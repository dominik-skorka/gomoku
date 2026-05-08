"""Tests for the chess-engine-style deep review (`review_game`).

These tests exercise the pure-logic layer in :mod:`gomoku.analyze` only —
no rich, no pydantic-ai, no Click.  They use a hand-rolled
:class:`FakeBrain` and a stubbed ``evaluate_board`` so the behaviour is
fully deterministic without depending on the JIT eval kernels.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from gomoku.analyze import (
    Advantage,
    Classification,
    ClassificationThresholds,
    DeepAnalysisReport,
    review_game,
)
from gomoku.board import Board
from gomoku.diagnostics import DiagnosticsHeader, MoveDiagnostics
from gomoku.models import GameConfig, Move, Stone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(size: int = 7) -> DiagnosticsHeader:
    return DiagnosticsHeader(
        started_at=datetime.now(tz=UTC),
        game_config=GameConfig(board_size=size, win_length=5),
        players=[
            {"stone": int(Stone.BLACK), "provider": "fake-black", "settings": {}},
            {"stone": int(Stone.WHITE), "provider": "fake-white", "settings": {}},
        ],
    )


def _move(n: int, stone: Stone, *, chosen: Move) -> MoveDiagnostics:
    return MoveDiagnostics(move_number=n, stone=stone, chosen_move=chosen)


def _write(
    path: Path,
    moves: list[MoveDiagnostics],
    *,
    size: int = 7,
) -> Path:
    lines = [_header(size).model_dump_json()]
    lines.extend(m.model_dump_json() for m in moves)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _FakeBrain:
    """Minimal ``MoveProvider`` returning a parametrised list of moves."""

    name = "fake"

    def __init__(self, moves: list[Move]) -> None:
        self._moves = list(moves)
        self._calls = 0

    def get_move(self, board: Board, stone: Stone) -> Move:
        del board, stone
        m = self._moves[self._calls]
        self._calls += 1
        return m


def _stub_eval(scores: dict[tuple[int, int, Stone], float]) -> object:
    """Build a stub ``evaluate_board(board, stone)`` keyed by (occupied stones)."""

    def _key(board: Board) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(board.occupied))

    table: dict[tuple[tuple[tuple[int, int], ...], Stone], float] = {}
    for (count, idx, stone), value in scores.items():
        # Encode (count, idx) as a synthetic key — the test fills `table`
        # directly via the wrapper below.  Unused; left for callers.
        del count, idx, stone, value

    def fn(board: Board, stone: Stone) -> float:
        # Look up by the exact occupancy tuple + stone perspective.
        k = (_key(board), stone)
        return table.get(k, 0.0)

    fn.table = table  # type: ignore[attr-defined]
    fn._key = _key  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# ClassificationThresholds — pure logic
# ---------------------------------------------------------------------------


class TestThresholds:
    @pytest.mark.parametrize(
        ("delta", "was_top", "expected"),
        [
            (0.0, True, Classification.BEST),
            (0.0, False, Classification.EXCELLENT),
            (199.0, False, Classification.EXCELLENT),
            (200.0, False, Classification.GOOD),
            (799.0, False, Classification.GOOD),
            (800.0, False, Classification.INACCURACY),
            (2_499.0, False, Classification.INACCURACY),
            (2_500.0, False, Classification.MISTAKE),
            (9_999.0, False, Classification.MISTAKE),
            (10_000.0, False, Classification.BLUNDER),
            (50_000.0, False, Classification.BLUNDER),
        ],
    )
    def test_classify_bands(
        self, delta: float, was_top: bool, expected: Classification
    ) -> None:
        t = ClassificationThresholds()
        assert t.classify(delta, was_top=was_top) is expected

    @pytest.mark.parametrize(
        ("eval_black", "expected"),
        [
            (60_000.0, Advantage.BLACK_WINNING),
            (50_000.0, Advantage.BLACK_WINNING),
            (5_000.0, Advantage.BLACK_BETTER),
            (1_500.0, Advantage.BLACK_BETTER),
            (0.0, Advantage.EQUAL),
            (-1_499.0, Advantage.EQUAL),
            (-1_500.0, Advantage.WHITE_BETTER),
            (-50_000.0, Advantage.WHITE_WINNING),
            (-99_000.0, Advantage.WHITE_WINNING),
        ],
    )
    def test_to_advantage_bands(self, eval_black: float, expected: Advantage) -> None:
        t = ClassificationThresholds()
        assert t.to_advantage(eval_black) is expected


# ---------------------------------------------------------------------------
# review_game — replay + classification
# ---------------------------------------------------------------------------


class TestReviewGame:
    def test_first_move_always_best(self, tmp_path: Path) -> None:
        played = Move(row=3, col=3)
        engine_pick = Move(row=4, col=4)  # different from played
        path = _write(tmp_path / "g.diag.jsonl", [_move(1, Stone.BLACK, chosen=played)])

        brain = _FakeBrain([engine_pick])

        # Even with a non-zero eval delta, move #1 must classify as BEST.
        def fake_eval(board: Board, stone: Stone) -> float:
            del board, stone
            return 5_000.0  # large bogus eval that would otherwise be a blunder

        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            evaluate=fake_eval,
        )
        assert isinstance(report, DeepAnalysisReport)
        assert len(report.reviews) == 1
        review = report.reviews[0]
        assert review.classification is Classification.BEST
        assert review.delta == 0.0

    def test_played_matches_engine_classified_best(self, tmp_path: Path) -> None:
        m1 = Move(row=3, col=3)
        m2 = Move(row=3, col=4)
        path = _write(
            tmp_path / "g.diag.jsonl",
            [
                _move(1, Stone.BLACK, chosen=m1),
                _move(2, Stone.WHITE, chosen=m2),
            ],
        )

        # Engine picks the same move played at every ply.
        brain = _FakeBrain([m1, m2])

        # Eval can be anything — was_top short-circuits to BEST.
        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            evaluate=lambda b, s: 100.0,
        )
        assert all(r.classification is Classification.BEST for r in report.reviews)

    def test_blunder_classification_from_delta(self, tmp_path: Path) -> None:
        played = Move(row=3, col=3)
        engine_pick = Move(row=4, col=4)
        path = _write(
            tmp_path / "g.diag.jsonl",
            [
                _move(1, Stone.BLACK, chosen=Move(row=0, col=0)),  # warm-up
                _move(2, Stone.WHITE, chosen=Move(row=0, col=1)),  # warm-up
                _move(3, Stone.BLACK, chosen=played),
            ],
        )
        brain = _FakeBrain(
            [Move(row=0, col=0), Move(row=0, col=1), engine_pick]
        )

        # Stub eval: only meaningful for ply 3.  After Black plays
        # `played`, mover-perspective eval = 0.  After Black plays
        # `engine_pick` (engine's choice), mover-perspective eval =
        # +20_000 — well into BLUNDER territory.
        def fake_eval(board: Board, stone: Stone) -> float:
            if (3, 3) in board.occupied and stone is Stone.BLACK:
                return 0.0
            if (4, 4) in board.occupied and stone is Stone.BLACK:
                return 20_000.0
            return 0.0

        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            evaluate=fake_eval,
        )
        # Plies 1 and 2 are warm-up: ply 1 forced BEST, ply 2 was-top BEST.
        assert report.reviews[2].classification is Classification.BLUNDER
        assert report.reviews[2].delta == pytest.approx(20_000.0)

    def test_inaccuracy_classification(self, tmp_path: Path) -> None:
        # Same as blunder test, but a smaller delta lands in INACCURACY.
        played = Move(row=3, col=3)
        engine_pick = Move(row=4, col=4)
        path = _write(
            tmp_path / "g.diag.jsonl",
            [
                _move(1, Stone.BLACK, chosen=Move(row=0, col=0)),
                _move(2, Stone.WHITE, chosen=Move(row=0, col=1)),
                _move(3, Stone.BLACK, chosen=played),
            ],
        )
        brain = _FakeBrain(
            [Move(row=0, col=0), Move(row=0, col=1), engine_pick]
        )

        def fake_eval(board: Board, stone: Stone) -> float:
            if (3, 3) in board.occupied and stone is Stone.BLACK:
                return 0.0
            if (4, 4) in board.occupied and stone is Stone.BLACK:
                return 1_000.0  # delta = 1_000 → INACCURACY
            return 0.0

        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            evaluate=fake_eval,
        )
        assert report.reviews[2].classification is Classification.INACCURACY

    def test_skips_missing_chosen_move(self, tmp_path: Path) -> None:
        """A diagnostics record without ``chosen_move`` is tolerated."""
        path = tmp_path / "g.diag.jsonl"
        lines = [_header().model_dump_json()]
        # Hand-construct a record with chosen_move=None.
        m = MoveDiagnostics(move_number=1, stone=Stone.BLACK, chosen_move=None)
        lines.append(m.model_dump_json())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        brain = _FakeBrain([])  # never called
        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            evaluate=lambda b, s: 0.0,
        )
        assert report.reviews == []

    def test_advantage_perspective_is_black(self, tmp_path: Path) -> None:
        """Advantage values come from the Black-perspective signed eval."""
        played = Move(row=3, col=3)
        path = _write(
            tmp_path / "g.diag.jsonl",
            [
                _move(1, Stone.BLACK, chosen=Move(row=0, col=0)),
                _move(2, Stone.WHITE, chosen=played),
            ],
        )
        brain = _FakeBrain([Move(row=0, col=0), played])

        # White plays.  evaluate is called from White's perspective —
        # we return +60_000 (White ahead).  In Black-perspective that's
        # -60_000 → WHITE_WINNING.
        def fake_eval(board: Board, stone: Stone) -> float:
            if stone is Stone.WHITE and (3, 3) in board.occupied:
                return 60_000.0
            return 0.0

        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            evaluate=fake_eval,
        )
        ply2 = report.reviews[1]
        assert ply2.advantage is Advantage.WHITE_WINNING
        assert ply2.eval_after_played == pytest.approx(-60_000.0)


# ---------------------------------------------------------------------------
# DeepAnalysisReport — accuracy + serialisation
# ---------------------------------------------------------------------------


class TestDeepReportShape:
    def test_classification_counts_and_accuracy(self, tmp_path: Path) -> None:
        # Build a 4-move game where Black plays one BEST + one BLUNDER,
        # and White plays two BESTs.
        played = Move(row=3, col=3)
        engine_pick = Move(row=4, col=4)
        m1 = Move(row=0, col=0)
        m2 = Move(row=0, col=1)
        m3 = played
        m4 = Move(row=1, col=0)
        path = _write(
            tmp_path / "g.diag.jsonl",
            [
                _move(1, Stone.BLACK, chosen=m1),
                _move(2, Stone.WHITE, chosen=m2),
                _move(3, Stone.BLACK, chosen=m3),
                _move(4, Stone.WHITE, chosen=m4),
            ],
        )
        brain = _FakeBrain([m1, m2, engine_pick, m4])

        def fake_eval(board: Board, stone: Stone) -> float:
            # Only ply 3 has a non-zero delta — and only via the
            # engine_pick branch — yielding BLUNDER.
            if (4, 4) in board.occupied and stone is Stone.BLACK:
                return 20_000.0
            return 0.0

        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            evaluate=fake_eval,
        )

        # Black: ply 1 forced BEST, ply 3 BLUNDER.
        assert report.classification_counts[Stone.BLACK][Classification.BEST] == 1
        assert report.classification_counts[Stone.BLACK][Classification.BLUNDER] == 1
        # White: ply 2 forced BEST, ply 4 was_top BEST.
        assert report.classification_counts[Stone.WHITE][Classification.BEST] == 2

        # Black accuracy = mean(1.0, 0.0) * 100 = 50; White = 100.
        assert report.accuracy[Stone.BLACK] == pytest.approx(50.0)
        assert report.accuracy[Stone.WHITE] == pytest.approx(100.0)

    def test_to_json_roundtrip(self, tmp_path: Path) -> None:
        played = Move(row=3, col=3)
        path = _write(tmp_path / "g.diag.jsonl", [_move(1, Stone.BLACK, chosen=played)])
        brain = _FakeBrain([played])

        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="alphabeta",
            analysis_depth=6,
            evaluate=lambda b, s: 0.0,
        )
        data = report.to_json()
        # Round-trip through json — verify it serialises cleanly.
        parsed = json.loads(json.dumps(data, default=str))
        assert parsed["analysis_brain"] == "alphabeta"
        assert parsed["analysis_depth"] == 6
        assert "reviews" in parsed and len(parsed["reviews"]) == 1
        assert "classification_counts" in parsed
        assert "accuracy" in parsed


# ---------------------------------------------------------------------------
# Commentator integration — uses a Python stub, no LLM
# ---------------------------------------------------------------------------


class _StubCommentator:
    """A fake commentator with a recorded payload."""

    def __init__(self, payload: object | None) -> None:
        self.payload = payload
        self.calls = 0

    def annotate(self, review, ctx):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.payload


class TestCommentatorIntegration:
    def test_attaches_commentary_when_provided(self, tmp_path: Path) -> None:
        played = Move(row=3, col=3)
        path = _write(tmp_path / "g.diag.jsonl", [_move(1, Stone.BLACK, chosen=played)])
        brain = _FakeBrain([played])

        payload = {"headline": "fine", "explanation": "ok", "key_threats": []}
        stub = _StubCommentator(payload)
        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            commentator=stub,
            evaluate=lambda b, s: 0.0,
        )
        assert stub.calls == 1
        assert report.reviews[0].commentary == payload

    def test_failing_commentator_leaves_commentary_none(self, tmp_path: Path) -> None:
        played = Move(row=3, col=3)
        path = _write(tmp_path / "g.diag.jsonl", [_move(1, Stone.BLACK, chosen=played)])
        brain = _FakeBrain([played])

        class _Boom:
            def annotate(self, review, ctx):  # type: ignore[no-untyped-def]
                raise RuntimeError("api down")

        report = review_game(
            path,
            analysis_brain=brain,
            analysis_brain_name="fake",
            analysis_depth=1,
            commentator=_Boom(),
            evaluate=lambda b, s: 0.0,
        )
        assert report.reviews[0].commentary is None
