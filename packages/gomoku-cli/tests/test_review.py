"""Blunder review: detection rules, escalation and the end-to-end command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from gomoku.models import Stone
from gomoku_cli.main import cli
from gomoku_cli.match import EngineSpec, GameRecord
from gomoku_cli.match_worker import merge
from gomoku_cli.review import (
    Blunder,
    EngineTally,
    GameReview,
    OracleConfig,
    _escalate,
    _Oracle,
    escalations,
    five_cells,
    format_report,
    review_game,
    run_review,
    verdict,
)

SIZE = 15


class FakeOracle(_Oracle):
    """Answers from a table keyed by the position; anything else is DISPROVED."""

    def __init__(self, table=None, *, screen_default: str = "DISPROVED") -> None:
        super().__init__(engine=None, config=OracleConfig(screen_moves=400))  # type: ignore[arg-type]
        self.table = table or {}
        self.screen_default = screen_default

    def status(self, moves, stone=None, *, screen=False):
        key = (tuple(moves), None if stone is None else int(stone))
        return self.table.get(key, (self.screen_default if screen else "DISPROVED", None))


class FakeSlot:
    def __init__(self, replies: dict[str, tuple[int, int]], brain_type: str = "threat", ms: float = 100.0) -> None:
        self.replies = replies
        self.brain_type = brain_type
        self.ms = ms
        self.seen: list[dict[str, Any]] = []

    def settings(self):
        return self.brain_type, {
            "depth": 5,
            "pns_nodes": 10_000,
            "pns_time_s": 5.0,
            "threat_search": "pns",
            "vct_depth": 12,
            "fallback": {"depth": 5, "max_candidates": 12},
        }

    def retry(self, moves, overrides, timeout_s):
        self.seen.append(overrides)
        name = next((n for n, o in [("repeat", {}), *escalations(*self.settings())] if o == overrides), "?")
        if name not in self.replies:
            return None  # a timeout
        return {"move": list(self.replies[name]), "ms": self.ms, "path": None}


def _record(moves: list[tuple[int, int]], *, reason: str = "move limit", paths=None) -> GameRecord:
    engine_moves = len(moves) - 2
    return GameRecord(
        opening=0,
        a_is_black=True,
        winner=None,
        reason=reason,
        moves=moves,
        a_ms=[100.0] * ((engine_moves + 1) // 2),
        b_ms=[100.0] * (engine_moves // 2),
        paths=paths or [None] * engine_moves,
    )


# Black has four in row 7 (cols 3-6, open at 2 and 7); White's stones are scattered.
FOUR = [(7, 3), (0, 0), (7, 4), (0, 2), (7, 5), (0, 4), (7, 6), (0, 6)]
# The same four with White blocking col 7: only (7, 2) completes it.
HALF_FOUR = [*FOUR[:-1], (7, 7)]


class TestFiveCells:
    def test_open_four_has_two_completions(self):
        assert sorted(five_cells(FOUR, Stone.BLACK, SIZE)) == [(7, 2), (7, 7)]
        assert five_cells(FOUR, Stone.WHITE, SIZE) == []

    def test_gap_and_overline_count(self):
        gapped = [(3, 3), (0, 14), (3, 4), (2, 14), (3, 6), (4, 14), (3, 7)]
        assert five_cells(gapped, Stone.BLACK, SIZE) == [(3, 5)]


class TestDetection:
    def test_missed_win_and_missed_block(self):
        # Black (to move at ply 8) ignores its own five; White then ignores the block.
        moves = [*HALF_FOUR, (10, 10), (10, 11)]
        review = review_game(0, _record(moves), 2, FakeOracle(), {}, SIZE, escalate=False)
        found = [(b.ply, b.kind, b.engine, b.better) for b in review.blunders]
        assert found == [(8, "missed_win", "a", (7, 2)), (9, "missed_block", "b", (7, 2))]

    def test_open_four_cannot_be_blocked(self):
        moves = [*FOUR, (7, 2)]  # White (ply 7) faced two completions: no blunder
        review = review_game(0, _record(moves, reason="five in a row"), 2, FakeOracle(), {}, SIZE, escalate=False)
        assert review.blunders == []
        assert review.statuses[-1] is None

    def test_losing_move_names_a_saving_move(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14)]
        # After Black's 3rd move (ply 2) White still has nothing, but after
        # White's ply-3 move Black is proven to win; a different ply-3 move held.
        table = {
            (tuple(moves[:4]), None): ("PROVED", (7, 9)),
            (tuple(moves[:3]), int(Stone.BLACK)): ("PROVED", (7, 9)),
        }
        for cell in [(r, c) for r in range(SIZE) for c in range(SIZE)]:
            if cell != (7, 6) and cell not in moves:
                table[((*moves[:3], cell), None)] = ("PROVED", (7, 9))
        review = review_game(0, _record(moves), 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert [(b.ply, b.kind, b.better) for b in review.blunders] == [(3, "losing_move", (7, 6))]

    def test_already_lost_position_is_not_a_blunder(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14)]
        cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        table = {((*moves[:3], cell), None): ("PROVED", None) for cell in cells}
        table[(tuple(moves[:4]), None)] = ("PROVED", None)
        review = review_game(0, _record(moves), 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert review.blunders == []

    def test_brain_claim_disproved_is_unsound(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14)]
        review = review_game(0, _record(moves, paths=["pns_win", None]), 2, FakeOracle(), {}, SIZE, escalate=True)
        assert [(b.ply, b.kind, b.verdict) for b in review.blunders] == [(2, "unsound_proof", "bug")]

    def test_oracle_proof_lost_after_its_own_move_is_unsound(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14), (3, 3), (4, 4)]
        table = {(tuple(moves[:2]), None): ("PROVED", (7, 8))}  # ply 2 plays the proof's move
        review = review_game(0, _record(moves), 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert [(b.ply, b.kind) for b in review.blunders] == [(2, "unsound_proof")]

    def test_win_kept_after_a_different_move_is_fine(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14), (3, 3), (4, 4)]
        table = {
            (tuple(moves[:2]), None): ("PROVED", (7, 9)),
            (tuple(moves[:4]), None): ("PROVED", (7, 9)),
            (tuple(moves[:6]), None): ("PROVED", (7, 9)),
        }
        # White's position stayed lost whatever it played.
        oracle = FakeOracle(table, screen_default="PROVED")
        review = review_game(0, _record(moves), 2, oracle, {}, SIZE, escalate=False)
        assert review.blunders == []


class TestEscalation:
    @staticmethod
    def _blunder(kind: str = "missed_block", better=(7, 2)) -> Blunder:
        return Blunder(0, 9, "b", "white", kind, (10, 11), better, "", None, 100.0)

    def test_setting_that_fixes_it_is_named_with_its_cost(self):
        slot = FakeSlot({"repeat": (10, 11), "more_pns": (10, 11), "deeper": (7, 2)}, ms=300.0)
        blunder = self._blunder()
        blunder.ms = 100.0
        _escalate(blunder, [*HALF_FOUR, (10, 10)], slot, FakeOracle(), SIZE, 10.0)
        assert blunder.verdict == "tuning"
        assert blunder.remedy == "deeper: fallback.depth=6 (3.0x think time)"
        assert [a.good for a in blunder.attempts] == [False, False, True]  # stops at the first fix

    def test_nothing_helps_means_a_bug(self):
        slot = FakeSlot(dict.fromkeys(["repeat", "more_pns", "deeper", "wider", "all"], (10, 11)))
        blunder = self._blunder()
        _escalate(blunder, [*HALF_FOUR, (10, 10)], slot, FakeOracle(), SIZE, 10.0)
        assert blunder.verdict == "bug"
        assert len(blunder.attempts) == 5

    def test_flaky_move_found_on_repeat(self):
        slot = FakeSlot({"repeat": (7, 2)})
        blunder = self._blunder()
        _escalate(blunder, [*HALF_FOUR, (10, 10)], slot, FakeOracle(), SIZE, 10.0)
        assert blunder.verdict == "tuning" and "timing" in blunder.remedy

    def test_losing_move_retry_is_judged_by_the_oracle(self):
        moves = [(7, 7), (0, 0), (7, 8)]
        oracle = FakeOracle({((*moves, (5, 5)), None): ("PROVED", None)})
        slot = FakeSlot({"repeat": (5, 5), "more_pns": (6, 6)})
        blunder = Blunder(0, 3, "b", "white", "losing_move", (14, 14), (7, 6), "", None, 100.0)
        _escalate(blunder, [*moves, (14, 14)], slot, oracle, SIZE, 10.0)
        assert [a.good for a in blunder.attempts[:2]] == [False, True]
        assert blunder.remedy.startswith("more_pns: pns_nodes=100000, pns_time_s=20.0")


class TestEscalations:
    def test_threat_with_fallback_table(self):
        settings = {
            "depth": 5,
            "pns_nodes": 10_000,
            "pns_time_s": 5.0,
            "threat_search": "pns",
            "vct_depth": 12,
            "fallback": {"depth": 5, "max_candidates": 12},
        }
        assert escalations("threat", settings) == [
            ("more_pns", {"pns_nodes": 100_000, "pns_time_s": 20.0}),
            ("deeper", {"fallback": {"depth": 6}}),
            ("wider", {"fallback": {"max_candidates": 24}}),
            ("all", {"pns_nodes": 100_000, "pns_time_s": 20.0, "fallback": {"depth": 6, "max_candidates": 24}}),
        ]

    def test_threat_without_fallback_keeps_its_depth(self):
        settings = {"depth": 4, "pns_nodes": 10_000, "pns_time_s": 30.0, "threat_search": "vct", "vct_depth": 12}
        variants = dict(escalations("threat", {**settings, "fallback": None}))
        assert variants["more_pns"] == {"pns_nodes": 100_000, "pns_time_s": 60.0, "vct_depth": 16}
        assert variants["deeper"] == {"depth": 5}
        assert variants["wider"] == {"fallback": {"depth": 4, "max_candidates": 40}}
        assert variants["all"]["fallback"] == {"depth": 4, "max_candidates": 40}

    def test_other_brains(self):
        assert [n for n, _ in escalations("alphabeta", {"depth": 4, "max_candidates": 20})] == [
            "deeper",
            "wider",
            "all",
        ]
        assert escalations("mcts", {"iterations": 1000}) == [("more_iterations", {"iterations": 4000})]
        assert escalations("random", {}) == []

    def test_variants_validate_against_the_settings_models(self):
        from gomoku.settings import GomokuSettings
        from gomoku_brain import BRAINS, resolve_brain

        for name in ("threat", "alphabeta", "mcts"):
            brain_type, settings = resolve_brain(name, GomokuSettings().brains)
            for _, overrides in escalations(brain_type, settings):
                BRAINS[brain_type].settings_cls(**merge(settings, overrides))


class TestMerge:
    def test_nested_tables_merge(self):
        base = {"depth": 5, "fallback": {"depth": 5, "max_candidates": 12}}
        assert merge(base, {"fallback": {"depth": 6}}) == {"depth": 5, "fallback": {"depth": 6, "max_candidates": 12}}
        assert merge({"fallback": None}, {"fallback": {"depth": 4}}) == {"fallback": {"depth": 4}}
        assert base["fallback"]["depth"] == 5


class TestVerdict:
    @staticmethod
    def _totals(a_bugs: int, b_bugs: int, b_blunders: int) -> dict[str, EngineTally]:
        return {
            "a": EngineTally(moves=10, blunders=a_bugs, bugs=a_bugs),
            "b": EngineTally(moves=10, blunders=b_blunders, bugs=b_bugs),
        }

    @pytest.mark.parametrize(
        ("a_bugs", "b_bugs", "b_blunders", "status"),
        [(0, 0, 0, "PASS"), (0, 0, 2, "WARN"), (1, 1, 1, "WARN"), (0, 1, 1, "FAIL"), (1, 2, 2, "FAIL")],
    )
    def test_candidate_fails_only_on_extra_bugs(self, a_bugs, b_bugs, b_blunders, status):
        assert verdict(self._totals(a_bugs, b_bugs, b_blunders))[0] == status

    def test_report_lists_candidate_blunders_first(self):
        record = _record([*FOUR, (10, 10), (10, 11)])
        blunders = [
            Blunder(0, 8, "a", "black", "missed_win", (10, 10), (7, 2), "x", None, 50.0, verdict="bug"),
            Blunder(0, 9, "b", "white", "missed_block", (10, 11), (7, 2), "y", "alphabeta", 50.0, verdict="bug"),
        ]
        text = format_report(EngineSpec("x"), EngineSpec("y"), [GameReview(0, record, [], blunders)])
        assert text.index("move 10: B") < text.index("move 9: A")
        assert "Verdict: WARN" in text


class TestReviewEndToEnd:
    """Real engine and oracle processes; the random brain keeps it fast."""

    def test_random_vs_random(self):
        seen: list[GameReview] = []
        reviews = run_review(
            EngineSpec("random"),
            EngineSpec("random"),
            games=2,
            seed=3,
            size=9,
            oracle=OracleConfig(nodes=2_000, time_s=1.0, screen_nodes=1_000, screen_time_s=0.5, screen_moves=5),
            on_game=seen.append,
        )
        assert len(reviews) == len(seen) == 2
        for review in reviews:
            record = review.record
            assert len(record.paths) == len(record.moves) - 7
            assert len(review.statuses) == len(record.moves) - 7 + 1
            assert all(s is None or s[0] in {"PROVED", "DISPROVED", "UNKNOWN"} for s in review.statuses)
            for blunder in review.blunders:
                assert blunder.verdict in {"tuning", "bug"}
                assert blunder.attempts or blunder.kind == "unsound_proof"

    def test_cli_report_json_and_exit_code(self, tmp_path: Path):
        out = tmp_path / "review.json"
        args = ["review", "--a", "random", "--b", "random", "--games", "2", "--board-size", "9"]
        args += ["--oracle-nodes", "2000", "--oracle-time", "1", "--no-escalate", "--out", str(out)]
        result = CliRunner().invoke(cli, args)
        payload = json.loads(out.read_text())
        assert "Verdict:" in result.output
        assert result.exit_code == (1 if payload["verdict"]["status"] == "FAIL" else 0), result.output
        assert len(payload["games"]) == 2
        assert {"oracle", "blunders", "record"} <= set(payload["games"][0])
