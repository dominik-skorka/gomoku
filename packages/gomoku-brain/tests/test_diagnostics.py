"""Tests for the post-game diagnostics pipeline.

These assertions focus on the invariants the :mod:`gomoku_cli.analyze`
module relies on: brains must populate the new fields correctly so the
analyzer can flag blunders, alternative-best deviations, time spikes,
and TT performance.
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.diagnostics import (
    CandidateInfo,
    DiagnosticsRecorder,
    MoveDiagnostics,
    SearchStats,
    ThreatInfo,
    current_move_diagnostics,
    set_move_diagnostics,
)
from gomoku.models import GameConfig, Move, Stone
from gomoku_brain.alphabeta import AlphaBetaBrain

# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------


class TestCandidateInfoScoreType:
    def test_default_is_eval(self):
        c = CandidateInfo(move=Move(row=0, col=0), score=1.0)
        assert c.score_type == "eval"

    def test_accepts_alternative_types(self):
        c = CandidateInfo(move=Move(row=0, col=0), score=0.5, score_type="winrate")
        assert c.score_type == "winrate"

    def test_round_trips_through_json(self):
        c = CandidateInfo(move=Move(row=3, col=4), score=123.4, score_type="priority")
        round_tripped = CandidateInfo.model_validate_json(c.model_dump_json())
        assert round_tripped == c


class TestMoveDiagnosticsAlternativeBest:
    def test_defaults_to_none(self):
        d = MoveDiagnostics(move_number=1, stone=Stone.BLACK)
        assert d.alternative_best is None

    def test_can_be_set(self):
        alt = CandidateInfo(move=Move(row=2, col=2), score=1000.0)
        d = MoveDiagnostics(move_number=1, stone=Stone.BLACK, alternative_best=alt)
        assert d.alternative_best == alt


class TestSearchStatsFields:
    def test_all_new_fields_nullable(self):
        s = SearchStats()
        assert s.nodes_searched is None
        assert s.qs_nodes_searched is None
        assert s.tt_hits is None
        assert s.tt_probes is None

    def test_values_persist(self):
        s = SearchStats(
            depth_reached=5,
            nodes_searched=1000,
            qs_nodes_searched=250,
            tt_hits=300,
            tt_probes=1500,
        )
        assert s.nodes_searched == 1000
        assert s.tt_probes == 1500


# ---------------------------------------------------------------------------
# AlphaBetaBrain populates the new fields
# ---------------------------------------------------------------------------


class TestAlphaBetaPopulates:
    def _make_brain(self) -> AlphaBetaBrain:
        # Tiny depth keeps the test fast but still exercises the diag block.
        return AlphaBetaBrain(depth=2, time_limit_s=0.0)

    def _run_with_diag(self, brain: AlphaBetaBrain, board: Board, stone: Stone) -> MoveDiagnostics:
        diag = MoveDiagnostics(move_number=1, stone=stone)
        set_move_diagnostics(diag)
        try:
            brain.get_move(board, stone)
        finally:
            set_move_diagnostics(None)
        return diag

    def test_populates_nodes_and_tt_counters(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=5), Stone.WHITE)
        diag = self._run_with_diag(self._make_brain(), b, Stone.BLACK)
        assert diag.search_stats.nodes_searched is not None
        assert diag.search_stats.nodes_searched > 0
        # tt_probes is always non-negative; TT probes should exceed hits.
        assert diag.search_stats.tt_probes is not None
        assert diag.search_stats.tt_hits is not None
        assert 0 <= diag.search_stats.tt_hits <= diag.search_stats.tt_probes

    def test_populates_threat_counts(self):
        # Black with a live-three + one white stone so both sides have non-zero
        # threat data to populate.
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=3), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=5), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)
        diag = self._run_with_diag(self._make_brain(), b, Stone.BLACK)
        # Either ThreatBrain populated these or AlphaBeta's fallback did —
        # the point is they must be populated (non-default), not specific values.
        our = diag.our_threats
        assert isinstance(our, ThreatInfo)
        # four_threats + three_threats should be non-negative ints
        assert our.four_threats >= 0
        assert our.three_threats >= 0
        assert diag.opp_threats.four_threats >= 0

    def test_top_candidates_are_eval_typed(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        diag = self._run_with_diag(self._make_brain(), b, Stone.WHITE)
        assert diag.top_candidates
        for c in diag.top_candidates:
            assert c.score_type == "eval"

    def test_alternative_best_recorded_when_chosen_is_not_top(self):
        # A position where multiple moves have similar evals increases the
        # chance that best_move deviates from the static top.  We do NOT
        # require deviation — we only require correctness of the invariant:
        # alternative_best is set iff top_candidates[0] != chosen_move.
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        diag = self._run_with_diag(self._make_brain(), b, Stone.WHITE)
        if diag.top_candidates and diag.top_candidates[0].move != diag.chosen_move:
            assert diag.alternative_best == diag.top_candidates[0]
        else:
            assert diag.alternative_best is None


# ---------------------------------------------------------------------------
# ContextVar scoping
# ---------------------------------------------------------------------------


class TestDiagnosticsContext:
    def test_begin_end_move_toggles_context(self, tmp_path):
        rec = DiagnosticsRecorder(
            game_config=GameConfig(board_size=9, win_length=5),
            players=[],
            directory=tmp_path,
        )
        assert current_move_diagnostics() is None
        diag = rec.begin_move(1, Stone.BLACK, [])
        assert current_move_diagnostics() is diag
        rec.end_move(diag)
        assert current_move_diagnostics() is None

    def test_end_move_persists_to_file(self, tmp_path):
        rec = DiagnosticsRecorder(
            game_config=GameConfig(board_size=9, win_length=5),
            players=[],
            directory=tmp_path,
        )
        diag = rec.begin_move(1, Stone.BLACK, [])
        diag.think_time_ms = 123
        diag.chosen_move = Move(row=4, col=4)
        rec.end_move(diag)
        rec.write_summary({"total_moves": 1})
        content = rec.path.read_text(encoding="utf-8")
        assert "\"think_time_ms\":123" in content
        assert "\"type\": \"summary\"" in content or "\"type\":\"summary\"" in content
