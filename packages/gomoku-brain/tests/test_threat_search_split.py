"""Smoke tests for the ``threat_search`` module split.

The VCF and VCT recursive search cores were lifted out of
:mod:`gomoku_brain.threat` into :mod:`gomoku_brain.threat_search`.
The original module re-exports the search names so that downstream
modules (:mod:`gomoku_brain.alphabeta`, :mod:`gomoku_brain.mcts`,
:mod:`gomoku_brain.pns`) and tests that still import them from
``gomoku_brain.threat`` keep working.

This file pins:

1. The new module is importable directly.
2. ``threat.py`` re-exports the search symbols and the re-exports are
   the **same** function objects (not silent copies).
3. A round-trip VCF win is found via both the direct and the re-exported
   entry points and they agree.
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone


class TestThreatSearchDirectImport:
    def test_module_is_importable(self):
        import gomoku_brain.threat_search as ts

        assert hasattr(ts, "_vcf")
        assert hasattr(ts, "_vct")
        assert hasattr(ts, "classify_vcf_breakers")
        assert hasattr(ts, "_state_is_win")

    def test_vcf_finds_simple_forced_win(self):
        from gomoku_brain.threat_search import _vcf

        # Black has open-three on row 7 cols 4-6; placing at (7,3) or (7,7)
        # gives a live four — VCF wins in 1 ply.
        b = Board(size=15)
        for c in range(4, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        result = _vcf(b, Stone.BLACK, depth=2)
        assert result is not None
        # Either end of the live three completes a four-threat.
        assert (result.row, result.col) in {(7, 3), (7, 7)}


class TestThreatBackCompatReExport:
    """Downstream modules still do ``from gomoku_brain.threat import _vcf``;
    that import path must keep working and must point at the *same*
    function object as the direct one."""

    def test_vcf_reexport_is_same_object(self):
        from gomoku_brain.threat import _vcf as via_threat
        from gomoku_brain.threat_search import _vcf as direct

        assert via_threat is direct

    def test_vct_reexport_is_same_object(self):
        from gomoku_brain.threat import _vct as via_threat
        from gomoku_brain.threat_search import _vct as direct

        assert via_threat is direct

    def test_classify_vcf_breakers_reexport_is_same_object(self):
        from gomoku_brain.threat import classify_vcf_breakers as via_threat
        from gomoku_brain.threat_search import classify_vcf_breakers as direct

        assert via_threat is direct

    def test_state_is_win_reexport_is_same_object(self):
        from gomoku_brain.threat import _state_is_win as via_threat
        from gomoku_brain.threat_search import _state_is_win as direct

        assert via_threat is direct

    def test_vct_defence_moves_reexport_is_same_object(self):
        from gomoku_brain.threat import _vct_defence_moves as via_threat
        from gomoku_brain.threat_search import _vct_defence_moves as direct

        assert via_threat is direct


class TestRoundTripBehaviourMatches:
    """Belt-and-braces: call the same search via both paths and verify the
    behaviour is identical, not just the symbol identity."""

    def test_vcf_via_both_paths_agrees(self):
        from gomoku_brain.threat import _vcf as via_threat
        from gomoku_brain.threat_search import _vcf as direct

        b = Board(size=15)
        for c in range(4, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        a = via_threat(b, Stone.BLACK, depth=2)
        d = direct(b, Stone.BLACK, depth=2)
        assert a == d

    def test_classify_vcf_breakers_via_both_paths_agrees(self):
        from gomoku_brain.threat import classify_vcf_breakers as via_threat
        from gomoku_brain.threat_search import classify_vcf_breakers as direct

        # Empty board — every candidate is a "breaker" because no VCF
        # exists; the return is just (durable, all) for the same input.
        b = Board(size=9)
        moves = [Move(row=4, col=4), Move(row=3, col=3)]
        assert via_threat(b, moves, Stone.BLACK, vcf_depth=2) == direct(b, moves, Stone.BLACK, vcf_depth=2)
