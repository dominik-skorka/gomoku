"""Smoke tests for the post-split module layout.

The threat-detection helpers and the alpha-beta move-classification
helpers were extracted from the two large brain modules into dedicated
sub-modules so each concern can be imported and reasoned about
independently.  These tests pin two things:

1. The new modules are importable directly (``threat_detect``,
   ``alphabeta_ordering``).
2. ``threat.py`` still re-exports the detection symbols by their
   historical names — the old ``from gomoku_brain.threat import
   _find_four_threats`` style of import (used by ``pns.py``,
   ``mcts.py``, ``test_threat.py``, ``test_pns.py``,
   ``test_parity.py``) keeps working.

If either invariant breaks, the rest of the suite would still pass
under the right import order, but downstream callers that haven't
been migrated would stop importing.
"""

from gomoku.board import Board
from gomoku.models import Move, Stone


class TestThreatDetectDirectImport:
    def test_finds_four_threat(self):
        from gomoku_brain.threat_detect import _find_four_threats

        b = Board(size=15)
        for c in range(3, 6):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        threats = _find_four_threats(b, Stone.BLACK)
        attacks = {(t[0].row, t[0].col) for t in threats}
        # (7, 2) and (7, 6) both extend to a live four.
        assert {(7, 2), (7, 6)} <= attacks

    def test_probe_immediate_returns_win_tag(self):
        from gomoku_brain.threat_detect import probe_immediate

        b = Board(size=15)
        for c in range(3, 7):  # 4-in-a-row
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        move, tag = probe_immediate(b, Stone.BLACK)
        assert tag == "instant_win"
        assert move is not None


class TestThreatBackCompatReExport:
    """``threat.py`` must continue to expose the detection helpers under
    their historical names — there are call sites in ``pns.py``,
    ``mcts.py``, and the test suite that still import them this way."""

    def test_find_four_threats_reexport(self):
        from gomoku_brain.threat import _find_four_threats
        from gomoku_brain.threat_detect import _find_four_threats as direct

        # Re-export must be the *same* function object, not a copy.
        assert _find_four_threats is direct

    def test_find_three_threats_reexport(self):
        from gomoku_brain.threat import _find_three_threats
        from gomoku_brain.threat_detect import _find_three_threats as direct

        assert _find_three_threats is direct

    def test_find_winning_move_reexport(self):
        from gomoku_brain.threat import _find_winning_move
        from gomoku_brain.threat_detect import _find_winning_move as direct

        assert _find_winning_move is direct

    def test_probe_immediate_reexport(self):
        from gomoku_brain.threat import probe_immediate
        from gomoku_brain.threat_detect import probe_immediate as direct

        assert probe_immediate is direct


class TestAlphabetaOrderingDirectImport:
    def test_split_threats_partitions_correctly(self):
        from gomoku_brain.alphabeta_ordering import _split_threats

        b = Board(size=15)
        for c in range(3, 6):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        # (7, 6) extends to a four — threat.  (0, 0) is far away — non-threat.
        threats, non = _split_threats(b, [Move(row=7, col=6), Move(row=0, col=0)], Stone.BLACK)
        assert Move(row=7, col=6) in threats
        assert Move(row=0, col=0) in non

    def test_count_open_threes_at_zero_on_empty(self):
        from gomoku_brain.alphabeta_ordering import _count_open_threes_at

        assert _count_open_threes_at(Board(size=9), 4, 4, Stone.BLACK) == 0

    def test_count_open_threes_at_detects_extension(self):
        from gomoku_brain.alphabeta_ordering import _count_open_threes_at

        b = Board(size=15)
        # Two stones in a row: placing at (7, 4) creates an open three.
        b = b.place_stone(Move(row=7, col=5), Stone.BLACK)
        b = b.place_stone(Move(row=7, col=6), Stone.BLACK)
        assert _count_open_threes_at(b, 7, 4, Stone.BLACK) == 1
