"""Tests for the VCF/VCT threat-based brain."""

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.threat import ThreatBrain, _find_four_threats, _find_three_threats, _find_winning_move, _vcf, _vct

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _place(board: Board, moves: list[tuple[int, int]], stone: Stone) -> Board:
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
    return board


# ---------------------------------------------------------------------------
# Four-threat detection
# ---------------------------------------------------------------------------


class TestFindFourThreats:
    def test_straight_four_detected(self):
        """Three-in-a-row with both ends open → placing at either end creates a four."""
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        threats = _find_four_threats(b, Stone.BLACK)
        attack_moves = {t[0] for t in threats}
        # Extending to col=2 or col=6 should be among the attack moves
        assert Move(row=4, col=2) in attack_moves or Move(row=4, col=6) in attack_moves

    def test_four_in_a_row_creates_five(self):
        """Four-in-a-row → placing at either end is an immediate five (win)."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        threats = _find_four_threats(b, Stone.BLACK)
        # Should find at least one winning threat with defence=None
        wins = [t for t in threats if t[1] is None]
        assert len(wins) >= 1


# ---------------------------------------------------------------------------
# Three-threat detection
# ---------------------------------------------------------------------------


class TestFindThreeThreats:
    def test_live_three_detected(self):
        """Two-in-a-row with both ends open → placing at either end creates a live three."""
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5)], Stone.BLACK)
        threats = _find_three_threats(b, Stone.BLACK)
        # At least one live-three threat should be detected
        assert len(threats) >= 1


# ---------------------------------------------------------------------------
# VCF
# ---------------------------------------------------------------------------


class TestVCF:
    def test_finds_immediate_win(self):
        """Four-in-a-row → VCF should find the winning move."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        # Add opponent stones so the board is not trivially empty
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        move = _vcf(b, Stone.BLACK, depth=20, is_attacker=True)
        assert move is not None
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_returns_none_when_no_threat(self):
        """No four-threats → VCF returns None."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)
        move = _vcf(b, Stone.BLACK, depth=20, is_attacker=True)
        assert move is None

    def test_chained_vcf(self):
        """A position needing two consecutive four-threats to win."""
        # Black has two separate open threes that can chain into a forced win.
        # White has scattered stones with no threat.
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4)], Stone.BLACK)
        b = _place(b, [(6, 2), (6, 3), (6, 4)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 8), (2, 0), (2, 8), (8, 0), (8, 8)], Stone.WHITE)

        move = _vcf(b, Stone.BLACK, depth=20, is_attacker=True)
        # VCF should find a first four-threat move in one of the two rows.
        # Extending an open three creates a live four (unblockable).
        assert move is not None


# ---------------------------------------------------------------------------
# VCT
# ---------------------------------------------------------------------------


class TestVCT:
    def test_finds_immediate_win(self):
        """Four-in-a-row → VCT (which calls VCF internally) should find the win."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        move = _vct(b, Stone.BLACK, depth=8, is_attacker=True)
        assert move is not None
        assert move in (Move(row=4, col=1), Move(row=4, col=6))


# ---------------------------------------------------------------------------
# ThreatBrain (integration)
# ---------------------------------------------------------------------------


class TestThreatBrain:
    def test_finds_winning_move(self):
        """Standard test: four-in-a-row, brain should complete to five."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_fallback_on_empty_board(self):
        """On an empty-ish board with no threats, should fall back gracefully."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)
        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        # Should return a valid move without crashing
        assert b.is_valid_move(move)

    def test_blocks_opponent_vcf(self):
        """When opponent has a VCF, brain should try to block it."""
        b = Board(size=9)
        # White has four-in-a-row → imminent win
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.WHITE)
        b = _place(b, [(2, 2), (2, 3)], Stone.BLACK)
        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        # Should block at col=1 or col=6
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_name(self):
        brain = ThreatBrain()
        assert brain.name == "threat"

    def test_blocks_opponent_immediate_win_over_own_vcf(self):
        """Regression: VCF must not ignore an opponent four-in-a-row.

        Reproduces the game where Black had a VCF via F5 (live four on row 5)
        but White already had H6-I7-J8-K9 on a diagonal needing only L10.
        Black must block at L10 rather than play F5.

        Board before move #20 (Black to play, 15x15):

            A B C D E F G H I J K L M N O
         4  · · ○ · · · · · · · · · · · ·
         5  · · · ● ● · ● · · · · · · · ·   ← Black has D5, E5, G5
         6  · · · · ● ○ ○ ○ ○ ● · · · · ·   ← White: F6, G6, H6, I6
         7  · · · · · ● · ● ○ · · · · · ·   ← White: I7
         8  · · · · · ○ ● ● ● ○ · · · · ·   ← White: J8
         9  · · · · · · · ○ · · ○ · · · ·   ← White: K9
        10  · · · · · · · · · · · · · · ·   ← L10 empty → White wins
        """
        b = Board(size=15)
        # Black stones
        b = _place(b, [(4, 3), (4, 4), (4, 6)], Stone.BLACK)  # D5, E5, G5
        b = _place(b, [(5, 4)], Stone.BLACK)  # E6
        b = _place(b, [(6, 5), (6, 7)], Stone.BLACK)  # F7, H7
        b = _place(b, [(7, 6), (7, 7), (7, 8)], Stone.BLACK)  # G8, H8, I8
        b = _place(b, [(5, 9)], Stone.BLACK)  # J6
        # White stones
        b = _place(b, [(3, 2)], Stone.WHITE)  # C4
        b = _place(b, [(5, 5), (5, 6), (5, 7), (5, 8)], Stone.WHITE)  # F6, G6, H6, I6
        b = _place(b, [(6, 8)], Stone.WHITE)  # I7
        b = _place(b, [(7, 5), (7, 9)], Stone.WHITE)  # F8, J8
        b = _place(b, [(8, 7)], Stone.WHITE)  # H9
        b = _place(b, [(8, 10)], Stone.WHITE)  # K9

        # White has diagonal H6(5,7)-I7(6,8)-J8(7,9)-K9(8,10) → needs L10(9,11)
        assert _find_winning_move(b, Stone.WHITE) == Move(row=9, col=11)

        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        # Black MUST block at L10 (row=9, col=11)
        assert move == Move(row=9, col=11)

    def test_vcf_rejects_live_four_when_opponent_has_five(self):
        """VCF should not return a live-four if opponent can complete five."""
        b = Board(size=15)
        # Black has three in a row with both ends open → live four candidate
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        # White has four in a diagonal → can complete five next move
        b = _place(b, [(0, 0), (1, 1), (2, 2), (3, 3)], Stone.WHITE)
        # White can win at (4,4) but that's occupied; let's adjust:
        # White diagonal: rows 5-8, cols 10-13
        b2 = Board(size=15)
        b2 = _place(b2, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b2 = _place(b2, [(5, 10), (6, 11), (7, 12), (8, 13)], Stone.WHITE)
        # White can win at (4,9) or (9,14)
        # VCF should NOT return the live-four extension for Black
        result = _vcf(b2, Stone.BLACK, depth=20, is_attacker=True)
        # If VCF returns a move, placing it must not let White win
        if result is not None:
            trial = b2.place_stone(result, Stone.BLACK)
            assert _find_winning_move(trial, Stone.WHITE) is None
