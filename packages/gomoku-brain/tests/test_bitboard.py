"""Tests for the bitboard representation and JIT functions.

Verifies that bitboard-based operations produce results consistent with
the existing numpy-grid and pure-Python implementations.
"""

import numpy as np
import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.bitboard import BitboardState
from gomoku_brain.bitboard_jit import (
    bitboard_to_grid,
    check_winner_at_bb,
    check_winner_bb,
    evaluate_bb,
    get_candidates_bb,
    get_fork_bonus,
    get_fork_threshold,
    get_open_three_bonus,
    get_pattern_table,
    get_threat_moves_bb,
    grid_to_bitboard,
    is_four_threat_bb,
    occupied_from_bb,
)
from gomoku_brain.grid_utils import board_to_bitboard, board_to_grid, occupied_coords

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _place(board: Board, moves: list[tuple[int, int]], stone: Stone) -> Board:
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
    return board


def _eval_bb(board: Board, stone: Stone) -> float:
    """Evaluate board using the bitboard path."""
    black, white, size = board_to_bitboard(board)
    return evaluate_bb(
        black,
        white,
        size,
        int(stone),
        get_pattern_table(),
        get_fork_threshold(),
        get_fork_bonus(),
        get_open_three_bonus(),
    )


def _eval_grid(board: Board, stone: Stone) -> float:
    """Evaluate board using the numpy-grid JIT path."""
    from gomoku_brain.evaluate_jit import (
        evaluate_grid,
        get_dirs,
    )
    from gomoku_brain.evaluate_jit import (
        get_fork_bonus as gfb,
    )
    from gomoku_brain.evaluate_jit import (
        get_fork_threshold as gft,
    )
    from gomoku_brain.evaluate_jit import (
        get_open_three_bonus as gotb,
    )
    from gomoku_brain.evaluate_jit import (
        get_pattern_table as gpt,
    )

    grid = board_to_grid(board)
    occ = occupied_coords(board)
    return evaluate_grid(grid, board.size, occ, int(stone), get_dirs(), gpt(), gft(), gfb(), gotb())


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------


class TestConversion:
    def test_empty_board_roundtrip(self):
        b = Board(size=9)
        grid = np.array(b.grid, dtype=np.int8)
        black, white = grid_to_bitboard(grid, 9)
        assert (black == 0).all()
        assert (white == 0).all()
        out = bitboard_to_grid(black, white, 9)
        np.testing.assert_array_equal(grid, out)

    def test_with_stones_roundtrip(self):
        b = Board(size=15)
        b = _place(b, [(7, 7), (3, 10), (14, 0)], Stone.BLACK)
        b = _place(b, [(0, 14), (7, 8)], Stone.WHITE)
        grid = np.array(b.grid, dtype=np.int8)
        black, white = grid_to_bitboard(grid, 15)
        out = bitboard_to_grid(black, white, 15)
        np.testing.assert_array_equal(grid, out)

    def test_board_to_bitboard_helper(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        black, white, size = board_to_bitboard(b)
        assert size == 9
        assert black[4] == (1 << 4)
        assert (white == 0).all()

    def test_bitboard_state_from_board(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        bb = BitboardState.from_board(b)
        assert bb.at(4, 4) is Stone.BLACK
        assert bb.at(0, 0) is Stone.EMPTY
        assert bb.size == 9


# ---------------------------------------------------------------------------
# BitboardState wrapper tests
# ---------------------------------------------------------------------------


class TestBitboardState:
    def test_empty(self):
        bb = BitboardState.empty(15)
        assert bb.size == 15
        assert len(bb.occupied) == 0
        assert bb.at(0, 0) is Stone.EMPTY

    def test_place_and_at(self):
        bb = BitboardState.empty(15)
        bb2 = bb.place_stone(Move(row=7, col=7), Stone.BLACK)
        assert bb2.at(7, 7) is Stone.BLACK
        assert bb.at(7, 7) is Stone.EMPTY  # original unchanged

    def test_is_valid_move(self):
        bb = BitboardState.empty(9)
        bb2 = bb.place_stone(Move(row=4, col=4), Stone.BLACK)
        assert not bb2.is_valid_move(Move(row=4, col=4))
        assert bb2.is_valid_move(Move(row=4, col=5))
        assert not bb2.is_valid_move(Move(row=0, col=9))  # out of bounds col

    def test_occupied(self):
        bb = BitboardState.empty(9)
        bb = bb.place_stone(Move(row=0, col=0), Stone.BLACK)
        bb = bb.place_stone(Move(row=8, col=8), Stone.WHITE)
        assert bb.occupied == {(0, 0), (8, 8)}

    def test_is_full(self):
        # Fill a tiny 5x5 board
        bb = BitboardState.empty(5)
        for r in range(5):
            for c in range(5):
                stone = Stone.BLACK if (r + c) % 2 == 0 else Stone.WHITE
                bb = bb.place_stone(Move(row=r, col=c), stone)
        assert bb.is_full()


# ---------------------------------------------------------------------------
# Win detection tests
# ---------------------------------------------------------------------------


class TestWinDetection:
    def test_horizontal_win(self):
        b = Board(size=15)
        b = _place(b, [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)], Stone.BLACK)
        black, white, _ = board_to_bitboard(b)
        assert check_winner_bb(black, 15, 5)
        assert check_winner_at_bb(black, 15, 7, 5, 5)
        assert not check_winner_bb(white, 15, 5)

    def test_vertical_win(self):
        b = Board(size=15)
        b = _place(b, [(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)], Stone.WHITE)
        _, white, _ = board_to_bitboard(b)
        assert check_winner_bb(white, 15, 5)
        assert check_winner_at_bb(white, 15, 5, 7, 5)

    def test_diagonal_down_win(self):
        b = Board(size=15)
        b = _place(b, [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)], Stone.BLACK)
        black, _, _ = board_to_bitboard(b)
        assert check_winner_bb(black, 15, 5)
        assert check_winner_at_bb(black, 15, 5, 5, 5)

    def test_diagonal_up_win(self):
        b = Board(size=15)
        b = _place(b, [(7, 3), (6, 4), (5, 5), (4, 6), (3, 7)], Stone.BLACK)
        black, _, _ = board_to_bitboard(b)
        assert check_winner_bb(black, 15, 5)
        assert check_winner_at_bb(black, 15, 5, 5, 5)

    def test_no_win_four(self):
        b = Board(size=15)
        b = _place(b, [(7, 3), (7, 4), (7, 5), (7, 6)], Stone.BLACK)
        black, _, _ = board_to_bitboard(b)
        assert not check_winner_bb(black, 15, 5)

    def test_six_in_a_row_is_win(self):
        """Freestyle gomoku: 6+ in a row still wins."""
        b = Board(size=15)
        b = _place(b, [(7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)], Stone.BLACK)
        black, _, _ = board_to_bitboard(b)
        assert check_winner_bb(black, 15, 5)


# ---------------------------------------------------------------------------
# Candidate generation tests
# ---------------------------------------------------------------------------


class TestCandidates:
    def test_candidates_near_center(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        black, white, _ = board_to_bitboard(b)
        cands = get_candidates_bb(black, white, 9, 2)
        # (4,4) itself should NOT be a candidate
        cand_set = {(int(cands[i, 0]), int(cands[i, 1])) for i in range(cands.shape[0])}
        assert (4, 4) not in cand_set
        # But neighbours should be
        assert (4, 5) in cand_set
        assert (3, 4) in cand_set
        # Radius-2 diagonals
        assert (2, 2) in cand_set
        assert (6, 6) in cand_set

    def test_candidates_match_grid_version(self):
        """Bitboard candidates should produce the same set as grid-based."""
        from gomoku_brain.evaluate_jit import get_candidates_grid

        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5), (3, 3)], Stone.BLACK)
        b = _place(b, [(5, 5)], Stone.WHITE)

        black, white, _ = board_to_bitboard(b)
        bb_cands = get_candidates_bb(black, white, 9, 2)
        bb_set = {(int(bb_cands[i, 0]), int(bb_cands[i, 1])) for i in range(bb_cands.shape[0])}

        grid = board_to_grid(b)
        occ = occupied_coords(b)
        grid_cands = get_candidates_grid(grid, 9, occ, 2)
        grid_set = {(int(grid_cands[i, 0]), int(grid_cands[i, 1])) for i in range(grid_cands.shape[0])}

        assert bb_set == grid_set


# ---------------------------------------------------------------------------
# Evaluation parity tests
# ---------------------------------------------------------------------------


class TestEvaluationParity:
    """Verify bitboard evaluation matches the numpy-grid JIT evaluation."""

    def test_empty_board(self):
        b = Board(size=9)
        assert _eval_bb(b, Stone.BLACK) == _eval_grid(b, Stone.BLACK)

    def test_single_stone(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        assert _eval_bb(b, Stone.BLACK) == pytest.approx(_eval_grid(b, Stone.BLACK), abs=1.0)

    def test_symmetric_position(self):
        """Same position evaluated for BLACK and WHITE should be negated."""
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5), (4, 6)], Stone.BLACK)
        b = _place(b, [(5, 4), (5, 5)], Stone.WHITE)
        bb_black = _eval_bb(b, Stone.BLACK)
        bb_white = _eval_bb(b, Stone.WHITE)
        grid_black = _eval_grid(b, Stone.BLACK)
        grid_white = _eval_grid(b, Stone.WHITE)
        assert bb_black == pytest.approx(grid_black, abs=1.0)
        assert bb_white == pytest.approx(grid_white, abs=1.0)

    def test_complex_mid_game(self):
        b = Board(size=15)
        b = _place(b, [(7, 7), (7, 8), (7, 9), (6, 7), (8, 7)], Stone.BLACK)
        b = _place(b, [(6, 8), (8, 8), (6, 6), (7, 6)], Stone.WHITE)
        assert _eval_bb(b, Stone.BLACK) == pytest.approx(_eval_grid(b, Stone.BLACK), abs=1.0)
        assert _eval_bb(b, Stone.WHITE) == pytest.approx(_eval_grid(b, Stone.WHITE), abs=1.0)


# ---------------------------------------------------------------------------
# Threat detection tests
# ---------------------------------------------------------------------------


class TestThreatDetection:
    def test_four_threat(self):
        b = Board(size=15)
        b = _place(b, [(7, 5), (7, 6), (7, 7), (7, 8)], Stone.BLACK)
        black, _, _ = board_to_bitboard(b)
        # Placing at (7,4) or (7,9) should complete 5 → four-threat
        test_b_left = black.copy()
        test_b_left[7] |= np.uint32(1) << np.uint32(4)
        assert is_four_threat_bb(test_b_left, 15, 7, 4)
        test_b_right = black.copy()
        test_b_right[7] |= np.uint32(1) << np.uint32(9)
        assert is_four_threat_bb(test_b_right, 15, 7, 9)

    def test_not_four_threat(self):
        b = Board(size=15)
        b = _place(b, [(7, 5), (7, 6), (7, 8)], Stone.BLACK)
        black, _, _ = board_to_bitboard(b)
        test_b = black.copy()
        test_b[7] |= np.uint32(1) << np.uint32(0)
        assert not is_four_threat_bb(test_b, 15, 7, 0)

    def test_threat_moves(self):
        b = Board(size=15)
        b = _place(b, [(7, 5), (7, 6), (7, 7), (7, 8)], Stone.BLACK)
        black, white, _ = board_to_bitboard(b)
        cands = get_candidates_bb(black, white, 15, 2)
        threats = get_threat_moves_bb(black, white, 15, cands, 1)
        threat_set = {(int(threats[i, 0]), int(threats[i, 1])) for i in range(threats.shape[0])}
        assert (7, 4) in threat_set
        assert (7, 9) in threat_set


# ---------------------------------------------------------------------------
# Occupied extraction test
# ---------------------------------------------------------------------------


class TestOccupied:
    def test_occupied_from_bb(self):
        b = Board(size=9)
        b = _place(b, [(0, 0), (4, 4), (8, 8)], Stone.BLACK)
        b = _place(b, [(1, 1)], Stone.WHITE)
        black, white, _ = board_to_bitboard(b)
        occ = occupied_from_bb(black, white, 9)
        occ_set = {(int(occ[i, 0]), int(occ[i, 1])) for i in range(occ.shape[0])}
        assert occ_set == {(0, 0), (1, 1), (4, 4), (8, 8)}
