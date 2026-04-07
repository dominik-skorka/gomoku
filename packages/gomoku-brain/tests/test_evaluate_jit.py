"""Tests for the Numba JIT evaluation path.

Verifies that the JIT-compiled ``evaluate_grid`` produces identical results
to the pure-Python ``evaluate_board`` and that conversion helpers work.
"""

import numpy as np
import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.evaluate_jit import (
    check_winner_grid,
    evaluate_grid,
    get_candidates_grid,
    get_dirs,
    get_fork_bonus,
    get_fork_threshold,
    get_open_three_bonus,
    get_pattern_table,
    get_threat_moves_grid,
    is_four_threat,
    occupied_from_grid,
)
from gomoku_brain.grid_utils import _GridBoard, board_to_grid, grid_to_board, occupied_coords

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _place(board: Board, moves: list[tuple[int, int]], stone: Stone) -> Board:
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
    return board


def _eval_via_jit(board: Board, stone: Stone) -> float:
    """Evaluate board using JIT path directly."""
    return evaluate_grid(
        board_to_grid(board),
        board.size,
        occupied_coords(board),
        int(stone),
        get_dirs(),
        get_pattern_table(),
        get_fork_threshold(),
        get_fork_bonus(),
        get_open_three_bonus(),
    )


# ---------------------------------------------------------------------------
# grid_utils
# ---------------------------------------------------------------------------


class TestGridUtils:
    def test_board_to_grid_empty(self):
        b = Board(size=9)
        grid = board_to_grid(b)
        assert grid.shape == (9, 9)
        assert grid.dtype == np.int8
        assert (grid == 0).all()

    def test_board_to_grid_with_stones(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)
        grid = board_to_grid(b)
        assert grid[4, 4] == int(Stone.BLACK)
        assert grid[0, 0] == int(Stone.WHITE)
        assert grid[1, 1] == 0

    def test_occupied_coords_empty(self):
        b = Board(size=9)
        occ = occupied_coords(b)
        assert occ.shape == (0, 2)
        assert occ.dtype == np.int32

    def test_occupied_coords_matches_board(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (0, 0), (8, 8)], Stone.BLACK)
        occ = occupied_coords(b)
        assert occ.shape == (3, 2)
        occ_set = {(int(occ[i, 0]), int(occ[i, 1])) for i in range(occ.shape[0])}
        assert occ_set == {(4, 4), (0, 0), (8, 8)}


# ---------------------------------------------------------------------------
# JIT evaluation vs pure-Python
# ---------------------------------------------------------------------------


class TestEvaluateJit:
    def test_empty_board_zero(self):
        b = Board(size=9)
        assert _eval_via_jit(b, Stone.BLACK) == 0.0

    def test_single_stone_is_zero(self):
        """A lone stone has no consecutive neighbours, so score is 0."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        assert _eval_via_jit(b, Stone.BLACK) == 0.0

    def test_advantage_positive(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5), (4, 6)], Stone.BLACK)
        assert _eval_via_jit(b, Stone.BLACK) > 0

    def test_opponent_advantage_negative(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5), (4, 6), (4, 7)], Stone.WHITE)
        assert _eval_via_jit(b, Stone.BLACK) < 0

    def test_symmetry(self):
        """Score for BLACK on a BLACK-dominant board should be the negation of
        the score for WHITE on the same board (both see the same patterns)."""
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0)], Stone.WHITE)
        s_black = _eval_via_jit(b, Stone.BLACK)
        s_white = _eval_via_jit(b, Stone.WHITE)
        assert s_black == pytest.approx(-s_white)

    def test_matches_evaluate_board(self):
        """JIT path must produce the same result as :func:`evaluate_board`."""
        from gomoku_brain.evaluate import evaluate_board

        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (3, 5), (2, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (1, 1)], Stone.WHITE)
        expected = evaluate_board(b, Stone.BLACK)
        actual = _eval_via_jit(b, Stone.BLACK)
        assert actual == pytest.approx(expected)

    def test_fork_bonus_present(self):
        """A fork arrangement should score higher than scattered stones."""
        b_fork = Board(size=9)
        b_fork = _place(b_fork, [(4, 3), (4, 4), (3, 5), (2, 5)], Stone.BLACK)
        b_fork = _place(b_fork, [(0, 0)], Stone.WHITE)

        b_scattered = Board(size=9)
        b_scattered = _place(b_scattered, [(4, 3), (4, 4), (0, 7), (1, 7)], Stone.BLACK)
        b_scattered = _place(b_scattered, [(0, 0)], Stone.WHITE)

        assert _eval_via_jit(b_fork, Stone.BLACK) > _eval_via_jit(b_scattered, Stone.BLACK)


# ---------------------------------------------------------------------------
# Tier 2: JIT candidate generation
# ---------------------------------------------------------------------------


class TestGetCandidatesGrid:
    def test_single_stone_radius_2(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        grid = board_to_grid(b)
        occ = occupied_coords(b)
        result = get_candidates_grid(grid, b.size, occ, 2)
        coords = {(int(result[i, 0]), int(result[i, 1])) for i in range(result.shape[0])}
        # Should include all empty cells within radius 2 of (4,4)
        assert (4, 5) in coords
        assert (3, 3) in coords
        assert (2, 2) in coords  # radius 2
        # Should NOT include the occupied cell itself
        assert (4, 4) not in coords
        # Should NOT include cells beyond radius 2
        assert (1, 1) not in coords

    def test_matches_python_candidates(self):
        from gomoku_brain.candidates import _get_candidates_python

        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (3, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (8, 8)], Stone.WHITE)

        python_moves = _get_candidates_python(b, 2)
        python_set = {(m.row, m.col) for m in python_moves}

        grid = board_to_grid(b)
        occ = occupied_coords(b)
        result = get_candidates_grid(grid, b.size, occ, 2)
        jit_set = {(int(result[i, 0]), int(result[i, 1])) for i in range(result.shape[0])}

        assert jit_set == python_set

    def test_sorted_output(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (0, 0), (8, 8)], Stone.BLACK)
        grid = board_to_grid(b)
        occ = occupied_coords(b)
        result = get_candidates_grid(grid, b.size, occ, 2)
        # Verify sorted by (row, col)
        for i in range(1, result.shape[0]):
            prev = (int(result[i - 1, 0]), int(result[i - 1, 1]))
            curr = (int(result[i, 0]), int(result[i, 1]))
            assert prev <= curr


# ---------------------------------------------------------------------------
# Tier 2: JIT threat detection
# ---------------------------------------------------------------------------


class TestIsFourThreat:
    def test_horizontal_four(self):
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        grid = board_to_grid(b)
        dirs = get_dirs()
        # Placing at (4,6) creates 4 in a row
        assert is_four_threat(grid, 9, 4, 6, int(Stone.BLACK), dirs) is True
        # Placing at (4,2) also creates 4 in a row
        assert is_four_threat(grid, 9, 4, 2, int(Stone.BLACK), dirs) is True

    def test_not_four_with_two(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5)], Stone.BLACK)
        grid = board_to_grid(b)
        dirs = get_dirs()
        # Only 3 in a row — not a four-threat
        assert is_four_threat(grid, 9, 4, 6, int(Stone.BLACK), dirs) is False

    def test_diagonal_four(self):
        b = Board(size=9)
        b = _place(b, [(2, 2), (3, 3), (4, 4)], Stone.WHITE)
        grid = board_to_grid(b)
        dirs = get_dirs()
        assert is_four_threat(grid, 9, 5, 5, int(Stone.WHITE), dirs) is True

    def test_wrong_stone_no_threat(self):
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        grid = board_to_grid(b)
        dirs = get_dirs()
        # WHITE placing here doesn't make 4
        assert is_four_threat(grid, 9, 4, 6, int(Stone.WHITE), dirs) is False


class TestGetThreatMovesGrid:
    def test_finds_four_threats(self):
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0)], Stone.WHITE)
        grid = board_to_grid(b)
        occ = occupied_coords(b)
        cands = get_candidates_grid(grid, b.size, occ, 2)
        dirs = get_dirs()
        threats = get_threat_moves_grid(grid, b.size, cands, int(Stone.BLACK), int(Stone.WHITE), dirs)
        threat_set = {(int(threats[i, 0]), int(threats[i, 1])) for i in range(threats.shape[0])}
        # (4,2) and (4,6) extend the line to 4
        assert (4, 2) in threat_set
        assert (4, 6) in threat_set

    def test_no_threats_on_sparse_board(self):
        b = Board(size=9)
        b = _place(b, [(4, 4)], Stone.BLACK)
        b = _place(b, [(0, 0)], Stone.WHITE)
        grid = board_to_grid(b)
        occ = occupied_coords(b)
        cands = get_candidates_grid(grid, b.size, occ, 2)
        dirs = get_dirs()
        threats = get_threat_moves_grid(grid, b.size, cands, int(Stone.BLACK), int(Stone.WHITE), dirs)
        assert threats.shape[0] == 0


# ---------------------------------------------------------------------------
# Tier 3: check_winner_grid, occupied_from_grid, _GridBoard, grid_to_board
# ---------------------------------------------------------------------------


class TestCheckWinnerGrid:
    def test_horizontal_win(self):
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7)], Stone.BLACK)
        grid = board_to_grid(b)
        dirs = get_dirs()
        assert check_winner_grid(grid, 9, 4, 5, 5, dirs) is True

    def test_no_win_with_four(self):
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5), (4, 6)], Stone.BLACK)
        grid = board_to_grid(b)
        dirs = get_dirs()
        assert check_winner_grid(grid, 9, 4, 5, 5, dirs) is False

    def test_diagonal_win(self):
        b = Board(size=9)
        b = _place(b, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], Stone.WHITE)
        grid = board_to_grid(b)
        dirs = get_dirs()
        assert check_winner_grid(grid, 9, 2, 2, 5, dirs) is True

    def test_empty_cell_returns_false(self):
        b = Board(size=9)
        grid = board_to_grid(b)
        dirs = get_dirs()
        assert check_winner_grid(grid, 9, 4, 4, 5, dirs) is False


class TestOccupiedFromGrid:
    def test_empty_board(self):
        grid = np.zeros((9, 9), dtype=np.int8)
        occ = occupied_from_grid(grid, 9)
        assert occ.shape == (0, 2)

    def test_matches_board_occupied(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (0, 0), (8, 8)], Stone.BLACK)
        grid = board_to_grid(b)
        occ = occupied_from_grid(grid, 9)
        occ_set = {(int(occ[i, 0]), int(occ[i, 1])) for i in range(occ.shape[0])}
        assert occ_set == {(4, 4), (0, 0), (8, 8)}
        assert occ.shape == (3, 2)

    def test_sorted_row_major(self):
        b = Board(size=9)
        b = _place(b, [(8, 0), (0, 8), (4, 4)], Stone.BLACK)
        grid = board_to_grid(b)
        occ = occupied_from_grid(grid, 9)
        for i in range(1, occ.shape[0]):
            prev = (int(occ[i - 1, 0]), int(occ[i - 1, 1]))
            curr = (int(occ[i, 0]), int(occ[i, 1]))
            assert prev < curr


class TestGridBoard:
    def test_place_stone_returns_grid_board(self):
        b = Board(size=9)
        grid = board_to_grid(b)
        gb = _GridBoard(grid, 9)
        gb2 = gb.place_stone(Move(row=4, col=4), Stone.BLACK)
        assert isinstance(gb2, _GridBoard)
        assert gb2.grid[4, 4] == int(Stone.BLACK)
        # Original unchanged
        assert gb.grid[4, 4] == 0

    def test_at_returns_stone(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        gb = _GridBoard(board_to_grid(b), 9)
        assert gb.at(4, 4) is Stone.BLACK
        assert gb.at(0, 0) is Stone.EMPTY

    def test_occupied_property(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (0, 0)], Stone.BLACK)
        gb = _GridBoard(board_to_grid(b), 9)
        assert gb.occupied == {(4, 4), (0, 0)}

    def test_is_valid_move(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        gb = _GridBoard(board_to_grid(b), 9)
        assert gb.is_valid_move(Move(row=0, col=0)) is True
        assert gb.is_valid_move(Move(row=4, col=4)) is False

    def test_is_full(self):
        # 3x3 board fully filled
        grid = np.ones((3, 3), dtype=np.int8)
        gb = _GridBoard(grid, 3)
        assert gb.is_full() is True
        grid2 = np.zeros((3, 3), dtype=np.int8)
        gb2 = _GridBoard(grid2, 3)
        assert gb2.is_full() is False

    def test_board_to_grid_returns_reference(self):
        grid = np.zeros((9, 9), dtype=np.int8)
        gb = _GridBoard(grid, 9)
        # board_to_grid on _GridBoard returns the grid reference (no copy)
        assert board_to_grid(gb) is gb.grid

    def test_occupied_coords_uses_jit(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (0, 0), (8, 8)], Stone.BLACK)
        gb = _GridBoard(board_to_grid(b), 9)
        occ = occupied_coords(gb)
        occ_set = {(int(occ[i, 0]), int(occ[i, 1])) for i in range(occ.shape[0])}
        assert occ_set == {(4, 4), (0, 0), (8, 8)}


class TestGridToBoard:
    def test_round_trip(self):
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0)], Stone.WHITE)
        grid = board_to_grid(b)
        b2 = grid_to_board(grid, 9)
        assert b2.size == 9
        assert b2.at(4, 4) is Stone.BLACK
        assert b2.at(0, 0) is Stone.WHITE
        assert b2.at(1, 1) is Stone.EMPTY
        assert b2.occupied == b.occupied

    def test_empty_board(self):
        grid = np.zeros((9, 9), dtype=np.int8)
        b = grid_to_board(grid, 9)
        assert len(b.occupied) == 0
