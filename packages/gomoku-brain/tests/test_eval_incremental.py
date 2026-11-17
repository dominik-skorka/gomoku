"""Parity tests for incremental evaluation on ``SearchState``.

The incremental path (``evaluate._eval_state_black`` backed by the
``evaluate_delta_grid`` kernel) must be *bit-identical* to a full
``evaluate_grid`` recomputation: every term is an integer accumulated
in float64, so any mismatch indicates a locality bug in the delta
kernel, not floating-point drift.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.evaluate import _eval_state_black, _should_use_jit, evaluate_board
from gomoku_brain.search_state import SearchState

pytestmark = pytest.mark.skipif(not _should_use_jit(), reason="incremental path requires JIT")


def _full_eval_black(grid: np.ndarray, size: int) -> float:
    """Ground truth: full evaluate_grid of the position, black-positive."""
    from gomoku_brain.evaluate_jit import (
        evaluate_grid,
        get_dirs,
        get_fork_bonus,
        get_fork_threshold,
        get_open_three_bonus,
        get_pattern_table,
    )

    rows, cols = np.nonzero(grid)
    occ = np.empty((rows.shape[0], 2), dtype=np.int32)
    occ[:, 0] = rows
    occ[:, 1] = cols
    return float(
        evaluate_grid(
            grid,
            size,
            occ,
            1,
            get_dirs(),
            get_pattern_table(),
            get_fork_threshold(),
            get_fork_bonus(),
            get_open_three_bonus(),
        )
    )


def _delta(grid: np.ndarray, size: int, row: int, col: int) -> float:
    from gomoku_brain.evaluate_jit import (
        evaluate_delta_grid,
        get_dirs,
        get_fork_bonus,
        get_fork_threshold,
        get_open_three_bonus,
        get_pattern_table,
    )

    return float(
        evaluate_delta_grid(
            grid,
            size,
            row,
            col,
            get_dirs(),
            get_pattern_table(),
            get_fork_threshold(),
            get_fork_bonus(),
            get_open_three_bonus(),
        )
    )


class TestDeltaKernel:
    def test_delta_equals_full_diff_random(self):
        """delta == E(after) - E(before) on random dense positions."""
        rng = random.Random(42)
        size = 15
        for _ in range(40):
            grid = np.zeros((size, size), dtype=np.int8)
            n_stones = rng.randint(0, 60)
            cells = rng.sample([(r, c) for r in range(size) for c in range(size)], n_stones + 1)
            for r, c in cells[:-1]:
                grid[r, c] = rng.choice([1, 2])
            r, c = cells[-1]
            before = _full_eval_black(grid, size)
            grid[r, c] = rng.choice([1, 2])
            after = _full_eval_black(grid, size)
            assert _delta(grid, size, r, c) == after - before

    def test_delta_long_runs_and_edges(self):
        """Overlines and board-edge runs exercise the locality bounds."""
        size = 15
        grid = np.zeros((size, size), dtype=np.int8)
        # Long horizontal black run hugging the edge, gap at (0, 6).
        for c in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]:
            grid[0, c] = 1
        # White wall nearby.
        for c in range(4, 9):
            grid[1, c] = 2
        before = _full_eval_black(grid, size)
        grid[0, 6] = 1  # fill the gap → 11-long overline
        after = _full_eval_black(grid, size)
        assert _delta(grid, size, 0, 6) == after - before

    def test_delta_restores_grid(self):
        size = 15
        grid = np.zeros((size, size), dtype=np.int8)
        grid[7, 7] = 1
        grid[7, 8] = 2
        snapshot = grid.copy()
        _delta(grid, size, 7, 8)
        assert np.array_equal(grid, snapshot)


class TestIncrementalStateEval:
    def _random_state_walk(self, seed: int, size: int = 15) -> None:
        """Random make/unmake walk asserting incremental == full at random points."""
        rng = random.Random(seed)
        board = Board(size=size)
        stone = Stone.BLACK
        # Random prefix on the Board so the state starts mid-game.
        for _ in range(rng.randint(0, 10)):
            empties = [(r, c) for r in range(size) for c in range(size) if board.at(r, c) is Stone.EMPTY]
            r, c = rng.choice(empties)
            board = board.place_stone(Move(row=r, col=c), stone)
            stone = stone.opponent()

        state = SearchState.from_board(board, stone)
        for _ in range(120):
            action = rng.random()
            if action < 0.55 or state.ply == 0:
                empties = [(r, c) for r in range(size) for c in range(size) if state.is_empty(r, c)]
                r, c = rng.choice(empties)
                state.make(r, c, rng.choice([Stone.BLACK, Stone.WHITE]))
            else:
                state.unmake()
            if rng.random() < 0.5:
                got = _eval_state_black(state)
                expected = _full_eval_black(state.grid.copy(), size)
                assert got == expected, f"ply={state.ply} got={got} expected={expected}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_walk_parity(self, seed: int):
        self._random_state_walk(seed)

    def test_evaluate_board_sign_and_clamp(self):
        """evaluate_board on a SearchState matches the black-positive cache
        with sign flip for white and the ±99_000 clamp."""
        board = Board(size=15)
        stone = Stone.BLACK
        for r, c in [(7, 7), (7, 8), (8, 7), (6, 8), (8, 8), (6, 7)]:
            board = board.place_stone(Move(row=r, col=c), stone)
            stone = stone.opponent()
        state = SearchState.from_board(board, Stone.BLACK)
        e = _eval_state_black(state)
        clamp = lambda v: max(-99_000.0, min(99_000.0, v))  # noqa: E731
        assert evaluate_board(state, Stone.BLACK) == clamp(e)
        assert evaluate_board(state, Stone.WHITE) == clamp(-e)

    def test_walkback_gap_fill(self):
        """Several makes without intermediate evals — the gap fill must
        strip and re-apply moves correctly and leave the grid intact."""
        board = Board(size=15)
        state = SearchState.from_board(board, Stone.BLACK)
        _eval_state_black(state)  # cache ply 0
        moves = [(7, 7, Stone.BLACK), (7, 8, Stone.WHITE), (8, 7, Stone.BLACK)]
        for r, c, s in moves:
            state.make(r, c, s)
        snapshot = state.grid.copy()
        got = _eval_state_black(state)
        assert np.array_equal(state.grid, snapshot)
        assert got == _full_eval_black(state.grid.copy(), 15)
        # Intermediate plies were back-filled along the way.
        state.unmake()
        assert _eval_state_black(state) == _full_eval_black(state.grid.copy(), 15)
