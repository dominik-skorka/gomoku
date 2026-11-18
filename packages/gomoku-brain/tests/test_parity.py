"""JIT vs pure-Python parity property tests.

Each evaluator / detector has three implementations: pure Python on a
Pydantic :class:`Board`, a Numba grid JIT on an ``int8[size, size]`` array,
and a Numba bitboard JIT on a pair of ``uint32[size]`` arrays.  They're
selected by ``GomokuSettings`` at runtime — a silent divergence here would
mean the search sees a *different game* depending on configuration.

These tests feed many random positions through every path and assert
equality.  They complement the single-scenario sanity checks in
``test_evaluate_jit.py`` by covering the fork/open-three/defence paths
with positions the author would never think to hand-craft.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner
from gomoku_brain.bitboard_jit import (
    check_winner_at_bb,
    evaluate_bb,
    get_candidates_bb,
)
from gomoku_brain.bitboard_jit import (
    get_fork_bonus as bb_fork_bonus,
)
from gomoku_brain.bitboard_jit import (
    get_fork_threshold as bb_fork_threshold,
)
from gomoku_brain.bitboard_jit import (
    get_open_three_bonus as bb_open_three_bonus,
)
from gomoku_brain.bitboard_jit import (
    get_pattern_table as bb_pattern_table,
)
from gomoku_brain.candidates import _get_candidates_python
from gomoku_brain.evaluate import _evaluate_board_python
from gomoku_brain.evaluate_jit import (
    check_winner_grid,
    evaluate_grid,
    find_four_threats_grid,
    find_three_threats_grid,
    get_candidates_grid,
    get_dirs,
    get_fork_bonus,
    get_fork_threshold,
    get_open_three_bonus,
    get_pattern_table,
)
from gomoku_brain.grid_utils import board_to_bitboard, board_to_grid, occupied_coords
from gomoku_brain.threat import _find_four_threats_python, _find_three_threats_python

# ---------------------------------------------------------------------------
# Random-position helpers
# ---------------------------------------------------------------------------


def _random_board(seed: int, size: int, n_stones: int) -> Board:
    """Build a Board with *n_stones* random moves alternating BLACK/WHITE."""
    rng = random.Random(seed)
    b = Board(size=size)
    side = Stone.BLACK
    placed = 0
    while placed < n_stones:
        r, c = rng.randrange(size), rng.randrange(size)
        if (r, c) in b.occupied:
            continue
        b = b.place_stone(Move(row=r, col=c), side)
        side = side.opponent()
        placed += 1
    return b


# Each param is (seed, size, n_stones).  We keep sizes <= 15 so bitboards fit
# in uint32 and the tests stay fast.
_RANDOM_POSITIONS: list[tuple[int, int, int]] = [
    (1, 9, 5),
    (2, 9, 15),
    (3, 11, 20),
    (4, 11, 30),
    (5, 13, 25),
    (6, 15, 10),
    (7, 15, 30),
    (8, 15, 50),
]


# ---------------------------------------------------------------------------
# Canonicalisation helpers — convert heterogeneous return types to a
# comparable form (tuple of tuples of ints) before asserting equality.
# ---------------------------------------------------------------------------


def _canon_four_python(
    threats: list[tuple[Move, Move | None]],
) -> set[tuple[int, int, int, int]]:
    """Python returns one ``(attack, defence)`` per cell.  The cell is unique
    after per-cell aggregation, so a set of (ar, ac, dr, dc) is canonical."""
    out: set[tuple[int, int, int, int]] = set()
    for atk, defence in threats:
        if defence is None:
            out.add((atk.row, atk.col, -1, -1))
        else:
            out.add((atk.row, atk.col, defence.row, defence.col))
    return out


def _canon_four_grid(result: np.ndarray) -> set[tuple[int, int, int, int]]:
    return {
        (int(result[i, 0]), int(result[i, 1]), int(result[i, 2]), int(result[i, 3])) for i in range(result.shape[0])
    }


def _canon_three_python(
    threats: list[tuple[Move, list[Move]]],
) -> list[tuple[int, int, int, int, int, int]]:
    """Three-threats emit one entry per ``(cell, direction)`` pair.  The
    attack cell can repeat, and direction is implicit — so we compare as a
    sorted tuple of (ar, ac, d1r, d1c, d2r, d2c), which uniquely identifies
    one entry because any cell+direction pair has a unique pair of open
    ends."""
    return sorted(
        (
            atk.row,
            atk.col,
            defs[0].row,
            defs[0].col,
            defs[1].row,
            defs[1].col,
        )
        for atk, defs in threats
    )


def _canon_three_grid(result: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    return sorted(
        (
            int(result[i, 0]),
            int(result[i, 1]),
            int(result[i, 2]),
            int(result[i, 3]),
            int(result[i, 4]),
            int(result[i, 5]),
        )
        for i in range(result.shape[0])
    )


# ---------------------------------------------------------------------------
# evaluate — Python vs grid-JIT vs bitboard-JIT
# ---------------------------------------------------------------------------


class TestEvaluateParity:
    """Three evaluators, identical math — identical result."""

    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    @pytest.mark.parametrize("stone", [Stone.BLACK, Stone.WHITE])
    def test_grid_jit_matches_python(self, seed, size, n, stone):
        b = _random_board(seed, size, n)
        py = _evaluate_board_python(b, stone)
        jit = evaluate_grid(
            board_to_grid(b),
            size,
            occupied_coords(b),
            int(stone),
            get_dirs(),
            get_pattern_table(),
            get_fork_threshold(),
            get_fork_bonus(),
            get_open_three_bonus(),
        )
        assert jit == pytest.approx(py)

    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    @pytest.mark.parametrize("stone", [Stone.BLACK, Stone.WHITE])
    def test_bitboard_jit_matches_python(self, seed, size, n, stone):
        b = _random_board(seed, size, n)
        py = _evaluate_board_python(b, stone)
        black, white, bb_size = board_to_bitboard(b)
        bb = evaluate_bb(
            black,
            white,
            bb_size,
            int(stone),
            bb_pattern_table(),
            bb_fork_threshold(),
            bb_fork_bonus(),
            bb_open_three_bonus(),
        )
        assert bb == pytest.approx(py)


# ---------------------------------------------------------------------------
# Candidate generation — Python vs grid-JIT vs bitboard-JIT
# ---------------------------------------------------------------------------


class TestCandidatesParity:
    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    @pytest.mark.parametrize("radius", [1, 2])
    def test_grid_jit_matches_python(self, seed, size, n, radius):
        b = _random_board(seed, size, n)
        py = {(m.row, m.col) for m in _get_candidates_python(b, radius)}
        result = get_candidates_grid(board_to_grid(b), size, occupied_coords(b), radius)
        jit = {(int(result[i, 0]), int(result[i, 1])) for i in range(result.shape[0])}
        assert jit == py

    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    @pytest.mark.parametrize("radius", [1, 2])
    def test_bitboard_jit_matches_python(self, seed, size, n, radius):
        b = _random_board(seed, size, n)
        py = {(m.row, m.col) for m in _get_candidates_python(b, radius)}
        black, white, bb_size = board_to_bitboard(b)
        result = get_candidates_bb(black, white, bb_size, radius)
        bb = {(int(result[i, 0]), int(result[i, 1])) for i in range(result.shape[0])}
        assert bb == py

    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    def test_grid_jit_sorted_row_major(self, seed, size, n):
        """The JIT kernel is documented as emitting sorted output — several
        callers rely on this for reproducible move ordering."""
        b = _random_board(seed, size, n)
        result = get_candidates_grid(board_to_grid(b), size, occupied_coords(b), 2)
        coords = [(int(result[i, 0]), int(result[i, 1])) for i in range(result.shape[0])]
        assert coords == sorted(coords)


# ---------------------------------------------------------------------------
# Four-threat detection — Python vs grid-JIT
# ---------------------------------------------------------------------------


class TestFourThreatsParity:
    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    @pytest.mark.parametrize("stone", [Stone.BLACK, Stone.WHITE])
    def test_grid_jit_matches_python(self, seed, size, n, stone):
        b = _random_board(seed, size, n)
        occ = occupied_coords(b)
        if occ.shape[0] == 0:
            return
        grid = board_to_grid(b)
        candidates = get_candidates_grid(grid, size, occ, 2)
        jit_result = find_four_threats_grid(grid, size, candidates, int(stone), get_dirs())

        py_threats = _find_four_threats_python(b, stone)

        assert _canon_four_grid(jit_result) == _canon_four_python(py_threats)


# ---------------------------------------------------------------------------
# Three-threat detection — Python vs grid-JIT
# ---------------------------------------------------------------------------


class TestThreeThreatsParity:
    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    @pytest.mark.parametrize("stone", [Stone.BLACK, Stone.WHITE])
    def test_grid_jit_matches_python(self, seed, size, n, stone):
        b = _random_board(seed, size, n)
        occ = occupied_coords(b)
        if occ.shape[0] == 0:
            return
        grid = board_to_grid(b)
        candidates = get_candidates_grid(grid, size, occ, 2)
        jit_result = find_three_threats_grid(grid, size, candidates, int(stone), get_dirs())

        py_threats = _find_three_threats_python(b, stone)

        assert _canon_three_grid(jit_result) == _canon_three_python(py_threats)


# ---------------------------------------------------------------------------
# Winner detection — rules.check_winner vs grid-JIT vs bitboard-JIT
# ---------------------------------------------------------------------------


def _winning_positions() -> list[tuple[str, Board, Move, Stone, bool]]:
    """Return a list of ``(name, board, last_move, stone, expected_is_win)``."""
    out: list[tuple[str, Board, Move, Stone, bool]] = []

    # Horizontal 5
    b = Board(size=15)
    for c in range(3, 8):
        b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
    out.append(("horiz-5", b, Move(row=7, col=5), Stone.BLACK, True))

    # Vertical 5
    b = Board(size=15)
    for r in range(3, 8):
        b = b.place_stone(Move(row=r, col=5), Stone.WHITE)
    out.append(("vert-5", b, Move(row=5, col=5), Stone.WHITE, True))

    # Diagonal ↘
    b = Board(size=15)
    for k in range(5):
        b = b.place_stone(Move(row=3 + k, col=3 + k), Stone.BLACK)
    out.append(("diag-down-5", b, Move(row=5, col=5), Stone.BLACK, True))

    # Diagonal ↗
    b = Board(size=15)
    for k in range(5):
        b = b.place_stone(Move(row=7 - k, col=3 + k), Stone.WHITE)
    out.append(("diag-up-5", b, Move(row=5, col=5), Stone.WHITE, True))

    # Four in a row (no win)
    b = Board(size=15)
    for c in range(3, 7):
        b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
    out.append(("horiz-4-no-win", b, Move(row=7, col=5), Stone.BLACK, False))

    # Six in a row (overline — still a freestyle win since >= 5)
    b = Board(size=15)
    for c in range(2, 8):
        b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
    out.append(("overline-6", b, Move(row=7, col=5), Stone.BLACK, True))

    return out


class TestCheckWinnerParity:
    @pytest.mark.parametrize("name,board,last_move,stone,expected", _winning_positions())
    def test_rules_agrees(self, name, board, last_move, stone, expected):
        """Baseline: pure-Python rules.check_winner gives the expected result."""
        result = check_winner(board, last_move)
        if expected:
            assert result is not None
            assert result.winner is stone
        else:
            assert result is None

    @pytest.mark.parametrize("name,board,last_move,stone,expected", _winning_positions())
    def test_grid_jit_agrees(self, name, board, last_move, stone, expected):
        grid = board_to_grid(board)
        dirs = get_dirs()
        assert check_winner_grid(grid, board.size, last_move.row, last_move.col, 5, dirs) is expected

    @pytest.mark.parametrize("name,board,last_move,stone,expected", _winning_positions())
    def test_bitboard_jit_agrees(self, name, board, last_move, stone, expected):
        black, white, bb_size = board_to_bitboard(board)
        rows = black if stone is Stone.BLACK else white
        assert check_winner_at_bb(rows, bb_size, last_move.row, last_move.col, 5) is expected

    @pytest.mark.parametrize("seed,size,n", _RANDOM_POSITIONS)
    def test_random_positions_no_false_wins(self, seed, size, n):
        """On a random position we can't predict whether a win exists, but
        whatever ``rules.check_winner`` says must match both JIT kernels for
        every occupied cell (checked from the stone-at-cell perspective)."""
        b = _random_board(seed, size, n)
        grid = board_to_grid(b)
        black, white, _ = board_to_bitboard(b)
        dirs = get_dirs()
        for r, c in sorted(b.occupied):
            stone = b.at(r, c)
            rules_says = check_winner(b, Move(row=r, col=c)) is not None
            grid_says = bool(check_winner_grid(grid, size, r, c, 5, dirs))
            rows = black if stone is Stone.BLACK else white
            bb_says = bool(check_winner_at_bb(rows, size, r, c, 5))
            assert rules_says == grid_says == bb_says, (
                f"disagreement at ({r},{c}) on seed={seed} size={size}: "
                f"rules={rules_says} grid={grid_says} bb={bb_says}"
            )
