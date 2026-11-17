"""Conversion helpers between Board (Pydantic) and numpy arrays for JIT code."""

from __future__ import annotations

import numpy as np
from gomoku.board import Board
from gomoku.models import Move, Stone
from numpy.typing import NDArray


class _GridBoard:
    """Lightweight numpy-backed board for JIT-accelerated search.

    Duck-type compatible with :class:`Board` for the operations used inside
    the alpha-beta search, but *much* cheaper to copy: ``place_stone`` does a
    225-byte ``numpy.copy`` instead of a Python list-of-lists copy + set copy +
    Pydantic ``model_construct``.
    """

    __slots__ = ("_occ", "grid", "size")

    def __init__(self, grid: NDArray[np.int8], size: int) -> None:
        self.grid = grid
        self.size = size
        self._occ: set[tuple[int, int]] | None = None

    def place_stone(self, move: Move, stone: Stone) -> _GridBoard:
        g = self.grid.copy()
        g[move.row, move.col] = int(stone)
        return _GridBoard(g, self.size)

    def at(self, row: int, col: int) -> Stone:
        return Stone(int(self.grid[row, col]))

    @property
    def occupied(self) -> set[tuple[int, int]]:
        if self._occ is None:
            rows, cols = np.nonzero(self.grid)
            self._occ = {(int(rows[i]), int(cols[i])) for i in range(len(rows))}
        return self._occ

    def is_valid_move(self, move: Move) -> bool:
        return 0 <= move.row < self.size and 0 <= move.col < self.size and int(self.grid[move.row, move.col]) == 0

    def is_full(self) -> bool:
        return int(np.count_nonzero(self.grid)) == self.size * self.size


def board_to_grid(board: Board | _GridBoard) -> NDArray[np.int8]:
    """Convert a Board (or _GridBoard) to a numpy int8 2-D array (size x size)."""
    from gomoku_brain.bitboard import BitboardState
    from gomoku_brain.search_state import SearchState

    if isinstance(board, BitboardState | SearchState):
        return board.grid
    if isinstance(board, _GridBoard):
        return board.grid
    return np.array(board.grid, dtype=np.int8)


def occupied_coords(board: Board | _GridBoard) -> NDArray[np.int32]:
    """Return occupied cell coordinates as an (N, 2) int32 array.

    Each row is ``[row, col]``.  Returns an empty ``(0, 2)`` array when the
    board is empty.

    For :class:`SearchState` this is a zero-cost slice of the
    incrementally-maintained occupied stack (row-major seed + move
    order); callers must treat it as read-only and order-insensitive.
    """
    from gomoku_brain.bitboard import BitboardState
    from gomoku_brain.search_state import SearchState

    if isinstance(board, SearchState):
        return board._occ_arr[: board.occ_count]
    if isinstance(board, BitboardState):
        from gomoku_brain.bitboard_jit import occupied_from_bb

        return occupied_from_bb(board.black, board.white, board.size)
    if isinstance(board, _GridBoard):
        from gomoku_brain.evaluate_jit import occupied_from_grid

        return occupied_from_grid(board.grid, board.size)
    if not board.occupied:
        return np.empty((0, 2), dtype=np.int32)
    occ = np.array(sorted(board.occupied), dtype=np.int32)
    return occ


def board_to_bitboard(board: Board | _GridBoard) -> tuple[NDArray[np.uint32], NDArray[np.uint32], int]:
    """Convert any board type to bitboard arrays ``(black, white, size)``."""
    from gomoku_brain.bitboard import BitboardState
    from gomoku_brain.bitboard_jit import grid_to_bitboard
    from gomoku_brain.search_state import SearchState

    if isinstance(board, BitboardState | SearchState):
        return board.black, board.white, board.size
    grid = board_to_grid(board)
    size = board.size
    black, white = grid_to_bitboard(grid, size)
    return black, white, size


def grid_to_board(grid: NDArray[np.int8], size: int) -> Board:
    """Reconstruct a Pydantic Board from a numpy grid."""
    grid_list = [[Stone(int(grid[r, c])) for c in range(size)] for r in range(size)]
    occupied = {(r, c) for r in range(size) for c in range(size) if grid[r, c] != 0}
    return Board.model_construct(size=size, grid=grid_list, occupied=occupied)
