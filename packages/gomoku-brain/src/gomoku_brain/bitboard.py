"""Python-level bitboard wrapper for Gomoku.

Provides :class:`BitboardState` — a lightweight, immutable-style container
holding row-based ``uint32`` bitboard arrays.  Duck-type compatible with
:class:`~gomoku_brain.grid_utils._GridBoard` for the operations used by the
alpha-beta and threat search layers.

Use :func:`from_board` to convert a :class:`~gomoku.board.Board` or
:class:`~gomoku_brain.grid_utils._GridBoard` to a ``BitboardState``.
"""

from __future__ import annotations

import numpy as np
from gomoku.board import Board
from gomoku.models import Move, Stone
from numpy.typing import NDArray

from gomoku_brain.bitboard_jit import (
    _bit_test,
    bitboard_to_grid,
    grid_to_bitboard,
    occupied_from_bb,
)


class BitboardState:
    """Row-based uint32 bitboard representation of a Gomoku board.

    Two arrays of ``uint32[size]``: one for BLACK, one for WHITE.
    Bit ``c`` of ``rows[r]`` is 1 iff there is a stone at ``(r, c)``.

    Implements the same interface used by alpha-beta / threat search:
    ``at()``, ``place_stone()``, ``is_valid_move()``, ``is_full()``,
    ``occupied``, ``size``.
    """

    __slots__ = ("_occ", "black", "size", "white")

    def __init__(
        self,
        black: NDArray[np.uint32],
        white: NDArray[np.uint32],
        size: int,
    ) -> None:
        self.black = black
        self.white = white
        self.size = size
        self._occ: set[tuple[int, int]] | None = None

    # -- Construction --------------------------------------------------------

    @classmethod
    def empty(cls, size: int = 15) -> BitboardState:
        """Create an empty bitboard of the given size."""
        return cls(
            black=np.zeros(size, dtype=np.uint32),
            white=np.zeros(size, dtype=np.uint32),
            size=size,
        )

    @classmethod
    def from_grid(cls, grid: NDArray[np.int8], size: int) -> BitboardState:
        """Construct from a numpy int8 grid."""
        black, white = grid_to_bitboard(grid, size)
        return cls(black=black, white=white, size=size)

    @classmethod
    def from_board(cls, board: Board) -> BitboardState:
        """Construct from a Pydantic Board."""
        grid = np.array(board.grid, dtype=np.int8)
        return cls.from_grid(grid, board.size)

    # -- Board-compatible interface ------------------------------------------

    def at(self, row: int, col: int) -> Stone:
        if _bit_test(self.black, row, col):
            return Stone.BLACK
        if _bit_test(self.white, row, col):
            return Stone.WHITE
        return Stone.EMPTY

    def place_stone(self, move: Move, stone: Stone) -> BitboardState:
        """Return a new BitboardState with the stone placed."""
        if stone is Stone.BLACK:
            new_black = self.black.copy()
            new_black[move.row] |= np.uint32(1) << np.uint32(move.col)
            return BitboardState(black=new_black, white=self.white, size=self.size)
        new_white = self.white.copy()
        new_white[move.row] |= np.uint32(1) << np.uint32(move.col)
        return BitboardState(black=self.black, white=new_white, size=self.size)

    def is_valid_move(self, move: Move) -> bool:
        if not (0 <= move.row < self.size and 0 <= move.col < self.size):
            return False
        return not _bit_test(self.black, move.row, move.col) and not _bit_test(self.white, move.row, move.col)

    def is_full(self) -> bool:
        total = 0
        for r in range(self.size):
            v = self.black[r] | self.white[r]
            # Count set bits
            v = np.uint32(v)
            v = v - ((v >> np.uint32(1)) & np.uint32(0x55555555))
            v = (v & np.uint32(0x33333333)) + ((v >> np.uint32(2)) & np.uint32(0x33333333))
            v = (v + (v >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
            total += int((v * np.uint32(0x01010101)) >> np.uint32(24))
        return total == self.size * self.size

    @property
    def occupied(self) -> set[tuple[int, int]]:
        if self._occ is None:
            occ_arr = occupied_from_bb(self.black, self.white, self.size)
            self._occ = {(int(occ_arr[i, 0]), int(occ_arr[i, 1])) for i in range(occ_arr.shape[0])}
        return self._occ

    @property
    def grid(self) -> NDArray[np.int8]:
        """Return an int8[size, size] grid — for interop with numpy-based code."""
        return bitboard_to_grid(self.black, self.white, self.size)


# ---------------------------------------------------------------------------
# Convenience conversion
# ---------------------------------------------------------------------------

_use_bitboard: bool | None = None


def _should_use_bitboard() -> bool:
    """Return whether bitboard mode is enabled (cached after first call)."""
    global _use_bitboard
    if _use_bitboard is None:
        from gomoku.settings import GomokuSettings

        _use_bitboard = GomokuSettings().bitboard.enabled
    return _use_bitboard
