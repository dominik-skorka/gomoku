"""Mutable bitboard with make/unmake — internal search representation.

This class is designed to be the single representation used throughout the
alpha-beta search.  Unlike ``Board`` / ``_GridBoard`` / ``BitboardState`` —
all of which return a *new* object from ``place_stone`` — ``SearchState``
mutates in place and supports O(1) ``unmake`` via a preallocated undo stack.

Key properties:

* **O(1) make/unmake** — no allocations on the search hot path.
* **Incremental zobrist** — XOR deltas applied on each make, undone on unmake.
* **Occupied count tracked** — no per-node ``len(occupied)`` recomputation.
* **Row-based uint32 bitboards** — same layout as
  :class:`~gomoku_brain.bitboard.BitboardState`, so existing numba kernels
  in ``bitboard_jit.py`` can be called directly on ``state.black``,
  ``state.white``.

Phase 1 scope: standalone container + tests.  Phase 2 threads it through
``AlphaBetaBrain._alphabeta``; Phase 3 adds incremental line-score caches
alongside the bitboards.
"""

from __future__ import annotations

import numpy as np
from gomoku.board import Board
from gomoku.models import Stone
from numpy.typing import NDArray

from gomoku_brain.zobrist import _TABLE, SIDE_KEY, hash_board


class SearchState:
    """Mutable bitboard with a preallocated undo stack.

    All mutation goes through :meth:`make` / :meth:`unmake`.  Reads
    (``at``, ``black``, ``white``, ``zobrist``) are zero-cost attribute
    lookups.

    Use :meth:`from_board` to build one from a Pydantic ``Board``.
    """

    __slots__ = (
        "_undo_col",
        "_undo_row",
        "_undo_stone",
        "black",
        "occ_count",
        "ply",
        "size",
        "white",
        "zobrist",
    )

    def __init__(
        self,
        size: int,
        black: NDArray[np.uint32],
        white: NDArray[np.uint32],
        zobrist: int,
        occ_count: int,
        max_ply: int | None = None,
    ) -> None:
        self.size = size
        self.black = black
        self.white = white
        self.zobrist = zobrist
        self.occ_count = occ_count
        self.ply = 0

        # Preallocated undo stack.  The search tree depth can in principle reach
        # size*size (fill the board), but typical alpha-beta depth is far less.
        # We size to size*size + small headroom so it is never the bottleneck.
        cap = max_ply if max_ply is not None else (size * size + 16)
        self._undo_row = np.empty(cap, dtype=np.int32)
        self._undo_col = np.empty(cap, dtype=np.int32)
        self._undo_stone = np.empty(cap, dtype=np.int8)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_board(cls, board: Board, side_to_move: Stone) -> SearchState:
        """Build a ``SearchState`` from a Pydantic ``Board``.

        The initial zobrist hash encodes ``side_to_move`` so that the first
        :meth:`make` call flips it in the usual way.
        """
        size = board.size
        black = np.zeros(size, dtype=np.uint32)
        white = np.zeros(size, dtype=np.uint32)
        for r, c in board.occupied:
            bit = np.uint32(1) << np.uint32(c)
            stone = board.at(r, c)
            if stone is Stone.BLACK:
                black[r] |= bit
            elif stone is Stone.WHITE:
                white[r] |= bit
        zobrist = hash_board(board, side_to_move)
        occ_count = len(board.occupied)
        return cls(size=size, black=black, white=white, zobrist=zobrist, occ_count=occ_count)

    @classmethod
    def from_bitboards(
        cls,
        black: NDArray[np.uint32],
        white: NDArray[np.uint32],
        size: int,
        side_to_move: Stone,
    ) -> SearchState:
        """Build from raw bitboard arrays.  ``black`` and ``white`` are copied."""
        black_copy = black.copy()
        white_copy = white.copy()
        zobrist = 0
        occ_count = 0
        for r in range(size):
            bits_b = int(black_copy[r])
            bits_w = int(white_copy[r])
            for c in range(size):
                m = 1 << c
                if bits_b & m:
                    zobrist ^= _TABLE[r][c][0]
                    occ_count += 1
                elif bits_w & m:
                    zobrist ^= _TABLE[r][c][1]
                    occ_count += 1
        if side_to_move is Stone.WHITE:
            zobrist ^= SIDE_KEY
        return cls(size=size, black=black_copy, white=white_copy, zobrist=zobrist, occ_count=occ_count)

    # ------------------------------------------------------------------
    # Mutation (hot path)
    # ------------------------------------------------------------------

    def make(self, row: int, col: int, stone: Stone) -> None:
        """Place ``stone`` at ``(row, col)`` and push an undo record.

        Precondition: the cell is empty.  No bounds check — callers in the
        search are expected to have validated the candidate already.
        """
        stone_int = int(stone)
        # Zobrist: XOR in the stone, flip the side-to-move bit.
        self.zobrist ^= _TABLE[row][col][stone_int - 1]
        self.zobrist ^= SIDE_KEY
        # Set the bit on the colour's bitboard.
        bit = np.uint32(1) << np.uint32(col)
        if stone_int == 1:
            self.black[row] |= bit
        else:
            self.white[row] |= bit
        # Push undo.
        p = self.ply
        self._undo_row[p] = row
        self._undo_col[p] = col
        self._undo_stone[p] = stone_int
        self.ply = p + 1
        self.occ_count += 1

    def unmake(self) -> None:
        """Undo the most recent :meth:`make`.

        Precondition: ``self.ply > 0``.
        """
        p = self.ply - 1
        row = int(self._undo_row[p])
        col = int(self._undo_col[p])
        stone_int = int(self._undo_stone[p])
        # Clear the bit.
        bit_mask = ~(np.uint32(1) << np.uint32(col))
        if stone_int == 1:
            self.black[row] &= bit_mask
        else:
            self.white[row] &= bit_mask
        # Reverse the zobrist update — XOR is its own inverse.
        self.zobrist ^= _TABLE[row][col][stone_int - 1]
        self.zobrist ^= SIDE_KEY
        self.ply = p
        self.occ_count -= 1

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def at(self, row: int, col: int) -> Stone:
        bit = np.uint32(1) << np.uint32(col)
        if self.black[row] & bit:
            return Stone.BLACK
        if self.white[row] & bit:
            return Stone.WHITE
        return Stone.EMPTY

    def is_empty(self, row: int, col: int) -> bool:
        bit = np.uint32(1) << np.uint32(col)
        return not (self.black[row] & bit) and not (self.white[row] & bit)

    def last_move(self) -> tuple[int, int, Stone] | None:
        """Return the most recent move on the undo stack, or ``None`` at root."""
        if self.ply == 0:
            return None
        p = self.ply - 1
        return (
            int(self._undo_row[p]),
            int(self._undo_col[p]),
            Stone(int(self._undo_stone[p])),
        )
