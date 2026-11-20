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

# Precomputed per-column bit masks.  ``make``/``unmake`` run millions of
# times per search; building ``np.uint32`` scalars and shifting on every
# call dominated their cost.
_BIT: list[np.uint32] = [np.uint32(1 << c) for c in range(32)]
_BIT_INV: list[np.uint32] = [np.uint32(~(1 << c) & 0xFFFFFFFF) for c in range(32)]


class SearchState:
    """Mutable bitboard with a preallocated undo stack.

    All mutation goes through :meth:`make` / :meth:`unmake`.  Reads
    (``at``, ``black``, ``white``, ``zobrist``) are zero-cost attribute
    lookups.

    Use :meth:`from_board` to build one from a Pydantic ``Board``.
    """

    __slots__ = (
        "_ctx_candidates",
        "_ctx_moves",
        "_ctx_zobrist",
        "_eval_stack",
        "_eval_valid",
        "_occ",
        "_occ_arr",
        "_undo_col",
        "_undo_row",
        "_undo_stone",
        "black",
        "grid",
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
        grid: NDArray[np.int8] | None = None,
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
        # Plain lists: only Python reads these, and storing one value in a
        # NumPy array costs several times a list store on this hot path.
        self._undo_row: list[int] = [0] * cap
        self._undo_col: list[int] = [0] * cap
        self._undo_stone: list[int] = [0] * cap
        # Eager int8 grid mirror — maintained O(1) per make/unmake so the
        # grid-JIT path (bitboard mode off) doesn't pay an O(size*size)
        # bitboard→grid conversion per node.
        if grid is None:
            from gomoku_brain.bitboard_jit import bitboard_to_grid

            grid = bitboard_to_grid(black, white, size)
        self.grid = grid
        # Incremental occupied-coordinate stack: the first ``occ_count``
        # rows are the occupied cells.  Seeded row-major; ``make`` appends,
        # ``unmake`` pops (LIFO matches the undo stack).  Consumers via
        # ``grid_utils.occupied_coords`` get a zero-cost slice instead of
        # an O(size^2) bitboard scan; all kernels treat the order as
        # irrelevant (sums and seen-grid dedups).
        self._occ_arr = np.empty((cap, 2), dtype=np.int32)
        rows, cols = np.nonzero(grid)
        self._occ_arr[: rows.shape[0], 0] = rows
        self._occ_arr[: rows.shape[0], 1] = cols
        # Per-ply incremental evaluation stack (black-positive score).
        # ``_eval_valid[k]`` says whether ``_eval_stack[k]`` holds the
        # evaluation of the position after k plies; ``make`` invalidates
        # the new ply only, so cached ancestor evaluations survive and
        # ``gomoku_brain.evaluate`` can fill gaps with local deltas.
        self._eval_stack = np.zeros(cap, dtype=np.float64)
        self._eval_valid: list[bool] = [False] * cap
        # Per-ply memo of the radius-2 candidate scan, keyed by the
        # zobrist hash of the position each slot was derived from (no
        # explicit invalidation: a make at the same ply changes the
        # hash, so a stale slot simply misses; ancestor slots survive
        # the make/unmake churn of child subtrees).  ``_ctx_candidates``
        # holds the int32 arrays consumed by JIT kernels;
        # ``_ctx_moves`` the corresponding Move lists (returned as
        # copies — callers slice and re-order their lists).
        self._ctx_zobrist: list[int] = [0] * cap
        self._ctx_candidates: list[NDArray[np.int32] | None] = [None] * cap
        self._ctx_moves: list[list | None] = [None] * cap  # list[Move] slots
        # Lazy set cache — off the hot path, invalidated on make/unmake.
        self._occ: set[tuple[int, int]] | None = None

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
        # Accumulate row-bitmasks as plain Python ints, then convert each row
        # to ``np.uint32`` once.  Per-cell ``np.uint32`` shifts box/unbox the
        # numpy scalar and dominate construction cost on the search hot path
        # (every ``classify_vcf_breakers`` and ``pns_threat_search`` call).
        black_ints = [0] * size
        white_ints = [0] * size
        for r, c in board.occupied:
            stone = board.at(r, c)
            if stone is Stone.BLACK:
                black_ints[r] |= 1 << c
            elif stone is Stone.WHITE:
                white_ints[r] |= 1 << c
        black = np.array(black_ints, dtype=np.uint32)
        white = np.array(white_ints, dtype=np.uint32)
        zobrist = hash_board(board, side_to_move)
        occ_count = len(board.occupied)
        return cls(size=size, black=black, white=white, zobrist=zobrist, occ_count=occ_count)

    def fork(self) -> SearchState:
        """A new state holding the current position, with its own arrays.

        For searching the same position on another thread: the copy shares
        nothing with this state, so each can make and unmake freely.  Its
        undo history starts empty (``ply`` 0).
        """
        return SearchState(
            size=self.size,
            black=self.black.copy(),
            white=self.white.copy(),
            zobrist=self.zobrist,
            occ_count=self.occ_count,
            grid=self.grid.copy(),
        )

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
        bit = _BIT[col]
        if stone_int == 1:
            self.black[row] |= bit
        else:
            self.white[row] |= bit
        # Mirror into the int8 grid.
        self.grid[row, col] = stone_int
        # Push undo.
        p = self.ply
        self._undo_row[p] = row
        self._undo_col[p] = col
        self._undo_stone[p] = stone_int
        self.ply = p + 1
        # Push onto the occupied stack and invalidate the new ply's eval.
        oc = self.occ_count
        self._occ_arr[oc, 0] = row
        self._occ_arr[oc, 1] = col
        self.occ_count = oc + 1
        self._eval_valid[p + 1] = False
        self._occ = None

    def unmake(self) -> None:
        """Undo the most recent :meth:`make`.

        Precondition: ``self.ply > 0``.
        """
        p = self.ply - 1
        row = self._undo_row[p]
        col = self._undo_col[p]
        stone_int = self._undo_stone[p]
        # Clear the bit.
        bit_mask = _BIT_INV[col]
        if stone_int == 1:
            self.black[row] &= bit_mask
        else:
            self.white[row] &= bit_mask
        # Clear the grid cell too.
        self.grid[row, col] = 0
        # Reverse the zobrist update — XOR is its own inverse.
        self.zobrist ^= _TABLE[row][col][stone_int - 1]
        self.zobrist ^= SIDE_KEY
        self.ply = p
        self.occ_count -= 1
        self._occ = None

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def at(self, row: int, col: int) -> Stone:
        # The int8 grid mirror is kept in lockstep by make/unmake — one
        # indexed read beats two bitboard mask tests.
        return Stone(int(self.grid[row, col]))

    def is_empty(self, row: int, col: int) -> bool:
        return int(self.grid[row, col]) == 0

    @property
    def occupied(self) -> set[tuple[int, int]]:
        """Lazy set of occupied ``(row, col)`` coords — duck-type for Board/_GridBoard.

        Not used on the hot path (which consumes ``.black``/``.white`` directly),
        but kept so helpers that accept any board-like object keep working.
        """
        if self._occ is None:
            from gomoku_brain.bitboard_jit import occupied_from_bb

            arr = occupied_from_bb(self.black, self.white, self.size)
            self._occ = {(int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])}
        return self._occ

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
