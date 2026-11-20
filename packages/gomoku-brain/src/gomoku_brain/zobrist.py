"""Zobrist hashing and transposition table for Gomoku alpha-beta search."""

import random
from enum import IntEnum
from typing import NamedTuple

from gomoku.board import Board
from gomoku.models import Move, Stone

# ---------------------------------------------------------------------------
# Zobrist random table
# ---------------------------------------------------------------------------
# Deterministic seed so hashes are stable within a process.
_RNG = random.Random(0x60_0C0E_2026)

_MAX_SIZE = 25  # Board.size upper bound

# _TABLE[row][col][stone_index] — stone_index: 0 = BLACK, 1 = WHITE
_TABLE: list[list[tuple[int, int]]] = []
for _r in range(_MAX_SIZE):
    _row: list[tuple[int, int]] = []
    for _c in range(_MAX_SIZE):
        _row.append((_RNG.getrandbits(64), _RNG.getrandbits(64)))
    _TABLE.append(_row)

# XOR key flipped each ply to distinguish "Black to move" from "White to move".
SIDE_KEY: int = _RNG.getrandbits(64)


def _stone_index(stone: Stone) -> int:
    """Map Stone.BLACK -> 0, Stone.WHITE -> 1."""
    return stone - 1  # BLACK=1 -> 0, WHITE=2 -> 1


def hash_board(board: Board, side_to_move: Stone) -> int:
    """Compute the full Zobrist hash of *board* with *side_to_move*."""
    h = 0
    for r, c in board.occupied:
        s = board.at(r, c)
        h ^= _TABLE[r][c][_stone_index(s)]
    if side_to_move is Stone.WHITE:
        h ^= SIDE_KEY
    return h


def incremental_hash(h: int, row: int, col: int, stone: Stone) -> int:
    """Return the updated hash after placing *stone* at (row, col).

    Also flips the side-to-move bit.
    """
    return h ^ _TABLE[row][col][_stone_index(stone)] ^ SIDE_KEY


# ---------------------------------------------------------------------------
# Transposition-table entry
# ---------------------------------------------------------------------------


class BoundType(IntEnum):
    EXACT = 0
    LOWER = 1  # score is a lower bound (failed high / beta cutoff)
    UPPER = 2  # score is an upper bound (failed low / all-node)


class TTEntry(NamedTuple):
    key: int  # full Zobrist hash for collision detection
    depth: int
    score: float
    flag: BoundType
    best_move: Move | None


# Default capacity of one TT generation (see :class:`TranspositionTable`).
# At roughly 250 B per entry the worst case (two full generations) stays
# around 250 MB, which a long ``time_limit_s`` search can no longer exceed.
DEFAULT_TT_MAX_ENTRIES = 500_000


class TranspositionTable:
    """Bounded dict-based transposition table with depth-preferred replacement.

    Memory is bounded by two generations of at most ``max_entries`` each.
    Stores go to the *current* generation; when it fills up it becomes the
    *old* generation (the previous old one is dropped) and a fresh current
    one starts.  Probes check the current generation first, then the old
    one and promote an old hit, so recently used positions survive a
    rollover while stale ones age out.
    """

    __slots__ = ("_current", "_old", "hits", "max_entries", "probes")

    def __init__(self, max_entries: int = DEFAULT_TT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._current: dict[int, TTEntry] = {}
        self._old: dict[int, TTEntry] = {}
        # Lightweight counters for diagnostics; they never affect search behaviour.
        self.hits: int = 0
        self.probes: int = 0

    def probe(self, key: int) -> TTEntry | None:
        self.probes += 1
        entry = self._current.get(key)
        if entry is None:
            entry = self._old.get(key)
            if entry is None:
                return None
            self._insert(key, entry)  # promote so it survives the next rollover
        if entry.key == key:
            self.hits += 1
            return entry
        return None

    def store(self, key: int, depth: int, score: float, flag: BoundType, best_move: Move | None) -> None:
        existing = self._current.get(key)
        if existing is None:
            existing = self._old.get(key)
        # Depth-preferred replacement: keep deeper entries.
        if existing is not None and existing.key == key and existing.depth > depth:
            return
        self._insert(key, TTEntry(key, depth, score, flag, best_move))

    def _insert(self, key: int, entry: TTEntry) -> None:
        if len(self._current) >= self.max_entries and key not in self._current:
            self._old = self._current
            self._current = {}
        self._current[key] = entry

    def __len__(self) -> int:
        return len(self._current) + sum(1 for k in self._old if k not in self._current)
