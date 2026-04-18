"""Zobrist hashing and transposition table for Gomoku alpha-beta search."""

from __future__ import annotations

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


class TranspositionTable:
    """Simple dict-based transposition table with depth-preferred replacement."""

    __slots__ = ("_table", "hits", "probes")

    def __init__(self) -> None:
        self._table: dict[int, TTEntry] = {}
        # Lightweight counters for diagnostics; they never affect search behaviour.
        self.hits: int = 0
        self.probes: int = 0

    def probe(self, key: int) -> TTEntry | None:
        self.probes += 1
        entry = self._table.get(key)
        if entry is not None and entry.key == key:
            self.hits += 1
            return entry
        return None

    def store(self, key: int, depth: int, score: float, flag: BoundType, best_move: Move | None) -> None:
        existing = self._table.get(key)
        # Depth-preferred replacement: keep deeper entries.
        if existing is not None and existing.key == key and existing.depth > depth:
            return
        self._table[key] = TTEntry(key, depth, score, flag, best_move)

    def __len__(self) -> int:
        return len(self._table)
