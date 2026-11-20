"""File-based opening book — load, save, and lookup.

The book is stored as a JSON file with canonical positions (8-fold symmetry)
mapping to evaluated moves.  At runtime the current board is canonicalised
and looked up in the loaded book to retrieve a pre-computed best move.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gomoku.models import Move, Stone
from pydantic import BaseModel, Field

from gomoku_brain.opening import _INVERSE, _canonicalize, _transform

if TYPE_CHECKING:
    from gomoku.interfaces import BoardView


class BookEntry(BaseModel):
    """A single opening-book entry for a canonical position."""

    # The best move in the canonical frame.
    move: tuple[int, int]
    # Minimax-backed evaluation score.
    score: float
    # Search depth used for the evaluation.
    depth: int = 0


class OpeningBookFile(BaseModel):
    """Serialisable opening book."""

    format_version: int = 1
    board_size: int = 15
    # brain + settings used to generate the book.
    generator_brain: str = ""
    generator_settings: dict[str, Any] = Field(default_factory=dict)
    # DOE parameters used during generation.
    search_depth: int = 0
    max_positions: int = 0
    error_threshold: float = 0.0
    # Positions: canonical key (JSON string of sorted stone-tuples) → BookEntry.
    positions: dict[str, BookEntry] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Write the book to *path* as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OpeningBookFile:
        """Load a book from *path*."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)


def _stones_from_board(board: BoardView) -> set[tuple[int, int, int]]:
    """Extract ``{(stone_int, row, col), …}`` from any board type."""
    stones: set[tuple[int, int, int]] = set()
    for r, c in board.occupied:
        stones.add((int(board.at(r, c)), r, c))
    return stones


def _canonical_key(stones: set[tuple[int, int, int]], size: int) -> str:
    """Return the JSON-string key for the canonical form of *stones*."""
    canon, _t = _canonicalize(stones, size)
    return json.dumps(canon)


def lookup_file_book(
    book: OpeningBookFile,
    board: BoardView,
    stone: Stone,  # kept for API consistency
) -> Move | None:
    """Look up the current position in *book* and return the book move, or ``None``."""
    if board.size != book.board_size:
        return None

    stones = _stones_from_board(board)
    if not stones and not book.positions:
        return None

    canon, t = _canonicalize(stones, board.size)
    key = json.dumps(canon)
    entry = book.positions.get(key)
    if entry is None:
        return None

    # Map the canonical move back to the original orientation.
    inv_t = _INVERSE[t]
    actual = _transform(entry.move[0], entry.move[1], board.size, inv_t)
    move = Move(row=actual[0], col=actual[1])

    # Sanity: the cell must be empty.
    if (move.row, move.col) in board.occupied:
        return None

    return move
