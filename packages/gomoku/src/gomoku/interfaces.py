"""The interfaces the packages depend on, as structural types.

A class implements one of these by having the members, not by inheriting
from it; the type checker (pyright, in CI) verifies that it does.

* :class:`BoardView`: a position to read.  :class:`~gomoku.board.Board`
  (immutable, for games) and ``gomoku_brain.search_state.SearchState``
  (mutable, for searches) both are one, so evaluation and threat detection
  take a ``BoardView`` and work on either.
* :class:`SearchBoard`: a position a search changes in place and takes
  back (``SearchState``).
* :class:`Player`: anything that chooses moves for a game, a brain or a
  human at the keyboard.  :class:`~gomoku.game.Game` and the API's runner
  depend on nothing else.  New brains subclass
  ``gomoku_brain.base.Brain``, which implements it.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gomoku.board import Board
    from gomoku.models import Move, Stone

type Cell = tuple[int, int]
"""A board cell as ``(row, col)``, both from 0."""


class BoardView(Protocol):
    """A position to read: its size, its stones, what is on a cell."""

    @property
    def size(self) -> int:
        """Side length; cells are ``0 <= row, col < size``."""
        ...

    @property
    def occupied(self) -> Collection[Cell]:
        """The cells with a stone on them."""
        ...

    def at(self, row: int, col: int) -> Stone:
        """The stone on a cell, ``Stone.EMPTY`` if none."""
        ...


class SearchBoard(BoardView, Protocol):
    """A position a search changes in place and takes back, last move first."""

    def make(self, row: int, col: int, stone: Stone) -> None:
        """Put *stone* on an empty cell."""
        ...

    def unmake(self) -> None:
        """Take back the most recent :meth:`make`."""
        ...


@runtime_checkable
class Player(Protocol):
    """Anything that chooses moves for one side of a game."""

    @property
    def name(self) -> str:
        """Short name, as shown to people and written to recordings."""
        ...

    def get_move(self, board: Board, stone: Stone) -> Move:
        """A legal move for *stone* on *board* (not full)."""
        ...
