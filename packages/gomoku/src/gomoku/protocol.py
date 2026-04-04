"""MoveProvider protocol — the plugin interface for brains and keyboard input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gomoku.board import Board
    from gomoku.models import Move, Stone


@runtime_checkable
class MoveProvider(Protocol):
    """Any object that can provide a move given the current board state."""

    @property
    def name(self) -> str: ...

    def get_move(self, board: Board, stone: Stone) -> Move: ...
