"""Board model with game operations."""

from __future__ import annotations

import copy

from pydantic import BaseModel, Field, model_validator

from gomoku.models import Move, Stone


class Board(BaseModel):
    """Immutable-style gomoku board. Use place_stone() to get a new board with a move applied."""

    size: int = Field(default=15, ge=5, le=25)
    grid: list[list[Stone]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _init_grid(self) -> Board:
        if not self.grid:
            self.grid = [[Stone.EMPTY] * self.size for _ in range(self.size)]
        return self

    def at(self, row: int, col: int) -> Stone:
        return self.grid[row][col]

    def is_valid_move(self, move: Move) -> bool:
        if not (0 <= move.row < self.size and 0 <= move.col < self.size):
            return False
        return self.grid[move.row][move.col] is Stone.EMPTY

    def get_valid_moves(self) -> list[Move]:
        return [
            Move(row=r, col=c) for r in range(self.size) for c in range(self.size) if self.grid[r][c] is Stone.EMPTY
        ]

    def is_full(self) -> bool:
        return all(self.grid[r][c] is not Stone.EMPTY for r in range(self.size) for c in range(self.size))

    def place_stone(self, move: Move, stone: Stone) -> Board:
        """Return a new Board with the stone placed. Does not mutate self."""
        new_grid = copy.deepcopy(self.grid)
        new_grid[move.row][move.col] = stone
        return Board(size=self.size, grid=new_grid)

    def stone_count(self) -> int:
        return sum(1 for r in range(self.size) for c in range(self.size) if self.grid[r][c] is not Stone.EMPTY)
