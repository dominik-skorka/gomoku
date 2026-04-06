"""Board model with game operations."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from gomoku.models import Move, Stone


class Board(BaseModel):
    """Immutable-style gomoku board. Use place_stone() to get a new board with a move applied."""

    size: int = Field(default=15, ge=5, le=25)
    grid: list[list[Stone]] = Field(default_factory=list)
    occupied: set[tuple[int, int]] = Field(default_factory=set)

    @model_validator(mode="after")
    def _init_grid(self) -> Board:
        if not self.grid:
            self.grid = [[Stone.EMPTY] * self.size for _ in range(self.size)]
        if not self.occupied:
            self.occupied = {
                (r, c) for r in range(self.size) for c in range(self.size) if self.grid[r][c] is not Stone.EMPTY
            }
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
        return len(self.occupied) == self.size * self.size

    def place_stone(self, move: Move, stone: Stone) -> Board:
        """Return a new Board with the stone placed. Does not mutate self."""
        new_grid = [row[:] for row in self.grid]
        new_grid[move.row][move.col] = stone
        new_occupied = self.occupied | {(move.row, move.col)}
        return Board.model_construct(size=self.size, grid=new_grid, occupied=new_occupied)

    def stone_count(self) -> int:
        return len(self.occupied)
