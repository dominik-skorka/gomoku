"""Core data models for the Gomoku game."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field, computed_field, model_validator


class Stone(IntEnum):
    """Represents the state of a board cell."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def opponent(self) -> Stone:
        if self is Stone.BLACK:
            return Stone.WHITE
        if self is Stone.WHITE:
            return Stone.BLACK
        msg = "EMPTY has no opponent"
        raise ValueError(msg)

    def __str__(self) -> str:
        return self.name.capitalize()


class Move(BaseModel, frozen=True):
    """A board position (row, col), both zero-indexed."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def position(self) -> str:
        """Human-readable notation, e.g. ``H8``."""
        col_label = chr(ord("A") + self.col)
        row_label = self.row + 1
        return f"{col_label}{row_label}"

    def __str__(self) -> str:
        return self.position


class GameConfig(BaseModel, frozen=True):
    """Configuration for a Gomoku game."""

    board_size: int = Field(default=15, ge=5, le=25)
    win_length: int = Field(default=5, ge=3)

    @model_validator(mode="after")
    def _win_length_fits_board(self) -> GameConfig:
        if self.win_length > self.board_size:
            msg = f"win_length ({self.win_length}) cannot exceed board_size ({self.board_size})"
            raise ValueError(msg)
        return self


class GameResult(BaseModel, frozen=True):
    """Outcome of a completed game."""

    winner: Stone | None = None
    winning_moves: list[Move] | None = None

    @property
    def is_draw(self) -> bool:
        return self.winner is None
