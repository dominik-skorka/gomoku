"""Random brain — picks a uniformly random valid move."""

from __future__ import annotations

import random as _random

from gomoku.board import Board
from gomoku.models import Move, Stone


class RandomBrain:
    """MoveProvider that chooses a random valid move."""

    @property
    def name(self) -> str:
        return "random"

    def get_move(self, board: Board, stone: Stone) -> Move:
        valid = board.get_valid_moves()
        return _random.choice(valid)
