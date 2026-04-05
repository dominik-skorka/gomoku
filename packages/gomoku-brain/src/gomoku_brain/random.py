"""Random brain — picks a uniformly random valid move."""

from __future__ import annotations

import random as _random

from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.candidates import get_candidate_moves


class RandomBrain:
    """MoveProvider that chooses a random valid move near existing stones."""

    @property
    def name(self) -> str:
        return "random"

    def get_move(self, board: Board, stone: Stone) -> Move:
        candidates = get_candidate_moves(board)
        return _random.choice(candidates)
