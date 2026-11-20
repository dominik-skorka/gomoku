"""Random brain — picks a uniformly random valid move."""

from __future__ import annotations

import random as _random
from typing import override

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.settings import RandomBrainSettings

from gomoku_brain.base import Brain
from gomoku_brain.brain_base import BrainSession
from gomoku_brain.candidates import get_candidate_moves


class RandomBrain(Brain[RandomBrainSettings], name="random"):
    """Player that chooses a random valid move near existing stones."""

    @override
    def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
        candidates = get_candidate_moves(board)
        return _random.choice(candidates)
