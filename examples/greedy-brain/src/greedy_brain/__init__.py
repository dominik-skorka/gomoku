"""A minimal Gomoku brain: win if it can, block if it must, else play greedily.

The tutorial in ``docs/WRITING_A_BRAIN.md`` walks through this file.
"""

from typing import override

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner
from gomoku_brain import BaseBrainSettings, Brain
from gomoku_brain.brain_base import BrainSession
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board
from pydantic import Field


class GreedySettings(BaseBrainSettings):
    """Settings of the greedy brain (``gomoku play --black greedy,radius=1``)."""

    radius: int = Field(default=2, ge=1, le=4, description="Look at empty cells this close to a stone.")


class GreedyBrain(Brain[GreedySettings], name="greedy"):
    """One-ply brain: an immediate win, else a block, else the best evaluation."""

    @override
    def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
        candidates = get_candidate_moves(board, radius=self.settings.radius)
        opponent = stone.opponent()

        # 1. Five in a row now?
        for move in candidates:
            if check_winner(board.place_stone(move, stone), move):
                return move

        # 2. Would the opponent make five there next move?  Take the cell first.
        for move in candidates:
            if check_winner(board.place_stone(move, opponent), move):
                return move

        # 3. Otherwise the move after which the position looks best for us.
        return max(candidates, key=lambda move: evaluate_board(board.place_stone(move, stone), stone))
