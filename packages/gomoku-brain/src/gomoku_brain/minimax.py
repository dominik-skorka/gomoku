"""Minimax brain — classic minimax search."""

from typing import override

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner
from gomoku.settings import MinimaxBrainSettings

from gomoku_brain.base import Brain
from gomoku_brain.brain_base import BrainSession
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board


class MinimaxBrain(Brain[MinimaxBrainSettings], name="minimax"):
    """Player using classic minimax search."""

    @override
    def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
        _, best_move = self._minimax(board, self.settings.depth, stone, stone)
        assert best_move is not None
        return best_move

    def _minimax(
        self,
        board: Board,
        depth: int,
        current: Stone,
        maximizing_stone: Stone,
    ) -> tuple[float, Move | None]:
        candidates = get_candidate_moves(board)
        if not candidates:
            return 0.0, None

        is_maximizing = current is maximizing_stone
        best_score = float("-inf") if is_maximizing else float("inf")
        best_move: Move | None = None

        for move in candidates:
            new_board = board.place_stone(move, current)

            result = check_winner(new_board, move)
            if result is not None:
                if result.winner is maximizing_stone:
                    return 100_000.0 + depth, move
                return -(100_000.0 + depth), move

            if depth <= 1:
                score = evaluate_board(new_board, maximizing_stone)
            else:
                score, _ = self._minimax(new_board, depth - 1, current.opponent(), maximizing_stone)

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move
