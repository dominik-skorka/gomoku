"""Negamax brain — negamax simplification of minimax."""

from __future__ import annotations

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board


class NegamaxBrain:
    """MoveProvider using negamax search."""

    def __init__(self, depth: int = 3) -> None:
        self.depth = depth

    @property
    def name(self) -> str:
        return "negamax"

    def get_move(self, board: Board, stone: Stone) -> Move:
        with logfire.span("brain.negamax", depth=self.depth, stone=str(stone)):
            _, best_move = self._negamax(board, self.depth, stone)
            assert best_move is not None
            return best_move

    def _negamax(
        self,
        board: Board,
        depth: int,
        stone: Stone,
    ) -> tuple[float, Move | None]:
        candidates = get_candidate_moves(board)
        if not candidates:
            return 0.0, None

        best_score = float("-inf")
        best_move: Move | None = None

        for move in candidates:
            new_board = board.place_stone(move, stone)

            result = check_winner(new_board, move)
            if result is not None:
                if result.winner is stone:
                    return 100_000.0 + depth, move
                return -(100_000.0 + depth), move

            if depth <= 1:
                score = evaluate_board(new_board, stone)
            else:
                opp_score, _ = self._negamax(new_board, depth - 1, stone.opponent())
                score = -opp_score

            if score > best_score:
                best_score = score
                best_move = move

        return best_score, best_move
