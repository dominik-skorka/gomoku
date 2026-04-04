"""Alpha-beta brain — negamax with alpha-beta pruning and move ordering."""

from __future__ import annotations

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board


class AlphaBetaBrain:
    """MoveProvider using alpha-beta pruning on top of negamax."""

    def __init__(self, depth: int = 4) -> None:
        self.depth = depth

    @property
    def name(self) -> str:
        return "alphabeta"

    def get_move(self, board: Board, stone: Stone) -> Move:
        with logfire.span("brain.alphabeta", depth=self.depth, stone=str(stone)):
            _, best_move = self._alphabeta(board, self.depth, float("-inf"), float("inf"), stone)
            assert best_move is not None
            return best_move

    def _order_moves(self, board: Board, candidates: list[Move], stone: Stone) -> list[Move]:
        """Sort candidates by shallow evaluation for better pruning."""
        scored = []
        for move in candidates:
            new_board = board.place_stone(move, stone)
            score = evaluate_board(new_board, stone)
            scored.append((score, move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _alphabeta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        stone: Stone,
    ) -> tuple[float, Move | None]:
        candidates = get_candidate_moves(board)
        if not candidates:
            return 0.0, None

        # Order moves for better pruning at higher depths
        if depth >= 2:
            candidates = self._order_moves(board, candidates, stone)

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
                opp_score, _ = self._alphabeta(new_board, depth - 1, -beta, -alpha, stone.opponent())
                score = -opp_score

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score, best_move
