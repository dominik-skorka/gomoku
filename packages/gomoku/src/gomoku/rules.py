"""Win detection and game status checks."""

from __future__ import annotations

from gomoku.board import Board
from gomoku.interfaces import BoardView
from gomoku.models import GameResult, Move, Stone

# The four directions to check: horizontal, vertical, diagonal-down, diagonal-up.
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def check_winner(board: BoardView, last_move: Move, win_length: int = 5) -> GameResult | None:
    """Check if the last move created a winning line of ``win_length`` or more (freestyle)."""
    stone = board.at(last_move.row, last_move.col)
    if stone is Stone.EMPTY:
        return None

    for dr, dc in _DIRECTIONS:
        line = [last_move]

        # Scan in the positive direction
        for sign in (1, -1):
            r, c = last_move.row + dr * sign, last_move.col + dc * sign
            while 0 <= r < board.size and 0 <= c < board.size and board.at(r, c) is stone:
                line.append(Move(row=r, col=c))
                r += dr * sign
                c += dc * sign

        if len(line) >= win_length:
            return GameResult(winner=stone, winning_moves=sorted(line, key=lambda m: (m.row, m.col)))

    return None


def get_game_status(board: Board, last_move: Move, win_length: int = 5) -> GameResult | None:
    """Return a GameResult if the game is over (win or draw), else None."""
    result = check_winner(board, last_move, win_length)
    if result is not None:
        return result
    if board.is_full():
        return GameResult()
    return None
