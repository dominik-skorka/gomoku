"""Heuristic board evaluation for Gomoku AI."""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Stone

# Pattern scores: (count, open_ends) -> score
# open_ends: 2 = both ends open, 1 = one end open, 0 = blocked both
_PATTERN_SCORES: dict[tuple[int, int], int] = {
    (5, 2): 100_000,
    (5, 1): 100_000,
    (5, 0): 100_000,
    (4, 2): 10_000,  # live-4
    (4, 1): 1_000,  # dead-4
    (3, 2): 1_000,  # live-3
    (3, 1): 100,  # dead-3
    (2, 2): 100,  # live-2
    (2, 1): 10,  # dead-2
    (1, 2): 10,
    (1, 1): 1,
}

_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _score_direction(board: Board, row: int, col: int, dr: int, dc: int, stone: Stone) -> int:
    """Score a single direction from (row, col) for the given stone."""
    size = board.size
    count = 0
    open_ends = 0

    # Count consecutive stones in positive direction
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
        count += 1
        r += dr
        c += dc
    # Check if the end is open
    if 0 <= r < size and 0 <= c < size and board.at(r, c) is Stone.EMPTY:
        open_ends += 1

    # Count consecutive stones in negative direction
    r, c = row - dr, col - dc
    while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
        count += 1
        r -= dr
        c -= dc
    if 0 <= r < size and 0 <= c < size and board.at(r, c) is Stone.EMPTY:
        open_ends += 1

    return _PATTERN_SCORES.get((count, open_ends), 0)


def evaluate_board(board: Board, stone: Stone) -> float:
    """Evaluate the board from ``stone``'s perspective. Higher is better."""
    score = 0.0
    opponent = stone.opponent()

    for r in range(board.size):
        for c in range(board.size):
            cell = board.at(r, c)
            if cell is Stone.EMPTY:
                continue
            for dr, dc in _DIRECTIONS:
                if cell is stone:
                    score += _score_direction(board, r, c, dr, dc, stone)
                else:
                    score -= _score_direction(board, r, c, dr, dc, opponent)

    return score
