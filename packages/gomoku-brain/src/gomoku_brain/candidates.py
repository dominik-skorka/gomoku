"""Smart move candidate generation for Gomoku AI.

Instead of evaluating all empty cells (up to 225 on a 15x15 board),
restrict candidates to cells within a radius of existing stones.
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone

_RADIUS = 2


def get_candidate_moves(board: Board, radius: int = _RADIUS) -> list[Move]:
    """Return empty cells within ``radius`` of any placed stone.

    On an empty board, returns the center cell.
    """
    if board.stone_count() == 0:
        center = board.size // 2
        return [Move(row=center, col=center)]

    candidates: set[tuple[int, int]] = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.at(r, c) is not Stone.EMPTY:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < board.size and 0 <= nc < board.size and board.at(nr, nc) is Stone.EMPTY:
                            candidates.add((nr, nc))

    return [Move(row=r, col=c) for r, c in sorted(candidates)]
