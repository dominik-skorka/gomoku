"""Smart move candidate generation for Gomoku AI.

Instead of evaluating all empty cells (up to 225 on a 15x15 board),
restrict candidates to cells within a radius of existing stones.
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move

_RADIUS = 2


def get_candidate_moves(board: Board, radius: int = _RADIUS) -> list[Move]:
    """Return empty cells within ``radius`` of any placed stone.

    On an empty board, returns the center cell.
    Iterates only over ``board.occupied`` instead of scanning the full grid.
    """
    if not board.occupied:
        center = board.size // 2
        return [Move(row=center, col=center)]

    size = board.size
    candidates: set[tuple[int, int]] = set()
    for r, c in board.occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in board.occupied:
                    candidates.add((nr, nc))

    return [Move(row=r, col=c) for r, c in sorted(candidates)]
