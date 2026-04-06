"""Heuristic board evaluation for Gomoku AI."""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone

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

# Minimum threat level to count as a fork contributor: live-2 (count=2, open_ends=2).
_FORK_THRESHOLD = 100

# Bonus per fork cell: each additional converging threat line beyond the first
# is a multiplicative danger signal.
_FORK_BONUS = 3_000

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


def _threat_lines_at(board: Board, row: int, col: int, stone: Stone) -> int:
    """Count how many distinct threat lines for *stone* pass through (row, col).

    A "threat line" is a direction where placing *stone* at (row, col) would
    create a pattern scoring >= ``_FORK_THRESHOLD`` (i.e. at least a live-2).
    """
    count = 0
    size = board.size
    for dr, dc in _DIRECTIONS:
        consecutive = 0
        open_ends = 0
        # positive direction
        r, c = row + dr, col + dc
        while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
            consecutive += 1
            r += dr
            c += dc
        if 0 <= r < size and 0 <= c < size and board.at(r, c) is Stone.EMPTY:
            open_ends += 1
        # negative direction
        r, c = row - dr, col - dc
        while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
            consecutive += 1
            r -= dr
            c -= dc
        if 0 <= r < size and 0 <= c < size and board.at(r, c) is Stone.EMPTY:
            open_ends += 1
        # The pattern after placing stone here: (consecutive + 1, open_ends)
        # but for fork detection we check the *existing* neighbours: a cell
        # touching >=1 stone with open ends is a threat contributor.
        score = _PATTERN_SCORES.get((consecutive, open_ends), 0)
        if score >= _FORK_THRESHOLD:
            count += 1
    return count


def _fork_score(board: Board, stone: Stone) -> float:
    """Score fork potential: cells where multiple threat lines converge.

    For each empty candidate cell near occupied stones, count how many
    distinct directions form a significant threat for *stone*. If >= 2
    directions converge, this cell is a fork point — placing *stone* there
    would create a multi-threat (e.g. double-three, three-four).

    Returns the total fork bonus (always >= 0).
    """
    bonus = 0.0
    size = board.size
    seen: set[tuple[int, int]] = set()

    for occ_r, occ_c in board.occupied:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = occ_r + dr, occ_c + dc
                if (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                if board.at(nr, nc) is not Stone.EMPTY:
                    continue
                lines = _threat_lines_at(board, nr, nc, stone)
                if lines >= 2:
                    # Each additional converging line beyond the first adds bonus
                    bonus += _FORK_BONUS * (lines - 1)
    return bonus


def find_fork_cells(board: Board, stone: Stone) -> list[Move]:
    """Return empty cells where >= 2 distinct threat lines for *stone* converge.

    Used by threat brain to detect imminent fork points that need blocking.
    """
    forks: list[Move] = []
    size = board.size
    seen: set[tuple[int, int]] = set()

    for occ_r, occ_c in board.occupied:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = occ_r + dr, occ_c + dc
                if (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                if board.at(nr, nc) is not Stone.EMPTY:
                    continue
                lines = _threat_lines_at(board, nr, nc, stone)
                if lines >= 2:
                    forks.append(Move(row=nr, col=nc))
    return forks


def evaluate_board(board: Board, stone: Stone) -> float:
    """Evaluate the board from ``stone``'s perspective. Higher is better.

    Iterates only over ``board.occupied`` instead of scanning the full grid.
    Includes fork potential bonus/penalty.
    """
    score = 0.0
    opponent = stone.opponent()

    for r, c in board.occupied:
        cell = board.at(r, c)
        for dr, dc in _DIRECTIONS:
            if cell is stone:
                score += _score_direction(board, r, c, dr, dc, stone)
            else:
                score -= _score_direction(board, r, c, dr, dc, opponent)

    # Fork potential: bonus for our forks, penalty for opponent forks
    score += _fork_score(board, stone)
    score -= _fork_score(board, opponent)

    return score
