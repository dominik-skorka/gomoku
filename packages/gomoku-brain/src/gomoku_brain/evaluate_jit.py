"""Numba-JIT compiled board evaluation for Gomoku AI.

All functions in this module operate on raw numpy arrays (not Pydantic models)
so that Numba can compile them to machine code. Use :func:`evaluate_grid` as
the main entry point.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Pattern-score lookup table — replaces the dict ``_PATTERN_SCORES``.
# Axes: [count 0..5][open_ends 0..2] → score.
# ---------------------------------------------------------------------------
_PATTERN_TABLE = np.zeros((6, 3), dtype=np.int32)
_PATTERN_TABLE[5, 2] = 100_000
_PATTERN_TABLE[5, 1] = 100_000
_PATTERN_TABLE[5, 0] = 100_000
_PATTERN_TABLE[4, 2] = 10_000
_PATTERN_TABLE[4, 1] = 1_000
_PATTERN_TABLE[3, 2] = 1_000
_PATTERN_TABLE[3, 1] = 100
_PATTERN_TABLE[2, 2] = 100
_PATTERN_TABLE[2, 1] = 10
_PATTERN_TABLE[1, 2] = 10
_PATTERN_TABLE[1, 1] = 1

_FORK_THRESHOLD: int = 100
_FORK_BONUS: int = 3_000
_OPEN_THREE_BONUS: int = 1_500

# Four scanning directions: (dr, dc).
_DIRS = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)


# ---------------------------------------------------------------------------
# Core JIT helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def _pattern_score(count: int, open_ends: int, table: NDArray) -> int:  # type: ignore[type-arg]
    """Look up the pattern score, clamping indices to the table bounds."""
    if count < 0 or count > 5 or open_ends < 0 or open_ends > 2:
        return 0
    return table[count, open_ends]


@njit(cache=True)
def _score_direction(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    dr: int,
    dc: int,
    stone: int,
    table: NDArray,  # type: ignore[type-arg]
) -> int:
    """Score a single direction from (row, col) for *stone*."""
    count = 0
    open_ends = 0

    # Positive direction
    r = row + dr
    c = col + dc
    while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
        count += 1
        r += dr
        c += dc
    if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
        open_ends += 1

    # Negative direction
    r = row - dr
    c = col - dc
    while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
        count += 1
        r -= dr
        c -= dc
    if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
        open_ends += 1

    return _pattern_score(count, open_ends, table)


@njit(cache=True)
def _threat_lines_at(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
) -> int:
    """Count distinct threat-level lines for *stone* passing through (row, col)."""
    line_count = 0
    for d in range(dirs.shape[0]):
        dr = dirs[d, 0]
        dc = dirs[d, 1]
        consecutive = 0
        open_ends = 0

        # Positive
        r = row + dr
        c = col + dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            consecutive += 1
            r += dr
            c += dc
        if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
            open_ends += 1

        # Negative
        r = row - dr
        c = col - dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            consecutive += 1
            r -= dr
            c -= dc
        if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
            open_ends += 1

        score = _pattern_score(consecutive, open_ends, table)
        if score >= fork_threshold:
            line_count += 1
    return line_count


@njit(cache=True)
def _fork_score(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    occ: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
    fork_bonus: int,
) -> float:
    """Compute the fork bonus for *stone*.

    Scans radius-2 neighborhood of every occupied cell.  For empty cells
    where >= 2 threat-level directions converge, adds a bonus.
    """
    bonus = 0.0
    n_occ = occ.shape[0]
    # Flat boolean grid used as ``seen`` set — avoids Python set objects.
    seen = np.zeros((size, size), dtype=np.bool_)

    for i in range(n_occ):
        occ_r = occ[i, 0]
        occ_c = occ[i, 1]
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr = occ_r + dr
                nc = occ_c + dc
                if nr < 0 or nr >= size or nc < 0 or nc >= size:
                    continue
                if seen[nr, nc]:
                    continue
                seen[nr, nc] = True
                if grid[nr, nc] != 0:
                    continue
                lines = _threat_lines_at(grid, size, nr, nc, stone, dirs, table, fork_threshold)
                if lines >= 2:
                    bonus += fork_bonus * (lines - 1)
    return bonus


@njit(cache=True)
def find_fork_cells_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    occ: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
) -> NDArray:  # type: ignore[type-arg]
    """Return empty cells where >= 2 threat-level lines for *stone* converge.

    Mirrors ``_fork_score`` but emits coordinates instead of a bonus.
    Returns an int32[K, 2] array of (row, col) pairs in scan order.
    """
    n_occ = occ.shape[0]
    seen = np.zeros((size, size), dtype=np.bool_)
    # Upper bound: every cell can be visited once.
    out = np.empty((size * size, 2), dtype=np.int32)
    out_count = 0

    for i in range(n_occ):
        occ_r = occ[i, 0]
        occ_c = occ[i, 1]
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr = occ_r + dr
                nc = occ_c + dc
                if nr < 0 or nr >= size or nc < 0 or nc >= size:
                    continue
                if seen[nr, nc]:
                    continue
                seen[nr, nc] = True
                if grid[nr, nc] != 0:
                    continue
                lines = _threat_lines_at(grid, size, nr, nc, stone, dirs, table, fork_threshold)
                if lines >= 2:
                    out[out_count, 0] = nr
                    out[out_count, 1] = nc
                    out_count += 1

    return out[:out_count].copy()


@njit(cache=True)
def _open_three_score(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    occ: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    open_three_bonus: int,
) -> float:
    """Count distinct open-three formations for *stone* and return a bonus.

    An open-three is exactly 3 consecutive same-colour stones with both ends
    empty.  Each formation is counted once (deduplicated via canonical start).
    """
    bonus = 0.0
    n_occ = occ.shape[0]
    # Track seen (start_row, start_col, dir_index) in a flat grid.
    # 4 directions × size × size entries.
    seen = np.zeros((4, size, size), dtype=np.bool_)

    for i in range(n_occ):
        r = occ[i, 0]
        c = occ[i, 1]
        if grid[r, c] != stone:
            continue
        for d in range(dirs.shape[0]):
            dr = dirs[d, 0]
            dc = dirs[d, 1]
            # Walk to canonical start of the run
            sr = r
            sc = c
            while True:
                pr = sr - dr
                pc = sc - dc
                if 0 <= pr < size and 0 <= pc < size and grid[pr, pc] == stone:
                    sr = pr
                    sc = pc
                else:
                    break
            if seen[d, sr, sc]:
                continue
            seen[d, sr, sc] = True

            # Count consecutive
            count = 0
            er = sr
            ec = sc
            while 0 <= er < size and 0 <= ec < size and grid[er, ec] == stone:
                count += 1
                er += dr
                ec += dc

            if count != 3:
                continue

            # Check both ends empty
            end_open = 0 <= er < size and 0 <= ec < size and grid[er, ec] == 0
            br = sr - dr
            bc = sc - dc
            start_open = 0 <= br < size and 0 <= bc < size and grid[br, bc] == 0

            if start_open and end_open:
                bonus += open_three_bonus
    return bonus


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@njit(cache=True)
def evaluate_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    occ: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
    fork_bonus: int,
    open_three_bonus: int,
) -> float:
    """Evaluate the board from *stone*'s perspective.  Higher is better.

    Parameters
    ----------
    grid : int8[size, size]
        Board state (0=empty, 1=BLACK, 2=WHITE).
    size : int
        Board edge length.
    occ : int32[N, 2]
        Coordinates of occupied cells (row, col).
    stone : int
        The stone whose perspective we evaluate (1 or 2).
    dirs : int32[4, 2]
        Direction vectors.
    table : int32[6, 3]
        Pattern-score lookup table.
    fork_threshold : int
        Minimum pattern score to count as a fork contributor.
    fork_bonus : int
        Score bonus per additional converging threat line.
    open_three_bonus : int
        Bonus per distinct open-three formation.
    """
    opponent = 3 - stone  # BLACK(1)↔WHITE(2)
    score = 0.0
    n_occ = occ.shape[0]

    for i in range(n_occ):
        r = occ[i, 0]
        c = occ[i, 1]
        cell = grid[r, c]
        for d in range(dirs.shape[0]):
            dr = dirs[d, 0]
            dc = dirs[d, 1]
            if cell == stone:
                score += _score_direction(grid, size, r, c, dr, dc, stone, table)
            else:
                score -= _score_direction(grid, size, r, c, dr, dc, opponent, table)

    # Fork potential
    score += _fork_score(grid, size, occ, stone, dirs, table, fork_threshold, fork_bonus)
    score -= _fork_score(grid, size, occ, opponent, dirs, table, fork_threshold, fork_bonus)

    # Open-three bonus
    score += _open_three_score(grid, size, occ, stone, dirs, open_three_bonus)
    score -= _open_three_score(grid, size, occ, opponent, dirs, open_three_bonus)

    return score


# ---------------------------------------------------------------------------
# Module-level constants exposed for callers
# ---------------------------------------------------------------------------


def get_pattern_table() -> NDArray[np.int32]:
    """Return the pattern-score lookup table."""
    return _PATTERN_TABLE


def get_dirs() -> NDArray[np.int32]:
    """Return the direction vectors array."""
    return _DIRS


def get_fork_threshold() -> int:
    return _FORK_THRESHOLD


def get_fork_bonus() -> int:
    return _FORK_BONUS


def get_open_three_bonus() -> int:
    return _OPEN_THREE_BONUS


# ---------------------------------------------------------------------------
# Tier 2: Candidate generation + threat detection
# ---------------------------------------------------------------------------


@njit(cache=True)
def get_candidates_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    occ: NDArray,  # type: ignore[type-arg]
    radius: int,
) -> NDArray:  # type: ignore[type-arg]
    """Return empty cells within *radius* of any occupied cell.

    Returns an int32[M, 2] array of (row, col) pairs, sorted by row then col.
    """
    seen = np.zeros((size, size), dtype=np.bool_)
    n_occ = occ.shape[0]
    # Upper bound: each occupied cell can contribute (2*radius+1)^2 neighbours.
    buf = np.empty((n_occ * (2 * radius + 1) ** 2, 2), dtype=np.int32)
    count = 0

    for i in range(n_occ):
        occ_r = occ[i, 0]
        occ_c = occ[i, 1]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = occ_r + dr
                nc = occ_c + dc
                if nr < 0 or nr >= size or nc < 0 or nc >= size:
                    continue
                if seen[nr, nc]:
                    continue
                seen[nr, nc] = True
                if grid[nr, nc] != 0:
                    continue
                buf[count, 0] = nr
                buf[count, 1] = nc
                count += 1

    result = buf[:count].copy()
    # Simple insertion sort — candidate lists are small (typically < 100).
    for i in range(1, count):
        key_r = result[i, 0]
        key_c = result[i, 1]
        j = i - 1
        while j >= 0 and (result[j, 0] > key_r or (result[j, 0] == key_r and result[j, 1] > key_c)):
            result[j + 1, 0] = result[j, 0]
            result[j + 1, 1] = result[j, 1]
            j -= 1
        result[j + 1, 0] = key_r
        result[j + 1, 1] = key_c
    return result


@njit(cache=True)
def find_winning_move_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
) -> NDArray:  # type: ignore[type-arg]
    """Return a ``(row, col)`` that completes 5-in-a-row for *stone*, or
    an empty ``(0, 2)`` array if none exists.

    Scans the directional neighbours of each candidate cell in place.
    Mirrors the pure-Python ``_find_winning_move`` but uses the int8
    grid for speed.
    """
    out = np.empty((1, 2), dtype=np.int32)
    n = candidates.shape[0]
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        for d in range(dirs.shape[0]):
            dr = dirs[d, 0]
            dc = dirs[d, 1]
            count = 1  # the stone we would place
            nr = r + dr
            nc = c + dc
            while 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == stone:
                count += 1
                nr += dr
                nc += dc
            nr = r - dr
            nc = c - dc
            while 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == stone:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 5:
                out[0, 0] = r
                out[0, 1] = c
                return out
    return np.empty((0, 2), dtype=np.int32)


@njit(cache=True)
def is_four_threat(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
) -> bool:
    """Return True if placing *stone* at (row, col) creates >= 4 in a row.

    Checks all four directions.  Used for threat ordering, threat extensions,
    and quiescence threat detection.
    """
    for d in range(dirs.shape[0]):
        dr = dirs[d, 0]
        dc = dirs[d, 1]
        count = 1  # the placed stone itself
        # Positive direction
        r = row + dr
        c = col + dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            count += 1
            r += dr
            c += dc
        # Negative direction
        r = row - dr
        c = col - dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            count += 1
            r -= dr
            c -= dc
        if count >= 4:
            return True
    return False


@njit(cache=True)
def has_any_four_threat_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
) -> bool:
    """Return True if any cell in *candidates* would create a four-threat.

    Short-circuits on the first hit — cheaper than ``find_four_threats_grid``
    when the caller only needs a boolean (e.g. VCT counter-four detection).
    """
    n = candidates.shape[0]
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        if grid[r, c] != 0:
            continue
        if is_four_threat(grid, size, r, c, stone, dirs):
            return True
    return False


@njit(cache=True)
def count_distinct_three_cells_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    cap: int,
) -> int:
    """Count distinct candidate cells that form a live-three for *stone*.

    Mirrors the unique-attacker-cell count used by
    ``ThreatBrain._check_opponent_vct``'s early-reject heuristic.  Stops
    counting once the running total reaches ``cap`` so callers asking
    "is this >= cap?" don't pay for the full scan.
    """
    n = candidates.shape[0]
    count = 0
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        if grid[r, c] != 0:
            continue
        for d in range(dirs.shape[0]):
            dr = dirs[d, 0]
            dc = dirs[d, 1]
            consec = 0
            pos_open = False
            neg_open = False
            rr = r + dr
            cc = c + dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                consec += 1
                rr += dr
                cc += dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                pos_open = True
            rr = r - dr
            cc = c - dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                consec += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                neg_open = True
            if consec + 1 == 3 and pos_open and neg_open:
                count += 1
                if count >= cap:
                    return count
                break
    return count


@njit(cache=True)
def get_threat_moves_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    opponent: int,
    dirs: NDArray,  # type: ignore[type-arg]
) -> NDArray:  # type: ignore[type-arg]
    """Return the subset of *candidates* that are four-threats for either side.

    Returns an int32[K, 2] array of (row, col) pairs.
    """
    n = candidates.shape[0]
    buf = np.empty((n, 2), dtype=np.int32)
    count = 0
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        if is_four_threat(grid, size, r, c, stone, dirs) or is_four_threat(grid, size, r, c, opponent, dirs):
            buf[count, 0] = r
            buf[count, 1] = c
            count += 1
    return buf[:count].copy()


@njit(cache=True)
def find_four_threats_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
) -> NDArray:  # type: ignore[type-arg]
    """Find four-threat moves for *stone* across *candidates*.

    Returns an int32[K, 4] array; each row is ``(atk_r, atk_c, def_r, def_c)``.
    ``def_r == -1`` (and ``def_c == -1``) means the threat is unblockable
    (live-four, five, or a double-four from the same cell).

    Mirrors the per-direction scan + per-move aggregation logic in
    ``gomoku_brain.threat._find_four_threats``.  See that docstring for
    classification rules.
    """
    n = candidates.shape[0]
    # Per-cell aggregation.  A move can produce several (direction-specific)
    # fours; if any is unblockable, the overall threat is unblockable; if
    # two or more distinct dead-fours fire from the same cell, it is also
    # effectively unblockable (double-four).
    unblockable = np.zeros((size, size), dtype=np.bool_)
    def_count = np.zeros((size, size), dtype=np.int32)
    def_r = np.full((size, size), -1, dtype=np.int32)
    def_c = np.full((size, size), -1, dtype=np.int32)

    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        for d in range(dirs.shape[0]):
            dr = dirs[d, 0]
            dc = dirs[d, 1]
            count = 0
            pos_end_r = -1
            pos_end_c = -1
            neg_end_r = -1
            neg_end_c = -1

            # Positive direction
            rr = r + dr
            cc = c + dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                count += 1
                rr += dr
                cc += dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                pos_end_r = rr
                pos_end_c = cc

            # Negative direction
            rr = r - dr
            cc = c - dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                count += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                neg_end_r = rr
                neg_end_c = cc

            open_ends = 0
            if pos_end_r >= 0:
                open_ends += 1
            if neg_end_r >= 0:
                open_ends += 1

            total = count + 1

            if total == 5:
                unblockable[r, c] = True
            elif total == 4 and open_ends == 2:
                unblockable[r, c] = True
            elif total == 4 and open_ends == 1:
                if pos_end_r >= 0:
                    dr_end = pos_end_r
                    dc_end = pos_end_c
                else:
                    dr_end = neg_end_r
                    dc_end = neg_end_c
                def_count[r, c] += 1
                def_r[r, c] = dr_end
                def_c[r, c] = dc_end

    # Emit one row per candidate that produced any threat.  Iterate the
    # candidate list in order to preserve the caller's expected ordering.
    out = np.empty((n, 4), dtype=np.int32)
    out_count = 0
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        if unblockable[r, c] or def_count[r, c] >= 2:
            out[out_count, 0] = r
            out[out_count, 1] = c
            out[out_count, 2] = -1
            out[out_count, 3] = -1
            out_count += 1
        elif def_count[r, c] == 1:
            out[out_count, 0] = r
            out[out_count, 1] = c
            out[out_count, 2] = def_r[r, c]
            out[out_count, 3] = def_c[r, c]
            out_count += 1
    return out[:out_count].copy()


@njit(cache=True)
def find_three_threats_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
) -> NDArray:  # type: ignore[type-arg]
    """Find live-three threats for *stone* across *candidates*.

    Returns an int32[K, 6] array; each row is
    ``(atk_r, atk_c, def1_r, def1_c, def2_r, def2_c)`` — the two open-end
    defence cells of the live three.

    One entry per ``(candidate, direction)`` pair that forms a live three.
    The same candidate may appear multiple times for different directions,
    matching the pure-Python implementation.
    """
    n = candidates.shape[0]
    # Upper bound: n candidates × 4 directions.
    out = np.empty((n * 4, 6), dtype=np.int32)
    out_count = 0

    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        for d in range(dirs.shape[0]):
            dr = dirs[d, 0]
            dc = dirs[d, 1]
            count = 0
            pos_end_r = -1
            pos_end_c = -1
            neg_end_r = -1
            neg_end_c = -1

            # Positive direction
            rr = r + dr
            cc = c + dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                count += 1
                rr += dr
                cc += dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                pos_end_r = rr
                pos_end_c = cc

            # Negative direction
            rr = r - dr
            cc = c - dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                count += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                neg_end_r = rr
                neg_end_c = cc

            open_ends = 0
            if pos_end_r >= 0:
                open_ends += 1
            if neg_end_r >= 0:
                open_ends += 1

            total = count + 1
            if total == 3 and open_ends == 2:
                out[out_count, 0] = r
                out[out_count, 1] = c
                # Match the Python order: positive-sign end first, then
                # negative-sign end.
                out[out_count, 2] = pos_end_r
                out[out_count, 3] = pos_end_c
                out[out_count, 4] = neg_end_r
                out[out_count, 5] = neg_end_c
                out_count += 1

    return out[:out_count].copy()


# ---------------------------------------------------------------------------
# Tier 3: Win detection + occupied derivation
# ---------------------------------------------------------------------------


@njit(cache=True)
def check_winner_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    win_length: int,
    dirs: NDArray,  # type: ignore[type-arg]
) -> bool:
    """Return True if there is a winning line through (row, col).

    Checks the stone at ``grid[row, col]`` and scans all four directions.
    Returns False when the cell is empty.
    """
    stone = grid[row, col]
    if stone == 0:
        return False
    for d in range(dirs.shape[0]):
        dr = dirs[d, 0]
        dc = dirs[d, 1]
        count = 1
        # Positive direction
        r = row + dr
        c = col + dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            count += 1
            r += dr
            c += dc
        # Negative direction
        r = row - dr
        c = col - dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            count += 1
            r -= dr
            c -= dc
        if count >= win_length:
            return True
    return False


@njit(cache=True)
def occupied_from_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
) -> NDArray:  # type: ignore[type-arg]
    """Derive occupied coordinates from a grid as an int32[N, 2] array.

    Scans row-major order so the result is sorted by (row, col).
    """
    buf = np.empty((size * size, 2), dtype=np.int32)
    count = 0
    for r in range(size):
        for c in range(size):
            if grid[r, c] != 0:
                buf[count, 0] = r
                buf[count, 1] = c
                count += 1
    return buf[:count].copy()


# ---------------------------------------------------------------------------
# MCTS prior scoring
# ---------------------------------------------------------------------------


@njit(cache=True)
def compute_priors_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    opponent: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
) -> NDArray:  # type: ignore[type-arg]
    """Return normalised MCTS prior probabilities for *candidates*.

    Mirrors the handwritten ``_score_move_prior`` + ``_compute_priors`` in
    ``gomoku_brain.mcts`` — per candidate, sums our own offensive pattern
    score plus 0.8 × the opponent's defensive pattern score across all four
    directions.  Output is an L1-normalised ``float64[N]`` array; uniform
    when every raw score is zero (only possible for an empty ``candidates``).
    """
    n = candidates.shape[0]
    raw = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        row = candidates[i, 0]
        col = candidates[i, 1]
        score = 1.0  # baseline so no move gets zero prior
        for d in range(dirs.shape[0]):
            dr = dirs[d, 0]
            dc = dirs[d, 1]

            # --- offensive: our own consecutive stones through (row, col)
            count = 0
            open_ends = 0
            rr = row + dr
            cc = col + dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                count += 1
                rr += dr
                cc += dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                open_ends += 1
            rr = row - dr
            cc = col - dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
                count += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                open_ends += 1
            score += _pattern_score(count, open_ends, table)

            # --- defensive: opponent consecutive stones through (row, col)
            count = 0
            open_ends = 0
            rr = row + dr
            cc = col + dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == opponent:
                count += 1
                rr += dr
                cc += dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                open_ends += 1
            rr = row - dr
            cc = col - dc
            while 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == opponent:
                count += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == 0:
                open_ends += 1
            score += 0.8 * _pattern_score(count, open_ends, table)

        raw[i] = score
        total += score

    if total == 0.0:
        if n == 0:
            return raw
        return np.full(n, 1.0 / n, dtype=np.float64)
    for i in range(n):
        raw[i] = raw[i] / total
    return raw
