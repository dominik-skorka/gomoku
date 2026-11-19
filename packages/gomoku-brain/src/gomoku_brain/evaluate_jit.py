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
def _fork_lines_at(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
) -> int:
    """Count the lines a fork at (row, col) would threaten on.

    As :func:`_threat_lines_at` (unbroken neighbours scoring a live two or
    better), plus every line where the move makes a three or a four —
    split ones included (``XX_X``, ``XXX_X``), which the run table can't
    see.  Used for fork *cells*; the evaluation's fork bonus keeps
    :func:`_threat_lines_at`.
    """
    line_count = 0
    for d in range(dirs.shape[0]):
        dr = dirs[d, 0]
        dc = dirs[d, 1]
        consecutive = 0
        open_ends = 0
        r = row + dr
        c = col + dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            consecutive += 1
            r += dr
            c += dc
        if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
            open_ends += 1
        r = row - dr
        c = col - dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            consecutive += 1
            r -= dr
            c -= dc
        if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
            open_ends += 1
        if (
            _pattern_score(consecutive, open_ends, table) >= fork_threshold
            or _three_defences(grid, size, row, col, stone, dr, dc)[0] > 0
            or _five_cells_through(grid, size, row, col, stone, dr, dc)[1] > 0
        ):
            line_count += 1
    return line_count


@njit(cache=True)
def find_fork_cells_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    occ: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
    split: bool = False,
) -> NDArray:  # type: ignore[type-arg]
    """Return empty cells where >= 2 threat-level lines for *stone* converge.

    With *split*, lines are counted by :func:`_fork_lines_at` (split threes
    and fours count), otherwise by :func:`_threat_lines_at`.

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
                if split:
                    lines = _fork_lines_at(grid, size, nr, nc, stone, dirs, table, fork_threshold)
                else:
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
# Incremental evaluation — local delta for a single placed stone
# ---------------------------------------------------------------------------


@njit(cache=True)
def _line_local_terms(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    dr: int,
    dc: int,
    table: NDArray,  # type: ignore[type-arg]
    open_three_bonus: int,
) -> float:
    """Black-positive directional + open-three terms for one full line.

    Decomposes the line through (row, col) along (dr, dc) into maximal
    same-colour runs.  Every stone in a run of length L sees
    ``count = L - 1`` consecutive neighbours and the run's two end
    opennesses, so the per-stone ``_score_direction`` sum for the line
    equals ``L * _pattern_score(L - 1, open_ends)`` per run; a run with
    ``L == 3`` and both ends empty additionally scores one open-three
    bonus — matching ``_open_three_score``'s canonical-start dedup,
    which counts each maximal run once.
    """
    # Walk to the line's first in-bounds cell.
    r = row
    c = col
    while 0 <= r - dr < size and 0 <= c - dc < size:
        r -= dr
        c -= dc

    total = 0.0
    prev = -1  # colour of the previous cell; -1 = board edge
    run_color = 0
    run_len = 0
    left_open = False
    while 0 <= r < size and 0 <= c < size:
        cell = grid[r, c]
        if cell == run_color and run_color != 0:
            run_len += 1
        else:
            if run_color != 0:
                right_open = cell == 0
                open_ends = (1 if left_open else 0) + (1 if right_open else 0)
                term = float(run_len * _pattern_score(run_len - 1, open_ends, table))
                if run_len == 3 and left_open and right_open:
                    term += open_three_bonus
                total += term if run_color == 1 else -term
            if cell != 0:
                run_color = cell
                run_len = 1
                left_open = prev == 0
            else:
                run_color = 0
                run_len = 0
        prev = cell
        r += dr
        c += dc
    if run_color != 0:
        # Run ran off the board — the far end is closed.
        open_ends = 1 if left_open else 0
        term = float(run_len * _pattern_score(run_len - 1, open_ends, table))
        total += term if run_color == 1 else -term
    return total


@njit(cache=True)
def _near_any_stone(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
) -> bool:
    """True if any occupied cell lies within Chebyshev distance 2."""
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r = row + dr
            c = col + dc
            if 0 <= r < size and 0 <= c < size and grid[r, c] != 0:
                return True
    return False


@njit(cache=True)
def _fork_cell_terms(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
    fork_bonus: int,
) -> float:
    """Black-minus-white fork contribution of the empty cell (row, col).

    Mirrors one cell of ``_fork_score`` for both colours; the caller is
    responsible for the eligibility gate (empty + near a stone).
    """
    total = 0.0
    lines_b = _threat_lines_at(grid, size, row, col, 1, dirs, table, fork_threshold)
    if lines_b >= 2:
        total += fork_bonus * (lines_b - 1)
    lines_w = _threat_lines_at(grid, size, row, col, 2, dirs, table, fork_threshold)
    if lines_w >= 2:
        total -= fork_bonus * (lines_w - 1)
    return total


@njit(cache=True)
def evaluate_delta_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    dirs: NDArray,  # type: ignore[type-arg]
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
    fork_bonus: int,
    open_three_bonus: int,
) -> float:
    """Black-positive evaluation delta caused by the stone at (row, col).

    ``grid`` must already contain the placed stone.  Returns
    ``E_after - E_before`` where ``E = evaluate_grid(..., stone=BLACK)``;
    the stone is toggled off and back on internally, so the grid is
    unchanged on return.

    Locality argument (all terms are restricted to a bounded window):

    * Directional + open-three terms are line-decomposable, and a stone
      changes only the run structure of the 4 lines through its cell —
      both states recompute those lines in full via
      :func:`_line_local_terms`.
    * The fork score of an empty cell *e* changes only if (a) its
      eligibility flips — possible only within Chebyshev distance 2 of
      the changed cell — or (b) one of its directional scans reads the
      changed cell.  A scan reads at most 5 consecutive stones plus one
      terminator per side (longer runs score 0 either way), so any
      affected *e* lies on one of the 4 lines through (row, col) within
      distance 6.
    """
    # Fork-dirty cell set — purely geometric, identical for both states.
    dirty = np.empty((1 + 4 * 12 + 24, 2), dtype=np.int32)
    seen = np.zeros((size, size), dtype=np.bool_)
    nd = 0
    seen[row, col] = True
    dirty[nd, 0] = row
    dirty[nd, 1] = col
    nd += 1
    for d in range(dirs.shape[0]):
        dr = dirs[d, 0]
        dc = dirs[d, 1]
        for sign in (1, -1):
            for k in range(1, 7):
                r = row + dr * k * sign
                c = col + dc * k * sign
                if r < 0 or r >= size or c < 0 or c >= size:
                    break
                if not seen[r, c]:
                    seen[r, c] = True
                    dirty[nd, 0] = r
                    dirty[nd, 1] = c
                    nd += 1
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r = row + dr
            c = col + dc
            if 0 <= r < size and 0 <= c < size and not seen[r, c]:
                seen[r, c] = True
                dirty[nd, 0] = r
                dirty[nd, 1] = c
                nd += 1

    stone_int = grid[row, col]

    after = 0.0
    for d in range(dirs.shape[0]):
        after += _line_local_terms(grid, size, row, col, dirs[d, 0], dirs[d, 1], table, open_three_bonus)
    for i in range(nd):
        r = dirty[i, 0]
        c = dirty[i, 1]
        if grid[r, c] == 0 and _near_any_stone(grid, size, r, c):
            after += _fork_cell_terms(grid, size, r, c, dirs, table, fork_threshold, fork_bonus)

    grid[row, col] = 0
    before = 0.0
    for d in range(dirs.shape[0]):
        before += _line_local_terms(grid, size, row, col, dirs[d, 0], dirs[d, 1], table, open_three_bonus)
    for i in range(nd):
        r = dirty[i, 0]
        c = dirty[i, 1]
        if grid[r, c] == 0 and _near_any_stone(grid, size, r, c):
            before += _fork_cell_terms(grid, size, r, c, dirs, table, fork_threshold, fork_bonus)
    grid[row, col] = stone_int

    return after - before


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

    Occupied candidate cells are skipped, so a candidate list computed
    *before* additional stones were placed stays valid — any cell that
    completes five for *stone* is adjacent to an existing *stone* stone
    and therefore already in the stale list.
    """
    out = np.empty((1, 2), dtype=np.int32)
    n = candidates.shape[0]
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        if grid[r, c] != 0:
            continue
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
def _five_cells_through(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    stone: int,
    dr: int,
    dc: int,
    split: bool = True,
) -> tuple[int, int, int, int, int, int]:
    """Scan the line through an empty (row, col) as if *stone* were placed there.

    Returns ``(run, n, r1, c1, r2, c2)``: ``run`` is the length of the
    unbroken *stone* run through the cell, and the ``n`` (0-2) cells
    ``(r1, c1)``, ``(r2, c2)`` are the empty cells that would then complete
    five or more along this line.  Only the first empty cell past each end
    of the run can (a five through the placed stone may not contain a
    second gap), and it does when the run, that cell and the *stone* run
    beyond it add up to five.  This catches the split fours
    (``XXX_X``, ``XX_XX``) a run count alone misses; with ``split=False``
    the stones beyond are not counted (unbroken runs only).
    """
    # Unbroken run on each side of the cell; (pr, pc) / (nr, nc) is the
    # first cell past it.
    pos = 0
    pr = row + dr
    pc = col + dc
    while 0 <= pr < size and 0 <= pc < size and grid[pr, pc] == stone:
        pos += 1
        pr += dr
        pc += dc
    neg = 0
    nr = row - dr
    nc = col - dc
    while 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == stone:
        neg += 1
        nr -= dr
        nc -= dc
    run = 1 + pos + neg
    n = 0
    r1 = -1
    c1 = -1
    r2 = -1
    c2 = -1
    if run >= 5:
        return run, n, r1, c1, r2, c2
    need = 4 - run  # stones needed past the end cell to complete five
    if 0 <= pr < size and 0 <= pc < size and grid[pr, pc] == 0:
        beyond = 0
        rr = pr + dr
        cc = pc + dc
        while split and beyond < need and 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
            beyond += 1
            rr += dr
            cc += dc
        if beyond >= need:
            r1 = pr
            c1 = pc
            n = 1
    if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0:
        beyond = 0
        rr = nr - dr
        cc = nc - dc
        while split and beyond < need and 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == stone:
            beyond += 1
            rr -= dr
            cc -= dc
        if beyond >= need:
            if n == 0:
                r1 = nr
                c1 = nc
            else:
                r2 = nr
                c2 = nc
            n += 1
    return run, n, r1, c1, r2, c2


@njit(cache=True)
def _three_defences(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    stone: int,
    dr: int,
    dc: int,
) -> tuple[int, int, int, int, int, int, int]:
    """Does placing *stone* at the empty (row, col) make a three on this line?

    A three is a move after which one more stone makes a *live* four
    (``_XXXX_``, completable at both ends): some window of four cells
    through the move holds three of *stone* and one empty cell, and the
    cells on both sides of the window are empty.  That covers the
    straight three ``_XXX_`` and the split threes ``_XX_X_``, ``_X_XX_``
    — and leaves out ``O_XXX_O``, which no move turns into a live four.

    Returns ``(n, r1, c1, r2, c2, r3, c3)``: the ``n`` (0-3) cells that
    stop every such live four (``n == 0``: no three).  For ``_XXX_``
    those are the two ends; for ``_XX_X_`` the gap and the two ends.
    """
    inter = -1  # bit i: offset i - 4 from (row, col), along (dr, dc)
    union = 0
    for k in range(-3, 1):
        ours = 0
        empty_off = 99
        ok = True
        for off in range(k, k + 4):
            if off == 0:
                ours += 1
                continue
            r = row + off * dr
            c = col + off * dc
            if not (0 <= r < size and 0 <= c < size):
                ok = False
                break
            v = grid[r, c]
            if v == stone:
                ours += 1
            elif v == 0 and empty_off == 99:
                empty_off = off
            else:
                ok = False
                break
        if not ok or ours != 3 or empty_off == 99:
            continue
        # Both cells flanking the window must be empty for a live four.
        ok = True
        for off in (k - 1, k + 4):
            r = row + off * dr
            c = col + off * dc
            if not (0 <= r < size and 0 <= c < size) or grid[r, c] != 0:
                ok = False
        if not ok:
            continue
        mask = (1 << (empty_off + 4)) | (1 << (k - 1 + 4)) | (1 << (k + 4 + 4))
        inter &= mask
        union |= mask
    if union == 0:
        return 0, -1, -1, -1, -1, -1, -1
    # A defence must stop every live four; if no single cell does, list
    # the cells of all of them (more defences is the safe side).
    bits = inter if inter != 0 else union
    n = 0
    r1 = c1 = r2 = c2 = r3 = c3 = -1
    for i in range(9):
        if (bits >> i) & 1:
            r = row + (i - 4) * dr
            c = col + (i - 4) * dc
            if n == 0:
                r1 = r
                c1 = c
            elif n == 1:
                r2 = r
                c2 = c
            elif n == 2:
                r3 = r
                c3 = c
            n += 1
    return min(n, 3), r1, c1, r2, c2, r3, c3


@njit(cache=True)
def _straight_three_defences(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    stone: int,
    dr: int,
    dc: int,
) -> tuple[int, int, int, int, int, int, int]:
    """The classic three test: an unbroken run of three with both ends empty.

    Same return shape as :func:`_three_defences`; the defences are the two
    ends.  Used when split threes are off (``split=False``).
    """
    run = 1
    pr = row + dr
    pc = col + dc
    while 0 <= pr < size and 0 <= pc < size and grid[pr, pc] == stone:
        run += 1
        pr += dr
        pc += dc
    nr = row - dr
    nc = col - dc
    while 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == stone:
        run += 1
        nr -= dr
        nc -= dc
    if (
        run == 3
        and 0 <= pr < size
        and 0 <= pc < size
        and grid[pr, pc] == 0
        and 0 <= nr < size
        and 0 <= nc < size
        and grid[nr, nc] == 0
    ):
        return 2, pr, pc, nr, nc, -1, -1
    return 0, -1, -1, -1, -1, -1, -1


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

    Checks all four directions.  An alpha-beta heuristic, used for threat
    ordering, threat extensions and quiescence move generation: it counts
    unbroken runs only, blocked or not.  Split fours (``XXX_X``,
    ``XX_XX``) are left out on purpose — quiescence also plays every cell
    where the opponent would make such a four, and counting split fours
    there nearly tripled its node count without a measurable gain.  The
    threat searches (VCF, PNS) use :func:`makes_four_threat` instead.
    """
    for d in range(dirs.shape[0]):
        dr = dirs[d, 0]
        dc = dirs[d, 1]
        count = 1  # the placed stone itself
        r = row + dr
        c = col + dc
        while 0 <= r < size and 0 <= c < size and grid[r, c] == stone:
            count += 1
            r += dr
            c += dc
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
def makes_four_threat(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    split: bool = True,
) -> bool:
    """Return True if placing *stone* at (row, col) threatens five next move.

    That is a five already, or a four — straight or split (``XXX_X``,
    ``XX_XX``) — with an empty completion cell.
    """
    for d in range(dirs.shape[0]):
        run, n_five, _r1, _c1, _r2, _c2 = _five_cells_through(
            grid, size, row, col, stone, dirs[d, 0], dirs[d, 1], split
        )
        if run >= 5 or n_five > 0:
            return True
    return False


@njit(cache=True)
def has_any_four_threat_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    split: bool = True,
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
        if makes_four_threat(grid, size, r, c, stone, dirs, split):
            return True
    return False


@njit(cache=True)
def has_unblockable_four_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    split: bool = True,
) -> bool:
    """Return True if *stone* has an unblockable four-threat cell.

    Unblockable means a five-or-more completion, a live four (both ends
    open), or a double-four (two dead-fours from the same cell) — the
    same classification ``find_four_threats_grid`` encodes as
    ``def_r == -1``, but aggregated per candidate and short-circuiting
    on the first hit.  Used by the VCF defender-refutation check, which
    only needs the boolean.

    Occupied candidate cells are skipped, so stale candidate lists are
    safe (see ``find_winning_move_grid``).
    """
    n = candidates.shape[0]
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        if grid[r, c] != 0:
            continue
        # Distinct cells that would complete five: two or more (a live
        # four, or fours on two lines) cannot all be blocked.
        first_r = -1
        first_c = -1
        for d in range(dirs.shape[0]):
            run, n_five, r1, c1, _r2, _c2 = _five_cells_through(grid, size, r, c, stone, dirs[d, 0], dirs[d, 1], split)
            if run >= 5 or n_five == 2:
                return True
            if n_five == 1:
                if first_r >= 0 and (first_r != r1 or first_c != c1):
                    return True
                first_r = r1
                first_c = c1
    return False


@njit(cache=True)
def count_distinct_three_cells_grid(
    grid: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]
    stone: int,
    dirs: NDArray,  # type: ignore[type-arg]
    cap: int,
    split: bool = False,
) -> int:
    """Count distinct candidate cells that form a three for *stone* (split ones with *split*).

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
            if split:
                found = _three_defences(grid, size, r, c, stone, dirs[d, 0], dirs[d, 1])[0] > 0
            else:
                found = _straight_three_defences(grid, size, r, c, stone, dirs[d, 0], dirs[d, 1])[0] > 0
            if found:
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
    split: bool = True,
) -> NDArray:  # type: ignore[type-arg]
    """Find four-threat moves for *stone* across *candidates*.

    Returns an int32[K, 4] array; each row is ``(atk_r, atk_c, def_r, def_c)``.
    ``def_r == -1`` (and ``def_c == -1``) means the threat is unblockable
    (live-four, five, or a double-four from the same cell).

    Mirrors the per-direction scan + per-move aggregation logic in
    ``gomoku_brain.threat._find_four_threats``.  See that docstring for
    classification rules.

    Occupied candidate cells are skipped, so stale candidate lists are
    safe (see ``find_winning_move_grid``).
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
        if grid[r, c] != 0:
            continue
        for d in range(dirs.shape[0]):
            run, n_five, r1, c1, _r2, _c2 = _five_cells_through(grid, size, r, c, stone, dirs[d, 0], dirs[d, 1], split)
            # >= 5 (not == 5): freestyle counts overlines as wins, so a
            # move completing six-or-more is just as unblockable.  Two
            # completion cells on one line is a live four.
            if run >= 5 or n_five == 2:
                unblockable[r, c] = True
            elif n_five == 1:
                def_count[r, c] += 1
                def_r[r, c] = r1
                def_c[r, c] = c1

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
    split: bool = False,
) -> NDArray:  # type: ignore[type-arg]
    """Find three-threats for *stone* across *candidates*.

    With *split*, split threes (``_XX_X_``) count and a three is any move
    after which one stone makes a live four (:func:`_three_defences`);
    otherwise only unbroken runs of three with both ends empty
    (:func:`_straight_three_defences`).

    Returns an int32[K, 8] array; each row is
    ``(atk_r, atk_c, d1_r, d1_c, d2_r, d2_c, d3_r, d3_c)`` — the cells that
    stop the three (see :func:`_three_defences`), unused ones ``-1``: the
    two ends of a straight three, or the gap and both ends of a split
    three.

    One entry per ``(candidate, direction)`` pair that forms a three.
    The same candidate may appear multiple times for different directions,
    matching the pure-Python implementation.

    Occupied candidate cells are skipped, so stale candidate lists are
    safe (see ``find_winning_move_grid``).
    """
    n = candidates.shape[0]
    # Upper bound: n candidates × 4 directions.
    out = np.empty((n * 4, 8), dtype=np.int32)
    out_count = 0

    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        if grid[r, c] != 0:
            continue
        for d in range(dirs.shape[0]):
            if split:
                n_def, r1, c1, r2, c2, r3, c3 = _three_defences(grid, size, r, c, stone, dirs[d, 0], dirs[d, 1])
            else:
                n_def, r1, c1, r2, c2, r3, c3 = _straight_three_defences(
                    grid, size, r, c, stone, dirs[d, 0], dirs[d, 1]
                )
            if n_def == 0:
                continue
            out[out_count, 0] = r
            out[out_count, 1] = c
            out[out_count, 2] = r1
            out[out_count, 3] = c1
            out[out_count, 4] = r2
            out[out_count, 5] = c2
            out[out_count, 6] = r3
            out[out_count, 7] = c3
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
