"""Numba JIT-compiled bitboard operations for Gomoku.

Row-based representation: two ``uint32[size]`` arrays (one per stone colour).
Bit ``c`` of ``rows[r]`` is set iff a stone occupies cell ``(r, c)``.

Maximum supported board size: 25 (fits in ``uint32``).

All public functions in this module are ``@njit(cache=True)`` so they can be
called from other JIT code or from Python with automatic compilation on first
use.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

# Re-use the pattern table layout from evaluate_jit:
# Axes: [count 0..5][open_ends 0..2] → score.
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


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def grid_to_bitboard(
    grid: NDArray,  # type: ignore[type-arg]  # int8[size, size]
    size: int,
) -> tuple[NDArray, NDArray]:  # type: ignore[type-arg]
    """Convert an int8 grid to a pair of uint32 row-based bitboard arrays.

    Returns ``(black_rows, white_rows)`` each of shape ``(size,)``.
    """
    black = np.zeros(size, dtype=np.uint32)
    white = np.zeros(size, dtype=np.uint32)
    for r in range(size):
        for c in range(size):
            v = grid[r, c]
            if v == 1:
                black[r] |= np.uint32(1) << np.uint32(c)
            elif v == 2:
                white[r] |= np.uint32(1) << np.uint32(c)
    return black, white


@njit(cache=True)
def bitboard_to_grid(
    black: NDArray,  # type: ignore[type-arg]  # uint32[size]
    white: NDArray,  # type: ignore[type-arg]  # uint32[size]
    size: int,
) -> NDArray:  # type: ignore[type-arg]
    """Convert bitboard arrays back to an int8[size, size] grid."""
    grid = np.zeros((size, size), dtype=np.int8)
    for r in range(size):
        for c in range(size):
            bit = np.uint32(1) << np.uint32(c)
            if black[r] & bit:
                grid[r, c] = 1
            elif white[r] & bit:
                grid[r, c] = 2
    return grid


# ---------------------------------------------------------------------------
# Bit manipulation helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def _bit_set(rows: NDArray, row: int, col: int) -> NDArray:  # type: ignore[type-arg]
    """Return a copy with bit (row, col) set."""
    out = rows.copy()
    out[row] |= np.uint32(1) << np.uint32(col)
    return out


@njit(cache=True)
def _bit_clear(rows: NDArray, row: int, col: int) -> NDArray:  # type: ignore[type-arg]
    """Return a copy with bit (row, col) cleared."""
    out = rows.copy()
    out[row] &= ~(np.uint32(1) << np.uint32(col))
    return out


@njit(cache=True)
def _bit_test(rows: NDArray, row: int, col: int) -> bool:  # type: ignore[type-arg]
    """Test whether bit (row, col) is set."""
    return bool(rows[row] & (np.uint32(1) << np.uint32(col)))


@njit(cache=True)
def _popcount32(x: int) -> int:
    """Population count for a 32-bit integer (Hamming weight)."""
    x = np.uint32(x)
    x = x - ((x >> np.uint32(1)) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
    x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return int((x * np.uint32(0x01010101)) >> np.uint32(24))


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------


@njit(cache=True)
def check_winner_bb(
    rows: NDArray,  # type: ignore[type-arg]  # uint32[size] for ONE colour
    size: int,
    win_length: int,
) -> bool:
    """Return True if *rows* contains a winning line of length >= win_length.

    Checks all four directions using bitwise operations.
    """
    # --- Horizontal ---
    for r in range(size):
        h = rows[r]
        for _ in range(1, win_length):
            h &= rows[r] >> np.uint32(_)
        # Unrolled: for win_length=5, h = row & (row>>1) & (row>>2) & (row>>3) & (row>>4)
        # But the loop is cleaner and Numba unrolls small constant loops.
        if h != 0:
            return True

    # --- Vertical ---
    for r in range(size - win_length + 1):
        v = rows[r]
        for k in range(1, win_length):
            v &= rows[r + k]
        if v != 0:
            return True

    # --- Diagonal ↘ (row+1, col+1) ---
    for r in range(size - win_length + 1):
        d = rows[r]
        for k in range(1, win_length):
            d &= rows[r + k] >> np.uint32(k)
        if d != 0:
            return True

    # --- Diagonal ↗ (row+1, col-1) ---
    for r in range(size - win_length + 1):
        d = rows[r]
        for k in range(1, win_length):
            d &= rows[r + k] << np.uint32(k)
        if d != 0:
            return True

    return False


@njit(cache=True)
def check_winner_at_bb(
    rows: NDArray,  # type: ignore[type-arg]  # uint32[size] for ONE colour
    size: int,
    row: int,
    col: int,
    win_length: int,
) -> bool:
    """Return True if there is a winning line through (row, col) for the stone in *rows*.

    More efficient than full-board scan when you know the last move.
    """
    stone_bit = np.uint32(1) << np.uint32(col)
    if not (rows[row] & stone_bit):
        return False

    # --- Horizontal ---
    count = 1
    c = col + 1
    while c < size and (rows[row] & (np.uint32(1) << np.uint32(c))):
        count += 1
        c += 1
    c = col - 1
    while c >= 0 and (rows[row] & (np.uint32(1) << np.uint32(c))):
        count += 1
        c -= 1
    if count >= win_length:
        return True

    # --- Vertical ---
    count = 1
    r = row + 1
    while r < size and (rows[r] & stone_bit):
        count += 1
        r += 1
    r = row - 1
    while r >= 0 and (rows[r] & stone_bit):
        count += 1
        r -= 1
    if count >= win_length:
        return True

    # --- Diagonal ↘ ---
    count = 1
    r, c = row + 1, col + 1
    while r < size and c < size and (rows[r] & (np.uint32(1) << np.uint32(c))):
        count += 1
        r += 1
        c += 1
    r, c = row - 1, col - 1
    while r >= 0 and c >= 0 and (rows[r] & (np.uint32(1) << np.uint32(c))):
        count += 1
        r -= 1
        c -= 1
    if count >= win_length:
        return True

    # --- Diagonal ↗ ---
    count = 1
    r, c = row + 1, col - 1
    while r < size and c >= 0 and (rows[r] & (np.uint32(1) << np.uint32(c))):
        count += 1
        r += 1
        c -= 1
    r, c = row - 1, col + 1
    while r >= 0 and c < size and (rows[r] & (np.uint32(1) << np.uint32(c))):
        count += 1
        r -= 1
        c += 1
    return count >= win_length


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


@njit(cache=True)
def get_candidates_bb(
    black: NDArray,  # type: ignore[type-arg]
    white: NDArray,  # type: ignore[type-arg]
    size: int,
    radius: int,
) -> NDArray:  # type: ignore[type-arg]
    """Return empty cells within *radius* of any occupied cell.

    Uses bitwise dilation of the occupied mask.
    Returns int32[M, 2] array of (row, col) sorted by row then col.
    """
    # Occupied mask per row
    occ = np.zeros(size, dtype=np.uint32)
    for r in range(size):
        occ[r] = black[r] | white[r]

    # Column mask: limit to valid column bits
    col_mask = (np.uint32(1) << np.uint32(size)) - np.uint32(1)

    # Dilate: for each row, expand the mask by radius in all directions
    dilated = np.zeros(size, dtype=np.uint32)
    for r in range(size):
        # Vertical range of rows to consider
        r_lo = max(0, r - radius)
        r_hi = min(size - 1, r + radius)
        acc = np.uint32(0)
        for rr in range(r_lo, r_hi + 1):
            row_bits = occ[rr]
            # Horizontal dilation: shift left and right by 1..radius
            expanded = row_bits
            for shift in range(1, radius + 1):
                expanded |= (row_bits << np.uint32(shift)) | (row_bits >> np.uint32(shift))
            acc |= expanded
        dilated[r] = acc & col_mask

    # Candidates = dilated & ~occupied
    buf = np.empty((size * size, 2), dtype=np.int32)
    count = 0
    for r in range(size):
        cand_bits = dilated[r] & ~occ[r] & col_mask
        # Extract set bits
        c = 0
        tmp = cand_bits
        while tmp:
            if tmp & np.uint32(1):
                buf[count, 0] = r
                buf[count, 1] = c
                count += 1
            tmp >>= np.uint32(1)
            c += 1

    return buf[:count].copy()


# ---------------------------------------------------------------------------
# Pattern scoring (for evaluation)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _pattern_score_bb(count: int, open_ends: int, table: NDArray) -> int:  # type: ignore[type-arg]
    """Look up pattern score, clamping indices."""
    if count < 0 or count > 5 or open_ends < 0 or open_ends > 2:
        return 0
    return table[count, open_ends]


@njit(cache=True)
def _score_direction_bb(
    rows: NDArray,  # type: ignore[type-arg]  # uint32[size] for ONE colour
    all_occ: NDArray,  # type: ignore[type-arg]  # uint32[size] occupied mask
    size: int,
    row: int,
    col: int,
    dr: int,
    dc: int,
    table: NDArray,  # type: ignore[type-arg]
) -> int:
    """Score a single direction from (row, col) for the stone colour in *rows*."""
    count = 0
    open_ends = 0

    # Positive direction
    r = row + dr
    c = col + dc
    while 0 <= r < size and 0 <= c < size and (rows[r] & (np.uint32(1) << np.uint32(c))):
        count += 1
        r += dr
        c += dc
    if 0 <= r < size and 0 <= c < size and not (all_occ[r] & (np.uint32(1) << np.uint32(c))):
        open_ends += 1

    # Negative direction
    r = row - dr
    c = col - dc
    while 0 <= r < size and 0 <= c < size and (rows[r] & (np.uint32(1) << np.uint32(c))):
        count += 1
        r -= dr
        c -= dc
    if 0 <= r < size and 0 <= c < size and not (all_occ[r] & (np.uint32(1) << np.uint32(c))):
        open_ends += 1

    return _pattern_score_bb(count, open_ends, table)


@njit(cache=True)
def _threat_lines_at_bb(
    rows: NDArray,  # type: ignore[type-arg]
    all_occ: NDArray,  # type: ignore[type-arg]
    size: int,
    row: int,
    col: int,
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
) -> int:
    """Count distinct threat-level lines for the colour in *rows* at (row, col)."""
    # 4 directions: (0,1), (1,0), (1,1), (1,-1)
    dirs_r = np.array([0, 1, 1, 1], dtype=np.int32)
    dirs_c = np.array([1, 0, 1, -1], dtype=np.int32)
    line_count = 0
    for d in range(4):
        dr = dirs_r[d]
        dc = dirs_c[d]
        score = _score_direction_bb(rows, all_occ, size, row, col, dr, dc, table)
        if score >= fork_threshold:
            line_count += 1
    return line_count


@njit(cache=True)
def _fork_score_bb(
    rows: NDArray,  # type: ignore[type-arg]
    black: NDArray,  # type: ignore[type-arg]
    white: NDArray,  # type: ignore[type-arg]
    size: int,
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
    fork_bonus: int,
) -> float:
    """Compute fork bonus for the colour in *rows*."""
    all_occ = np.zeros(size, dtype=np.uint32)
    for r in range(size):
        all_occ[r] = black[r] | white[r]

    col_mask = (np.uint32(1) << np.uint32(size)) - np.uint32(1)

    # Dilate by radius=2 to find candidate empty cells near stones
    dilated = np.zeros(size, dtype=np.uint32)
    for r in range(size):
        r_lo = max(0, r - 2)
        r_hi = min(size - 1, r + 2)
        acc = np.uint32(0)
        for rr in range(r_lo, r_hi + 1):
            row_bits = all_occ[rr]
            expanded = row_bits
            for shift in range(1, 3):
                expanded |= (row_bits << np.uint32(shift)) | (row_bits >> np.uint32(shift))
            acc |= expanded
        dilated[r] = acc & col_mask

    bonus = 0.0
    for r in range(size):
        cand_bits = dilated[r] & ~all_occ[r] & col_mask
        c = 0
        tmp = cand_bits
        while tmp:
            if tmp & np.uint32(1):
                lines = _threat_lines_at_bb(rows, all_occ, size, r, c, table, fork_threshold)
                if lines >= 2:
                    bonus += fork_bonus * (lines - 1)
            tmp >>= np.uint32(1)
            c += 1
    return bonus


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


@njit(cache=True)
def _open_three_score_bb(
    rows: NDArray,  # type: ignore[type-arg]  # uint32[size] for ONE colour
    all_occ: NDArray,  # type: ignore[type-arg]  # uint32[size] occupied mask
    size: int,
    open_three_bonus: int,
) -> float:
    """Count distinct open-three formations for the colour in *rows*.

    An open-three is exactly 3 consecutive same-colour stones with both ends
    empty.  Each formation is counted once.
    """
    dirs_r = np.array([0, 1, 1, 1], dtype=np.int32)
    dirs_c = np.array([1, 0, 1, -1], dtype=np.int32)
    bonus = 0.0
    # Track seen canonical starts: (dir, row, col)
    seen = np.zeros((4, size, size), dtype=np.bool_)

    for r in range(size):
        c = 0
        tmp = rows[r]
        while tmp:
            if tmp & np.uint32(1):
                for d in range(4):
                    dr = dirs_r[d]
                    dc = dirs_c[d]
                    # Walk to canonical start
                    sr = r
                    sc = c
                    while True:
                        pr = sr - dr
                        pc = sc - dc
                        if 0 <= pr < size and 0 <= pc < size and (rows[pr] & (np.uint32(1) << np.uint32(pc))):
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
                    while 0 <= er < size and 0 <= ec < size and (rows[er] & (np.uint32(1) << np.uint32(ec))):
                        count += 1
                        er += dr
                        ec += dc
                    if count != 3:
                        continue

                    # Check both ends empty
                    end_open = 0 <= er < size and 0 <= ec < size and not (all_occ[er] & (np.uint32(1) << np.uint32(ec)))
                    br = sr - dr
                    bc = sc - dc
                    start_open = (
                        0 <= br < size and 0 <= bc < size and not (all_occ[br] & (np.uint32(1) << np.uint32(bc)))
                    )
                    if start_open and end_open:
                        bonus += open_three_bonus
            tmp >>= np.uint32(1)
            c += 1
    return bonus


@njit(cache=True)
def evaluate_bb(
    black: NDArray,  # type: ignore[type-arg]
    white: NDArray,  # type: ignore[type-arg]
    size: int,
    stone: int,
    table: NDArray,  # type: ignore[type-arg]
    fork_threshold: int,
    fork_bonus: int,
    open_three_bonus: int,
) -> float:
    """Evaluate the board from *stone*'s perspective using bitboards.

    Parameters
    ----------
    black : uint32[size]
        Row-based bitboard for BLACK stones.
    white : uint32[size]
        Row-based bitboard for WHITE stones.
    size : int
        Board edge length.
    stone : int
        The stone whose perspective we evaluate (1=BLACK, 2=WHITE).
    table : int32[6, 3]
        Pattern-score lookup table.
    fork_threshold, fork_bonus : int
        Fork detection parameters.
    open_three_bonus : int
        Bonus per distinct open-three formation.
    """
    if stone == 1:
        my_rows = black
        opp_rows = white
    else:
        my_rows = white
        opp_rows = black

    all_occ = np.zeros(size, dtype=np.uint32)
    for r in range(size):
        all_occ[r] = black[r] | white[r]

    # 4 directions
    dirs_r = np.array([0, 1, 1, 1], dtype=np.int32)
    dirs_c = np.array([1, 0, 1, -1], dtype=np.int32)

    score = 0.0

    # Score occupied cells
    for r in range(size):
        row_occ = all_occ[r]
        c = 0
        tmp = row_occ
        while tmp:
            if tmp & np.uint32(1):
                is_mine = bool(my_rows[r] & (np.uint32(1) << np.uint32(c)))
                for d in range(4):
                    dr = dirs_r[d]
                    dc = dirs_c[d]
                    if is_mine:
                        score += _score_direction_bb(my_rows, all_occ, size, r, c, dr, dc, table)
                    else:
                        score -= _score_direction_bb(opp_rows, all_occ, size, r, c, dr, dc, table)
            tmp >>= np.uint32(1)
            c += 1

    # Fork potential
    score += _fork_score_bb(my_rows, black, white, size, table, fork_threshold, fork_bonus)
    score -= _fork_score_bb(opp_rows, black, white, size, table, fork_threshold, fork_bonus)

    # Open-three bonus
    score += _open_three_score_bb(my_rows, all_occ, size, open_three_bonus)
    score -= _open_three_score_bb(opp_rows, all_occ, size, open_three_bonus)

    return score


# ---------------------------------------------------------------------------
# Four-threat detection
# ---------------------------------------------------------------------------


@njit(cache=True)
def is_four_threat_bb(
    rows: NDArray,  # type: ignore[type-arg]  # uint32[size] for ONE colour
    size: int,
    row: int,
    col: int,
) -> bool:
    """Return True if placing a stone at (row, col) creates >= 4 in a row for *rows*.

    Assumes (row, col) already has the bit set in *rows*.
    """
    dirs_r = np.array([0, 1, 1, 1], dtype=np.int32)
    dirs_c = np.array([1, 0, 1, -1], dtype=np.int32)
    for d in range(4):
        dr = dirs_r[d]
        dc = dirs_c[d]
        count = 1
        # Positive
        r = row + dr
        c = col + dc
        while 0 <= r < size and 0 <= c < size and (rows[r] & (np.uint32(1) << np.uint32(c))):
            count += 1
            r += dr
            c += dc
        # Negative
        r = row - dr
        c = col - dc
        while 0 <= r < size and 0 <= c < size and (rows[r] & (np.uint32(1) << np.uint32(c))):
            count += 1
            r -= dr
            c -= dc
        if count >= 4:
            return True
    return False


@njit(cache=True)
def get_threat_moves_bb(
    black: NDArray,  # type: ignore[type-arg]
    white: NDArray,  # type: ignore[type-arg]
    size: int,
    candidates: NDArray,  # type: ignore[type-arg]  # int32[N, 2]
    stone: int,
) -> NDArray:  # type: ignore[type-arg]
    """Return subset of candidates that are four-threats for either side.

    Returns int32[K, 2] array of (row, col).
    """
    if stone == 1:
        my_rows = black
        opp_rows = white
    else:
        my_rows = white
        opp_rows = black

    n = candidates.shape[0]
    buf = np.empty((n, 2), dtype=np.int32)
    count = 0
    for i in range(n):
        r = candidates[i, 0]
        c = candidates[i, 1]
        # Temporarily set the bit to check
        my_test = my_rows.copy()
        my_test[r] |= np.uint32(1) << np.uint32(c)
        opp_test = opp_rows.copy()
        opp_test[r] |= np.uint32(1) << np.uint32(c)
        if is_four_threat_bb(my_test, size, r, c) or is_four_threat_bb(opp_test, size, r, c):
            buf[count, 0] = r
            buf[count, 1] = c
            count += 1
    return buf[:count].copy()


# ---------------------------------------------------------------------------
# Occupied cells extraction
# ---------------------------------------------------------------------------


@njit(cache=True)
def occupied_from_bb(
    black: NDArray,  # type: ignore[type-arg]
    white: NDArray,  # type: ignore[type-arg]
    size: int,
) -> NDArray:  # type: ignore[type-arg]
    """Return occupied coordinates as int32[N, 2] from bitboard arrays."""
    buf = np.empty((size * size, 2), dtype=np.int32)
    count = 0
    for r in range(size):
        occ = black[r] | white[r]
        c = 0
        tmp = occ
        while tmp:
            if tmp & np.uint32(1):
                buf[count, 0] = r
                buf[count, 1] = c
                count += 1
            tmp >>= np.uint32(1)
            c += 1
    return buf[:count].copy()


# ---------------------------------------------------------------------------
# Module-level constant accessors (mirrors evaluate_jit API)
# ---------------------------------------------------------------------------


def get_pattern_table() -> NDArray[np.int32]:
    return _PATTERN_TABLE


def get_fork_threshold() -> int:
    return _FORK_THRESHOLD


def get_fork_bonus() -> int:
    return _FORK_BONUS


def get_open_three_bonus() -> int:
    return _OPEN_THREE_BONUS
