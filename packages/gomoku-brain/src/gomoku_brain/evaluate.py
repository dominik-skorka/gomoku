"""Heuristic board evaluation for Gomoku AI."""

from gomoku.interfaces import BoardView
from gomoku.models import Move, Stone

from gomoku_brain.directions import DIRECTIONS as _DIRECTIONS
from gomoku_brain.line_patterns import five_cells_through, three_defences

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

# Bonus per distinct open-three (3 consecutive, both ends empty).
# An open-three is one move away from an unstoppable live-4, so it
# deserves a dedicated bonus on top of the per-stone pattern scoring.
_OPEN_THREE_BONUS = 1_500

# Hard ceiling for static evaluation.  Must stay below the win-score
# constant (100_000) used by the search so that heuristic sums can never
# be mistaken for a proven win/loss.
_EVAL_CLAMP = 99_000


def _score_direction(board: BoardView, row: int, col: int, dr: int, dc: int, stone: Stone) -> int:
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


def _threat_lines_at(board: BoardView, row: int, col: int, stone: Stone) -> int:
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


def _fork_lines_at(board: BoardView, row: int, col: int, stone: Stone) -> int:
    """Count the lines a fork at (row, col) would threaten on.

    As :func:`_threat_lines_at`, plus every line where the move makes a
    three or a four, split ones included (``XX_X``, ``XXX_X``).  Used for
    fork cells; the evaluation's fork bonus keeps :func:`_threat_lines_at`.
    """
    count = 0
    for dr, dc in _DIRECTIONS:
        if (
            _threat_lines_at_direction(board, row, col, dr, dc, stone)
            or three_defences(board, row, col, dr, dc, stone)
            or five_cells_through(board, row, col, dr, dc, stone)[1]
        ):
            count += 1
    return count


def _threat_lines_at_direction(board: BoardView, row: int, col: int, dr: int, dc: int, stone: Stone) -> bool:
    """One direction of :func:`_threat_lines_at`."""
    size = board.size
    consecutive = 0
    open_ends = 0
    for sign in (1, -1):
        r, c = row + sign * dr, col + sign * dc
        while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
            consecutive += 1
            r, c = r + sign * dr, c + sign * dc
        if 0 <= r < size and 0 <= c < size and board.at(r, c) is Stone.EMPTY:
            open_ends += 1
    return _PATTERN_SCORES.get((consecutive, open_ends), 0) >= _FORK_THRESHOLD


def _fork_score(board: BoardView, stone: Stone) -> float:
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


def find_fork_cells(board: BoardView, stone: Stone, split: bool = False) -> list[Move]:
    """Return empty cells where >= 2 distinct threat lines for *stone* converge.

    With *split*, a line also counts when the move makes a three or a four
    on it, split ones included (``XX_X``, ``XXX_X``), which the run-based
    threat lines miss.

    Used by threat brain to detect imminent fork points that need blocking.
    Dispatches to a Numba JIT kernel when available; the pure-Python
    fallback below is kept for the JIT-disabled path.
    """
    if _should_use_jit():
        from gomoku_brain.evaluate_jit import (
            find_fork_cells_grid,
            get_dirs,
            get_fork_threshold,
            get_pattern_table,
        )
        from gomoku_brain.grid_utils import board_to_grid, occupied_coords

        occ = occupied_coords(board)
        if occ.shape[0] == 0:
            return []
        grid = board_to_grid(board)
        result = find_fork_cells_grid(
            grid,
            board.size,
            occ,
            int(stone),
            get_dirs(),
            get_pattern_table(),
            get_fork_threshold(),
            split,
        )
        return [Move(row=int(result[i, 0]), col=int(result[i, 1])) for i in range(result.shape[0])]

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
                lines = _fork_lines_at(board, nr, nc, stone) if split else _threat_lines_at(board, nr, nc, stone)
                if lines >= 2:
                    forks.append(Move(row=nr, col=nc))
    return forks


def _open_three_score(board: BoardView, stone: Stone) -> float:
    """Count distinct open-three formations for *stone* and return a bonus.

    An open-three is exactly 3 consecutive same-colour stones in a line
    with both ends empty (i.e. a live-3 that can become a live-4 in one move).
    Each formation is counted once (deduplicated by starting cell + direction).
    """
    bonus = 0.0
    size = board.size
    seen: set[tuple[int, int, int, int]] = set()  # (row, col, dr, dc)

    for r, c in board.occupied:
        if board.at(r, c) is not stone:
            continue
        for dr, dc in _DIRECTIONS:
            # Walk to the start of the consecutive run in this direction
            sr, sc = r, c
            while True:
                pr, pc = sr - dr, sc - dc
                if 0 <= pr < size and 0 <= pc < size and board.at(pr, pc) is stone:
                    sr, sc = pr, pc
                else:
                    break

            # Deduplicate: only process from the canonical start
            key = (sr, sc, dr, dc)
            if key in seen:
                continue
            seen.add(key)

            # Count consecutive stones from (sr, sc)
            count = 0
            er, ec = sr, sc
            while 0 <= er < size and 0 <= ec < size and board.at(er, ec) is stone:
                count += 1
                er += dr
                ec += dc

            if count != 3:
                continue

            # Check both ends are empty
            # End after the run: (er, ec) is already one past
            end_open = 0 <= er < size and 0 <= ec < size and board.at(er, ec) is Stone.EMPTY
            # End before the run: one step before (sr, sc)
            br, bc = sr - dr, sc - dc
            start_open = 0 <= br < size and 0 <= bc < size and board.at(br, bc) is Stone.EMPTY

            if start_open and end_open:
                bonus += _OPEN_THREE_BONUS
    return bonus


# ---------------------------------------------------------------------------
# JIT setting — resolved once on first call
# ---------------------------------------------------------------------------

_use_jit: bool | None = None


def _should_use_jit() -> bool:
    """Return whether JIT evaluation is enabled (cached after first call)."""
    global _use_jit
    if _use_jit is None:
        from gomoku.settings import GomokuSettings

        _use_jit = GomokuSettings().evaluation.use_jit
    return _use_jit


# ---------------------------------------------------------------------------
# Pure-Python evaluate_board (fallback)
# ---------------------------------------------------------------------------


def _evaluate_board_python(board: BoardView, stone: Stone) -> float:
    """Evaluate the board using pure-Python loops."""
    opponent = Stone.WHITE if stone is Stone.BLACK else Stone.BLACK
    score = 0.0

    for r, c in board.occupied:
        cell = board.at(r, c)
        for dr, dc in _DIRECTIONS:
            if cell is stone:
                score += _score_direction(board, r, c, dr, dc, stone)
            else:
                score -= _score_direction(board, r, c, dr, dc, opponent)

    score += _fork_score(board, stone)
    score -= _fork_score(board, opponent)

    score += _open_three_score(board, stone)
    score -= _open_three_score(board, opponent)
    return score


# ---------------------------------------------------------------------------
# Incremental evaluation on SearchState
# ---------------------------------------------------------------------------

# Maximum plies to walk back looking for a cached ancestor evaluation.
# Each gap ply costs one local-delta kernel call (~a quarter of a full
# evaluation), so beyond a few plies a fresh full evaluation wins.
_EVAL_WALKBACK_CAP = 4

# Lazily-resolved (dirs, table, fork_threshold, fork_bonus, open_three_bonus)
# kernel-constant bundle — avoids five accessor calls per evaluation.
_jit_consts: tuple | None = None  # type: ignore[type-arg]


def _eval_state_black(state) -> float:  # type: ignore[no-untyped-def]
    """Black-positive evaluation of a :class:`SearchState`, incrementally.

    Uses the per-ply cache on the state: if an ancestor ply within
    ``_EVAL_WALKBACK_CAP`` has a cached value, the gap is filled with
    one :func:`evaluate_delta_grid` call per ply (the gap moves are
    stripped from the grid mirror and re-applied one by one, so each
    delta sees exactly its own position).  Otherwise a full
    ``evaluate_grid`` runs, and the parent ply is back-filled with one
    delta so sibling probes after the next make/unmake hit the cache.

    Every term is an integer accumulated in float64, so the incremental
    sum is bit-identical to a full recomputation.
    """
    global _jit_consts
    if _jit_consts is None:
        from gomoku_brain.evaluate_jit import (
            get_dirs,
            get_fork_bonus,
            get_fork_threshold,
            get_open_three_bonus,
            get_pattern_table,
        )

        _jit_consts = (
            get_dirs(),
            get_pattern_table(),
            get_fork_threshold(),
            get_fork_bonus(),
            get_open_three_bonus(),
        )
    from gomoku_brain.evaluate_jit import evaluate_delta_grid, evaluate_grid

    dirs, table, fork_threshold, fork_bonus, open_three_bonus = _jit_consts
    p = state.ply
    valid = state._eval_valid
    stack = state._eval_stack
    if valid[p]:
        return float(stack[p])

    grid = state.grid
    size = state.size

    # Nearest cached ancestor within the walk-back cap.
    anc = -1
    floor = p - _EVAL_WALKBACK_CAP
    q = p - 1
    while q >= 0 and q >= floor:
        if valid[q]:
            anc = q
            break
        q -= 1

    if anc >= 0 and anc == p - 1:
        # Common case: parent cached — one delta for the last move.
        e = float(stack[anc]) + evaluate_delta_grid(
            grid,
            size,
            int(state._undo_row[anc]),
            int(state._undo_col[anc]),
            dirs,
            table,
            fork_threshold,
            fork_bonus,
            open_three_bonus,
        )
        stack[p] = e
        valid[p] = True
    elif anc >= 0:
        # Strip the gap moves from the grid mirror, then re-apply them
        # one by one so each delta is computed on its own position.
        # Bitboards / zobrist are untouched — only the int8 mirror.
        for k in range(anc, p):
            grid[state._undo_row[k], state._undo_col[k]] = 0
        e = float(stack[anc])
        try:
            for k in range(anc, p):
                r = int(state._undo_row[k])
                c = int(state._undo_col[k])
                grid[r, c] = state._undo_stone[k]
                e += evaluate_delta_grid(grid, size, r, c, dirs, table, fork_threshold, fork_bonus, open_three_bonus)
                stack[k + 1] = e
                valid[k + 1] = True
        finally:
            for k in range(anc, p):
                grid[state._undo_row[k], state._undo_col[k]] = state._undo_stone[k]
    else:
        from gomoku_brain.grid_utils import occupied_coords

        occ = occupied_coords(state)
        e = float(evaluate_grid(grid, size, occ, 1, dirs, table, fork_threshold, fork_bonus, open_three_bonus))
        stack[p] = e
        valid[p] = True
        if p > 0:
            # Back-fill the parent so sibling evaluations hit the cache.
            stack[p - 1] = e - evaluate_delta_grid(
                grid,
                size,
                int(state._undo_row[p - 1]),
                int(state._undo_col[p - 1]),
                dirs,
                table,
                fork_threshold,
                fork_bonus,
                open_three_bonus,
            )
            valid[p - 1] = True
    return e


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_board(board: BoardView, stone: Stone) -> float:
    """Evaluate the board from ``stone``'s perspective. Higher is better.

    :class:`SearchState` boards use the incremental per-ply evaluation
    cache (see :func:`_eval_state_black`) when JIT is enabled.  Other
    board types dispatch to bitboard JIT when ``[bitboard] enabled =
    true``, else to numpy-grid JIT when ``[evaluation] use_jit = true``,
    otherwise pure Python.

    The return value is clamped to [-``_EVAL_CLAMP``, ``_EVAL_CLAMP``] so that
    heuristic sums can never be confused with the win/loss scores (±100 000)
    used by the search.
    """
    if _should_use_jit():
        from gomoku_brain.search_state import SearchState

        if isinstance(board, SearchState):
            e = _eval_state_black(board)
            raw = e if stone is Stone.BLACK else -e
            return max(-_EVAL_CLAMP, min(_EVAL_CLAMP, raw))

    from gomoku_brain.bitboard import _should_use_bitboard

    if _should_use_bitboard():
        from gomoku_brain.bitboard_jit import (
            evaluate_bb,
            get_fork_bonus,
            get_fork_threshold,
            get_open_three_bonus,
            get_pattern_table,
        )
        from gomoku_brain.grid_utils import board_to_bitboard

        black, white, size = board_to_bitboard(board)
        raw = evaluate_bb(
            black,
            white,
            size,
            int(stone),
            get_pattern_table(),
            get_fork_threshold(),
            get_fork_bonus(),
            get_open_three_bonus(),
        )
        return max(-_EVAL_CLAMP, min(_EVAL_CLAMP, raw))
    if _should_use_jit():
        from gomoku_brain.evaluate_jit import (
            evaluate_grid,
            get_dirs,
            get_fork_bonus,
            get_fork_threshold,
            get_open_three_bonus,
            get_pattern_table,
        )
        from gomoku_brain.grid_utils import board_to_grid, occupied_coords

        grid = board_to_grid(board)
        occ = occupied_coords(board)
        raw = evaluate_grid(
            grid,
            board.size,
            occ,
            int(stone),
            get_dirs(),
            get_pattern_table(),
            get_fork_threshold(),
            get_fork_bonus(),
            get_open_three_bonus(),
        )
        return max(-_EVAL_CLAMP, min(_EVAL_CLAMP, raw))
    return max(-_EVAL_CLAMP, min(_EVAL_CLAMP, _evaluate_board_python(board, stone)))
