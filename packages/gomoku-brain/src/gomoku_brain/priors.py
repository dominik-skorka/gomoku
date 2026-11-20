"""Cheap pattern-based move priors shared by MCTS and alpha-beta."""

import numpy as np

from gomoku_brain.search_state import SearchState


def compute_priors(
    state: SearchState,
    candidates: list[tuple[int, int]],
    stone_val: int,
    opp_val: int,
) -> list[float]:
    """Return normalised move priors for *candidates*, ``(row, col)`` tuples.

    Each candidate scores 1 plus, per direction, the pattern score of the
    line it would extend for the mover plus 0.8x that of the opponent line
    it would block.  Used by MCTS as PUCT priors and by alpha-beta to rank
    candidates cheaply at shallow depths.

    Delegates to the Numba ``compute_priors_grid`` kernel on the JIT path;
    falls back to a compact pure-Python scan otherwise.  The Python path
    is only hit when JIT evaluation is disabled via
    :class:`~gomoku.settings.GomokuSettings`.
    """
    if not candidates:
        return []

    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import (
            compute_priors_grid,
            get_dirs,
            get_pattern_table,
        )

        cand_arr = np.asarray(candidates, dtype=np.int32).reshape(-1, 2)
        result = compute_priors_grid(
            state.grid,
            state.size,
            cand_arr,
            stone_val,
            opp_val,
            get_dirs(),
            get_pattern_table(),
        )
        return [float(x) for x in result]

    from gomoku_brain.directions import DIRECTIONS

    size = state.size
    grid = state.grid
    raw = []
    for row, col in candidates:
        score = 1.0
        for dr, dc in DIRECTIONS:
            for scanned, weight in ((stone_val, 1.0), (opp_val, 0.8)):
                count = 0
                open_ends = 0
                for sign in (1, -1):
                    r, c = row + dr * sign, col + dc * sign
                    while 0 <= r < size and 0 <= c < size and grid[r, c] == scanned:
                        count += 1
                        r += dr * sign
                        c += dc * sign
                    if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
                        open_ends += 1
                score += weight * _PY_PATTERN_SCORES.get((count, open_ends), 0.0)
        raw.append(score)
    total = sum(raw)
    if total == 0:
        uniform = 1.0 / len(candidates)
        return [uniform] * len(candidates)
    return [s / total for s in raw]


# Pure-Python fallback table — mirrors ``evaluate_jit._PATTERN_TABLE`` except
# the hot JIT path uses the numpy table directly.
_PY_PATTERN_SCORES: dict[tuple[int, int], float] = {
    (4, 2): 10_000.0,
    (4, 1): 1_000.0,
    (3, 2): 1_000.0,
    (3, 1): 100.0,
    (2, 2): 100.0,
    (2, 1): 10.0,
    (1, 2): 10.0,
    (1, 1): 1.0,
}
