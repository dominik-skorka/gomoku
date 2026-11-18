"""Pure-Python line patterns: which cells complete five, which stop a three.

The fallback for the JIT kernels ``_five_cells_through`` and
``_three_defences`` in :mod:`gomoku_brain.evaluate_jit`, with the same
rules.  Each function looks at one line through an empty cell as if
*stone* were placed there.
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Stone


def five_cells_through(board: Board, row: int, col: int, dr: int, dc: int, stone: Stone) -> tuple[int, list]:
    """Return ``(run, cells)``: the unbroken run through the cell, and the
    empty cells (0-2) that would then complete five or more on this line.

    Only the first cell past each end of the run can, when the run, that
    cell and the *stone* run beyond it add up to five — which catches the
    split fours ``XXX_X`` and ``XX_XX``.
    """
    size = board.size
    ends = []
    run = 1
    for sign in (1, -1):
        r, c = row + sign * dr, col + sign * dc
        while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
            run += 1
            r, c = r + sign * dr, c + sign * dc
        ends.append((r, c, sign))
    if run >= 5:
        return run, []
    cells = []
    for r, c, sign in ends:
        if not (0 <= r < size and 0 <= c < size) or board.at(r, c) is not Stone.EMPTY:
            continue
        beyond = 0
        br, bc = r + sign * dr, c + sign * dc
        while beyond < 4 - run and 0 <= br < size and 0 <= bc < size and board.at(br, bc) is stone:
            beyond += 1
            br, bc = br + sign * dr, bc + sign * dc
        if run + 1 + beyond >= 5:
            cells.append((r, c))
    return run, cells


def three_defences(board: Board, row: int, col: int, dr: int, dc: int, stone: Stone) -> list[tuple[int, int]]:
    """The cells that stop the three placing *stone* here makes on this line.

    A three is a move after which one more stone makes a *live* four
    (completable at both ends): a window of four cells through the move
    holding three of *stone* and one empty cell, with empty cells on both
    sides of the window.  The defences are the cells common to every such
    window's empty cell and flanks: the two ends of ``_XXX_``, the gap and
    both ends of ``_XX_X_``.  An empty list means no three.
    """
    size = board.size

    def cell(off: int) -> tuple[int, int] | None:
        r, c = row + off * dr, col + off * dc
        return (r, c) if 0 <= r < size and 0 <= c < size else None

    windows = []
    for k in range(-3, 1):
        ours, empty = 0, None
        for off in range(k, k + 4):
            if off == 0:
                ours += 1
                continue
            pos = cell(off)
            if pos is None:
                break
            value = board.at(*pos)
            if value is stone:
                ours += 1
            elif value is Stone.EMPTY and empty is None:
                empty = pos
            else:
                break
        else:
            flanks = [cell(k - 1), cell(k + 4)]
            if ours == 3 and empty is not None and all(f is not None and board.at(*f) is Stone.EMPTY for f in flanks):
                windows.append({empty, *flanks})
    if not windows:
        return []
    common = set.intersection(*windows)
    # No single cell stops every live four: list them all (the safe side).
    chosen = common or set.union(*windows)
    return sorted(chosen)[:3]
