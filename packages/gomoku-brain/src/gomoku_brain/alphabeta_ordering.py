"""Move-classification helpers for ``AlphaBetaBrain``.

Pure pattern-scanning helpers that decide whether a candidate move is
*tactically interesting* — used for move ordering, threat extensions,
and quiescence triggering inside :mod:`gomoku_brain.alphabeta`.

Each function dispatches to a Numba JIT kernel when available and falls
back to a pure-Python scan otherwise.  All helpers accept either a
Pydantic :class:`Board`, a :class:`_GridBoard`, or a
:class:`SearchState` — the line scanners only read cell values, never
mutate state.
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.directions import DIRECTIONS as _DIRECTIONS


def _check_win(board: Board, move: Move) -> bool:
    """Return True if *move* creates a winning line on *board*.

    Dispatches to JIT ``check_winner_grid`` when *board* is a ``_GridBoard``
    or a ``SearchState``, otherwise falls back to ``check_winner``.
    """
    from gomoku_brain.grid_utils import _GridBoard
    from gomoku_brain.search_state import SearchState

    if isinstance(board, _GridBoard | SearchState):
        from gomoku_brain.evaluate_jit import check_winner_grid, get_dirs

        return check_winner_grid(board.grid, board.size, move.row, move.col, 5, get_dirs())
    return check_winner(board, move) is not None


def _count_open_threes_at(board: Board, row: int, col: int, stone: Stone) -> int:
    """Count open-three lines through (row, col) if *stone* were placed there.

    An open-three is exactly 3 consecutive stones (including the hypothetical
    one at *row*, *col*) with both ends empty.  Returns 0-4.
    """
    size = board.size
    open_threes = 0
    for dr, dc in _DIRECTIONS:
        count = 1  # the stone we'd place
        # Positive direction
        pr, pc = row + dr, col + dc
        while 0 <= pr < size and 0 <= pc < size and board.at(pr, pc) is stone:
            count += 1
            pr += dr
            pc += dc
        end1_open = 0 <= pr < size and 0 <= pc < size and board.at(pr, pc) is Stone.EMPTY
        # Negative direction
        nr, nc = row - dr, col - dc
        while 0 <= nr < size and 0 <= nc < size and board.at(nr, nc) is stone:
            count += 1
            nr -= dr
            nc -= dc
        end2_open = 0 <= nr < size and 0 <= nc < size and board.at(nr, nc) is Stone.EMPTY
        if count == 3 and end1_open and end2_open:
            open_threes += 1
    return open_threes


def _split_threats(
    board: Board,
    moves: list[Move],
    stone: Stone,
    *,
    include_forks: bool = False,
) -> tuple[list[Move], list[Move]]:
    """Partition *moves* into (threat_moves, non_threat_moves).

    A move is a threat if placing *stone* or *opponent* there creates >= 4
    in a row.  When *include_forks* is True, moves that create a double-three
    fork (2+ open threes) for either side are also promoted.
    Dispatches to JIT when enabled.
    """
    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import get_dirs, is_four_threat
        from gomoku_brain.grid_utils import board_to_grid

        grid = board_to_grid(board)
        size = board.size
        stone_int = int(stone)
        opp_int = int(stone.opponent())
        dirs = get_dirs()
        threat_list: list[Move] = []
        non_threat: list[Move] = []
        for m in moves:
            if is_four_threat(grid, size, m.row, m.col, stone_int, dirs) or is_four_threat(
                grid, size, m.row, m.col, opp_int, dirs
            ) or (include_forks and (
                _count_open_threes_at(board, m.row, m.col, stone) >= 2
                or _count_open_threes_at(board, m.row, m.col, stone.opponent()) >= 2
            )):
                threat_list.append(m)
            else:
                non_threat.append(m)
        return threat_list, non_threat

    # Pure-Python fallback
    opponent = stone.opponent()
    size = board.size
    threat_list = []
    non_threat = []
    for move in moves:
        is_threat = False
        for dr, dc in _DIRECTIONS:
            own = 1
            for sign in (1, -1):
                r, c = move.row + dr * sign, move.col + dc * sign
                while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
                    own += 1
                    r += dr * sign
                    c += dc * sign
            if own >= 4:
                is_threat = True
                break
        if not is_threat:
            for dr, dc in _DIRECTIONS:
                opp = 1
                for sign in (1, -1):
                    r, c = move.row + dr * sign, move.col + dc * sign
                    while 0 <= r < size and 0 <= c < size and board.at(r, c) is opponent:
                        opp += 1
                        r += dr * sign
                        c += dc * sign
                if opp >= 4:
                    is_threat = True
                    break
        if is_threat or (include_forks and (
            _count_open_threes_at(board, move.row, move.col, stone) >= 2
            or _count_open_threes_at(board, move.row, move.col, opponent) >= 2
        )):
            threat_list.append(move)
        else:
            non_threat.append(move)
    return threat_list, non_threat


def _is_four_threat_move(
    board: Board,
    move: Move,
    stone: Stone,
) -> bool:
    """Return True if *move* creates a four-threat for *stone* or blocks one for opponent.

    *board* may be either the pre- or post-move view: the line scanners only
    look at cells strictly adjacent to (``move.row``, ``move.col``), never at
    the centre itself, so both views give the same answer.
    """
    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import get_dirs, is_four_threat
        from gomoku_brain.grid_utils import board_to_grid

        dirs = get_dirs()
        grid = board_to_grid(board)
        size = board.size
        if is_four_threat(grid, size, move.row, move.col, int(stone), dirs):
            return True
        return is_four_threat(grid, size, move.row, move.col, int(stone.opponent()), dirs)

    # Pure-Python fallback
    size = board.size
    for dr, dc in _DIRECTIONS:
        cnt = 1
        for sign in (1, -1):
            r, c = move.row + dr * sign, move.col + dc * sign
            while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
                cnt += 1
                r += dr * sign
                c += dc * sign
        if cnt >= 4:
            return True
    opp = stone.opponent()
    for dr, dc in _DIRECTIONS:
        cnt = 1
        for sign in (1, -1):
            r, c = move.row + dr * sign, move.col + dc * sign
            while 0 <= r < size and 0 <= c < size and board.at(r, c) is opp:
                cnt += 1
                r += dr * sign
                c += dc * sign
        if cnt >= 4:
            return True

    # Fork detection: move creates a double-three (2+ open threes) for
    # either side — nearly as forcing as a four-threat.
    if _count_open_threes_at(board, move.row, move.col, stone) >= 2:
        return True
    return _count_open_threes_at(board, move.row, move.col, opp) >= 2
