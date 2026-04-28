"""Threat-detection primitives — pure pattern scanners with no search.

This module owns the low-level "what's threatening on the current
board?" question: counting consecutive lines, listing four-threats,
listing live-three threats, finding immediate wins, and the
``probe_immediate`` shortcut shared by all three brains.

It deliberately holds **no** search logic — VCF / VCT / PNS recursion
lives in :mod:`gomoku_brain.threat` (which re-exports the helpers below
for backward compatibility, since tests, ``pns.py``, ``mcts.py``, and
``alphabeta.py`` all import these symbols by name).
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.directions import DIRECTIONS as _DIRECTIONS


def _line_cells(
    board: Board,
    row: int,
    col: int,
    dr: int,
    dc: int,
    stone: Stone,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Count consecutive ``stone`` cells in both directions from (row, col).

    Returns (count, open_ends, list_of_empty_neighbours_at_open_ends).
    The stone at (row, col) is *not* included in count (caller decides context).
    """
    size = board.size
    count = 0
    open_ends = 0
    end_cells: list[tuple[int, int]] = []

    for sign in (1, -1):
        r, c = row + dr * sign, col + dc * sign
        while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
            count += 1
            r += dr * sign
            c += dc * sign
        if 0 <= r < size and 0 <= c < size and board.at(r, c) is Stone.EMPTY:
            open_ends += 1
            end_cells.append((r, c))

    return count, open_ends, end_cells


def _find_four_threats_python(board: Board, stone: Stone) -> list[tuple[Move, Move | None]]:
    """Pure-Python fallback for :func:`_find_four_threats`.

    Kept for boards that aren't routed through the JIT path (tests that
    instantiate Pydantic ``Board`` directly with JIT disabled).
    """
    by_move: dict[tuple[int, int], tuple[Move, list[Move | None]]] = {}
    candidates = get_candidate_moves(board, radius=2)

    for move in candidates:
        r, c = move.row, move.col
        for dr, dc in _DIRECTIONS:
            count, open_ends, end_cells = _line_cells(board, r, c, dr, dc, stone)
            total = count + 1  # +1 for the placed stone itself

            defence: Move | None
            if total == 5:
                defence = None
            elif total == 4 and open_ends == 2:
                defence = None
            elif total == 4 and open_ends == 1 and end_cells:
                dr2, dc2 = end_cells[0]
                defence = Move(row=dr2, col=dc2)
            else:
                continue

            key = (r, c)
            if key not in by_move:
                by_move[key] = (move, [])
            by_move[key][1].append(defence)

    threats: list[tuple[Move, Move | None]] = []
    for move, defences in by_move.values():
        if any(d is None for d in defences):
            threats.append((move, None))
        elif len(defences) >= 2:
            threats.append((move, None))
        else:
            threats.append((move, defences[0]))
    return threats


def _find_four_threats(board: Board, stone: Stone) -> list[tuple[Move, Move | None]]:
    """Find all moves that create a four-in-a-row for ``stone``.

    Returns a list of ``(attack_move, defence_move | None)`` sorted by
    *strength*: unblockable fours (``defence is None`` — live-fours,
    double-fours, or completed fives) come first, blockable dead-fours
    after.  Callers that iterate (VCF attacker loop, counter-threat
    scan) therefore hit wins and refutations earliest, maximising
    VCF / alpha-beta early termination.

    Detects:
    - Straight fours: placing a stone extends three-in-a-row to four.
    - Gap fours: placing a stone fills a gap inside a would-be five
      (e.g. XX_XX pattern).

    Dispatches to a Numba JIT kernel when available; pure-Python fallback
    is kept for the JIT-disabled path.  ``_line_cells`` scans outward from
    ``(r, c)`` and never reads the centre, so it is invariant to whether
    the stone has actually been placed there — we save the per-candidate
    board copy.
    """
    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import find_four_threats_grid, get_dirs
        from gomoku_brain.grid_utils import board_to_grid, occupied_coords

        occ = occupied_coords(board)
        if occ.shape[0] == 0:
            return []

        from gomoku_brain.evaluate_jit import get_candidates_grid

        grid = board_to_grid(board)
        candidates = get_candidates_grid(grid, board.size, occ, 2)
        result = find_four_threats_grid(grid, board.size, candidates, int(stone), get_dirs())
        threats: list[tuple[Move, Move | None]] = []
        for i in range(result.shape[0]):
            atk = Move(row=int(result[i, 0]), col=int(result[i, 1]))
            dr_end = int(result[i, 2])
            if dr_end < 0:
                threats.append((atk, None))
            else:
                threats.append((atk, Move(row=dr_end, col=int(result[i, 3]))))
    else:
        threats = _find_four_threats_python(board, stone)

    # Stable sort by strength: unblockables (defence is None) first.
    threats.sort(key=lambda t: 0 if t[1] is None else 1)
    return threats


def _find_three_threats_python(board: Board, stone: Stone) -> list[tuple[Move, list[Move]]]:
    """Pure-Python fallback for :func:`_find_three_threats`."""
    threats: list[tuple[Move, list[Move]]] = []
    candidates = get_candidate_moves(board, radius=2)

    for move in candidates:
        r, c = move.row, move.col
        for dr, dc in _DIRECTIONS:
            count, open_ends, end_cells = _line_cells(board, r, c, dr, dc, stone)
            total = count + 1
            if total == 3 and open_ends == 2:
                defence = [Move(row=er, col=ec) for er, ec in end_cells]
                threats.append((move, defence))
    return threats


def _find_three_threats(board: Board, stone: Stone) -> list[tuple[Move, list[Move]]]:
    """Find moves that create a *live* three (open at both ends).

    Returns ``(attack_move, defence_moves)`` — ``defence_moves`` are the
    two open-end cells of the threat.  (The full defence expansion,
    including counter-four cells, lives in :func:`_vct_defence_moves`.)

    Results are sorted by *strength*: moves that create multiple live
    threes (double- / triple-threes — i.e. forks that converge toward
    unblockable follow-ups) come first.  Both the JIT and Python paths
    emit one entry per ``(candidate, direction)`` pair, so the number of
    entries per attack cell is the fork degree and provides a natural
    strength proxy for the VCT attacker loop.
    """
    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import find_three_threats_grid, get_candidates_grid, get_dirs
        from gomoku_brain.grid_utils import board_to_grid, occupied_coords

        occ = occupied_coords(board)
        if occ.shape[0] == 0:
            return []

        grid = board_to_grid(board)
        candidates = get_candidates_grid(grid, board.size, occ, 2)
        result = find_three_threats_grid(grid, board.size, candidates, int(stone), get_dirs())
        threats: list[tuple[Move, list[Move]]] = []
        for i in range(result.shape[0]):
            atk = Move(row=int(result[i, 0]), col=int(result[i, 1]))
            defence = [
                Move(row=int(result[i, 2]), col=int(result[i, 3])),
                Move(row=int(result[i, 4]), col=int(result[i, 5])),
            ]
            threats.append((atk, defence))
    else:
        threats = _find_three_threats_python(board, stone)

    # Stable sort: attacks with higher fork degree (more directions
    # producing a live-three) first.
    counts: dict[tuple[int, int], int] = {}
    for atk, _ in threats:
        k = (atk.row, atk.col)
        counts[k] = counts.get(k, 0) + 1
    threats.sort(key=lambda t: -counts[(t[0].row, t[0].col)])
    return threats


def _find_winning_move(board: Board, stone: Stone) -> Move | None:
    """Return a move that immediately completes five-in-a-row, or ``None``.

    Scans neighbours of each candidate cell in place: no board copies, no
    ``check_winner`` call.  Matches the ``check_winner`` freestyle rule
    (``count >= 5`` wins).  Dispatches to a Numba kernel when JIT is on
    — critical for SearchState-based callers where ``board.at`` is a
    bitboard lookup.
    """
    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import find_winning_move_grid, get_candidates_grid, get_dirs
        from gomoku_brain.grid_utils import board_to_grid, occupied_coords

        occ = occupied_coords(board)
        if occ.shape[0] == 0:
            return None
        grid = board_to_grid(board)
        candidates = get_candidates_grid(grid, board.size, occ, 2)
        result = find_winning_move_grid(grid, board.size, candidates, int(stone), get_dirs())
        if result.shape[0] == 0:
            return None
        return Move(row=int(result[0, 0]), col=int(result[0, 1]))

    size = board.size
    for move in get_candidate_moves(board, radius=2):
        r, c = move.row, move.col
        for dr, dc in _DIRECTIONS:
            count = 1
            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and board.at(nr, nc) is stone:
                count += 1
                nr += dr
                nc += dc
            nr, nc = r - dr, c - dc
            while 0 <= nr < size and 0 <= nc < size and board.at(nr, nc) is stone:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 5:
                return move
    return None


def probe_immediate(board: Board, stone: Stone) -> tuple[Move | None, str | None]:
    """Return ``(move, tag)`` for a pre-search tactical shortcut.

    * ``(move, "instant_win")`` — ``stone`` has a move completing
      five-in-a-row.
    * ``(move, "instant_block")`` — the opponent has such a move; the
      returned cell is the block.
    * ``(None, None)`` — no immediate tactic; the caller should run its
      full search.

    Shared between :class:`AlphaBetaBrain`, :class:`ThreatBrain`, and
    :class:`MCTSBrain` so the three brains agree on which shortcut they
    take and what ``decision_path`` tag they emit.  Accepts any
    board-like object that is compatible with
    :func:`_find_winning_move` (Pydantic :class:`Board` or
    :class:`SearchState`).
    """
    ours = _find_winning_move(board, stone)
    if ours is not None:
        return ours, "instant_win"
    opp = _find_winning_move(board, stone.opponent())
    if opp is not None:
        return opp, "instant_block"
    return None, None
