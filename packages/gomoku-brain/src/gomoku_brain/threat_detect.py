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
from gomoku_brain.search_state import SearchState

# Shared JIT scan context: ``(grid, size, candidates, dirs)``.
#
# Building one costs an occupied-coords scan plus a candidate dilation —
# the dominant fixed cost of every detection call.  Search loops that fire
# several detection calls per node build the context once via
# :func:`_scan_context` and pass it down.  ``grid`` is the live
# ``SearchState`` mirror, so make/unmake mutations are visible to the
# kernels without rebuilding the context.
#
# Staleness contract: the candidate snapshot stays valid for detecting
# threats of a colour that has NOT placed a stone since the context was
# built — any win / four / three cell for that colour is adjacent to one
# of its (then-existing) stones and was therefore already a candidate,
# and the kernels skip cells that have become occupied.  After a colour
# moves, detection *for that colour* needs a fresh context (new threat
# cells may neighbour only the new stone).
_ScanCtx = tuple

# Lazily-resolved (get_candidates_grid, dirs) pair: ``None`` = not yet
# resolved, ``False`` = JIT disabled.  ``_scan_context`` runs ~20k times
# per search; re-importing the kernels and re-checking the JIT setting on
# every call used to cost more than the (memoised) kernel work itself.
_SCAN_KIT: tuple | None | bool = None


def _resolve_scan_kit() -> tuple | bool:
    global _SCAN_KIT
    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import get_candidates_grid, get_dirs

        _SCAN_KIT = (get_candidates_grid, get_dirs())
    else:
        _SCAN_KIT = False
    return _SCAN_KIT


def _scan_context(board: Board) -> _ScanCtx | None:
    """Build the shared JIT scan context for *board*, or ``None`` when JIT is off.

    For :class:`SearchState` the derived candidate array is memoised
    per ply, keyed by the zobrist hash it was derived from — quiescence,
    the VCF guard, and the VCF root each build a context for the same
    position, so the dilation kernel runs once per (ply, position)
    instead of once per call.  A make at the same ply changes the hash,
    so a stale slot simply misses; ancestor slots survive the
    make/unmake churn of child subtrees.
    """
    kit = _SCAN_KIT
    if kit is None:
        kit = _resolve_scan_kit()
    if kit is False:
        return None
    get_candidates_grid, dirs = kit

    if isinstance(board, SearchState):
        ply = board.ply
        cached = board._ctx_candidates[ply]
        if cached is not None and board._ctx_zobrist[ply] == board.zobrist:
            return board.grid, board.size, cached, dirs
        candidates = get_candidates_grid(board.grid, board.size, board._occ_arr[: board.occ_count], 2)
        board._ctx_zobrist[ply] = board.zobrist
        board._ctx_candidates[ply] = candidates
        board._ctx_moves[ply] = None
        return board.grid, board.size, candidates, dirs

    from gomoku_brain.grid_utils import board_to_grid, occupied_coords

    occ = occupied_coords(board)
    grid = board_to_grid(board)
    candidates = get_candidates_grid(grid, board.size, occ, 2)
    return grid, board.size, candidates, dirs


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

    size = board.size
    for move in candidates:
        r, c = move.row, move.col
        for dr, dc in _DIRECTIONS:
            count, _open_ends, end_cells = _line_cells(board, r, c, dr, dc, stone)
            total = count + 1  # +1 for the placed stone itself
            # An end cell completes five when the run, the cell and the
            # stones beyond it reach five — which also catches the split
            # fours (XXX_X, XX_XX) a run count alone misses.
            completions = []
            for er, ec in end_cells:
                sr, sc = (1, 1) if (er - r) * dr + (ec - c) * dc > 0 else (-1, -1)
                beyond = 0
                br, bc = er + sr * dr, ec + sc * dc
                while 0 <= br < size and 0 <= bc < size and board.at(br, bc) is stone:
                    beyond += 1
                    br, bc = br + sr * dr, bc + sc * dc
                if total + 1 + beyond >= 5:
                    completions.append(Move(row=er, col=ec))

            defence: Move | None
            # >= 5 (not == 5): freestyle counts overlines as wins, so a
            # move completing six-or-more is just as unblockable.
            if total >= 5 or len(completions) == 2:
                defence = None
            elif len(completions) == 1:
                defence = completions[0]
            else:
                continue

            key = (r, c)
            if key not in by_move:
                by_move[key] = (move, [])
            by_move[key][1].append(defence)

    threats: list[tuple[Move, Move | None]] = []
    for move, defences in by_move.values():
        if any(d is None for d in defences) or len(defences) >= 2:
            threats.append((move, None))
        else:
            threats.append((move, defences[0]))
    return threats


def _find_four_threats(board: Board, stone: Stone, ctx: _ScanCtx | None = None) -> list[tuple[Move, Move | None]]:
    """Find all moves that create a four-in-a-row for ``stone``.

    Returns a list of ``(attack_move, defence_move | None)`` sorted by
    *strength*: unblockable fours (``defence is None`` — live-fours,
    double-fours, or completed fives-and-longer) come first, blockable
    dead-fours after.  Callers that iterate (VCF attacker loop,
    counter-threat scan) therefore hit wins and refutations earliest,
    maximising VCF / alpha-beta early termination.

    A four-threat is a move after which one empty cell (its defence)
    would complete five or more through it; two such cells (a live four,
    or fours on two lines) make it unblockable.  Detects:
    - Straight fours: placing a stone extends three-in-a-row to four.
    - Split fours: the move leaves ``XXX_X`` or ``XX_XX``, whose gap
      completes five (from ``XX__X``, ``X_X_X`` and the like).
    - Fives: placing a stone fills a gap inside a would-be five
      (e.g. ``XX_XX``) or extends a four.

    Dispatches to a Numba JIT kernel when available; pure-Python fallback
    is kept for the JIT-disabled path.  ``_line_cells`` scans outward from
    ``(r, c)`` and never reads the centre, so it is invariant to whether
    the stone has actually been placed there — we save the per-candidate
    board copy.

    ``ctx`` is an optional precomputed :func:`_scan_context` — pass it
    when several detection calls share one position to skip the repeated
    occupied/candidate derivation.
    """
    if ctx is None:
        ctx = _scan_context(board)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import find_four_threats_grid

        grid, size, candidates, dirs = ctx
        result = find_four_threats_grid(grid, size, candidates, int(stone), dirs)
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


def _find_three_threats(board: Board, stone: Stone, ctx: _ScanCtx | None = None) -> list[tuple[Move, list[Move]]]:
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

    ``ctx`` is an optional precomputed :func:`_scan_context`.
    """
    if ctx is None:
        ctx = _scan_context(board)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import find_three_threats_grid

        grid, size, candidates, dirs = ctx
        result = find_three_threats_grid(grid, size, candidates, int(stone), dirs)
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


def _find_winning_move(board: Board, stone: Stone, ctx: _ScanCtx | None = None) -> Move | None:
    """Return a move that immediately completes five-in-a-row, or ``None``.

    Scans neighbours of each candidate cell in place: no board copies, no
    ``check_winner`` call.  Matches the ``check_winner`` freestyle rule
    (``count >= 5`` wins).  Dispatches to a Numba kernel when JIT is on
    — critical for SearchState-based callers where ``board.at`` is a
    bitboard lookup.

    ``ctx`` is an optional precomputed :func:`_scan_context`.
    """
    if ctx is None:
        ctx = _scan_context(board)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import find_winning_move_grid

        grid, size, candidates, dirs = ctx
        result = find_winning_move_grid(grid, size, candidates, int(stone), dirs)
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


def _has_any_four_threat(board: Board, stone: Stone, ctx: _ScanCtx | None = None) -> bool:
    """Return True if *stone* has any four-threat candidate.

    Short-circuits on the first hit — cheaper than enumerating every
    threat via :func:`_find_four_threats`.  Used by the threat brain's
    early-reject heuristic for opponent-VCT checks.
    """
    if ctx is None:
        ctx = _scan_context(board)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import has_any_four_threat_grid

        grid, size, candidates, dirs = ctx
        return bool(has_any_four_threat_grid(grid, size, candidates, int(stone), dirs))

    return bool(_find_four_threats_python(board, stone))


def _has_unblockable_four(board: Board, stone: Stone, ctx: _ScanCtx | None = None) -> bool:
    """Return True if *stone* has an unblockable four-threat.

    Unblockable = five-or-more completion, live four, or double-four —
    the threats :func:`_find_four_threats` reports with ``defence is
    None``, but detected with a short-circuiting kernel instead of a
    full enumeration.  Used by the VCF defender-refutation check.
    """
    if ctx is None:
        ctx = _scan_context(board)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import has_unblockable_four_grid

        grid, size, candidates, dirs = ctx
        return bool(has_unblockable_four_grid(grid, size, candidates, int(stone), dirs))

    return any(d is None for _, d in _find_four_threats_python(board, stone))


def _count_distinct_three_cells(board: Board, stone: Stone, cap: int, ctx: _ScanCtx | None = None) -> int:
    """Count distinct cells that create a live-three for *stone*, up to *cap*.

    Stops counting once the running total reaches ``cap`` — callers
    typically just need the answer to ``>= cap?``.
    """
    if ctx is None:
        ctx = _scan_context(board)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import count_distinct_three_cells_grid

        grid, size, candidates, dirs = ctx
        return int(count_distinct_three_cells_grid(grid, size, candidates, int(stone), dirs, cap))

    distinct: set[tuple[int, int]] = set()
    for atk, _ in _find_three_threats_python(board, stone):
        distinct.add((atk.row, atk.col))
        if len(distinct) >= cap:
            return len(distinct)
    return len(distinct)


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
