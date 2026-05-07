"""VCF and VCT recursive search cores.

Pure search logic — no orchestration, no diagnostics, no logfire.
Drives a mutable :class:`SearchState` via ``make`` / ``unmake`` and
delegates pattern detection to the helpers defined in
:mod:`gomoku_brain.threat` (``_find_four_threats``,
``_find_three_threats``, ``_find_winning_move``).

These functions are re-exported from :mod:`gomoku_brain.threat` for
backward compatibility, since :mod:`gomoku_brain.alphabeta`,
:mod:`gomoku_brain.mcts`, :mod:`gomoku_brain.pns`, and several test
files still import them by name from there.

Public surface:

* :func:`_state_is_win`              — terminal-position check on a
  ``SearchState``.
* :func:`_vcf` / :func:`_vcf_state`  — Victory-by-Continuous-Fours.
  *Attacker* may only play four-threats; *defender* must play the
  forced single-cell block.
* :func:`_vct` / :func:`_vct_state`  — Victory-by-Continuous-Threats.
  Extends VCF with live-three attacks; defender enumerates the open-end
  blocks plus any counter four-threat.
* :func:`classify_vcf_breakers`      — partition candidate replies into
  (durable, all) breakers when the opponent has a known VCF.
* :func:`_vct_defence_moves` /
  :func:`_vct_defence_moves_state`   — full defence enumeration for a
  three-threat (open-end blocks + counter-fours).
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.directions import DIRECTIONS as _DIRECTIONS
from gomoku_brain.search_state import SearchState

# Imported from :mod:`gomoku_brain.threat` — that module's top-level
# definitions execute *before* it imports from this module, so by the
# time this import resolves the helper names are present on the
# partially-loaded ``threat`` module.  See ``threat.py`` for the
# definitions.
from gomoku_brain.threat import (
    _find_four_threats,
    _find_three_threats,
    _find_winning_move,
)

# Cache entry: (proven, depth_searched, attack_move)
# * proven=True  -> winning attack found within ``depth_searched`` plies.
#                   Valid whenever the query depth is >= depth_searched.
# * proven=False -> no winning sequence within ``depth_searched`` plies.
#                   Valid whenever the query depth is <= depth_searched.
_CacheEntry = tuple[bool, int, "Move | None"]
_ThreatCache = dict[int, _CacheEntry]


# ---------------------------------------------------------------------------
# Terminal check
# ---------------------------------------------------------------------------


def _state_is_win(state: SearchState, row: int, col: int) -> bool:
    """Return True if the stone at ``(row, col)`` completes 5-in-a-row.

    Uses the JIT grid kernel; falls back to a Python loop if JIT is off.
    """
    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import check_winner_grid, get_dirs

        return bool(check_winner_grid(state.grid, state.size, row, col, 5, get_dirs()))

    stone = state.at(row, col)
    if stone is Stone.EMPTY:
        return False
    size = state.size
    for dr, dc in _DIRECTIONS:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < size and 0 <= c < size and state.at(r, c) is stone:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < size and 0 <= c < size and state.at(r, c) is stone:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            return True
    return False


# ---------------------------------------------------------------------------
# VCF — Victory by Continuous Fours
# ---------------------------------------------------------------------------


def _vcf_state(
    state: SearchState,
    stone: Stone,
    depth: int,
    is_attacker: bool,
    cache: _ThreatCache,
) -> Move | None:
    """Recursive VCF worker operating on a mutable :class:`SearchState`.

    ``state`` is mutated via make/unmake during the search; on return the
    undo stack is always restored to the ply count at entry.  The public
    :func:`_vcf` wrapper builds a ``SearchState`` once and delegates here.
    """
    if depth <= 0:
        return None

    zh = state.zobrist
    entry = cache.get(zh)
    if entry is not None:
        proven, stored_depth, stored_move = entry
        if proven and stored_depth <= depth:
            return stored_move
        if not proven and stored_depth >= depth:
            return None

    if is_attacker:
        threats = _find_four_threats(state, stone)
        opp = stone.opponent()
        for attack, defence in threats:
            state.make(attack.row, attack.col, stone)
            try:
                if _state_is_win(state, attack.row, attack.col):
                    cache[zh] = (True, depth, attack)
                    return attack

                # Opponent gets to respond — can they win first?
                if _find_winning_move(state, opp) is not None:
                    continue

                if defence is None:
                    cache[zh] = (True, depth, attack)
                    return attack

                # Defender must not have a live-four refutation elsewhere.
                opp_threats = _find_four_threats(state, opp)
                if any(d is None for _, d in opp_threats):
                    continue

                state.make(defence.row, defence.col, opp)
                try:
                    if _vcf_state(state, stone, depth - 2, is_attacker=True, cache=cache) is not None:
                        cache[zh] = (True, depth, attack)
                        return attack
                finally:
                    state.unmake()
            finally:
                state.unmake()

    cache[zh] = (False, depth, None)
    return None


def _vcf(
    board: Board,
    stone: Stone,
    depth: int,
    is_attacker: bool,
    cache: _ThreatCache | None = None,
) -> Move | None:
    """Recursive VCF search.

    *Attacker* can only play moves that create four-threats.
    *Defender* must respond to each four-threat with the forced block.
    Returns the first attacking move of a winning sequence, or ``None``.

    ``cache`` is a transposition table scoped to a single top-level call —
    transposed positions (reached via different attack orderings) are
    resolved once and reused.  Accepts a Pydantic :class:`Board` at the
    entry point; recursion uses a mutable :class:`SearchState` to avoid
    per-node allocations.
    """
    if depth <= 0:
        return None
    if cache is None:
        cache = {}
    state = SearchState.from_board(board, stone)
    return _vcf_state(state, stone, depth, is_attacker, cache)


# ---------------------------------------------------------------------------
# VCT — Victory by Continuous Threats (VCF + live-threes)
# ---------------------------------------------------------------------------


def classify_vcf_breakers(
    board: Board,
    candidates: list[Move],
    stone: Stone,
    vcf_depth: int,
) -> tuple[list[Move], list[Move]]:
    """Classify VCF breakers as (durable, all).

    A *breaker* is any candidate move that makes the opponent's VCF
    disappear when we play it.  A breaker is *durable* when the opponent
    cannot restore the VCF via their forced reply to any counter four-
    threat we happen to create.

    Specifically, after we play ``move``:

    1. If we have no four-threat ourselves, the VCF is genuinely broken
       and ``move`` is durable.
    2. Otherwise we enumerate our four-threats.  For each with a forced
       blocking cell, we simulate the forced block and re-check the
       opponent's VCF.  If any forced block restores the VCF, ``move``
       is only a *temporary* breaker (the counter-threat was just a
       distraction).  Unblockable fours (``defence is None``) are wins
       for us and count as durable.

    Shared between :class:`ThreatBrain._vcf_aware_search` and
    :meth:`AlphaBetaBrain._root_vcf` so both enforce the same semantics.
    """
    state = SearchState.from_board(board, stone)
    opp = stone.opponent()
    breakers: list[Move] = []
    durable: list[Move] = []

    for move in candidates:
        state.make(move.row, move.col, stone)
        try:
            if _vcf_state(state, opp, vcf_depth, is_attacker=True, cache={}) is not None:
                continue
            breakers.append(move)

            our_threats = _find_four_threats(state, stone)
            if not our_threats:
                durable.append(move)
                continue

            is_durable = True
            for _attack, defence in our_threats:
                if defence is None:
                    # Unblockable four — we win, so the VCF concern is moot.
                    continue
                state.make(defence.row, defence.col, opp)
                try:
                    if _vcf_state(state, opp, vcf_depth, is_attacker=True, cache={}) is not None:
                        is_durable = False
                        break
                finally:
                    state.unmake()
            if is_durable:
                durable.append(move)
        finally:
            state.unmake()

    return durable, breakers


def _vct_defence_moves_state(state: SearchState, attack_move: Move, stone: Stone) -> list[Move]:
    """Compute the full defence set for a three-threat played by *stone*.

    Mirrors :func:`_vct_defence_moves` but operates on a mutable
    :class:`SearchState` — ``state`` is expected to be the pre-attack
    position and is returned to the same ply on exit.

    On the JIT path the per-candidate counter-four probe uses
    ``has_any_four_threat_grid`` (short-circuits on first hit) over
    a candidate list regenerated after each defender placement, replacing
    the previous full ``_find_four_threats`` enumeration.
    """
    from gomoku_brain.evaluate import _should_use_jit

    opp = stone.opponent()

    # Natural end-cell defences of the live three
    three_threats = _find_three_threats(state, stone)
    end_cells: set[tuple[int, int]] = set()
    for atk, defs in three_threats:
        if atk == attack_move:
            for d in defs:
                end_cells.add((d.row, d.col))

    state.make(attack_move.row, attack_move.col, stone)
    try:
        if _should_use_jit():
            from gomoku_brain.evaluate_jit import (
                get_candidates_grid,
                get_dirs,
                has_any_four_threat_grid,
            )
            from gomoku_brain.grid_utils import occupied_coords

            dirs = get_dirs()
            opp_int = int(opp)
            size = state.size
            grid = state.grid  # mutable mirror updated by make/unmake

            for candidate in get_candidate_moves(state, radius=2):
                if (candidate.row, candidate.col) in end_cells:
                    continue
                state.make(candidate.row, candidate.col, opp)
                try:
                    occ = occupied_coords(state)
                    cands = get_candidates_grid(grid, size, occ, 2)
                    if has_any_four_threat_grid(grid, size, cands, opp_int, dirs):
                        end_cells.add((candidate.row, candidate.col))
                finally:
                    state.unmake()
        else:
            for candidate in get_candidate_moves(state, radius=2):
                if (candidate.row, candidate.col) in end_cells:
                    continue
                state.make(candidate.row, candidate.col, opp)
                try:
                    if _find_four_threats(state, opp):
                        end_cells.add((candidate.row, candidate.col))
                finally:
                    state.unmake()
    finally:
        state.unmake()

    return [Move(row=r, col=c) for r, c in sorted(end_cells)]


def _vct_defence_moves(board: Board, attack_move: Move, stone: Stone) -> list[Move]:
    """Board-accepting wrapper over :func:`_vct_defence_moves_state`."""
    state = SearchState.from_board(board, stone)
    return _vct_defence_moves_state(state, attack_move, stone)


def _vct_state(
    state: SearchState,
    stone: Stone,
    depth: int,
    is_attacker: bool,
    vct_cache: _ThreatCache,
    vcf_cache: _ThreatCache,
) -> Move | None:
    """Recursive VCT worker operating on a mutable :class:`SearchState`."""
    if depth <= 0:
        return None

    zh = state.zobrist
    entry = vct_cache.get(zh)
    if entry is not None:
        proven, stored_depth, stored_move = entry
        if proven and stored_depth <= depth:
            return stored_move
        if not proven and stored_depth >= depth:
            return None

    if is_attacker:
        # 1) Try VCF first (four-threats are stronger)
        vcf_result = _vcf_state(state, stone, depth, is_attacker=True, cache=vcf_cache)
        if vcf_result is not None:
            vct_cache[zh] = (True, depth, vcf_result)
            return vcf_result

        # 2) Try three-threats
        three_threats = _find_three_threats(state, stone)
        opp = stone.opponent()
        for attack, _naive_defences in three_threats:
            # Defence set is computed on the *pre-attack* state.
            defence_moves = _vct_defence_moves_state(state, attack, stone)

            state.make(attack.row, attack.col, stone)
            try:
                if _state_is_win(state, attack.row, attack.col):
                    vct_cache[zh] = (True, depth, attack)
                    return attack

                if _find_winning_move(state, opp) is not None:
                    continue

                all_refuted = True
                if not defence_moves:
                    all_refuted = False
                else:
                    for defence in defence_moves:
                        if not state.is_empty(defence.row, defence.col):
                            continue
                        state.make(defence.row, defence.col, opp)
                        try:
                            opponent_fours = _find_four_threats(state, opp)
                            if opponent_fours:
                                if any(d is None for _, d in opponent_fours):
                                    all_refuted = False
                                    break
                                unique_blocks = {(m.row, m.col) for _, m in opponent_fours if m is not None}
                                if len(unique_blocks) >= 2:
                                    all_refuted = False
                                    break
                                _, block_move = opponent_fours[0]
                                if block_move is None or not state.is_empty(block_move.row, block_move.col):
                                    all_refuted = False
                                    break
                                state.make(block_move.row, block_move.col, stone)
                                try:
                                    if (
                                        _vct_state(
                                            state,
                                            stone,
                                            depth - 2,
                                            is_attacker=True,
                                            vct_cache=vct_cache,
                                            vcf_cache=vcf_cache,
                                        )
                                        is None
                                    ):
                                        all_refuted = False
                                        break
                                finally:
                                    state.unmake()
                            elif (
                                _vct_state(
                                    state,
                                    stone,
                                    depth - 2,
                                    is_attacker=True,
                                    vct_cache=vct_cache,
                                    vcf_cache=vcf_cache,
                                )
                                is None
                            ):
                                all_refuted = False
                                break
                        finally:
                            state.unmake()

                if all_refuted:
                    vct_cache[zh] = (True, depth, attack)
                    return attack
            finally:
                state.unmake()

    vct_cache[zh] = (False, depth, None)
    return None


def _vct(
    board: Board,
    stone: Stone,
    depth: int,
    is_attacker: bool,
    vct_cache: _ThreatCache | None = None,
    vcf_cache: _ThreatCache | None = None,
) -> Move | None:
    """Recursive VCT search.

    Attacker plays four-threats *and* live-three threats.
    Defender tries all reasonable defence moves: the two open-end cells of the
    threat **and** any move that creates a counter four-threat.

    ``vct_cache`` memoises VCT results across transpositions in the current
    top-level call.  ``vcf_cache`` is a separate table shared with the inner
    ``_vcf`` probes — VCT and VCF prove different properties, so their
    entries cannot be mixed, but within a single VCT search the same VCF
    sub-positions are re-visited repeatedly and benefit from caching.
    """
    if depth <= 0:
        return None
    if vct_cache is None:
        vct_cache = {}
    if vcf_cache is None:
        vcf_cache = {}
    state = SearchState.from_board(board, stone)
    return _vct_state(state, stone, depth, is_attacker, vct_cache, vcf_cache)
