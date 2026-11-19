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
from gomoku_brain.threat_detect import (
    _find_four_threats,
    _find_three_threats,
    _find_winning_move,
    _has_unblockable_four,
    _scan_context,
    _ScanCtx,
)
from gomoku_brain.threat_space import DEFAULT_THREAT_SPACE, ThreatSpace

# Cache entry: (proven, depth_searched, attack_move)
# * proven=True  -> winning attack found within ``depth_searched`` plies.
#                   Valid whenever the query depth is >= depth_searched.
# * proven=False -> no winning sequence within ``depth_searched`` plies.
#                   Valid whenever the query depth is <= depth_searched.
_CacheEntry = tuple[bool, int, "Move | None"]
# Keyed by (zobrist, attacker): the zobrist hash encodes the side to move,
# not the attacker, and one cache is shared by searches for both sides
# within a decision (e.g. the opponent's VCF probed with us to move).
_ThreatCache = dict[tuple[int, int], _CacheEntry]


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
    cache: _ThreatCache,
    space: ThreatSpace = DEFAULT_THREAT_SPACE,
) -> Move | None:
    """Recursive VCF worker operating on a mutable :class:`SearchState`.

    *space* chooses the attacker's fours (``fours``, ``split_fours``) and
    the defender's replies (``block_fours``, ``counter_fours``,
    ``split_counter_fours``); *cache* must only ever see one *space*.

    ``state`` is mutated via make/unmake during the search; on return the
    undo stack is always restored to the ply count at entry.  The public
    :func:`_vcf` wrapper builds a ``SearchState`` once and delegates here.
    """
    if depth <= 0 or not space.fours:
        return None

    zh = (state.zobrist, int(stone))
    entry = cache.get(zh)
    if entry is not None:
        proven, stored_depth, stored_move = entry
        if proven and stored_depth <= depth:
            return stored_move
        if not proven and stored_depth >= depth:
            return None

    # One scan context per node, shared by the attacker enumeration and
    # the per-attack opponent probes below.  The probes run after *our*
    # stone is placed, but they detect *opponent* threats, so the
    # staleness contract (see :data:`_ScanCtx`) holds.
    ctx = _scan_context(state)
    threats = _find_four_threats(state, stone, ctx, space.split_fours)
    opp = stone.opponent()
    for attack, defence in threats:
        state.make(attack.row, attack.col, stone)
        try:
            if _state_is_win(state, attack.row, attack.col):
                cache[zh] = (True, depth, attack)
                return attack

            # Opponent gets to respond — can they win first?
            if _find_winning_move(state, opp, ctx) is not None:
                continue

            # Unblockable, or the defender may not block (block_fours off).
            if defence is None or not space.block_fours:
                cache[zh] = (True, depth, attack)
                return attack

            # Defender must not have a live-four refutation elsewhere.
            if space.counter_fours and _has_unblockable_four(state, opp, ctx, space.split_counter_fours):
                continue

            state.make(defence.row, defence.col, opp)
            try:
                if _vcf_state(state, stone, depth - 2, cache=cache, space=space) is not None:
                    cache[zh] = (True, depth, attack)
                    return attack
            finally:
                state.unmake()
        finally:
            state.unmake()

    cache[zh] = (False, depth, None)
    return None


def _vcf(
    board: Board | SearchState,
    stone: Stone,
    depth: int,
    cache: _ThreatCache | None = None,
    space: ThreatSpace = DEFAULT_THREAT_SPACE,
) -> Move | None:
    """Recursive VCF search.

    The attacker can only play moves that create four-threats; the
    defender must respond to each four-threat with the forced block.
    Returns the first attacking move of a winning sequence, or ``None``.

    ``cache`` is a transposition table scoped to a single top-level call —
    transposed positions (reached via different attack orderings) are
    resolved once and reused.  Accepts a Pydantic :class:`Board` or an
    existing :class:`SearchState` at the entry point; recursion uses the
    mutable state to avoid per-node allocations.
    """
    if depth <= 0:
        return None
    if cache is None:
        cache = {}
    state = board if isinstance(board, SearchState) else SearchState.from_board(board, stone)
    return _vcf_state(state, stone, depth, cache, space)


# ---------------------------------------------------------------------------
# VCT — Victory by Continuous Threats (VCF + live-threes)
# ---------------------------------------------------------------------------


def classify_vcf_breakers(
    board: Board | SearchState,
    candidates: list[Move],
    stone: Stone,
    vcf_depth: int,
    cache: _ThreatCache | None = None,
    space: ThreatSpace = DEFAULT_THREAT_SPACE,
) -> tuple[list[Move], list[Move]]:
    """Classify VCF breakers as (durable, all).

    *space* is the opponent's VCF space; our counter-fours follow its
    defence fields (``counter_fours``, ``split_counter_fours``).

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

    ``cache`` is an optional shared VCF transposition table — when
    multiple VCF calls within one decision visit overlapping
    sub-positions (e.g. our breaker probes after attacker fours), reusing
    proofs across them avoids redundant deep search.

    Shared between :class:`ThreatBrain._vcf_aware_search` and
    :meth:`AlphaBetaBrain._root_vcf` so both enforce the same semantics.
    """
    state = board if isinstance(board, SearchState) else SearchState.from_board(board, stone)
    opp = stone.opponent()
    if cache is None:
        cache = {}
    breakers: list[Move] = []
    durable: list[Move] = []

    for move in candidates:
        state.make(move.row, move.col, stone)
        try:
            if _vcf_state(state, opp, vcf_depth, cache=cache, space=space) is not None:
                continue
            breakers.append(move)

            our_threats = (
                _find_four_threats(state, stone, split=space.split_counter_fours) if space.counter_fours else []
            )
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
                    if _vcf_state(state, opp, vcf_depth, cache=cache, space=space) is not None:
                        is_durable = False
                        break
                finally:
                    state.unmake()
            if is_durable:
                durable.append(move)
        finally:
            state.unmake()

    return durable, breakers


def _vct_defence_moves_state(
    state: SearchState,
    attack_move: Move,
    stone: Stone,
    ctx: _ScanCtx | None = None,
    space: ThreatSpace = DEFAULT_THREAT_SPACE,
) -> list[Move]:
    """Compute the full defence set for a three-threat played by *stone*.

    The three's own defences with ``space.block_threes``; cells where the
    defender makes a four with ``space.counter_fours``.

    Mirrors :func:`_vct_defence_moves` but operates on a mutable
    :class:`SearchState` — ``state`` is expected to be the pre-attack
    position and is returned to the same ply on exit.  ``ctx`` is an
    optional scan context for that pre-attack position.

    On the JIT path the per-candidate counter-four probe uses
    ``has_any_four_threat_grid`` (short-circuits on first hit) over
    a candidate list regenerated after each defender placement, replacing
    the previous full ``_find_four_threats`` enumeration.
    """
    from gomoku_brain.evaluate import _should_use_jit

    opp = stone.opponent()

    # Natural defences of the three (its ends; for a split three also the gap)
    end_cells: set[tuple[int, int]] = set()
    if space.block_threes:
        for atk, defs in _find_three_threats(state, stone, ctx, space.split_threes):
            if atk == attack_move:
                for d in defs:
                    end_cells.add((d.row, d.col))
    if not space.counter_fours:
        return [Move(row=r, col=c) for r, c in sorted(end_cells)]

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
                    if has_any_four_threat_grid(grid, size, cands, opp_int, dirs, space.split_counter_fours):
                        end_cells.add((candidate.row, candidate.col))
                finally:
                    state.unmake()
        else:
            for candidate in get_candidate_moves(state, radius=2):
                if (candidate.row, candidate.col) in end_cells:
                    continue
                state.make(candidate.row, candidate.col, opp)
                try:
                    if _find_four_threats(state, opp, split=space.split_counter_fours):
                        end_cells.add((candidate.row, candidate.col))
                finally:
                    state.unmake()
    finally:
        state.unmake()

    return [Move(row=r, col=c) for r, c in sorted(end_cells)]


def _vct_defence_moves(board: Board, attack_move: Move, stone: Stone) -> list[Move]:
    """Board-accepting wrapper over :func:`_vct_defence_moves_state`.

    Production callers (``_vct_state``) operate on a ``SearchState`` and
    invoke ``_vct_defence_moves_state`` directly; this wrapper exists for
    tests that pass a Pydantic ``Board``.
    """
    state = SearchState.from_board(board, stone)
    return _vct_defence_moves_state(state, attack_move, stone)


def _defence_is_refuted(
    state: SearchState,
    stone: Stone,
    depth: int,
    vct_cache: _ThreatCache,
    vcf_cache: _ThreatCache,
    space: ThreatSpace = DEFAULT_THREAT_SPACE,
) -> bool:
    """Return True if the defender's reply (just played) still loses to VCT.

    ``state`` is positioned right after a defence move.  If that move
    created counter four-threats for the defender, the attacker must
    neutralise them first: an unblockable counter-four, or two distinct
    forced block cells, refute the attack outright; a single forced
    block is played and the search continues below it.
    """
    opp = stone.opponent()
    opponent_fours = _find_four_threats(state, opp, split=space.split_counter_fours) if space.counter_fours else []
    if not opponent_fours:
        return _vct_state(state, stone, depth - 2, vct_cache, vcf_cache, space) is not None
    if any(d is None for _, d in opponent_fours):
        return False
    unique_blocks = {(m.row, m.col) for _, m in opponent_fours if m is not None}
    if len(unique_blocks) >= 2:
        return False
    _, block_move = opponent_fours[0]
    if block_move is None or not state.is_empty(block_move.row, block_move.col):
        return False
    state.make(block_move.row, block_move.col, stone)
    try:
        return _vct_state(state, stone, depth - 2, vct_cache, vcf_cache, space) is not None
    finally:
        state.unmake()


def _vct_state(
    state: SearchState,
    stone: Stone,
    depth: int,
    vct_cache: _ThreatCache,
    vcf_cache: _ThreatCache,
    space: ThreatSpace = DEFAULT_THREAT_SPACE,
) -> Move | None:
    """Recursive VCT worker operating on a mutable :class:`SearchState`.

    *space* chooses the threats and replies (see :class:`ThreatSpace`);
    ``vct_cache`` and ``vcf_cache`` must only ever see one *space*.
    """
    if depth <= 0:
        return None

    zh = (state.zobrist, int(stone))
    entry = vct_cache.get(zh)
    if entry is not None:
        proven, stored_depth, stored_move = entry
        if proven and stored_depth <= depth:
            return stored_move
        if not proven and stored_depth >= depth:
            return None

    # 1) Try VCF first (four-threats are stronger)
    vcf_result = _vcf_state(state, stone, depth, cache=vcf_cache, space=space)
    if vcf_result is not None:
        vct_cache[zh] = (True, depth, vcf_result)
        return vcf_result

    # 2) Try three-threats.  One scan context for this node, shared by the
    # attacker enumeration, per-attack defence sets (computed pre-attack),
    # and the post-attack opponent win probes (opponent hasn't moved).
    ctx = _scan_context(state)
    three_threats = _find_three_threats(state, stone, ctx, space.split_threes) if space.threes else []
    opp = stone.opponent()
    for attack, _naive_defences in three_threats:
        # Defence set is computed on the *pre-attack* state.
        defence_moves = _vct_defence_moves_state(state, attack, stone, ctx, space)

        state.make(attack.row, attack.col, stone)
        try:
            if _state_is_win(state, attack.row, attack.col):
                vct_cache[zh] = (True, depth, attack)
                return attack

            if _find_winning_move(state, opp, ctx) is not None:
                continue

            # No reply at all proves the attack only when the space takes
            # the three's own defences away (block_threes off).
            all_refuted = bool(defence_moves) or not space.block_threes
            for defence in defence_moves:
                if not state.is_empty(defence.row, defence.col):
                    continue
                state.make(defence.row, defence.col, opp)
                try:
                    if not _defence_is_refuted(state, stone, depth, vct_cache, vcf_cache, space):
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
    board: Board | SearchState,
    stone: Stone,
    depth: int,
    vct_cache: _ThreatCache | None = None,
    vcf_cache: _ThreatCache | None = None,
    space: ThreatSpace = DEFAULT_THREAT_SPACE,
) -> Move | None:
    """Recursive VCT search.

    The attacker plays four-threats *and* live-three threats.  The
    defender tries all reasonable defence moves: the two open-end cells
    of the threat **and** any move that creates a counter four-threat.

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
    state = board if isinstance(board, SearchState) else SearchState.from_board(board, stone)
    return _vct_state(state, stone, depth, vct_cache, vcf_cache, space)
