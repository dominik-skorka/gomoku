"""Threat-based brain — VCF (Victory by Continuous Fours) and VCT (Victory by Continuous Threats).

VCF searches only four-threats (forcing single-cell responses), enabling very
deep searches.  VCT extends this to also consider live-three threats, where the
defender has more responses but a forced win can still be proven.

When neither finds a forced win the brain falls back to alpha-beta search.

The pure threat-detection helpers (``_find_four_threats``,
``_find_three_threats``, ``_find_winning_move``, ``probe_immediate``,
``_line_cells``) live in :mod:`gomoku_brain.threat_detect` and are
re-exported below for backward compatibility.
"""

from __future__ import annotations

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.brain_base import brain_session
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.directions import DIRECTIONS as _DIRECTIONS
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import (  # noqa: F401  (re-exports for back-compat)
    _find_four_threats,
    _find_four_threats_python,
    _find_three_threats,
    _find_three_threats_python,
    _find_winning_move,
    _line_cells,
    probe_immediate,
)

# Cache entry: (proven, depth_searched, attack_move)
# * proven=True  -> winning attack found within ``depth_searched`` plies.
#                   Valid whenever the query depth is >= depth_searched.
# * proven=False -> no winning sequence within ``depth_searched`` plies.
#                   Valid whenever the query depth is <= depth_searched.
_CacheEntry = tuple[bool, int, "Move | None"]
_ThreatCache = dict[int, _CacheEntry]

# Re-exports for back-compat: callers still do
#     from gomoku_brain.threat import _find_four_threats
__all_detect__ = (
    "_find_four_threats",
    "_find_four_threats_python",
    "_find_three_threats",
    "_find_three_threats_python",
    "_find_winning_move",
    "_line_cells",
    "probe_immediate",
)


# ---------------------------------------------------------------------------
# VCF search
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
# VCT search
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
    """
    opp = stone.opponent()

    # Natural end-cell defences of the live three
    three_threats = _find_three_threats(state, stone)
    end_cells: set[tuple[int, int]] = set()
    for atk, defs in three_threats:
        if atk == attack_move:
            for d in defs:
                end_cells.add((d.row, d.col))

    # Place the attack, then scan for counter four-threats
    state.make(attack_move.row, attack_move.col, stone)
    try:
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


# ---------------------------------------------------------------------------
# ThreatBrain
# ---------------------------------------------------------------------------


class ThreatBrain:
    """MoveProvider using VCF / VCT or PNS with alpha-beta fallback.

    *threat_search* selects the algorithm used after VCF:
    - ``"pns"`` (default): Proof-Number Search — sound, covers all defender moves.
    - ``"vct"`` : depth-limited VCT — faster but weaker (expanded defence set).
    """

    def __init__(
        self,
        depth: int = 4,
        vcf_depth: int = 20,
        vct_depth: int = 12,
        threat_search: str = "pns",
        pns_nodes: int = 10_000,
        pns_time_s: float = 5.0,
    ) -> None:
        self.depth = depth
        self.vcf_depth = vcf_depth
        self.vct_depth = vct_depth
        self.threat_search = threat_search
        self.pns_nodes = pns_nodes
        self.pns_time_s = pns_time_s
        self._fallback = AlphaBetaBrain(depth=depth)

    @property
    def name(self) -> str:
        return "threat"

    def get_move(self, board: Board, stone: Stone) -> Move:
        with brain_session(
            "threat",
            vcf_depth=self.vcf_depth,
            vct_depth=self.vct_depth,
            threat_search=self.threat_search,
            pns_nodes=self.pns_nodes,
            stone=str(stone),
        ) as session:
            from gomoku.diagnostics import ThreatInfo

            diag = session.diag

            # Collect threat landscape for diagnostics.
            if diag is not None:
                opp = stone.opponent()
                our_fours = _find_four_threats(board, stone)
                our_threes = _find_three_threats(board, stone)
                opp_fours = _find_four_threats(board, opp)
                opp_threes = _find_three_threats(board, opp)
                diag.our_threats = ThreatInfo(
                    four_threats=len(our_fours),
                    three_threats=len(our_threes),
                )
                diag.opp_threats = ThreatInfo(
                    four_threats=len(opp_fours),
                    three_threats=len(opp_threes),
                )

            # 0) If we can win immediately, do it
            win = _find_winning_move(board, stone)
            if win is not None:
                logfire.info("Immediate win: {move}", move=str(win))
                if diag is not None:
                    diag.decision_path = "immediate_win"
                    diag.chosen_move = win
                return win

            # 1) If opponent can win next turn, we must block
            opp = stone.opponent()
            opp_win = _find_winning_move(board, opp)
            if opp_win is not None:
                logfire.info("Blocking opponent immediate win at: {move}", move=str(opp_win))
                if diag is not None:
                    diag.decision_path = "block_immediate_win"
                    diag.chosen_move = opp_win
                return opp_win

            # 2) Try VCF — very fast, deep search
            move = _vcf(board, stone, self.vcf_depth, is_attacker=True)
            if move is not None:
                logfire.info("VCF found winning move: {move}", move=str(move))
                if diag is not None:
                    diag.decision_path = "vcf_win"
                    diag.chosen_move = move
                    diag.our_threats = diag.our_threats.model_copy(update={"vcf_result": "win"})
                return move

            # 3) Try broader threat search (PNS or VCT)
            if self.threat_search == "pns":
                from gomoku_brain.pns import pns_threat_search

                move = pns_threat_search(board, stone, max_nodes=self.pns_nodes, max_time_s=self.pns_time_s)
                if move is not None:
                    logfire.info("PNS found winning move: {move}", move=str(move))
                    if diag is not None:
                        diag.decision_path = "pns_win"
                        diag.chosen_move = move
                    return move
            else:
                move = _vct(board, stone, self.vct_depth, is_attacker=True)
                if move is not None:
                    logfire.info("VCT found winning move: {move}", move=str(move))
                    if diag is not None:
                        diag.decision_path = "vct_win"
                        diag.chosen_move = move
                        diag.our_threats = diag.our_threats.model_copy(update={"vct_result": "win"})
                    return move

            if diag is not None:
                diag.notes.append("No forced win found (VCF/PNS/VCT)")

            # 4) Check if opponent has a VCF threat
            opp_vcf = _vcf(board, opp, self.vcf_depth, is_attacker=True)
            if opp_vcf is not None:
                if diag is not None:
                    diag.opp_threats = diag.opp_threats.model_copy(update={"vcf_result": "win"})
                    diag.notes.append(f"Opponent has VCF at {opp_vcf}")
                result = self._vcf_aware_search(board, stone, opp_vcf)
                if diag is not None:
                    diag.decision_path = diag.decision_path or "vcf_aware_search"
                    diag.chosen_move = result
                return result

            # 5) Check if opponent has a VCT / PNS threat
            opp_vct_move = self._check_opponent_vct(board, stone)
            if opp_vct_move is not None:
                if diag is not None:
                    diag.decision_path = "vct_aware_search"
                    diag.chosen_move = opp_vct_move
                return opp_vct_move

            # 6) Check for opponent fork points (converging threat lines)
            fork_move = self._block_fork(board, stone)
            if fork_move is not None:
                if diag is not None:
                    diag.decision_path = "block_fork"
                    diag.chosen_move = fork_move
                return fork_move

            # 7) Fallback to alpha-beta
            logfire.info("No threat sequence found, falling back to alpha-beta")
            if diag is not None:
                diag.notes.append("Falling back to alpha-beta")
                # Don't set decision_path here — alpha-beta will set its own
            return self._fallback.get_move(board, stone)

    # -- Opponent VCT detection ---------------------------------------------

    def _check_opponent_vct(
        self,
        board: Board,
        stone: Stone,
    ) -> Move | None:
        """Check if the opponent has a forced VCT win and try to break it.

        Runs the same threat search (PNS or VCT) from the opponent's
        perspective.  If a forced win is found, restrict candidates to
        moves that break it and pick the best positionally.

        Gated by a cheap material heuristic: PNS/VCT only has a realistic
        chance of proving a combination win when the opponent already has
        enough tactical material on the board (≥ 1 dead-four, since any
        single forcing sequence starts there, or ≥ 2 live-threes, since
        a double-three is a known fork seed).  When neither is present we
        skip the full search — this is the dominant case in quiet
        positions and each PNS call costs up to ``pns_time_s`` seconds.

        The earlier ``_vcf`` check in :meth:`get_move` has already ruled
        out unblockable fours, so any four-threat seen here is a
        dead-four.
        """
        opp = stone.opponent()

        opp_fours = _find_four_threats(board, opp)
        opp_threes = _find_three_threats(board, opp)
        # Distinct attack cells rather than raw entry count: the JIT path
        # emits one three-threat entry per direction, so a double-three
        # shows up as two entries at the same cell.
        distinct_three_cells = {(a.row, a.col) for a, _ in opp_threes}
        if not opp_fours and len(distinct_three_cells) < 2:
            return None

        if self.threat_search == "pns":
            from gomoku_brain.pns import pns_threat_search

            opp_threat = pns_threat_search(
                board,
                opp,
                max_nodes=self.pns_nodes,
                max_time_s=self.pns_time_s,
            )
        else:
            opp_threat = _vct(board, opp, self.vct_depth, is_attacker=True)

        if opp_threat is None:
            return None

        logfire.info(
            "Opponent has VCT/PNS threat starting at: {move}",
            move=str(opp_threat),
        )
        return self._vct_aware_search(board, stone, opp_threat)

    def _vct_aware_search(
        self,
        board: Board,
        stone: Stone,
        opp_threat_move: Move,
    ) -> Move:
        """Pick the best move when the opponent has a VCT/PNS forced win.

        Order candidates by static evaluation (descending), then probe in
        order until we find the first move that breaks the opponent's
        threat sequence.  Because the candidate list is eval-sorted, the
        first breaker is also the positionally strongest, so a single
        pass replaces the old "collect-all-then-score" loop.

        A global wall-clock cap (``self.pns_time_s``) prevents the
        worst-case blow-up that used to be possible when every one of
        the 40-50 candidates was probed with a half-budget PNS call.
        Per-candidate PNS budgets are additionally shrunk to the
        remaining global budget.
        """
        import time as _time

        from gomoku_brain.evaluate import evaluate_board

        opp = stone.opponent()
        state = SearchState.from_board(board, stone)
        candidates = get_candidate_moves(state, radius=2)

        # Pre-score every candidate with a static eval so strong moves
        # are probed first.  Static eval is cheap on the JIT path
        # (<~50us per call).
        scored: list[tuple[float, Move]] = []
        for move in candidates:
            state.make(move.row, move.col, stone)
            try:
                score = evaluate_board(state, stone)
            finally:
                state.unmake()
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)

        deadline = _time.monotonic() + self.pns_time_s
        probed = 0
        for score, move in scored:
            if _time.monotonic() > deadline:
                logfire.info(
                    "VCT-aware search: time budget exhausted after {n} candidates",
                    n=probed,
                )
                break
            probed += 1

            state.make(move.row, move.col, stone)
            try:
                # Fast VCF probe first.
                if _vcf_state(state, opp, self.vcf_depth, is_attacker=True, cache={}) is not None:
                    continue
                # Expensive threat probe — bounded by remaining global budget.
                if self.threat_search == "pns":
                    from gomoku_brain.grid_utils import grid_to_board
                    from gomoku_brain.pns import pns_threat_search

                    remaining = max(0.05, deadline - _time.monotonic())
                    trial_board = grid_to_board(state.grid, state.size)
                    opp_result = pns_threat_search(
                        trial_board,
                        opp,
                        max_nodes=self.pns_nodes // 2,
                        max_time_s=min(self.pns_time_s / 2, remaining),
                    )
                    if opp_result is not None:
                        continue
                else:
                    if (
                        _vct_state(
                            state,
                            opp,
                            self.vct_depth,
                            is_attacker=True,
                            vct_cache={},
                            vcf_cache={},
                        )
                        is not None
                    ):
                        continue
                logfire.info(
                    "VCT-aware search: first durable breaker {move} (eval={score:.0f}) after {n} probes",
                    move=str(move),
                    score=score,
                    n=probed,
                )
                return move
            finally:
                state.unmake()

        logfire.info(
            "No VCT breaker found, blocking opponent threat at: {move}",
            move=str(opp_threat_move),
        )
        if board.is_valid_move(opp_threat_move):
            return opp_threat_move
        return self._fallback.get_move(board, stone)

    # -- Fork detection -----------------------------------------------------

    def _block_fork(
        self,
        board: Board,
        stone: Stone,
    ) -> Move | None:
        """Block opponent fork points — cells where >= 2 threat lines converge.

        If the opponent has a fork cell, blocking it is urgent.  Among
        multiple fork cells, pick the one that scores best for us.
        """
        from gomoku_brain.evaluate import evaluate_board, find_fork_cells

        opp = stone.opponent()
        forks = find_fork_cells(board, opp)

        if not forks:
            return None

        logfire.info(
            "Opponent fork cells detected: {n} cells",
            n=len(forks),
        )

        if len(forks) == 1:
            return forks[0]

        # Multiple fork cells — block the most dangerous one
        # (the one giving opponent the highest eval after placing there)
        state = SearchState.from_board(board, stone)
        best_score = float("-inf")
        best_move = forks[0]
        for move in forks:
            state.make(move.row, move.col, stone)
            try:
                score = evaluate_board(state, stone)
            finally:
                state.unmake()
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # -- VCF-aware search ---------------------------------------------------

    def _vcf_aware_search(
        self,
        board: Board,
        stone: Stone,
        opp_vcf_move: Move,
    ) -> Move:
        """Pick the best move when the opponent has a VCF.

        Classify candidates into durable breakers (VCF stays broken
        after any forced reply) and merely-temporary breakers (opp
        restores the VCF via the forced block of one of our counter-
        threats).  Prefer durable breakers when any exist; fall back
        to all breakers otherwise — matching the semantics of
        :meth:`AlphaBetaBrain._root_vcf`.  Among the chosen set, pick
        the positionally best move.
        """
        from gomoku_brain.evaluate import evaluate_board

        candidates = get_candidate_moves(board, radius=2)
        durable, breakers = classify_vcf_breakers(board, candidates, stone, self.vcf_depth)

        if not breakers:
            logfire.info(
                "No VCF breaker found, blocking opponent VCF at: {move}",
                move=str(opp_vcf_move),
            )
            if board.is_valid_move(opp_vcf_move):
                return opp_vcf_move
            return self._fallback.get_move(board, stone)

        effective = durable if durable else breakers
        logfire.info(
            "VCF breakers: {total} total, {durable} durable, using {n}",
            total=len(breakers),
            durable=len(durable),
            n=len(effective),
        )

        if len(effective) == 1:
            return effective[0]

        state = SearchState.from_board(board, stone)
        best_score = float("-inf")
        best_move = effective[0]
        for move in effective:
            state.make(move.row, move.col, stone)
            try:
                score = evaluate_board(state, stone)
            finally:
                state.unmake()
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
