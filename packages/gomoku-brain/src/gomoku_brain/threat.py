"""Threat-based brain — VCF (Victory by Continuous Fours) and VCT (Victory by Continuous Threats).

VCF searches only four-threats (forcing single-cell responses), enabling very
deep searches.  VCT extends this to also consider live-three threats, where the
defender has more responses but a forced win can still be proven.

When neither finds a forced win the brain falls back to alpha-beta search.

The detection primitives live in :mod:`gomoku_brain.threat_detect`; the
recursive search workers live in :mod:`gomoku_brain.threat_search`.  Both
sets are re-exported below for backward compatibility — external callers
and tests still import them from this module.
"""

from __future__ import annotations

import time

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.alphabeta_ordering import _scored_moves
from gomoku_brain.brain_base import brain_session
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import find_fork_cells
from gomoku_brain.pns import PNSStatus, pns_threat_search, pns_threat_status
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import (
    _count_distinct_three_cells,
    _find_four_threats,
    _find_four_threats_python,
    _find_three_threats,
    _find_three_threats_python,
    _find_winning_move,
    _has_any_four_threat,
    _line_cells,
    _scan_context,
    probe_immediate,
)
from gomoku_brain.threat_search import (
    _state_is_win,
    _ThreatCache,
    _vcf,
    _vcf_state,
    _vct,
    _vct_defence_moves,
    _vct_defence_moves_state,
    _vct_state,
    classify_vcf_breakers,
)

__all__ = (
    "ThreatBrain",
    "_find_four_threats",
    "_find_four_threats_python",
    "_find_three_threats",
    "_find_three_threats_python",
    "_find_winning_move",
    "_line_cells",
    "_state_is_win",
    "_vcf",
    "_vcf_state",
    "_vct",
    "_vct_defence_moves",
    "_vct_defence_moves_state",
    "_vct_state",
    "classify_vcf_breakers",
    "probe_immediate",
)

# Per-side fraction of the configured PNS budget used for opponent-perspective
# probes inside :meth:`ThreatBrain._vct_aware_search`.  The original threat
# search has already consumed up to one full ``pns_time_s``; allowing each
# breaker probe a full second budget would let total wall time blow past the
# configured cap, so we halve both the node and time budgets for those probes.
_OPP_PNS_NODE_FRACTION = 0.5
_OPP_PNS_TIME_FRACTION = 0.5

# Floor for per-candidate PNS probe time when the global VCT-aware deadline
# is nearly exhausted — keeps progress when the deadline is close.
_MIN_PROBE_TIME_S = 0.05

# Radius used for candidate generation throughout the threat brain.
_CANDIDATE_RADIUS = 2


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
        fallback: dict[str, object] | None = None,
    ) -> None:
        self.depth = depth
        self.vcf_depth = vcf_depth
        self.vct_depth = vct_depth
        self.threat_search = threat_search
        self.pns_nodes = pns_nodes
        self.pns_time_s = pns_time_s
        # ``fallback`` carries AlphaBetaBrain tuning (qs_vcf_depth, LMR,
        # null-move, time limit, ...) for the no-forced-win path —
        # without it the fallback ran with bare defaults while the
        # standalone [brains.alphabeta] alias was fully tuned.  Keys in
        # ``fallback`` win over the top-level ``depth``.  ``vcf_depth``
        # is deliberately not defaulted here: the threat brain has
        # already proven both sides VCF-free before falling back, so a
        # root-VCF precheck in the fallback would be redundant work.
        self._fallback = AlphaBetaBrain(**{"depth": depth, **(fallback or {})})

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
            opp = stone.opponent()

            # One mutable state per perspective for the whole decision —
            # every detection helper and search worker accepts a
            # ``SearchState``, so the Pydantic board is converted exactly
            # twice instead of once per detection call.  Two perspectives
            # because the VCF/VCT caches key on zobrist hashes that encode
            # the side to move.
            state = SearchState.from_board(board, stone)
            opp_state = SearchState.from_board(board, opp)

            # 0/1) Immediate win, else forced block of the opponent's win.
            # Runs before the diagnostics block so the shortcut paths skip
            # the four full pattern scans they don't need.
            probe_move, probe_tag = probe_immediate(state, stone)
            if probe_move is not None:
                if probe_tag == "instant_win":
                    tag = "immediate_win"
                    logfire.info("Immediate win: {move}", move=str(probe_move))
                else:
                    tag = "block_immediate_win"
                    logfire.info("Blocking opponent immediate win at: {move}", move=str(probe_move))
                if diag is not None:
                    diag.decision_path = tag
                    diag.chosen_move = probe_move
                return probe_move

            # Collect threat landscape for diagnostics on the search-bound
            # paths (VCF / PNS / VCT / fork / fallback).  The instant-win
            # and instant-block shortcuts above return without it.
            if diag is not None:
                ctx = _scan_context(state)
                our_fours = _find_four_threats(state, stone, ctx)
                our_threes = _find_three_threats(state, stone, ctx)
                opp_fours = _find_four_threats(state, opp, ctx)
                opp_threes = _find_three_threats(state, opp, ctx)
                diag.our_threats = ThreatInfo(
                    four_threats=len(our_fours),
                    three_threats=len(our_threes),
                )
                diag.opp_threats = ThreatInfo(
                    four_threats=len(opp_fours),
                    three_threats=len(opp_threes),
                )

            # Single VCF transposition cache shared across this decision —
            # our VCF, opponent VCF, and breaker probes all visit
            # overlapping sub-positions; reusing proofs cuts redundant
            # deep search.
            vcf_cache: _ThreatCache = {}

            # 2) Try VCF — very fast, deep search
            move = _vcf_state(state, stone, self.vcf_depth, cache=vcf_cache)
            if move is not None:
                logfire.info("VCF found winning move: {move}", move=str(move))
                if diag is not None:
                    diag.decision_path = "vcf_win"
                    diag.chosen_move = move
                    diag.our_threats = diag.our_threats.model_copy(update={"vcf_result": "win"})
                return move

            # 3) Try broader threat search (PNS or VCT)
            if self.threat_search == "pns":
                move = pns_threat_search(state, stone, max_nodes=self.pns_nodes, max_time_s=self.pns_time_s)
                if move is not None:
                    logfire.info("PNS found winning move: {move}", move=str(move))
                    if diag is not None:
                        diag.decision_path = "pns_win"
                        diag.chosen_move = move
                    return move
            else:
                move = _vct(state, stone, self.vct_depth, vcf_cache=vcf_cache)
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
            opp_vcf = _vcf_state(opp_state, opp, self.vcf_depth, cache=vcf_cache)
            if opp_vcf is not None:
                if diag is not None:
                    diag.opp_threats = diag.opp_threats.model_copy(update={"vcf_result": "win"})
                    diag.notes.append(f"Opponent has VCF at {opp_vcf}")
                result = self._vcf_aware_search(board, state, stone, opp_vcf, vcf_cache)
                if diag is not None:
                    diag.decision_path = diag.decision_path or "vcf_aware_search"
                    diag.chosen_move = result
                return result

            # 5) Check if opponent has a VCT / PNS threat
            opp_vct_move = self._check_opponent_vct(board, state, opp_state, stone, vcf_cache)
            if opp_vct_move is not None:
                if diag is not None:
                    diag.decision_path = "vct_aware_search"
                    diag.chosen_move = opp_vct_move
                return opp_vct_move

            # 6) Check for opponent fork points (converging threat lines)
            fork_move = self._block_fork(state, stone)
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
        state: SearchState,
        opp_state: SearchState,
        stone: Stone,
        vcf_cache: _ThreatCache,
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

        The earlier ``_vcf_state`` check in :meth:`get_move` has already
        ruled out unblockable fours, so any four-threat seen here is a
        dead-four.
        """
        opp = stone.opponent()

        # Cheap predicates short-circuit on the first hit — building full
        # threat lists just to ask "any?" / "≥ 2?" used to dominate the
        # quiet-position cost of every threat-brain ``get_move``.
        ctx = _scan_context(state)
        if not _has_any_four_threat(state, opp, ctx) and _count_distinct_three_cells(state, opp, 2, ctx) < 2:
            return None

        if self.threat_search == "pns":
            opp_threat = pns_threat_search(
                opp_state,
                opp,
                max_nodes=self.pns_nodes,
                max_time_s=self.pns_time_s,
            )
        else:
            opp_threat = _vct(opp_state, opp, self.vct_depth, vcf_cache=vcf_cache)

        if opp_threat is None:
            return None

        logfire.info(
            "Opponent has VCT/PNS threat starting at: {move}",
            move=str(opp_threat),
        )
        return self._vct_aware_search(board, state, stone, opp_threat, vcf_cache)

    def _vct_aware_search(
        self,
        board: Board,
        state: SearchState,
        stone: Stone,
        opp_threat_move: Move,
        vcf_cache: _ThreatCache,
    ) -> Move:
        """Pick the best move when the opponent has a VCT/PNS forced win.

        Order candidates by static evaluation (descending), then probe in
        order until we find the first move after which the opponent
        provably has no forced win.  Because the candidate list is
        eval-sorted, that breaker is also the positionally strongest.

        A probe that runs out of budget proves nothing, so such a
        candidate is only remembered: the search keeps looking for a
        proven breaker and falls back to the best unproven candidate
        when the time budget ends without one.  (Treating "budget ran
        out" as "safe" used to pick an unproven move in over half of the
        positions that reach this search.)

        A global wall-clock cap (``self.pns_time_s``) prevents the
        worst-case blow-up that used to be possible when every one of
        the 40-50 candidates was probed with a half-budget PNS call.
        Per-candidate PNS budgets are additionally shrunk to the
        remaining global budget.
        """
        opp = stone.opponent()
        candidates = get_candidate_moves(state, radius=_CANDIDATE_RADIUS)

        # Pre-score every candidate with a static eval so strong moves
        # are probed first.
        scored = _scored_moves(state, candidates, stone)

        # Probe caches shared across candidates — the positions reached
        # after different first moves transpose heavily.
        vct_probe_cache: _ThreatCache = {}

        deadline = time.monotonic() + self.pns_time_s
        probed = 0
        unproven: tuple[float, Move] | None = None
        for score, move in scored:
            if time.monotonic() > deadline:
                logfire.info(
                    "VCT-aware search: time budget exhausted after {n} candidates",
                    n=probed,
                )
                break
            probed += 1

            state.make(move.row, move.col, stone)
            try:
                # Fast VCF probe first.
                if _vcf_state(state, opp, self.vcf_depth, cache=vcf_cache) is not None:
                    continue
                # Expensive threat probe — bounded by remaining global budget.
                if self.threat_search == "pns":
                    remaining = max(_MIN_PROBE_TIME_S, deadline - time.monotonic())
                    status, _ = pns_threat_status(
                        state,
                        opp,
                        max_nodes=int(self.pns_nodes * _OPP_PNS_NODE_FRACTION),
                        max_time_s=min(self.pns_time_s * _OPP_PNS_TIME_FRACTION, remaining),
                    )
                    if status is PNSStatus.PROVED:
                        continue
                    if status is PNSStatus.UNKNOWN:
                        if unproven is None:
                            unproven = (score, move)
                        continue
                else:
                    if (
                        _vct_state(
                            state,
                            opp,
                            self.vct_depth,
                            vct_cache=vct_probe_cache,
                            vcf_cache=vcf_cache,
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

        if unproven is not None:
            score, move = unproven
            logfire.info(
                "VCT-aware search: no proven breaker, best unproven {move} (eval={score:.0f})",
                move=str(move),
                score=score,
            )
            return move

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
        state: SearchState,
        stone: Stone,
    ) -> Move | None:
        """Block opponent fork points — cells where >= 2 threat lines converge.

        If the opponent has a fork cell, blocking it is urgent.  Among
        multiple fork cells, pick the one that scores best for us.
        """
        opp = stone.opponent()
        forks = find_fork_cells(state, opp)

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
        return _scored_moves(state, forks, stone)[0][1]

    # -- VCF-aware search ---------------------------------------------------

    def _vcf_aware_search(
        self,
        board: Board,
        state: SearchState,
        stone: Stone,
        opp_vcf_move: Move,
        vcf_cache: _ThreatCache,
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
        candidates = get_candidate_moves(state, radius=_CANDIDATE_RADIUS)
        durable, breakers = classify_vcf_breakers(
            state, candidates, stone, self.vcf_depth, cache=vcf_cache
        )

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

        return _scored_moves(state, effective, stone)[0][1]
