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
# VCF / VCT search core
#
# The recursive workers live in :mod:`gomoku_brain.threat_search` —
# pure search logic with no orchestration / diagnostics / logfire.
# We import them at the bottom of the helper section, *after* the
# detection helpers (``_find_four_threats`` etc.) are defined, so
# ``threat_search.py``'s top-level import resolves cleanly against the
# partially-loaded ``threat`` module.
#
# Re-exported so that downstream modules (``alphabeta``, ``mcts``,
# ``pns``) and tests that still import these names from
# ``gomoku_brain.threat`` continue to work unchanged.
# ---------------------------------------------------------------------------


from gomoku_brain.threat_search import (  # noqa: E402, F401, I001
    _state_is_win,
    _vcf,
    _vcf_state,
    _vct,
    _vct_defence_moves,
    _vct_defence_moves_state,
    _vct_state,
    classify_vcf_breakers,
)


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
