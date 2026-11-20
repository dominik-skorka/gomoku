"""Alpha-beta brain — negamax with alpha-beta pruning, move ordering, and killer heuristic.

Move-classification helpers (``_check_win``, ``_count_open_threes_at``,
``_split_threats``, ``_is_four_threat_move``) live in
:mod:`gomoku_brain.alphabeta_ordering` so the search engine itself is
not interleaved with pattern scanners.
"""

from __future__ import annotations

import time
from typing import Any, override

import logfire
import numpy as np
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.settings import AlphaBetaBrainSettings

from gomoku_brain.alphabeta_ordering import (
    _check_win,
    _is_four_threat_move,
    _split_threats,
)
from gomoku_brain.alphabeta_quiescence import EvalCache, QuiesceCtx, quiesce
from gomoku_brain.alphabeta_root import root_vcf
from gomoku_brain.base import Brain
from gomoku_brain.brain_base import BrainSession
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.opening import deduplicate_by_symmetry, lookup_opening_book
from gomoku_brain.priors import compute_priors
from gomoku_brain.search_state import SearchState
from gomoku_brain.zobrist import (
    SIDE_KEY,
    BoundType,
    TranspositionTable,
)

# Number of killer-move slots per depth level.
_KILLER_SLOTS = 2

# Maximum number of candidate moves to search at each node.
_MAX_CANDIDATES = 20

# Maximum extra plies for quiescence search at leaves.
_MAX_QS_DEPTH = 4

# Type alias for the history heuristic table: (stone, row, col) -> cumulative score.
HistoryTable = dict[tuple[int, int, int], int]

# Half-width of the aspiration window around the previous iteration's score.
_ASPIRATION_WINDOW = 50


class _SearchTimeout(Exception):
    """Raised when the search exceeds its time budget."""


# Futility pruning margin: if static eval + margin < alpha at a frontier node,
# the move is unlikely to raise alpha.
_FUTILITY_MARGIN = 1500


class AlphaBetaBrain(Brain[AlphaBetaBrainSettings], name="alphabeta"):
    """Player using alpha-beta pruning on top of negamax."""

    def __init__(self, settings: AlphaBetaBrainSettings | None = None, /, **overrides: Any) -> None:
        super().__init__(settings, **overrides)
        s = self.settings
        self.depth = s.depth
        # PNS budget for checking that a root VCF breaker leaves the
        # opponent no other forced win (0: pick by evaluation only).
        self.vcf_breaker_pns_nodes = s.vcf_breaker_pns_nodes
        self.vcf_breaker_time_s = s.vcf_breaker_time_s
        # Threads probing those breakers in parallel (gomoku_brain.probe).
        self.probe_threads = s.probe_threads
        self.vcf_depth = s.vcf_depth
        self.qs_vcf_depth = s.qs_vcf_depth
        self.threat_ordering = s.threat_ordering
        self.threat_extensions = s.threat_extensions
        self.null_move_pruning = s.null_move_pruning
        self.null_move_reduction = s.null_move_reduction
        self.lmr = s.lmr
        self.lmr_threshold = s.lmr_threshold
        self.lmr_min_depth = s.lmr_min_depth
        self.early_game_radius = s.early_game_radius
        self.early_game_threshold = s.early_game_threshold
        self.early_game_max_candidates = s.early_game_max_candidates
        self.early_game_lmr_threshold = s.early_game_lmr_threshold
        self.early_game_futility_margin = s.early_game_futility_margin
        # Branching cap and futility margin after the early game.
        self.max_candidates = s.max_candidates
        self.futility_margin = s.futility_margin
        # Nodes at least this deep rank candidates by a full evaluation
        # of each move; shallower nodes use the one-kernel pattern prior.
        self.eval_ordering_depth = s.eval_ordering_depth
        self.time_limit_s = s.time_limit_s
        # Benchmark instrumentation — reset on each get_move, updated inside
        # _alphabeta / _quiesce.  Read by bench_alphabeta.py.  Zero overhead
        # when ignored (one int increment per node).
        self.node_count = 0
        self.qs_node_count = 0
        # Per-search quiescence context — created once per get_move so the
        # leaf entry points don't allocate a fresh ctx each call.
        self._qs_ctx: QuiesceCtx | None = None
        self._file_book = None
        if s.opening_book:
            from pathlib import Path

            from gomoku_brain.opening_book import OpeningBookFile

            book_path = Path(s.opening_book)
            if book_path.exists():
                self._file_book = OpeningBookFile.load(book_path)
                logfire.info(
                    "Loaded opening book: {path} ({n} positions)", path=str(book_path), n=len(self._file_book.positions)
                )

    @override
    def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
        diag = session.diag

        # -- Opening book (file-based, then hardcoded) -------------------
        book_move = None
        if self._file_book is not None:
            from gomoku_brain.opening_book import lookup_file_book

            book_move = lookup_file_book(self._file_book, board, stone)
            if book_move is not None:
                logfire.info("File opening book: {move}", move=str(book_move))
        if book_move is None:
            book_move = lookup_opening_book(board, stone)
        if book_move is not None:
            logfire.info("Opening book: {move}", move=str(book_move))
            if diag is not None:
                diag.decision_path = "opening_book"
                diag.chosen_move = book_move
            return book_move

        # -- Root-level VCF pre-check ------------------------------------
        if self.vcf_depth > 0:
            vcf_move = root_vcf(
                board,
                stone,
                self.vcf_depth,
                self.vcf_breaker_pns_nodes,
                self.vcf_breaker_time_s,
                self.probe_threads,
            )
            if vcf_move is not None:
                if diag is not None:
                    # root_vcf names the case (instant_win, vcf_win,
                    # vcf_breaker, vcf_block_no_breaker); keep it.
                    diag.decision_path = diag.decision_path or "vcf_precheck"
                    diag.chosen_move = vcf_move
                    diag.notes.append(f"VCF found at depth {self.vcf_depth}")
                return vcf_move

        # -- Build the mutable SearchState used throughout the search ---
        # make/unmake on this single object replaces per-node Board /
        # _GridBoard allocations.  All helpers duck-type it via
        # ``grid_utils``.
        state = SearchState.from_board(board, stone)

        tt = TranspositionTable()
        best_move: Move | None = None
        # Reset per-move node counters for benchmarks / diagnostics.
        self.node_count = 0
        self.qs_node_count = 0
        # History table shared across all ID iterations so it accumulates.
        history: HistoryTable = {}
        eval_cache: EvalCache = {}
        self._qs_ctx = QuiesceCtx(qs_vcf_depth=self.qs_vcf_depth, eval_cache=eval_cache)
        prev_score: float | None = None
        aspiration_resets = 0

        # Time management: 0 means unlimited.  Deadline is independent
        # of the brain_session timer because it gates the search itself
        # (not just diagnostics reporting).
        deadline = (time.monotonic() + self.time_limit_s) if self.time_limit_s > 0 else 0.0

        # Static eval before the move (for diagnostics).
        if diag is not None:
            diag.eval_before = evaluate_board(board, stone)

        # Iterative deepening: search at increasing depths so that each
        # pass seeds the TT with best-move info for the next, deeper pass.
        depth_reached = 0
        for d in range(1, self.depth + 1):
            # Fresh killers each iteration — they are depth-relative.
            killers: dict[int, list[Move]] = {}

            try:
                # Aspiration window: after the first iteration, search with a
                # narrow window around the previous score.  Widen on failure.
                if prev_score is not None and abs(prev_score) < 90_000:
                    a = prev_score - _ASPIRATION_WINDOW
                    b = prev_score + _ASPIRATION_WINDOW
                    score, move = self._alphabeta(
                        state,
                        d,
                        a,
                        b,
                        stone,
                        killers,
                        tt,
                        history,
                        eval_cache,
                        self.threat_extensions,
                        deadline=deadline,
                    )
                    # Re-search on fail-low or fail-high.
                    if score <= a or score >= b:
                        aspiration_resets += 1
                        score, move = self._alphabeta(
                            state,
                            d,
                            float("-inf"),
                            float("inf"),
                            stone,
                            killers,
                            tt,
                            history,
                            eval_cache,
                            self.threat_extensions,
                            deadline=deadline,
                        )
                else:
                    score, move = self._alphabeta(
                        state,
                        d,
                        float("-inf"),
                        float("inf"),
                        stone,
                        killers,
                        tt,
                        history,
                        eval_cache,
                        self.threat_extensions,
                        deadline=deadline,
                    )
            except _SearchTimeout:
                logfire.info(
                    "ID depth {d}/{max_d}: time limit reached, using best from depth {prev_d}",
                    d=d,
                    max_d=self.depth,
                    prev_d=d - 1,
                )
                break

            prev_score = score
            depth_reached = d
            if move is not None:
                best_move = move
            if diag is not None:
                diag.notes.append(f"ID d={d}/{self.depth}: score={score}, move={move}, tt={len(tt)}")
            logfire.info(
                "ID depth {d}/{max_d}: score={score}, move={move}, tt_size={tt_size}",
                d=d,
                max_d=self.depth,
                score=score,
                move=str(move),
                tt_size=len(tt),
            )
            # Early exit on proven win.
            if score >= 100_000:
                break

        self.qs_node_count = self._qs_ctx.nodes
        assert best_move is not None

        if diag is not None:
            from gomoku.diagnostics import CandidateInfo, SearchStats, ThreatInfo

            from gomoku_brain.threat_detect import _find_four_threats, _find_three_threats

            search_ms = session.elapsed_ms()
            diag.decision_path = diag.decision_path or "alphabeta"
            diag.chosen_move = best_move
            diag.search_stats = SearchStats(
                depth_reached=depth_reached,
                nodes_searched=self.node_count,
                qs_nodes_searched=self.qs_node_count,
                tt_size=len(tt),
                tt_hits=tt.hits,
                tt_probes=tt.probes,
                time_ms=search_ms,
                aspiration_resets=aspiration_resets,
            )
            # Collect top candidates from TT root entry and eval scores.
            root_candidates = get_candidate_moves(board, radius=2)
            if len(root_candidates) > 8 and len(board.occupied) <= 8:
                root_candidates = deduplicate_by_symmetry(board, root_candidates)
            scored_cands: list[tuple[float, Move]] = []
            for m in root_candidates[:20]:
                trial = board.place_stone(m, stone)
                sc = evaluate_board(trial, stone)
                scored_cands.append((sc, m))
            scored_cands.sort(key=lambda x: x[0], reverse=True)
            diag.top_candidates = [CandidateInfo(move=m, score=sc, score_type="eval") for sc, m in scored_cands[:10]]
            # Alternative-best: the top-ranked candidate when it differs
            # from the move we actually chose.  Analyzers flag this as a
            # potential blunder when the gap is large.
            if diag.top_candidates and diag.top_candidates[0].move != best_move:
                diag.alternative_best = diag.top_candidates[0]
            # Threat landscape before the move — populate only when a
            # downstream layer (threat.py) didn't already.
            opp = Stone.BLACK if stone is Stone.WHITE else Stone.WHITE
            if diag.our_threats == ThreatInfo():
                diag.our_threats = ThreatInfo(
                    four_threats=len(_find_four_threats(board, stone)),
                    three_threats=len(_find_three_threats(board, stone)),
                )
            if diag.opp_threats == ThreatInfo():
                diag.opp_threats = ThreatInfo(
                    four_threats=len(_find_four_threats(board, opp)),
                    three_threats=len(_find_three_threats(board, opp)),
                )
            # Eval after the chosen move.
            after_board = board.place_stone(best_move, stone)
            diag.eval_after = evaluate_board(after_board, stone)

        return best_move

    # -- Instant-win / forced-block pre-check --------------------------------

    @staticmethod
    def _find_urgent_move(
        state: SearchState,
        candidates: list[Move],
        stone: Stone,
    ) -> Move | None:
        """Return an immediate winning move for *stone*, or a move that blocks
        the opponent's immediate win.  Returns ``None`` if no urgent move.

        On the JIT path the node's candidate list is packed into one array
        and probed with two ``find_winning_move_grid`` calls — the kernel
        scans line neighbours in place, so no make/unmake is needed (this
        used to cost two make/win-check/unmake cycles per candidate per
        interior node).  The pure-Python fallback keeps the probe loop.
        """
        from gomoku_brain.evaluate import _should_use_jit

        opponent = stone.opponent()

        if _should_use_jit():
            from gomoku_brain.evaluate_jit import find_winning_move_grid, get_dirs

            cand_arr = np.empty((len(candidates), 2), dtype=np.int32)
            for i, m in enumerate(candidates):
                cand_arr[i, 0] = m.row
                cand_arr[i, 1] = m.col
            dirs = get_dirs()
            result = find_winning_move_grid(state.grid, state.size, cand_arr, int(stone), dirs)
            if result.shape[0] > 0:
                return Move(row=int(result[0, 0]), col=int(result[0, 1]))
            result = find_winning_move_grid(state.grid, state.size, cand_arr, int(opponent), dirs)
            if result.shape[0] > 0:
                return Move(row=int(result[0, 0]), col=int(result[0, 1]))
            return None

        block: Move | None = None
        for move in candidates:
            # Check our win
            state.make(move.row, move.col, stone)
            try:
                if _check_win(state, move):
                    return move
            finally:
                state.unmake()
            # Check opponent's win (only need to find first one)
            if block is None:
                state.make(move.row, move.col, opponent)
                try:
                    if _check_win(state, move):
                        block = move
                finally:
                    state.unmake()
        return block

    # -- Move ordering -------------------------------------------------------

    def _prior_order(
        self,
        state: SearchState,
        candidates: list[Move],
        stone: Stone,
        depth: int,
        killers: dict[int, list[Move]],
        tt_move: Move | None,
    ) -> list[Move]:
        """Rank candidates by pattern priors; TT move and killers first."""
        coords = {(m.row, m.col) for m in candidates}
        promoted: list[Move] = []
        skip: set[tuple[int, int]] = set()
        for m in ([tt_move] if tt_move is not None else []) + killers.get(depth, []):
            key = (m.row, m.col)
            if key in coords and key not in skip:
                promoted.append(m)
                skip.add(key)
        rest = [m for m in candidates if (m.row, m.col) not in skip]
        if not rest:
            return promoted
        prior = compute_priors(state, [(m.row, m.col) for m in rest], int(stone), int(stone.opponent()))
        order = sorted(range(len(rest)), key=lambda i: -prior[i])
        return promoted + [rest[i] for i in order]

    def _order_moves(
        self,
        state: SearchState,
        candidates: list[Move],
        stone: Stone,
        depth: int,
        killers: dict[int, list[Move]],
        tt_move: Move | None = None,
        history: HistoryTable | None = None,
        eval_cache: EvalCache | None = None,
    ) -> list[Move]:
        """Sort candidates by shallow evaluation for better pruning.

        TT best-move is promoted first, then killer moves (moves that recently
        caused a beta-cutoff at this depth), then evaluation-sorted remainder.
        History heuristic scores are added to the evaluation to boost moves
        that have historically caused cutoffs.
        """
        skip: set[tuple[int, int]] = set()
        promoted: list[Move] = []
        cand_coords = {(m.row, m.col) for m in candidates}

        # 1. TT best-move first
        if tt_move is not None and (tt_move.row, tt_move.col) in cand_coords:
            promoted.append(tt_move)
            skip.add((tt_move.row, tt_move.col))

        # 2. Killer moves
        if depth in killers:
            for km in killers[depth]:
                if (km.row, km.col) not in skip and (km.row, km.col) in cand_coords:
                    promoted.append(km)
                    skip.add((km.row, km.col))

        rest = [m for m in candidates if (m.row, m.col) not in skip]

        # 3. Threat-tier (optional): moves that create a four or five for us,
        #    or block one for the opponent.  Cheap inline scan.
        if self.threat_ordering:
            threat_moves, non_threat = _split_threats(state, rest, stone, include_forks=True)
        else:
            threat_moves = []
            non_threat = rest

        # 4. Eval + history sorted remainder.  Score threats and non-threats
        #    the same way, but keep them in separate tiers so the threat tier
        #    always precedes non-threats.  Within each tier we sort by score
        #    so strong threats (create-five, block-four) come before weaker
        #    ones (fork) and strong positional moves before weak ones.
        stone_int = int(stone)

        opponent = stone.opponent()

        def _score_move(move: Move) -> float:
            state.make(move.row, move.col, stone)
            try:
                # The cache holds evaluations from the perspective of the
                # side to move in the cached position — here the opponent —
                # because futility pruning, null-move pruning and
                # quiescence read the same entries that way.  Evaluation
                # is antisymmetric, so negating gives the mover's score.
                zh = state.zobrist
                if eval_cache is not None and zh in eval_cache:
                    score = -eval_cache[zh]
                else:
                    to_move_eval = evaluate_board(state, opponent)
                    if eval_cache is not None:
                        eval_cache[zh] = to_move_eval
                    score = -to_move_eval
            finally:
                state.unmake()
            if history:
                score += history.get((stone_int, move.row, move.col), 0)
            return score

        scored_threats = [(_score_move(m), m) for m in threat_moves]
        scored_threats.sort(key=lambda x: x[0], reverse=True)
        scored_rest = [(_score_move(m), m) for m in non_threat]
        scored_rest.sort(key=lambda x: x[0], reverse=True)

        return promoted + [m for _, m in scored_threats] + [m for _, m in scored_rest]

    # -- Quiescence search ---------------------------------------------------

    def _quiesce(
        self,
        state: SearchState,
        alpha: float,
        beta: float,
        stone: Stone,
        qs_depth: int,
    ) -> float:
        """Quiescence search — thin wrapper around
        :func:`gomoku_brain.alphabeta_quiescence.quiesce`.

        Delegates to the free function with the per-search
        :class:`QuiesceCtx` created in :meth:`get_move`; the node
        counter is synced back into ``self.qs_node_count`` once after
        the iterative-deepening loop.
        """
        assert self._qs_ctx is not None
        return quiesce(state, alpha, beta, stone, self._qs_ctx, qs_depth)

    # -- Core search ---------------------------------------------------------

    @staticmethod
    def _record_killer(killers: dict[int, list[Move]], depth: int, move: Move) -> None:
        """Store *move* as a killer at *depth* (LRU with _KILLER_SLOTS slots)."""
        slots = killers.setdefault(depth, [])
        if move in slots:
            return
        slots.insert(0, move)
        if len(slots) > _KILLER_SLOTS:
            slots.pop()

    def _alphabeta(
        self,
        state: SearchState,
        depth: int,
        alpha: float,
        beta: float,
        stone: Stone,
        killers: dict[int, list[Move]],
        tt: TranspositionTable,
        history: HistoryTable,
        eval_cache: EvalCache,
        ext_left: int = 0,
        allow_null: bool = True,
        deadline: float = 0.0,
    ) -> tuple[float, Move | None]:
        self.node_count += 1
        # -- Time check (skip at shallow depths to limit overhead) -----------
        if deadline and depth >= 2 and time.monotonic() >= deadline:
            raise _SearchTimeout

        zh = state.zobrist

        # -- Transposition-table probe ---------------------------------------
        alpha_orig = alpha
        tt_move: Move | None = None
        tt_entry = tt.probe(zh)
        if tt_entry is not None and tt_entry.depth >= depth:
            tt_move = tt_entry.best_move
            if tt_entry.flag is BoundType.EXACT:
                return tt_entry.score, tt_entry.best_move
            if tt_entry.flag is BoundType.LOWER:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag is BoundType.UPPER:
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score, tt_entry.best_move
        elif tt_entry is not None:
            # Shallower entry — still useful for move ordering.
            tt_move = tt_entry.best_move

        # -- Null Move Pruning -----------------------------------------------
        # If the side to move is likely far ahead, pass the turn and search at
        # reduced depth.  If the opponent still can't beat beta even with a
        # free move, prune this branch.
        # Guards: enabled, not consecutive (allow_null), sufficient depth,
        # non-PV node (zero-window), and board has enough pieces.
        if self.null_move_pruning and allow_null and depth >= 3 and (beta - alpha) == 1 and state.occ_count >= 4:
            # Use cached static eval when available.
            if zh in eval_cache:
                null_eval = eval_cache[zh]
            else:
                null_eval = evaluate_board(state, stone)
                eval_cache[zh] = null_eval

            if null_eval >= beta:
                r = self.null_move_reduction
                null_depth = max(depth - 1 - r, 0)
                # Pass the turn: flip the Zobrist side-to-move bit only.
                # We toggle it on the state in place and restore it after
                # the recursive call so the child sees the correct hash.
                state.zobrist ^= SIDE_KEY
                try:
                    if null_depth <= 0:
                        null_score = -self._quiesce(
                            state,
                            -beta,
                            -beta + 1,
                            stone.opponent(),
                            _MAX_QS_DEPTH,
                        )
                    else:
                        null_score, _ = self._alphabeta(
                            state,
                            null_depth,
                            -beta,
                            -beta + 1,
                            stone.opponent(),
                            killers,
                            tt,
                            history,
                            eval_cache,
                            ext_left,
                            allow_null=False,
                            deadline=deadline,
                        )
                        null_score = -null_score
                finally:
                    state.zobrist ^= SIDE_KEY
                if null_score >= beta:
                    return beta, None

        # -- Candidate generation --------------------------------------------
        occ_count = state.occ_count
        is_early = occ_count <= self.early_game_threshold
        radius = self.early_game_radius if is_early else 2
        candidates = get_candidate_moves(state, radius=radius)
        if not candidates:
            return 0.0, None

        # Symmetry dedup: remove equivalent moves in early game.
        if occ_count <= 8:
            candidates = deduplicate_by_symmetry(state, candidates)

        # Resolve early-game adaptive parameters once.
        futility_margin = self.early_game_futility_margin if is_early else self.futility_margin
        lmr_threshold = self.early_game_lmr_threshold if is_early else self.lmr_threshold
        lmr_min_depth = 2 if is_early else self.lmr_min_depth

        # Fast path: check for an instant win or a forced block before the
        # expensive evaluation-based sort.  When urgent is found, play it
        # immediately (winning) or promote it (blocking).
        urgent = self._find_urgent_move(state, candidates, stone)
        if urgent is not None:
            state.make(urgent.row, urgent.col, stone)
            try:
                is_win = _check_win(state, urgent)
            finally:
                state.unmake()
            if is_win:
                return 100_000.0 + depth, urgent
            # It's a block — promote it to head but still search alternatives
            candidates = [urgent, *(m for m in candidates if m != urgent)]
        elif depth >= self.eval_ordering_depth:
            candidates = self._order_moves(state, candidates, stone, depth, killers, tt_move, history, eval_cache)
        else:
            # Shallow nodes are the bulk of the tree, so they get a cheap
            # one-kernel ranking instead of an evaluation per candidate.
            # They must be ranked before the cap below: the candidate list
            # is in board order, and cutting it unranked kept whichever
            # cells came first on the board.
            candidates = self._prior_order(state, candidates, stone, depth, killers, tt_move)

        # Cap branching factor to avoid combinatorial explosion.
        cap = self.early_game_max_candidates if is_early else self.max_candidates
        candidates = candidates[:cap]

        best_score = float("-inf")
        best_move: Move | None = None

        # Lazy static eval for futility pruning (computed at most once).
        futility_eval: float | None = None

        for i, move in enumerate(candidates):
            state.make(move.row, move.col, stone)
            try:
                if _check_win(state, move):
                    score = 100_000.0 + depth
                    # Terminal — store exact and return.
                    tt.store(zh, depth, score, BoundType.EXACT, move)
                    return score, move

                # -- Futility pruning at frontier (depth 2) ------------------
                # If the position's static eval + margin can't reach alpha,
                # skip non-first moves (first move is always searched fully
                # for safety).  Evaluated on the pre-move position.
                if depth == 2 and i > 0:
                    if futility_eval is None:
                        if zh in eval_cache:
                            futility_eval = eval_cache[zh]
                        else:
                            # Unmake briefly to evaluate the pre-move position,
                            # then redo.  Cheaper than an isinstance-heavy view.
                            state.unmake()
                            futility_eval = evaluate_board(state, stone)
                            eval_cache[zh] = futility_eval
                            state.make(move.row, move.col, stone)
                    if futility_eval + futility_margin <= alpha:
                        continue

                # -- Threat extension: extend by 1 ply on forcing moves ------
                child_depth = depth - 1
                child_ext = ext_left
                is_threat = False
                if (ext_left > 0 or self.lmr) and depth >= 2:
                    is_threat = _is_four_threat_move(state, move, stone)
                    if is_threat and ext_left > 0:
                        child_depth = depth  # extend: don't decrement
                        child_ext = ext_left - 1

                # -- Determine if this move is a known good move (skip LMR) --
                is_tactical = is_threat or move == tt_move or (depth in killers and move in killers[depth])

                if depth <= 1:
                    score = -self._quiesce(
                        state,
                        -beta,
                        -alpha,
                        stone.opponent(),
                        _MAX_QS_DEPTH,
                    )
                elif i == 0:
                    # First move (expected best): search with full window.
                    opp_score, _ = self._alphabeta(
                        state,
                        child_depth,
                        -beta,
                        -alpha,
                        stone.opponent(),
                        killers,
                        tt,
                        history,
                        eval_cache,
                        child_ext,
                        deadline=deadline,
                    )
                    score = -opp_score
                else:
                    # -- Late Move Reduction (LMR) ---------------------------
                    # Later moves that aren't tactical get searched at reduced
                    # depth first.  If the reduced search beats alpha, re-search
                    # at full depth via the normal PVS fail-high path below.
                    lmr_reduced = False
                    scout_depth = child_depth
                    if self.lmr and depth >= lmr_min_depth and i >= lmr_threshold and not is_tactical:
                        scout_depth = max(child_depth - 1, 1)
                        lmr_reduced = True

                    # PVS zero-window scout (possibly at reduced depth).
                    opp_score, _ = self._alphabeta(
                        state,
                        scout_depth,
                        -alpha - 1,
                        -alpha,
                        stone.opponent(),
                        killers,
                        tt,
                        history,
                        eval_cache,
                        child_ext,
                        deadline=deadline,
                    )
                    score = -opp_score

                    # Re-search at full depth if LMR scout fails high.
                    if lmr_reduced and score > alpha:
                        opp_score, _ = self._alphabeta(
                            state,
                            child_depth,
                            -alpha - 1,
                            -alpha,
                            stone.opponent(),
                            killers,
                            tt,
                            history,
                            eval_cache,
                            child_ext,
                            deadline=deadline,
                        )
                        score = -opp_score

                    # Re-search with full window if scout fails high.
                    if alpha < score < beta:
                        opp_score, _ = self._alphabeta(
                            state,
                            child_depth,
                            -beta,
                            -score,
                            stone.opponent(),
                            killers,
                            tt,
                            history,
                            eval_cache,
                            child_ext,
                            deadline=deadline,
                        )
                        score = -opp_score
            finally:
                state.unmake()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                self._record_killer(killers, depth, move)
                # History heuristic: reward this move for causing a cutoff.
                key = (int(stone), move.row, move.col)
                history[key] = history.get(key, 0) + depth * depth
                break

        # -- Store in TT -----------------------------------------------------
        if best_score <= alpha_orig:
            flag = BoundType.UPPER
        elif best_score >= beta:
            flag = BoundType.LOWER
        else:
            flag = BoundType.EXACT
        tt.store(zh, depth, best_score, flag, best_move)

        return best_score, best_move
