"""Alpha-beta brain — negamax with alpha-beta pruning, move ordering, and killer heuristic."""

from __future__ import annotations

import time

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.opening import deduplicate_by_symmetry, lookup_opening_book
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

# Directions for line scanning (horizontal, vertical, two diagonals).
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Type alias for the history heuristic table: (stone, row, col) -> cumulative score.
HistoryTable = dict[tuple[int, int, int], int]

# Type alias for the evaluation cache: zobrist_hash -> evaluation score.
EvalCache = dict[int, float]

# Half-width of the aspiration window around the previous iteration's score.
_ASPIRATION_WINDOW = 50


class _SearchTimeout(Exception):
    """Raised when the search exceeds its time budget."""


# Futility pruning margin: if static eval + margin < alpha at a frontier node,
# the move is unlikely to raise alpha.
_FUTILITY_MARGIN = 1500


# ---------------------------------------------------------------------------
# JIT-dispatched helpers (used by class methods below)
# ---------------------------------------------------------------------------


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
    one at *row*, *col*) with both ends empty.  Returns 0–4.
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
            ):
                threat_list.append(m)
            elif include_forks and (
                _count_open_threes_at(board, m.row, m.col, stone) >= 2
                or _count_open_threes_at(board, m.row, m.col, stone.opponent()) >= 2
            ):
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
        if is_threat:
            threat_list.append(move)
        elif include_forks and (
            _count_open_threes_at(board, move.row, move.col, stone) >= 2
            or _count_open_threes_at(board, move.row, move.col, opponent) >= 2
        ):
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
    if _count_open_threes_at(board, move.row, move.col, opp) >= 2:
        return True
    return False


class AlphaBetaBrain:
    """MoveProvider using alpha-beta pruning on top of negamax."""

    def __init__(
        self,
        depth: int = 4,
        vcf_depth: int = 0,
        qs_vcf_depth: int = 0,
        *,
        threat_ordering: bool = True,
        threat_extensions: int = 0,
        null_move_pruning: bool = False,
        null_move_reduction: int = 2,
        lmr: bool = False,
        lmr_threshold: int = 3,
        lmr_min_depth: int = 3,
        early_game_radius: int = 1,
        early_game_threshold: int = 6,
        early_game_max_candidates: int = 10,
        early_game_lmr_threshold: int = 2,
        early_game_futility_margin: int = 800,
        time_limit_s: float = 0.0,
        opening_book: str = "",
    ) -> None:
        self.depth = depth
        self.vcf_depth = vcf_depth
        self.qs_vcf_depth = qs_vcf_depth
        self.threat_ordering = threat_ordering
        self.threat_extensions = threat_extensions
        self.null_move_pruning = null_move_pruning
        self.null_move_reduction = null_move_reduction
        self.lmr = lmr
        self.lmr_threshold = lmr_threshold
        self.lmr_min_depth = lmr_min_depth
        self.early_game_radius = early_game_radius
        self.early_game_threshold = early_game_threshold
        self.early_game_max_candidates = early_game_max_candidates
        self.early_game_lmr_threshold = early_game_lmr_threshold
        self.early_game_futility_margin = early_game_futility_margin
        self.time_limit_s = time_limit_s
        # Benchmark instrumentation — reset on each get_move, updated inside
        # _alphabeta / _quiesce.  Read by bench_alphabeta.py.  Zero overhead
        # when ignored (one int increment per node).
        self.node_count = 0
        self.qs_node_count = 0
        self._file_book = None
        if opening_book:
            from pathlib import Path

            from gomoku_brain.opening_book import OpeningBookFile

            book_path = Path(opening_book)
            if book_path.exists():
                self._file_book = OpeningBookFile.load(book_path)
                logfire.info(
                    "Loaded opening book: {path} ({n} positions)", path=str(book_path), n=len(self._file_book.positions)
                )

    @property
    def name(self) -> str:
        return "alphabeta"

    def get_move(self, board: Board, stone: Stone) -> Move:
        with logfire.span("brain.alphabeta", depth=self.depth, vcf_depth=self.vcf_depth, stone=str(stone)):
            from gomoku.diagnostics import current_move_diagnostics

            diag = current_move_diagnostics()

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
                vcf_move = self._root_vcf(board, stone)
                if vcf_move is not None:
                    if diag is not None:
                        diag.decision_path = "vcf_precheck"
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
            prev_score: float | None = None
            aspiration_resets = 0

            # Time management: 0 means unlimited.
            deadline = (time.monotonic() + self.time_limit_s) if self.time_limit_s > 0 else 0.0
            search_start = time.monotonic()

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

            assert best_move is not None

            if diag is not None:
                from gomoku.diagnostics import CandidateInfo, SearchStats

                search_ms = int((time.monotonic() - search_start) * 1000)
                diag.decision_path = diag.decision_path or "alphabeta"
                diag.chosen_move = best_move
                diag.search_stats = SearchStats(
                    depth_reached=depth_reached,
                    tt_size=len(tt),
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
                diag.top_candidates = [CandidateInfo(move=m, score=sc) for sc, m in scored_cands[:10]]
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

        Uses :meth:`SearchState.make` / :meth:`SearchState.unmake` for zero
        allocation per candidate.
        """
        opponent = stone.opponent()
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

    # -- Root-level VCF pre-check -------------------------------------------

    def _root_vcf(self, board: Board, stone: Stone) -> Move | None:
        """Run VCF at the root for *stone* and its opponent.

        Returns immediately if *stone* has a forced VCF win.
        If the opponent has a VCF, restricts candidates to moves that break
        it and returns the positionally best breaker.
        Returns ``None`` when VCF is irrelevant and normal search should run.
        """
        from gomoku.diagnostics import current_move_diagnostics

        from gomoku_brain.threat import _find_winning_move, _vcf, classify_vcf_breakers

        diag = current_move_diagnostics()

        # 0) Instant win — check before the expensive VCF search.
        instant = _find_winning_move(board, stone)
        if instant is not None:
            logfire.info("AlphaBeta instant win: {move}", move=str(instant))
            if diag is not None:
                diag.decision_path = "instant_win"
            return instant

        # 1) Do we have a forced win?
        our_vcf = _vcf(board, stone, self.vcf_depth, is_attacker=True)
        if our_vcf is not None:
            logfire.info("AlphaBeta VCF win: {move}", move=str(our_vcf))
            if diag is not None:
                diag.decision_path = "vcf_win"
                diag.our_threats = diag.our_threats.model_copy(update={"vcf_result": "win"})
            return our_vcf

        # 2) Does the opponent have a forced win we must disrupt?
        opp = stone.opponent()
        opp_vcf = _vcf(board, opp, self.vcf_depth, is_attacker=True)
        if opp_vcf is None:
            if diag is not None:
                diag.notes.append("No VCF for either side")
            return None  # No VCF concern — proceed with normal search.

        logfire.info("Opponent has VCF starting at {move}, searching breakers", move=str(opp_vcf))
        if diag is not None:
            diag.opp_threats = diag.opp_threats.model_copy(update={"vcf_result": "win"})
            diag.notes.append(f"Opponent VCF at {opp_vcf}, searching breakers")

        candidates = get_candidate_moves(board, radius=2)
        durable, breakers = classify_vcf_breakers(board, candidates, stone, self.vcf_depth)

        if not breakers:
            # Nothing breaks the VCF — position is lost.  Prefer blocking an
            # instant winning move (five-in-a-row) over the multi-step VCF
            # start, since the instant win ends the game immediately.
            block = _find_winning_move(board, opp)
            if block is None:
                block = opp_vcf
            logfire.info(
                "No VCF breaker found, blocking opponent VCF start: {move}",
                move=str(block),
            )
            if diag is not None:
                diag.decision_path = "vcf_block_no_breaker"
                diag.chosen_move = block
                diag.notes.append(f"No VCF breaker, blocking opponent at {block}")
            return block

        effective = durable if durable else breakers
        logfire.info(
            "VCF breakers: {total} total, {durable} durable, using {n}",
            total=len(breakers),
            durable=len(durable),
            n=len(effective),
        )
        if diag is not None:
            diag.decision_path = "vcf_breaker"
            diag.notes.append(f"VCF breakers: {len(breakers)} total, {len(durable)} durable, using {len(effective)}")

        if len(effective) == 1:
            logfire.info("Single VCF breaker: {move}", move=str(effective[0]))
            return effective[0]

        # Multiple breakers — pick via evaluation.
        best_score = float("-inf")
        best_move = effective[0]
        for move in effective:
            trial = board.place_stone(move, stone)
            score = evaluate_board(trial, stone)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    # -- Move ordering -------------------------------------------------------

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

        # 1. TT best-move first
        if tt_move is not None and tt_move in candidates:
            promoted.append(tt_move)
            skip.add((tt_move.row, tt_move.col))

        # 2. Killer moves
        if depth in killers:
            for km in killers[depth]:
                if (km.row, km.col) not in skip and km in candidates:
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

        def _score_move(move: Move) -> float:
            state.make(move.row, move.col, stone)
            try:
                zh = state.zobrist
                if eval_cache is not None and zh in eval_cache:
                    score = eval_cache[zh]
                else:
                    score = evaluate_board(state, stone)
                    if eval_cache is not None:
                        eval_cache[zh] = score
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

    @staticmethod
    def _get_threat_moves(state: SearchState, stone: Stone) -> list[Move]:
        """Return moves that create a four-in-a-row for *stone* or block one for the opponent.

        Only considers candidates within radius 2 of occupied stones.
        This is a lightweight detector for quiescence — not exhaustive.
        """
        candidates = get_candidate_moves(state)
        threats, _ = _split_threats(state, candidates, stone)
        return threats

    def _quiesce(
        self,
        state: SearchState,
        alpha: float,
        beta: float,
        stone: Stone,
        tt: TranspositionTable,
        qs_depth: int,
        eval_cache: EvalCache,
    ) -> float:
        """Quiescence search: extend leaves for forcing (four-threat) moves."""
        self.qs_node_count += 1
        zh = state.zobrist
        # -- VCF probe: detect multi-move forced wins that the one-ply
        #    threat loop would miss.
        if self.qs_vcf_depth > 0:
            from gomoku_brain.grid_utils import grid_to_board
            from gomoku_brain.threat import _find_winning_move, _vcf

            # _vcf requires a Pydantic Board — reconstruct from bitboards.
            vcf_board = grid_to_board(state.grid, state.size)

            # Guard: if the opponent already has a winning move (four/five
            # on the board), we must block — we cannot freely start our own
            # VCF.  Skipping this check caused the engine to return a false
            # +100 000 even when the opponent had an unstoppable live-4.
            opp = stone.opponent()
            if _find_winning_move(vcf_board, opp) is None:
                if _vcf(vcf_board, stone, self.qs_vcf_depth, is_attacker=True) is not None:
                    return 100_000.0

        # Standing pat — static evaluation as a lower bound.
        if zh in eval_cache:
            stand_pat = eval_cache[zh]
        else:
            stand_pat = evaluate_board(state, stone)
            eval_cache[zh] = stand_pat
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        if qs_depth <= 0:
            return stand_pat

        # Only search forcing moves.
        threats = self._get_threat_moves(state, stone)
        if not threats:
            return stand_pat

        for move in threats:
            state.make(move.row, move.col, stone)
            try:
                if _check_win(state, move):
                    return 100_000.0

                score = -self._quiesce(
                    state,
                    -beta,
                    -alpha,
                    stone.opponent(),
                    tt,
                    qs_depth - 1,
                    eval_cache,
                )
            finally:
                state.unmake()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

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
                            tt,
                            _MAX_QS_DEPTH,
                            eval_cache,
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
        futility_margin = self.early_game_futility_margin if is_early else _FUTILITY_MARGIN
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
        elif depth >= 3:
            candidates = self._order_moves(state, candidates, stone, depth, killers, tt_move, history, eval_cache)

        # Cap branching factor to avoid combinatorial explosion.
        cap = self.early_game_max_candidates if occ_count <= self.early_game_threshold else _MAX_CANDIDATES
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
                        tt,
                        _MAX_QS_DEPTH,
                        eval_cache,
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
