"""Alpha-beta brain — negamax with alpha-beta pruning, move ordering, and killer heuristic."""

from __future__ import annotations

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.zobrist import (
    BoundType,
    TranspositionTable,
    hash_board,
    incremental_hash,
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

# Futility pruning margin: if static eval + margin < alpha at a frontier node,
# the move is unlikely to raise alpha.
_FUTILITY_MARGIN = 1500


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
    ) -> None:
        self.depth = depth
        self.vcf_depth = vcf_depth
        self.qs_vcf_depth = qs_vcf_depth
        self.threat_ordering = threat_ordering
        self.threat_extensions = threat_extensions

    @property
    def name(self) -> str:
        return "alphabeta"

    def get_move(self, board: Board, stone: Stone) -> Move:
        with logfire.span("brain.alphabeta", depth=self.depth, vcf_depth=self.vcf_depth, stone=str(stone)):
            # -- Root-level VCF pre-check ------------------------------------
            if self.vcf_depth > 0:
                vcf_move = self._root_vcf(board, stone)
                if vcf_move is not None:
                    return vcf_move

            tt = TranspositionTable()
            zh = hash_board(board, stone)
            best_move: Move | None = None
            # History table shared across all ID iterations so it accumulates.
            history: HistoryTable = {}
            eval_cache: EvalCache = {}
            prev_score: float | None = None

            # Iterative deepening: search at increasing depths so that each
            # pass seeds the TT with best-move info for the next, deeper pass.
            for d in range(1, self.depth + 1):
                # Fresh killers each iteration — they are depth-relative.
                killers: dict[int, list[Move]] = {}

                # Aspiration window: after the first iteration, search with a
                # narrow window around the previous score.  Widen on failure.
                if prev_score is not None and abs(prev_score) < 90_000:
                    a = prev_score - _ASPIRATION_WINDOW
                    b = prev_score + _ASPIRATION_WINDOW
                    score, move = self._alphabeta(
                        board, d, a, b, stone, killers, tt, zh, history, eval_cache, self.threat_extensions,
                    )
                    # Re-search on fail-low or fail-high.
                    if score <= a or score >= b:
                        score, move = self._alphabeta(
                            board, d, float("-inf"), float("inf"), stone, killers, tt, zh, history, eval_cache,
                            self.threat_extensions,
                        )
                else:
                    score, move = self._alphabeta(
                        board, d, float("-inf"), float("inf"), stone, killers, tt, zh, history, eval_cache,
                        self.threat_extensions,
                    )

                prev_score = score
                if move is not None:
                    best_move = move
                logfire.info(
                    "ID depth {d}/{max_d}: score={score}, move={move}, tt_size={tt_size}",
                    d=d, max_d=self.depth, score=score, move=str(move), tt_size=len(tt),
                )
                # Early exit on proven win.
                if score >= 100_000:
                    break

            assert best_move is not None
            return best_move

    # -- Instant-win / forced-block pre-check --------------------------------

    @staticmethod
    def _find_urgent_move(
        board: Board,
        candidates: list[Move],
        stone: Stone,
    ) -> Move | None:
        """Return an immediate winning move for *stone*, or a move that blocks
        the opponent's immediate win.  Returns ``None`` if no urgent move."""
        opponent = stone.opponent()
        block: Move | None = None
        for move in candidates:
            # Check our win
            new_board = board.place_stone(move, stone)
            if check_winner(new_board, move) is not None:
                return move
            # Check opponent's win (only need to find first one)
            if block is None:
                opp_board = board.place_stone(move, opponent)
                if check_winner(opp_board, move) is not None:
                    block = move
        return block

    # -- Root-level VCF pre-check -------------------------------------------

    def _root_vcf(self, board: Board, stone: Stone) -> Move | None:
        """Run VCF at the root for *stone* and its opponent.

        Returns immediately if *stone* has a forced VCF win.
        If the opponent has a VCF, restricts candidates to moves that break
        it and returns the positionally best breaker.
        Returns ``None`` when VCF is irrelevant and normal search should run.
        """
        from gomoku_brain.threat import _vcf

        # 1) Do we have a forced win?
        our_vcf = _vcf(board, stone, self.vcf_depth, is_attacker=True)
        if our_vcf is not None:
            logfire.info("AlphaBeta VCF win: {move}", move=str(our_vcf))
            return our_vcf

        # 2) Does the opponent have a forced win we must disrupt?
        opp = stone.opponent()
        opp_vcf = _vcf(board, opp, self.vcf_depth, is_attacker=True)
        if opp_vcf is None:
            return None  # No VCF concern — proceed with normal search.

        logfire.info("Opponent has VCF starting at {move}, searching breakers", move=str(opp_vcf))

        candidates = get_candidate_moves(board, radius=2)
        breakers: list[Move] = []
        for move in candidates:
            trial = board.place_stone(move, stone)
            if _vcf(trial, opp, self.vcf_depth, is_attacker=True) is None:
                breakers.append(move)

        if not breakers:
            # Nothing breaks the VCF — block at the threatened cell as last resort.
            logfire.info("No VCF breaker, blocking at {move}", move=str(opp_vcf))
            if board.is_valid_move(opp_vcf):
                return opp_vcf
            return None  # Let normal search handle it.

        if len(breakers) == 1:
            logfire.info("Single VCF breaker: {move}", move=str(breakers[0]))
            return breakers[0]

        # Multiple breakers — pick via evaluation.
        logfire.info("VCF-aware: {n} breakers, picking best positionally", n=len(breakers))
        best_score = float("-inf")
        best_move = breakers[0]
        for move in breakers:
            trial = board.place_stone(move, stone)
            score = evaluate_board(trial, stone)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    # -- Move ordering -------------------------------------------------------

    def _order_moves(
        self,
        board: Board,
        candidates: list[Move],
        stone: Stone,
        depth: int,
        killers: dict[int, list[Move]],
        tt_move: Move | None = None,
        history: HistoryTable | None = None,
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
            opponent = stone.opponent()
            size = board.size
            threat_moves: list[Move] = []
            non_threat: list[Move] = []
            for move in rest:
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
                    threat_moves.append(move)
                else:
                    non_threat.append(move)
        else:
            threat_moves = []
            non_threat = rest

        # 4. Eval + history sorted remainder (non-threat moves only).
        stone_int = int(stone)
        scored = []
        for move in non_threat:
            new_board = board.place_stone(move, stone)
            score = evaluate_board(new_board, stone)
            if history:
                score += history.get((stone_int, move.row, move.col), 0)
            scored.append((score, move))
        scored.sort(key=lambda x: x[0], reverse=True)

        return promoted + threat_moves + [m for _, m in scored]

    # -- Quiescence search ---------------------------------------------------

    @staticmethod
    def _get_threat_moves(board: Board, stone: Stone) -> list[Move]:
        """Return moves that create a four-in-a-row for *stone* or block one for the opponent.

        Only considers candidates within radius 2 of occupied stones.
        This is a lightweight detector for quiescence — not exhaustive.
        """
        opponent = stone.opponent()
        size = board.size
        threats: list[Move] = []
        seen: set[tuple[int, int]] = set()

        for move in get_candidate_moves(board):
            if (move.row, move.col) in seen:
                continue
            is_threat = False
            # Check if placing *stone* here creates a four or five.
            for dr, dc in _DIRECTIONS:
                count = 1  # the placed stone itself
                for sign in (1, -1):
                    r, c = move.row + dr * sign, move.col + dc * sign
                    while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
                        count += 1
                        r += dr * sign
                        c += dc * sign
                if count >= 4:
                    is_threat = True
                    break
            # Check if placing *opponent* here creates a four or five (need to block).
            if not is_threat:
                for dr, dc in _DIRECTIONS:
                    count = 1
                    for sign in (1, -1):
                        r, c = move.row + dr * sign, move.col + dc * sign
                        while 0 <= r < size and 0 <= c < size and board.at(r, c) is opponent:
                            count += 1
                            r += dr * sign
                            c += dc * sign
                    if count >= 4:
                        is_threat = True
                        break
            if is_threat:
                seen.add((move.row, move.col))
                threats.append(move)
        return threats

    def _quiesce(
        self,
        board: Board,
        alpha: float,
        beta: float,
        stone: Stone,
        tt: TranspositionTable,
        zh: int,
        qs_depth: int,
        eval_cache: EvalCache,
    ) -> float:
        """Quiescence search: extend leaves for forcing (four-threat) moves."""
        # -- VCF probe: detect multi-move forced wins that the one-ply
        #    threat loop would miss.
        if self.qs_vcf_depth > 0:
            from gomoku_brain.threat import _vcf

            if _vcf(board, stone, self.qs_vcf_depth, is_attacker=True) is not None:
                return 100_000.0

        # Standing pat — static evaluation as a lower bound.
        if zh in eval_cache:
            stand_pat = eval_cache[zh]
        else:
            stand_pat = evaluate_board(board, stone)
            eval_cache[zh] = stand_pat
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        if qs_depth <= 0:
            return stand_pat

        # Only search forcing moves.
        threats = self._get_threat_moves(board, stone)
        if not threats:
            return stand_pat

        for move in threats:
            new_board = board.place_stone(move, stone)
            child_zh = incremental_hash(zh, move.row, move.col, stone)

            result = check_winner(new_board, move)
            if result is not None:
                return 100_000.0 if result.winner is stone else -100_000.0

            score = -self._quiesce(
                new_board, -beta, -alpha, stone.opponent(), tt, child_zh, qs_depth - 1, eval_cache,
            )

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
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        stone: Stone,
        killers: dict[int, list[Move]],
        tt: TranspositionTable,
        zh: int,
        history: HistoryTable,
        eval_cache: EvalCache,
        ext_left: int = 0,
    ) -> tuple[float, Move | None]:
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

        # -- Candidate generation --------------------------------------------
        candidates = get_candidate_moves(board)
        if not candidates:
            return 0.0, None

        # Fast path: check for an instant win or a forced block before the
        # expensive evaluation-based sort.  When urgent is found, play it
        # immediately (winning) or promote it (blocking).
        urgent = self._find_urgent_move(board, candidates, stone)
        if urgent is not None:
            new_board = board.place_stone(urgent, stone)
            result = check_winner(new_board, urgent)
            if result is not None and result.winner is stone:
                return 100_000.0 + depth, urgent
            # It's a block — promote it to head but still search alternatives
            candidates = [urgent, *(m for m in candidates if m != urgent)]
        elif depth >= 3:
            candidates = self._order_moves(board, candidates, stone, depth, killers, tt_move, history)

        # Cap branching factor to avoid combinatorial explosion in early game.
        candidates = candidates[:_MAX_CANDIDATES]

        best_score = float("-inf")
        best_move: Move | None = None

        # Lazy static eval for futility pruning (computed at most once).
        futility_eval: float | None = None

        for i, move in enumerate(candidates):
            new_board = board.place_stone(move, stone)
            child_zh = incremental_hash(zh, move.row, move.col, stone)

            result = check_winner(new_board, move)
            if result is not None:
                score = 100_000.0 + depth if result.winner is stone else -(100_000.0 + depth)
                # Terminal — store exact and return.
                tt.store(zh, depth, score, BoundType.EXACT, move)
                return score, move

            # -- Futility pruning at frontier (depth 2) ----------------------
            # If the position's static eval + margin can't reach alpha, skip
            # non-first moves (first move is always searched fully for safety).
            if depth == 2 and i > 0:
                if futility_eval is None:
                    if zh in eval_cache:
                        futility_eval = eval_cache[zh]
                    else:
                        futility_eval = evaluate_board(board, stone)
                        eval_cache[zh] = futility_eval
                if futility_eval + _FUTILITY_MARGIN <= alpha:
                    continue

            # -- Threat extension: extend by 1 ply on forcing moves ----------
            child_depth = depth - 1
            child_ext = ext_left
            if ext_left > 0 and depth >= 2:
                is_four = False
                size = board.size
                for dr, dc in _DIRECTIONS:
                    cnt = 1
                    for sign in (1, -1):
                        r, c = move.row + dr * sign, move.col + dc * sign
                        while 0 <= r < size and 0 <= c < size and new_board.at(r, c) is stone:
                            cnt += 1
                            r += dr * sign
                            c += dc * sign
                    if cnt >= 4:
                        is_four = True
                        break
                if not is_four:
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
                            is_four = True
                            break
                if is_four:
                    child_depth = depth  # extend: don't decrement
                    child_ext = ext_left - 1

            if depth <= 1:
                score = self._quiesce(
                    new_board, -beta, -alpha, stone.opponent(), tt, child_zh, _MAX_QS_DEPTH, eval_cache,
                )
                score = -score
            elif i == 0:
                # First move (expected best): search with full window.
                opp_score, _ = self._alphabeta(
                    new_board, child_depth, -beta, -alpha,
                    stone.opponent(), killers, tt, child_zh, history, eval_cache, child_ext,
                )
                score = -opp_score
            else:
                # PVS zero-window scout.
                opp_score, _ = self._alphabeta(
                    new_board, child_depth, -alpha - 1, -alpha,
                    stone.opponent(), killers, tt, child_zh, history, eval_cache, child_ext,
                )
                score = -opp_score
                # Re-search with full window if scout fails high.
                if alpha < score < beta:
                    opp_score, _ = self._alphabeta(
                        new_board, child_depth, -beta, -score,
                        stone.opponent(), killers, tt, child_zh, history, eval_cache, child_ext,
                    )
                    score = -opp_score

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
