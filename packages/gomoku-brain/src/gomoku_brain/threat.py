"""Threat-based brain — VCF (Victory by Continuous Fours) and VCT (Victory by Continuous Threats).

VCF searches only four-threats (forcing single-cell responses), enabling very
deep searches.  VCT extends this to also consider live-three threats, where the
defender has more responses but a forced win can still be proven.

When neither finds a forced win the brain falls back to alpha-beta search.
"""

from __future__ import annotations

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.zobrist import hash_board

# Cache entry: (proven, depth_searched, attack_move)
# * proven=True  -> winning attack found within ``depth_searched`` plies.
#                   Valid whenever the query depth is >= depth_searched.
# * proven=False -> no winning sequence within ``depth_searched`` plies.
#                   Valid whenever the query depth is <= depth_searched.
_CacheEntry = tuple[bool, int, "Move | None"]
_ThreatCache = dict[int, _CacheEntry]

# ---------------------------------------------------------------------------
# Direction vectors (horizontal, vertical, two diagonals)
# ---------------------------------------------------------------------------
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# ---------------------------------------------------------------------------
# Threat detection helpers
# ---------------------------------------------------------------------------


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


def _find_four_threats(board: Board, stone: Stone) -> list[tuple[Move, Move | None]]:
    """Find all moves that create a four-in-a-row for ``stone``.

    Returns a list of ``(attack_move, defence_move | None)``.
    ``defence_move`` is the single cell the opponent *must* play to block, or
    ``None`` when the four is a *double-four* (two open ends → unblockable,
    i.e. immediate win).

    Detects:
    - Straight fours: placing a stone extends three-in-a-row to four.
    - Gap fours: placing a stone fills a gap inside a would-be five
      (e.g. XX_XX pattern).
    """
    # Per-move aggregation: a single move can generate multiple four-threats
    # across different directions.  If it produces an unblockable (five or
    # live-four) in any direction, it's unblockable overall.  If it produces
    # two or more distinct dead-fours, the defender can only block one — it's
    # effectively unblockable (classic double-four).
    #
    # ``_line_cells`` scans outward from (r, c) and never reads the centre,
    # so it is invariant to whether the stone has actually been placed there
    # — the pre-move board gives the same answer and saves one full board
    # copy per candidate.
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
            # Two or more dead-fours from the same cell — opponent can only
            # block one, so this is effectively unblockable.
            threats.append((move, None))
        else:
            threats.append((move, defences[0]))
    return threats


def _find_three_threats(board: Board, stone: Stone) -> list[tuple[Move, list[Move]]]:
    """Find moves that create a *live* three (open at both ends).

    Returns ``(attack_move, defence_moves)`` — the defence moves are the cells
    the opponent could play to neutralise the threat (the two open ends plus
    any adjacent cells that break the line).
    """
    # See _find_four_threats: _line_cells is invariant to whether the
    # hypothetical stone has been placed at (r, c), so we avoid the
    # per-candidate board copy.
    threats: list[tuple[Move, list[Move]]] = []
    candidates = get_candidate_moves(board, radius=2)

    for move in candidates:
        r, c = move.row, move.col
        for dr, dc in _DIRECTIONS:
            count, open_ends, end_cells = _line_cells(board, r, c, dr, dc, stone)
            total = count + 1

            if total == 3 and open_ends == 2:
                # Live three — both ends open ⇒ two natural defence cells
                defence = [Move(row=er, col=ec) for er, ec in end_cells]
                threats.append((move, defence))

    return threats


def _find_winning_move(board: Board, stone: Stone) -> Move | None:
    """Return a move that immediately completes five-in-a-row, or ``None``.

    Scans neighbours of each candidate cell in place: no board copies, no
    ``check_winner`` call.  Matches the ``check_winner`` freestyle rule
    (``count >= 5`` wins).
    """
    size = board.size
    for move in get_candidate_moves(board, radius=2):
        r, c = move.row, move.col
        for dr, dc in _DIRECTIONS:
            count = 1  # the stone we would place
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


# ---------------------------------------------------------------------------
# VCF search
# ---------------------------------------------------------------------------


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
    resolved once and reused.  Callers who wish to share a cache across
    multiple related queries (e.g. VCT's inner _vcf calls) can pass an
    explicit dict.
    """
    if depth <= 0:
        return None
    if cache is None:
        cache = {}

    zh = hash_board(board, stone)
    entry = cache.get(zh)
    if entry is not None:
        proven, stored_depth, stored_move = entry
        if proven and stored_depth <= depth:
            return stored_move
        if not proven and stored_depth >= depth:
            return None

    if is_attacker:
        threats = _find_four_threats(board, stone)
        opp = stone.opponent()
        for attack, defence in threats:
            # Check immediate win
            trial = board.place_stone(attack, stone)
            result = check_winner(trial, attack)
            if result is not None and result.winner is stone:
                cache[zh] = (True, depth, attack)
                return attack

            # Opponent gets to respond — can they win first?
            if _find_winning_move(trial, opp) is not None:
                continue

            if defence is None:
                # Unblockable four (live-four or five) → win
                cache[zh] = (True, depth, attack)
                return attack

            # Before assuming the defender is forced to block, verify the
            # defender cannot instead play an unblockable four (live-four)
            # elsewhere.  Without this check, VCF can return false positives
            # when the opponent has a counter-live-four refutation.
            opp_threats = _find_four_threats(trial, opp)
            if any(d is None for _, d in opp_threats):
                continue

            # Defender is forced to block
            trial2 = trial.place_stone(defence, opp)

            # Recurse: attacker tries another four-threat
            if _vcf(trial2, stone, depth - 2, is_attacker=True, cache=cache) is not None:
                cache[zh] = (True, depth, attack)
                return attack

    cache[zh] = (False, depth, None)
    return None


# ---------------------------------------------------------------------------
# VCT search
# ---------------------------------------------------------------------------


def _vct_defence_moves(board: Board, attack_move: Move, stone: Stone) -> list[Move]:
    """Compute the full defence set for a three-threat played by *stone*.

    Includes the two open-end cells of the threat **plus** any move that
    creates a four-threat for the *opponent* (counter-threats).  This prevents
    the VCT from declaring false forced wins when the opponent can create
    counter-threats from unexpected cells (e.g. the K8 bug).
    """
    opp = stone.opponent()
    trial = board.place_stone(attack_move, stone)

    # Start with the natural blocking cells (open ends of the live three)
    three_threats = _find_three_threats(board, stone)
    end_cells: set[tuple[int, int]] = set()
    for atk, defs in three_threats:
        if atk == attack_move:
            for d in defs:
                end_cells.add((d.row, d.col))

    # Add any candidate move that would create a four-threat for the opponent
    for candidate in get_candidate_moves(trial, radius=2):
        if (candidate.row, candidate.col) in end_cells:
            continue
        cand_board = trial.place_stone(candidate, opp)
        opp_fours = _find_four_threats(cand_board, opp)
        if opp_fours:
            end_cells.add((candidate.row, candidate.col))

    return [Move(row=r, col=c) for r, c in sorted(end_cells)]


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

    zh = hash_board(board, stone)
    entry = vct_cache.get(zh)
    if entry is not None:
        proven, stored_depth, stored_move = entry
        if proven and stored_depth <= depth:
            return stored_move
        if not proven and stored_depth >= depth:
            return None

    if is_attacker:
        # 1) Try VCF first (four-threats are stronger)
        vcf_result = _vcf(board, stone, depth, is_attacker=True, cache=vcf_cache)
        if vcf_result is not None:
            vct_cache[zh] = (True, depth, vcf_result)
            return vcf_result

        # 2) Try three-threats
        three_threats = _find_three_threats(board, stone)
        for attack, _naive_defences in three_threats:
            trial = board.place_stone(attack, stone)

            # Check if attack itself wins
            result = check_winner(trial, attack)
            if result is not None and result.winner is stone:
                vct_cache[zh] = (True, depth, attack)
                return attack

            # Opponent gets to respond — can they win first?
            if _find_winning_move(trial, stone.opponent()) is not None:
                continue

            # Compute full defence set (end cells + counter-threat moves)
            defence_moves = _vct_defence_moves(board, attack, stone)

            # Defender responds: must refute ALL defence moves to prove win
            all_refuted = True
            if not defence_moves:
                all_refuted = False
            else:
                for defence in defence_moves:
                    if not trial.is_valid_move(defence):
                        continue
                    trial2 = trial.place_stone(defence, stone.opponent())

                    # Defender may also play a counter four-threat
                    opponent_fours = _find_four_threats(trial2, stone.opponent())
                    if opponent_fours:
                        # An unblockable counter (live-four / five) refutes
                        # the attack outright.
                        if any(d is None for _, d in opponent_fours):
                            all_refuted = False
                            break

                        # Two or more *independent* counter dead-fours mean
                        # the attacker can only neutralise one — the defender
                        # presses the other.  Treat as a refutation.
                        unique_blocks = {
                            (m.row, m.col) for _, m in opponent_fours if m is not None
                        }
                        if len(unique_blocks) >= 2:
                            all_refuted = False
                            break

                        # Exactly one dead-four: attacker blocks it, then
                        # recurse.  (Old code assumed the attacker has to
                        # block every listed counter sequentially with a
                        # per-counter depth_cost — that was both unsound
                        # and depth-inflated.)
                        _, block_move = opponent_fours[0]
                        if block_move is None or not trial2.is_valid_move(block_move):
                            all_refuted = False
                            break
                        board_after_block = trial2.place_stone(block_move, stone)
                        if (
                            _vct(
                                board_after_block,
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
                    elif (
                        _vct(
                            trial2,
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

            if all_refuted:
                vct_cache[zh] = (True, depth, attack)
                return attack

    vct_cache[zh] = (False, depth, None)
    return None


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
        with logfire.span(
            "brain.threat",
            vcf_depth=self.vcf_depth,
            vct_depth=self.vct_depth,
            threat_search=self.threat_search,
            pns_nodes=self.pns_nodes,
            stone=str(stone),
        ):
            import time as _time

            from gomoku.diagnostics import ThreatInfo, current_move_diagnostics

            diag = current_move_diagnostics()
            _t0 = _time.monotonic()

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
        """
        opp = stone.opponent()

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

        Test every candidate: does our move break the opponent's threat
        sequence?  Among breakers, pick the positionally strongest.
        Falls back to blocking the threat start cell.
        """
        from gomoku_brain.evaluate import evaluate_board

        opp = stone.opponent()
        candidates = get_candidate_moves(board, radius=2)

        breakers: list[Move] = []
        for move in candidates:
            trial = board.place_stone(move, stone)
            # Check VCF first (fast)
            if _vcf(trial, opp, self.vcf_depth, is_attacker=True) is not None:
                continue
            # Then check VCT/PNS
            if self.threat_search == "pns":
                from gomoku_brain.pns import pns_threat_search

                opp_result = pns_threat_search(
                    trial,
                    opp,
                    max_nodes=self.pns_nodes // 2,
                    max_time_s=self.pns_time_s / 2,
                )
                if opp_result is not None:
                    continue
            else:
                if _vct(trial, opp, self.vct_depth, is_attacker=True) is not None:
                    continue
            breakers.append(move)

        if not breakers:
            logfire.info(
                "No VCT breaker found, blocking opponent threat at: {move}",
                move=str(opp_threat_move),
            )
            if board.is_valid_move(opp_threat_move):
                return opp_threat_move
            return self._fallback.get_move(board, stone)

        if len(breakers) == 1:
            logfire.info("Single VCT breaker: {move}", move=str(breakers[0]))
            return breakers[0]

        logfire.info(
            "VCT-aware search: {n} breakers, picking best positionally",
            n=len(breakers),
        )
        best_score = float("-inf")
        best_move = breakers[0]
        for move in breakers:
            trial = board.place_stone(move, stone)
            score = evaluate_board(trial, stone)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

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
        best_score = float("-inf")
        best_move = forks[0]
        for move in forks:
            trial = board.place_stone(move, stone)
            score = evaluate_board(trial, stone)
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

        Instead of unconditionally blocking at *opp_vcf_move*, we check
        every candidate: does our move break the opponent's VCF?  Among
        the VCF-breakers we let alpha-beta choose the positionally best
        one.  If nothing breaks it we fall back to the direct block.
        """
        from gomoku_brain.evaluate import evaluate_board

        opp = stone.opponent()
        candidates = get_candidate_moves(board, radius=2)

        # Partition candidates into those that break opp VCF and those that don't.
        breakers: list[Move] = []
        for move in candidates:
            trial = board.place_stone(move, stone)
            if _vcf(trial, opp, self.vcf_depth, is_attacker=True) is None:
                breakers.append(move)

        if not breakers:
            # Nothing breaks the VCF — block at the threatened cell as last resort.
            logfire.info(
                "No VCF breaker found, blocking opponent VCF at: {move}",
                move=str(opp_vcf_move),
            )
            if board.is_valid_move(opp_vcf_move):
                return opp_vcf_move
            return self._fallback.get_move(board, stone)

        if len(breakers) == 1:
            logfire.info("Single VCF breaker: {move}", move=str(breakers[0]))
            return breakers[0]

        # Multiple breakers — use alpha-beta among them for positional quality.
        logfire.info(
            "VCF-aware search: {n} breakers, picking best via alpha-beta",
            n=len(breakers),
        )
        best_score = float("-inf")
        best_move = breakers[0]
        for move in breakers:
            trial = board.place_stone(move, stone)
            score = evaluate_board(trial, stone)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
