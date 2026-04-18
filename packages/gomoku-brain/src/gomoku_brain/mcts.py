"""MCTS brain — Monte Carlo Tree Search with PUCT and heuristic priors.

This module implements an AlphaGo Zero-inspired tree search adapted for
Gomoku **without** neural networks.  Key ideas borrowed from the book
*Deep Learning and the Game of Go* (Manning, 2019):

* **PUCT formula** (Predictor + UCT) replaces plain UCB1.  Each branch
  carries a *prior probability* so the search focuses on tactically
  promising moves first.  Formula:  ``Q + c · P · √N / (n + 1)``.
* **Heuristic priors** - a lightweight raw-grid pattern scorer assigns
  prior probabilities based on consecutive-stone / open-end patterns,
  standing in for the neural-network prior used in AGZ.
* **Dirichlet noise** at the root ensures exploration breadth and
  prevents the search from developing blind spots (AGZ §14.4).
* **Heuristic leaf evaluation** replaces random rollouts.  Instead of
  playing random moves to a terminal state, we score the leaf position
  using the pattern table and map the net advantage through a sigmoid.
  This eliminates the O(depth) rollout cost per iteration, giving a
  ~30-60x speedup on sparse early-game boards.
"""

from __future__ import annotations

import math
import random as _random

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.threat import _find_four_threats, _vcf

# ---------------------------------------------------------------------------
# Directions for win-checking / pattern scoring on raw grids
# ---------------------------------------------------------------------------
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# ---------------------------------------------------------------------------
# Lightweight helpers operating on raw int grids
# ---------------------------------------------------------------------------


def _fast_check_winner(
    grid: list[list[int]],
    size: int,
    row: int,
    col: int,
    stone_val: int,
    win_length: int,
) -> bool:
    """Check whether *stone_val* at (row, col) creates a winning line."""
    for dr, dc in _DIRECTIONS:
        count = 1
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < size and 0 <= c < size and grid[r][c] == stone_val:
                count += 1
                r += dr * sign
                c += dc * sign
        if count >= win_length:
            return True
    return False


def _find_winning_move(
    grid: list[list[int]],
    size: int,
    candidates: list[tuple[int, int]],
    stone_val: int,
    win_length: int,
) -> tuple[int, int] | None:
    """Return a candidate that wins immediately, or None."""
    for r, c in candidates:
        grid[r][c] = stone_val
        won = _fast_check_winner(grid, size, r, c, stone_val, win_length)
        grid[r][c] = 0
        if won:
            return (r, c)
    return None


def _candidate_moves(
    occupied: set[tuple[int, int]],
    size: int,
    radius: int = 2,
) -> list[tuple[int, int]]:
    """Empty cells within *radius* of any occupied cell (raw-grid version)."""
    if not occupied:
        center = size // 2
        return [(center, center)]
    cands: set[tuple[int, int]] = set()
    for r, c in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in occupied:
                    cands.add((nr, nc))
    return list(cands)


# ---------------------------------------------------------------------------
# Heuristic prior scoring on raw int grids
# ---------------------------------------------------------------------------

# (consecutive, open_ends) → raw score  (mirrors evaluate.py but works on
# raw grids so we avoid constructing Board objects in the hot path).
_PRIOR_PATTERN_SCORES: dict[tuple[int, int], float] = {
    (4, 2): 10_000.0,  # live-4
    (4, 1): 1_000.0,  # dead-4
    (3, 2): 1_000.0,  # live-3
    (3, 1): 100.0,  # dead-3
    (2, 2): 100.0,  # live-2
    (2, 1): 10.0,  # dead-2
    (1, 2): 10.0,
    (1, 1): 1.0,
}


def _score_move_prior(
    grid: list[list[int]],
    size: int,
    row: int,
    col: int,
    stone_val: int,
    opp_val: int,
) -> float:
    """Score how tactically promising placing *stone_val* at (row, col) is.

    Considers both our own patterns (offensive) and opponent patterns we
    would disrupt (defensive), weighted 1.0 and 0.8 respectively.
    """
    score = 1.0  # baseline so no move gets zero prior

    for dr, dc in _DIRECTIONS:
        # --- offensive: how many of *our* stones line up through (row,col)
        count = 0
        open_ends = 0
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < size and 0 <= c < size and grid[r][c] == stone_val:
                count += 1
                r += dr * sign
                c += dc * sign
            if 0 <= r < size and 0 <= c < size and grid[r][c] == 0:
                open_ends += 1
        score += _PRIOR_PATTERN_SCORES.get((count, open_ends), 0.0)

        # --- defensive: how many *opponent* stones line up through (row,col)
        count = 0
        open_ends = 0
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < size and 0 <= c < size and grid[r][c] == opp_val:
                count += 1
                r += dr * sign
                c += dc * sign
            if 0 <= r < size and 0 <= c < size and grid[r][c] == 0:
                open_ends += 1
        score += 0.8 * _PRIOR_PATTERN_SCORES.get((count, open_ends), 0.0)

    return score


def _compute_priors(
    grid: list[list[int]],
    size: int,
    candidates: list[tuple[int, int]],
    stone_val: int,
    opp_val: int,
) -> list[float]:
    """Return normalised prior probabilities for *candidates*."""
    raw = [_score_move_prior(grid, size, r, c, stone_val, opp_val) for r, c in candidates]
    total = sum(raw)
    if total == 0:
        uniform = 1.0 / len(candidates)
        return [uniform] * len(candidates)
    return [s / total for s in raw]


def _apply_dirichlet_noise(
    priors: list[float],
    alpha: float,
    noise_weight: float,
) -> list[float]:
    """Mix Dirichlet noise into *priors*.  AGZ §14.4.

    ``new_prior = (1 - noise_weight) * prior + noise_weight * noise``
    """
    # Use the Gamma-function trick to sample Dirichlet without numpy:
    # Dir(alpha) ~ normalise(Gamma(alpha, 1) for each dim).
    noise = [_random.gammavariate(alpha, 1.0) for _ in priors]
    noise_total = sum(noise) or 1.0
    noise = [n / noise_total for n in noise]
    return [(1.0 - noise_weight) * p + noise_weight * n for p, n in zip(priors, noise, strict=True)]


# ---------------------------------------------------------------------------
# Heuristic leaf evaluation (replaces random rollout)
# ---------------------------------------------------------------------------


def _evaluate_position(
    grid: list[list[int]],
    size: int,
    occupied: set[tuple[int, int]],
    root_stone_val: int,
) -> float:
    """Heuristic board evaluation → value in [0, 1] for *root_stone_val*.

    Scans every consecutive-stone line on the board, scores patterns using
    ``_PRIOR_PATTERN_SCORES``, and maps the net advantage through a sigmoid.
    """
    score = 0.0
    for r, c in occupied:
        sv = grid[r][c]
        if sv == 0:
            continue
        for dr, dc in _DIRECTIONS:
            # Only process lines starting at this cell (avoid double-counting)
            pr, pc = r - dr, c - dc
            if 0 <= pr < size and 0 <= pc < size and grid[pr][pc] == sv:
                continue

            # Count consecutive stones in the positive direction
            count = 1
            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == sv:
                count += 1
                nr += dr
                nc += dc

            # Count open ends
            open_ends = 0
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                open_ends += 1
            if 0 <= pr < size and 0 <= pc < size and grid[pr][pc] == 0:
                open_ends += 1

            pattern_score = _PRIOR_PATTERN_SCORES.get((count, open_ends), 0.0)
            if sv == root_stone_val:
                score += pattern_score
            else:
                score -= 0.8 * pattern_score

    # Sigmoid: k=1000 so a live-3 advantage (≈1000 pts) ≈ 0.73.
    clamped = max(-10.0, min(10.0, score / 1000.0))
    return 1.0 / (1.0 + math.exp(-clamped))


# ---------------------------------------------------------------------------
# MCTS Node — uses raw (row, col) tuples, not Pydantic Move objects
# ---------------------------------------------------------------------------


class _Node:
    """A node in the MCTS tree."""

    __slots__ = (
        "_untried",
        "_untried_priors",
        "children",
        "move",
        "parent",
        "prior",
        "stone",
        "visits",
        "wins",
    )

    def __init__(
        self,
        move: tuple[int, int] | None,
        stone: Stone,
        parent: _Node | None,
        untried: list[tuple[int, int]],
        prior: float = 0.0,
        untried_priors: list[float] | None = None,
    ) -> None:
        self.move = move
        self.stone = stone  # stone that was *placed* to reach this node
        self.parent = parent
        self.prior = prior
        self.children: list[_Node] = []
        self.wins: float = 0.0
        self.visits: int = 0
        self._untried = untried
        # Per-move priors aligned with _untried; uniform if not supplied
        if untried_priors is not None:
            self._untried_priors = untried_priors
        else:
            n = len(untried)
            self._untried_priors = [1.0 / n] * n if n else []

    # -- PUCT (Predictor + UCT) -----------------------------------------------

    def puct(self, exploration: float) -> float:
        """AGZ-style PUCT score: Q + c · P · √N / (n + 1)."""
        q = self.wins / self.visits if self.visits else 0.0
        parent_visits = self.parent.visits if self.parent else 1
        return q + exploration * self.prior * math.sqrt(parent_visits) / (self.visits + 1)

    # -- Selection -----------------------------------------------------------

    def best_child(self, exploration: float) -> _Node:
        return max(self.children, key=lambda c: c.puct(exploration))

    # -- Expansion -----------------------------------------------------------

    def has_untried(self) -> bool:
        return len(self._untried) > 0

    def pop_untried(self) -> tuple[tuple[int, int], float]:
        """Remove and return a random untried move together with its prior."""
        idx = _random.randrange(len(self._untried))
        # Swap-and-pop for O(1) removal (both lists in sync)
        self._untried[idx], self._untried[-1] = self._untried[-1], self._untried[idx]
        self._untried_priors[idx], self._untried_priors[-1] = (
            self._untried_priors[-1],
            self._untried_priors[idx],
        )
        prior = self._untried_priors.pop()
        move = self._untried.pop()
        return move, prior

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# MCTS Brain
# ---------------------------------------------------------------------------


class MCTSBrain:
    """MoveProvider using Monte Carlo Tree Search with UCT."""

    def __init__(
        self,
        iterations: int = 10_000,
        exploration: float = 2.0,
        dirichlet_alpha: float = 0.03,
        dirichlet_weight: float = 0.25,
        vcf_depth: int = 10,
    ) -> None:
        self.iterations = iterations
        self.exploration = exploration
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.vcf_depth = vcf_depth

    @property
    def name(self) -> str:
        return "mcts"

    # -- Public interface ----------------------------------------------------

    def get_move(self, board: Board, stone: Stone) -> Move:
        with logfire.span(
            "brain.mcts",
            iterations=self.iterations,
            exploration=self.exploration,
            stone=str(stone),
        ):
            import time as _time

            from gomoku.diagnostics import current_move_diagnostics

            diag = current_move_diagnostics()
            _t0 = _time.monotonic()

            candidates = get_candidate_moves(board)
            if len(candidates) == 1:
                if diag is not None:
                    diag.decision_path = "single_candidate"
                    diag.chosen_move = candidates[0]
                return candidates[0]

            # Convert to raw grid once — all MCTS work uses this
            size = board.size
            board_grid: list[list[int]] = [[int(board.grid[r][c]) for c in range(size)] for r in range(size)]
            board_occupied: frozenset[tuple[int, int]] = frozenset(board.occupied)
            stone_val = int(stone)

            # Instant-win check on raw grid (no Board.place_stone overhead)
            for move in candidates:
                r, c = move.row, move.col
                board_grid[r][c] = stone_val
                if _fast_check_winner(board_grid, size, r, c, stone_val, 5):
                    board_grid[r][c] = 0
                    if diag is not None:
                        diag.decision_path = "instant_win"
                        diag.chosen_move = move
                    return move
                board_grid[r][c] = 0

            # Instant-block: if opponent wins in 1 ply, we must block
            opp_val = int(stone.opponent())
            raw_cands = [(m.row, m.col) for m in candidates]
            opp_winning = _find_winning_move(board_grid, size, raw_cands, opp_val, 5)
            if opp_winning is not None:
                result_move = Move(row=opp_winning[0], col=opp_winning[1])
                if diag is not None:
                    diag.decision_path = "instant_block"
                    diag.chosen_move = result_move
                return result_move

            # VCF safety gate: if opponent has a proven forced win,
            # restrict MCTS candidates to blocking cells only
            vcf_gated = False
            if self.vcf_depth > 0:
                opp = stone.opponent()
                vcf_move = _vcf(board, opp, self.vcf_depth, is_attacker=True)
                if vcf_move is not None:
                    opp_fours = _find_four_threats(board, opp)
                    blocking: set[tuple[int, int]] = {(vcf_move.row, vcf_move.col)}
                    for attack, defence in opp_fours:
                        blocking.add((attack.row, attack.col))
                        if defence is not None:
                            blocking.add((defence.row, defence.col))
                    restricted = [m for m in candidates if (m.row, m.col) in blocking]
                    if restricted:
                        logfire.info(
                            "VCF safety gate: opponent has forced win, restricting to {n} blocking candidates",
                            n=len(restricted),
                        )
                        candidates = restricted
                        vcf_gated = True
                    if diag is not None:
                        diag.opp_threats = diag.opp_threats.model_copy(update={"vcf_result": "win"})
                        diag.notes.append(
                            f"Opponent VCF at {vcf_move}, restricted to {len(restricted)} blocking candidates"
                            if vcf_gated
                            else f"Opponent VCF at {vcf_move}, no blocking candidates found"
                        )

            root_untried = [(m.row, m.col) for m in candidates]
            root_priors = _compute_priors(
                board_grid,
                size,
                root_untried,
                stone_val,
                opp_val,
            )
            # Dirichlet noise for exploration breadth (AGZ §14.4)
            if self.dirichlet_alpha > 0 and self.dirichlet_weight > 0:
                root_priors = _apply_dirichlet_noise(
                    root_priors,
                    self.dirichlet_alpha,
                    self.dirichlet_weight,
                )
            root = _Node(
                move=None,
                stone=stone.opponent(),  # "opponent placed last" to reach root
                parent=None,
                untried=root_untried,
                untried_priors=root_priors,
            )

            for _ in range(self.iterations):
                self._run_iteration(root, board_grid, board_occupied, size, stone)

            # Choose the child with the most visits (most robust choice)
            best = max(root.children, key=lambda c: c.visits)
            assert best.move is not None
            br, bc = best.move
            logfire.info(
                "MCTS chose {move} ({visits}/{total} visits, win-rate {wr:.1%})",
                move=str(Move(row=br, col=bc)),
                visits=best.visits,
                total=root.visits,
                wr=best.wins / best.visits if best.visits else 0,
            )

            chosen = Move(row=br, col=bc)

            if diag is not None:
                from gomoku.diagnostics import CandidateInfo, SearchStats

                search_ms = int((_time.monotonic() - _t0) * 1000)
                diag.decision_path = diag.decision_path or ("mcts_vcf_gated" if vcf_gated else "mcts")
                diag.chosen_move = chosen
                diag.search_stats = SearchStats(
                    iterations=self.iterations,
                    nodes_searched=root.visits,
                    time_ms=search_ms,
                )
                # Top children by visits (most informative for MCTS).
                children_sorted = sorted(root.children, key=lambda c: c.visits, reverse=True)
                diag.top_candidates = [
                    CandidateInfo(
                        move=Move(row=c.move[0], col=c.move[1]),
                        score=round(c.wins / c.visits, 4) if c.visits else 0.0,
                        score_type="winrate",
                    )
                    for c in children_sorted[:10]
                ]
                if diag.top_candidates and diag.top_candidates[0].move != chosen:
                    diag.alternative_best = diag.top_candidates[0]
                diag.notes.append(
                    f"Best: {chosen} visits={best.visits}/{root.visits} wr={best.wins / best.visits:.1%}"
                    if best.visits
                    else f"Best: {chosen} visits=0"
                )

            return chosen

    # -- Core MCTS loop (operates entirely on raw int grids) -----------------

    def _run_iteration(
        self,
        root: _Node,
        board_grid: list[list[int]],
        board_occupied: frozenset[tuple[int, int]],
        size: int,
        root_stone: Stone,
    ) -> None:
        # Fresh mutable grid copy for this iteration (row[:] is ~50x faster
        # than copy.deepcopy used by Board.place_stone)
        grid = [row[:] for row in board_grid]
        occupied = set(board_occupied)

        node = root

        # 1. Selection — descend using UCB1, applying moves on the raw grid
        while not node.has_untried() and not node.is_leaf():
            node = node.best_child(self.exploration)
            r, c = node.move  # type: ignore[misc]
            grid[r][c] = int(node.stone)
            occupied.add((r, c))

        # 2. Expansion — add one child
        if node.has_untried():
            move_rc, move_prior = node.pop_untried()
            child_stone = root_stone if node.stone is not root_stone else root_stone.opponent()
            r, c = move_rc
            sv = int(child_stone)
            grid[r][c] = sv
            occupied.add((r, c))

            child_candidates = _candidate_moves(occupied, size)
            child_opp_val = int(child_stone.opponent())
            child_priors = _compute_priors(
                grid,
                size,
                child_candidates,
                child_opp_val,
                sv,
            )
            child = _Node(
                move=move_rc,
                stone=child_stone,
                parent=node,
                untried=child_candidates,
                prior=move_prior,
                untried_priors=child_priors,
            )
            node.children.append(child)
            node = child

            # Check if the expansion move wins immediately
            if _fast_check_winner(grid, size, r, c, sv, 5):
                value = 1.0 if child_stone is root_stone else 0.0
                self._backpropagate(node, value, root_stone)
                return

            # Decisive move: if opponent can win immediately, treat as loss
            opp_sv = int(child_stone.opponent())
            if _find_winning_move(grid, size, child_candidates, opp_sv, 5) is not None:
                value = 0.0 if child_stone is root_stone else 1.0
                self._backpropagate(node, value, root_stone)
                return

        # 3. Heuristic leaf evaluation (replaces random rollout)
        value = _evaluate_position(grid, size, occupied, int(root_stone))

        # 4. Backpropagation
        self._backpropagate(node, value, root_stone)

    # -- Backpropagation -----------------------------------------------------

    @staticmethod
    def _backpropagate(node: _Node, value: float, root_stone: Stone) -> None:
        """Walk up to root, updating visit counts and wins.

        *value* is in [0, 1] from *root_stone*'s perspective.
        """
        n: _Node | None = node
        while n is not None:
            n.visits += 1
            if n.stone is root_stone:
                n.wins += value
            else:
                n.wins += 1.0 - value
            n = n.parent
