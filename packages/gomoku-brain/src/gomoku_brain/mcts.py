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
  Skipped when a VCF safety gate has already restricted the root to a
  handful of forced-defence candidates.
* **Heuristic leaf evaluation** replaces random rollouts.  Instead of
  playing random moves to a terminal state, we score the leaf position
  with :func:`gomoku_brain.evaluate.evaluate_board` (shared with
  alpha-beta) and map the net advantage through a sigmoid.  This
  eliminates the O(depth) rollout cost per iteration and reuses the
  single static evaluator across brains.

The search operates on a mutable :class:`SearchState` with O(1)
``make`` / ``unmake`` — each iteration walks down, optionally expands
one child, evaluates, and walks back up — with no per-iteration grid
or occupied-set copies.
"""

from __future__ import annotations

import math
import random as _random

import logfire
import numpy as np
from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.brain_base import brain_session
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat import classify_vcf_breakers, probe_immediate

# ---------------------------------------------------------------------------
# Heuristic prior scoring (JIT-backed, with Python fallback)
# ---------------------------------------------------------------------------


def _compute_priors(
    state: SearchState,
    candidates: list[tuple[int, int]],
    stone_val: int,
    opp_val: int,
) -> list[float]:
    """Return normalised MCTS prior probabilities for *candidates*.

    Delegates to the Numba ``compute_priors_grid`` kernel on the JIT path;
    falls back to a compact pure-Python scan otherwise.  The Python path
    is only hit when JIT evaluation is disabled via
    :class:`~gomoku.settings.GomokuSettings`.
    """
    if not candidates:
        return []

    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import (
            compute_priors_grid,
            get_dirs,
            get_pattern_table,
        )

        cand_arr = np.asarray(candidates, dtype=np.int32).reshape(-1, 2)
        result = compute_priors_grid(
            state.grid,
            state.size,
            cand_arr,
            stone_val,
            opp_val,
            get_dirs(),
            get_pattern_table(),
        )
        return [float(x) for x in result]

    from gomoku_brain.directions import DIRECTIONS

    size = state.size
    grid = state.grid
    raw = []
    for row, col in candidates:
        score = 1.0
        for dr, dc in DIRECTIONS:
            for scanned, weight in ((stone_val, 1.0), (opp_val, 0.8)):
                count = 0
                open_ends = 0
                for sign in (1, -1):
                    r, c = row + dr * sign, col + dc * sign
                    while 0 <= r < size and 0 <= c < size and grid[r, c] == scanned:
                        count += 1
                        r += dr * sign
                        c += dc * sign
                    if 0 <= r < size and 0 <= c < size and grid[r, c] == 0:
                        open_ends += 1
                score += weight * _PY_PATTERN_SCORES.get((count, open_ends), 0.0)
        raw.append(score)
    total = sum(raw)
    if total == 0:
        uniform = 1.0 / len(candidates)
        return [uniform] * len(candidates)
    return [s / total for s in raw]


# Pure-Python fallback table — mirrors ``evaluate_jit._PATTERN_TABLE`` except
# the hot JIT path uses the numpy table directly.
_PY_PATTERN_SCORES: dict[tuple[int, int], float] = {
    (4, 2): 10_000.0,
    (4, 1): 1_000.0,
    (3, 2): 1_000.0,
    (3, 1): 100.0,
    (2, 2): 100.0,
    (2, 1): 10.0,
    (1, 2): 10.0,
    (1, 1): 1.0,
}


def _apply_dirichlet_noise(
    priors: list[float],
    alpha: float,
    noise_weight: float,
) -> list[float]:
    """Mix Dirichlet noise into *priors*.  AGZ §14.4.

    ``new_prior = (1 - noise_weight) * prior + noise_weight * noise``
    """
    noise = [_random.gammavariate(alpha, 1.0) for _ in priors]
    noise_total = sum(noise) or 1.0
    noise = [n / noise_total for n in noise]
    return [(1.0 - noise_weight) * p + noise_weight * n for p, n in zip(priors, noise, strict=True)]


# ---------------------------------------------------------------------------
# Heuristic leaf value — sigmoid over the shared static evaluator
# ---------------------------------------------------------------------------


def _leaf_value(state: SearchState, root_stone: Stone) -> float:
    """Return a leaf value in [0, 1] from *root_stone*'s perspective.

    Uses the same :func:`gomoku_brain.evaluate.evaluate_board` as
    alpha-beta — keeping MCTS and AB aligned on what a "good" position
    means — and maps the signed heuristic score through a sigmoid.
    ``k = 1000`` so a live-3 advantage (~1000 pts) maps to ~0.73.
    """
    from gomoku_brain.evaluate import evaluate_board

    raw = evaluate_board(state, root_stone)
    clamped = max(-10.0, min(10.0, raw / 1000.0))
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
        if untried_priors is not None:
            self._untried_priors = untried_priors
        else:
            n = len(untried)
            self._untried_priors = [1.0 / n] * n if n else []

    def puct(self, exploration: float) -> float:
        """AGZ-style PUCT score: Q + c · P · √N / (n + 1)."""
        q = self.wins / self.visits if self.visits else 0.0
        parent_visits = self.parent.visits if self.parent else 1
        return q + exploration * self.prior * math.sqrt(parent_visits) / (self.visits + 1)

    def best_child(self, exploration: float) -> _Node:
        return max(self.children, key=lambda c: c.puct(exploration))

    def has_untried(self) -> bool:
        return len(self._untried) > 0

    def pop_untried(self) -> tuple[tuple[int, int], float]:
        """Remove and return a random untried move together with its prior."""
        idx = _random.randrange(len(self._untried))
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
    """MoveProvider using Monte Carlo Tree Search with PUCT."""

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
        with brain_session(
            "mcts",
            iterations=self.iterations,
            exploration=self.exploration,
            stone=str(stone),
        ) as session:
            diag = session.diag

            candidates = get_candidate_moves(board)
            if len(candidates) == 1:
                if diag is not None:
                    diag.decision_path = "single_candidate"
                    diag.chosen_move = candidates[0]
                return candidates[0]

            # One shared SearchState for the whole search — make/unmake
            # threads it through every iteration without copying.
            state = SearchState.from_board(board, stone)

            # Instant win / instant block shortcut, shared with the other brains.
            shortcut_move, shortcut_tag = probe_immediate(state, stone)
            if shortcut_move is not None:
                if diag is not None:
                    diag.decision_path = shortcut_tag
                    diag.chosen_move = shortcut_move
                return shortcut_move

            # VCF safety gate: if the opponent has a proven forced VCF win,
            # restrict the root candidate set to *durable* breakers (moves
            # that don't hand the win back via a counter-threat).  Matches
            # the semantics enforced in AlphaBeta and ThreatBrain via
            # :func:`classify_vcf_breakers`.
            vcf_gated = False
            if self.vcf_depth > 0:
                from gomoku_brain.threat import _vcf

                opp_vcf = _vcf(board, stone.opponent(), self.vcf_depth)
                if opp_vcf is not None:
                    durable, breakers = classify_vcf_breakers(
                        board, candidates, stone, self.vcf_depth
                    )
                    restricted = durable if durable else breakers
                    if restricted:
                        logfire.info(
                            "VCF safety gate: opponent VCF at {vcf}, restricting to "
                            "{n} breakers ({durable} durable)",
                            vcf=str(opp_vcf),
                            n=len(restricted),
                            durable=len(durable),
                        )
                        candidates = restricted
                        vcf_gated = True
                        note = (
                            f"Opponent VCF at {opp_vcf}, restricted to "
                            f"{len(restricted)} breakers ({len(durable)} durable)"
                        )
                    else:
                        note = f"Opponent VCF at {opp_vcf}, no breakers found"
                    if diag is not None:
                        diag.opp_threats = diag.opp_threats.model_copy(
                            update={"vcf_result": "win"}
                        )
                        diag.notes.append(note)

            stone_val = int(stone)
            opp_val = int(stone.opponent())

            root_untried = [(m.row, m.col) for m in candidates]
            root_priors = _compute_priors(state, root_untried, stone_val, opp_val)

            # Dirichlet noise for exploration breadth (AGZ §14.4).  Skipped
            # on a VCF-gated root — the candidate set is tiny and every move
            # is a near-mandatory defence, so uniform noise only dilutes the
            # heuristic signal.
            if not vcf_gated and self.dirichlet_alpha > 0 and self.dirichlet_weight > 0:
                root_priors = _apply_dirichlet_noise(
                    root_priors,
                    self.dirichlet_alpha,
                    self.dirichlet_weight,
                )

            root = _Node(
                move=None,
                stone=stone.opponent(),
                parent=None,
                untried=root_untried,
                untried_priors=root_priors,
            )

            for _ in range(self.iterations):
                self._run_iteration(root, state, stone)

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

                search_ms = session.elapsed_ms()
                diag.decision_path = diag.decision_path or (
                    "mcts_vcf_gated" if vcf_gated else "mcts"
                )
                diag.chosen_move = chosen
                diag.search_stats = SearchStats(
                    iterations=self.iterations,
                    nodes_searched=root.visits,
                    time_ms=search_ms,
                )
                children_sorted = sorted(
                    root.children, key=lambda c: c.visits, reverse=True
                )
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
                    f"Best: {chosen} visits={best.visits}/{root.visits} "
                    f"wr={best.wins / best.visits:.1%}"
                    if best.visits
                    else f"Best: {chosen} visits=0"
                )

            return chosen

    # -- Core MCTS loop (operates entirely on one SearchState) ---------------

    def _run_iteration(
        self,
        root: _Node,
        state: SearchState,
        root_stone: Stone,
    ) -> None:
        """One select-expand-evaluate-backpropagate iteration.

        The shared ``state`` is mutated via ``make`` while descending and
        restored via ``unmake`` on the way back out, so every iteration
        starts and ends on the same position with zero allocations on the
        hot path.
        """
        node = root
        depth = 0

        # 1. Selection — descend using PUCT, replaying moves into state.
        while not node.has_untried() and not node.is_leaf():
            node = node.best_child(self.exploration)
            r, c = node.move  # type: ignore[misc]
            state.make(r, c, node.stone)
            depth += 1

        # 2. Expansion — add one child.
        if node.has_untried():
            move_rc, move_prior = node.pop_untried()
            child_stone = (
                root_stone if node.stone is not root_stone else root_stone.opponent()
            )
            r, c = move_rc
            state.make(r, c, child_stone)
            depth += 1

            child_moves = get_candidate_moves(state, radius=2)
            child_untried = [(m.row, m.col) for m in child_moves]
            child_priors = _compute_priors(
                state,
                child_untried,
                int(child_stone.opponent()),
                int(child_stone),
            )
            child = _Node(
                move=move_rc,
                stone=child_stone,
                parent=node,
                untried=child_untried,
                prior=move_prior,
                untried_priors=child_priors,
            )
            node.children.append(child)
            node = child

            # Terminal check at the expansion ply.
            terminal_value = self._terminal_value(state, r, c, child_stone, root_stone)
            if terminal_value is not None:
                self._backpropagate(node, terminal_value, root_stone)
                self._restore(state, depth)
                return

        # 3. Heuristic leaf evaluation (replaces random rollout).
        value = _leaf_value(state, root_stone)

        # 4. Backpropagation + restore the shared state to the root position.
        self._backpropagate(node, value, root_stone)
        self._restore(state, depth)

    @staticmethod
    def _restore(state: SearchState, depth: int) -> None:
        """Pop *depth* ``make`` records off ``state``'s undo stack."""
        for _ in range(depth):
            state.unmake()

    @staticmethod
    def _terminal_value(
        state: SearchState,
        row: int,
        col: int,
        mover: Stone,
        root_stone: Stone,
    ) -> float | None:
        """Return a terminal value in [0, 1] (from *root_stone*'s perspective).

        * *mover* just placed at (row, col).  If that move wins, return 1.0
          for root_stone or 0.0 otherwise.
        * If the *opponent* of *mover* can win immediately on the next ply,
          the position is a decisive loss for *mover* (who has no time to
          respond) — report that value.
        * Otherwise return ``None`` (non-terminal leaf).
        """
        from gomoku_brain.evaluate import _should_use_jit
        from gomoku_brain.threat import _find_winning_move

        if _should_use_jit():
            from gomoku_brain.evaluate_jit import check_winner_grid, get_dirs

            if check_winner_grid(state.grid, state.size, row, col, 5, get_dirs()):
                return 1.0 if mover is root_stone else 0.0
        elif _fast_check_winner_state(state, row, col, mover):
            return 1.0 if mover is root_stone else 0.0

        opp = mover.opponent()
        if _find_winning_move(state, opp) is not None:
            return 0.0 if mover is root_stone else 1.0
        return None

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


# ---------------------------------------------------------------------------
# Non-JIT fallback win detector (used by ``_terminal_value``)
# ---------------------------------------------------------------------------


def _fast_check_winner_state(state: SearchState, row: int, col: int, mover: Stone) -> bool:
    """Pure-Python 5-in-a-row check for the JIT-disabled path."""
    from gomoku_brain.directions import DIRECTIONS

    size = state.size
    stone_val = int(mover)
    grid = state.grid
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < size and 0 <= c < size and grid[r, c] == stone_val:
                count += 1
                r += dr * sign
                c += dc * sign
        if count >= 5:
            return True
    return False
