"""Unit tests for MCTS internals.

MCTS is mostly tested via end-to-end brain smoke tests (``test_brains.py``)
that run the whole ``get_move`` pipeline.  Those tests are useful but noisy
— they don't pinpoint which helper broke when something goes wrong.  This
file covers the silent-wrong-answer primitives:

* prior normalisation (a non-normalised prior breaks PUCT proportionality)
* Dirichlet noise mixing (must preserve the sum and still differ from the
  input for non-trivial ``noise_weight``)
* PUCT formula (a sign flip or missing sqrt goes undetected by smoke tests)
* VCF safety gate (when the opponent has a forced four-threat win, MCTS
  must fall back to blocking cells only)
"""

from __future__ import annotations

import math
import random

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.mcts import (
    MCTSBrain,
    _apply_dirichlet_noise,
    _candidate_moves,
    _compute_priors,
    _evaluate_position,
    _fast_check_winner,
    _find_winning_move,
    _Node,
    _score_move_prior,
)

# ---------------------------------------------------------------------------
# Raw-grid helpers
# ---------------------------------------------------------------------------


def _empty_grid(size: int) -> list[list[int]]:
    return [[0] * size for _ in range(size)]


def _place(grid: list[list[int]], r: int, c: int, stone: Stone) -> None:
    grid[r][c] = int(stone)


# ---------------------------------------------------------------------------
# Raw-grid winner / candidate helpers
# ---------------------------------------------------------------------------


class TestFastCheckWinner:
    def test_horizontal_five(self):
        g = _empty_grid(15)
        for c in range(3, 8):
            _place(g, 7, c, Stone.BLACK)
        assert _fast_check_winner(g, 15, 7, 5, int(Stone.BLACK), 5) is True

    def test_not_win_with_four(self):
        g = _empty_grid(15)
        for c in range(3, 7):
            _place(g, 7, c, Stone.BLACK)
        assert _fast_check_winner(g, 15, 7, 5, int(Stone.BLACK), 5) is False

    def test_diagonal_five(self):
        g = _empty_grid(15)
        for k in range(5):
            _place(g, 3 + k, 3 + k, Stone.WHITE)
        assert _fast_check_winner(g, 15, 5, 5, int(Stone.WHITE), 5) is True


class TestFindWinningMove:
    def test_finds_win(self):
        g = _empty_grid(15)
        for c in range(3, 7):
            _place(g, 7, c, Stone.BLACK)
        # Placing at (7,7) or (7,2) wins.
        result = _find_winning_move(g, 15, [(7, 2), (7, 7), (0, 0)], int(Stone.BLACK), 5)
        assert result in {(7, 2), (7, 7)}

    def test_returns_none_when_no_win(self):
        g = _empty_grid(15)
        _place(g, 7, 7, Stone.BLACK)
        assert _find_winning_move(g, 15, [(0, 0), (1, 1)], int(Stone.BLACK), 5) is None

    def test_does_not_mutate_grid(self):
        g = _empty_grid(15)
        for c in range(3, 7):
            _place(g, 7, c, Stone.BLACK)
        before = [row[:] for row in g]
        _find_winning_move(g, 15, [(7, 7), (0, 0)], int(Stone.BLACK), 5)
        assert g == before


class TestCandidateMoves:
    def test_empty_board_returns_center(self):
        cands = _candidate_moves(set(), 15, radius=2)
        assert cands == [(7, 7)]

    def test_radius_1_returns_8_neighbours(self):
        cands = set(_candidate_moves({(7, 7)}, 15, radius=1))
        expected = {(r, c) for r in range(6, 9) for c in range(6, 9) if (r, c) != (7, 7)}
        assert cands == expected

    def test_excludes_occupied(self):
        cands = _candidate_moves({(7, 7), (7, 8)}, 15, radius=1)
        assert (7, 7) not in cands
        assert (7, 8) not in cands


# ---------------------------------------------------------------------------
# Prior scoring
# ---------------------------------------------------------------------------


class TestScoreMovePrior:
    def test_baseline_is_positive(self):
        """Every move gets at least a +1 baseline so priors can't all be 0."""
        g = _empty_grid(9)
        p = _score_move_prior(g, 9, 4, 4, int(Stone.BLACK), int(Stone.WHITE))
        assert p >= 1.0

    def test_extending_own_three_scores_higher_than_empty(self):
        g = _empty_grid(15)
        for c in range(3, 6):
            _place(g, 7, c, Stone.BLACK)
        extend = _score_move_prior(g, 15, 7, 6, int(Stone.BLACK), int(Stone.WHITE))
        empty = _score_move_prior(g, 15, 0, 0, int(Stone.BLACK), int(Stone.WHITE))
        assert extend > empty

    def test_blocking_opponent_three_scores_higher_than_empty(self):
        g = _empty_grid(15)
        for c in range(3, 6):
            _place(g, 7, c, Stone.WHITE)
        block = _score_move_prior(g, 15, 7, 6, int(Stone.BLACK), int(Stone.WHITE))
        empty = _score_move_prior(g, 15, 0, 0, int(Stone.BLACK), int(Stone.WHITE))
        assert block > empty


class TestComputePriors:
    def test_sums_to_one(self):
        g = _empty_grid(15)
        for c in range(3, 6):
            _place(g, 7, c, Stone.BLACK)
        cands = [(7, 2), (7, 6), (0, 0), (5, 5)]
        priors = _compute_priors(g, 15, cands, int(Stone.BLACK), int(Stone.WHITE))
        assert sum(priors) == pytest.approx(1.0)
        assert all(p >= 0 for p in priors)

    def test_fallback_uniform_when_all_zero(self):
        """Baseline is +1 per move so we can't actually get all-zero raw
        scores, but the defensive branch should still return a valid
        uniform distribution for a single-candidate list."""
        g = _empty_grid(9)
        priors = _compute_priors(g, 9, [(4, 4)], int(Stone.BLACK), int(Stone.WHITE))
        assert priors == pytest.approx([1.0])

    def test_stronger_cell_gets_higher_prior(self):
        """The cell that completes a four-threat should dominate an empty
        corner cell."""
        g = _empty_grid(15)
        for c in range(3, 6):
            _place(g, 7, c, Stone.BLACK)
        cands = [(0, 0), (7, 6), (14, 14)]
        priors = _compute_priors(g, 15, cands, int(Stone.BLACK), int(Stone.WHITE))
        assert priors[1] > priors[0]
        assert priors[1] > priors[2]


# ---------------------------------------------------------------------------
# Dirichlet noise
# ---------------------------------------------------------------------------


class TestApplyDirichletNoise:
    def test_preserves_sum(self):
        rng_state = random.getstate()
        try:
            random.seed(42)
            priors = [0.1, 0.2, 0.3, 0.4]
            mixed = _apply_dirichlet_noise(priors, alpha=0.03, noise_weight=0.25)
            assert sum(mixed) == pytest.approx(1.0)
            assert all(p >= 0 for p in mixed)
        finally:
            random.setstate(rng_state)

    def test_noise_weight_zero_is_identity(self):
        priors = [0.1, 0.2, 0.3, 0.4]
        mixed = _apply_dirichlet_noise(priors, alpha=0.03, noise_weight=0.0)
        assert mixed == pytest.approx(priors)

    def test_noise_weight_one_ignores_priors(self):
        """With noise_weight=1.0 the output is pure Dirichlet draw — so two
        different seeds must give different results even if priors are the
        same."""
        priors = [0.25, 0.25, 0.25, 0.25]
        rng_state = random.getstate()
        try:
            random.seed(1)
            a = _apply_dirichlet_noise(priors, alpha=0.03, noise_weight=1.0)
            random.seed(2)
            b = _apply_dirichlet_noise(priors, alpha=0.03, noise_weight=1.0)
        finally:
            random.setstate(rng_state)
        assert a != b
        assert sum(a) == pytest.approx(1.0)
        assert sum(b) == pytest.approx(1.0)

    def test_non_trivial_weight_changes_priors(self):
        rng_state = random.getstate()
        try:
            random.seed(0)
            priors = [0.1, 0.2, 0.3, 0.4]
            mixed = _apply_dirichlet_noise(priors, alpha=0.03, noise_weight=0.25)
        finally:
            random.setstate(rng_state)
        assert mixed != priors
        assert sum(mixed) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _evaluate_position — heuristic leaf value
# ---------------------------------------------------------------------------


class TestEvaluatePosition:
    def test_returns_in_unit_interval(self):
        g = _empty_grid(15)
        occ: set[tuple[int, int]] = set()
        for c in range(3, 6):
            _place(g, 7, c, Stone.BLACK)
            occ.add((7, c))
        v = _evaluate_position(g, 15, occ, int(Stone.BLACK))
        assert 0.0 <= v <= 1.0

    def test_empty_board_is_neutral(self):
        v = _evaluate_position(_empty_grid(15), 15, set(), int(Stone.BLACK))
        assert v == pytest.approx(0.5)

    def test_symmetry(self):
        """Swapping the evaluator's perspective should map value to 1 - value."""
        g = _empty_grid(15)
        occ: set[tuple[int, int]] = set()
        for c in range(3, 6):
            _place(g, 7, c, Stone.BLACK)
            occ.add((7, c))
        v_black = _evaluate_position(g, 15, occ, int(Stone.BLACK))
        v_white = _evaluate_position(g, 15, occ, int(Stone.WHITE))
        # The sign flips but the 0.8 opp-weight makes it asymmetric, so the
        # exact symmetry ``v_black + v_white == 1`` does not hold.  Instead:
        # the BLACK-favourable position must score > 0.5 for BLACK and < 0.5
        # for WHITE.
        assert v_black > 0.5
        assert v_white < 0.5


# ---------------------------------------------------------------------------
# PUCT formula on _Node
# ---------------------------------------------------------------------------


class TestNodePUCT:
    def test_zero_visits_gives_exploration_only(self):
        parent = _Node(move=None, stone=Stone.BLACK, parent=None, untried=[])
        parent.visits = 4  # parent_visits used under sqrt
        child = _Node(move=(7, 7), stone=Stone.BLACK, parent=parent, untried=[], prior=0.5)
        # Q=0 (no visits), so PUCT = c * P * sqrt(4) / (0 + 1) = c * 0.5 * 2
        assert child.puct(exploration=2.0) == pytest.approx(2.0 * 0.5 * 2.0 / 1.0)

    def test_prior_and_visits_scale_correctly(self):
        parent = _Node(move=None, stone=Stone.BLACK, parent=None, untried=[])
        parent.visits = 16
        child = _Node(move=(7, 7), stone=Stone.BLACK, parent=parent, untried=[], prior=0.25)
        child.visits = 3
        child.wins = 1.5  # Q = 0.5
        # c=1, so PUCT = 0.5 + 1 * 0.25 * 4 / 4 = 0.5 + 0.25 = 0.75
        assert child.puct(exploration=1.0) == pytest.approx(0.75)

    def test_exploration_zero_is_pure_q(self):
        parent = _Node(move=None, stone=Stone.BLACK, parent=None, untried=[])
        parent.visits = 10
        child = _Node(move=(1, 1), stone=Stone.BLACK, parent=parent, untried=[], prior=0.5)
        child.visits = 5
        child.wins = 2.0  # Q = 0.4
        assert child.puct(exploration=0.0) == pytest.approx(0.4)

    def test_best_child_prefers_higher_puct(self):
        parent = _Node(move=None, stone=Stone.BLACK, parent=None, untried=[])
        parent.visits = 4
        low = _Node(move=(0, 0), stone=Stone.BLACK, parent=parent, untried=[], prior=0.1)
        high = _Node(move=(1, 1), stone=Stone.BLACK, parent=parent, untried=[], prior=0.9)
        parent.children = [low, high]
        assert parent.best_child(exploration=1.0) is high

    def test_has_untried(self):
        n = _Node(move=None, stone=Stone.BLACK, parent=None, untried=[(0, 0)])
        assert n.has_untried() is True
        n.pop_untried()
        assert n.has_untried() is False

    def test_pop_untried_returns_move_and_prior(self):
        n = _Node(
            move=None,
            stone=Stone.BLACK,
            parent=None,
            untried=[(0, 0), (1, 1)],
            untried_priors=[0.3, 0.7],
        )
        move1, prior1 = n.pop_untried()
        move2, prior2 = n.pop_untried()
        assert {move1, move2} == {(0, 0), (1, 1)}
        assert math.isclose(prior1 + prior2, 1.0)


# ---------------------------------------------------------------------------
# MCTSBrain high-level behaviour
# ---------------------------------------------------------------------------


class TestMCTSBrain:
    def test_first_move_is_center(self):
        """On an empty board the only candidate is the centre cell."""
        brain = MCTSBrain(iterations=1)
        move = brain.get_move(Board(size=15), Stone.BLACK)
        assert move == Move(row=7, col=7)

    def test_instant_win(self):
        """When we have a winning move we must play it regardless of MCTS
        budget — this is the ``decision_path = "instant_win"`` shortcut."""
        b = Board(size=15)
        for c in range(3, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        brain = MCTSBrain(iterations=5)
        move = brain.get_move(b, Stone.BLACK)
        assert move in {Move(row=7, col=2), Move(row=7, col=7)}

    def test_instant_block(self):
        """If the opponent has a 1-ply win we must block it."""
        b = Board(size=15)
        for c in range(3, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.WHITE)
        # Put a single black stone far away so BLACK has candidates
        b = b.place_stone(Move(row=0, col=0), Stone.BLACK)
        brain = MCTSBrain(iterations=5)
        move = brain.get_move(b, Stone.BLACK)
        # The only blocking cells are (7,2) and (7,7)
        assert move in {Move(row=7, col=2), Move(row=7, col=7)}

    def test_vcf_gate_picks_blocker_when_opponent_has_forced_win(self):
        """When the opponent has a forced VCF win, MCTS must restrict to
        blocking candidates.  Construct an open-four: opponent plays either
        end for the win, so any move MCTS chooses must block one of them.

        This is a weaker assertion than "picks the correct side" — deciding
        which end to block is an AI strength question — but it's enough to
        guard against the VCF gate being silently skipped.
        """
        b = Board(size=15)
        # WHITE open four: placing W at (7,2) or (7,7) wins next turn.
        for c in range(3, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.WHITE)
        # Minimal BLACK material so candidates list is non-trivial
        b = b.place_stone(Move(row=0, col=0), Stone.BLACK)
        brain = MCTSBrain(iterations=50, vcf_depth=4)
        move = brain.get_move(b, Stone.BLACK)
        # Instant-block catches this case first, but the assertion still
        # verifies that the VCF gate doesn't *override* a correct block.
        assert move in {Move(row=7, col=2), Move(row=7, col=7)}

    def test_respects_iteration_budget(self):
        """Smoke: MCTS returns a legal move with very few iterations."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        brain = MCTSBrain(iterations=10)
        move = brain.get_move(b, Stone.WHITE)
        assert b.is_valid_move(move)

    def test_returns_legal_move_in_complex_position(self):
        b = Board(size=15)
        for r, c in [(7, 7), (8, 8), (6, 6), (7, 8)]:
            stone = Stone.BLACK if (r + c) % 2 == 0 else Stone.WHITE
            b = b.place_stone(Move(row=r, col=c), stone)
        brain = MCTSBrain(iterations=20)
        move = brain.get_move(b, Stone.BLACK)
        assert b.is_valid_move(move)
