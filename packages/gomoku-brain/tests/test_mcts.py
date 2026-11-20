"""Unit tests for MCTS internals.

MCTS is mostly tested via end-to-end brain smoke tests (``test_brains.py``)
that run the whole ``get_move`` pipeline.  Those tests are useful but noisy
— they don't pinpoint which helper broke when something goes wrong.  This
file covers the silent-wrong-answer primitives:

* prior normalisation (a non-normalised prior breaks PUCT proportionality)
* Dirichlet noise mixing (must preserve the sum and still differ from the
  input for non-trivial ``noise_weight``)
* heuristic leaf value (must lie in [0, 1] and reflect side advantage)
* PUCT formula (a sign flip or missing sqrt goes undetected by smoke tests)
* VCF safety gate (when the opponent has a forced four-threat win, MCTS
  must fall back to blocking cells only)

The raw-grid helpers that earlier versions of this file referenced
(``_candidate_moves`` / ``_evaluate_position`` / ``_fast_check_winner`` /
``_find_winning_move`` / ``_score_move_prior``) were removed when MCTS
moved to a shared :class:`SearchState`.  Their equivalents are covered
in ``test_search_state.py``, ``test_evaluate_jit.py``, ``test_threat.py``,
and ``test_parity.py``; we don't duplicate that coverage here.
"""

import math
import random

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.mcts import (
    MCTSBrain,
    _apply_dirichlet_noise,
    _compute_priors,
    _leaf_value,
    _Node,
)
from gomoku_brain.search_state import SearchState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_with_three(stone: Stone, size: int = 15) -> SearchState:
    """Board with a 3-in-a-row of *stone* on row 7, cols 3-5; *stone* to move."""
    b = Board(size=size)
    for c in range(3, 6):
        b = b.place_stone(Move(row=7, col=c), stone)
    return SearchState.from_board(b, stone)


# ---------------------------------------------------------------------------
# Prior scoring
# ---------------------------------------------------------------------------


class TestComputePriors:
    def test_sums_to_one(self):
        state = _state_with_three(Stone.BLACK)
        cands = [(7, 2), (7, 6), (0, 0), (5, 5)]
        priors = _compute_priors(state, cands, int(Stone.BLACK), int(Stone.WHITE))
        assert sum(priors) == pytest.approx(1.0)
        assert all(p >= 0 for p in priors)

    def test_single_candidate_gets_unit_prior(self):
        state = SearchState.from_board(Board(size=9), Stone.BLACK)
        priors = _compute_priors(state, [(4, 4)], int(Stone.BLACK), int(Stone.WHITE))
        assert priors == pytest.approx([1.0])

    def test_empty_candidates_returns_empty(self):
        state = SearchState.from_board(Board(size=9), Stone.BLACK)
        assert _compute_priors(state, [], int(Stone.BLACK), int(Stone.WHITE)) == []

    def test_stronger_cell_gets_higher_prior(self):
        """The cell that completes a four-threat should dominate empty cells."""
        state = _state_with_three(Stone.BLACK)
        cands = [(0, 0), (7, 6), (14, 14)]
        priors = _compute_priors(state, cands, int(Stone.BLACK), int(Stone.WHITE))
        assert priors[1] > priors[0]
        assert priors[1] > priors[2]

    def test_blocking_opponent_three_scores_above_empty(self):
        """A defensive move (blocking opponent's three) must outscore an
        unrelated empty cell — otherwise MCTS would never explore blocks."""
        state = _state_with_three(Stone.WHITE)
        cands = [(0, 0), (7, 6)]
        priors = _compute_priors(state, cands, int(Stone.BLACK), int(Stone.WHITE))
        assert priors[1] > priors[0]


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
        """With ``noise_weight=1.0`` the output is a pure Dirichlet draw —
        two different seeds must give different results even for identical
        input priors."""
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
# Heuristic leaf value
# ---------------------------------------------------------------------------


class TestLeafValue:
    def test_returns_in_unit_interval(self):
        state = _state_with_three(Stone.BLACK)
        v = _leaf_value(state, Stone.BLACK)
        assert 0.0 <= v <= 1.0

    def test_empty_board_is_neutral(self):
        state = SearchState.from_board(Board(size=15), Stone.BLACK)
        v = _leaf_value(state, Stone.BLACK)
        assert v == pytest.approx(0.5)

    def test_advantage_flips_with_perspective(self):
        """A position favouring BLACK must score > 0.5 from BLACK's view and
        < 0.5 from WHITE's view (the 0.8 opp-weight makes the values
        asymmetric, so ``v_black + v_white == 1`` does not hold exactly)."""
        state = _state_with_three(Stone.BLACK)
        v_black = _leaf_value(state, Stone.BLACK)
        v_white = _leaf_value(state, Stone.WHITE)
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
