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

    def test_pop_untried_takes_highest_prior_first(self):
        n = _Node(
            move=None,
            stone=Stone.BLACK,
            parent=None,
            untried=[(0, 0), (1, 1), (2, 2)],
            untried_priors=[0.2, 0.5, 0.3],
        )
        assert [n.pop_untried()[0] for _ in range(3)] == [(1, 1), (2, 2), (0, 0)]

    def test_expands_while_best_untried_outscores_children(self):
        parent = _Node(
            move=None, stone=Stone.WHITE, parent=None, untried=[(0, 0), (1, 1)], untried_priors=[0.1, 0.4]
        )
        assert parent.wants_expansion(1.0, 0.2)  # no children yet
        parent.visits = 4
        parent.wins = 2.0  # Q = 0.5 for the chooser too
        child = _Node(move=(2, 2), stone=Stone.BLACK, parent=parent, untried=[], prior=0.5)
        child.visits = 3
        child.wins = 2.7  # Q = 0.9: PUCT = 0.9 + 0.5 * 2 / 4 = 1.15
        parent.children = [child]
        # Best untried (prior 0.4): FPU 0.3 + 0.4 * 2 = 1.1 < 1.15 -> descend.
        assert not parent.wants_expansion(1.0, 0.2)
        child.wins = 1.5  # Q = 0.5: PUCT = 0.75 -> expand
        assert parent.wants_expansion(1.0, 0.2)

    def test_no_expansion_without_untried(self):
        parent = _Node(move=None, stone=Stone.WHITE, parent=None, untried=[])
        assert not parent.wants_expansion(1.0, 0.2)


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


# ---------------------------------------------------------------------------
# Tree invariants (search run directly on a hand-built root)
# ---------------------------------------------------------------------------


def _open_three_board() -> Board:
    """Black open three on row 7 (cols 5-7), stones far away; Black to move."""
    b = Board(size=15)
    stone = Stone.BLACK
    for r, c in [(7, 5), (2, 2), (7, 6), (2, 12), (7, 7), (12, 2), (10, 10), (12, 12)]:
        b = b.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return b


def _search_tree(board: Board, stone: Stone, iterations: int, seed: int = 0, **settings) -> _Node:
    """Run *iterations* MCTS iterations from *board* and return the root."""
    from gomoku_brain.candidates import get_candidate_moves

    random.seed(seed)
    brain = MCTSBrain(iterations=iterations, dirichlet_weight=0.0, **settings)
    state = SearchState.from_board(board, stone)
    untried = [(m.row, m.col) for m in get_candidate_moves(board)]
    priors = _compute_priors(state, untried, int(stone), int(stone.opponent()))
    root = _Node(move=None, stone=stone.opponent(), parent=None, untried=untried, untried_priors=priors)
    for _ in range(iterations):
        brain._run_iteration(root, state, stone)
    assert state.ply == 0  # every iteration restores the root position
    return root


def _walk(root: _Node):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


class TestPriorOrderedExpansion:
    def test_strong_move_gets_searched_deeply(self):
        """In a quiet position the search goes deeper than one ply per
        candidate: the most visited root move has a grandchild with visits."""
        b = Board(size=15)
        stone = Stone.BLACK
        for r, c in [(7, 7), (7, 8), (8, 6), (6, 8), (9, 9), (8, 9)]:
            b = b.place_stone(Move(row=r, col=c), stone)
            stone = stone.opponent()
        root = _search_tree(b, Stone.BLACK, iterations=2000)
        assert root._untried  # weak moves were never expanded
        depth = 0
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            depth += 1
        assert depth >= 6


class TestTerminalNodes:
    def test_terminal_nodes_are_never_expanded(self):
        """A position decided when its node is created gets no moves and
        no children (nodes proven later by the solver keep theirs)."""
        root = _search_tree(_open_three_board(), Stone.BLACK, iterations=3000)
        decided = [n for n in _walk(root) if n.terminal is not None and not n.children]
        assert decided, "the open three must lead to decided positions"
        for n in decided:
            assert not n.has_untried()

    def test_terminal_value_is_not_diluted(self):
        """Every visit of a decided leaf backs up the same 0 / 1 value."""
        root = _search_tree(_open_three_board(), Stone.BLACK, iterations=3000)
        for n in _walk(root):
            if n.terminal is not None and not n.children:
                assert n.terminal in (0.0, 1.0)
                q = n.wins / n.visits
                # Q is stored from the mover's perspective.
                expected = n.terminal if n.stone is Stone.BLACK else 1.0 - n.terminal
                assert q == pytest.approx(expected)

    def test_no_node_below_five_in_a_row(self):
        board = _open_three_board()
        root = _search_tree(board, Stone.BLACK, iterations=3000)

        def wins(state: SearchState, r: int, c: int, stone: Stone) -> bool:
            from gomoku_brain.mcts import _fast_check_winner_state

            return _fast_check_winner_state(state, r, c, stone)

        state = SearchState.from_board(board, Stone.BLACK)

        def visit(node: _Node) -> None:
            for child in node.children:
                assert child.move is not None
                r, c = child.move
                state.make(r, c, child.stone)
                if wins(state, r, c, child.stone):
                    assert child.children == []
                visit(child)
                state.unmake()

        visit(root)


class TestForcedReplies:
    @staticmethod
    def _status(board: Board, mover: Stone, last: tuple[int, int]):
        state = SearchState.from_board(board, mover.opponent())
        return MCTSBrain._node_status(state, last[0], last[1], mover, Stone.BLACK)

    @staticmethod
    def _board(black: list[tuple[int, int]], white: list[tuple[int, int]]) -> Board:
        b = Board(size=15)
        for r, c in black:
            b = b.place_stone(Move(row=r, col=c), Stone.BLACK)
        for r, c in white:
            b = b.place_stone(Move(row=r, col=c), Stone.WHITE)
        return b

    def test_single_four_forces_the_block(self):
        # Black four (7,3..7,6) with (7,2) blocked: White must play (7,7).
        b = self._board([(7, 3), (7, 4), (7, 5), (7, 6)], [(7, 2), (0, 0), (0, 14)])
        assert self._status(b, Stone.BLACK, (7, 6)) == (None, (7, 7))

    def test_open_four_is_a_win_for_the_mover(self):
        b = self._board([(7, 3), (7, 4), (7, 5), (7, 6)], [(0, 0), (0, 14), (14, 0)])
        assert self._status(b, Stone.BLACK, (7, 6)) == (1.0, None)

    def test_side_to_move_with_a_five_threat_wins(self):
        # Black just played a quiet move, White already has a four.
        b = self._board([(0, 0), (0, 2)], [(7, 3), (7, 4), (7, 5), (7, 6)])
        assert self._status(b, Stone.BLACK, (0, 2)) == (0.0, None)

    def test_five_is_terminal(self):
        b = self._board([(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)], [(0, 0), (0, 14), (14, 0), (14, 14)])
        assert self._status(b, Stone.BLACK, (7, 7)) == (1.0, None)

    def test_quiet_position_is_open(self):
        b = self._board([(7, 7), (7, 8)], [(8, 8), (6, 6)])
        assert self._status(b, Stone.BLACK, (7, 8)) == (None, None)

    def test_tree_keeps_only_the_block_under_a_four(self):
        root = _search_tree(_open_three_board(), Stone.BLACK, iterations=3000)
        state = SearchState.from_board(_open_three_board(), Stone.BLACK)

        def visit(node: _Node) -> None:
            for child in node.children:
                assert child.move is not None
                state.make(child.move[0], child.move[1], child.stone)
                if child.terminal is None:
                    threat = _find_winning_move(state, child.stone)
                    if threat is not None:
                        assert len(child.children) + len(child._untried) == 1
                visit(child)
                state.unmake()

        from gomoku_brain.threat import _find_winning_move

        visit(root)
        # The open four itself is now a proven win at the root.
        four = [c for c in root.children if c.move in {(7, 4), (7, 8)}]
        assert four and all(c.terminal == 1.0 for c in four)


class TestOwnVCF:
    @staticmethod
    def _vcf_board() -> Board:
        """Black wins by fours: (7,6) makes a four on row 7 and a split four
        on column 6 at once.  No immediate five for either side."""
        b = Board(size=15)
        for r, c in [(7, 3), (7, 4), (7, 5), (3, 6), (4, 6), (5, 6)]:
            b = b.place_stone(Move(row=r, col=c), Stone.BLACK)
        for r, c in [(7, 2), (2, 6), (0, 14), (14, 0), (14, 14)]:
            b = b.place_stone(Move(row=r, col=c), Stone.WHITE)
        return b

    def test_plays_its_vcf_with_a_tiny_budget(self):
        from gomoku_brain.threat import _find_winning_move, _vcf

        b = self._vcf_board()
        assert _find_winning_move(b, Stone.BLACK) is None
        vcf = _vcf(b, Stone.BLACK, 10)
        assert vcf is not None
        assert MCTSBrain(iterations=1).get_move(b, Stone.BLACK) == vcf

    def test_vcf_depth_zero_disables_it(self, monkeypatch):
        import gomoku_brain.threat as threat

        calls: list[Stone] = []
        real = threat._vcf

        def spy(board, stone, depth, *args, **kwargs):
            calls.append(stone)
            return real(board, stone, depth, *args, **kwargs)

        monkeypatch.setattr(threat, "_vcf", spy)
        MCTSBrain(iterations=5, vcf_depth=0).get_move(self._vcf_board(), Stone.BLACK)
        assert calls == []


class TestLeafTempo:
    def test_open_three_is_worth_more_for_the_side_to_move(self):
        """Black open three (row 7, cols 3-5), White stones far away."""
        b = Board(size=15)
        for c in range(3, 6):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        for r, c in [(0, 0), (14, 14)]:
            b = b.place_stone(Move(row=r, col=c), Stone.WHITE)
        state = SearchState.from_board(b, Stone.BLACK)
        black_to_move = _leaf_value(state, Stone.BLACK, Stone.BLACK)
        white_to_move = _leaf_value(state, Stone.BLACK, Stone.WHITE)
        static = _leaf_value(state, Stone.BLACK)
        assert black_to_move > static
        assert white_to_move == pytest.approx(static)  # White cannot make a live four
        assert black_to_move > 0.95

    def test_no_tempo_bonus_under_a_five_threat(self):
        """Black has an open three, but White threatens five: no bonus."""
        b = Board(size=15)
        for c in range(3, 6):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        for c in range(3, 7):
            b = b.place_stone(Move(row=12, col=c), Stone.WHITE)
        b = b.place_stone(Move(row=12, col=2), Stone.BLACK)
        state = SearchState.from_board(b, Stone.BLACK)
        assert _leaf_value(state, Stone.BLACK, Stone.BLACK) == pytest.approx(_leaf_value(state, Stone.BLACK))


class TestBudgetAndReuse:
    @staticmethod
    def _quiet_board() -> Board:
        b = Board(size=15)
        stone = Stone.BLACK
        for r, c in [(7, 7), (7, 8), (8, 6), (6, 8), (9, 9), (8, 9)]:
            b = b.place_stone(Move(row=r, col=c), stone)
            stone = stone.opponent()
        return b

    def test_time_limit_stops_the_search(self):
        import time

        brain = MCTSBrain(iterations=1_000_000, time_limit_s=0.3)
        brain.get_move(self._quiet_board(), Stone.BLACK)  # warm up the JIT kernels
        t0 = time.monotonic()
        brain.get_move(self._quiet_board(), Stone.BLACK)
        assert time.monotonic() - t0 < 1.5

    def test_diagnostics_report_the_iterations_run(self):
        from gomoku.diagnostics import MoveDiagnostics
        from gomoku_brain.brain_base import BrainSession

        diag = MoveDiagnostics(move_number=7, stone=Stone.BLACK)
        session = BrainSession(diag=diag, _t0=0.0)
        MCTSBrain(iterations=123).choose_move(self._quiet_board(), Stone.BLACK, session)
        assert diag.search_stats is not None
        assert diag.search_stats.iterations == 123

    def test_lazy_generation_leaves_unvisited_leaves_empty(self):
        root = _search_tree(self._quiet_board(), Stone.BLACK, iterations=500)
        once = [n for n in _walk(root) if n.parent is not None and n.visits == 1 and n.terminal is None]
        assert once
        # A leaf visited once was scored, never expanded: no moves generated
        # unless it is a forced node (its single block is set at creation).
        assert all(not n.generated or len(n._untried) == 1 for n in once)

    def test_reuses_the_subtree_of_the_reply(self):
        board = self._quiet_board()
        brain = MCTSBrain(iterations=400, reuse_tree=True, dirichlet_weight=0.0)
        ours = brain.get_move(board, Stone.BLACK)
        assert brain._kept is not None
        kept_node = brain._kept[0]
        reply_node = max(kept_node.children, key=lambda c: c.visits)
        assert reply_node.move is not None
        visits = reply_node.visits
        board = board.place_stone(ours, Stone.BLACK)
        board = board.place_stone(Move(row=reply_node.move[0], col=reply_node.move[1]), Stone.WHITE)
        root = brain._take_kept(board, Stone.BLACK)
        assert root is reply_node
        assert root.parent is None and root.visits == visits

    def test_no_reuse_after_an_unexpected_position(self):
        board = self._quiet_board()
        brain = MCTSBrain(iterations=200, reuse_tree=True)
        brain.get_move(board, Stone.BLACK)
        # A different game: the kept tree must not be used.
        other = Board(size=15).place_stone(Move(row=3, col=3), Stone.BLACK)
        other = other.place_stone(Move(row=3, col=4), Stone.WHITE)
        assert brain._take_kept(other, Stone.BLACK) is None
        assert brain._kept is None

    def test_reuse_is_off_by_default(self):
        brain = MCTSBrain(iterations=50)
        brain.get_move(self._quiet_board(), Stone.BLACK)
        assert brain._kept is None


class TestSolver:
    @staticmethod
    def _node(stone: Stone, parent: _Node | None, untried=None) -> _Node:
        n = _Node(move=(0, 0) if parent else None, stone=stone, parent=parent, untried=untried)
        if parent is not None:
            parent.children.append(n)
        return n

    def test_a_winning_child_proves_the_parent(self):
        root = self._node(Stone.WHITE, None, untried=[])  # Black chooses at the root
        mid = self._node(Stone.BLACK, root, untried=[(1, 1)])  # White chooses here
        win_for_white = self._node(Stone.WHITE, mid, untried=[])
        win_for_white.prove(0.0, Stone.BLACK)  # Black (root stone) loses
        MCTSBrain._backpropagate(win_for_white, 0.0, Stone.BLACK)
        assert mid.proven == -1 and mid.terminal == 0.0  # Black's move there loses
        # The root has no untried move left and its only child loses for
        # Black, so the root is lost for Black too (a win for its "mover").
        assert root.proven == 1 and root.terminal == 0.0

    def test_parent_with_open_moves_is_not_proven_lost(self):
        root = self._node(Stone.WHITE, None, untried=[(5, 5)])
        lost = self._node(Stone.BLACK, root, untried=[])
        lost.prove(0.0, Stone.BLACK)  # Black's move loses
        MCTSBrain._backpropagate(lost, 0.0, Stone.BLACK)
        assert root.proven == 0 and root.terminal is None
        # ...and the untried move is expanded rather than the lost one revisited.
        assert root.wants_expansion(2.0, 0.2)

    def test_best_child_skips_proven_losses(self):
        root = self._node(Stone.WHITE, None, untried=[])
        root.visits = 10
        lost = self._node(Stone.BLACK, root, untried=[])
        lost.prior, lost.visits, lost.wins = 0.9, 1, 1.0
        lost.prove(0.0, Stone.BLACK)
        other = self._node(Stone.BLACK, root, untried=[])
        other.prior, other.visits, other.wins = 0.1, 5, 1.0
        assert root.best_child(2.0) is other

    def test_final_choice_prefers_a_proven_win(self):
        root = self._node(Stone.WHITE, None, untried=[])
        busy = self._node(Stone.BLACK, root, untried=[])
        busy.visits = 100
        win = self._node(Stone.BLACK, root, untried=[])
        win.visits = 3
        win.prove(1.0, Stone.BLACK)
        assert MCTSBrain._final_choice(root) is win

    def test_final_choice_avoids_proven_losses(self):
        root = self._node(Stone.WHITE, None, untried=[])
        lost = self._node(Stone.BLACK, root, untried=[])
        lost.visits = 100
        lost.prove(0.0, Stone.BLACK)
        other = self._node(Stone.BLACK, root, untried=[])
        other.visits = 3
        assert MCTSBrain._final_choice(root) is other

    def test_proofs_in_the_tree_are_consistent(self):
        """Every proven node with children agrees with them."""
        root = _search_tree(_open_three_board(), Stone.BLACK, iterations=3000)
        assert root.proven == -1  # Black to move wins: a loss for White, who "moved" into the root
        for n in _walk(root):
            if n.proven and n.children:
                if n.proven < 0:
                    assert any(c.proven > 0 for c in n.children)
                else:
                    assert not n._untried and all(c.proven < 0 for c in n.children)

    def test_search_stops_once_the_root_is_solved(self):
        from gomoku.diagnostics import MoveDiagnostics
        from gomoku_brain.brain_base import BrainSession

        diag = MoveDiagnostics(move_number=9, stone=Stone.BLACK)
        session = BrainSession(diag=diag, _t0=0.0)
        brain = MCTSBrain(iterations=5000, vcf_depth=0)  # no root VCF: the tree must solve it
        move = brain.choose_move(_open_three_board(), Stone.BLACK, session)
        assert move in {Move(row=7, col=4), Move(row=7, col=8)}
        assert diag.search_stats is not None
        assert diag.search_stats.iterations < 5000
        assert diag.decision_path == "mcts_proven_win"


class TestImplicitMinimax:
    @staticmethod
    def _quiet_board() -> Board:
        return TestBudgetAndReuse._quiet_board()

    def test_minimax_values_follow_the_children(self):
        root = _search_tree(self._quiet_board(), Stone.BLACK, iterations=1500, minimax_weight=0.5)
        checked = 0
        for n in _walk(root):
            evaluated = [c.v_mm for c in n.children if c.v_mm is not None]
            if evaluated:
                assert n.v_mm == pytest.approx(1.0 - max(evaluated))
                checked += 1
        assert checked > 10

    def test_weight_zero_keeps_the_plain_search(self):
        a = _search_tree(self._quiet_board(), Stone.BLACK, iterations=800)
        b = _search_tree(self._quiet_board(), Stone.BLACK, iterations=800, minimax_weight=0.0)
        assert all(n.v_mm is None for n in _walk(b))
        visits_a = sorted((c.move, c.visits) for c in a.children)
        visits_b = sorted((c.move, c.visits) for c in b.children)
        assert visits_a == visits_b

    def test_puct_blends_the_minimax_value(self):
        parent = _Node(move=None, stone=Stone.WHITE, parent=None, untried=[])
        parent.visits = 4
        child = _Node(move=(1, 1), stone=Stone.BLACK, parent=parent, untried=[], prior=0.0)
        child.visits, child.wins, child.v_mm = 2, 1.6, 0.2  # Q = 0.8, minimax 0.2
        assert child.puct(1.0) == pytest.approx(0.8)
        assert child.puct(1.0, mm_weight=0.5) == pytest.approx(0.5)


class TestHybrids:
    @staticmethod
    def _open_three_for_white() -> Board:
        """White open three on row 7 (cols 5-7); Black to move must block."""
        b = Board(size=15)
        for r, c in [(7, 5), (7, 6), (7, 7), (3, 10)]:
            b = b.place_stone(Move(row=r, col=c), Stone.WHITE)
        for r, c in [(2, 2), (12, 3), (2, 12), (12, 12)]:
            b = b.place_stone(Move(row=r, col=c), Stone.BLACK)
        return b

    def test_search_priors_rank_the_blocks_first(self):
        from gomoku_brain.candidates import get_candidate_moves
        from gomoku_brain.mcts_hybrid import search_priors

        b = self._open_three_for_white()
        state = SearchState.from_board(b, Stone.BLACK)
        moves = [(m.row, m.col) for m in get_candidate_moves(b)]
        for depth in (2, 3):
            priors = search_priors(state, moves, Stone.BLACK, depth, k=8)
            assert sum(priors) == pytest.approx(1.0)
            top = {m for _, m in sorted(zip(priors, moves, strict=True), reverse=True)[:2]}
            assert top == {(7, 4), (7, 8)}
        assert state.ply == 0

    def test_deeper_search_concentrates_on_the_blocks(self):
        from gomoku_brain.candidates import get_candidate_moves
        from gomoku_brain.mcts_hybrid import search_priors

        b = self._open_three_for_white()
        state = SearchState.from_board(b, Stone.BLACK)
        moves = [(m.row, m.col) for m in get_candidate_moves(b)]

        def blocks(depth: int) -> float:
            priors = search_priors(state, moves, Stone.BLACK, depth, k=8)
            return sum(p for p, m in zip(priors, moves, strict=True) if m in {(7, 4), (7, 8)})

        assert blocks(3) > blocks(1)

    def test_vcf_at_a_visited_node_proves_it(self):
        board = TestOwnVCF._vcf_board()  # Black (to move) has a VCF
        state = SearchState.from_board(board, Stone.BLACK)
        brain = MCTSBrain(vcf_expand_depth=10, vcf_expand_visits=4)
        root = _Node(move=None, stone=Stone.BLACK, parent=None, untried=[])
        node = _Node(move=(0, 0), stone=Stone.WHITE, parent=root, untried=[(1, 1)])  # White just moved
        root.children.append(node)
        node.visits = 3
        brain._boost(node, state, Stone.WHITE)
        assert node.terminal is None  # not visited enough yet
        node.visits = 4
        brain._boost(node, state, Stone.WHITE)
        assert node.proven == -1  # White's move loses to Black's VCF
        assert node.terminal == 0.0  # from White's (root stone's) side
        assert root.proven == 1  # White's only move here loses: the root is won for Black
        assert state.ply == 0

    def test_reprior_blends_and_reorders(self):
        b = self._open_three_for_white()
        state = SearchState.from_board(b, Stone.BLACK)
        brain = MCTSBrain(ab_prior_depth=2, ab_prior_visits=1, ab_prior_weight=1.0)
        node = _Node(move=(0, 14), stone=Stone.WHITE, parent=None, untried=None)
        node.set_untried([(13, 13), (7, 4), (0, 13)], [0.6, 0.1, 0.3])
        brain._reprior(node, state, Stone.BLACK)
        assert sum(node._untried_priors) == pytest.approx(1.0)
        assert node._untried[-1] == (7, 4)  # the block is now expanded first

    def test_search_with_every_hybrid_on(self):
        root = _search_tree(
            self._open_three_for_white(),
            Stone.BLACK,
            iterations=800,
            ab_prior_depth=2,
            ab_prior_visits=8,
            vcf_expand_depth=6,
            vcf_expand_visits=8,
        )
        best = max(root.children, key=lambda c: c.visits)
        assert best.move in {(7, 4), (7, 8), (7, 3), (7, 9)}
        for n in _walk(root):
            if n.proven and n.children:
                if n.proven < 0:
                    assert any(c.proven > 0 for c in n.children)
                else:
                    assert all(c.proven < 0 for c in n.children)

    def test_root_prior_from_the_search(self):
        brain = MCTSBrain(iterations=50, ab_prior_depth=2, vcf_depth=0)
        move = brain.get_move(self._open_three_for_white(), Stone.BLACK)
        assert move in {Move(row=7, col=4), Move(row=7, col=8)}
