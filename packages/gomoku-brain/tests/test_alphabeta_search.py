"""Tests for invariants of the alpha-beta search itself."""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.search_state import SearchState


def _midgame_state() -> SearchState:
    moves = [(7, 7), (7, 8), (8, 7), (6, 8), (9, 6), (8, 8), (6, 6), (5, 5), (9, 9), (6, 9)]
    board = Board(size=15)
    stone = Stone.BLACK
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return SearchState.from_board(board, stone)


class TestEvalCachePerspective:
    """Every eval-cache entry is from the perspective of the side to move.

    Futility pruning, null-move pruning and quiescence read the cache that
    way.  Move ordering used to store the mover's perspective instead, so
    those readers got sign-flipped evaluations for ordered positions.
    """

    def test_ordering_stores_side_to_move_evaluations(self):
        state = _midgame_state()
        stone = Stone.BLACK
        cache: dict[int, float] = {}
        candidates = get_candidate_moves(state, radius=2)
        AlphaBetaBrain(depth=4)._order_moves(state, candidates, stone, 3, {}, eval_cache=cache)
        assert cache
        for move in candidates:
            state.make(move.row, move.col, stone)
            try:
                assert cache[state.zobrist] == evaluate_board(state, stone.opponent())
            finally:
                state.unmake()

    def test_ordering_ranks_by_the_movers_evaluation(self):
        state = _midgame_state()
        stone = Stone.BLACK
        cache: dict[int, float] = {}
        candidates = get_candidate_moves(state, radius=2)
        brain = AlphaBetaBrain(depth=4, threat_ordering=False)
        first = brain._order_moves(state, candidates, stone, 3, {}, eval_cache=cache)
        # A second pass reads every score back from the cache; the order
        # must not change.
        assert brain._order_moves(state, candidates, stone, 3, {}, eval_cache=cache) == first
        best = max(candidates, key=lambda m: _eval_after(state, m, stone))
        assert first[0] == best


def _eval_after(state: SearchState, move: Move, stone: Stone) -> float:
    state.make(move.row, move.col, stone)
    try:
        return evaluate_board(state, stone)
    finally:
        state.unmake()
