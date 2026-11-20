"""Tests for the zobrist-memoised scan context / candidate cache on SearchState."""

import numpy as np
import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import _should_use_jit
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import _scan_context

pytestmark = pytest.mark.skipif(not _should_use_jit(), reason="scan context requires JIT")


def _fresh_candidates(state: SearchState) -> np.ndarray:
    """Ground truth: derive the radius-2 candidate array without the memo."""
    from gomoku_brain.evaluate_jit import get_candidates_grid
    from gomoku_brain.grid_utils import occupied_coords

    return get_candidates_grid(state.grid, state.size, occupied_coords(state), 2)


def _mid_state() -> SearchState:
    board = Board(size=15)
    stone = Stone.BLACK
    for r, c in [(7, 7), (7, 8), (8, 7), (6, 8), (8, 8), (6, 7)]:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return SearchState.from_board(board, stone)


class TestScanContextMemo:
    def test_repeated_calls_reuse_candidates(self):
        state = _mid_state()
        ctx1 = _scan_context(state)
        ctx2 = _scan_context(state)
        assert ctx1 is not None and ctx2 is not None
        assert ctx1[2] is ctx2[2]  # same cached array object

    def test_memo_tracks_make_unmake(self):
        """After every make/unmake the context must match a fresh derivation."""
        state = _mid_state()
        moves = [(9, 9, Stone.BLACK), (5, 5, Stone.WHITE), (10, 10, Stone.BLACK)]
        for r, c, s in moves:
            _scan_context(state)  # populate memo at the pre-move position
            state.make(r, c, s)
            ctx = _scan_context(state)
            assert ctx is not None
            assert np.array_equal(ctx[2], _fresh_candidates(state))
        for _ in moves:
            state.unmake()
            ctx = _scan_context(state)
            assert ctx is not None
            assert np.array_equal(ctx[2], _fresh_candidates(state))

    def test_candidate_moves_match_and_are_copies(self):
        state = _mid_state()
        first = get_candidate_moves(state, radius=2)
        arr = _fresh_candidates(state)
        assert [(m.row, m.col) for m in first] == [(int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])]
        second = get_candidate_moves(state, radius=2)
        assert first == second
        assert first is not second  # callers may slice / re-order freely
        second.reverse()
        assert get_candidate_moves(state, radius=2) == first

    def test_radius_one_bypasses_memo(self):
        state = _mid_state()
        r2 = {(m.row, m.col) for m in get_candidate_moves(state, radius=2)}
        r1 = {(m.row, m.col) for m in get_candidate_moves(state, radius=1)}
        assert r1 < r2  # strictly fewer cells at radius 1 on this position
