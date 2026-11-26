"""The pattern evaluation with other weights (#105)."""

import random

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.pattern_weights import PatternWeights, pattern_evaluator
from gomoku_brain.search_state import SearchState

OTHER = PatternWeights(dead_four=3_000, live_three=1_700, fork_bonus=5_000, live_two=150)


def _random_board(seed: int) -> Board:
    rng = random.Random(seed)
    board = Board(size=15)
    cells = [(r, c) for r in range(15) for c in range(15)]
    rng.shuffle(cells)
    stone = Stone.BLACK
    for r, c in cells[: rng.randint(1, 60)]:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return board


@pytest.mark.parametrize("seed", range(40))
def test_default_weights_are_evaluate_board(seed: int) -> None:
    board = _random_board(seed)
    state = SearchState.from_board(board, Stone.BLACK)
    evaluate = pattern_evaluator(PatternWeights())
    for position in (board, state):
        for stone in (Stone.BLACK, Stone.WHITE):
            assert evaluate(position, stone) == evaluate_board(position, stone)


@pytest.mark.parametrize("seed", range(20))
def test_incremental_cache_matches_a_fresh_evaluation(seed: int) -> None:
    """make / unmake walks, interleaved with the default evaluator on the same state."""
    rng = random.Random(1000 + seed)
    board = _random_board(seed)
    state = SearchState.from_board(board, Stone.BLACK)
    other = pattern_evaluator(OTHER)
    stone = Stone.BLACK
    made = 0
    for _ in range(40):
        empty = [(r, c) for r in range(15) for c in range(15) if state.at(r, c) is Stone.EMPTY]
        if made and (rng.random() < 0.35 or not empty):
            state.unmake()
            made -= 1
            stone = stone.opponent()
        elif empty:
            r, c = rng.choice(empty)
            state.make(r, c, stone)
            made += 1
            stone = stone.opponent()
        fresh_board = Board(size=15)
        for r, c in state.occupied:
            fresh_board = fresh_board.place_stone(Move(row=r, col=c), state.at(r, c))
        fresh = SearchState.from_board(fresh_board, Stone.BLACK)
        # Alternate owners of the state's cache.
        assert other(state, Stone.BLACK) == other(fresh, Stone.BLACK)
        assert evaluate_board(state, Stone.WHITE) == evaluate_board(fresh_board, Stone.WHITE)
        assert other(state, Stone.WHITE) == other(fresh_board, Stone.WHITE)


def test_other_weights_change_the_score() -> None:
    other = pattern_evaluator(OTHER)
    boards = [_random_board(seed) for seed in range(20)]
    assert any(other(b, Stone.BLACK) != evaluate_board(b, Stone.BLACK) for b in boards)


@pytest.mark.parametrize("bad", [-1, 100_000])
def test_weights_out_of_range_are_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="dead_four"):
        PatternWeights(dead_four=bad)


def test_open_three_evaluator_is_registered() -> None:
    from gomoku.settings import AlphaBetaBrainSettings
    from gomoku.strategies import get_evaluator

    assert AlphaBetaBrainSettings(evaluator="pattern_open3").evaluator == "pattern_open3"
    evaluate = get_evaluator("pattern_open3")
    boards = [_random_board(seed) for seed in range(20)]
    assert any(evaluate(b, Stone.BLACK) != evaluate_board(b, Stone.BLACK) for b in boards)
