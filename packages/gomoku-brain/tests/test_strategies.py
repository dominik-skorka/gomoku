"""Pluggable search strategies (#77): the Evaluator / MoveGenerator protocols."""

import random

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.candidates import get_candidate_moves, radius_generator
from gomoku_brain.search_state import SearchState


def _random_board(seed: int, stones: int, size: int = 15) -> Board:
    rng = random.Random(seed)
    board = Board(size=size)
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)
    stone = Stone.BLACK
    for r, c in cells[:stones]:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return board


@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_radius_generator_is_get_candidate_moves(radius: int, seed: int) -> None:
    board = _random_board(seed, stones=8)
    generate = radius_generator(radius)
    for stone in (Stone.BLACK, Stone.WHITE):
        assert list(generate(board, stone)) == get_candidate_moves(board, radius)


@pytest.mark.parametrize("seed", range(3))
def test_radius_generator_on_search_state(seed: int) -> None:
    board = _random_board(seed, stones=10)
    state = SearchState.from_board(board, Stone.BLACK)
    assert list(radius_generator()(state, Stone.BLACK)) == get_candidate_moves(state)


def test_radius_generator_empty_board_is_centre() -> None:
    assert list(radius_generator()(Board(size=15), Stone.BLACK)) == [Move(row=7, col=7)]
