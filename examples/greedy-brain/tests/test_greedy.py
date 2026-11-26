"""Tests of the example greedy brain."""

import pytest
from gomoku.board import Board
from gomoku.game import Game
from gomoku.models import GameConfig, Move, Stone
from gomoku_brain import BRAINS, get_brain
from greedy_brain import GreedyBrain, GreedySettings
from pydantic import ValidationError


def board_with(black: list[tuple[int, int]], white: list[tuple[int, int]]) -> Board:
    board = Board()
    for r, c in black:
        board = board.place_stone(Move(row=r, col=c), Stone.BLACK)
    for r, c in white:
        board = board.place_stone(Move(row=r, col=c), Stone.WHITE)
    return board


def test_registered():
    assert BRAINS["greedy"].brain_cls is GreedyBrain
    assert BRAINS["greedy"].settings_cls is GreedySettings


def test_settings_are_validated():
    assert get_brain("greedy", radius=1).settings.radius == 1
    with pytest.raises(ValidationError):
        GreedyBrain(radius=9)


def test_takes_a_win():
    board = board_with(black=[(7, 3), (7, 4), (7, 5), (7, 6)], white=[(8, 3), (8, 4), (8, 5), (8, 6)])
    assert GreedyBrain().get_move(board, Stone.BLACK) in {Move(row=7, col=2), Move(row=7, col=7)}


def test_blocks_a_win():
    board = board_with(black=[(7, 3), (7, 4), (7, 5), (7, 6)], white=[(0, 0), (0, 14), (14, 0)])
    assert GreedyBrain().get_move(board, Stone.WHITE) in {Move(row=7, col=2), Move(row=7, col=7)}


def test_plays_a_whole_game():
    game = Game(GameConfig(), GreedyBrain(), get_brain("random"))
    result = game.play()
    assert result is not None
