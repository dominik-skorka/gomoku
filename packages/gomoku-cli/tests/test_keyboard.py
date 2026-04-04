from unittest.mock import patch

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_cli.keyboard import KeyboardProvider


class TestKeyboardProvider:
    def test_valid_move(self):
        b = Board(size=9)
        provider = KeyboardProvider()
        with patch("gomoku_cli.keyboard.Prompt.ask", return_value="E5"):
            move = provider.get_move(b, Stone.BLACK)
        assert move == Move(row=4, col=4)

    def test_invalid_then_valid(self):
        b = Board(size=9)
        # First call returns bad format, second returns valid
        provider = KeyboardProvider()
        with patch("gomoku_cli.keyboard.Prompt.ask", side_effect=["xyz", "A1"]):
            move = provider.get_move(b, Stone.BLACK)
        assert move == Move(row=0, col=0)

    def test_name(self):
        assert KeyboardProvider().name == "keyboard"
