from io import StringIO
from unittest.mock import patch

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_cli.display import render_board
from rich.console import Console


class TestDisplay:
    def test_render_empty_board(self):
        """Smoke test: render_board should not crash on an empty board."""
        b = Board(size=9)
        buf = StringIO()
        with patch("gomoku_cli.display.console", new=Console(file=buf, force_terminal=True)):
            render_board(b)
        output = buf.getvalue()
        # Should contain column labels
        assert "A" in output
        # Should contain row labels
        assert "1" in output

    def test_render_with_stones(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=5), Stone.WHITE)
        buf = StringIO()
        with patch("gomoku_cli.display.console", new=Console(file=buf, force_terminal=True)):
            render_board(b, last_move=Move(row=4, col=5), turn=2, current_stone=Stone.BLACK)
        output = buf.getvalue()
        assert "●" in output
        assert "○" in output
