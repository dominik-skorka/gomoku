from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.evaluate import evaluate_board


class TestEvaluate:
    def test_empty_board_zero(self):
        b = Board(size=9)
        assert evaluate_board(b, Stone.BLACK) == 0.0

    def test_advantage_positive(self):
        """A board with more black stones near each other should score positive for black."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=5), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=6), Stone.BLACK)
        score = evaluate_board(b, Stone.BLACK)
        assert score > 0

    def test_opponent_advantage_negative(self):
        """A board where white has a strong line should score negative for black."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.WHITE)
        b = b.place_stone(Move(row=4, col=5), Stone.WHITE)
        b = b.place_stone(Move(row=4, col=6), Stone.WHITE)
        b = b.place_stone(Move(row=4, col=7), Stone.WHITE)
        score = evaluate_board(b, Stone.BLACK)
        assert score < 0
