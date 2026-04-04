from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.random import RandomBrain


class TestRandomBrain:
    def test_name(self):
        assert RandomBrain().name == "random"

    def test_returns_valid_move(self):
        b = Board(size=9)
        brain = RandomBrain()
        move = brain.get_move(b, Stone.BLACK)
        assert b.is_valid_move(move)

    def test_avoids_occupied(self):
        b = Board(size=5)
        # Fill all but one cell
        for r in range(5):
            for c in range(5):
                if (r, c) != (2, 3):
                    b = b.place_stone(Move(row=r, col=c), Stone.BLACK if (r + c) % 2 == 0 else Stone.WHITE)
        brain = RandomBrain()
        move = brain.get_move(b, Stone.BLACK)
        assert move == Move(row=2, col=3)
