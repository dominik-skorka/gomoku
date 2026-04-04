from gomoku.board import Board
from gomoku.models import Move, Stone


class TestBoardInit:
    def test_default_size(self):
        b = Board()
        assert b.size == 15
        assert len(b.grid) == 15
        assert all(len(row) == 15 for row in b.grid)

    def test_custom_size(self):
        b = Board(size=9)
        assert b.size == 9
        assert len(b.grid) == 9

    def test_all_empty(self):
        b = Board(size=5)
        for r in range(5):
            for c in range(5):
                assert b.at(r, c) is Stone.EMPTY


class TestBoardOps:
    def test_place_stone(self):
        b = Board(size=5)
        m = Move(row=2, col=3)
        b2 = b.place_stone(m, Stone.BLACK)
        # Original unchanged
        assert b.at(2, 3) is Stone.EMPTY
        # New board has the stone
        assert b2.at(2, 3) is Stone.BLACK

    def test_is_valid_move(self):
        b = Board(size=5)
        assert b.is_valid_move(Move(row=0, col=0))
        b2 = b.place_stone(Move(row=0, col=0), Stone.BLACK)
        assert not b2.is_valid_move(Move(row=0, col=0))

    def test_out_of_bounds(self):
        b = Board(size=5)
        assert not b.is_valid_move(Move(row=5, col=0))
        assert not b.is_valid_move(Move(row=0, col=5))

    def test_get_valid_moves(self):
        b = Board(size=5)
        assert len(b.get_valid_moves()) == 25
        b2 = b.place_stone(Move(row=0, col=0), Stone.BLACK)
        assert len(b2.get_valid_moves()) == 24

    def test_is_full(self):
        b = Board(size=5)
        assert not b.is_full()
        # Fill the board
        current = Stone.BLACK
        for r in range(5):
            for c in range(5):
                b = b.place_stone(Move(row=r, col=c), current)
                current = current.opponent()
        assert b.is_full()

    def test_stone_count(self):
        b = Board(size=5)
        assert b.stone_count() == 0
        b = b.place_stone(Move(row=0, col=0), Stone.BLACK)
        assert b.stone_count() == 1
