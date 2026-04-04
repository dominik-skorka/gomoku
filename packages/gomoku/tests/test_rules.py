from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner, get_game_status


def _place_stones(board: Board, moves: list[tuple[int, int]], stone: Stone) -> Board:
    """Helper: place a series of stones on the board."""
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
    return board


class TestCheckWinner:
    def test_horizontal_win(self):
        b = Board(size=15)
        positions = [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        b = _place_stones(b, positions, Stone.BLACK)
        result = check_winner(b, Move(row=7, col=7))
        assert result is not None
        assert result.winner is Stone.BLACK
        assert len(result.winning_moves) == 5

    def test_vertical_win(self):
        b = Board(size=15)
        positions = [(3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]
        b = _place_stones(b, positions, Stone.WHITE)
        result = check_winner(b, Move(row=5, col=5))
        assert result is not None
        assert result.winner is Stone.WHITE

    def test_diagonal_down_win(self):
        b = Board(size=15)
        positions = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        b = _place_stones(b, positions, Stone.BLACK)
        result = check_winner(b, Move(row=4, col=4))
        assert result is not None
        assert result.winner is Stone.BLACK

    def test_diagonal_up_win(self):
        b = Board(size=15)
        positions = [(6, 2), (5, 3), (4, 4), (3, 5), (2, 6)]
        b = _place_stones(b, positions, Stone.BLACK)
        result = check_winner(b, Move(row=4, col=4))
        assert result is not None
        assert result.winner is Stone.BLACK

    def test_no_winner(self):
        b = Board(size=15)
        positions = [(7, 3), (7, 4), (7, 5), (7, 6)]  # Only 4
        b = _place_stones(b, positions, Stone.BLACK)
        result = check_winner(b, Move(row=7, col=6))
        assert result is None

    def test_six_in_a_row_freestyle(self):
        """Freestyle: 6 in a row should still win."""
        b = Board(size=15)
        positions = [(7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        b = _place_stones(b, positions, Stone.BLACK)
        result = check_winner(b, Move(row=7, col=5))
        assert result is not None
        assert result.winner is Stone.BLACK
        assert len(result.winning_moves) == 6


class TestGetGameStatus:
    def test_ongoing(self):
        b = Board(size=15)
        b = b.place_stone(Move(row=7, col=7), Stone.BLACK)
        result = get_game_status(b, Move(row=7, col=7))
        assert result is None

    def test_draw(self):
        """Fill a small board without a winner."""
        # 5x5 board filled in a pattern that avoids 5-in-a-row
        b = Board(size=5)
        # Row-by-row: alternating pairs to avoid lines
        pattern = [
            [Stone.BLACK, Stone.BLACK, Stone.WHITE, Stone.WHITE, Stone.BLACK],
            [Stone.WHITE, Stone.WHITE, Stone.BLACK, Stone.BLACK, Stone.WHITE],
            [Stone.BLACK, Stone.BLACK, Stone.WHITE, Stone.WHITE, Stone.BLACK],
            [Stone.WHITE, Stone.WHITE, Stone.BLACK, Stone.BLACK, Stone.WHITE],
            [Stone.BLACK, Stone.BLACK, Stone.WHITE, Stone.WHITE, Stone.BLACK],
        ]
        last = Move(row=0, col=0)
        for r in range(5):
            for c in range(5):
                b = b.place_stone(Move(row=r, col=c), pattern[r][c])
                last = Move(row=r, col=c)
        result = get_game_status(b, last, win_length=5)
        assert result is not None
        assert result.is_draw
