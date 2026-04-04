"""Test that each AI brain can find an obvious winning move."""

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.negamax import NegamaxBrain


def _board_with_four_in_a_row(stone: Stone) -> Board:
    """Create a 9x9 board where ``stone`` has 4 in a row at row 4, cols 2-5.
    Placing at (4, 6) wins."""
    b = Board(size=9)
    for c in range(2, 6):
        b = b.place_stone(Move(row=4, col=c), stone)
    # Place some opponent stones elsewhere
    opponent = stone.opponent()
    b = b.place_stone(Move(row=0, col=0), opponent)
    b = b.place_stone(Move(row=0, col=1), opponent)
    b = b.place_stone(Move(row=0, col=2), opponent)
    return b


class TestMinimaxFindsWin:
    def test_finds_winning_move(self):
        b = _board_with_four_in_a_row(Stone.BLACK)
        brain = MinimaxBrain(depth=2)
        move = brain.get_move(b, Stone.BLACK)
        # Should play at col=1 or col=6 to complete 5
        assert move in (Move(row=4, col=1), Move(row=4, col=6))


class TestNegamaxFindsWin:
    def test_finds_winning_move(self):
        b = _board_with_four_in_a_row(Stone.BLACK)
        brain = NegamaxBrain(depth=2)
        move = brain.get_move(b, Stone.BLACK)
        assert move in (Move(row=4, col=1), Move(row=4, col=6))


class TestAlphaBetaFindsWin:
    def test_finds_winning_move(self):
        b = _board_with_four_in_a_row(Stone.BLACK)
        brain = AlphaBetaBrain(depth=2)
        move = brain.get_move(b, Stone.BLACK)
        assert move in (Move(row=4, col=1), Move(row=4, col=6))
