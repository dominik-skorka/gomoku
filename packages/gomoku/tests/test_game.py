import pytest
from gomoku.board import Board
from gomoku.game import Game, InvalidMoveError
from gomoku.models import GameConfig, Move, Stone


class _ScriptedProvider:
    """A MoveProvider that plays a scripted sequence of moves."""

    def __init__(self, moves: list[Move], provider_name: str = "scripted") -> None:
        self._moves = list(moves)
        self._index = 0
        self._name = provider_name

    @property
    def name(self) -> str:
        return self._name

    def get_move(self, board: Board, stone: Stone) -> Move:
        move = self._moves[self._index]
        self._index += 1
        return move


class _BadProvider:
    """Always returns an invalid move."""

    @property
    def name(self) -> str:
        return "bad"

    def get_move(self, board: Board, stone: Stone) -> Move:
        return Move(row=0, col=0)


class TestGame:
    def test_black_wins_horizontal(self):
        """Black plays 5 in a row horizontally, White plays elsewhere."""
        black_moves = [Move(row=7, col=c) for c in range(5)]
        white_moves = [Move(row=0, col=c) for c in range(5)]

        black = _ScriptedProvider(black_moves, "black-script")
        white = _ScriptedProvider(white_moves, "white-script")
        game = Game(GameConfig(), black, white)
        result = game.play()

        assert result.winner is Stone.BLACK
        assert len(game.move_history) == 9  # Black plays 5, White plays 4

    def test_draw_detection(self):
        """On a 5x5 board with no winner, should end in draw."""
        config = GameConfig(board_size=5)
        # Generate moves that fill the board without creating 5-in-a-row
        # Row pattern: BB WW B / WW BB W / BB WW B / WW BB W / BB WW B
        patterns = [
            [
                (0, 0),
                (0, 1),
                (0, 4),
                (2, 0),
                (2, 1),
                (2, 4),
                (4, 0),
                (4, 1),
                (4, 4),
                (1, 2),
                (1, 3),
                (3, 2),
                (3, 3),
            ],  # Black: 13
            [
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 4),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 4),
                (4, 2),
                (4, 3),
            ],  # White: 12
        ]
        black = _ScriptedProvider([Move(row=r, col=c) for r, c in patterns[0]])
        white = _ScriptedProvider([Move(row=r, col=c) for r, c in patterns[1]])
        game = Game(config, black, white)
        result = game.play()
        assert result.is_draw

    def test_invalid_move_raises(self):
        """Provider that returns an occupied cell should raise InvalidMoveError."""
        black_moves = [Move(row=0, col=0)]
        white = _BadProvider()  # Always returns (0,0) which will be occupied

        game = Game(GameConfig(), _ScriptedProvider(black_moves), white)
        # Black plays (0,0), then White tries (0,0) -> invalid
        with pytest.raises(InvalidMoveError):
            game.play()
