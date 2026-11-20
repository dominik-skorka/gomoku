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


class _Log:
    """A GameObserver that logs what it is told, in order."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def on_move(self, stone: Stone, move: Move) -> None:
        self.events.append((stone, move))

    def on_game_end(self, result) -> None:
        self.events.append(result)


def _five_in_a_row_game(**kwargs) -> Game:
    black = _ScriptedProvider([Move(row=7, col=c) for c in range(5)], "black")
    white = _ScriptedProvider([Move(row=0, col=c) for c in range(5)], "white")
    return Game(GameConfig(), black, white, **kwargs)


class TestObservers:
    def test_every_observer_sees_every_move_then_the_end(self):
        first, second = _Log(), _Log()
        game = _five_in_a_row_game(observers=[first, second])
        result = game.play()
        assert first.events == second.events
        assert first.events[-1] == result == game.result
        assert first.events[:-1] == game.move_history

    def test_recorder_is_finalized_by_the_game(self, tmp_path):
        from gomoku.recorder import GameRecorder, load_record

        recorder = GameRecorder(game_config=GameConfig(), players=[], directory=tmp_path)
        _five_in_a_row_game(observers=[recorder]).play()
        record = load_record(recorder.path)
        assert record.result is not None and record.result.winner is Stone.BLACK
        assert len(record.moves) == 9
        assert recorder.path.with_suffix(".md5sum").exists()

    def test_diagnostics_get_one_line_per_move_and_the_summary_last(self, tmp_path):
        import json

        from gomoku.diagnostics import DiagnosticsRecorder

        diag = DiagnosticsRecorder(game_config=GameConfig(), players=[], directory=tmp_path)
        _five_in_a_row_game(diagnostics=diag).play()
        lines = [json.loads(line) for line in diag.path.read_text().splitlines()]
        moves = lines[1:-1]
        assert [m["move_number"] for m in moves] == list(range(1, 10))
        assert all(m["chosen_move"] is not None for m in moves)
        assert lines[-1]["type"] == "summary"
        assert (lines[-1]["total_moves"], lines[-1]["winner"], lines[-1]["is_draw"]) == (9, 1, False)

    def test_an_illegal_move_is_noted_in_its_diagnostics(self, tmp_path):
        import json

        from gomoku.diagnostics import DiagnosticsRecorder

        diag = DiagnosticsRecorder(game_config=GameConfig(), players=[], directory=tmp_path)
        game = Game(GameConfig(), _BadProvider(), _BadProvider(), diagnostics=diag)
        game.play_turn()
        with pytest.raises(InvalidMoveError):
            game.play_turn()
        last = json.loads(diag.path.read_text().splitlines()[-1])
        assert any("InvalidMoveError" in note for note in last["notes"])
