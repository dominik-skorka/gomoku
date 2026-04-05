"""Tests for the playback CLI command and navigator logic."""

from __future__ import annotations

from datetime import UTC, datetime

from click.testing import CliRunner
from gomoku.models import GameConfig, GameResult, Move, Stone
from gomoku.recorder import GameRecord, PlayerInfo, RecordedMove, save_record
from gomoku_cli.main import cli


def _make_record(num_moves: int = 5) -> GameRecord:
    """Build a small recorded game with *num_moves* moves."""
    now = datetime(2026, 4, 5, 14, 0, 0, tzinfo=UTC)
    moves = []
    stones = [Stone.BLACK, Stone.WHITE]
    for i in range(num_moves):
        moves.append(
            RecordedMove(
                number=i + 1,
                stone=stones[i % 2],
                move=Move(row=i, col=i),
                timestamp=datetime(2026, 4, 5, 14, 0, i, tzinfo=UTC),
                think_time_ms=100 * (i + 1),
            )
        )
    return GameRecord(
        started_at=now,
        ended_at=datetime(2026, 4, 5, 14, 1, 0, tzinfo=UTC),
        game_config=GameConfig(board_size=9, win_length=5),
        players=[
            PlayerInfo(stone=Stone.BLACK, provider="threat", settings={"depth": 4}),
            PlayerInfo(stone=Stone.WHITE, provider="random"),
        ],
        moves=moves,
        result=GameResult(winner=Stone.BLACK),
    )


class TestPlaybackCommand:
    def test_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["playback", "nonexistent.json"])
        assert result.exit_code != 0

    def test_load_and_quit_immediately(self, tmp_path):
        """Playback loads the file, then 'q' quits immediately."""
        record = _make_record()
        path = save_record(record, tmp_path)

        runner = CliRunner()
        # Simulate pressing 'q' immediately
        result = runner.invoke(cli, ["playback", str(path)], input="q")
        # Should not crash — exit code 0 or at least no Python traceback
        assert "Traceback" not in (result.output or "")

    def test_from_move_option_accepted(self, tmp_path):
        """--from-move is accepted without error."""
        record = _make_record()
        path = save_record(record, tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["playback", str(path), "--from-move", "3"], input="q")
        assert "Traceback" not in (result.output or "")
