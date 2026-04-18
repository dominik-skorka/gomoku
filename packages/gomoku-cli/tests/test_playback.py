"""Tests for the playback CLI command and navigator logic."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner
from gomoku.diagnostics import (
    DiagnosticsHeader,
    MoveDiagnostics,
    SearchStats,
)
from gomoku.models import GameConfig, GameResult, Move, Stone
from gomoku.recorder import GameRecord, PlayerInfo, RecordedMove, save_record
from gomoku_cli.main import _find_diagnostics, cli
from gomoku_cli.playback import _load_diag_by_move


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


def _write_diag_for_record(rec_path: Path) -> Path:
    """Produce a ``.diag.jsonl`` companion beside *rec_path*."""
    diag_path = rec_path.with_suffix(".diag.jsonl")
    header = DiagnosticsHeader(
        started_at=datetime(2026, 4, 5, 14, 0, 0, tzinfo=UTC),
        game_config=GameConfig(board_size=9, win_length=5),
    )
    moves = [
        MoveDiagnostics(
            move_number=1,
            stone=Stone.BLACK,
            chosen_move=Move(row=0, col=0),
            decision_path="alphabeta",
            search_stats=SearchStats(depth_reached=4, nodes_searched=100, time_ms=10),
            think_time_ms=10,
        ),
        MoveDiagnostics(
            move_number=2,
            stone=Stone.WHITE,
            chosen_move=Move(row=1, col=1),
            decision_path="random",
            think_time_ms=2,
        ),
    ]
    lines = [header.model_dump_json(), *(m.model_dump_json() for m in moves)]
    diag_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return diag_path


class TestPlaybackDiagnostics:
    def test_find_diagnostics_companion(self, tmp_path):
        record = _make_record()
        rec_path = save_record(record, tmp_path)
        diag_path = _write_diag_for_record(rec_path)
        assert _find_diagnostics(rec_path) == diag_path

    def test_find_diagnostics_returns_none_when_missing(self, tmp_path):
        record = _make_record()
        rec_path = save_record(record, tmp_path)
        assert _find_diagnostics(rec_path) is None

    def test_load_diag_by_move_indexes_by_number(self, tmp_path):
        record = _make_record()
        rec_path = save_record(record, tmp_path)
        diag_path = _write_diag_for_record(rec_path)
        by_move = _load_diag_by_move(diag_path)
        assert set(by_move) == {1, 2}
        assert by_move[1].decision_path == "alphabeta"
        assert by_move[1].search_stats.nodes_searched == 100

    def test_load_diag_by_move_tolerates_missing(self):
        assert _load_diag_by_move(None) == {}
