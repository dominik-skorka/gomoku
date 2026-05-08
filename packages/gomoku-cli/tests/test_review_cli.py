"""CLI integration tests for ``gomoku review``."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner
from gomoku.diagnostics import DiagnosticsHeader, MoveDiagnostics
from gomoku.models import GameConfig, Move, Stone
from gomoku_cli.main import cli


def _write_diag(tmp_path: Path, plies: int = 4, size: int = 7) -> Path:
    header = DiagnosticsHeader(
        started_at=datetime.now(tz=UTC),
        game_config=GameConfig(board_size=size, win_length=5),
        players=[
            {"stone": int(Stone.BLACK), "provider": "random", "settings": {}},
            {"stone": int(Stone.WHITE), "provider": "random", "settings": {}},
        ],
    )
    moves = []
    # Place moves diagonally so they don't overlap or trigger 5-in-a-row.
    coords = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    for i in range(plies):
        r, c = coords[i]
        moves.append(
            MoveDiagnostics(
                move_number=i + 1,
                stone=Stone.BLACK if i % 2 == 0 else Stone.WHITE,
                chosen_move=Move(row=r, col=c),
            )
        )
    path = tmp_path / "g.diag.jsonl"
    lines = [header.model_dump_json()] + [m.model_dump_json() for m in moves]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestReviewCommand:
    def test_runs_with_random_brain(self, tmp_path: Path) -> None:
        # ``random`` is fast and ignores depth — works on any board state.
        path = _write_diag(tmp_path, plies=4)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["review", str(path), "--analysis-brain", "random"],
        )
        assert result.exit_code == 0, result.output
        assert "Game review" in result.output
        assert "Move-by-move review" in result.output

    def test_writes_markdown(self, tmp_path: Path) -> None:
        path = _write_diag(tmp_path, plies=3)
        out = tmp_path / "out.md"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "review",
                str(path),
                "--analysis-brain",
                "random",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        assert text.startswith("# Gomoku game review")
        assert "## Move-by-move" in text

    def test_json_mode_produces_parseable_stdout(self, tmp_path: Path) -> None:
        path = _write_diag(tmp_path, plies=2)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "review",
                str(path),
                "--analysis-brain",
                "random",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "reviews" in data
        assert data["analysis_brain"] == "random"
        assert len(data["reviews"]) == 2

    def test_no_llm_does_not_import_pydantic_ai(self, tmp_path: Path) -> None:
        """When --llm is absent, pydantic-ai must not be imported.

        Guards the optional-dependency contract: users without the
        ``[llm]`` extra installed should never hit a missing-import error.
        """
        path = _write_diag(tmp_path, plies=2)

        # Snapshot any existing pydantic-ai modules so we can compare.
        before = {k for k in sys.modules if k.startswith("pydantic_ai")}

        runner = CliRunner()
        result = runner.invoke(
            cli, ["review", str(path), "--analysis-brain", "random"]
        )
        assert result.exit_code == 0, result.output

        after = {k for k in sys.modules if k.startswith("pydantic_ai")}
        # Make sure no *new* pydantic-ai modules were imported.
        assert after == before, f"Unexpectedly imported: {after - before}"

    def test_unknown_brain_errors(self, tmp_path: Path) -> None:
        path = _write_diag(tmp_path, plies=2)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["review", str(path), "--analysis-brain", "no-such-brain"],
        )
        assert result.exit_code != 0
        assert "Unknown brain" in result.output
