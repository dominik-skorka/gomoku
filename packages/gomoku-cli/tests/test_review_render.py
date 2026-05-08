"""Tests for the deep-review renderers (terminal / Markdown / JSON)."""

from __future__ import annotations

import json
import typing
from datetime import UTC, datetime
from pathlib import Path

from gomoku.analyze import (
    Advantage,
    AnalysisReport,
    Classification,
    DeepAnalysisReport,
    MoveReview,
    PlayerStats,
)
from gomoku.models import Move, Stone
from gomoku_cli.review import (
    render_deep_report,
    render_json_report,
    write_markdown_report,
)
from rich.console import Console


def _make_report(*, with_commentary: bool = False) -> DeepAnalysisReport:
    """Build a tiny DeepAnalysisReport without going through review_game."""
    base = AnalysisReport(
        path=Path("game.diag.jsonl"),
        header={"started_at": datetime.now(tz=UTC).isoformat()},
        summary={"total_moves": 2},
        moves=[],
        player_stats={
            Stone.BLACK: PlayerStats(stone=Stone.BLACK, provider="alphabeta", moves=1),
            Stone.WHITE: PlayerStats(stone=Stone.WHITE, provider="random", moves=1),
        },
        flagged=[],
    )

    class _FakeCommentary:
        headline = "Looks reasonable"
        explanation = "Played the same move the engine prefers."
        key_threats: typing.ClassVar[list[str]] = ["live-2 on row 8"]

    reviews = [
        MoveReview(
            move_number=1,
            stone=Stone.BLACK,
            played_move=Move(row=7, col=7),
            engine_best_move=Move(row=7, col=7),
            eval_before=0.0,
            eval_after_played=0.0,
            eval_after_best=0.0,
            delta=0.0,
            classification=Classification.BEST,
            advantage=Advantage.EQUAL,
            commentary=_FakeCommentary() if with_commentary else None,
        ),
        MoveReview(
            move_number=2,
            stone=Stone.WHITE,
            played_move=Move(row=8, col=8),
            engine_best_move=Move(row=8, col=9),
            eval_before=0.0,
            eval_after_played=-15_000.0,
            eval_after_best=-2_000.0,
            delta=13_000.0,
            classification=Classification.BLUNDER,
            advantage=Advantage.BLACK_WINNING,
        ),
    ]
    return DeepAnalysisReport.from_base(
        base, reviews, analysis_brain="alphabeta", analysis_depth=6
    )


class TestTerminalRenderer:
    def test_runs_and_prints_key_sections(self) -> None:
        report = _make_report(with_commentary=True)
        console = Console(record=True, width=200)
        render_deep_report(report, console)
        out = console.export_text()
        assert "Game review" in out
        assert "alphabeta" in out
        assert "Per-player accuracy" in out
        assert "Move-by-move review" in out
        # Classification badges
        assert "BLUNDER" in out
        assert "BEST" in out
        # Commentary headline column
        assert "Looks reasonable" in out

    def test_handles_empty_reviews(self) -> None:
        report = _make_report()
        report.reviews = []
        console = Console(record=True, width=200)
        render_deep_report(report, console)
        out = console.export_text()
        assert "No moves reviewed" in out


class TestMarkdownWriter:
    def test_writes_expected_sections(self, tmp_path: Path) -> None:
        report = _make_report(with_commentary=True)
        out = tmp_path / "review.md"
        write_markdown_report(report, out)
        text = out.read_text(encoding="utf-8")
        assert text.startswith("# Gomoku game review")
        assert "## Player accuracy" in text
        assert "## Move-by-move" in text
        assert "## Commentary" in text
        # Classification labels make it through (ANSI-stripped).
        assert "BLUNDER" in text
        # Move 1 commentary headline
        assert "Looks reasonable" in text

    def test_no_commentary_section_when_empty(self, tmp_path: Path) -> None:
        report = _make_report(with_commentary=False)
        out = tmp_path / "review.md"
        write_markdown_report(report, out)
        assert "## Commentary" not in out.read_text(encoding="utf-8")


class TestJSONWriter:
    def test_round_trip(self) -> None:
        report = _make_report(with_commentary=True)
        text = render_json_report(report)
        data = json.loads(text)
        assert data["analysis_brain"] == "alphabeta"
        assert data["analysis_depth"] == 6
        assert len(data["reviews"]) == 2
        assert data["reviews"][1]["classification"] == "blunder"
        assert "accuracy" in data
        assert "classification_counts" in data
