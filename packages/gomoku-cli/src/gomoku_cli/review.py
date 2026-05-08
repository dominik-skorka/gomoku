"""Renderers for the chess-engine-style deep review.

The pure data-processing layer (:func:`gomoku.analyze.review_game`)
produces a :class:`gomoku.analyze.DeepAnalysisReport`; this module turns
that report into:

* a rich-formatted terminal display (always on),
* a self-contained Markdown report (``--out path.md``),
* a structured JSON document (``--json``).

Three independent renderers so the user can pick any combination
without coupling.  All of them read from :class:`DeepAnalysisReport`
and never mutate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gomoku.analyze import (
    Advantage,
    Classification,
    DeepAnalysisReport,
)
from gomoku.models import Stone
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

# Per-classification tag used in both the rich table and the Markdown
# report.  Style strings are rich markup; the Markdown writer strips the
# bracketed segments and keeps the human label.
_CLASS_BADGE: dict[Classification, tuple[str, str]] = {
    Classification.BEST: ("[bold green]BEST[/bold green]", "BEST"),
    Classification.EXCELLENT: ("[green]excellent[/green]", "excellent"),
    Classification.GOOD: ("[cyan]good[/cyan]", "good"),
    Classification.INACCURACY: ("[yellow]inaccuracy[/yellow]", "inaccuracy"),
    Classification.MISTAKE: ("[red]mistake[/red]", "mistake"),
    Classification.BLUNDER: ("[bold red]BLUNDER[/bold red]", "BLUNDER"),
}

_ADVANTAGE_LABEL: dict[Advantage, str] = {
    Advantage.BLACK_WINNING: "Black ±",
    Advantage.BLACK_BETTER: "Black +",
    Advantage.EQUAL: "≈",
    Advantage.WHITE_BETTER: "White +",
    Advantage.WHITE_WINNING: "White ±",
}


def _format_eval(eval_black: float) -> str:
    """Format a Black-perspective signed eval as ``+1.2`` / ``-3.4`` (1000-scaled)."""
    scaled = eval_black / 1000.0
    return f"{scaled:+.1f}"


def _format_delta(delta: float) -> str:
    return f"{delta / 1000.0:.1f}" if delta else "0.0"


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------


def render_deep_report(report: DeepAnalysisReport, console: Console) -> None:
    """Print the deep review to *console* (rich-formatted).

    Layout:
      1. Header line (path, analysis brain + depth).
      2. Per-player accuracy / classification distribution table.
      3. Per-ply review table with classification badges.
      4. Quick legend.

    The pre-existing heuristic flags are *not* re-rendered here — the
    caller can still get them via the legacy ``analyze`` command if the
    light flagging output is wanted alongside.
    """
    console.print(f"\n[bold]Game review:[/bold] {report.path}")
    console.print(
        f"[dim]Analysis brain:[/dim] {report.analysis_brain}"
        f" (depth={report.analysis_depth})"
    )

    # ── Per-player summary ────────────────────────────────────────────
    summary_table = Table(title="Per-player accuracy", title_justify="left")
    summary_table.add_column("Stone", style="bold")
    summary_table.add_column("Provider")
    summary_table.add_column("Moves", justify="right")
    summary_table.add_column("Accuracy %", justify="right")
    for cls in Classification:
        summary_table.add_column(cls.value, justify="right")

    for stone in (Stone.BLACK, Stone.WHITE):
        ps = report.player_stats[stone]
        counts = report.classification_counts.get(stone, {})
        acc = report.accuracy.get(stone, 0.0)
        row = [
            str(stone),
            ps.provider or "-",
            str(ps.moves),
            f"{acc:.1f}",
        ]
        for cls in Classification:
            row.append(str(counts.get(cls, 0)))
        summary_table.add_row(*row)
    console.print(summary_table)

    # ── Per-ply review table ──────────────────────────────────────────
    if not report.reviews:
        console.print("\n[yellow]No moves reviewed.[/yellow]")
        return

    review_table = Table(
        title=f"Move-by-move review ({len(report.reviews)} plies)",
        title_justify="left",
    )
    review_table.add_column("#", justify="right", style="bold")
    review_table.add_column("Stone")
    review_table.add_column("Played")
    review_table.add_column("Engine best")
    review_table.add_column("Eval", justify="right")
    review_table.add_column("Δ", justify="right")
    review_table.add_column("Class")
    review_table.add_column("Advantage")
    review_table.add_column("Headline", style="dim")

    for review in report.reviews:
        headline = ""
        if review.commentary is not None:
            try:
                headline = review.commentary.headline
            except AttributeError:
                headline = str(review.commentary)
        review_table.add_row(
            str(review.move_number),
            str(review.stone),
            str(review.played_move),
            str(review.engine_best_move),
            _format_eval(review.eval_after_played),
            _format_delta(review.delta),
            _CLASS_BADGE[review.classification][0],
            _ADVANTAGE_LABEL[review.advantage],
            headline,
        )
    console.print(review_table)

    # ── Legend ────────────────────────────────────────────────────────
    console.print(
        "\n[dim]Eval shown from Black's perspective (scaled by 1000); "
        "delta is mover-perspective drop vs. engine best.  "
        "Classification cutoffs follow ClassificationThresholds.[/dim]"
    )


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def write_markdown_report(report: DeepAnalysisReport, out: Path) -> None:
    """Write *report* to *out* as a self-contained Markdown document."""
    lines: list[str] = []
    lines.append(f"# Gomoku game review — {report.path.name}")
    lines.append("")
    lines.append(
        f"_Analysis brain:_ **{report.analysis_brain}** "
        f"(depth={report.analysis_depth})"
    )
    if report.summary is not None:
        lines.append(f"_Result:_ {report.summary.get('ended_at', report.summary)}")
    lines.append("")

    # Per-player summary
    lines.append("## Player accuracy")
    lines.append("")
    cls_headers = " | ".join(c.value for c in Classification)
    lines.append(f"| Stone | Provider | Moves | Accuracy % | {cls_headers} |")
    lines.append("|---|---|---:|---:|" + "---:|" * len(Classification))
    for stone in (Stone.BLACK, Stone.WHITE):
        ps = report.player_stats[stone]
        counts = report.classification_counts.get(stone, {})
        acc = report.accuracy.get(stone, 0.0)
        cells = [
            str(stone),
            ps.provider or "-",
            str(ps.moves),
            f"{acc:.1f}",
        ]
        cells.extend(str(counts.get(c, 0)) for c in Classification)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Per-ply table
    lines.append("## Move-by-move")
    lines.append("")
    lines.append("| # | Stone | Played | Engine best | Eval | Δ | Class | Advantage |")
    lines.append("|---:|---|---|---|---:|---:|---|---|")
    for review in report.reviews:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(review.move_number),
                    str(review.stone),
                    str(review.played_move),
                    str(review.engine_best_move),
                    _format_eval(review.eval_after_played),
                    _format_delta(review.delta),
                    _CLASS_BADGE[review.classification][1],
                    _ADVANTAGE_LABEL[review.advantage],
                ]
            )
            + " |"
        )
    lines.append("")

    # Optional commentary section
    annotated = [r for r in report.reviews if r.commentary is not None]
    if annotated:
        lines.append("## Commentary")
        lines.append("")
        for review in annotated:
            lines.append(f"### Move {review.move_number} — {review.stone}")
            lines.append("")
            commentary = review.commentary
            headline = getattr(commentary, "headline", str(commentary))
            explanation = getattr(commentary, "explanation", "")
            threats = getattr(commentary, "key_threats", []) or []
            lines.append(f"**{headline}**")
            lines.append("")
            if explanation:
                lines.append(explanation)
                lines.append("")
            if threats:
                lines.append("Key threats: " + ", ".join(threats))
                lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def render_json_report(report: DeepAnalysisReport) -> str:
    """Serialise *report* as an indented JSON string (stable across calls)."""
    return json.dumps(report.to_json(), indent=2, default=str)


def write_json_report(report: DeepAnalysisReport, out: Path) -> None:
    out.write_text(render_json_report(report), encoding="utf-8")


__all__ = [
    "render_deep_report",
    "render_json_report",
    "write_json_report",
    "write_markdown_report",
]


# Suppress unused-import warning when this module is only imported for
# its side effects (it isn't, but keeping the import nice for IDEs).
_: Any = None
