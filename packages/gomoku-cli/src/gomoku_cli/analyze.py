"""CLI front-end for the post-game analyser.

The pure data-processing layer lives in :mod:`gomoku.analyze` so that
both the CLI and the FastAPI package can reuse it without depending on
``rich``.  This module re-exports those symbols (so the existing
``from gomoku_cli.analyze import analyze_diagnostics, AnalysisReport``
imports keep working) and adds :func:`render_report`, which renders an
``AnalysisReport`` as a rich-formatted table on a console.
"""

from __future__ import annotations

from typing import Any

# Re-exports — preserve the historical import surface.
from gomoku.analyze import (  # noqa: F401  (re-exports for back-compat)
    ALTERNATIVE_GAP,
    BLUNDER_EVAL_DROP,
    TIME_SPIKE_FACTOR,
    TIME_SPIKE_MIN_MS,
    AnalysisReport,
    FlaggedMove,
    PlayerStats,
    analyze_diagnostics,
    analyze_diagnostics_text,
    load_diagnostics,
)


def render_report(report: AnalysisReport, console: Any) -> None:
    """Print a human-readable analysis report to *console* (rich.Console)."""
    from rich.table import Table

    console.print(f"\n[bold]Diagnostics analysis:[/bold] {report.path}")

    # -- Per-player stats table --------------------------------------------
    stats_table = Table(title="Per-player statistics", title_justify="left")
    stats_table.add_column("Stone", style="bold")
    stats_table.add_column("Provider")
    stats_table.add_column("Moves", justify="right")
    stats_table.add_column("Total time (ms)", justify="right")
    stats_table.add_column("Avg time (ms)", justify="right")
    stats_table.add_column("Avg depth", justify="right")
    stats_table.add_column("Max depth", justify="right")
    stats_table.add_column("Total nodes", justify="right")
    stats_table.add_column("Blunders", justify="right")
    from gomoku.models import Stone

    for stone in (Stone.BLACK, Stone.WHITE):
        ps = report.player_stats[stone]
        stats_table.add_row(
            str(stone),
            ps.provider or "-",
            str(ps.moves),
            str(ps.total_think_ms),
            f"{ps.avg_think_ms:.1f}",
            f"{ps.avg_depth:.2f}" if ps.depth_samples else "-",
            str(ps.max_depth_reached) if ps.depth_samples else "-",
            str(ps.total_nodes),
            str(ps.blunders),
        )
    console.print(stats_table)

    # -- Flagged moves -----------------------------------------------------
    if not report.flagged:
        console.print("\n[green]No suspect moves flagged.[/green]")
        return

    flag_table = Table(title=f"Flagged moves ({len(report.flagged)})", title_justify="left")
    flag_table.add_column("#", justify="right", style="bold")
    flag_table.add_column("Stone")
    flag_table.add_column("Path")
    flag_table.add_column("Reasons", style="yellow")
    flag_table.add_column("Details")
    for f in report.flagged:
        details_str = " ".join(f"{k}={v}" for k, v in f.details.items())
        flag_table.add_row(
            str(f.move_number),
            str(f.stone),
            f.decision_path or "-",
            ", ".join(f.reasons),
            details_str,
        )
    console.print(flag_table)

    # -- Legend ------------------------------------------------------------
    legend = (
        "\n[dim]Legend: "
        "eval_drop = own eval fell ≥ "
        f"{BLUNDER_EVAL_DROP:.0f}; "
        "bypassed_top_candidate = chose a move ≥ "
        f"{ALTERNATIVE_GAP:.0f} worse than the top eval; "
        "time_spike = think time ≥ "
        f"{TIME_SPIKE_FACTOR:.1f}× player avg; "
        "opp_double_four = opponent has ≥2 fours after your move; "
        "opp_forced_win = opponent has a proven VCF/VCT.[/dim]"
    )
    console.print(legend)
