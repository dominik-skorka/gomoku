"""Diagnostics recording — captures per-move brain internals for post-game investigation.

Brains detect an active diagnostics context via :func:`current_move_diagnostics`
and populate it with search statistics, decision path, candidate scores, and
threat information.  The :class:`DiagnosticsRecorder` writes one JSON object
per line to a ``.diag.jsonl`` file following the recording naming convention.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gomoku.models import GameConfig, Move, Stone

# ── Per-move diagnostics model ──────────────────────────────────────────────


class CandidateInfo(BaseModel, frozen=True):
    """A candidate move with its evaluation score."""

    move: Move
    score: float


class ThreatInfo(BaseModel, frozen=True):
    """Summary of threats detected for one side."""

    four_threats: int = 0
    three_threats: int = 0
    vcf_result: str | None = None  # "win", "loss", or None
    vct_result: str | None = None


class SearchStats(BaseModel, frozen=True):
    """Statistics from the search algorithm."""

    depth_reached: int | None = None
    nodes_searched: int | None = None
    iterations: int | None = None
    tt_size: int | None = None
    tt_hits: int | None = None
    time_ms: int | None = None
    aspiration_resets: int | None = None


class MoveDiagnostics(BaseModel):
    """Diagnostics collected for a single move decision."""

    move_number: int = Field(ge=1)
    stone: Stone
    chosen_move: Move | None = None
    decision_path: str = ""
    think_time_ms: int = 0

    # Board state (compact: list of "B H8" / "W G7" strings)
    board_stones: list[str] = Field(default_factory=list)

    # Static evaluation before and after
    eval_before: float | None = None
    eval_after: float | None = None

    # Search statistics
    search_stats: SearchStats = SearchStats()

    # Top candidates considered (ordered by score, descending)
    top_candidates: list[CandidateInfo] = Field(default_factory=list)

    # Threat landscape
    our_threats: ThreatInfo = ThreatInfo()
    opp_threats: ThreatInfo = ThreatInfo()

    # Free-form notes from the brain (decision rationale, etc.)
    notes: list[str] = Field(default_factory=list)


# ── Context variable ────────────────────────────────────────────────────────

_current_diag: ContextVar[MoveDiagnostics | None] = ContextVar("_current_diag", default=None)


def current_move_diagnostics() -> MoveDiagnostics | None:
    """Return the active :class:`MoveDiagnostics` for the current move, or ``None``."""
    return _current_diag.get()


def set_move_diagnostics(diag: MoveDiagnostics | None) -> None:
    """Set (or clear) the active diagnostics context."""
    _current_diag.set(diag)


# ── Header model (first line of the .diag.jsonl file) ───────────────────────


class DiagnosticsHeader(BaseModel, frozen=True):
    """Metadata header written as the first line of a diagnostics file."""

    type: str = "header"
    format_version: int = 1
    started_at: datetime
    game_config: GameConfig
    players: list[dict[str, Any]] = Field(default_factory=list)


# ── Recorder ────────────────────────────────────────────────────────────────


class DiagnosticsRecorder:
    """Accumulates and writes per-move diagnostics to a ``.diag.jsonl`` file.

    The file uses the same naming scheme as game recordings::

        gomoku_{timestamp}.diag.jsonl
    """

    def __init__(
        self,
        game_config: GameConfig,
        players: list[dict[str, Any]],
        directory: Path,
        started_at: datetime | None = None,
    ) -> None:
        self._started_at = started_at or datetime.now(tz=UTC)
        directory.mkdir(parents=True, exist_ok=True)
        ts = self._started_at.strftime("%Y%m%d_%H%M%S")
        self._path = directory / f"gomoku_{ts}.diag.jsonl"

        header = DiagnosticsHeader(
            started_at=self._started_at,
            game_config=game_config,
            players=players,
        )
        self._path.write_text(
            header.model_dump_json() + "\n",
            encoding="utf-8",
        )

    @property
    def path(self) -> Path:
        return self._path

    def begin_move(self, move_number: int, stone: Stone, board_stones: list[str]) -> MoveDiagnostics:
        """Create a fresh :class:`MoveDiagnostics`, activate it, and return it."""
        diag = MoveDiagnostics(
            move_number=move_number,
            stone=stone,
            board_stones=board_stones,
        )
        set_move_diagnostics(diag)
        return diag

    def end_move(self, diag: MoveDiagnostics) -> None:
        """Finalise the diagnostics entry and append it to the file."""
        set_move_diagnostics(None)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(diag.model_dump_json() + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Append a final summary line (e.g. game result, total stats)."""
        line = json.dumps({"type": "summary", **summary}, default=str)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
