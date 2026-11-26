"""Post-game analysis of ``.diag.jsonl`` diagnostics files.

The :func:`analyze_diagnostics` entry point parses a diagnostics
recording and produces an :class:`AnalysisReport` summarising:

* Per-player aggregates (total time, avg depth, nodes searched).
* Suspect moves flagged by one or more heuristics
  (eval drops, blunders relative to the top candidate, missed opponent
  forks, time spikes, VCF gating).

This module is **pure data-processing** — it does not depend on
``rich`` or any other rendering library.  The CLI's
:func:`gomoku_cli.analyze.render_report` adds the rich-table renderer
on top; the FastAPI layer reuses these classes to populate
``POST /v1/analyze`` responses.

Two parsing entry points cover the two call patterns:

* :func:`analyze_diagnostics` — reads a path on disk (CLI / scripts).
* :func:`analyze_diagnostics_text` — accepts the JSONL text directly,
  for HTTP uploads where there is no local file.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gomoku.diagnostics import MoveDiagnostics
from gomoku.models import Stone

# ---------------------------------------------------------------------------
# Thresholds for flagging suspect moves.  Kept as module-level constants so
# tests can import / override them if needed.
# ---------------------------------------------------------------------------

# Eval swing that counts as a blunder, in centi-pawn-ish units used by
# :func:`gomoku_brain.evaluate_board`.  A full win is ≈ 100 000, so 500 is
# a meaningful deviation without being noisy.
BLUNDER_EVAL_DROP = 500.0

# Gap between the chosen move's score and the top candidate that we treat as
# "bypassed a clearly better move".  Same units as BLUNDER_EVAL_DROP.
ALTERNATIVE_GAP = 500.0

# Factor over the player's average think time before flagging a time spike.
TIME_SPIKE_FACTOR = 2.0

# Minimum absolute think time (ms) that a spike has to exceed.  Prevents
# noise on blitz-style games where moves take <50 ms.
TIME_SPIKE_MIN_MS = 200


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------


@dataclass
class FlaggedMove:
    """A single move flagged as suspect, with the reasons why."""

    move_number: int
    stone: Stone
    decision_path: str
    reasons: list[str] = field(default_factory=list)
    # Free-form details that help the user investigate (e.g. eval numbers,
    # chosen vs alternative-best, elapsed ms).  Rendered as key=value.
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerStats:
    """Aggregate counters for one player (one stone colour)."""

    stone: Stone
    provider: str | None = None
    moves: int = 0
    total_think_ms: int = 0
    total_nodes: int = 0
    max_depth_reached: int = 0
    sum_depth: int = 0
    depth_samples: int = 0
    blunders: int = 0

    @property
    def avg_think_ms(self) -> float:
        return self.total_think_ms / self.moves if self.moves else 0.0

    @property
    def avg_depth(self) -> float:
        return self.sum_depth / self.depth_samples if self.depth_samples else 0.0


@dataclass
class AnalysisReport:
    """Top-level report produced by :func:`analyze_diagnostics`."""

    path: Path
    header: dict[str, Any]
    summary: dict[str, Any] | None
    moves: list[MoveDiagnostics]
    player_stats: dict[Stone, PlayerStats]
    flagged: list[FlaggedMove]

    def to_json(self) -> dict[str, Any]:
        """Serialise to a plain ``dict`` suitable for ``json.dump``."""
        return {
            "path": str(self.path),
            "header": self.header,
            "summary": self.summary,
            "total_moves": len(self.moves),
            "player_stats": {
                str(stone): {
                    "provider": ps.provider,
                    "moves": ps.moves,
                    "total_think_ms": ps.total_think_ms,
                    "avg_think_ms": round(ps.avg_think_ms, 1),
                    "total_nodes": ps.total_nodes,
                    "max_depth_reached": ps.max_depth_reached,
                    "avg_depth": round(ps.avg_depth, 2),
                    "blunders": ps.blunders,
                }
                for stone, ps in self.player_stats.items()
            },
            "flagged": [
                {
                    "move_number": f.move_number,
                    "stone": str(f.stone),
                    "decision_path": f.decision_path,
                    "reasons": f.reasons,
                    "details": f.details,
                }
                for f in self.flagged
            ],
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_diagnostics_text(
    text: str, *, source: str = "<text>"
) -> tuple[dict[str, Any], list[MoveDiagnostics], dict[str, Any] | None]:
    """Parse JSONL text into ``(header, moves, summary)``.

    *source* is only used in error messages — it lets the caller report
    a useful filename even when the text didn't come from disk.
    """
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    if not raw_lines:
        raise ValueError(f"Empty diagnostics file: {source}")

    first = json.loads(raw_lines[0])
    if first.get("type") != "header":
        raise ValueError(f"{source}: first line is not a diagnostics header")
    header = first

    summary: dict[str, Any] | None = None
    body = raw_lines[1:]
    if body:
        last = json.loads(body[-1])
        if last.get("type") == "summary":
            summary = last
            body = body[:-1]

    moves: list[MoveDiagnostics] = []
    for i, line in enumerate(body, start=2):
        data = json.loads(line)
        # Tolerate the summary marker appearing mid-file for robustness; skip it.
        if data.get("type") in {"header", "summary"}:
            continue
        try:
            moves.append(MoveDiagnostics.model_validate(data))
        except Exception as exc:
            raise ValueError(f"{source}:{i}: malformed move record: {exc}") from exc
    return header, moves, summary


def load_diagnostics(
    path: Path,
) -> tuple[dict[str, Any], list[MoveDiagnostics], dict[str, Any] | None]:
    """Parse a ``.diag.jsonl`` file into (header, move_records, summary).

    The file format is:
    * line 1: header record (``type=header``).
    * lines 2..N: one :class:`MoveDiagnostics` JSON object per line.
    * optional last line: summary record (``type=summary``).
    """
    return _parse_diagnostics_text(path.read_text(encoding="utf-8"), source=str(path))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _player_providers(header: dict[str, Any]) -> dict[Stone, str]:
    """Pull ``{stone -> provider}`` out of the header's player list."""
    out: dict[Stone, str] = {}
    for p in header.get("players", []) or []:
        try:
            stone = Stone(p["stone"])
        except KeyError, ValueError:
            continue
        provider = p.get("provider")
        if provider is not None:
            out[stone] = str(provider)
    return out


def _compute_player_stats(moves: list[MoveDiagnostics], providers: dict[Stone, str]) -> dict[Stone, PlayerStats]:
    stats: dict[Stone, PlayerStats] = {
        Stone.BLACK: PlayerStats(stone=Stone.BLACK, provider=providers.get(Stone.BLACK)),
        Stone.WHITE: PlayerStats(stone=Stone.WHITE, provider=providers.get(Stone.WHITE)),
    }
    for m in moves:
        ps = stats[m.stone]
        ps.moves += 1
        ps.total_think_ms += m.think_time_ms
        if m.search_stats.nodes_searched:
            ps.total_nodes += m.search_stats.nodes_searched
        if m.search_stats.depth_reached is not None:
            ps.sum_depth += m.search_stats.depth_reached
            ps.depth_samples += 1
            if m.search_stats.depth_reached > ps.max_depth_reached:
                ps.max_depth_reached = m.search_stats.depth_reached
    return stats


def _flag_move(
    move: MoveDiagnostics,
    prev_by_stone: dict[Stone, MoveDiagnostics],
    player_stats: dict[Stone, PlayerStats],
) -> FlaggedMove | None:
    """Return a :class:`FlaggedMove` if *move* trips one or more heuristics."""
    reasons: list[str] = []
    details: dict[str, Any] = {}

    # 1. Eval drop vs. this player's previous move (compares eval_after t-1
    # with eval_after t — both are from the same side's perspective).
    prev = prev_by_stone.get(move.stone)
    if prev is not None and prev.eval_after is not None and move.eval_after is not None:
        drop = prev.eval_after - move.eval_after
        if drop >= BLUNDER_EVAL_DROP:
            reasons.append("eval_drop")
            details["eval_drop"] = round(drop, 1)
            details["eval_prev"] = round(prev.eval_after, 1)
            details["eval_after"] = round(move.eval_after, 1)

    # 2. Alternative-best deviation (only meaningful for eval-scored candidates).
    alt = move.alternative_best
    if alt is not None and move.chosen_move is not None and alt.move != move.chosen_move and alt.score_type == "eval":
        chosen_score: float | None = None
        for c in move.top_candidates:
            if c.move == move.chosen_move:
                chosen_score = c.score
                break
        if chosen_score is not None and alt.score - chosen_score >= ALTERNATIVE_GAP:
            reasons.append("bypassed_top_candidate")
            details["chosen_score"] = round(chosen_score, 1)
            details["alternative_score"] = round(alt.score, 1)
            details["alternative_move"] = str(alt.move)

    # 3. Time spike relative to this player's moving average so far.
    ps = player_stats.get(move.stone)
    if ps is not None and ps.moves > 2:
        avg_ms = ps.avg_think_ms
        if avg_ms > 0 and move.think_time_ms >= TIME_SPIKE_MIN_MS and move.think_time_ms >= avg_ms * TIME_SPIKE_FACTOR:
            reasons.append("time_spike")
            details["think_ms"] = move.think_time_ms
            details["avg_ms"] = round(avg_ms, 1)

    # 4. Opponent had a fork / counter-four we didn't respond to.  A common
    # symptom of a search-horizon miss: opponent's four_threats ≥ 2 *after*
    # we played but we did not choose a defensive decision_path.
    if move.opp_threats.four_threats >= 2 and "block" not in move.decision_path.lower():
        reasons.append("opp_double_four")
        details["opp_fours"] = move.opp_threats.four_threats

    # 5. VCF / VCT loss announced — the opponent has a forced win.
    if move.opp_threats.vcf_result == "win" or move.opp_threats.vct_result == "win":
        reasons.append("opp_forced_win")
        if move.opp_threats.vcf_result == "win":
            details["opp_vcf"] = True
        if move.opp_threats.vct_result == "win":
            details["opp_vct"] = True

    if not reasons:
        return None

    # Count blunders against the player's stats (eval-drop specifically).
    if "eval_drop" in reasons and ps is not None:
        ps.blunders += 1

    return FlaggedMove(
        move_number=move.move_number,
        stone=move.stone,
        decision_path=move.decision_path,
        reasons=reasons,
        details=details,
    )


def _build_report(
    path: Path,
    header: dict[str, Any],
    moves: list[MoveDiagnostics],
    summary: dict[str, Any] | None,
) -> AnalysisReport:
    """Run the flagging pipeline and assemble an :class:`AnalysisReport`."""
    providers = _player_providers(header)
    player_stats = _compute_player_stats(moves, providers)

    flagged: list[FlaggedMove] = []
    prev_by_stone: dict[Stone, MoveDiagnostics] = {}
    # Rolling stats so time-spike detection uses data up to (but excluding) the
    # current move — avoids a move trivially being its own average.
    rolling_stats: dict[Stone, PlayerStats] = {
        Stone.BLACK: PlayerStats(stone=Stone.BLACK, provider=providers.get(Stone.BLACK)),
        Stone.WHITE: PlayerStats(stone=Stone.WHITE, provider=providers.get(Stone.WHITE)),
    }
    for move in moves:
        f = _flag_move(move, prev_by_stone, rolling_stats)
        if f is not None:
            flagged.append(f)
        # Update rolling stats *after* flagging.
        ps = rolling_stats[move.stone]
        ps.moves += 1
        ps.total_think_ms += move.think_time_ms
        if move.search_stats.depth_reached is not None:
            ps.sum_depth += move.search_stats.depth_reached
            ps.depth_samples += 1
        prev_by_stone[move.stone] = move

    # Propagate blunder counts recorded on rolling_stats onto player_stats.
    for stone, rs in rolling_stats.items():
        player_stats[stone].blunders = rs.blunders

    return AnalysisReport(
        path=path,
        header=header,
        summary=summary,
        moves=moves,
        player_stats=player_stats,
        flagged=flagged,
    )


def analyze_diagnostics(path: Path) -> AnalysisReport:
    """Parse *path* and produce an :class:`AnalysisReport`."""
    header, moves, summary = load_diagnostics(path)
    return _build_report(path, header, moves, summary)


def analyze_diagnostics_text(text: str, *, source: str = "<text>") -> AnalysisReport:
    """Like :func:`analyze_diagnostics`, but takes the JSONL text directly.

    For the FastAPI layer where the upload arrives as bytes and there's
    no local file path.  *source* shows up in error messages and as the
    ``path`` field of the report.
    """
    header, moves, summary = _parse_diagnostics_text(text, source=source)
    return _build_report(Path(source), header, moves, summary)


__all__ = [
    "ALTERNATIVE_GAP",
    "BLUNDER_EVAL_DROP",
    "TIME_SPIKE_FACTOR",
    "TIME_SPIKE_MIN_MS",
    "AnalysisReport",
    "FlaggedMove",
    "PlayerStats",
    "analyze_diagnostics",
    "analyze_diagnostics_text",
    "load_diagnostics",
]
