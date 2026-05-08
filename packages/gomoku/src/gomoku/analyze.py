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

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from gomoku.diagnostics import MoveDiagnostics
from gomoku.models import Move, Stone

if TYPE_CHECKING:
    from gomoku.board import Board
    from gomoku.protocol import MoveProvider

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
        except (KeyError, ValueError):
            continue
        provider = p.get("provider")
        if provider is not None:
            out[stone] = str(provider)
    return out


def _compute_player_stats(
    moves: list[MoveDiagnostics], providers: dict[Stone, str]
) -> dict[Stone, PlayerStats]:
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
    if (
        alt is not None
        and move.chosen_move is not None
        and alt.move != move.chosen_move
        and alt.score_type == "eval"
    ):
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
        if (
            avg_ms > 0
            and move.think_time_ms >= TIME_SPIKE_MIN_MS
            and move.think_time_ms >= avg_ms * TIME_SPIKE_FACTOR
        ):
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


# ---------------------------------------------------------------------------
# Deep review — chess-engine-style classification with optional LLM commentary
# ---------------------------------------------------------------------------


class Classification(StrEnum):
    """Chess-engine-style move-quality buckets (best→blunder)."""

    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


class Advantage(StrEnum):
    """Coarse "who's ahead" derived from the Black-perspective eval."""

    BLACK_WINNING = "black_winning"
    BLACK_BETTER = "black_better"
    EQUAL = "equal"
    WHITE_BETTER = "white_better"
    WHITE_WINNING = "white_winning"


@dataclass(frozen=True)
class ClassificationThresholds:
    """Eval-delta and advantage cutoffs.

    All values are in the same unit as :func:`gomoku_brain.evaluate_board`
    (live-3 ≈ 1 000, live-4 ≈ 10 000, win ≈ 100 000).

    *Move-quality* uses ``*_delta`` cutoffs against the *mover-perspective*
    delta ``best_eval_after - played_eval_after`` (always >= 0).

    *Advantage* uses ``win_threshold`` / ``better_threshold`` against the
    Black-perspective signed eval (positive = Black ahead).
    """

    excellent_delta: float = 200.0
    good_delta: float = 800.0
    inaccuracy_delta: float = 2_500.0
    mistake_delta: float = 10_000.0
    win_threshold: float = 50_000.0
    better_threshold: float = 1_500.0

    def classify(self, delta: float, *, was_top: bool) -> Classification:
        """Bucket a non-negative *delta* into a :class:`Classification`."""
        if was_top or delta < self.excellent_delta:
            # ``BEST`` is reserved for "exactly matched the engine's pick".
            return Classification.BEST if was_top else Classification.EXCELLENT
        if delta < self.good_delta:
            return Classification.GOOD
        if delta < self.inaccuracy_delta:
            return Classification.INACCURACY
        if delta < self.mistake_delta:
            return Classification.MISTAKE
        return Classification.BLUNDER

    def to_advantage(self, eval_black: float) -> Advantage:
        """Map a Black-perspective signed eval to an :class:`Advantage`."""
        if eval_black >= self.win_threshold:
            return Advantage.BLACK_WINNING
        if eval_black >= self.better_threshold:
            return Advantage.BLACK_BETTER
        if eval_black <= -self.win_threshold:
            return Advantage.WHITE_WINNING
        if eval_black <= -self.better_threshold:
            return Advantage.WHITE_BETTER
        return Advantage.EQUAL


class MoveCommentator(Protocol):
    """Optional natural-language commentary callback.

    Implemented in :mod:`gomoku_cli.llm_review` for the pydantic-ai layer;
    kept abstract here so the core ``gomoku`` package stays LLM-free.
    """

    def annotate(self, review: MoveReview, context: ReviewContext) -> Any: ...


@dataclass(frozen=True)
class ReviewContext:
    """Per-ply context handed to a :class:`MoveCommentator`.

    Provides everything the LLM might want — the pre-move ASCII board,
    threat info from the diagnostics file, and the brain's preferred
    line — without forcing the commentator to import :mod:`gomoku.board`.
    """

    pre_board_ascii: str
    side_to_move: Stone
    decision_path: str
    our_threats: dict[str, Any]
    opp_threats: dict[str, Any]


@dataclass
class MoveReview:
    """Engine-eval review of a single played move."""

    move_number: int
    stone: Stone
    played_move: Move
    engine_best_move: Move
    eval_before: float           # Black-perspective, signed
    eval_after_played: float     # Black-perspective, signed
    eval_after_best: float       # Black-perspective, signed
    delta: float                 # mover-perspective, ≥ 0
    classification: Classification
    advantage: Advantage
    # Optional structured commentary attached after-the-fact.  ``Any``
    # because the concrete type is the pydantic-ai ``MoveCommentary``
    # model defined in :mod:`gomoku_cli.llm_review`, which the core
    # package must not import.
    commentary: Any | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "move_number": self.move_number,
            "stone": str(self.stone),
            "played_move": str(self.played_move),
            "engine_best_move": str(self.engine_best_move),
            "eval_before": round(self.eval_before, 1),
            "eval_after_played": round(self.eval_after_played, 1),
            "eval_after_best": round(self.eval_after_best, 1),
            "delta": round(self.delta, 1),
            "classification": str(self.classification),
            "advantage": str(self.advantage),
        }
        if self.commentary is not None:
            try:
                out["commentary"] = self.commentary.model_dump()  # pydantic
            except AttributeError:
                out["commentary"] = self.commentary
        return out


# Per-classification accuracy weight in [0, 1].  BEST/EXCELLENT count
# fully; BLUNDER not at all.  Mean across moves * 100 = chess-style
# accuracy %.
_ACCURACY_WEIGHTS: dict[Classification, float] = {
    Classification.BEST: 1.0,
    Classification.EXCELLENT: 0.95,
    Classification.GOOD: 0.85,
    Classification.INACCURACY: 0.55,
    Classification.MISTAKE: 0.25,
    Classification.BLUNDER: 0.0,
}


def _accuracy_from_classifications(items: list[Classification]) -> float:
    if not items:
        return 0.0
    return 100.0 * sum(_ACCURACY_WEIGHTS[c] for c in items) / len(items)


@dataclass
class DeepAnalysisReport(AnalysisReport):
    """Extends :class:`AnalysisReport` with chess-style review data."""

    reviews: list[MoveReview] = field(default_factory=list)
    analysis_brain: str = ""
    analysis_depth: int = 0
    classification_counts: dict[Stone, dict[Classification, int]] = field(default_factory=dict)
    accuracy: dict[Stone, float] = field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        base: AnalysisReport,
        reviews: list[MoveReview],
        analysis_brain: str,
        analysis_depth: int,
    ) -> DeepAnalysisReport:
        counts: dict[Stone, dict[Classification, int]] = {
            Stone.BLACK: {c: 0 for c in Classification},
            Stone.WHITE: {c: 0 for c in Classification},
        }
        per_stone: dict[Stone, list[Classification]] = {Stone.BLACK: [], Stone.WHITE: []}
        for r in reviews:
            counts[r.stone][r.classification] += 1
            per_stone[r.stone].append(r.classification)
        accuracy = {
            Stone.BLACK: _accuracy_from_classifications(per_stone[Stone.BLACK]),
            Stone.WHITE: _accuracy_from_classifications(per_stone[Stone.WHITE]),
        }
        return cls(
            path=base.path,
            header=base.header,
            summary=base.summary,
            moves=base.moves,
            player_stats=base.player_stats,
            flagged=base.flagged,
            reviews=reviews,
            analysis_brain=analysis_brain,
            analysis_depth=analysis_depth,
            classification_counts=counts,
            accuracy=accuracy,
        )

    def to_json(self) -> dict[str, Any]:
        data = super().to_json()
        data["analysis_brain"] = self.analysis_brain
        data["analysis_depth"] = self.analysis_depth
        data["accuracy"] = {str(s): round(v, 1) for s, v in self.accuracy.items()}
        data["classification_counts"] = {
            str(stone): {str(cls): n for cls, n in cls_counts.items()}
            for stone, cls_counts in self.classification_counts.items()
        }
        data["reviews"] = [r.to_json() for r in self.reviews]
        return data


def _board_to_ascii(board: Board) -> str:
    """Render a :class:`Board` as a multi-line ASCII grid for the LLM prompt."""
    rows: list[str] = []
    header = "   " + " ".join(chr(ord("A") + c) for c in range(board.size))
    rows.append(header)
    for r in range(board.size):
        cells: list[str] = []
        for c in range(board.size):
            s = board.at(r, c)
            cells.append("B" if s is Stone.BLACK else "W" if s is Stone.WHITE else ".")
        rows.append(f"{r + 1:>2} " + " ".join(cells))
    return "\n".join(rows)


def review_game(
    diag_path: Path,
    *,
    analysis_brain: MoveProvider,
    analysis_brain_name: str,
    analysis_depth: int,
    thresholds: ClassificationThresholds | None = None,
    commentator: MoveCommentator | None = None,
    evaluate: Any | None = None,
) -> DeepAnalysisReport:
    """Replay a ``.diag.jsonl`` and produce a :class:`DeepAnalysisReport`.

    The replay is driven entirely off the diagnostics file — every
    record carries ``chosen_move``, so a separate game record is not
    needed.  For each ply we reconstruct the pre-move position, ask
    *analysis_brain* for its preferred move, and classify the played
    move by the Black-perspective eval delta.

    *evaluate* is injected for testing (defaults to
    ``gomoku_brain.evaluate.evaluate_board``).  Production callers leave
    it ``None`` so the import only happens when a brain is actually run.
    """
    from gomoku.board import Board

    if evaluate is None:
        from gomoku_brain.evaluate import evaluate_board as evaluate  # type: ignore[assignment]

    if thresholds is None:
        thresholds = ClassificationThresholds()

    header, moves, summary = load_diagnostics(diag_path)
    base_report = _build_report(diag_path, header, moves, summary)

    size = int(header.get("game_config", {}).get("board_size", 15))
    board = Board(size=size)
    reviews: list[MoveReview] = []

    for entry in moves:
        if entry.chosen_move is None:
            # Forfeited / errored ply — skip the review pass but keep the
            # base report's flagging intact.
            continue

        stone = entry.stone
        pre_board = board

        # If the position is already terminal, replay is undefined — bail.
        if pre_board.is_full():
            break

        # 1) Engine's preferred move from this position.
        engine_best = analysis_brain.get_move(pre_board, stone)

        # 2) Apply both the played move and the engine's best move to
        #    fresh boards so we can re-eval each landing position.
        played_after = pre_board.place_stone(entry.chosen_move, stone)
        best_after = (
            played_after
            if engine_best == entry.chosen_move
            else pre_board.place_stone(engine_best, stone)
        )

        # 3) Mover-perspective deltas drive classification.
        played_eval_mover = float(evaluate(played_after, stone))
        best_eval_mover = float(evaluate(best_after, stone))
        delta = max(0.0, best_eval_mover - played_eval_mover)
        was_top = engine_best == entry.chosen_move

        if entry.move_number == 1:
            # First move on an empty board: every cell is symmetric.
            classification = Classification.BEST
            delta = 0.0
        else:
            classification = thresholds.classify(delta, was_top=was_top)

        # 4) Black-perspective signed evals for display.
        sign = 1 if stone is Stone.BLACK else -1
        played_eval_black = sign * played_eval_mover
        best_eval_black = sign * best_eval_mover
        eval_before_black = float(evaluate(pre_board, Stone.BLACK))

        review = MoveReview(
            move_number=entry.move_number,
            stone=stone,
            played_move=entry.chosen_move,
            engine_best_move=engine_best,
            eval_before=eval_before_black,
            eval_after_played=played_eval_black,
            eval_after_best=best_eval_black,
            delta=delta,
            classification=classification,
            advantage=thresholds.to_advantage(played_eval_black),
        )
        reviews.append(review)

        # 5) Advance the replay using the *played* move so subsequent
        #    plies see the actual game position, not the engine line.
        board = played_after

    if commentator is not None:
        for review in reviews:
            try:
                # Find the corresponding diagnostics entry for context.
                entry = next(m for m in moves if m.move_number == review.move_number)
            except StopIteration:
                continue
            ctx = ReviewContext(
                pre_board_ascii=_replay_pre_board_ascii(moves, review.move_number, size),
                side_to_move=review.stone,
                decision_path=entry.decision_path,
                our_threats=entry.our_threats.model_dump(),
                opp_threats=entry.opp_threats.model_dump(),
            )
            try:
                review.commentary = commentator.annotate(review, ctx)
            except Exception:
                review.commentary = None

    return DeepAnalysisReport.from_base(
        base_report,
        reviews,
        analysis_brain_name,
        analysis_depth,
    )


def _replay_pre_board_ascii(
    moves: list[MoveDiagnostics], target_move_number: int, size: int
) -> str:
    """Replay moves up to (but not including) *target_move_number* and render ASCII.

    Used only for the LLM commentary prompt — production replay drives off
    the same data inside :func:`review_game`.
    """
    from gomoku.board import Board

    board = Board(size=size)
    for m in moves:
        if m.move_number >= target_move_number:
            break
        if m.chosen_move is None:
            continue
        board = board.place_stone(m.chosen_move, m.stone)
    return _board_to_ascii(board)


__all__ = [
    "ALTERNATIVE_GAP",
    "BLUNDER_EVAL_DROP",
    "TIME_SPIKE_FACTOR",
    "TIME_SPIKE_MIN_MS",
    "Advantage",
    "AnalysisReport",
    "Classification",
    "ClassificationThresholds",
    "DeepAnalysisReport",
    "FlaggedMove",
    "MoveCommentator",
    "MoveReview",
    "PlayerStats",
    "ReviewContext",
    "analyze_diagnostics",
    "analyze_diagnostics_text",
    "load_diagnostics",
    "review_game",
]
