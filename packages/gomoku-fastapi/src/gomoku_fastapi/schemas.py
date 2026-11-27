"""Request / response envelopes for the API.

The API deliberately reuses the existing pydantic models from
:mod:`gomoku.models` and :mod:`gomoku.board` as the inner payloads —
``Board``, ``Move``, ``Stone`` already validate themselves and produce
sensible JSON.  This module only adds the request envelopes that the
core didn't need (``MoveRequest``, ``MoveResponse`` etc.) plus a couple
of metadata DTOs (``BrainInfo``).
"""

from datetime import date
from typing import Any, Literal

from gomoku.board import Board
from gomoku.diagnostics import MoveDiagnostics
from gomoku.models import GameResult, Move, Stone
from pydantic import BaseModel, ConfigDict, Field


class BrainInfo(BaseModel):
    """Metadata for a single registered brain — returned by ``GET /v1/brains``."""

    name: str = Field(description="Registry key — pass this as ``brain`` in /v1/move.")
    settings_schema: dict[str, Any] = Field(
        description=(
            "JSON Schema generated from the brain's pydantic settings model. "
            "Useful for clients that want to render a settings UI without "
            "hard-coding the knob list."
        ),
    )


class BrainsResponse(BaseModel):
    """Response payload for ``GET /v1/brains``."""

    brains: list[BrainInfo]


class BrainSettingsResponse(BaseModel):
    """Response payload for ``GET /v1/brains/{name}/settings``."""

    name: str = Field(description="The brain name or alias that was asked for.")
    type: str = Field(description="The registry brain it resolves to (``alphabeta``, ``threat``, …).")
    settings: dict[str, Any] = Field(
        description=(
            "Every setting a move with this brain uses when the request sends "
            "no ``brain_overrides``: the server's ``[brains.<name>]`` alias, "
            "the brain's defaults for what the alias leaves out, and the "
            "server's time cap.  Keys follow the type's ``settings_schema`` "
            "from ``GET /v1/brains``."
        ),
    )


class MoveRequest(BaseModel):
    """Request payload for ``POST /v1/move``.

    Stateless: the client passes the full board, the side to move, and a
    brain reference; the server replies with the brain's chosen move.
    No session is created.
    """

    model_config = ConfigDict(extra="forbid")

    board: Board = Field(description="Current position.  Must be a legal Gomoku board state.")
    side_to_move: Stone = Field(
        description=(
            "Whose turn it is.  ``Stone.EMPTY`` is rejected — only "
            "``Stone.BLACK`` (1) or ``Stone.WHITE`` (2) are valid."
        ),
    )
    brain: str = Field(
        description="Registry name or TOML alias of the brain to consult.",
        examples=["alphabeta", "threat", "mcts"],
    )
    brain_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-request tweaks merged on top of the resolved brain "
            "settings, validated against the brain's pydantic model.  "
            "Use sparingly — most clients should configure brains via "
            "the TOML aliases.  Settings that name server files "
            "(``opening_book``) cannot be changed: a different value is a 422."
        ),
    )
    diagnostics: bool = Field(
        default=False,
        description=(
            "When ``true``, the server activates a diagnostics "
            ":class:`~gomoku.diagnostics.MoveDiagnostics` ContextVar "
            "around the brain call and returns the captured record on "
            "the response.  Adds a measurable cost to the request "
            "(brains do extra evaluation when diagnostics are active), "
            "so leave it off when you only need the move."
        ),
    )


class MoveResponse(BaseModel):
    """Response payload for ``POST /v1/move``."""

    move: Move = Field(description="Move chosen by the brain.")
    brain: str = Field(description="Echo of the brain key used (post-alias resolution).")
    diagnostics: MoveDiagnostics | None = Field(
        default=None,
        description=(
            "Per-move diagnostics when the request set ``diagnostics: "
            "true``; ``null`` otherwise.  Schema matches the lines of a "
            "``.diag.jsonl`` recording so callers can pipe straight into "
            "``analyze_diagnostics``."
        ),
    )


class EvaluateRequest(BaseModel):
    """Request payload for ``POST /v1/evaluate`` — static heuristic eval.

    No brain is involved; this is a cheap call into
    :func:`gomoku_brain.evaluate.evaluate_board`.
    """

    model_config = ConfigDict(extra="forbid")

    board: Board = Field(description="Position to evaluate.")
    side_to_move: Stone = Field(
        description=(
            "Whose perspective the eval is from.  Higher = better for "
            "*this* side; the same board returns roughly opposite "
            "scores for opposite stones."
        ),
    )


class EvaluateResponse(BaseModel):
    """Response payload for ``POST /v1/evaluate``."""

    score: float = Field(
        description=(
            "Heuristic score from ``side_to_move``'s perspective.  "
            "Clamped to ±99 000 — never reaches the ±100 000 win/loss "
            "constants used by the search."
        ),
    )


# ---------------------------------------------------------------------------
# Sessions (POST/GET/DELETE /v1/games)
# ---------------------------------------------------------------------------


class PlayerConfig(BaseModel):
    """Configuration for one side of a session.

    A side may be **brain-driven** (set ``brain``) or **external**
    (omit ``brain``).  External sides advance via
    ``POST /v1/games/{id}/moves``; brain-driven sides advance via
    ``POST /v1/games/{id}/think``.  The two patterns mix freely — a
    typical "play against AI" UI configures one external side and one
    brain.
    """

    model_config = ConfigDict(extra="forbid")

    brain: str | None = Field(
        default=None,
        description=(
            "Registry name or TOML alias of the brain to use for this "
            "side.  ``null`` (or omit) marks the side as external — "
            "moves arrive via ``POST /v1/games/{id}/moves``."
        ),
        examples=["alphabeta", None],
    )
    brain_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-session tweaks merged on top of the resolved brain settings.  Ignored when ``brain`` is ``null``.  "
            "Settings that name server files (``opening_book``) cannot be changed: a different value is a 422."
        ),
    )


class CreateGameRequest(BaseModel):
    """Request payload for ``POST /v1/games``."""

    model_config = ConfigDict(extra="forbid")

    board_size: int = Field(default=15, ge=5, le=25, description="N for an NxN board.")
    win_length: int = Field(default=5, ge=3, description="Stones in a row needed to win.")
    black: PlayerConfig = Field(
        default_factory=PlayerConfig,
        description="Black-side configuration.  Default: external.",
    )
    white: PlayerConfig = Field(
        default_factory=PlayerConfig,
        description="White-side configuration.  Default: external.",
    )
    record: bool = Field(
        default=False,
        description=(
            "When ``true``, the session writes its moves to a "
            "``gomoku_*.json`` recording in a per-session temp dir.  "
            "The path appears on ``GET /v1/games/{id}`` as "
            "``record_dir`` so clients can fetch the file out-of-band.  "
            "The directory is removed when the session is deleted or expires."
        ),
    )
    diagnostics: bool = Field(
        default=False,
        description=(
            "When ``true``, the session activates a "
            ":class:`DiagnosticsRecorder` and writes per-move "
            "diagnostics to a ``gomoku_*.diag.jsonl`` companion file. "
            "Independent of the per-call ``think`` diagnostics — "
            "even when this flag is off, ``think`` still returns "
            "in-memory diagnostics on the response."
        ),
    )


class HistoryEntry(BaseModel):
    """One entry in ``GameStateResponse.history``."""

    stone: Stone
    move: Move


class GameStateResponse(BaseModel):
    """Response payload for ``POST /v1/games`` and ``GET /v1/games/{id}``.

    Captures everything a UI needs to render the current state: who's
    configured for each side, whose turn it is, the board, the move
    history, and the result if the game has finished.
    """

    game_id: str = Field(description="Session UUID — pass to subsequent /v1/games/{id}/* calls.")
    board: Board
    side_to_move: Stone
    move_number: int = Field(description="1-based number of the *next* move (== len(history) + 1).")
    history: list[HistoryEntry]
    black_brain: str | None = Field(description="Brain name for Black, or null when external.")
    white_brain: str | None = Field(description="Brain name for White, or null when external.")
    result: GameResult | None = Field(description="Set once the game ends; null while in progress.")
    record_dir: str | None = Field(default=None, description="Filesystem dir of the recording, when ``record=true``.")
    diag_dir: str | None = Field(
        default=None,
        description="Filesystem dir of the .diag.jsonl, when ``diagnostics=true``.",
    )


class SubmitMoveRequest(BaseModel):
    """Request payload for ``POST /v1/games/{id}/moves`` — external move."""

    model_config = ConfigDict(extra="forbid")

    stone: Stone = Field(description="The colour playing this move; must equal the session's current side_to_move.")
    move: Move = Field(description="Cell coordinates the external player chose.")


class ThinkRequest(BaseModel):
    """Request payload for ``POST /v1/games/{id}/think``.

    Optional knobs only — the brain is whatever the session was
    constructed with for the current side.
    """

    model_config = ConfigDict(extra="forbid")

    diagnostics: bool = Field(
        default=False,
        description=(
            "Return the per-move :class:`MoveDiagnostics` record on "
            "the response.  Independent of the session's recording "
            "flag — useful when a UI wants to render search internals "
            "for *this* move without producing a file."
        ),
    )


class ThinkResponse(BaseModel):
    """Response payload for ``POST /v1/games/{id}/think``."""

    move: Move = Field(description="Move the brain chose and the runner applied.")
    state: GameStateResponse = Field(description="Updated session state after the move.")
    diagnostics: MoveDiagnostics | None = Field(
        default=None,
        description="Populated when ``diagnostics: true`` was requested; ``null`` otherwise.",
    )


__all__ = [
    "BrainInfo",
    "BrainsResponse",
    "CreateGameRequest",
    "EvaluateRequest",
    "EvaluateResponse",
    "GameStateResponse",
    "HistoryEntry",
    "MoveRequest",
    "MoveResponse",
    "PlayerConfig",
    "SubmitMoveRequest",
    "ThinkRequest",
    "ThinkResponse",
]


class VersionInfo(BaseModel):
    """One API version and its lifecycle status — an entry of ``GET /versions``."""

    version: str = Field(description="Version name, which is also its URL prefix.", examples=["v1"])
    path: str = Field(description="URL prefix of every endpoint in this version.", examples=["/v1"])
    openapi: str = Field(description="OpenAPI document describing only this version.", examples=["/v1/openapi.json"])
    status: Literal["stable", "deprecated"] = Field(
        description=(
            "``deprecated`` versions still work but will be removed after their sunset date.  A "
            "version whose deprecation date is still ahead stays ``stable`` until then."
        )
    )
    released: date = Field(description="Date the version was first served.")
    deprecated: date | None = Field(default=None, description="Date the version was deprecated, if it is.")
    sunset: date | None = Field(default=None, description="Date after which the version may be removed.")
    successor: str | None = Field(default=None, description="Version that replaces this one, if any.")


class VersionsResponse(BaseModel):
    """Response payload for ``GET /versions``."""

    latest: str = Field(description="Newest version; new clients should use it.", examples=["v1"])
    versions: list[VersionInfo] = Field(description="Every served version, oldest first.")
