"""Request / response envelopes for the API.

The API deliberately reuses the existing pydantic models from
:mod:`gomoku.models` and :mod:`gomoku.board` as the inner payloads —
``Board``, ``Move``, ``Stone`` already validate themselves and produce
sensible JSON.  This module only adds the request envelopes that the
core didn't need (``MoveRequest``, ``MoveResponse`` etc.) plus a couple
of metadata DTOs (``BrainInfo``).
"""

from __future__ import annotations

from typing import Any

from gomoku.board import Board
from gomoku.diagnostics import MoveDiagnostics
from gomoku.models import Move, Stone
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
            "the TOML aliases."
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


__all__ = [
    "BrainInfo",
    "BrainsResponse",
    "EvaluateRequest",
    "EvaluateResponse",
    "MoveRequest",
    "MoveResponse",
]

