"""Optional pydantic-ai narrative commentary for the deep review.

This module is only imported when ``--llm`` is passed to
``gomoku-cli review``.  It depends on ``pydantic-ai``, which is an
*optional* extra (``pip install gomoku-cli[llm]``) — the rest of the
CLI never imports this module so the dependency stays opt-in.

LLMs are unreliable at exact numerical Gomoku evaluation, so they are
deliberately not in the loop for classification.  Their only job is to
produce a short coach-style narrative for each ply, given the engine's
already-computed verdict.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from gomoku.analyze import MoveReview, ReviewContext
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Gomoku (Five-in-a-Row) coach giving short, blunt feedback "
    "on a single played move.  Be concise.  Never invent moves not given to "
    "you.  Do not contradict the engine's verdict — explain it."
)


class MoveCommentary(BaseModel):
    """Structured output produced by the LLM for one ply."""

    headline: str = Field(description="One-line summary, e.g. 'Missed a double-three fork'.")
    explanation: str = Field(description="One to three sentences explaining the engine's verdict.")
    key_threats: list[str] = Field(
        default_factory=list,
        description="Short tactical labels relevant to this position, e.g. 'open-three on row 8'.",
    )


def _format_prompt(review: MoveReview, ctx: ReviewContext) -> str:
    """Build the per-ply user prompt from review + context."""
    eval_played = review.eval_after_played / 1000.0
    eval_best = review.eval_after_best / 1000.0
    return (
        f"Position before move {review.move_number} ({ctx.side_to_move} to play):\n"
        f"```\n{ctx.pre_board_ascii}\n```\n\n"
        f"Played:        {review.played_move}\n"
        f"Engine best:   {review.engine_best_move}\n"
        f"Eval (Black-perspective, scaled by 1000): "
        f"played={eval_played:+.1f}, best={eval_best:+.1f}, "
        f"delta(mover)={review.delta / 1000.0:.1f}\n"
        f"Engine verdict: {review.classification.value}\n"
        f"Decision path: {ctx.decision_path or '(unknown)'}\n"
        f"Threats — ours: {ctx.our_threats}; opponent: {ctx.opp_threats}\n\n"
        "Reply with a structured MoveCommentary."
    )


class MoveCommentatorImpl:
    """Pydantic-AI-backed commentator.  Constructs the agent lazily.

    Failures inside :meth:`annotate` are caught and logged — a single
    bad LLM call must not derail an otherwise valid review report.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key_env: str | None = None,
        ollama_base_url: str = "http://localhost:11434/v1",
    ) -> None:
        try:
            from pydantic_ai import Agent
        except ImportError as exc:  # pragma: no cover — exercised manually
            msg = (
                "pydantic-ai is not installed.  Re-install gomoku-cli with "
                "the 'llm' extra: pip install 'gomoku-cli[llm]'."
            )
            raise RuntimeError(msg) from exc

        self._provider = provider
        self._model = model

        # ``ollama`` is wired through OpenAI's SDK with a custom base
        # URL — pydantic-ai accepts an explicit OpenAIChatModel for that.
        if provider == "ollama":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            ollama_provider = OpenAIProvider(
                base_url=ollama_base_url,
                api_key="ollama",
            )
            ai_model: Any = OpenAIChatModel(model, provider=ollama_provider)
        else:
            # api_key_env defaults are picked by pydantic-ai itself, but
            # we honour the user override by setting the env var pydantic-ai
            # actually reads, when given a custom one.
            if api_key_env and api_key_env not in os.environ:
                _log.warning(
                    "AnalysisSettings.llm_api_key_env=%r is not set in the "
                    "environment; the provider may fail to authenticate.",
                    api_key_env,
                )
            ai_model = f"{provider}:{model}"

        self._agent = Agent(
            ai_model,
            output_type=MoveCommentary,
            system_prompt=_SYSTEM_PROMPT,
        )

    def annotate(self, review: MoveReview, ctx: ReviewContext) -> MoveCommentary | None:
        """Run the LLM on one ply.  Returns ``None`` on any failure."""
        prompt = _format_prompt(review, ctx)
        try:
            result = self._agent.run_sync(prompt)
        except Exception as exc:
            _log.warning(
                "LLM commentary failed for move %d: %s", review.move_number, exc
            )
            return None
        return result.output


__all__ = [
    "MoveCommentary",
    "MoveCommentatorImpl",
]
