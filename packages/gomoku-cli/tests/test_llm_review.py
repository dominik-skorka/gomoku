"""Tests for the optional pydantic-ai commentary layer.

These tests are skipped unless the ``pydantic-ai`` optional dependency
is installed (``pip install gomoku-cli[llm]``), which gives us a
deterministic ``TestModel`` we can use without any network call.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

from gomoku.analyze import (
    Advantage,
    Classification,
    MoveReview,
    ReviewContext,
)
from gomoku.models import Move, Stone
from gomoku_cli.llm_review import MoveCommentary, MoveCommentatorImpl
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


def _review() -> MoveReview:
    return MoveReview(
        move_number=12,
        stone=Stone.WHITE,
        played_move=Move(row=8, col=8),
        engine_best_move=Move(row=8, col=9),
        eval_before=0.0,
        eval_after_played=-15_000.0,
        eval_after_best=-2_000.0,
        delta=13_000.0,
        classification=Classification.BLUNDER,
        advantage=Advantage.BLACK_WINNING,
    )


def _ctx() -> ReviewContext:
    return ReviewContext(
        pre_board_ascii=". . .\n. . .\n. . .",
        side_to_move=Stone.WHITE,
        decision_path="alphabeta",
        our_threats={"four_threats": 0, "three_threats": 1},
        opp_threats={"four_threats": 0, "three_threats": 2},
    )


class TestCommentator:
    def test_uses_test_model(self) -> None:
        """A ``MoveCommentatorImpl`` swapped to TestModel returns a populated MoveCommentary."""
        commentator = MoveCommentatorImpl.__new__(MoveCommentatorImpl)
        # Build the agent directly with TestModel so we don't hit any
        # provider's auth path.  ``TestModel`` makes pydantic-ai
        # auto-populate every field of the structured output.
        commentator._agent = Agent(  # type: ignore[attr-defined]
            TestModel(),
            output_type=MoveCommentary,
            system_prompt="You are a Gomoku coach.",
        )
        commentator._provider = "anthropic"  # type: ignore[attr-defined]
        commentator._model = "claude-sonnet-4-5"  # type: ignore[attr-defined]

        result = commentator.annotate(_review(), _ctx())
        assert isinstance(result, MoveCommentary)
        # TestModel produces non-empty placeholder strings for required fields.
        assert isinstance(result.headline, str)
        assert isinstance(result.explanation, str)
        assert isinstance(result.key_threats, list)

    def test_failure_returns_none(self) -> None:
        """Any LLM-side error is swallowed and ``annotate`` returns None."""
        commentator = MoveCommentatorImpl.__new__(MoveCommentatorImpl)

        class _Boom:
            def run_sync(self, *_args, **_kwargs):
                raise RuntimeError("api down")

        commentator._agent = _Boom()  # type: ignore[attr-defined]
        commentator._provider = "anthropic"  # type: ignore[attr-defined]
        commentator._model = "claude-sonnet-4-5"  # type: ignore[attr-defined]

        assert commentator.annotate(_review(), _ctx()) is None

    def test_ollama_uses_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructing with provider='ollama' wires an OpenAIProvider with the given base URL."""
        captured: dict[str, object] = {}

        # Stub OpenAIProvider so we capture the constructor args without
        # any network connection.
        from pydantic_ai.providers import openai as openai_provider_mod

        real = openai_provider_mod.OpenAIProvider

        def _stub(*args, **kwargs):
            captured["base_url"] = kwargs.get("base_url")
            captured["api_key"] = kwargs.get("api_key")
            return real(*args, **kwargs)

        monkeypatch.setattr(openai_provider_mod, "OpenAIProvider", _stub)

        # Construction may fail downstream (model validation requires a
        # running Ollama).  We only care that our stub captured the args.
        import contextlib

        with contextlib.suppress(Exception):
            MoveCommentatorImpl(
                provider="ollama",
                model="llama3.2",
                ollama_base_url="http://localhost:9999/v1",
            )

        assert captured.get("base_url") == "http://localhost:9999/v1"
        assert captured.get("api_key") == "ollama"
