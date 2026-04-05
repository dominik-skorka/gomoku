"""Gomoku AI brains — pluggable move providers."""

from __future__ import annotations

from gomoku.protocol import MoveProvider
from gomoku.settings import BrainSettings

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.negamax import NegamaxBrain
from gomoku_brain.random import RandomBrain
from gomoku_brain.threat import ThreatBrain

BRAINS: dict[str, type] = {
    "random": RandomBrain,
    "minimax": MinimaxBrain,
    "negamax": NegamaxBrain,
    "alphabeta": AlphaBetaBrain,
    "threat": ThreatBrain,
}


def get_brain(name: str, **kwargs: object) -> MoveProvider:
    """Create a brain instance by name."""
    if name not in BRAINS:
        available = ", ".join(sorted(BRAINS))
        msg = f"Unknown brain {name!r}. Available: {available}"
        raise ValueError(msg)
    return BRAINS[name](**kwargs)


def get_brain_from_settings(name: str, settings: BrainSettings | None = None) -> MoveProvider:
    """Create a brain instance using the matching section from *settings*."""
    if name not in BRAINS:
        available = ", ".join(sorted(BRAINS))
        msg = f"Unknown brain {name!r}. Available: {available}"
        raise ValueError(msg)
    if settings is None:
        settings = BrainSettings()
    section = getattr(settings, name)
    return BRAINS[name](**section.model_dump())


__all__ = [
    "BRAINS",
    "AlphaBetaBrain",
    "MinimaxBrain",
    "NegamaxBrain",
    "RandomBrain",
    "ThreatBrain",
    "get_brain",
    "get_brain_from_settings",
]
