"""Gomoku AI brains — pluggable move providers."""

from gomoku.protocol import MoveProvider

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.negamax import NegamaxBrain
from gomoku_brain.random import RandomBrain

BRAINS: dict[str, type] = {
    "random": RandomBrain,
    "minimax": MinimaxBrain,
    "negamax": NegamaxBrain,
    "alphabeta": AlphaBetaBrain,
}


def get_brain(name: str, **kwargs: object) -> MoveProvider:
    """Create a brain instance by name."""
    if name not in BRAINS:
        available = ", ".join(sorted(BRAINS))
        msg = f"Unknown brain {name!r}. Available: {available}"
        raise ValueError(msg)
    return BRAINS[name](**kwargs)


__all__ = [
    "BRAINS",
    "AlphaBetaBrain",
    "MinimaxBrain",
    "NegamaxBrain",
    "RandomBrain",
    "get_brain",
]
