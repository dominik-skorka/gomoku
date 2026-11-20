"""Gomoku AI brains — pluggable move providers.

This module is the **single source of truth** for the brain registry.
Each entry in :data:`BRAINS` is a :class:`BrainSpec` pairing a brain
class with its pydantic settings model — adding a new brain means
appending exactly one entry here.

External consumers:

* :func:`get_brain` — instantiate by name with explicit kwargs.
* :func:`get_brain_from_settings` — instantiate by resolving an alias
  via a :class:`~gomoku.settings.BrainSettings` container.
* :func:`resolve_brain` — turn an alias name into a ``(type, kwargs)``
  pair without instantiating.
* :func:`register_brain` — add a brain at runtime (used by the CLI to
  register the keyboard provider).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gomoku.interfaces import Player
from gomoku.settings import (
    AlphaBetaBrainSettings,
    BrainSettings,
    MCTSBrainSettings,
    MinimaxBrainSettings,
    NegamaxBrainSettings,
    RandomBrainSettings,
    ThreatBrainSettings,
)
from pydantic import BaseModel

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.mcts import MCTSBrain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.negamax import NegamaxBrain
from gomoku_brain.random import RandomBrain
from gomoku_brain.threat import ThreatBrain


@dataclass(frozen=True)
class BrainSpec:
    """One row of the brain registry.

    Pairs a brain class with its pydantic settings model so the same
    name resolves to the same pair everywhere — no parallel dicts to
    keep in sync.
    """

    name: str
    brain_cls: type[Player]
    settings_cls: type[BaseModel]


BRAINS: dict[str, BrainSpec] = {
    spec.name: spec
    for spec in (
        BrainSpec("random", RandomBrain, RandomBrainSettings),
        BrainSpec("minimax", MinimaxBrain, MinimaxBrainSettings),
        BrainSpec("negamax", NegamaxBrain, NegamaxBrainSettings),
        BrainSpec("alphabeta", AlphaBetaBrain, AlphaBetaBrainSettings),
        BrainSpec("threat", ThreatBrain, ThreatBrainSettings),
        BrainSpec("mcts", MCTSBrain, MCTSBrainSettings),
    )
}
# ``KeyboardBrainSettings`` is defined in :mod:`gomoku.settings` for symmetry,
# but ``KeyboardProvider`` only exists in the CLI package — it is registered
# at startup via :func:`register_brain`.


def register_brain(name: str, brain_cls: type[Player], settings_cls: type[BaseModel]) -> None:
    """Add or overwrite a registry entry.

    Used by :mod:`gomoku_cli.main` to wire :class:`KeyboardProvider`
    into the registry without creating an upward dependency from
    :mod:`gomoku_brain` on the CLI.
    """
    BRAINS[name] = BrainSpec(name, brain_cls, settings_cls)


def get_brain(name: str, **kwargs: object) -> Player:
    """Instantiate a brain by name with explicit kwargs."""
    if name not in BRAINS:
        available = ", ".join(sorted(BRAINS))
        msg = f"Unknown brain {name!r}. Available: {available}"
        raise ValueError(msg)
    return BRAINS[name].brain_cls(**kwargs)


def resolve_brain(name: str, brain_settings: BrainSettings) -> tuple[str, dict[str, Any]]:
    """Resolve *name* through *brain_settings* into ``(type, kwargs)``.

    *name* may be either a built-in brain type (resolves to that type
    with default settings) or a TOML alias from ``[brains.<name>]``
    (resolves to the alias's ``type`` with its kwargs validated through
    the matching settings model).
    """
    raw = brain_settings.aliases().get(name)
    if raw is not None:
        if not isinstance(raw, dict) or "type" not in raw:
            msg = f"Brain alias {name!r} must be a table with a 'type' key"
            raise ValueError(msg)
        brain_type: str = raw["type"]
        params = {k: v for k, v in raw.items() if k != "type"}
    elif name in BRAINS:
        brain_type = name
        params = {}
    else:
        available = ", ".join(available_brains(brain_settings))
        msg = f"Unknown brain {name!r}. Available: {available}"
        raise ValueError(msg)

    if brain_type not in BRAINS:
        available = ", ".join(sorted(BRAINS))
        msg = f"Unknown brain type {brain_type!r} in alias {name!r}. Available types: {available}"
        raise ValueError(msg)

    settings_cls = BRAINS[brain_type].settings_cls
    validated = settings_cls(**params)
    return brain_type, validated.model_dump()


def available_brains(brain_settings: BrainSettings | None = None) -> list[str]:
    """Sorted list of all resolvable names — built-in types + TOML aliases.

    Used by the CLI to validate provider names and to build error messages.
    """
    names = set(BRAINS)
    if brain_settings is not None:
        names.update(brain_settings.aliases())
    return sorted(names)


def get_brain_from_settings(name: str, brain_settings: BrainSettings | None = None) -> Player:
    """Instantiate a brain by resolving *name* through *brain_settings*."""
    if brain_settings is None:
        brain_settings = BrainSettings()
    brain_type, kwargs = resolve_brain(name, brain_settings)
    return BRAINS[brain_type].brain_cls(**kwargs)


__all__ = [
    "BRAINS",
    "AlphaBetaBrain",
    "BrainSpec",
    "MCTSBrain",
    "MinimaxBrain",
    "NegamaxBrain",
    "RandomBrain",
    "ThreatBrain",
    "available_brains",
    "get_brain",
    "get_brain_from_settings",
    "register_brain",
    "resolve_brain",
]
