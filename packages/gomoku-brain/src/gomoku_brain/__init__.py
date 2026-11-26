"""Gomoku AI brains — pluggable move providers.

Every brain subclasses :class:`~gomoku_brain.base.Brain` and registers
itself by declaring its name (``class XBrain(Brain[XSettings], name="x")``);
see :mod:`gomoku_brain.base` for how to write one.  :data:`BRAINS` maps each
name to a :class:`BrainSpec` pairing the brain class with its settings
model.  Only brain modules imported here (or by the program) are
registered, so a new built-in brain is also imported below.

External consumers:

* :func:`get_brain` — instantiate by name with explicit kwargs.
* :func:`get_brain_from_settings` — instantiate by resolving an alias
  via a :class:`~gomoku.settings.BrainSettings` container.
* :func:`resolve_brain` — turn an alias name into a ``(type, kwargs)``
  pair without instantiating.
* :func:`register_brain` — add a brain at runtime (used by the CLI to
  register the keyboard provider).
* :func:`load_plugins` — import the brains other installed packages
  declare under the ``gomoku.brains`` entry-point group.  The registry
  calls it on first use, so the CLI, the API and match workers all see
  plug-in brains without doing anything.

A plug-in package declares its brain module in its ``pyproject.toml``::

    [project.entry-points."gomoku.brains"]
    greedy = "greedy_brain"

Importing the module registers the brain, as for a built-in one.  A
plug-in that fails to import (or whose brain name is taken) is skipped
with a :class:`PluginWarning`; the other brains still work.  See
``docs/WRITING_A_BRAIN.md``.
"""

import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, override

from gomoku.interfaces import Player
from gomoku.settings import BaseBrainSettings, BrainSettings
from pydantic import BaseModel

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.base import REGISTRY, Brain
from gomoku_brain.mcts import MCTSBrain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.negamax import NegamaxBrain
from gomoku_brain.random import RandomBrain
from gomoku_brain.threat import ThreatBrain

#: Entry-point group plug-in packages declare their brain modules under.
PLUGIN_GROUP = "gomoku.brains"


class PluginWarning(RuntimeWarning):
    """A plug-in brain could not be loaded and was skipped."""


_plugins_loaded = False


def load_plugins() -> list[str]:
    """Import every module declared under the ``gomoku.brains`` entry points.

    Runs once per process (later calls do nothing and return ``[]``); the
    registry calls it on first use.  Returns the entry-point names that
    loaded.  A plug-in that raises while importing, for example because
    its brain name is already taken, is skipped with a
    :class:`PluginWarning`.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return []
    _plugins_loaded = True
    loaded: list[str] = []
    for ep in entry_points(group=PLUGIN_GROUP):
        try:
            ep.load()
        except Exception as exc:
            msg = f"brain plug-in {ep.name!r} ({ep.value}) failed to load: {exc}"
            warnings.warn(msg, PluginWarning, stacklevel=2)
        else:
            loaded.append(ep.name)
    return loaded


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


class _Registry(Mapping[str, BrainSpec]):
    """The brain registry: every :class:`~gomoku_brain.base.Brain` subclass
    declared with a ``name=`` (they register themselves), plus players
    added with :func:`register_brain`.  A live view, so a brain defined
    after this module is imported (a plug-in, a test) shows up too.
    Plug-ins (:func:`load_plugins`) are imported on first access.
    """

    def __init__(self) -> None:
        self._extra: dict[str, BrainSpec] = {}

    def _specs(self) -> dict[str, BrainSpec]:
        load_plugins()
        specs = {name: BrainSpec(name, cls, cls.settings_type) for name, cls in REGISTRY.items()}
        return {**specs, **self._extra}

    @override
    def __getitem__(self, name: str) -> BrainSpec:
        load_plugins()
        if name in self._extra:
            return self._extra[name]
        cls = REGISTRY[name]
        return BrainSpec(name, cls, cls.settings_type)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._specs())

    @override
    def __len__(self) -> int:
        return len(self._specs())

    def add(self, spec: BrainSpec) -> None:
        self._extra[spec.name] = spec

    def remove(self, name: str) -> None:
        self._extra.pop(name, None)
        REGISTRY.pop(name, None)


#: name -> :class:`BrainSpec` for every registered brain and player.
BRAINS = _Registry()
# Built-in brains register themselves when their modules are imported above.
# ``KeyboardProvider`` only exists in the CLI package; it is a Player but not
# a Brain, and is added at startup via :func:`register_brain`.


def register_brain(name: str, brain_cls: type[Player], settings_cls: type[BaseModel]) -> None:
    """Add or overwrite a registry entry for a :class:`Player` class.

    Brains (subclasses of :class:`~gomoku_brain.base.Brain`) do not need
    this: declaring the class with ``name=`` registers it.  Used by
    :mod:`gomoku_cli.main` for :class:`KeyboardProvider`.
    """
    BRAINS.add(BrainSpec(name, brain_cls, settings_cls))


def unregister_brain(name: str) -> None:
    """Remove *name* from the registry (a brain or a registered player), if there."""
    BRAINS.remove(name)


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
    "PLUGIN_GROUP",
    "AlphaBetaBrain",
    "BaseBrainSettings",
    "Brain",
    "BrainSpec",
    "MCTSBrain",
    "MinimaxBrain",
    "NegamaxBrain",
    "PluginWarning",
    "RandomBrain",
    "ThreatBrain",
    "available_brains",
    "get_brain",
    "get_brain_from_settings",
    "load_plugins",
    "register_brain",
    "resolve_brain",
    "unregister_brain",
]
