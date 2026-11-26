"""Registries of the pluggable search strategies, by name.

A search brain's settings choose its :class:`~gomoku.interfaces.Evaluator`
and :class:`~gomoku.interfaces.MoveGenerator` by name
(``evaluator = "pattern"``, ``move_generator = "radius"``), like brains
are chosen.  The names live here, in the core package, so the settings
models can validate them; the implementations are registered by
``gomoku_brain`` when it is imported (``"pattern"`` = its
``evaluate_board``, ``"radius"`` = its ``get_candidate_moves``)::

    from gomoku.strategies import register_evaluator

    def stones_only(board, stone, /):
        ...

    register_evaluator("stones", stones_only)
    # gomoku play --black alphabeta,evaluator=stones

A move generator is registered as a factory taking the candidate radius
the brain would search at, so one name works at every radius a brain
uses.  Brains resolve their names once, when built: register before
building the brain.
"""

from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gomoku.interfaces import Evaluator, MoveGenerator

type MoveGeneratorFactory = Callable[[int], MoveGenerator]
"""Makes the :class:`~gomoku.interfaces.MoveGenerator` for a candidate radius."""

#: Evaluator by name.
EVALUATORS: dict[str, Evaluator] = {}
#: Move generator factory by name.
MOVE_GENERATORS: dict[str, MoveGeneratorFactory] = {}


def register_evaluator(name: str, evaluator: Evaluator) -> None:
    """Add (or replace) the evaluator called *name*."""
    EVALUATORS[name] = evaluator


def register_move_generator(name: str, factory: MoveGeneratorFactory) -> None:
    """Add (or replace) the move generator called *name*, as a factory of the radius."""
    MOVE_GENERATORS[name] = factory


def unregister_strategy(name: str) -> None:
    """Remove *name* from both registries, if there (for tests and plug-ins)."""
    EVALUATORS.pop(name, None)
    MOVE_GENERATORS.pop(name, None)


def _check(name: str, registry: Collection[str], kind: str) -> str:
    if name not in registry:
        available = ", ".join(sorted(registry)) or "none (is gomoku_brain imported?)"
        msg = f"unknown {kind} {name!r}; registered: {available}"
        raise ValueError(msg)
    return name


def check_evaluator(name: str) -> str:
    """*name* if it is a registered evaluator, else ``ValueError`` listing them."""
    return _check(name, EVALUATORS, "evaluator")


def check_move_generator(name: str) -> str:
    """*name* if it is a registered move generator, else ``ValueError`` listing them."""
    return _check(name, MOVE_GENERATORS, "move generator")


def get_evaluator(name: str) -> Evaluator:
    """The evaluator called *name* (``ValueError`` if none)."""
    return EVALUATORS[check_evaluator(name)]


def get_move_generator(name: str, radius: int) -> MoveGenerator:
    """The move generator called *name*, at *radius* (``ValueError`` if none)."""
    return MOVE_GENERATORS[check_move_generator(name)](radius)
