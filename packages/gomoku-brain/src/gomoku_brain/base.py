"""The base class of every brain.

A brain is a :class:`~gomoku.interfaces.Player` that decides by itself.
Subclass :class:`Brain` with the brain's settings model and a registry
name, and implement :meth:`Brain.choose_move`; that is all::

    class GreedySettings(BaseBrainSettings):
        radius: int = Field(default=2, ge=1, le=4)

    class GreedyBrain(Brain[GreedySettings], name="greedy"):
        @override
        def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
            ...  # self.settings.radius is validated and typed

The class statement registers the brain under its name, so it can be
played (``gomoku play --black greedy,radius=3``), aliased in
``gomoku.toml`` and matched like any built-in one.  Forgetting
``choose_move`` is a type error, and instantiating such a class raises
``TypeError``.

:meth:`Brain.get_move` is final: every brain gets the same tracing span
and diagnostics handle (:func:`~gomoku_brain.brain_base.brain_session`)
around its decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import get_original_bases
from typing import TYPE_CHECKING, Any, ClassVar, cast, final, get_args, get_origin

from gomoku.settings import BaseBrainSettings, NoBrainSettings

from gomoku_brain.brain_base import BrainSession, brain_session

if TYPE_CHECKING:
    from gomoku.board import Board
    from gomoku.models import Move, Stone

# Every Brain subclass declared with a name, by that name.
REGISTRY: dict[str, type[Brain[Any]]] = {}


class Brain[S: BaseBrainSettings = NoBrainSettings](ABC):
    """A move-choosing brain with settings of type *S*.

    Subclass as ``class XBrain(Brain[XSettings], name="x")``: the name
    registers it, and *S* (read from the class statement) is the model its
    settings are validated with.  Settings come from a model instance or
    as keyword arguments, checked by that model (unknown keys are errors)::

        XBrain()                      # the model's defaults
        XBrain(depth=6)               # some changed
        XBrain(XSettings(depth=6))    # a ready model
    """

    #: Registry name, set by the ``name=`` class argument.
    brain_name: ClassVar[str] = ""
    #: The settings model, read from ``Brain[S]`` in the class statement.
    settings_type: ClassVar[type[BaseBrainSettings]] = NoBrainSettings

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in get_original_bases(cls):
            if get_origin(base) is Brain:
                (cls.settings_type,) = get_args(base)
        if name is not None:
            if name in REGISTRY and REGISTRY[name] is not cls:
                msg = f"Brain name {name!r} is taken by {REGISTRY[name].__qualname__}"
                raise ValueError(msg)
            cls.brain_name = name
            REGISTRY[name] = cls

    def __init__(self, settings: S | None = None, /, **overrides: Any) -> None:
        if settings is not None and overrides:
            msg = "pass either a settings model or keyword settings, not both"
            raise TypeError(msg)
        self.settings: S = settings if settings is not None else cast("S", self.settings_type(**overrides))

    @property
    def name(self) -> str:
        """Registry name (``Player.name``)."""
        return self.brain_name

    @final
    def get_move(self, board: Board, stone: Stone) -> Move:
        """Choose a move: :meth:`choose_move` inside a traced brain session."""
        with brain_session(self.name, stone=str(stone), **self.span_attributes()) as session:
            return self.choose_move(board, stone, session)

    @abstractmethod
    def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
        """The brain's decision.

        *board* is not full and *stone* is the side to move.  *session*
        gives the per-move diagnostics record (``session.diag``, ``None``
        when diagnostics are off) and the elapsed time.
        """

    def span_attributes(self) -> dict[str, object]:
        """Attributes of the tracing span: the settings' plain values."""
        return {k: v for k, v in self.settings.model_dump().items() if isinstance(v, bool | int | float | str)}
