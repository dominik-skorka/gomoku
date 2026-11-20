"""Common helpers for ``Player`` implementations.

Each non-trivial brain's ``get_move`` opens the same kind of session:

1. A Logfire span for tracing (``brain.<name>``).
2. A monotonic-clock timer so ``SearchStats.time_ms`` can be filled at
   the end of the search.
3. A lookup of the active per-move diagnostics ``ContextVar``.

:func:`brain_session` packages all three so brains avoid copy-pasting
that setup at the top of every ``get_move``.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import logfire
from gomoku.diagnostics import MoveDiagnostics, current_move_diagnostics


@dataclass
class BrainSession:
    """Per-call handle yielded by :func:`brain_session`.

    Attributes:
        diag: The active :class:`MoveDiagnostics`, or ``None`` when
            diagnostics are disabled.  Brains opt in by checking
            ``if session.diag is not None: ...``.

    Use :meth:`elapsed_ms` to fill ``SearchStats.time_ms`` at the end
    of the search.
    """

    diag: MoveDiagnostics | None
    _t0: float

    def elapsed_ms(self) -> int:
        """Wall-clock milliseconds since the session opened."""
        return int((time.monotonic() - self._t0) * 1000)


@contextmanager
def brain_session(brain_name: str, **attrs: Any) -> Iterator[BrainSession]:
    """Open a Logfire span + diagnostics handle for one ``get_move`` call.

    *brain_name* is the registry name of the brain (``"alphabeta"``,
    ``"mcts"``, …); the Logfire span is named ``brain.<brain_name>``.
    Any keyword arguments are forwarded to the span as attributes — use
    them to attach search-relevant configuration (depth, iterations,
    stone) for tracing.

    Example::

        with brain_session("alphabeta", depth=self.depth, stone=str(stone)) as bs:
            ...
            if bs.diag is not None:
                bs.diag.search_stats = SearchStats(time_ms=bs.elapsed_ms(), ...)
    """
    with logfire.span(f"brain.{brain_name}", **attrs):
        yield BrainSession(diag=current_move_diagnostics(), _t0=time.monotonic())
