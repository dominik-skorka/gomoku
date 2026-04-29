"""Tests for the shared ``brain_session`` context manager."""

from __future__ import annotations

import time

from gomoku.diagnostics import (
    MoveDiagnostics,
    current_move_diagnostics,
    set_move_diagnostics,
)
from gomoku.models import Stone
from gomoku_brain.brain_base import BrainSession, brain_session


class TestBrainSession:
    def test_diag_is_none_when_no_active_context(self):
        # Make sure the ContextVar starts unset.
        set_move_diagnostics(None)
        with brain_session("test") as session:
            assert isinstance(session, BrainSession)
            assert session.diag is None

    def test_diag_picks_up_active_context(self):
        diag = MoveDiagnostics(move_number=1, stone=Stone.BLACK)
        set_move_diagnostics(diag)
        try:
            with brain_session("test") as session:
                assert session.diag is diag
        finally:
            set_move_diagnostics(None)

    def test_does_not_clear_context_on_exit(self):
        # ``brain_session`` is a passive observer — it must not touch
        # the ContextVar.  Ownership stays with whoever set it (the
        # CLI's DiagnosticsRecorder.begin_move / end_move pair).
        diag = MoveDiagnostics(move_number=1, stone=Stone.BLACK)
        set_move_diagnostics(diag)
        try:
            with brain_session("test"):
                pass
            assert current_move_diagnostics() is diag
        finally:
            set_move_diagnostics(None)

    def test_elapsed_ms_starts_at_zero(self):
        with brain_session("test") as session:
            # Cannot beat a few ms on any modern machine.
            assert session.elapsed_ms() < 50

    def test_elapsed_ms_advances(self):
        with brain_session("test") as session:
            time.sleep(0.01)
            assert session.elapsed_ms() >= 9  # allow ~1ms slack

    def test_attrs_forwarded_to_logfire_span(self):
        # Smoke test: if logfire didn't accept the kwargs, the context
        # manager would raise.  Real assertions about the span's
        # attributes would require instrumenting logfire, which is out
        # of scope here.
        with brain_session("test", depth=4, stone=str(Stone.BLACK)) as session:
            assert session.diag is None or isinstance(session.diag, MoveDiagnostics)
