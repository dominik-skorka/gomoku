"""Logfire instrumentation + request-correlation IDs for the API.

Two pieces:

* :func:`configure_logfire` — call once at app construction.  Wires
  :mod:`logfire` to (optionally) ship logs to logfire.dev and/or print
  structured records to stderr.  Also instruments ``gomoku_fastapi``'s
  pydantic models for free validation tracing.

* :func:`install_request_id_middleware` — adds a per-request UUID,
  echoes it back as ``X-Request-ID`` (preserving any incoming header
  the client sent), and stores it in a :class:`ContextVar` so log
  records can reference it without a parameter-passing scaffold.

Both are safe in tests: ``configure_logfire`` defaults to no-op (no
network, no console) when called without explicit settings, and the
request-id middleware is pure-Python with no external state.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, override

import logfire
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from gomoku.settings import ApiSettings
    from starlette.requests import Request
    from starlette.responses import Response


# ---------------------------------------------------------------------------
# Request ID propagation
# ---------------------------------------------------------------------------

# Read by handlers / log formatters that want to tag work with the
# current request.  ``""`` means "outside any request" (e.g. startup
# / shutdown / background tasks).
_REQUEST_ID: ContextVar[str] = ContextVar("gomoku_request_id", default="")


def get_request_id() -> str:
    """Return the request ID for the current task, or ``""`` if none.

    Safe to call from any code reachable from a request handler —
    including code dispatched to ``anyio.to_thread.run_sync`` (anyio
    copies the context across the boundary).
    """
    return _REQUEST_ID.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Tag every request with an ``X-Request-ID`` header.

    If the client supplied one, we trust it (within a 64-char limit
    so a malicious header can't bloat log lines).  Otherwise we mint a
    fresh UUID4.  The value is set in a ContextVar for the duration
    of the request and echoed on the response.
    """

    HEADER = "X-Request-ID"
    MAX_LEN = 64

    @override
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        incoming = request.headers.get(self.HEADER, "")
        # Strip + length-cap any client-supplied value so it can't be
        # used to inject log noise.  Empty / oversized => mint our own.
        rid = incoming.strip()[: self.MAX_LEN] if incoming.strip() else uuid.uuid4().hex
        token = _REQUEST_ID.set(rid)
        try:
            response = await call_next(request)
        finally:
            _REQUEST_ID.reset(token)
        response.headers[self.HEADER] = rid
        return response


# ---------------------------------------------------------------------------
# Logfire setup
# ---------------------------------------------------------------------------


def configure_logfire(app: FastAPI, settings: ApiSettings) -> None:
    """Configure logfire and instrument *app*.

    The ``logfire.configure`` part is process-wide and idempotent —
    the latest call wins, which is what we want when the test conftest
    builds a fresh app per test.

    The ``instrument_fastapi`` part is per-app and **not** idempotent
    (OpenTelemetry refuses to double-instrument).  We swallow that
    specific error so the second ``create_app`` on the *same* app
    object (rare outside tests) doesn't crash.

    The ``send_to_logfire="if-token-present"`` mode lets a missing
    ``LOGFIRE_TOKEN`` degrade to local-only operation rather than a
    hard failure — important for dev environments without a token.
    """
    logfire.configure(
        send_to_logfire="if-token-present" if settings.logfire_send else False,
        console=logfire.ConsoleOptions() if settings.logfire_console else False,
        service_name="gomoku-api",
    )
    try:
        logfire.instrument_fastapi(app, capture_headers=True)
    except ValueError as exc:
        # OpenTelemetry raises a plain ValueError("This app has already
        # been instrumented.") when called twice on the same app object.
        # Other ValueErrors should still surface.
        if "already been instrumented" not in str(exc):
            raise


__all__ = [
    "RequestIDMiddleware",
    "configure_logfire",
    "get_request_id",
]
