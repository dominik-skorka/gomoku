"""Per-IP rate limiting via slowapi.

slowapi wraps Flask-Limiter for ASGI; we use the in-memory backend
because the API is single-process by design (sessions are also in
memory — see ``sessions/store.py``).  When a Redis-backed deployment
arrives, swap the storage URI here and slowapi handles the rest.

Two distinct limits are enforced:

* ``compute_limit`` — applied to ``/v1/move`` and
  ``/v1/games/{id}/think``, the only endpoints that do real CPU work
  (alphabeta search runs for seconds, not microseconds).  Default
  cap: 30/minute per IP.
* ``default_limit`` — slowapi's ``default_limits``, applied to every
  route as a basic abuse guard.  Default: 120/minute per IP.

Both are sourced from ``settings.api`` so an operator can tighten or
loosen them via ``gomoku.toml`` without a code change.  The limit
strings are evaluated **per request** through callable hooks below
so a settings reload (``reset_settings_cache``) takes effect on the
next request, no app rebuild required.

Setting either limit to an empty string in ``gomoku.toml`` makes the
corresponding tier permissive — useful in tests that want to fire
rapid request loops without tripping 429s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.routing import iter_route_contexts
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware, _find_route_handler, _should_exempt, sync_check_limits
from slowapi.util import get_remote_address

from gomoku_fastapi.deps import get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from starlette.responses import Response


# Permissive sentinel used when an operator disables a tier.  slowapi
# requires *some* limit string for the callable to return; a value
# this large is effectively unbounded for any real workload.
_UNLIMITED = "1000000/minute"


def _default_limit() -> str:
    return get_settings().api.rate_limit_default or _UNLIMITED


def _compute_limit() -> str:
    return get_settings().api.rate_limit_compute or _UNLIMITED


# Module-level singleton so route decorators in
# ``routers/moves.py`` and ``routers/sessions.py`` import the same
# limiter ``app.state.limiter`` points at.  ``default_limits`` is a
# *callable* list — slowapi re-evaluates it per request, so a
# config reload picks up new numbers without a restart.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_default_limit],
    storage_uri="memory://",
)


def compute_limit() -> str:
    """Per-route limit for compute-heavy endpoints (callable form)."""
    return _compute_limit()


# A single shared bucket spans both ``/v1/move`` and
# ``/v1/games/{id}/think`` so a client can't dodge the cap by
# alternating between the two endpoints.  Routes opt in by decorating
# with ``@compute_shared_limit``.
compute_shared_limit = limiter.shared_limit(_compute_limit, scope="compute")


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Render :class:`RateLimitExceeded` as a clean 429 JSON envelope.

    slowapi's default handler returns ``text/plain``; we want the same
    ``{"detail": {"error": ..., ...}}`` shape every other 4xx in this
    API uses, so a client written against (e.g.) ``invalid_move`` can
    parse rate-limit responses with the same code path.
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={
            "detail": {
                "error": "rate_limit_exceeded",
                "limit": str(exc.detail),
            }
        },
        headers={"Retry-After": "60"},
    )


def install(app: FastAPI) -> None:
    """Attach the limiter + 429 handler + middleware to *app*.

    Idempotent — safe to call from each ``create_app`` invocation.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(_DefaultLimitMiddleware)


class _DefaultLimitMiddleware(SlowAPIMiddleware):
    """slowapi's middleware, finding the endpoint among *all* routes.

    slowapi looks the endpoint up in ``app.routes``.  FastAPI keeps
    included routers there as single nested entries without an
    ``endpoint``, so slowapi found no endpoint for any route of an
    included router, treated it as exempt, and the default limit
    silently applied to nothing.  ``iter_route_contexts`` lists every
    route with its full path, as FastAPI's own OpenAPI generation sees
    them.  Otherwise this is ``SlowAPIMiddleware.dispatch`` unchanged.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        app = request.app
        active: Limiter = app.state.limiter
        if not active.enabled:
            return await call_next(request)

        handler = _find_route_handler(list(iter_route_contexts(app.routes)), request.scope)
        if _should_exempt(active, handler):
            return await call_next(request)

        error_response, should_inject_headers = sync_check_limits(active, request, handler, app)
        if error_response is not None:
            return error_response

        response = await call_next(request)
        if should_inject_headers:
            response = active._inject_headers(response, request.state.view_rate_limit)
        return response


__all__ = [
    "compute_limit",
    "install",
    "limiter",
    "rate_limit_exceeded_handler",
]
