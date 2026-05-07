"""FastAPI application factory + ``gomoku-api`` entry point.

The factory pattern (``create_app``) lets tests build a fresh app per
test class without touching the module-level singleton.  Production
deploys either import ``app`` (the eagerly-constructed default) or run
the ``gomoku-api`` console script which delegates to uvicorn.

``create_app`` wires four cross-cutting concerns on top of the routers:

* **Lifespan** — graceful shutdown that flushes any open per-session
  recorders so partial recordings don't lose their last few moves on
  Ctrl-C.
* **Logfire** — structured access logs and request spans (silent unless
  ``settings.api.logfire_console`` or ``logfire_send`` is set).
* **CORS** — opt-in via ``settings.api.cors_origins``; not mounted at
  all when the list is empty (server-to-server deployments).
* **Rate limiting** — slowapi-based per-IP limits.  Always installed;
  the limit numbers come from ``settings.api`` so an operator can
  loosen them without a code change.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gomoku_fastapi import limits, observability
from gomoku_fastapi.deps import get_session_store, get_settings
from gomoku_fastapi.routers import analyze, brains, evaluate, health, moves, sessions

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan hook — clear live sessions on shutdown.

    Sessions are in-memory (see ``sessions/store.py``).  When uvicorn
    receives SIGTERM, walking the store on shutdown gives us a hook
    point for any future flush logic (a Redis-backed store would
    finalise here, for example).  Today's :class:`GameRecorder` writes
    incrementally so partial recordings are already on disk — this
    handler is a structural placeholder kept here so the dep is wired
    in from day one rather than retrofitted later.

    Yielded value is unused; the contextmanager shape is what
    FastAPI's lifespan API expects.
    """
    yield
    # Drop session references so any held resources (locks, recorder
    # file handles inside the GameRunner) get GC'd promptly.  This is
    # a no-op for the in-process store (the process is exiting anyway)
    # but matters for tests that build apps in a loop.
    store = get_session_store()
    store._sessions.clear()


def create_app() -> FastAPI:
    """Build a fresh FastAPI app with all routers and middleware mounted."""
    settings = get_settings()
    app = FastAPI(
        title="Gomoku API",
        version="0.4.0",
        description=(
            "REST API for the Gomoku engine.  Stateless surface (move, "
            "evaluate, analyze) plus stateful sessions for live play "
            "with mixed external/brain players.  Iteration 4 adds "
            "structured logging (logfire), per-IP rate limits "
            "(slowapi), and CORS.  See "
            "``packages/gomoku-fastapi/README.md`` for the workflow."
        ),
        lifespan=_lifespan,
    )

    # Order matters: FastAPI/Starlette runs middleware in *reverse*
    # add order on the request path — the LAST added runs FIRST.  We
    # want CORS to handle preflights before anything else, then the
    # request-id middleware so logs get a stable ID, then slowapi as
    # the innermost layer (so rate-limited requests still get a
    # request ID echoed on the 429 response).
    limits.install(app)
    app.add_middleware(observability.RequestIDMiddleware)
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    observability.configure_logfire(app, settings.api)

    app.include_router(health.router)
    app.include_router(brains.router, prefix="/v1")
    app.include_router(moves.router, prefix="/v1")
    app.include_router(evaluate.router, prefix="/v1")
    app.include_router(analyze.router, prefix="/v1")
    app.include_router(sessions.router, prefix="/v1")
    return app


# The module-level ``app`` is what ``uvicorn gomoku_fastapi.main:app``
# imports.  Tests should prefer ``create_app()`` for isolation.
app = create_app()


def run() -> None:
    """``gomoku-api`` console-script entry point.

    Defers the uvicorn import so that ``gomoku_fastapi`` is cheap to
    import in test contexts that don't actually want to start a server.
    """
    import uvicorn

    uvicorn.run("gomoku_fastapi.main:app", host="127.0.0.1", port=8000, reload=False)


__all__ = ["app", "create_app", "run"]
