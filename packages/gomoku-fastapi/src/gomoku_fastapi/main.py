"""FastAPI application factory + ``gomoku-api`` entry point.

The factory pattern (``create_app``) lets tests build a fresh app per
test class without touching the module-level singleton.  Production
deploys either import ``app`` (the eagerly-constructed default) or run
the ``gomoku-api`` console script which delegates to uvicorn.
"""

from __future__ import annotations

from fastapi import FastAPI

from gomoku_fastapi.routers import analyze, brains, evaluate, health, moves


def create_app() -> FastAPI:
    """Build a fresh FastAPI app with all routers mounted."""
    app = FastAPI(
        title="Gomoku API",
        version="0.2.0",
        description=(
            "REST API for the Gomoku engine.  Stateless surface: query "
            "a brain for its move on a given board, evaluate a "
            "position, analyse a diagnostics recording, list registered "
            "brains, and check liveness.  Stateful sessions arrive in a "
            "later iteration."
        ),
    )
    app.include_router(health.router)
    app.include_router(brains.router, prefix="/v1")
    app.include_router(moves.router, prefix="/v1")
    app.include_router(evaluate.router, prefix="/v1")
    app.include_router(analyze.router, prefix="/v1")
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
