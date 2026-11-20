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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from gomoku_fastapi import api_v1, limits, observability
from gomoku_fastapi.deps import get_session_store, get_settings
from gomoku_fastapi.routers import health, versions
from gomoku_fastapi.versioning import EXPOSED_HEADERS, VERSIONS, ApiVersion, VersionHeadersMiddleware

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
    store.clear()


def _package_version() -> str:
    """Version of the installed ``gomoku-fastapi`` package (its pyproject version)."""
    try:
        return package_version("gomoku-fastapi")
    except PackageNotFoundError:  # running from a source tree that was never installed
        return "0+unknown"


def _add_version_openapi(app: FastAPI, version: ApiVersion) -> None:
    """Serve ``/<version>/openapi.json``: an OpenAPI document of that version only.

    ``/openapi.json`` covers every version plus the infrastructure
    endpoints; a client of one version gets exactly its contract here.
    """
    cache: dict[str, object] = {}

    async def version_openapi() -> object:
        if "doc" not in cache:
            routes = [
                r
                for r in app.routes
                if getattr(r, "path", "") == version.prefix or getattr(r, "path", "").startswith(f"{version.prefix}/")
            ]
            cache["doc"] = get_openapi(
                title=f"{app.title} ({version.name})",
                version=app.version,
                description=app.description,
                routes=routes,
            )
        return cache["doc"]

    app.add_api_route(f"{version.prefix}/openapi.json", version_openapi, include_in_schema=False)


def create_app() -> FastAPI:
    """Build a fresh FastAPI app with all routers and middleware mounted."""
    settings = get_settings()
    app = FastAPI(
        title="Gomoku API",
        version=_package_version(),
        description=(
            "REST API for the Gomoku engine.  Stateless surface (move, "
            "evaluate, analyze) plus stateful sessions for live play "
            "with mixed external/brain players.\n\n"
            "Versioned by URL prefix: every endpoint except ``/healthz`` "
            "and ``/versions`` lives under a version such as ``/v1``, and "
            "a version only ever changes in backwards-compatible ways.  "
            "``GET /versions`` lists the served versions and their status; "
            "responses carry an ``API-Version`` header, plus ``Deprecation`` "
            "and ``Sunset`` headers once their version is deprecated.  See "
            "``packages/gomoku-fastapi/README.md`` for the workflow."
        ),
        lifespan=_lifespan,
    )

    # Order matters: FastAPI/Starlette runs middleware in *reverse*
    # add order on the request path — the LAST added runs FIRST.  We
    # want CORS to handle preflights before anything else, then the
    # version-header middleware, then the request-id middleware so logs
    # get a stable ID, then slowapi as the innermost layer (so
    # rate-limited requests still get a request ID and version headers
    # on the 429 response).
    limits.install(app)
    app.add_middleware(observability.RequestIDMiddleware)
    # Outside the request-ID and rate-limit layers, so every response
    # under a version prefix (429s and errors included) carries its
    # version headers.
    app.add_middleware(VersionHeadersMiddleware)
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            # Browsers hide response headers from scripts unless exposed.
            expose_headers=[observability.RequestIDMiddleware.HEADER, *EXPOSED_HEADERS],
        )

    observability.configure_logfire(app, settings.api)

    # Unversioned infrastructure endpoints, then one router per version.
    app.include_router(health.router)
    app.include_router(versions.router)
    app.include_router(api_v1.router)
    for version in VERSIONS:
        _add_version_openapi(app, version)
    return app


# The module-level ``app`` is what ``uvicorn gomoku_fastapi.main:app``
# imports.  Tests should prefer ``create_app()`` for isolation.
app = create_app()


def run() -> None:
    """``gomoku-api`` console-script entry point.

    Defers the uvicorn import so that ``gomoku_fastapi`` is cheap to
    import in test contexts that don't actually want to start a server.
    """
    import sysconfig

    import uvicorn

    # On a free-threaded Python, use the pure-Python h11 parser: httptools
    # (uvicorn's default when installed) has not declared it can run
    # without the GIL, so importing it would switch the GIL back on for
    # the whole process, brain threads included.
    http = "h11" if sysconfig.get_config_var("Py_GIL_DISABLED") else "auto"
    uvicorn.run("gomoku_fastapi.main:app", host="127.0.0.1", port=8000, reload=False, http=http)


__all__ = ["app", "create_app", "run"]
