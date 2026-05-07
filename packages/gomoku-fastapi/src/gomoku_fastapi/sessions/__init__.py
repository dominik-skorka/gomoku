"""Stateful session subsystem for the FastAPI app.

The :class:`SessionStore` is an in-memory ``dict[str, Session]`` with
TTL-based eviction; one instance per app lifecycle.  Each
:class:`Session` owns a :class:`gomoku.runner.GameRunner` plus optional
recorder / diagnostics-recorder, plus the metadata the API needs to
answer ``GET /v1/games/{id}`` (creation time, last access, brain
configuration).

Deliberately NOT persistent.  The architecture-review thread plan
calls this out: a process restart drops sessions, horizontal scaling
needs sticky routing.  Both are acceptable for the API's current
scope; a Redis-backed alternative can replace this module later
without touching the routers.
"""

from gomoku_fastapi.sessions.store import Session, SessionStore

__all__ = ["Session", "SessionStore"]
