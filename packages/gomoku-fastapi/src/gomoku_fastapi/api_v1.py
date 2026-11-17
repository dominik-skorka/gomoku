"""Routes of API version 1, served under ``/v1``.

When a breaking change needs a ``v2``, add an ``api_v2`` module that
includes the changed routers plus every unchanged one from here, and
register the version in :mod:`gomoku_fastapi.versioning`.
"""

from __future__ import annotations

from fastapi import APIRouter

from gomoku_fastapi.routers import analyze, brains, evaluate, moves, sessions
from gomoku_fastapi.versioning import V1

router = APIRouter(prefix=V1.prefix)
for _resource in (brains.router, moves.router, evaluate.router, analyze.router, sessions.router):
    router.include_router(_resource)
