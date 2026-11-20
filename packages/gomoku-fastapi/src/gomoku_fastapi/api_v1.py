"""Routes of API version 1, served under ``/v1``.

When a breaking change needs a ``v2``, add an ``api_v2`` module that
includes the changed routers plus every unchanged one from here, and
register the version in :mod:`gomoku_fastapi.versioning`.
"""

from fastapi import APIRouter

from gomoku_fastapi.routers import analyze, brains, evaluate, moves, sessions
from gomoku_fastapi.routers.versions import version_info
from gomoku_fastapi.schemas import VersionInfo
from gomoku_fastapi.versioning import V1

router = APIRouter(prefix=V1.prefix)


@router.get("", response_model=VersionInfo, tags=["meta"], summary="This API version and its status")
async def version_home() -> VersionInfo:
    """The version's home resource: its status, dates and OpenAPI document.

    A deprecated version's ``successor-version`` link points at the
    successor's home resource, so it always resolves.
    """
    return version_info(V1)


for _resource in (brains.router, moves.router, evaluate.router, analyze.router, sessions.router):
    router.include_router(_resource)
