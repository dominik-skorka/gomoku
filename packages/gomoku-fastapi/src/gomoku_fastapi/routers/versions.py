"""``GET /versions`` — which API versions are served, and their status."""

from __future__ import annotations

from fastapi import APIRouter

from gomoku_fastapi.schemas import VersionInfo, VersionsResponse
from gomoku_fastapi.versioning import LATEST, VERSIONS

router = APIRouter(tags=["meta"])


@router.get("/versions", response_model=VersionsResponse, summary="List API versions and their status")
async def list_versions() -> VersionsResponse:
    """Unversioned on purpose, like ``/healthz``: a client discovers the
    available versions here before it picks one.
    """
    return VersionsResponse(
        latest=LATEST.name,
        versions=[
            VersionInfo(
                version=v.name,
                path=v.prefix,
                status=v.status,
                released=v.released,
                deprecated=v.deprecated,
                sunset=v.sunset,
                successor=v.successor,
            )
            for v in VERSIONS
        ],
    )
