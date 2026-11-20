"""``GET /versions`` — which API versions are served, and their status."""

from fastapi import APIRouter

from gomoku_fastapi.schemas import VersionInfo, VersionsResponse
from gomoku_fastapi.versioning import LATEST, VERSIONS, ApiVersion

router = APIRouter(tags=["meta"])


def version_info(version: ApiVersion) -> VersionInfo:
    """The public description of *version*, with its status as of today."""
    return VersionInfo(
        version=version.name,
        path=version.prefix,
        openapi=f"{version.prefix}/openapi.json",
        status=version.status(),
        released=version.released,
        deprecated=version.deprecated,
        sunset=version.sunset,
        successor=version.successor,
    )


@router.get("/versions", response_model=VersionsResponse, summary="List API versions and their status")
async def list_versions() -> VersionsResponse:
    """Unversioned on purpose, like ``/healthz``: a client discovers the
    available versions here before it picks one.  Deprecated versions'
    ``deprecation`` and ``sunset`` links point here.
    """
    return VersionsResponse(latest=LATEST.name, versions=[version_info(v) for v in VERSIONS])
