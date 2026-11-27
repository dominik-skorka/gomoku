"""``GET /v1/brains`` — registered brains, their settings schemas and their effective settings."""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Path as PathParam
from gomoku_brain import BRAINS

from gomoku_fastapi.deps import SettingsDep, effective_brain_settings
from gomoku_fastapi.schemas import BrainInfo, BrainSettingsResponse, BrainsResponse
from gomoku_fastapi.server_only import public_schema

router = APIRouter(tags=["brains"])


@router.get(
    "/brains",
    response_model=BrainsResponse,
    summary="List brains and their settings schemas",
)
async def list_brains() -> BrainsResponse:
    """Return one entry per brain in :data:`gomoku_brain.BRAINS`.

    Each entry includes the brain's *settings JSON Schema* so a UI can
    auto-render a configuration form without hard-coding the per-brain
    knob list (depth, ``vcf_depth``, exploration, …).  Settings only the
    server's configuration may set (file paths such as ``opening_book``)
    are not listed.
    """
    items = [
        BrainInfo(
            name=spec.name,
            settings_schema=public_schema(spec.settings_cls),
        )
        for spec in sorted(BRAINS.values(), key=lambda s: s.name)
    ]
    return BrainsResponse(brains=items)


@router.get(
    "/brains/{name}/settings",
    response_model=BrainSettingsResponse,
    summary="The settings a brain plays with on this server",
    responses={
        404: {"description": "Unknown brain name or alias."},
        422: {"description": "The server's configuration for this alias is invalid."},
    },
)
async def get_brain_settings(
    name: Annotated[str, PathParam(description="Registry name or ``[brains.<name>]`` alias.")],
    settings: SettingsDep,
) -> BrainSettingsResponse:
    """Return every setting a move with *name* uses when the request sends
    no ``brain_overrides``.

    The schema from ``GET /v1/brains`` gives each setting's built-in
    default; this server may play differently (a ``[brains.<name>]``
    alias in ``gomoku.toml``, the ``[api].max_time_limit_s`` cap).  A
    client that lets users override settings can show these values as
    the baseline the overrides change.
    """
    brain_type, values = effective_brain_settings(name, settings)
    return BrainSettingsResponse(name=name, type=brain_type, settings=values)
