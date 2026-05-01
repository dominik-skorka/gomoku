"""``GET /v1/brains`` — list registered brains and their settings schemas."""

from __future__ import annotations

from fastapi import APIRouter
from gomoku_brain import BRAINS

from gomoku_fastapi.schemas import BrainInfo, BrainsResponse

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
    knob list (depth, ``vcf_depth``, exploration, …).
    """
    items = [
        BrainInfo(
            name=spec.name,
            settings_schema=spec.settings_cls.model_json_schema(),
        )
        for spec in sorted(BRAINS.values(), key=lambda s: s.name)
    ]
    return BrainsResponse(brains=items)
