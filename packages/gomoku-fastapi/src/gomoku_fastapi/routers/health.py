"""Liveness + readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["meta"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Returns 200 ``{"status": "ok"}`` when the process is up.

    Deliberately minimal: a successful response means the FastAPI
    handler chain is functional, not that brain JIT compilation has
    completed.  Use a separate readiness probe if you need that.
    """
    return {"status": "ok"}
