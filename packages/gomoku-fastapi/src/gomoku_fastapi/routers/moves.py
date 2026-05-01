"""``POST /v1/move`` — stateless single-shot brain query.

The client passes the full board, the side to move, and a brain
reference.  The server resolves the brain via the gomoku-brain
registry, runs ``get_move`` in a worker thread (so the event loop
isn't blocked by a multi-second alpha-beta search), and returns the
chosen move.
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, HTTPException, status
from gomoku.models import Stone

from gomoku_fastapi.deps import SettingsDep, build_brain
from gomoku_fastapi.schemas import MoveRequest, MoveResponse

router = APIRouter(tags=["moves"])


@router.post(
    "/move",
    response_model=MoveResponse,
    summary="Get a brain's chosen move for a given board",
    responses={
        404: {"description": "Unknown brain name."},
        422: {"description": "Invalid request payload (bad side, no candidates, etc.)."},
    },
)
async def post_move(req: MoveRequest, settings: SettingsDep) -> MoveResponse:
    """Resolve the brain, run ``get_move``, return the move.

    The brain runs in a worker thread via :func:`anyio.to_thread.run_sync`
    so that a several-second search doesn't block the event loop and
    starve other concurrent requests.  This is the standard FastAPI
    pattern for CPU-bound work — diagnostics ``ContextVar``s propagate
    through correctly.
    """
    if req.side_to_move is Stone.EMPTY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "invalid_side_to_move",
                "message": "side_to_move must be Stone.BLACK (1) or Stone.WHITE (2)",
            },
        )

    if req.board.is_full():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "board_full", "message": "no legal moves available"},
        )

    brain = build_brain(req.brain, req.brain_overrides, settings)
    move = await anyio.to_thread.run_sync(brain.get_move, req.board, req.side_to_move)
    return MoveResponse(move=move, brain=brain.name)
