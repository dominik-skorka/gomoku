"""``POST /v1/move`` — stateless single-shot brain query.

The client passes the full board, the side to move, and a brain
reference.  The server resolves the brain via the gomoku-brain
registry, runs ``get_move`` in a worker thread (so the event loop
isn't blocked by a multi-second alpha-beta search), and returns the
chosen move.  When ``diagnostics: true`` the server also activates the
:class:`MoveDiagnostics` ContextVar around the call and returns the
captured record alongside the move.
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, HTTPException, Request, status
from gomoku.diagnostics import (
    MoveDiagnostics,
    current_move_diagnostics,
    set_move_diagnostics,
)
from gomoku.models import Stone

from gomoku_fastapi.deps import SettingsDep, build_brain
from gomoku_fastapi.limits import compute_shared_limit
from gomoku_fastapi.schemas import MoveRequest, MoveResponse

router = APIRouter(tags=["moves"])


def _run_brain_with_diagnostics(brain, board, stone, capture: bool) -> tuple[object, MoveDiagnostics | None]:
    """Run ``brain.get_move`` and optionally capture diagnostics.

    Lives at module level (not inside the endpoint coroutine) so the
    whole call site can be passed straight to ``anyio.to_thread.run_sync``
    without requiring a thread-safe wrapper.

    The :class:`ContextVar` is set on the *worker thread* — anyio copies
    the parent task's context when ``run_sync`` dispatches, so each
    request gets its own isolated diagnostics record (no cross-talk
    between concurrent /v1/move calls).
    """
    if not capture:
        return brain.get_move(board, stone), None

    # Prefer an existing diag if one was already set by an outer caller
    # (eg. a future stateful-session endpoint that wants to record into
    # its own recorder).  Otherwise, create a fresh one for this call.
    diag = current_move_diagnostics()
    owns_diag = diag is None
    if owns_diag:
        diag = MoveDiagnostics(move_number=1, stone=stone)
        set_move_diagnostics(diag)
    try:
        move = brain.get_move(board, stone)
    finally:
        if owns_diag:
            set_move_diagnostics(None)
    return move, diag


@router.post(
    "/move",
    response_model=MoveResponse,
    summary="Get a brain's chosen move for a given board",
    responses={
        404: {"description": "Unknown brain name."},
        422: {"description": "Invalid request payload (bad side, no candidates, etc.)."},
        429: {"description": "Per-IP rate limit exceeded."},
    },
)
@compute_shared_limit
async def post_move(request: Request, req: MoveRequest, settings: SettingsDep) -> MoveResponse:
    """Resolve the brain, run ``get_move``, return the move (and optional diagnostics).

    The brain runs in a worker thread via :func:`anyio.to_thread.run_sync`
    so that a several-second search doesn't block the event loop and
    starve other concurrent requests.  Diagnostics ``ContextVar``s
    propagate through correctly because anyio copies the task context
    on dispatch.
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
    move, diag = await anyio.to_thread.run_sync(
        _run_brain_with_diagnostics, brain, req.board, req.side_to_move, req.diagnostics
    )
    return MoveResponse(move=move, brain=brain.name, diagnostics=diag)
