"""``POST /v1/evaluate`` — static heuristic evaluation of a board.

No brain is involved.  Calls
:func:`gomoku_brain.evaluate.evaluate_board` directly, returning the
clamped heuristic score from the requested side's perspective.

The whole call is a few hundred microseconds at most (the JIT path is
hot after the first request), so we do *not* dispatch to a worker
thread — the cost of context-switching would dominate the work itself.
"""

from fastapi import APIRouter, HTTPException, status
from gomoku.models import Stone
from gomoku_brain.evaluate import evaluate_board

from gomoku_fastapi.schemas import EvaluateRequest, EvaluateResponse

router = APIRouter(tags=["evaluate"])


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Static heuristic eval of a board for a given side",
    responses={
        422: {"description": "Invalid side (must be BLACK or WHITE, not EMPTY)."},
    },
)
async def post_evaluate(req: EvaluateRequest) -> EvaluateResponse:
    if req.side_to_move is Stone.EMPTY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "invalid_side_to_move",
                "message": "side_to_move must be Stone.BLACK (1) or Stone.WHITE (2)",
            },
        )
    score = evaluate_board(req.board, req.side_to_move)
    return EvaluateResponse(score=score)
