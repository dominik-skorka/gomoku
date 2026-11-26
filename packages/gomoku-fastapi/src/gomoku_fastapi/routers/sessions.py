"""``/v1/games`` — stateful session endpoints.

Sessions live in an in-memory :class:`SessionStore` (one process, no
persistence — see ``sessions/store.py``).  Each session owns a
:class:`gomoku.runner.GameRunner` plus optional recorder /
diagnostics-recorder.

Five endpoints make a complete UI workflow possible:

* ``POST /v1/games``                  — create a session
* ``GET  /v1/games/{id}``             — read state (board, history, result)
* ``DELETE /v1/games/{id}``           — drop the session
* ``POST /v1/games/{id}/moves``       — submit an external move
* ``POST /v1/games/{id}/think``       — ask the configured brain for its move

Errors map to clean status codes: 404 for unknown session, 422 for
wrong-turn / illegal move / no-brain-configured / game-already-over,
500 only for genuine surprises.
"""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Annotated

import anyio
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi import Path as PathParam
from gomoku.diagnostics import DiagnosticsRecorder
from gomoku.game import InvalidMoveError
from gomoku.models import GameConfig, Stone
from gomoku.recorder import GameRecorder, PlayerInfo
from gomoku.runner import (
    GameAlreadyOverError,
    GameRunner,
    NoBrainConfiguredError,
    WrongTurnError,
)

from gomoku_fastapi.deps import SessionStoreDep, SettingsDep, build_brain
from gomoku_fastapi.limits import compute_shared_limit
from gomoku_fastapi.schemas import (
    CreateGameRequest,
    GameStateResponse,
    HistoryEntry,
    PlayerConfig,
    SubmitMoveRequest,
    ThinkRequest,
    ThinkResponse,
)
from gomoku_fastapi.sessions import Session
from gomoku_fastapi.sessions.store import SessionNotFoundError

router = APIRouter(prefix="/games", tags=["games"])

# Path-param dependency reused by every per-session endpoint — declared
# once so the description appears identically in the OpenAPI docs.
GameIdPath = Annotated[
    str,
    PathParam(description="Session UUID returned by ``POST /v1/games``."),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(session: Session) -> GameStateResponse:
    """Snapshot the runner into a :class:`GameStateResponse`."""
    runner = session.runner
    return GameStateResponse(
        game_id=session.id,
        board=runner.board,
        side_to_move=runner.current_stone,
        move_number=len(runner.move_history) + 1,
        history=[HistoryEntry(stone=s, move=m) for s, m in runner.move_history],
        black_brain=session.black_brain,
        white_brain=session.white_brain,
        result=runner.result,
        record_dir=session.record_dir,
        diag_dir=session.diag_dir,
    )


def _lookup(store, game_id: str) -> Session:
    """Translate :class:`SessionNotFoundError` to a clean 404."""
    try:
        return store.get(game_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "game_id": exc.session_id},
        ) from exc


def _player_info(stone: Stone, cfg: PlayerConfig) -> PlayerInfo:
    """Build a :class:`PlayerInfo` for the recorder."""
    return PlayerInfo(
        stone=stone,
        provider=cfg.brain or "external",
        settings=cfg.brain_overrides or {},
    )


# ---------------------------------------------------------------------------
# Create / read / delete
# ---------------------------------------------------------------------------


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a session",
    responses={
        404: {"description": "One of the requested brains is unknown."},
        422: {"description": "Invalid request payload (bad config, brain overrides)."},
    },
)
def create_game(
    req: CreateGameRequest,
    settings: SettingsDep,
    store: SessionStoreDep,
) -> GameStateResponse:
    """Build a fresh :class:`GameRunner` and register it in the store.

    The returned ``game_id`` is the only handle clients need — pass it
    in subsequent ``/v1/games/{id}/...`` calls.  Brains (when
    requested) are instantiated immediately so a bad config produces a
    clean error at create-time, not on the first ``/think`` call.
    """
    config = GameConfig(board_size=req.board_size, win_length=req.win_length)

    # Resolve brains up-front.  ``build_brain`` raises HTTPException on
    # unknown brains / invalid overrides — propagate as-is.
    black_brain = build_brain(req.black.brain, req.black.brain_overrides, settings) if req.black.brain else None
    white_brain = build_brain(req.white.brain, req.white.brain_overrides, settings) if req.white.brain else None

    # Recording / diagnostics live in a per-session temp dir so two
    # sessions can't fight over filenames.  ``mkdtemp`` returns an
    # absolute path; recorders create files inside it.
    record_dir: str | None = None
    diag_dir: str | None = None
    recorder: GameRecorder | None = None
    diag_recorder: DiagnosticsRecorder | None = None

    started_at = datetime.now(tz=UTC)
    if req.record:
        record_dir = mkdtemp(prefix="gomoku-session-rec-")
        recorder = GameRecorder(
            game_config=config,
            players=[_player_info(Stone.BLACK, req.black), _player_info(Stone.WHITE, req.white)],
            directory=Path(record_dir),
        )
    if req.diagnostics:
        diag_dir = mkdtemp(prefix="gomoku-session-diag-")
        diag_recorder = DiagnosticsRecorder(
            game_config=config,
            players=[
                {"stone": int(Stone.BLACK), "provider": req.black.brain or "external"},
                {"stone": int(Stone.WHITE), "provider": req.white.brain or "external"},
            ],
            directory=Path(diag_dir),
            started_at=started_at,
        )

    runner = GameRunner(
        config,
        black=black_brain,
        white=white_brain,
        diagnostics_recorder=diag_recorder,
        # The recorder observes the game, so the recording is finished
        # (result, checksum) when it ends; it used to stay partial.
        observers=[recorder] if recorder is not None else [],
    )
    session = store.create(
        runner,
        black_brain=req.black.brain,
        white_brain=req.white.brain,
        record_dir=record_dir,
        diag_dir=diag_dir,
    )
    return _state(session)


@router.get(
    "/{game_id}",
    summary="Read the current state of a session",
    responses={404: {"description": "Session not found or expired."}},
)
def get_game(game_id: GameIdPath, store: SessionStoreDep) -> GameStateResponse:
    return _state(_lookup(store, game_id))


@router.delete(
    "/{game_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Drop a session",
)
def delete_game(game_id: GameIdPath, store: SessionStoreDep) -> Response:
    """Idempotent: deleting a missing session is a no-op (still 204).

    Returning 204 for unknown IDs lets clients retry safely without
    branching on whether the session is "really" gone.
    """
    store.delete(game_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Move + think
# ---------------------------------------------------------------------------


@router.post(
    "/{game_id}/moves",
    summary="Submit an external move",
    responses={
        404: {"description": "Session not found or expired."},
        422: {"description": "Wrong turn, illegal move, or game already over."},
    },
)
def submit_move(game_id: GameIdPath, req: SubmitMoveRequest, store: SessionStoreDep) -> GameStateResponse:
    """Apply *req.move* on behalf of the external player at *req.stone*.

    The runner enforces side-of-move and legality; mismatches surface
    as 422 with a structured ``detail.error`` code so clients can
    discriminate (``wrong_turn`` vs ``invalid_move`` vs
    ``game_already_over``).
    """
    session = _lookup(store, game_id)
    with session.lock:
        try:
            session.runner.submit_move(req.stone, req.move)
        except WrongTurnError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "error": "wrong_turn",
                    "expected": int(exc.expected),
                    "got": int(exc.got),
                },
            ) from exc
        except GameAlreadyOverError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"error": "game_already_over", "result": exc.result.model_dump(mode="json")},
            ) from exc
        except InvalidMoveError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "error": "invalid_move",
                    "move": exc.move.model_dump(mode="json"),
                    "reason": exc.reason,
                },
            ) from exc
    return _state(session)


@router.post(
    "/{game_id}/think",
    summary="Ask the brain on the current side to play its move",
    responses={
        404: {"description": "Session not found or expired."},
        422: {"description": "Current side is external, or game already over."},
        429: {"description": "Per-IP rate limit exceeded."},
    },
)
@compute_shared_limit
async def think(
    request: Request,
    game_id: GameIdPath,
    req: ThinkRequest,
    store: SessionStoreDep,
) -> ThinkResponse:
    """Run the brain configured for the side-to-move and apply its move.

    The brain runs on a worker thread (anyio) so a multi-second
    search doesn't block the event loop.  The session lock is released
    *only after* the apply completes — concurrent ``think`` /
    ``submit_move`` calls on the same session are serialised, but
    different sessions don't block each other.
    """
    session = _lookup(store, game_id)

    def _run_locked() -> ThinkResponse:
        with session.lock:
            stone = session.runner.current_stone
            try:
                move, _result, diag = session.runner.think(stone)
            except NoBrainConfiguredError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={"error": "external_side", "stone": int(exc.stone)},
                ) from exc
            except GameAlreadyOverError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={"error": "game_already_over", "result": exc.result.model_dump(mode="json")},
                ) from exc
            except InvalidMoveError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "brain_returned_illegal_move",
                        "move": exc.move.model_dump(mode="json"),
                        "reason": exc.reason,
                    },
                ) from exc
            return ThinkResponse(
                move=move,
                state=_state(session),
                diagnostics=diag if req.diagnostics else None,
            )

    return await anyio.to_thread.run_sync(_run_locked)
