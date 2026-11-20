"""``POST /v1/analyze`` — upload a ``.diag.jsonl`` and get an
:class:`AnalysisReport`.

Accepts the file as multipart form-data (``file=@recording.diag.jsonl``)
to keep curl / browser uploads ergonomic.  The body is decoded as
UTF-8 and handed straight to :func:`analyze_diagnostics_text`; we
never touch the filesystem.

The report's ``to_json()`` shape is the same dict that
``gomoku analyze --json`` prints, so a client can replace a local
file-based pipeline with this endpoint without changing downstream
``jq`` filters.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from gomoku.analyze import analyze_diagnostics_text

# A few MB easily covers a long game's worth of diagnostics; larger
# uploads probably indicate misuse and we'd rather 413 than wait.
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze",
    summary="Analyse an uploaded .diag.jsonl recording",
    responses={
        400: {"description": "File body could not be parsed as a diagnostics recording."},
        413: {"description": "Upload exceeds the size limit."},
    },
)
async def post_analyze(
    file: UploadFile = File(..., description=".diag.jsonl recording"),  # noqa: B008  (FastAPI idiom)
) -> dict:
    """Parse the upload and return an :class:`AnalysisReport`.

    The response body is the dict produced by :meth:`AnalysisReport.to_json`
    — same shape as ``gomoku analyze --json``.  We bypass FastAPI's
    automatic ``response_model`` so dataclass-keyed dicts (e.g. the
    ``Stone``-keyed ``player_stats``) serialise without ceremony.
    """
    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "upload_too_large",
                "limit_bytes": _MAX_UPLOAD_BYTES,
                "received_bytes": len(raw),
            },
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "not_utf8",
                "message": "diagnostics files must be UTF-8 encoded",
            },
        ) from exc

    try:
        report = analyze_diagnostics_text(text, source=file.filename or "<upload>")
    except ValueError as exc:
        # ``analyze_diagnostics_text`` raises ValueError for malformed
        # JSONL (missing header, bad move record, empty body, …).
        # Surface as 400 with the parser's message so the user can fix
        # their file rather than guess.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "malformed_diagnostics", "message": str(exc)},
        ) from exc

    return report.to_json()
