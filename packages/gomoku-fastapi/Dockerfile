# syntax=docker/dockerfile:1.7
#
# Multi-stage build for ``gomoku-fastapi``.
#
# * **builder** — installs uv, syncs the workspace, produces ``/app/.venv``.
#   The uv cache mount preserves wheel downloads across source-edit
#   invalidations, so a second build takes seconds even though the
#   ``uv sync`` layer technically re-runs.
# * **runtime** — ``python:slim`` with the prebuilt venv copied in.  No
#   build tools, no uv, no apt-get install.  Runs as a non-root user
#   (uid 1000) so a host volume mount inherits sensible ownership.
#
# Build:   docker build -t gomoku-api -f packages/gomoku-fastapi/Dockerfile .
# Run:     docker run --rm -p 8000:8000 gomoku-api
# Compose: docker compose up   (from the repo root)

ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Pull a static uv binary from Astral's image — saves a pip-install
# round-trip and pins the build-tool version explicitly.  Bump the tag
# in lockstep with the dev-env uv version (``uv --version``) to keep
# host and container behaviour aligned.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

# UV_COMPILE_BYTECODE: pre-compile .py to .pyc inside the image so the
#   first request doesn't pay startup compilation cost.
# UV_LINK_MODE=copy: avoid hardlinks across the cache mount boundary —
#   docker's overlayfs doesn't always preserve them and uv warns
#   loudly on a fallback.
# UV_PYTHON_DOWNLOADS=never: use the system python from the base image
#   rather than letting uv pull a managed interpreter (we already have one).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy the workspace and lockfile, then sync.  The ``--package
# gomoku-fastapi`` flag tells uv to install just the deps reachable
# from that one workspace member — gomoku + gomoku-brain come along
# automatically, but the CLI doesn't, which keeps the image lean.
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package gomoku-fastapi

# Ship the repository's configuration so the API serves the same brains
# and aliases as the CLI.  Mount another file over /app/gomoku.toml (see
# compose.yaml) to change it without rebuilding.
COPY gomoku.toml ./



# ---------------------------------------------------------------------------
# Runtime: minimal slim image with the prebuilt venv copied in.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Non-root user.  uid 1000 matches the typical first-user uid on the
# host so bind-mounted volumes don't end up root-owned.
RUN useradd --uid 1000 --user-group --create-home --shell /bin/bash app

WORKDIR /app

# Copy the venv + source from the builder.  Ownership is set so the
# unprivileged runtime user can read everything (including the venv,
# which is referenced by ``PATH`` below).
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# Compile the Numba kernels at build time instead of on the first
# request (about 5 s otherwise, paid again by every new container).
# Running each brain once under the shipped config exercises the kernels
# the API uses, and Numba caches them next to the sources.
#
# Numba discards a cache whose recorded source timestamp differs from the
# file's current one, and image layers keep timestamps only to whole
# seconds, so the sources' sub-second timestamps are dropped *before*
# compiling; otherwise the finished image would invalidate every entry.
# On a CPU that differs from the build machine's, Numba recompiles on
# first use, as it would without this step.
RUN python - <<'PY'
import os

for root, _dirs, files in os.walk("/app/packages"):
    for name in files:
        if name.endswith(".py"):
            path = os.path.join(root, name)
            st = os.stat(path)
            os.utime(path, (int(st.st_atime), int(st.st_mtime)))

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.settings import GomokuSettings
from gomoku_brain import BRAINS, resolve_brain
from gomoku_brain.evaluate import evaluate_board

settings = GomokuSettings()
board = Board(size=15)
stone = Stone.BLACK
for r, c in [(7, 7), (7, 8), (8, 7), (6, 8), (9, 6), (8, 8), (6, 6), (5, 5), (9, 9), (6, 9)]:
    board = board.place_stone(Move(row=r, col=c), stone)
    stone = stone.opponent()
evaluate_board(board, stone)
for name in ("threat", "alphabeta", "mcts"):
    brain_type, kwargs = resolve_brain(name, settings.brains)
    if brain_type == "mcts":
        kwargs["iterations"] = 200
    BRAINS[brain_type].brain_cls(**kwargs).get_move(board, stone)
PY

# Liveness probe — a slim image has no curl, but Python is right there.
# A 200 from /healthz means the FastAPI app booted and is serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# Bind to 0.0.0.0 so the published port reaches the app — the
# ``gomoku-api`` console script binds to 127.0.0.1 by default
# (intentionally, so dev runs aren't exposed to the LAN), so we
# invoke uvicorn directly with the right host here.
CMD ["python", "-m", "uvicorn", "gomoku_fastapi.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
