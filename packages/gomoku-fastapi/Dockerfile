# syntax=docker/dockerfile:1
#
# Multi-stage build for ``gomoku-fastapi``.
#
# * **builder** — installs uv (from PyPI), lets it install the free-threaded
#   interpreter the project runs on (CPython 3.14t, ``.python-version``),
#   and syncs the locked workspace into ``/app/.venv``.  The uv cache
#   mount keeps wheel downloads across source-edit invalidations.
# * **runtime** — Debian slim with only the interpreter and the venv.  No
#   build tools, no uv.  Runs as a non-root user (uid 1000) so a host
#   volume mount inherits sensible ownership.  Numba kernels are compiled
#   at build time, so the first move is as fast as later ones.
#
# Build:   docker build -t gomoku-api -f packages/gomoku-fastapi/Dockerfile .
# Run:     docker run --rm -p 8000:8000 gomoku-api
# Compose: docker compose up   (from the repo root; see compose.yaml)
#
# Behind a TLS-intercepting proxy, pass its CA to every download step:
#   docker build --secret id=extra_ca,src=/path/to/ca.pem ...
#
# Brain plug-ins (docs/WRITING_A_BRAIN.md): BRAIN_PLUGINS lists pip
# requirement specs installed into the venv, separated by spaces.  Local
# packages come from the ``plugins`` build context, mounted at /plugins
# (empty unless given):
#   docker build --build-context plugins=examples \
#     --build-arg BRAIN_PLUGINS=/plugins/greedy-brain ...
#   docker build --build-arg BRAIN_PLUGINS="git+https://host/me/my-brain.git" ...
# The build fails if a plug-in does not load.

ARG UV_VERSION=0.11.32
ARG PYTHON_VERSION=3.14t

# Default ``plugins`` build context: empty.  ``--build-context plugins=DIR``
# (compose: ``additional_contexts``) replaces this stage with DIR.
FROM scratch AS plugins

FROM python:3.14-slim-trixie AS builder
ARG UV_VERSION
ARG PYTHON_VERSION

# uv from PyPI rather than ghcr.io/astral-sh/uv: PyPI is reachable from
# almost every network (the GHCR blob host is not always), and the pin
# keeps host and container uv in step (``uv --version``).
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install --no-cache-dir "uv==${UV_VERSION}"

# git, for brain plug-ins given as ``git+https://`` specs (BRAIN_PLUGINS
# below).  Builder stage only: the runtime image does not get it.  apt
# fetches over HTTPS (the image has ca-certificates), so the extra CA of a
# TLS-intercepting proxy applies to it as to the other downloads (readable
# by all: apt downloads as the unprivileged _apt user).
RUN --mount=type=secret,id=extra_ca,required=false,mode=0444 \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    ca=""; if [ -f /run/secrets/extra_ca ]; then ca="-o Acquire::https::CAInfo=/run/secrets/extra_ca"; fi; \
    apt-get $ca update \
    && apt-get $ca install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# UV_COMPILE_BYTECODE: .pyc files in the image, no compile cost at start.
# UV_LINK_MODE=copy: no hardlinks across the cache mount boundary.
# UV_PYTHON_INSTALL_DIR: the managed interpreter lands where the runtime
#   stage copies it from; the venv's symlinks point there.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_PREFERENCE=only-managed

RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv python install "${PYTHON_VERSION}"

WORKDIR /app

# Dependencies first (a cached layer that survives source edits), then the
# workspace members.  ``--package gomoku-fastapi`` installs only what that
# member reaches: gomoku + gomoku-brain, not the CLI.
COPY pyproject.toml uv.lock ./
COPY packages/gomoku/pyproject.toml packages/gomoku/pyproject.toml
COPY packages/gomoku-brain/pyproject.toml packages/gomoku-brain/pyproject.toml
COPY packages/gomoku-cli/pyproject.toml packages/gomoku-cli/pyproject.toml
COPY packages/gomoku-fastapi/pyproject.toml packages/gomoku-fastapi/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --no-install-workspace --package gomoku-fastapi
COPY packages/gomoku/ packages/gomoku/
COPY packages/gomoku-brain/ packages/gomoku-brain/
COPY packages/gomoku-fastapi/ packages/gomoku-fastapi/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --no-editable --package gomoku-fastapi

# Brain plug-ins, after the locked sync so changing them rebuilds only this
# layer.  ``uv pip install`` keeps what the sync installed (gomoku-brain is
# not on PyPI; the venv's copy satisfies a plug-in's dependency on it) and
# resolves a plug-in's other dependencies from the index.
ARG BRAIN_PLUGINS=""
RUN --mount=type=bind,from=plugins,target=/plugins \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then \
        export SSL_CERT_FILE=/run/secrets/extra_ca GIT_SSL_CAINFO=/run/secrets/extra_ca; \
    fi; \
    if [ -n "${BRAIN_PLUGINS}" ]; then \
        uv pip install --python /app/.venv/bin/python ${BRAIN_PLUGINS}; \
    fi


# ---------------------------------------------------------------------------
# Runtime: interpreter + venv, nothing else.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS runtime
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="gomoku-api" \
      org.opencontainers.image.source="https://github.com/dominik-skorka/gomoku" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

# Non-root user.  uid 1000 matches the typical first-user uid on the host
# so bind-mounted volumes don't end up root-owned.
RUN useradd --uid 1000 --user-group --create-home --shell /bin/bash app

COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=app:app /app/.venv /app/.venv
# Ship the repository's configuration so the API serves the same brains and
# aliases as the CLI.  Mount another file over /app/gomoku.toml (see
# compose.yaml) to change it without rebuilding.
COPY --chown=app:app gomoku.toml /app/gomoku.toml

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    GOMOKU_CONFIG=/app/gomoku.toml \
    GOMOKU_BUILD_VERSION=${VERSION} \
    GOMOKU_BUILD_REVISION=${REVISION}

USER app

# Compile the Numba kernels at build time instead of on the first request
# (about 5 s otherwise, paid again by every new container).  Running each
# brain once under the shipped config exercises the kernels the API uses,
# and Numba caches them next to the installed sources.
#
# Numba discards a cache whose recorded source timestamp differs from the
# file's current one, and image layers keep timestamps only to whole
# seconds, so the sources' sub-second timestamps are dropped *before*
# compiling; otherwise the finished image would invalidate every entry.
# On a CPU that differs from the build machine's, Numba recompiles on
# first use, as it would without this step.
RUN python - <<'PY'
import os
import sysconfig

site = sysconfig.get_paths()["purelib"]
for root, _dirs, files in os.walk(site):
    for name in files:
        if name.endswith(".py"):
            path = os.path.join(root, name)
            st = os.stat(path)
            os.utime(path, (int(st.st_atime), int(st.st_mtime)))

import warnings

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.settings import GomokuSettings
from gomoku_brain import BRAINS, PluginWarning, load_plugins, resolve_brain
from gomoku_brain.evaluate import evaluate_board

# A plug-in that does not load fails the build instead of a running server
# quietly missing a brain.
with warnings.catch_warnings():
    warnings.simplefilter("error", PluginWarning)
    loaded = load_plugins()
print("brain plug-ins:", ", ".join(loaded) or "none")

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

EXPOSE 8000

# Liveness probe.  A slim image has no curl, but Python is right there.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]

# uvicorn directly: the ``gomoku-api`` console script deliberately binds to
# 127.0.0.1 so dev runs aren't exposed to the LAN.
# * --http h11: httptools has not declared free-threading support, and
#   importing it would switch the GIL back on (see gomoku_fastapi.main.run).
# * --proxy-headers: behind the edge proxy, X-Forwarded-For becomes the
#   client address, so per-IP rate limits see real clients.  Trusted from
#   any address (FORWARDED_ALLOW_IPS) because in compose.yaml the API is
#   reachable through the proxy alone; compose.dev.yaml narrows it.
ENV FORWARDED_ALLOW_IPS="*"
CMD ["uvicorn", "gomoku_fastapi.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--http", "h11", "--proxy-headers", "--no-server-header"]
