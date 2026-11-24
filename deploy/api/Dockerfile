# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# gomoku-api: the FastAPI service (gomoku core + gomoku-brain + gomoku-fastapi).
# Build context: repository root (needs uv.lock and every workspace package).
#
#   docker build -f deploy/api/Dockerfile -t gomoku-api .
# -----------------------------------------------------------------------------
ARG UV_VERSION=0.11.32
ARG PYTHON_VERSION=3.14t

# ---- build: resolve the locked environment into /app/.venv -----------------
# uv comes from PyPI (reachable from almost any network); it then installs the
# free-threaded interpreter (python-build-standalone) into /opt/python.
FROM python:3.14-slim-trixie AS build
ARG UV_VERSION
ARG PYTHON_VERSION
# Optional extra CA for a TLS-intercepting proxy, used by every download step:
#   docker build --secret id=extra_ca,src=/path/to/ca.pem ...
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install --no-cache-dir "uv==${UV_VERSION}"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_PREFERENCE=only-managed
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv python install ${PYTHON_VERSION}
WORKDIR /app
# Dependencies first (cached layer), then the workspace sources.
COPY pyproject.toml uv.lock ./
COPY packages/gomoku/pyproject.toml packages/gomoku/pyproject.toml
COPY packages/gomoku-brain/pyproject.toml packages/gomoku-brain/pyproject.toml
COPY packages/gomoku-cli/pyproject.toml packages/gomoku-cli/pyproject.toml
COPY packages/gomoku-fastapi/pyproject.toml packages/gomoku-fastapi/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --no-install-workspace --package gomoku-fastapi
COPY packages/gomoku/ packages/gomoku/
COPY packages/gomoku-brain/ packages/gomoku-brain/
COPY packages/gomoku-fastapi/ packages/gomoku-fastapi/
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --no-editable --package gomoku-fastapi

# ---- runtime: interpreter + venv, nothing else ------------------------------
FROM debian:trixie-slim AS runtime
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="gomoku-api" \
      org.opencontainers.image.source="https://github.com/dominik-skorka/gomoku" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
RUN useradd --system --uid 10001 --home-dir /app gomoku
COPY --from=build /opt/python /opt/python
COPY --from=build --chown=gomoku:gomoku /app/.venv /app/.venv
COPY --chown=gomoku:gomoku gomoku.toml /app/gomoku.toml
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    GOMOKU_CONFIG=/app/gomoku.toml \
    GOMOKU_BUILD_VERSION=${VERSION} \
    GOMOKU_BUILD_REVISION=${REVISION} \
    NUMBA_CACHE_DIR=/tmp/numba-cache
USER gomoku
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]
# h11: httptools has not declared free-threading support, importing it would
# re-enable the GIL (see gomoku_fastapi.main.run).  --proxy-headers: the edge
# proxy's X-Forwarded-For becomes the client address, so per-IP rate limits
# see real clients instead of the proxy.
CMD ["uvicorn", "gomoku_fastapi.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--http", "h11", "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]
