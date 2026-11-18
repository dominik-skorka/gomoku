# gomoku-fastapi

REST API exposing the Gomoku engine over HTTP.  Built with FastAPI;
reuses the pydantic models and brain registry from `gomoku` and
`gomoku-brain` directly — no parallel DTO layer.

## Quick start

```bash
uv sync
uv run gomoku-api          # binds 127.0.0.1:8000

# Liveness
curl http://127.0.0.1:8000/healthz
# {"status":"ok"}
```

### Docker

```bash
# From the repo root.  Multi-stage build, slim runtime, non-root user.
docker compose up --build

# Or without Compose:
docker build -t gomoku-api -f packages/gomoku-fastapi/Dockerfile .
docker run --rm -p 8000:8000 gomoku-api

# All [api] settings are configurable via env vars (see compose.yaml).
docker run --rm -p 8000:8000 \
  -e GOMOKU_API__RATE_LIMIT_COMPUTE="60/minute" \
  -e GOMOKU_API__CORS_ORIGINS='["http://localhost:5173"]' \
  -e LOGFIRE_TOKEN="$LOGFIRE_TOKEN" \
  gomoku-api
```

The image runs `uvicorn` directly (bypassing the `gomoku-api` console
script, which deliberately binds to localhost).  Built-in
`HEALTHCHECK` polls `/healthz` every 30s — `docker ps` shows the
state.

The image ships the repository's `gomoku.toml`, so the API serves the
same brains and aliases as the CLI; mount an edited copy over
`/app/gomoku.toml` to change them without rebuilding.  Numba kernels are
compiled while the image is built, so the first move request is as fast
as later ones (on a CPU that differs from the build machine's, Numba
recompiles once on first use).

### API examples

```bash
# List brains
curl http://127.0.0.1:8000/v1/brains | jq '.brains[].name'

# Get a move
curl -X POST http://127.0.0.1:8000/v1/move \
  -H "Content-Type: application/json" \
  -d '{"board": {"size": 9}, "side_to_move": 1, "brain": "alphabeta",
       "brain_overrides": {"depth": 2}}'
# {"move":{"row":4,"col":4,"position":"E5"},"brain":"alphabeta",
#  "diagnostics":null}

# Get a move + per-move diagnostics
curl -X POST http://127.0.0.1:8000/v1/move \
  -H "Content-Type: application/json" \
  -d '{"board": {"size": 9}, "side_to_move": 1, "brain": "alphabeta",
       "diagnostics": true}'

# Static heuristic eval
curl -X POST http://127.0.0.1:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"board": {"size": 9}, "side_to_move": 1}'

# Analyse a recording (multipart upload)
curl -X POST http://127.0.0.1:8000/v1/analyze \
  -F "file=@recordings/gomoku_*.diag.jsonl"

# Stateful session: human (Black) vs alphabeta (White)
GAME=$(curl -s -X POST http://127.0.0.1:8000/v1/games \
  -H "Content-Type: application/json" \
  -d '{"board_size": 9, "white": {"brain": "alphabeta",
       "brain_overrides": {"depth": 2}}}' | jq -r .game_id)

# Black plays the centre
curl -X POST http://127.0.0.1:8000/v1/games/$GAME/moves \
  -H "Content-Type: application/json" \
  -d '{"stone": 1, "move": {"row": 4, "col": 4}}'

# White's brain replies
curl -X POST http://127.0.0.1:8000/v1/games/$GAME/think \
  -H "Content-Type: application/json" -d '{}'

# Tear it down
curl -X DELETE http://127.0.0.1:8000/v1/games/$GAME
```

Interactive OpenAPI docs are live at `http://127.0.0.1:8000/docs`.

## Endpoints

| Path | Verb | Purpose |
|---|---|---|
| `/healthz` | GET | Liveness probe |
| `/versions` | GET | Served API versions and their status (see [Versioning](#versioning)) |
| `/v1` | GET | This version's status and dates |
| `/v1/openapi.json` | GET | OpenAPI document of `v1` only |
| `/v1/brains` | GET | Brain registry + per-brain settings JSON Schema |
| `/v1/move` | POST | Stateless: ask a brain for its move on a given board (optional `diagnostics: true`) |
| `/v1/evaluate` | POST | Static heuristic eval — no brain |
| `/v1/analyze` | POST | Upload a `.diag.jsonl`; returns the same report shape as `gomoku analyze --json` |
| `/v1/games` | POST | Create a session (each side may be brain-driven or external) |
| `/v1/games/{id}` | GET | Read session state (board, history, result) |
| `/v1/games/{id}` | DELETE | Drop a session (idempotent — 204 even if unknown) |
| `/v1/games/{id}/moves` | POST | Submit an externally-played move |
| `/v1/games/{id}/think` | POST | Run the configured brain on the side-to-move |

Sessions are in-process and TTL-evicted (default: 1 hour of
inactivity).  No persistence — restarting the server drops everything.

## Versioning

The API follows the URL-path scheme of
[Google AIP-185](https://google.aip.dev/185): the major version is the
first path segment (`/v1`), and minor versions never appear in URLs.
Every endpoint except `/healthz` and `/versions` lives under a version.

### What clients can rely on

A version only changes in backwards-compatible ways
([AIP-180](https://google.aip.dev/180)):

- **Compatible:** new endpoints, new optional request fields, new
  response fields and new enum values.  Clients must ignore response
  fields they don't recognise and tolerate enum values they don't know.
- **Breaking, so only ever in a new version:** removing or renaming an
  endpoint, parameter, field or enum value; making a parameter, request
  body or request field required; changing a type, a default value or a
  validation constraint; changing what a field means.

### Discovering versions

| Path | What it returns |
|---|---|
| `GET /versions` | Every served version with its status and dates, and the latest |
| `GET /v1` | The same description for `v1` alone (the version's home resource) |
| `GET /v1/openapi.json` | An OpenAPI document of `v1` only; `/openapi.json` covers everything |

```json
{
  "latest": "v1",
  "versions": [
    {"version": "v1", "path": "/v1", "openapi": "/v1/openapi.json", "status": "stable",
     "released": "2026-05-01", "deprecated": null, "sunset": null, "successor": null}
  ]
}
```

### Lifecycle and headers

Every response under a version prefix, errors and `429`s included,
carries `API-Version: v1`.  Retiring a version takes two steps:

1. **Deprecation**, at least 180 days before the sunset (AIP-185's
   recommended period; the registry rejects anything shorter).  The
   version's responses then carry:

   ```http
   Deprecation: @1780272000
   Sunset: Thu, 31 Dec 2026 00:00:00 GMT
   Link: </versions>; rel="deprecation", </versions>; rel="sunset", </v2>; rel="successor-version"
   ```

   `Deprecation` ([RFC 9745](https://www.rfc-editor.org/rfc/rfc9745)) may
   name a future date to announce an upcoming deprecation; the version
   stays `stable` in `/versions` until that date.  `Sunset`
   ([RFC 8594](https://www.rfc-editor.org/rfc/rfc8594)) is never earlier
   than `Deprecation`.  These headers describe the **whole version**, not
   only the resource that returned them.  A deprecated version keeps
   working unchanged until its sunset.
2. **Retirement**, after the sunset.  Requests to the version get
   `410 Gone` with a body naming the successor.

A request to a version that was never served (say `/v2/...` today) gets
a `404` listing the supported versions.  When CORS is enabled, all of
these headers, and `X-Request-ID`, are exposed to browser scripts.

### Making a breaking change

Register the new version in `gomoku_fastapi/versioning.py` and add an
`api_v2.py` beside `api_v1.py` that includes the changed routers plus
every unchanged one, so both versions are served side by side.  To
retire the old one, set its `deprecated`, `sunset` and `successor`
dates, and after the sunset move it to `RETIRED`.  The registry is
validated at import: names must be `v<number>` in ascending order, a
deprecated version needs a sunset at least 180 days later and a newer
successor, and the latest version can't be deprecated.

### The contract guard

`tests/contracts/v1.openapi.json` is a snapshot of `/v1/openapi.json`.
`tests/test_api_contract.py` fails on any breaking change to `v1` (the
AIP-180 rules above, distinguishing request schemas from response
schemas), and on compatible changes until the snapshot is refreshed, so
every change to the public surface shows up in review:

```bash
uv run python packages/gomoku-fastapi/tests/test_api_contract.py --update
```

The OpenAPI documents' `info.version` is the `gomoku-fastapi` package
version from its `pyproject.toml`; it tracks releases of the server, not
the URL version.

## Operations

| Concern | How it's handled |
|---|---|
| Structured logs | [logfire](https://logfire.pydantic.dev/) — auto-instruments FastAPI requests + spans the brain calls already made via `logfire.span` in `gomoku.game`.  Set `LOGFIRE_TOKEN` to ship to logfire.dev; otherwise records stay local. |
| Request correlation | `X-Request-ID` middleware — echoes the client's ID or mints a UUID, and exposes it via `gomoku_fastapi.observability.get_request_id()` for downstream code (incl. work dispatched to anyio threads). |
| CORS | Opt-in via `settings.api.cors_origins`.  Empty list = no middleware mounted (server-to-server only). |
| Rate limiting | [slowapi](https://github.com/laurentS/slowapi), in-memory backend.  Two tiers: a default per-IP limit on every endpoint and a tighter shared bucket on the compute-heavy `/v1/move` + `/v1/games/{id}/think`.  Set the limit strings to `""` to disable a tier. |
| Graceful shutdown | FastAPI lifespan handler clears the session store on shutdown so a restart doesn't carry stale state forward. |

Configure via `[api]` in `gomoku.toml`:

```toml
[api]
cors_origins = ["http://localhost:5173"]
rate_limit_default = "120/minute"
rate_limit_compute = "30/minute"
logfire_send = false
logfire_console = true
```

All knobs are also overridable via the `GOMOKU_API__<NAME>` env var
(e.g. `GOMOKU_API__RATE_LIMIT_COMPUTE="10/minute"`).

## Configuration

Re-uses `gomoku.toml` via the same `find_config_file` walk-up the CLI
uses — drop a `[brains.<alias>]` section and it'll be visible in
`POST /v1/move` as a `"brain": "<alias>"` value.  Set `GOMOKU_CONFIG`
to point at an explicit file.

## Architecture

```
client ── HTTP ──▶ FastAPI router
                       │
                       ├─ build_brain(name, overrides, settings)
                       │     └─ resolve_brain (gomoku-brain registry)
                       │
                       ├─ SessionStore (in-memory, TTL-evicted)
                       │     └─ Session { GameRunner, lock, recorder, ... }
                       │
                       └─ anyio.to_thread.run_sync(brain.get_move)
                              ↑ keeps the event loop responsive while
                                a multi-second alpha-beta search runs
```

The CPU-bound `brain.get_move` runs on the default anyio thread pool
so that one slow request can't starve other concurrent ones.
Diagnostics `ContextVar`s propagate across the thread-pool boundary
correctly (locked in by tests in iteration 2).

`GameRunner` (added in iteration 3, in `gomoku/runner.py`) generalises
`Game`'s loop so each side may be brain-driven *or* external.  The
`/v1/games/{id}/moves` endpoint advances the external side; `/think`
dispatches the configured brain.  Per-session locking keeps concurrent
calls on the same session serialised without blocking different
sessions.

Iteration 4 wraps the request pipeline with logfire instrumentation, a
slowapi limiter, optional CORS, and a request-ID middleware — all
configured from `[api]` in `gomoku.toml` so an operator can tune
without a code change.
