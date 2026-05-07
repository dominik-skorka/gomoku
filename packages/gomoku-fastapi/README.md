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
