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
# {"move":{"row":4,"col":4,"position":"E5"},"brain":"alphabeta"}
```

Interactive OpenAPI docs are live at `http://127.0.0.1:8000/docs`.

## Endpoints (iteration 1)

| Path | Verb | Purpose |
|---|---|---|
| `/healthz` | GET | Liveness probe |
| `/v1/brains` | GET | Brain registry + per-brain settings JSON Schema |
| `/v1/move` | POST | Stateless: ask a brain for its move on a given board |

The plan from the architecture-review thread schedules sessions,
diagnostics, and analyze in iterations 2-4.

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
                       └─ anyio.to_thread.run_sync(brain.get_move)
                              ↑ keeps the event loop responsive while
                                a multi-second alpha-beta search runs
```

The CPU-bound `brain.get_move` runs on the default anyio thread pool
so that one slow request can't starve other concurrent ones.
Diagnostics `ContextVar`s propagate across the thread-pool boundary
correctly (locked in by tests in iteration 2).
