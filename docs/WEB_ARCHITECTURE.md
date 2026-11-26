# Web architecture

How the Gomoku website is built, run and operated: the browser game in
`web/`, the REST API in `packages/gomoku-fastapi`, and the containers in
`deploy/` and `compose.yaml`.  Work on the website is tracked in
[issue #82](https://github.com/dominik-skorka/gomoku/issues/82).

## Goals

| Goal | How it is met |
|---|---|
| Runs on a laptop and on any host | One `compose.yaml`; the same images everywhere |
| Modular, each part replaceable | Three images (`api`, `web`, `edge`), each rebuilt, upgraded and restarted alone |
| Survives failures | Stateless API replicas; restart policies + healthchecks + autoheal; the proxy retries another replica |
| Scales | `docker compose up --scale api=N`; the edge balances over every replica |
| Modern look on phone, tablet and laptop | React 19 SPA, responsive layout, SVG board, light/dark theme |
| Versioned | SemVer per component, `/version.json` (web) and `/v1` + OpenAPI (API) |
| Observable | `/healthz` per container, request IDs end to end, Logfire traces (see [DIAGNOSTICS](DIAGNOSTICS.md)) |

## Components

```mermaid
flowchart LR
  subgraph Browser
    UI["React SPA<br/>game state + rules"]
  end
  subgraph Host["docker compose"]
    E["edge<br/>Caddy: TLS, routing,<br/>load balancing"]
    W["web<br/>Caddy: static files"]
    A1["api #1<br/>FastAPI + brains"]
    A2["api #N"]
    H["autoheal"]
  end
  UI -- "HTTPS /" --> E
  UI -- "HTTPS /api/v1/move" --> E
  E -- "/*" --> W
  E -- "/api/* (prefix stripped)" --> A1
  E -.-> A2
  H -. "restart if unhealthy" .-> A1
  H -.-> W
```

| Component | Image | Technology | Responsibility |
|---|---|---|---|
| `web` | `gomoku-web` | React 19, Vite 8, TypeScript 6, Tailwind CSS 4, Motion | The game UI; built once into static files served by Caddy |
| `api` | `gomoku-api` | Python 3.14 free-threaded, FastAPI, uvicorn (`packages/gomoku-fastapi/Dockerfile`) | Asks a brain for a move (`POST /v1/move`); stateless; Numba kernels precompiled in the image |
| `edge` | `caddy` | Caddy 2.11 | The only published port: TLS (automatic certificates), `/api` routing, load balancing, security headers |
| `autoheal` | `willfarrell/autoheal` | — | Restarts containers whose healthcheck turns `unhealthy` |

## A move, end to end

```mermaid
sequenceDiagram
  participant U as Player
  participant B as Browser (React)
  participant E as edge (Caddy)
  participant A as api replica
  U->>B: taps a point
  B->>B: place stone, check five-in-a-row
  B->>E: POST /api/v1/move {board, side_to_move, brain}
  E->>A: POST /v1/move (least-connections replica)
  A->>A: brain.get_move() in a worker thread
  A-->>E: 200 {move} + X-Request-ID, API-Version
  E-->>B: 200
  B->>B: place engine stone, check for a win
```

## Decisions

### ADR-1: Game state lives in the browser

**Decision.** The browser holds the board, applies the rules (`web/src/game/`)
and sends the whole position to the stateless `POST /v1/move`.  The
stateful `/v1/games` sessions stay in the API for other clients but the
website does not use them.

**Why.** Sessions are kept in one process's memory: a restart loses every
game and a second replica cannot see the first one's games (sticky sessions
or Redis would be needed).  Stateless calls make every replica equivalent,
so replicas can be added, killed or upgraded mid-game without a player
noticing.  The cost is a second copy of the win rule (about 30 lines of
TypeScript, unit-tested against the same cases as `gomoku.rules`).

**Revisit when** accounts, saved games or multiplayer arrive: then
sessions move to a shared store (Redis/Valkey) behind the existing API.

### ADR-1a: Each colour is a human or an engine

**Decision.** Game settings name a player kind per colour (`black`,
`white`: `human` or `engine` at a difficulty preset).  Human vs engine,
human vs human (one device, taking turns) and engine vs engine all follow
from that one model; the engine loop simply asks `POST /v1/move` whenever
an engine plays the side to move.

**Engine vs engine** plays itself, with pause, single step and a delay
between moves (none, 0.5 s, 1.5 s) so a game can be followed.  When the
API rate-limits a fast game (`429`), the client waits for `Retry-After`
and retries by itself.

**Resign** ends the game for the human whose turn it is (in human vs
human: the side to move), after a confirmation, and cannot be undone.
**Move navigation** (⏮ first move, ◀ ▶, ⏭ latest; under the board on phones and tablets, above the move list on desktop; Home / ← / → / End, or a click in the move
list) only *shows* earlier positions: the game, an engine thinking in the
background included, goes on, and ⏭ catches up with it.  Playing from an
earlier position is what Undo is for.

**Player bars** frame the board chess-style: White above, Black below,
each with an avatar frame (a placeholder until avatars exist), the
player's name, what they are doing (thinking, paused, an API error with
Retry, the result) and a clock.  The clock counts the time a player has
spent in the game, chess-style: a human's from the start of each turn,
an engine's only while its request is in flight, so pauses and errors do
not count.  Undo takes the time of the undone moves back.

### ADR-7: The web UI owns its configuration; the API stays generic

**Decision.** What a difficulty level is, which engine and parameters it
uses, and every text the page shows belong to the web UI, not to the API
or `gomoku.toml`.  They live in files the web container serves and the page
reads at start-up:

| File (repository) | Served at | Holds |
|---|---|---|
| `web/public/config/config.json` | `/config/config.json` | `apiBase`, `defaultLevel`, `levels[]`: `id`, `brain`, `overrides`, `typicalSeconds` |
| `web/public/locales/en.json`, `pl.json` | `/locales/<lang>.json` | All UI texts; level names under `levels.<id>` |

The API only receives what it always accepted: a `brain` name and
`brain_overrides`.  The level editor (gear icon) builds its form from the
engines' settings schemas (`GET /v1/brains`), so a new engine parameter
appears there without a web change.  Each field shows the value a move
uses: the level's override, or what the server plays with
(`GET /v1/brains/{name}/settings`: the `gomoku.toml` alias, the brain's
defaults and the server's time cap; generic engine information, no
client concepts).  Values that differ from the server's (the defaults) are
highlighted, with the default beside them and a button back to it; a
value equal to the default is not stored as an override.  Explanations
from the schema sit in (i) tooltips (hover, focus, or tap on touch).

Nested settings (e.g. `threats.own.split_fours`, the `fallback` alpha-beta
tuning of the threat brain) are collapsible groups.  The API merges
`brain_overrides` one level deep, so a change inside a group sends the
whole top-level group: the server's values with the change applied
(sending only the changed key would reset the group's other values to
their schema defaults).  A group back at the server's values is no
longer overridden.

**Why.** The engine and its API serve any client; client concepts (levels,
their names in two languages) would make them specific to this one.
Reading the files at run time means changing a level or a text needs an
edit and a browser refresh, not a rebuild.

**Layers.**  Levels in use = `config.json` levels, with this browser's
changes from the level editor on top (localStorage).  "Restore" drops the
changes; "Export" downloads the levels in use as a `config.json`, to make
them everyone's default.  Settings shared across devices wait for user
accounts.

**Safety nets.**  A missing or invalid `config.json` falls back to the copy
built into the bundle (the console says what is wrong); a locale file with
missing keys is filled from the bundled English; a level without a
translation shows its id.  Parameters without an override are not sent,
so the server's own settings apply (they may differ from the schema
defaults, e.g. `vcf_depth` in `gomoku.toml`, which is why the editor asks
the server rather than showing schema defaults).  Settings that name server
files (`opening_book`) are never offered.

### ADR-2: A static SPA, not a server-rendered framework

**Decision.** React + Vite, built to static files.  No Next.js/SSR.

**Why.** A game behind a board has nothing to render on the server and no
SEO beyond one page.  Static files need no Node process in production,
cache perfectly (hashed bundles are immutable), and roll back by switching
an image tag.  React was chosen over Svelte for the size of its ecosystem.

### ADR-3: Same origin through an edge proxy

**Decision.** Caddy serves the site and proxies `/api/*` to the API on the
same origin; uvicorn runs with `--root-path /api`.

**Why.** No CORS configuration, one certificate, one place for security
headers.  Caddy obtains and renews TLS certificates by itself, so moving
from the laptop (`SITE_ADDRESS=:80`) to a domain is one variable.

### ADR-4: TypeScript types generated from OpenAPI

**Decision.** `web/openapi/v1.json` is exported from the running app code
(`scripts/export_openapi.py`), and `web/src/api/schema.d.ts` is generated
from it.  CI fails if either is stale.

**Why.** An incompatible API change then breaks the web build in the same
pull request, never the game in production.  The API's own contract test
(`tests/test_api_contract.py`) already forbids breaking changes inside `v1`.

### ADR-5: Docker Compose, not Kubernetes

**Decision.** One Compose file for the laptop and for a single host.

**Why.** Three services do not justify a cluster's operating cost.  The
images are plain OCI images, so moving to Kubernetes, Fly.io or Cloud Run
later changes deployment files, not the application.

**Caveat.** Docker restarts a container when its process exits, not when
its healthcheck fails.  The `autoheal` container closes that gap; it needs
read access to the Docker socket, which is acceptable on a single-purpose
host.

### ADR-6: Free-threaded Python in the API image

**Decision.** The image runs CPython 3.14t (no GIL) installed by `uv`, and
uvicorn's pure-Python `h11` parser.

**Why.** It is the project's interpreter (`.python-version`): brains that
probe in parallel really run in parallel.  `httptools` has not declared
free-threading support, and importing it would turn the GIL back on.

## Versioning

| Component | Version source | Where to see it |
|---|---|---|
| Web | `web/package.json` + git revision | Page footer, `GET /version.json` |
| API (server) | `packages/gomoku-fastapi/pyproject.toml` | `info.version` in `GET /api/openapi.json` |
| API (contract) | URL major version `v1` (AIP-185) | `API-Version` response header, `GET /api/versions` |
| Images | Build args `VERSION`, `REVISION` | OCI labels `org.opencontainers.image.version` / `.revision` |

## Running it

```bash
docker compose up --build
```

Then open <http://localhost:8080>.  The API's interactive docs are at
<http://localhost:8080/api/docs>.

To serve your own brain plug-ins (see
[WRITING_A_BRAIN.md](WRITING_A_BRAIN.md#on-the-website)), name them when
building:

```bash
GOMOKU_BRAIN_PLUGINS=/plugins/greedy-brain docker compose up --build
```

For frontend development with hot reload, run the API (from source or in
its container) and Vite separately:

```bash
uv run gomoku-api
```

or

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build api
```

```bash
cd web && pnpm install && pnpm dev
```

Vite proxies `/api` to `127.0.0.1:8000` exactly as Caddy does in production.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HTTP_PORT` | `8080` | Port published on the host |
| `SITE_ADDRESS` | `:80` | Caddy site address; a domain name enables automatic HTTPS |
| `GOMOKU_RATE_LIMIT_DEFAULT` | `240/minute` | Per-IP limit on every API endpoint, per replica |
| `GOMOKU_RATE_LIMIT_COMPUTE` | `60/minute` | Per-IP limit on `/v1/move`, per replica |
| `GOMOKU_MAX_TIME_LIMIT_S` | `10` | Upper bound on a brain's think time per request |
| `LOGFIRE_TOKEN` | empty | Ship traces to Logfire when set |
| `GOMOKU_VERSION`, `GOMOKU_WEB_VERSION` | `dev` | Image tags to run |

The web UI's own settings (API address, difficulty levels, texts) are in
`web/public/config/config.json` and `web/public/locales/`, mounted into the
web container (see ADR-7): edit, then refresh the browser.  The directories
are mounted, not single files, because a single-file bind mount keeps the
old content after an editor replaces the file.

Rate limits are kept in each replica's memory, so with N replicas a client
can get up to N times the limit.  That is fine for abuse protection; a
shared limit would need Redis.

### Operating

| Task | Command |
|---|---|
| Status of every container | `docker compose ps` |
| Logs of one component | `docker compose logs -f api` |
| Restart one component | `docker compose restart api` |
| Upgrade only the web UI | `docker compose up -d --build web` |
| Run three engine replicas | `docker compose up -d --scale api=3` |

### Building behind a TLS-intercepting proxy

Both Dockerfiles accept an optional extra CA certificate as a build secret:

```bash
docker build --secret id=extra_ca,src=/path/to/ca.pem -f packages/gomoku-fastapi/Dockerfile .
```
