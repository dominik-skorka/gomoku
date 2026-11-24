# gomoku-web

The browser game: React 19 + Vite 8 + TypeScript + Tailwind CSS 4. It keeps
the game state itself and asks the API (`POST /api/v1/move`) for engine
moves. Architecture and decisions: [docs/WEB_ARCHITECTURE.md](../docs/WEB_ARCHITECTURE.md).

```bash
pnpm install
pnpm dev          # http://localhost:5173, proxies /api to 127.0.0.1:8000
pnpm test         # unit tests (rules, game state)
pnpm check        # Biome lint + format check, TypeScript
pnpm build        # static files in dist/
```

| Path              | What                                                          |
| ----------------- | ------------------------------------------------------------- |
| `src/game/`       | Rules, game state reducer, engine loop, difficulty presets    |
| `src/api/`        | Typed API client; `schema.d.ts` is generated (`pnpm gen:api`) |
| `src/components/` | Board (SVG), dialogs, panels                                  |
| `openapi/v1.json` | API contract, exported by `scripts/export_openapi.py`         |

When the API changes, refresh the contract and the types:

```bash
uv run python scripts/export_openapi.py
```

```bash
pnpm gen:api
```
