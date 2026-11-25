# gomoku-web

The browser game: React 19 + Vite 8 + TypeScript + Tailwind CSS 4. It keeps
the game state itself and asks the API (`POST /api/v1/move`) for engine
moves. Architecture and decisions: [docs/WEB_ARCHITECTURE.md](../docs/WEB_ARCHITECTURE.md).

```bash
pnpm install
pnpm dev          # http://localhost:5173, proxies /api to 127.0.0.1:8000
pnpm test         # unit tests (rules, game state, config, texts, levels)
pnpm test:e2e     # browser tests (Playwright, stubbed API; desktop + phone)
pnpm check        # Biome lint + format check, TypeScript
pnpm build        # static files in dist/
```

| Path                        | What                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `src/game/`                 | Rules, game state reducer, engine loop, difficulty levels (config + this browser's changes) |
| `src/lib/`                  | Runtime config (`config.ts`), texts (`i18n.ts`), status line, theme                         |
| `public/config/config.json` | API address and difficulty levels (engine + parameters), read at start-up                   |
| `public/locales/*.json`     | UI texts per language, read at start-up                                                     |
| `src/api/`                  | Typed API client; `schema.d.ts` is generated (`pnpm gen:api`)                               |
| `src/components/`           | Board (SVG), dialogs, panels                                                                |
| `openapi/v1.json`           | API contract, exported by `scripts/export_openapi.py`                                       |

`public/config/config.json` and `public/locales/*.json` are served with the
page and read when it starts, so a deployment can change them without a
rebuild (Docker mounts them, see `compose.yaml`). The copies in the bundle
are the fallback. Levels can also be tuned per browser from the gear icon;
"Export" turns those changes into a `config.json`.

When the API changes, refresh the contract and the types:

```bash
uv run python scripts/export_openapi.py
```

```bash
pnpm gen:api
```
