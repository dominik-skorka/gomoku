# Gomoku

A **Gomoku (Five in a Row)** engine with multiple AI brains, a terminal
interface, game recording, diagnostics, and opening book support.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Play as Black (keyboard) against the strongest AI (threat brain)
uv run gomoku play --black keyboard --white threat

# Watch two AIs play each other
uv run gomoku play --black threat --white alphabeta --record

# Replay a recorded game
uv run gomoku playback recordings/gomoku_20260405_143012.json
```

---

## Features

| Feature | Description |
|---------|-------------|
| **6 AI brains** | Random, minimax, negamax, alpha-beta (8 enhancements), threat (VCF/VCT/PNS), MCTS |
| **Human play** | Keyboard input via terminal |
| **Game recording** | Incremental JSON with MD5 checksums |
| **Diagnostics** | Per-move brain internals (`.diag.jsonl`) + `gomoku analyze` blunder/time-spike reporter |
| **Playback overlay** | Arrow-key replay overlays diagnostics when a companion file exists |
| **Opening book** | Hardcoded + file-based with 8-fold symmetry |
| **Configuration** | TOML + environment variables + CLI flags |
| **Observability** | Pydantic Logfire instrumentation |
| **JIT acceleration** | Optional Numba-compiled evaluation and bitboard |

---

## Architecture

The project is a **uv workspace** with three packages:

```
gomoku-cli  ──▶  gomoku-brain  ──▶  gomoku (core)
   CLI              AI brains         Board, rules, models
```

- **gomoku** — Core engine: board, rules, game loop, recording, configuration.
- **gomoku-brain** — AI move providers: threat search, alpha-beta, MCTS, evaluation.
- **gomoku-cli** — Terminal interface: game display, keyboard input, playback.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture
description.

---

## AI Brains

| Brain | Algorithm | Strength | Best at |
|-------|-----------|----------|---------|
| `random` | Uniform random | Weakest | Testing |
| `minimax` | Classic minimax | Weak | Educational |
| `negamax` | Negamax | Weak–medium | Educational |
| `alphabeta` | Negamax + alpha-beta + 8 enhancements | Strong | Positional play |
| `threat` | VCF + PNS/VCT + alpha-beta fallback | **Strongest** | Tactical endgame |
| `mcts` | PUCT + heuristic priors | Medium | Exploratory play |

The **threat brain** is the strongest — it chains forcing moves (threats the
opponent must answer) up to 20 plies deep, then falls back to alpha-beta for
non-tactical positions.

The **alpha-beta brain** uses Zobrist transposition tables, iterative
deepening, principal variation search, quiescence search, aspiration windows,
futility pruning, late move reduction, null-move pruning, and threat
extensions.

---

## CLI Commands

### `play` — Start a game

```bash
uv run gomoku play [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--black NAME` | `keyboard` | Black player (brain name or alias) |
| `--white NAME` | `random` | White player (brain name or alias) |
| `--board-size N` | from config | Board size (5–25) |
| `--config PATH` | `gomoku.toml` | Configuration file |
| `--record / --no-record` | from config | Enable game recording |
| `--record-dir PATH` | from config | Recording directory |
| `--diagnostics / --no-diagnostics` | from config | Enable per-move diagnostics |
| `--diagnostics-dir PATH` | from config | Diagnostics directory |
| `--opening-book PATH` | — | Opening book JSON file |

### `playback` — Replay a recorded game

```bash
uv run gomoku playback <file.json> [--from-move N]
```

Navigate with `←` previous, `→` next, `q` quit.  If a companion
`<file>.diag.jsonl` file sits next to the recording (default when playing
with `--record --diagnostics`), per-move brain internals are overlaid under
each position automatically.

### `analyze` — Post-game diagnostics review

```bash
uv run gomoku analyze <file.diag.jsonl> [--json]
```

Prints per-player stats (moves, think time, avg/max depth, total nodes,
blunder count) and a table of suspect moves flagged by five heuristics
(`eval_drop`, `bypassed_top_candidate`, `time_spike`, `opp_double_four`,
`opp_forced_win`).  `--json` emits a single machine-readable document.
See [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md#post-game-analysis-gomoku-analyze)
for details.

### `generate-book` — Build a file-based opening book

```bash
uv run gomoku generate-book [OPTIONS]
```

Runs a configurable brain on the early-game search tree and saves an
opening book JSON file.  See
[docs/OPENING_BOOK.md](docs/OPENING_BOOK.md#generating-a-book).

---

## Configuration

Runtime settings are managed via `gomoku.toml` with pydantic-settings.

**Priority** (highest → lowest):
1. CLI flags
2. Environment variables (`GOMOKU_*` prefix)
3. `gomoku.toml`
4. Built-in defaults

Brain aliases allow multiple configurations of the same engine:

```toml
[brains.threat]
type = "threat"
vcf_depth = 20
depth = 4

[brains.alphabeta_deep]
type = "alphabeta"
depth = 6
lmr = true
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Workspace structure, packages, data models, configuration system |
| [docs/THREATS.md](docs/THREATS.md) | Threat classification, pattern scoring, fork detection, defensive strategies |
| [docs/OPENING_BOOK.md](docs/OPENING_BOOK.md) | 8-fold symmetry, hardcoded and file-based books, generation |
| [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md) | Per-move diagnostics: format, context API, reading and analysing output |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Manual for analysing games — blunders, mistakes, inefficiencies, root causes, fixes |
| [packages/gomoku-brain/docs/THREAT_BRAIN.md](packages/gomoku-brain/docs/THREAT_BRAIN.md) | VCF, VCT, PNS algorithms with pseudocode and examples |
| [packages/gomoku-brain/docs/ALPHABETA_BRAIN.md](packages/gomoku-brain/docs/ALPHABETA_BRAIN.md) | Eight alpha-beta enhancements with worked examples |
| [packages/gomoku-brain/docs/MCTS_BRAIN.md](packages/gomoku-brain/docs/MCTS_BRAIN.md) | PUCT, heuristic priors, Dirichlet noise, VCF safety gate |
| [packages/gomoku/docs/RECORDING.md](packages/gomoku/docs/RECORDING.md) | Game recording JSON format, playback controls, Python API |

---

## Development

```bash
# Sync all packages
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

### Project layout

```
pyproject.toml              ← uv workspace root
gomoku.toml                 ← runtime configuration
packages/
  gomoku/                   ← core engine
  gomoku-brain/             ← AI brains
  gomoku-cli/               ← terminal interface
recordings/                 ← saved games & diagnostics
docs/                       ← project documentation
```

---

## License

See the project root for licence information.
