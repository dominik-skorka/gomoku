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

### `match` — Measure which engine is stronger

```bash
# A git revision against the working tree
uv run gomoku match --a brain=threat,rev=master --b brain=threat --games 40 --jobs 3

# Two settings of the same brain
uv run gomoku match --a brain=threat --b brain=threat,config=other.toml --games 40
```

Plays engine A against engine B over generated openings, each opening
twice with colours swapped, and reports wins, score with a 95% interval
(computed per opening pair, which cancels colour bias), the implied Elo
difference and think time per move.  An engine is
`brain=NAME[,rev=GIT_REV][,config=PATH]`: without `rev` it runs the
working tree, with `rev` that commit is exported to a per-commit cache
under the system temp directory (the first start of a revision compiles
its kernels).  Openings where either colour already has a forced win are
skipped unless `--any-openings` is given.  `--out games.json` saves every
game.  Think times are only comparable at `--jobs 1`.

### `review` — Check a change for blunders before a long match

```bash
uv run gomoku review --a brain=threat,rev=master --b brain=threat --games 6 --jobs 3
```

A quick gate to run before `match`: plays a few games between the
baseline (A) and the candidate (B), then checks every engine move with
an *oracle*, a threat-space proof-number search with a much larger
budget (`--oracle-nodes`, `--oracle-time`) running from a separate
source tree (`--oracle-rev`, by default A's revision), so a bug in the
candidate's search can't vouch for its own moves.  It flags:

| Blunder | Meaning |
|---|---|
| `missed_win` | The mover could complete five, or had a proven forced win that is disproved after its move and the reply |
| `missed_block` | The opponent threatened five on a single cell and the mover played elsewhere |
| `losing_move` | After the move the opponent has a proven forced win; the report shows a move that held |
| `unsound_proof` | The brain claimed a forced win the oracle disproves, or the oracle's own proof failed after it was followed |

Each blunder (except `unsound_proof`, which is always a bug) is then
replayed with the same engine: once unchanged, then with larger
settings, cheapest first — for the threat brain a 10× PNS budget, one
more fallback ply, a doubled fallback width, and all of them together.
The first setting that finds a good move is reported as **TUNING**,
with its think-time cost; if none does, the blunder is a **BUG** in the
brain's logic:

```
game 1, move 16: B (threat, white) played G9 — missed win: forced win starting I8; gone after the reply
    decision: alphabeta, 0.7 s
    retried: repeat no G9 (0.6 s) | more_pns ok I8 (0.4 s)
    => TUNING: more_pns: pns_nodes=100000, pns_time_s=20.0 (0.6x think time)

Verdict: WARN — B has 2 blunder(s), A has 2; no more of them need a code fix than A's
```

The verdict is `FAIL` (exit status 1) when B has more BUG blunders than
A, `WARN` when B has blunders, and `PASS` otherwise.  Fix a `FAIL`
before spending time on a long `match`.  `--out review.json` saves the
games, the oracle's verdict for every position and each blunder with
its retries; `--no-escalate` skips the retries.

**In CI.**  Every pull request that touches the engine or `gomoku.toml`
runs a 6-game review against its base branch (`.github/workflows/review.yml`)
and fails on `FAIL`.  Adding the `match` label to a pull request runs the
review and then a 300-game match against the base
(`.github/workflows/match.yml`; remove and re-add the label to run it
again, or start it by hand).  The match fails only when the pull request
is significantly weaker (`gomoku match --fail-if-weaker`).  Results are
in the run summary and as `review.json` / `match.json` artifacts.

---

## Configuration

Runtime settings are managed via `gomoku.toml` with pydantic-settings.

**Priority** (highest → lowest):
1. CLI flags
2. Environment variables (`GOMOKU_*` prefix)
3. `gomoku.toml`
4. Built-in defaults

The TOML file is discovered automatically: the loader walks up from the
current working directory until it finds `gomoku.toml` or hits a project
root marker (`pyproject.toml` / `.git`).  Set `GOMOKU_CONFIG=/path/to/config.toml`
to bypass discovery and use an explicit file.

Brain aliases allow multiple configurations of the same engine:

```toml
[brains.threat_fast]
type = "threat"
pns_nodes = 10_000      # a third of the shipped threat brain's PNS budget

[brains.threat_fast.fallback]
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
