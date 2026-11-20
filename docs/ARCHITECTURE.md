# Architecture

This document describes the structure, design decisions, and component
relationships of the Gomoku workspace.

---

## Table of Contents

1. [Workspace Layout](#workspace-layout)
2. [Package Overview](#package-overview)
3. [Dependency Graph](#dependency-graph)
4. [Core Data Models](#core-data-models)
5. [Interfaces](#interfaces)
6. [Game Engine](#game-engine)
7. [AI Brains](#ai-brains)
8. [Board Evaluation](#board-evaluation)
9. [Candidate Generation](#candidate-generation)
10. [Configuration System](#configuration-system)
11. [HTTP API](#http-api)
12. [Strength Testing](#strength-testing)
13. [Observability](#observability)

---

## Workspace Layout

```
gomoku/
├── pyproject.toml            ← uv workspace root
├── gomoku.toml               ← runtime configuration (TOML)
├── .github/workflows/        ← ci.yml (lint + tests), review.yml, match.yml
├── docs/                     ← project-level documentation
├── packages/
│   ├── gomoku/               ← core game engine
│   │   ├── src/gomoku/
│   │   │   ├── models.py     ← Stone, Move, GameConfig, GameResult
│   │   │   ├── board.py      ← Board (Pydantic, immutable-style)
│   │   │   ├── rules.py      ← Win detection
│   │   │   ├── game.py       ← Game: both sides brain-driven
│   │   │   ├── runner.py     ← GameRunner: each side brain-driven or external
│   │   │   ├── interfaces.py ← BoardView, SearchBoard, Player
│   │   │   ├── protocol.py   ← MoveProvider (old name of Player)
│   │   │   ├── recorder.py   ← Game recording & playback data
│   │   │   ├── diagnostics.py← Per-move diagnostics capture
│   │   │   ├── analyze.py    ← Post-game diagnostics analysis
│   │   │   └── settings.py   ← Pydantic-settings configuration
│   │   ├── docs/RECORDING.md
│   │   └── tests/
│   ├── gomoku-brain/         ← AI move providers
│   │   ├── src/gomoku_brain/
│   │   │   ├── __init__.py   ← Brain registry & factory
│   │   │   ├── brain_base.py ← Shared per-move span / diagnostics session
│   │   │   ├── random.py, minimax.py, negamax.py
│   │   │   ├── alphabeta.py  ← AlphaBetaBrain (8 enhancements)
│   │   │   ├── alphabeta_ordering.py ← Move classification and ordering
│   │   │   ├── alphabeta_quiescence.py ← Quiescence search
│   │   │   ├── alphabeta_root.py     ← Root VCF pre-check and breaker probe
│   │   │   ├── threat.py     ← ThreatBrain (VCF / PNS or VCT / defence / fallback)
│   │   │   ├── threat_detect.py      ← Four / three / five detection
│   │   │   ├── threat_search.py      ← VCF and VCT searches, VCF breakers
│   │   │   ├── threat_space.py       ← ThreatSpace: threats a search plays and answers
│   │   │   ├── pns.py        ← Proof-number search in threat space
│   │   │   ├── probe.py      ← Breaker probes, serial or on several threads
│   │   │   ├── line_patterns.py      ← Pure-Python line rules (five / three cells)
│   │   │   ├── mcts.py       ← MCTSBrain (PUCT + heuristic eval)
│   │   │   ├── priors.py     ← Move priors (MCTS, alpha-beta ordering)
│   │   │   ├── evaluate.py   ← Heuristic board evaluation (dispatch + Python)
│   │   │   ├── evaluate_jit.py ← Numba kernels: evaluation, threat detection
│   │   │   ├── bitboard.py, bitboard_jit.py ← Bitboard representation and kernels
│   │   │   ├── search_state.py ← Mutable make/unmake board with Zobrist hash
│   │   │   ├── zobrist.py    ← Zobrist keys & transposition table
│   │   │   ├── candidates.py ← Candidate move generation
│   │   │   ├── directions.py ← The four line directions
│   │   │   ├── grid_utils.py ← Raw int-grid board helpers
│   │   │   ├── opening.py    ← Hardcoded opening book + symmetry
│   │   │   ├── opening_book.py, opening_book_gen.py ← File-based book, generator
│   │   ├── docs/             ← THREAT_BRAIN.md, ALPHABETA_BRAIN.md, MCTS_BRAIN.md
│   │   └── tests/
│   ├── gomoku-cli/           ← Terminal user interface and tooling
│   │   ├── src/gomoku_cli/
│   │   │   ├── main.py       ← Click CLI (play, playback, analyze, generate-book, match, review)
│   │   │   ├── display.py    ← Rich-based board rendering
│   │   │   ├── keyboard.py   ← Human player input provider
│   │   │   ├── playback.py   ← Arrow-key game replay + diagnostics overlay
│   │   │   ├── analyze.py    ← Post-game .diag.jsonl analyser
│   │   │   ├── match.py      ← Engine-vs-engine matches (strength measurement)
│   │   │   ├── match_worker.py ← One brain per process, JSON lines
│   │   │   └── review.py     ← Blunder review with a proof-search oracle
│   │   └── tests/
│   └── gomoku-fastapi/       ← HTTP API (see its README.md)
│       ├── src/gomoku_fastapi/
│       │   ├── main.py, api_v1.py, versioning.py ← App, versioned router, version lifecycle
│       │   ├── routers/      ← brains, moves, evaluate, analyze, sessions, health, versions
│       │   ├── sessions/     ← In-memory game sessions (GameRunner)
│       │   └── deps.py, limits.py, observability.py, schemas.py
│       └── tests/            ← incl. contracts/v1.openapi.json (API contract snapshot)
└── recordings/               ← Saved games and diagnostics (created on demand)
```

---

## Package Overview

| Package | Import name | Purpose | Key dependencies |
|---------|-------------|---------|------------------|
| **gomoku** | `gomoku` | Core engine: board, rules, game loop, recording, settings | pydantic, pydantic-settings, logfire |
| **gomoku-brain** | `gomoku_brain` | AI brains, evaluation, search algorithms | gomoku, numba, numpy, logfire |
| **gomoku-cli** | `gomoku_cli` | Terminal interface, display, human input, `match` / `review` tooling | gomoku, gomoku-brain, click, rich |
| **gomoku-fastapi** | `gomoku_fastapi` | Versioned HTTP API over the engine and brains | gomoku, gomoku-brain, fastapi, uvicorn, slowapi, logfire |

Numba is a hard dependency; `[evaluation] use_jit = false` switches the
JIT kernels off in favour of the pure-Python paths.

---

## Dependency Graph

```
gomoku-cli ─────┐
                ├── gomoku-brain
gomoku-fastapi ─┘       └── gomoku (core)
   (both also import gomoku directly)
```

- **gomoku** has zero internal package dependencies — it defines the models,
  rules, and interfaces that everything else builds on.
- **gomoku-brain** depends on **gomoku** for `Board`, `Stone`, `Move`, and the
  interfaces in `gomoku.interfaces`.
- **gomoku-cli** and **gomoku-fastapi** depend on both and wire them
  together: one in the terminal, the other over HTTP.

---

## Core Data Models

All models live in `gomoku.models` and use Pydantic for validation.

### Stone

```python
class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
```

`Stone.opponent()` returns the other colour.

### Move

```python
class Move(BaseModel, frozen=True):
    row: int  # 0-indexed
    col: int  # 0-indexed
```

`str(Move(row=7, col=7))` → `"H8"` (chess-style notation: column A–O, row
1–15).

### GameConfig

```python
class GameConfig(BaseModel, frozen=True):
    board_size: int = 15    # NxN grid
    win_length: int = 5     # consecutive stones to win
```

### GameResult

```python
class GameResult(BaseModel):
    winner: Stone | None          # BLACK, WHITE, or None (draw)
    winning_moves: list[Move] | None  # the 5 cells forming the winning line
    is_draw: bool                 # property: True when winner is None
```

### Board

`gomoku.board.Board` is a Pydantic model with an **immutable-style** API:

| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Board side length (N) |
| `grid` | `list[list[Stone]]` | NxN grid |
| `occupied` | `set[tuple[int,int]]` | Occupied cell coordinates (avoids full-grid scans) |

Key methods:
- `at(row, col) → Stone` — read a cell.
- `is_valid_move(move) → bool` — bounds check + empty check.
- `place_stone(move, stone) → Board` — returns a **new** board with the
  stone placed (the original is unchanged).
- `get_valid_moves()`, `is_full()`, `stone_count()`.

Searches don't copy boards: they use `gomoku_brain.search_state.SearchState`,
a mutable grid with `make` / `unmake` and an incrementally updated Zobrist
hash (which encodes the side to move).

---

## Interfaces

`gomoku.interfaces` holds the structural types (`typing.Protocol`) the
packages depend on.  A class implements one by having its members, not by
inheriting; pyright checks this in CI (`uv run pyright`).

```python
class BoardView(Protocol):          # a position to read
    @property
    def size(self) -> int: ...
    @property
    def occupied(self) -> Collection[Cell]: ...
    def at(self, row: int, col: int) -> Stone: ...

class SearchBoard(BoardView, Protocol):   # changed in place by a search
    def make(self, row: int, col: int, stone: Stone) -> None: ...
    def unmake(self) -> None: ...

@runtime_checkable
class Player(Protocol):             # chooses moves for one side
    @property
    def name(self) -> str: ...
    def get_move(self, board: Board, stone: Stone) -> Move: ...
```

- `Board` (immutable, for games) and `SearchState` (mutable, for searches)
  are both a `BoardView`; `SearchState` is also a `SearchBoard`.
  Evaluation, candidate generation and threat detection take a `BoardView`,
  so they work on either.  Each class carries a `TYPE_CHECKING`-only
  function returning itself as the interface, so a change that breaks
  conformance fails the type check.
- Every brain, and the keyboard input, is a `Player`.  The game engine calls
  `get_move()` each turn and expects a valid `Move` back.  The registry
  (`BrainSpec.brain_cls: type[Player]`) is checked against it.
  `MoveProvider` is the old name, kept as an alias.
- Overriding methods are marked `@typing.override`; pyright reports one
  that is not (`reportImplicitOverride`), so renaming a base method cannot
  leave a subclass silently defining a new one.

---

## Game Engine

`gomoku.game.Game` orchestrates a full match:

```
Game(config, black, white, on_move=callback)
```

Loop:
1. `play_turn()` calls the current provider's `get_move()`.
2. Validates the returned move (`InvalidMoveError` if illegal).
3. Places the stone on the board.
4. Calls `on_move(stone, move)` (used for recording / diagnostics).
5. Checks win condition via `rules.check_winner()`.
6. Alternates stone colour.

The caller drives the loop turn by turn (`play_turn()`), or `play()` runs
it to the end.

`gomoku.runner.GameRunner` generalises this for the HTTP API: each side is
either brain-driven (`think()` runs its brain) or external
(`submit_move()` takes the move from the client).

---

## AI Brains

Every brain subclasses `gomoku_brain.base.Brain[S]`, generic over its
settings model `S` (a `gomoku.settings.BaseBrainSettings`):

```python
class GreedySettings(BaseBrainSettings):
    radius: int = Field(default=2, ge=1, le=4)

class GreedyBrain(Brain[GreedySettings], name="greedy"):
    @override
    def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
        ...  # self.settings.radius: validated and typed
```

That is the whole contract:

- **`choose_move`** is the one abstract method: the decision, for a board
  that is not full and the side to move.  `session.diag` is the per-move
  diagnostics record (`None` when off).  A class without it is a type
  error, and instantiating it raises `TypeError`.
- **`get_move`** (the `Player` method) is `@final` in the base: it opens
  the brain's tracing span and diagnostics session (`brain_session`) and
  calls `choose_move`, so every brain is traced the same way.
- **Settings** are the model's: `GreedyBrain()`, `GreedyBrain(radius=3)` or
  `GreedyBrain(GreedySettings(radius=3))`.  `BaseBrainSettings` forbids
  unknown keys, so a misspelt setting is an error everywhere: in code, on
  the command line and in `gomoku.toml`.  `settings_type` is read from the
  class statement (`types.get_original_bases`).
- **Registration** is the class statement: `name="greedy"` puts it in the
  registry, so it can be played, aliased in `gomoku.toml` and matched like
  a built-in brain.  Built-in brains are imported by `gomoku_brain` so they
  register on import.

`gomoku_brain.BRAINS` is a live mapping from name to `BrainSpec` (brain
class + settings model) over those registrations, plus players added with
`register_brain` (the CLI's keyboard input, a `Player` that is not a
`Brain`).

Public surface:

- `get_brain(name, **kwargs)` — instantiate by name with explicit kwargs.
- `resolve_brain(name, brain_settings)` — turn a brain name or TOML
  alias into a validated `(type, kwargs)` pair without instantiating.
- `get_brain_from_settings(name, brain_settings)` — `resolve_brain` +
  instantiate.
- `register_brain(name, player_cls, settings_cls)` — add a `Player` class
  that is not a `Brain` (used by the CLI for the keyboard provider);
  `unregister_brain(name)` removes an entry.
- `available_brains(brain_settings)` — sorted list of all resolvable
  names (built-ins + TOML aliases).

| Brain | Module | Algorithm | Strength |
|-------|--------|-----------|----------|
| `random` | `random.py` | Uniform random candidate | Weakest |
| `minimax` | `minimax.py` | Classic minimax (no pruning) | Weak |
| `negamax` | `negamax.py` | Negamax framework | Weak–medium |
| `alphabeta` | `alphabeta.py` | Negamax + alpha-beta + 8 enhancements | Strong |
| `threat` | `threat.py` | VCF + PNS/VCT + alpha-beta fallback | **Strongest** |
| `mcts` | `mcts.py` | PUCT + heuristic priors + heuristic eval | Medium |
| `keyboard` | `gomoku_cli/keyboard.py` | Human terminal input | N/A |

See [THREAT_BRAIN.md](../packages/gomoku-brain/docs/THREAT_BRAIN.md),
[ALPHABETA_BRAIN.md](../packages/gomoku-brain/docs/ALPHABETA_BRAIN.md), and
[MCTS_BRAIN.md](../packages/gomoku-brain/docs/MCTS_BRAIN.md) for
in-depth documentation of the three heaviest brains.

### Module split & import graph

`alphabeta.py` and `threat.py` were each split along behavioural seams
(detection helpers, search core, root pre-check, quiescence, move
ordering).  Module-scope imports inside `gomoku_brain` form a **DAG**;
there are no load-order tricks.  By layer — each module imports only from
the layers above it in this table:

| Layer | Modules |
|-------|---------|
| 0 (leaves) | `threat_space`, `line_patterns`, `directions`, `candidates`, `zobrist`, `brain_base`, `grid_utils`, `opening`, `evaluate_jit`, `bitboard_jit` |
| 1 | `search_state`, `evaluate`, `bitboard`, `opening_book`, `random` |
| 2 | `threat_detect`, `probe`, `alphabeta_ordering`, `priors`, `minimax`, `negamax`, `opening_book_gen` |
| 3 | `threat_search` (VCF, VCT, VCF breakers) |
| 4 | `pns`, `alphabeta_root`, `alphabeta_quiescence` |
| 5 | `alphabeta` |
| 6 | `threat` (ThreatBrain) |
| 7 | `mcts` (imports `classify_vcf_breakers` / `probe_immediate` via `threat`) |
| 8 | `__init__` (the registry) |

- `threat_detect` (detection primitives) and `alphabeta_ordering` (move
  classification, `_scored_moves`) sit just above the leaves.
- `threat.py` re-exports the detection and search helpers for backward
  compatibility (`test_module_split.py` and `test_threat_search_split.py`
  lock in the `is`-identity of those re-exports).
- Imports inside function bodies are reserved for optional / heavy
  dependencies (numba kernels, `grid_utils`), never to defuse a cycle.

`tests/test_import_graph.py` fails if a module-scope import introduces a
cycle.  When splitting another module, move shared helpers *down* into a
leaf module rather than importing back up.

---

## Board Evaluation

`gomoku_brain.evaluate.evaluate_board(board, stone) → float`

The heuristic scores a position by scanning every occupied cell in four
directions (horizontal, vertical, two diagonals):

| Pattern (consecutive, open ends) | Score |
|---|---|
| (5, *) — five | 100,000 |
| (4, 2) — live-four | 10,000 |
| (4, 1) — dead-four | 1,000 |
| (3, 2) — live-three | 1,000 |
| (3, 1) — dead-three | 100 |
| (2, 2) — live-two | 100 |
| (2, 1) — dead-two | 10 |

Additional scoring:
- **Fork detection** — cells where ≥ 2 threat lines converge receive a
  `_FORK_BONUS` (3,000 per extra line).
- **Open-three bonus** — each live-three earns an extra 1,500 points.
- **Symmetric weighting** — the opponent's patterns are subtracted at the
  same weight (1.0×).  The *move priors* (`priors.py`, used by MCTS and to
  rank alpha-beta candidates) weigh the line a move blocks at 0.8× of the
  line it extends: blocking is valuable but slightly less important than
  building your own threats.
- **Clamped** to ±99,000 so heuristic scores never equal the 100,000 win
  constant.

`evaluate_board` dispatches to the bitboard kernels (`bitboard_jit.py`)
when `[bitboard] enabled = true`, else to the grid kernels
(`evaluate_jit.py`) when `[evaluation] use_jit = true`, else to Python.
`test_parity.py` keeps the three in agreement.

The evaluation scores unbroken runs only.  Threat detection for the
searches — fours including split ones (`XXX_X`), threes — is separate; see
[THREATS.md](THREATS.md).

---

## Candidate Generation

`gomoku_brain.candidates.get_candidate_moves(board, radius=2) → list[Move]`

Instead of evaluating all ~225 cells on a 15×15 board, only cells **within
`radius`** of at least one occupied cell are considered.  On an empty board
the center cell is returned.

This reduces candidates from ~225 to ~10–40 in typical mid-game positions —
a massive branching factor reduction critical for search performance.

---

## Configuration System

Configuration is managed by `pydantic-settings` with a TOML backend.

### Priority (highest → lowest)

1. CLI flags (`--board-size 9`)
2. Environment variables (`GOMOKU_GAME__BOARD_SIZE=9`; `__` separates
   nested keys)
3. `gomoku.toml` file (discovered via `gomoku.settings.find_config_file`)
4. Field defaults in settings models

### Config discovery

`gomoku.toml` is located by `find_config_file` on each `GomokuSettings()`
instantiation:

1. If `GOMOKU_CONFIG` is set, use that file (no fallback if it's missing).
2. Otherwise walk up from the current working directory toward the
   filesystem root.  At each directory:
   - if it contains `gomoku.toml`, return it;
   - else if it contains `pyproject.toml` or `.git`, stop — that's a
     project boundary and we don't leak into the parent.
3. Return `None` (use field defaults) when no config exists in the
   search path.

This means `uv run gomoku ...` works the same whether you run it from
the project root or any subdirectory.

### Key sections

```toml
[game]
board_size = 15
win_length = 5

[brains.alphabeta]
type = "alphabeta"
depth = 4
# ... brain-specific knobs

[brains.threat]
type = "threat"
vcf_depth = 20
pns_nodes = 120_000
# ...

[brains.threat.fallback]        # alpha-beta settings for the fallback
depth = 5

[brains.threat.threats.own]     # threat spaces: own / opponent / defence / forks
split_threes = false

[evaluation]
use_jit = true

[bitboard]
enabled = true

[recording]
enabled = false
directory = "recordings"

[diagnostics]
enabled = false
directory = "recordings"
top_candidates = 10

[api]                           # HTTP API: CORS, rate limits, logfire
rate_limit_compute = "30/minute"
```

### Brain aliases

Brain aliases are defined under `[brains.<name>]` with a `type` key mapping
to a built-in brain engine.  Multiple aliases can share the same engine with
different parameters:

```toml
[brains.alphabeta_depth_6]
type = "alphabeta"
depth = 6
```

`gomoku_brain.resolve_brain(name, settings.brains)` turns a brain name or
alias into `(brain_type, validated_kwargs)`; `get_brain_from_settings`
also instantiates it.  The CLI, the HTTP API and the match workers all go
through it.

---

## HTTP API

`gomoku-fastapi` serves the engine over HTTP under a versioned prefix
(`/v1`): stateless moves and evaluation, recording analysis, and stateful
game sessions built on `GameRunner`.  CPU-bound brain calls run on a thread
pool.  The version lifecycle (deprecation and sunset headers, `410 Gone`
after retirement) and a contract test against an OpenAPI snapshot are
described in [packages/gomoku-fastapi/README.md](../packages/gomoku-fastapi/README.md).

---

## Strength Testing

Two CLI commands measure brain changes; both run each engine in its own
worker process (`match_worker.py`), optionally from another git revision
(`brain=threat,rev=master`) or config (`config=other.toml`):

- **`gomoku match`** plays engine A against engine B over generated
  openings, each opening with colours swapped, and reports the score, Elo
  difference with a 95% interval, and think time.
- **`gomoku review`** plays a few games and checks every move with a
  proof-number search oracle (missed wins, missed blocks, losing moves,
  unsound proofs), follows each claimed forced win to the end of the game
  (unconverted wins), retries each blunder with larger settings to tell
  tuning issues from bugs, and draws each blunder's position.

CI runs `review` on every pull request that touches the engine and `match`
on demand (the `match` label); see the main README.

---

## Observability

The project uses [Pydantic Logfire](https://logfire.pydantic.dev/) as the
default observability framework.  All brains and the game engine emit
structured spans and log messages via `logfire.span()` and `logfire.info()`.

Key instrumentation points:
- `gomoku.game` — one span per game; `gomoku.turn` — one per move.
- `brain.<name>` — one span per `get_move()` call with search parameters
  (`brain_base.brain_session`).
- Opening book lookups, VCF / PNS / VCT results, fallback decisions.

The CLI configures Logfire at startup with
`logfire.configure(send_to_logfire=False, inspect_arguments=False)` for
local-only logging.  The HTTP API instruments FastAPI requests and is
configured from `[api]` (`logfire_send`, `logfire_console`).
