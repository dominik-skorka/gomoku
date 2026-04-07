# Architecture

This document describes the structure, design decisions, and component
relationships of the Gomoku workspace.

---

## Table of Contents

1. [Workspace Layout](#workspace-layout)
2. [Package Overview](#package-overview)
3. [Dependency Graph](#dependency-graph)
4. [Core Data Models](#core-data-models)
5. [MoveProvider Protocol](#moveprovider-protocol)
6. [Game Engine](#game-engine)
7. [AI Brains](#ai-brains)
8. [Board Evaluation](#board-evaluation)
9. [Candidate Generation](#candidate-generation)
10. [Configuration System](#configuration-system)
11. [Observability](#observability)

---

## Workspace Layout

```
gomoku/
├── pyproject.toml            ← uv workspace root
├── gomoku.toml               ← runtime configuration (TOML)
├── docs/                     ← project-level documentation
├── packages/
│   ├── gomoku/               ← core game engine
│   │   ├── src/gomoku/
│   │   │   ├── models.py     ← Stone, Move, GameConfig, GameResult
│   │   │   ├── board.py      ← Board (Pydantic, immutable-style)
│   │   │   ├── rules.py      ← Win detection
│   │   │   ├── game.py       ← Game orchestration loop
│   │   │   ├── protocol.py   ← MoveProvider protocol
│   │   │   ├── recorder.py   ← Game recording & playback data
│   │   │   ├── diagnostics.py← Per-move diagnostics capture
│   │   │   └── settings.py   ← Pydantic-settings configuration
│   │   └── tests/
│   ├── gomoku-brain/         ← AI move providers
│   │   ├── src/gomoku_brain/
│   │   │   ├── random.py     ← RandomBrain
│   │   │   ├── minimax.py    ← MinimaxBrain
│   │   │   ├── negamax.py    ← NegamaxBrain
│   │   │   ├── alphabeta.py  ← AlphaBetaBrain (8 enhancements)
│   │   │   ├── threat.py     ← ThreatBrain (VCF/VCT/PNS)
│   │   │   ├── mcts.py       ← MCTSBrain (PUCT + heuristic eval)
│   │   │   ├── pns.py        ← Proof-Number Search
│   │   │   ├── evaluate.py   ← Heuristic board evaluation
│   │   │   ├── evaluate_jit.py ← Numba JIT evaluation
│   │   │   ├── candidates.py ← Candidate move generation
│   │   │   ├── opening.py    ← Hardcoded opening book + symmetry
│   │   │   ├── opening_book.py ← File-based opening book
│   │   │   ├── zobrist.py    ← Zobrist hashing & transposition table
│   │   │   ├── bitboard.py   ← Bitboard representation
│   │   │   ├── bitboard_jit.py ← Numba JIT bitboard ops
│   │   │   ├── grid_utils.py ← Raw int-grid board helpers
│   │   │   └── __init__.py   ← Brain registry & factory
│   │   ├── docs/
│   │   │   ├── THREAT_BRAIN.md
│   │   │   └── ALPHABETA_BRAIN.md
│   │   └── tests/
│   └── gomoku-cli/           ← Terminal user interface
│       ├── src/gomoku_cli/
│       │   ├── main.py       ← Click CLI (play, playback commands)
│       │   ├── display.py    ← Rich-based board rendering
│       │   ├── keyboard.py   ← Human player input provider
│       │   └── playback.py   ← Arrow-key game replay
│       └── tests/
├── recordings/               ← Saved game JSON files + MD5 checksums
└── scripts/                  ← Analysis & generation utilities
```

---

## Package Overview

| Package | Import name | Purpose | Key dependencies |
|---------|-------------|---------|------------------|
| **gomoku** | `gomoku` | Core engine: board, rules, game loop, recording, settings | pydantic, pydantic-settings, logfire |
| **gomoku-brain** | `gomoku_brain` | AI brains, evaluation, search algorithms | gomoku, numba (optional), numpy |
| **gomoku-cli** | `gomoku_cli` | Terminal interface, display, human input | gomoku, gomoku-brain, click, rich |

---

## Dependency Graph

```
gomoku-cli
  ├── gomoku-brain
  │     └── gomoku (core)
  └── gomoku (core)
```

- **gomoku** has zero internal package dependencies — it defines the models,
  rules, and protocol that everything else builds on.
- **gomoku-brain** depends on **gomoku** for `Board`, `Stone`, `Move`, and the
  `MoveProvider` protocol.
- **gomoku-cli** depends on both packages and wires them together at the CLI
  entrypoint.

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
- `is_valid_move(row, col) → bool` — bounds check + empty check.
- `place_stone(row, col, stone) → Board` — returns a **new** board with the
  stone placed (the original is unchanged).

---

## MoveProvider Protocol

```python
@runtime_checkable
class MoveProvider(Protocol):
    @property
    def name(self) -> str: ...

    def get_move(self, board: Board, stone: Stone) -> Move: ...
```

Every brain (AI or human) implements this protocol.  The game engine calls
`get_move()` each turn and expects a valid `Move` back.

---

## Game Engine

`gomoku.game.Game` orchestrates a full match:

```
Game(config, black_provider, white_provider, on_move=callback)
```

Loop:
1. `play_turn()` calls the current provider's `get_move()`.
2. Validates the returned move (`InvalidMoveError` if illegal).
3. Places the stone on the board.
4. Calls `on_move(stone, move)` (used for recording / diagnostics).
5. Checks win condition via `rules.check_winner()`.
6. Alternates stone colour.

The game is played turn by turn — the caller drives the loop (the CLI calls
`play_turn()` in a `while True` loop).

---

## AI Brains

All brains are registered in `gomoku_brain.__init__` and created via the
`get_brain()` / `get_brain_from_settings()` factory functions.

| Brain | Module | Algorithm | Strength |
|-------|--------|-----------|----------|
| `random` | `random.py` | Uniform random candidate | Weakest |
| `minimax` | `minimax.py` | Classic minimax (no pruning) | Weak |
| `negamax` | `negamax.py` | Negamax framework | Weak–medium |
| `alphabeta` | `alphabeta.py` | Negamax + alpha-beta + 8 enhancements | Strong |
| `threat` | `threat.py` | VCF + PNS/VCT + alpha-beta fallback | **Strongest** |
| `mcts` | `mcts.py` | PUCT + heuristic priors + heuristic eval | Medium |
| `keyboard` | `gomoku_cli/keyboard.py` | Human terminal input | N/A |

See [THREAT_BRAIN.md](../packages/gomoku-brain/docs/THREAT_BRAIN.md) and
[ALPHABETA_BRAIN.md](../packages/gomoku-brain/docs/ALPHABETA_BRAIN.md) for
in-depth documentation of the two strongest brains.

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
- **Defensive weighting** — opponent patterns scored at 0.8× (blocking is
  valuable but slightly less important than building your own threats).
- **Clamped** to ±99,000 so heuristic scores never equal the 100,000 win
  constant.

When Numba is available and `evaluation.use_jit = true`, `evaluate_jit.py`
provides a JIT-compiled implementation for significantly faster scoring.

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

1. CLI flags (`--depth 4`)
2. Environment variables (`GOMOKU_BRAINS__ALPHABETA__DEPTH=4`)
3. `gomoku.toml` file
4. Field defaults in settings models

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
# ...

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

The CLI resolves brain names via `settings.brains.resolve(name)` which returns
`(brain_type, validated_kwargs)`.

---

## Observability

The project uses [Pydantic Logfire](https://logfire.pydantic.dev/) as the
default observability framework.  All brains and the game engine emit
structured spans and log messages via `logfire.span()` and `logfire.info()`.

Key instrumentation points:
- `game.play_turn` — one span per move.
- `brain.<name>` — one span per `get_move()` call with search parameters.
- Opening book lookups, VCF/VCT results, fallback decisions.

Logfire is configured at CLI startup with `logfire.configure(send_to_logfire=False)`
for local-only logging by default.
