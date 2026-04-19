# Diagnostics

Per-move diagnostics recording captures the internal decision-making of each
brain during a game.  The output is a `.diag.jsonl` file (JSON Lines) that
can be analysed after the game to understand why specific moves were chosen
and to spot blunders, missed threats, or search inefficiencies.

---

## Table of Contents

1. [Overview](#overview)
2. [Enabling Diagnostics](#enabling-diagnostics)
3. [File Format](#file-format)
4. [Data Model](#data-model)
5. [Context API](#context-api)
6. [What Brains Record](#what-brains-record)
7. [Reading Diagnostics](#reading-diagnostics)
8. [Post-game Analysis (`gomoku analyze`)](#post-game-analysis-gomoku-analyze)
9. [Playback Overlay](#playback-overlay)
10. [Configuration](#configuration)

---

## Overview

| Property | Value |
|----------|-------|
| File format | JSON Lines (`.diag.jsonl`) |
| Naming | `gomoku_{timestamp}.diag.jsonl` |
| Location | Same directory as recordings (configurable) |
| Content | Header + one JSON object per move + summary |
| Writers | `DiagnosticsRecorder` in `gomoku.diagnostics` |
| Readers | `gomoku analyze` (see below) or any JSON Lines parser (Python, `jq`, etc.) |

---

## Enabling Diagnostics

### CLI

```bash
# Enable diagnostics (saves alongside recordings)
uv run gomoku-cli play --black threat --white alphabeta --diagnostics

# Custom output directory
uv run gomoku-cli play --black threat --white alphabeta \
    --diagnostics --diagnostics-dir my_diags/
```

### gomoku.toml

```toml
[diagnostics]
enabled = true
directory = "recordings"
top_candidates = 10
```

### Both recording and diagnostics

When both `--record` and `--diagnostics` are enabled
and the same directory is used, the files share a timestamp so they are easy
to correlate:

```
recordings/gomoku_20260405_143012.json          ← game recording
recordings/gomoku_20260405_143012.diag.jsonl     ← diagnostics
recordings/gomoku_20260405_143012.md5sum         ← recording checksum
```

---

## File Format

The `.diag.jsonl` file uses JSON Lines — one JSON object per line:

```
{"type":"header","format_version":1,"started_at":"...","game_config":{...},"players":[...]}
{"move_number":1,"stone":1,"chosen_move":{"row":7,"col":7},"decision_path":"opening_book",...}
{"move_number":2,"stone":2,"chosen_move":{"row":6,"col":6},"decision_path":"alphabeta",...}
...
{"type":"summary","total_moves":42,"winner":1,"is_draw":false}
```

### Line types

| Line | Content |
|------|---------|
| **Header** (line 1) | Game metadata: config, players, start time |
| **Move entries** (lines 2–N) | One `MoveDiagnostics` per move |
| **Summary** (last line) | Game outcome, total moves |

---

## Data Model

### MoveDiagnostics

The central data structure, one per move:

| Field | Type | Description |
|-------|------|-------------|
| `move_number` | `int` | 1-based move number |
| `stone` | `Stone` | Which colour is moving (1=Black, 2=White) |
| `chosen_move` | `Move \| None` | The move the brain selected |
| `decision_path` | `str` | Algorithm that chose the move: `"opening_book"`, `"vcf_precheck"`, `"vcf_win"`, `"pns_win"`, `"vct_win"`, `"vcf_aware_search"`, `"vct_aware_search"`, `"block_fork"`, `"immediate_win"`, `"block_immediate_win"`, `"alphabeta"`, `"mcts"`, `"mcts_vcf_gated"`, `"single_candidate"`, `"instant_win"`, `"instant_block"`, etc. |
| `think_time_ms` | `int` | Wall-clock time the brain spent thinking (stamped by the CLI around each `get_move`) |
| `board_stones` | `list[str]` | Board state snapshot: `["B H8", "W G7", ...]` |
| `eval_before` | `float \| None` | Static evaluation before the move |
| `eval_after` | `float \| None` | Static evaluation after the move |
| `search_stats` | `SearchStats` | Algorithm statistics (see below) |
| `top_candidates` | `list[CandidateInfo]` | Top N candidate moves with scores |
| `alternative_best` | `CandidateInfo \| None` | When the chosen move is not the top-ranked candidate, the one that was ranked highest. `None` when the brain picked its own top candidate. |
| `our_threats` | `ThreatInfo` | Threats for the moving side |
| `opp_threats` | `ThreatInfo` | Threats for the opponent |
| `notes` | `list[str]` | Free-form brain annotations |

### SearchStats

| Field | Type | Description |
|-------|------|-------------|
| `depth_reached` | `int \| None` | Deepest ply searched |
| `nodes_searched` | `int \| None` | Total nodes expanded in the main search |
| `qs_nodes_searched` | `int \| None` | Nodes expanded in quiescence search (alpha-beta only) |
| `iterations` | `int \| None` | MCTS iterations / PNS iterations |
| `tt_size` | `int \| None` | Transposition table entries |
| `tt_hits` | `int \| None` | TT lookup hits |
| `tt_probes` | `int \| None` | Total TT lookups — hit rate is `tt_hits / tt_probes` |
| `time_ms` | `int \| None` | Search time in milliseconds |
| `aspiration_resets` | `int \| None` | Number of aspiration window re-searches |

### ThreatInfo

| Field | Type | Description |
|-------|------|-------------|
| `four_threats` | `int` | Number of four-threats detected |
| `three_threats` | `int` | Number of three-threats detected |
| `vcf_result` | `str \| None` | `"win"` or `"loss"` if VCF found a result |
| `vct_result` | `str \| None` | `"win"` or `"loss"` if VCT found a result |

### CandidateInfo

| Field | Type | Description |
|-------|------|-------------|
| `move` | `Move` | The candidate move |
| `score` | `float` | Score for this candidate (units depend on `score_type`) |
| `score_type` | `str` | How to interpret `score`. One of: `"eval"` (heuristic eval — alpha-beta), `"winrate"` (average value ∈ [0,1] — MCTS), `"visits"` (raw visit count), `"priority"` (prior / ordering score). Default: `"eval"`. |

---

## Context API

Brains detect an active diagnostics context via a `ContextVar` and populate it
during `get_move()`:

```python
from gomoku.diagnostics import current_move_diagnostics

diag = current_move_diagnostics()
if diag is not None:
    diag.decision_path = "vcf_win"
    diag.chosen_move = move
    diag.notes.append("VCF found forced win at depth 12")
    diag.search_stats = SearchStats(
        depth_reached=12,
        nodes_searched=5000,
        tt_hits=128,
        tt_probes=2048,
        time_ms=45,
    )
    diag.our_threats = ThreatInfo(four_threats=2, vcf_result="win")
```

### Lifecycle

1. **`begin_move()`** — the CLI creates a `MoveDiagnostics` and sets it as the
   active context (via `set_move_diagnostics()`).
2. **Brain runs** — the brain detects the context via
   `current_move_diagnostics()` and populates fields.
3. **`end_move()`** — the CLI clears the context and appends the entry to the
   `.diag.jsonl` file.

This design is non-intrusive: brains that don't check for diagnostics work
unchanged.  The overhead is zero when diagnostics are disabled.

---

## What Brains Record

| Brain | Decision paths | Stats recorded |
|-------|---------------|----------------|
| **ThreatBrain** | `"immediate_win"`, `"block_immediate_win"`, `"vcf_win"`, `"pns_win"`, `"vct_win"`, `"vcf_aware_search"`, `"vct_aware_search"`, `"block_fork"`; falls through to the alpha-beta paths below | Threats, VCF/VCT results, PNS nodes, fallback depth |
| **AlphaBetaBrain** | `"opening_book"`, `"vcf_precheck"`, `"alphabeta"`; VCF-aware: `"instant_win"`, `"vcf_win"`, `"vcf_block_no_breaker"`, `"vcf_breaker"` | TT size / hits / probes, depth, main and QS nodes, aspiration resets, top candidates (`score_type="eval"`), `alternative_best`, our/opp threats |
| **MCTSBrain** | `"mcts"`, `"mcts_vcf_gated"`, `"single_candidate"`, `"instant_win"`, `"instant_block"` | Iterations, root visits (`nodes_searched`), top candidates as average-value winrates (`score_type="winrate"`), `alternative_best` |

---

## Reading Diagnostics

For day-to-day use prefer [`gomoku analyze`](#post-game-analysis-gomoku-analyze),
which consumes the same file and produces per-player stats plus a list of
suspect moves.  The raw recipes below are still useful for custom queries.

### Python

```python
import json
from pathlib import Path

lines = Path("recordings/gomoku_20260405_143012.diag.jsonl").read_text().splitlines()
header = json.loads(lines[0])
moves = [json.loads(line) for line in lines[1:] if not line.startswith('{"type":"summary"')]
summary = json.loads(lines[-1])

# Find moves where the brain fell back to plain alpha-beta
fallbacks = [m for m in moves if m["decision_path"] == "alphabeta"]
print(f"{len(fallbacks)} moves used alpha-beta fallback")
```

### jq (command line)

```bash
# List all decision paths
cat recordings/*.diag.jsonl | jq -r 'select(.decision_path) | .decision_path' | sort | uniq -c

# Find moves with high think time
cat recordings/*.diag.jsonl | jq 'select(.think_time_ms > 5000)'

# Show threat landscape for a specific move
cat recordings/*.diag.jsonl | jq 'select(.move_number == 15)'

# TT hit-rate per move (alpha-beta only)
cat recordings/*.diag.jsonl | jq 'select(.search_stats.tt_probes) |
    {m:.move_number, hit_rate:(.search_stats.tt_hits / .search_stats.tt_probes)}'
```

---

## Post-game Analysis (`gomoku analyze`)

The `gomoku analyze` subcommand parses a `.diag.jsonl` file and reports
per-player statistics plus a list of suspect moves:

```bash
uv run gomoku analyze recordings/gomoku_20260405_143012.diag.jsonl

# Machine-readable output for further scripting
uv run gomoku analyze recordings/gomoku_20260405_143012.diag.jsonl --json
```

### Flagging heuristics

Each move is checked against five independent heuristics.  A move is
flagged when **any** of them fires and all matching reasons are recorded.

| Reason | Fires when | Default threshold |
|--------|-----------|-------------------|
| `eval_drop` | Your own `eval_after` dropped sharply versus your previous move | ≥ 500 eval units |
| `bypassed_top_candidate` | `alternative_best` outscored the chosen move by a wide margin (only when `score_type == "eval"`) | ≥ 500 eval units |
| `time_spike` | Think time spiked compared to this player's running average | ≥ 2× average and ≥ 200 ms |
| `opp_double_four` | Opponent has ≥ 2 four-threats after your move and you did not pick a `"block_*"` decision path | — |
| `opp_forced_win` | The opponent's `vcf_result` or `vct_result` is `"win"` | — |

Thresholds live as module constants in `gomoku_cli.analyze`
(`BLUNDER_EVAL_DROP`, `ALTERNATIVE_GAP`, `TIME_SPIKE_FACTOR`,
`TIME_SPIKE_MIN_MS`) — edit them in place if you need different sensitivity.

### Report shape

The `--json` form serialises an `AnalysisReport`:

```json
{
  "path": "recordings/gomoku_20260405_143012.diag.jsonl",
  "header": { ... },
  "summary": { ... },
  "total_moves": 42,
  "player_stats": {
    "Black": {
      "provider": "threat",
      "moves": 21,
      "total_think_ms": 4210,
      "avg_think_ms": 200.5,
      "total_nodes": 128500,
      "max_depth_reached": 6,
      "avg_depth": 4.1,
      "blunders": 1
    },
    "White": { ... }
  },
  "flagged": [
    {
      "move_number": 27,
      "stone": "Black",
      "decision_path": "vcf_aware_search",
      "reasons": ["eval_drop", "opp_double_four"],
      "details": { "eval_drop": 820.0, "opp_fours": 2 }
    }
  ]
}
```

---

## Playback Overlay

When a `.diag.jsonl` file sits next to a `.json` recording (default layout
from `--record --diagnostics`), `gomoku playback` auto-loads it and renders
a compact diagnostics overlay under each move:

```bash
uv run gomoku playback recordings/gomoku_20260405_143012.json
```

The overlay shows:

- `path=<decision_path>`, `depth`, `nodes`, `tt_hits/tt_probes (rate%)`,
  `iters`, `search_ms`, and the `eval_before → eval_after` swing.
- A `threats:` line summarising own/opponent four- and three-threat counts
  when either is non-zero.
- A `top:` line listing the first five candidates (move=score).
- An `alt-best:` line whenever `alternative_best` is set — i.e. the brain
  picked something other than the top-ranked candidate.

Pass an explicit diagnostics path via the `run_playback()` Python API when
the companion file lives elsewhere.

---

## Configuration

### Settings model

```python
class DiagnosticsSettings(BaseModel):
    enabled: bool = False
    directory: Path = Path("recordings")
    top_candidates: int = Field(default=10, ge=1, le=50)
```

### gomoku.toml

```toml
[diagnostics]
enabled = false          # Set to true for always-on diagnostics
directory = "recordings" # Output directory
top_candidates = 10      # Max candidate moves to record per entry
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--diagnostics` / `--no-diagnostics` | Override config setting |
| `--diagnostics-dir PATH` | Override output directory |
