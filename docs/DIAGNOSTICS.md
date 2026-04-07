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
8. [Configuration](#configuration)

---

## Overview

| Property | Value |
|----------|-------|
| File format | JSON Lines (`.diag.jsonl`) |
| Naming | `gomoku_{timestamp}.diag.jsonl` |
| Location | Same directory as recordings (configurable) |
| Content | Header + one JSON object per move + summary |
| Writers | `DiagnosticsRecorder` in `gomoku.diagnostics` |
| Readers | Any JSON Lines parser (Python, `jq`, etc.) |

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
{"move_number":2,"stone":2,"chosen_move":{"row":6,"col":6},"decision_path":"Alpha-Beta",...}
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
| `decision_path` | `str` | Algorithm that chose the move: `"VCF"`, `"PNS"`, `"VCT"`, `"Alpha-Beta"`, `"opening_book"`, `"vcf_precheck"`, etc. |
| `think_time_ms` | `int` | Wall-clock time the brain spent thinking |
| `board_stones` | `list[str]` | Board state snapshot: `["B H8", "W G7", ...]` |
| `eval_before` | `float \| None` | Static evaluation before the move |
| `eval_after` | `float \| None` | Static evaluation after the move |
| `search_stats` | `SearchStats` | Algorithm statistics (see below) |
| `top_candidates` | `list[CandidateInfo]` | Top N candidate moves with scores |
| `our_threats` | `ThreatInfo` | Threats for the moving side |
| `opp_threats` | `ThreatInfo` | Threats for the opponent |
| `notes` | `list[str]` | Free-form brain annotations |

### SearchStats

| Field | Type | Description |
|-------|------|-------------|
| `depth_reached` | `int \| None` | Deepest ply searched |
| `nodes_searched` | `int \| None` | Total nodes expanded |
| `iterations` | `int \| None` | MCTS iterations / PNS iterations |
| `tt_size` | `int \| None` | Transposition table entries |
| `tt_hits` | `int \| None` | TT lookup hits |
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
| `score` | `float` | Evaluation score for this candidate |

---

## Context API

Brains detect an active diagnostics context via a `ContextVar` and populate it
during `get_move()`:

```python
from gomoku.diagnostics import current_move_diagnostics

diag = current_move_diagnostics()
if diag is not None:
    diag.decision_path = "VCF"
    diag.chosen_move = move
    diag.notes.append("VCF found forced win at depth 12")
    diag.search_stats = SearchStats(
        depth_reached=12,
        nodes_searched=5000,
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
| **ThreatBrain** | `"VCF"`, `"PNS"`, `"VCT"`, `"VCF-defence"`, `"Alpha-Beta"` | Threats, VCF/VCT results, PNS nodes, fallback depth |
| **AlphaBetaBrain** | `"opening_book"`, `"vcf_precheck"`, `"Alpha-Beta"` | TT size/hits, depth, nodes, aspiration resets, candidates |
| **MCTSBrain** | `"MCTS"` | Iterations, root visits |

---

## Reading Diagnostics

### Python

```python
import json
from pathlib import Path

lines = Path("recordings/gomoku_20260405_143012.diag.jsonl").read_text().splitlines()
header = json.loads(lines[0])
moves = [json.loads(line) for line in lines[1:] if not line.startswith('{"type":"summary"')]
summary = json.loads(lines[-1])

# Find moves where the brain fell back to alpha-beta
fallbacks = [m for m in moves if m["decision_path"] == "Alpha-Beta"]
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
```

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
