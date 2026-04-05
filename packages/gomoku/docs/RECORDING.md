# Game Recording & Playback

Record full Gomoku games to JSON files and replay them interactively in the
terminal with arrow-key navigation.

---

## Table of Contents

1. [Overview](#overview)
2. [Recording a Game](#recording-a-game)
3. [File Format](#file-format)
4. [Playback](#playback)
5. [Python API](#python-api)
6. [Configuration](#configuration)

---

## Overview

| Feature    | Description                                    |
| ---------- | ---------------------------------------------- |
| Format     | JSON (one file per game)                       |
| Location   | `recordings/` directory (configurable)         |
| Naming     | `gomoku_YYYYMMDD_HHMMSS.json`                 |
| Content    | Board config, players, every move with timing, result |
| Writing    | Incremental — file created on game start, updated after each move |
| Integrity  | `.md5sum` checksum file written on game completion |
| Playback   | Arrow-key navigation, move-by-move board view  |

---

## Recording a Game

### CLI

Pass `--record` to the `play` command:

```bash
# Record the game (saves to recordings/)
uv run gomoku play --black threat --white alphabeta --record

# Record to a custom directory
uv run gomoku play --black threat --white random --record --record-dir my_games/
```

The recording file is created immediately when the game starts and updated
after every move, so partial data survives crashes.  When the game ends the
final result is written and a companion `.md5sum` checksum is produced:

```
Game recorded: recordings/gomoku_20260405_143012.json
```

### Always-on recording

Set `enabled = true` in `gomoku.toml` to record every game without `--record`:

```toml
[recording]
enabled = true
directory = "recordings"
```

---

## File Format

Each recording is a JSON file with the following structure:

```json
{
  "format_version": 1,
  "started_at": "2026-04-05T12:06:11.123456+00:00",
  "ended_at": "2026-04-05T12:06:30.654321+00:00",
  "game_config": {
    "board_size": 15,
    "win_length": 5
  },
  "players": [
    {
      "stone": 1,
      "provider": "threat",
      "settings": { "depth": 4, "vcf_depth": 20, "vct_depth": 12 }
    },
    {
      "stone": 2,
      "provider": "alphabeta",
      "settings": { "depth": 4 }
    }
  ],
  "moves": [
    {
      "number": 1,
      "stone": 1,
      "move": { "row": 7, "col": 7 },
      "timestamp": "2026-04-05T12:06:11.200000+00:00",
      "think_time_ms": 76
    }
  ],
  "result": {
    "winner": 1,
    "is_draw": false,
    "winning_moves": [
      { "row": 7, "col": 7 },
      { "row": 7, "col": 8 },
      { "row": 7, "col": 9 },
      { "row": 7, "col": 10 },
      { "row": 7, "col": 11 }
    ]
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | `int` | Schema version (currently `1`) |
| `started_at` | `datetime` | UTC timestamp when the game began |
| `ended_at` | `datetime\|null` | UTC timestamp when the game ended (`null` while in progress) |
| `game_config` | `object` | Board size and win length |
| `players` | `array` | One entry per player with stone colour, brain name, and settings |
| `moves` | `array` | Ordered list of every move |
| `moves[].number` | `int` | 1-based move number |
| `moves[].stone` | `int` | `1` = Black, `2` = White |
| `moves[].move` | `object` | `{row, col}` — zero-indexed board coordinates |
| `moves[].timestamp` | `datetime` | UTC time the move was played |
| `moves[].think_time_ms` | `int` | Milliseconds the brain spent thinking |
| `result` | `object\|null` | Game outcome (`null` while in progress) |
| `result.winner` | `int\|null` | Winning stone (`1`/`2`) or `null` for draw |
| `result.is_draw` | `bool` | Whether the game ended in a draw |
| `result.winning_moves` | `array` | The five cells forming the winning line |

---

## Playback

### CLI

```bash
# Replay from the start
uv run gomoku playback recordings/gomoku_20260405_120611.json

# Start from a specific move
uv run gomoku playback recordings/gomoku_20260405_120611.json --from-move 20
```

### Controls

| Key | Action |
|-----|--------|
| `→` (right arrow) | Advance to the next move |
| `←` (left arrow) | Go back to the previous move |
| `q` | Quit playback |

### Display

At each step the terminal shows:

- The board state with the last move highlighted
- Move number, stone colour, cell, and think time
- At the final move: the game result and winning line

---

## Python API

### Recording

```python
from pathlib import Path
from gomoku.models import GameConfig, Stone
from gomoku.recorder import GameRecorder, PlayerInfo

# Set up recorder — file is created immediately in the given directory
config = GameConfig(board_size=15, win_length=5)
players = [
    PlayerInfo(stone=Stone.BLACK, provider="threat", settings={"depth": 4}),
    PlayerInfo(stone=Stone.WHITE, provider="alphabeta", settings={"depth": 4}),
]
recorder = GameRecorder(game_config=config, players=players, directory=Path("recordings"))
print(recorder.path)  # recordings/gomoku_20260405_143012.json (exists on disk)

# Record moves during the game — file is rewritten after each move
recorder.record_move(Stone.BLACK, move)
recorder.record_move(Stone.WHITE, move)
# ...

# Finalize: writes result + ended_at, produces .md5sum checksum file
record = recorder.finalize(result)
# recordings/gomoku_20260405_143012.json      ← final JSON
# recordings/gomoku_20260405_143012.md5sum     ← MD5 checksum
```

> **Backward compatibility:** `GameRecorder` still works without a `directory`
> argument — moves are accumulated in memory and `finalize()` returns the
> `GameRecord` without any file I/O.  The standalone `save_record()` function
> also remains available for one-shot writes.

### Loading

```python
from pathlib import Path
from gomoku.recorder import load_record

record = load_record(Path("recordings/gomoku_20260405_120611.json"))
print(f"Moves: {len(record.moves)}")
print(f"Winner: {record.result.winner}")
```

### Playback (programmatic)

```python
from gomoku.recorder import load_record
from gomoku_cli.playback import run_playback

record = load_record(Path("recordings/gomoku_20260405_120611.json"))
run_playback(record, from_move=1)
```

---

## Configuration

### gomoku.toml

```toml
[recording]
enabled = false        # Set to true to record every game
directory = "recordings"
```

### Settings model

```python
class RecordingSettings(BaseModel):
    enabled: bool = False
    directory: Path = Path("recordings")
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--record` / `--no-record` | Override the config file setting |
| `--record-dir PATH` | Override the recording directory |
