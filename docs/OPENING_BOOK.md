# Opening Book

Pre-computed moves for the first few plies of a game, stored either as
hardcoded entries or in a JSON file.  The opening book uses **8-fold symmetry
reduction** so that a single stored position covers all rotated and reflected
equivalents.

---

## Table of Contents

1. [Overview](#overview)
2. [Why an Opening Book?](#why-an-opening-book)
3. [8-Fold Symmetry](#8-fold-symmetry)
4. [Hardcoded Book](#hardcoded-book)
5. [File-Based Book](#file-based-book)
6. [Lookup Flow](#lookup-flow)
7. [Generating a Book](#generating-a-book)
8. [Configuration & Usage](#configuration--usage)

---

## Overview

| Property | Value |
|----------|-------|
| Symmetry | 8 transforms (4 rotations × 2 reflections) |
| Hardcoded entries | 3 positions (move 2–3 for standard 15×15) |
| File format | JSON (`OpeningBookFile`) |
| Lookup cost | O(8 × S) per position (S = number of stones) |
| Supported brains | `alphabeta` (file-based), all (hardcoded) |

---

## Why an Opening Book?

The opening phase of Gomoku is well-studied.  On a 15×15 board the first
player (Black) has a significant advantage and the optimal opening moves are
known.  Rather than spending search time re-deriving these moves every game,
the brain looks them up instantly.

Benefits:
- **Zero think-time** for book moves.
- **Optimal play** in studied positions.
- **Consistency** — avoids randomness in opening choice.

---

## 8-Fold Symmetry

A square Gomoku board is symmetric under **8 transforms**:

| Index | Transform | Description |
|-------|-----------|-------------|
| 0 | Identity | No change |
| 1 | Rotate 90° CW | `(r, c) → (c, m−r)` |
| 2 | Rotate 180° | `(r, c) → (m−r, m−c)` |
| 3 | Rotate 270° CW | `(r, c) → (m−c, r)` |
| 4 | Reflect horizontal | `(r, c) → (r, m−c)` |
| 5 | Reflect vertical | `(r, c) → (m−r, c)` |
| 6 | Reflect main diagonal | `(r, c) → (c, r)` |
| 7 | Reflect anti-diagonal | `(r, c) → (m−c, m−r)` |

Where `m = board_size − 1`.

### Canonicalisation

Any position is converted to its **canonical form** by applying all 8
transforms and picking the **lexicographically smallest** one:

```python
def _canonicalize(stones, size):
    best_key = None
    best_t = 0
    for t in range(8):
        key = tuple(sorted(
            (stone_val, *_transform(r, c, size, t))
            for stone_val, r, c in stones
        ))
        if best_key is None or key < best_key:
            best_key = key
            best_t = t
    return best_key, best_t
```

This means a single book entry covers all 8 orientations.

### Inverse mapping

When a book move is found in the canonical frame, it must be mapped back to
the original board orientation.  Each transform has an inverse:

| Transform | Inverse |
|-----------|---------|
| 0 (identity) | 0 |
| 1 (90° CW) | 3 (270° CW) |
| 2 (180°) | 2 (180°) |
| 3 (270° CW) | 1 (90° CW) |
| 4 (horiz flip) | 4 |
| 5 (vert flip) | 5 |
| 6 (diag) | 6 |
| 7 (anti-diag) | 7 |

---

## Hardcoded Book

The hardcoded book lives in `gomoku_brain.opening` and covers the first 2–3
moves of standard 15×15 freestyle Gomoku.

### Registered positions

| Position | Stones on board | Recommended move | Rationale |
|----------|----------------|------------------|-----------|
| Move 2 | Black at center (H8) | `G7` (diagonal) | Main-line diagonal response |
| Move 3a | Black H8, White G7 | `I9` (extend diagonal) | Build connected diagonal |
| Move 3b | Black H8, White G8 (orthogonal) | `I8` (extend other way) | Mirror the extension |

All entries are stored in canonical form via `_add_book()` and looked up via
`lookup_opening_book()`.

### Limitations

- Only 15×15 board.
- Only covers ≤ 4 occupied cells (exits book early).
- **No** deep-searched evaluations (entries are hand-picked main-line moves).

---

## File-Based Book

The file-based book (`gomoku_brain.opening_book`) extends the hardcoded book
with positions evaluated by a brain (typically alpha-beta at high depth).

### JSON schema

```json
{
  "format_version": 1,
  "board_size": 15,
  "generator_brain": "alphabeta",
  "generator_settings": { "depth": 6 },
  "search_depth": 6,
  "max_positions": 1000,
  "error_threshold": 0.0,
  "positions": {
    "[[1,7,7]]": {
      "move": [6, 6],
      "score": 150.0,
      "depth": 6
    }
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | `int` | Schema version (currently 1) |
| `board_size` | `int` | Board size the book was generated for |
| `generator_brain` | `str` | Brain used to evaluate positions |
| `generator_settings` | `dict` | Settings of the generator brain |
| `search_depth` | `int` | Search depth used during generation |
| `max_positions` | `int` | Maximum positions evaluated |
| `error_threshold` | `float` | Score threshold for including entries |
| `positions` | `dict` | Canonical key → `BookEntry` |

### BookEntry

```python
class BookEntry(BaseModel):
    move: tuple[int, int]   # best move in canonical frame
    score: float            # minimax evaluation score
    depth: int              # search depth used
```

---

## Lookup Flow

```
get_move() called
│
├── File-based book loaded? ──yes──▶ lookup_file_book()
│       ├── Canonicalise current position
│       ├── Look up in book.positions
│       ├── Map canonical move back via inverse transform
│       └── Sanity-check cell is empty → return Move or None
│
├── (file book miss) ──▶ lookup_opening_book()  (hardcoded)
│       ├── Board size ≠ 15 or > 4 stones → None
│       ├── Canonicalise current position
│       ├── Look up in _BOOK_15 dict
│       ├── Map back via inverse transform
│       └── Sanity-check → return Move or None
│
└── (both miss) ──▶ proceed to normal search
```

### Candidate deduplication

In the opening phase, `opening.deduplicate_by_symmetry()` removes
symmetry-equivalent candidate moves.  It detects the board's symmetry
subgroup (e.g., an empty board has all 8 symmetries; a board with one stone
at center has 8; a board with off-center stones may have only 1) and keeps
one representative per equivalence class.

---

## Generating a Book

A file-based opening book can be generated by running alpha-beta at high depth
on early-game positions and collecting the results.  The typical workflow:

1. Enumerate early-game positions (e.g., all 1-stone and 2-stone setups).
2. For each position, run `AlphaBetaBrain(depth=6+)` and record the best
   move and score.
3. Canonicalise and store in an `OpeningBookFile`.
4. Save to JSON.

```python
from gomoku_brain.opening_book import OpeningBookFile, BookEntry
from gomoku_brain.opening import _canonicalize

book = OpeningBookFile(
    board_size=15,
    generator_brain="alphabeta",
    generator_settings={"depth": 6},
    search_depth=6,
)

# After evaluating a position:
stones = {(1, 7, 7)}  # Black at center
canon, t = _canonicalize(stones, 15)
key = json.dumps(canon)
book.positions[key] = BookEntry(move=(6, 6), score=150.0, depth=6)

book.save(Path("opening_book.json"))
```

---

## Configuration & Usage

### CLI

```bash
# Use a file-based opening book
uv run gomoku play --black alphabeta --white threat --opening-book books/15x15.json
```

The `--opening-book` flag injects the book path into all `alphabeta`-type
brains participating in the game.

### gomoku.toml

```toml
[brains.alphabeta]
type = "alphabeta"
opening_book = "books/15x15.json"
```

### Python API

```python
from gomoku_brain.alphabeta import AlphaBetaBrain

# File-based book
brain = AlphaBetaBrain(depth=4, opening_book="books/15x15.json")

# Hardcoded book (used automatically by all brains that call
# lookup_opening_book() — currently alphabeta does this as a fallback
# when no file book is loaded or the file book has no match)
from gomoku_brain.opening import lookup_opening_book
move = lookup_opening_book(board, stone)  # Move or None
```
