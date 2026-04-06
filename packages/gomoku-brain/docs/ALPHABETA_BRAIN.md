# Alpha-Beta Brain — Negamax with Advanced Search Techniques

A **general-purpose** Gomoku brain that uses negamax with alpha-beta pruning,
enhanced with eight improvements: Zobrist transposition table, iterative
deepening, quiescence search, principal variation search, history heuristic,
aspiration windows, futility pruning, and evaluation caching.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Zobrist Hashing & Transposition Table](#zobrist-hashing--transposition-table)
4. [Iterative Deepening](#iterative-deepening)
5. [Move Ordering](#move-ordering)
6. [Principal Variation Search](#principal-variation-search)
7. [Quiescence Search](#quiescence-search)
8. [Aspiration Windows](#aspiration-windows)
9. [Futility Pruning](#futility-pruning)
10. [Evaluation Cache](#evaluation-cache)
11. [Parameters](#parameters)
12. [Example Scenarios](#example-scenarios)
13. [Configuration & Usage](#configuration--usage)
14. [Brain Comparison](#brain-comparison)

---

## Overview

| Property        | Value                                             |
| --------------- | ------------------------------------------------- |
| Brain name      | `alphabeta`                                       |
| Algorithm       | Negamax + alpha-beta + 8 enhancements             |
| Default depth   | 4                                                 |
| Strength        | Strong positional / general play                  |
| Best at         | Mid-game positions without clear forcing sequences |

The brain evaluates all candidate moves up to a fixed depth using pattern-based
evaluation with fork detection.  The eight enhancements reduce the number of
nodes searched by orders of magnitude compared to plain negamax.

---

## Architecture

```
get_move()
│
├── Create shared state: TT, history table, eval cache
│
└── Iterative deepening: d = 1 .. max_depth
    │
    ├── Aspiration window (narrow α/β from previous score)
    │   └── Re-search on fail-low / fail-high
    │
    └── _alphabeta(board, depth, α, β, stone, ...)
        │
        ├── TT probe → exact/bound cutoff or best-move hint
        │
        ├── Candidate generation (radius-2 of occupied)
        │
        ├── Urgent move fast path (instant win / forced block)
        │
        ├── Move ordering (depth ≥ 3):
        │   TT move → killers → eval + history sorted
        │
        ├── For each candidate (capped at 20):
        │   │
        │   ├── Terminal check (five-in-a-row)
        │   │
        │   ├── Futility pruning (depth = 2, non-first moves)
        │   │
        │   ├── depth ≤ 1 → quiescence search
        │   │
        │   └── depth > 1 → PVS:
        │       ├── First move: full window [-β, -α]
        │       └── Rest: zero-window scout [-α-1, -α]
        │           └── Re-search on fail-high
        │
        ├── Killer + history update on β-cutoff
        │
        └── TT store (EXACT / LOWER / UPPER)
```

---

## Zobrist Hashing & Transposition Table

### Zobrist hashing

Each (row, col, stone) triple has a pre-computed random 64-bit key.  A special
`SIDE_KEY` distinguishes Black-to-move from White-to-move positions.

The full board hash is the XOR of all placed stones' keys, XORed with
`SIDE_KEY` if White is to move.  After placing a stone, the hash is updated
incrementally in O(1):

```python
new_hash = old_hash ^ TABLE[row][col][stone_index] ^ SIDE_KEY
```

The random table is seeded deterministically (`0x60_0C0E_2026`) so hashes are
stable within a process.

### Transposition table

A dict-based transposition table stores previously searched positions:

| Field       | Description                                  |
| ----------- | -------------------------------------------- |
| `key`       | Full Zobrist hash (collision detection)      |
| `depth`     | Search depth of the stored result            |
| `score`     | Evaluation score                             |
| `flag`      | Bound type: `EXACT`, `LOWER`, or `UPPER`     |
| `best_move` | Best move found at this position             |

**Replacement policy:** depth-preferred — an entry is only overwritten if the
new search is at equal or greater depth.

**Probe logic:** If the stored depth ≥ current depth:

- `EXACT` → return immediately (no search needed).
- `LOWER` → tighten α (`alpha = max(alpha, stored_score)`).
- `UPPER` → tighten β (`beta = min(beta, stored_score)`).
- If α ≥ β after tightening → cutoff.

Even when the stored depth is too shallow for a cutoff, the `best_move` is
extracted for move ordering.

---

## Iterative Deepening

Instead of searching directly at the target depth, the brain searches at
increasing depths 1, 2, ..., `max_depth`:

```
d=1:  fast, seeds TT with best-move info
d=2:  uses TT from d=1, better ordering
d=3:  uses TT from d=2, even better ordering
d=4:  final search, excellent move ordering from accumulated TT
```

### Properties

- The TT, history table, and eval cache are **shared** across all iterations,
  so information accumulates.
- Killer tables are **reset** each iteration because they are depth-relative.
- An **early exit** triggers if a proven win (score ≥ 100,000) is found.
- The overhead of re-searching shallower depths is small because each depth
  level expands the tree exponentially — d=1 through d=3 combined cost much
  less than d=4 alone.

---

## Move Ordering

Good move ordering is critical for alpha-beta efficiency.  Moves are sorted
in priority order:

| Priority | Source           | Rationale                                     |
| -------- | ---------------- | --------------------------------------------- |
| 1        | TT best-move    | Best move from a previous (shallower) search  |
| 2        | Killer moves     | Moves that caused β-cutoff at this depth      |
| 3        | Eval + history   | `evaluate_board()` + history heuristic score  |

Move ordering is applied at depth ≥ 3 (at shallower depths the overhead of
evaluating every candidate outweighs the benefit).

### Killer heuristic

When a move causes a beta-cutoff, it is stored as a "killer" at that depth.
Two slots per depth (LRU replacement).  Killers are tried right after the TT
move because they recently worked well at the same depth.

### History heuristic

A shared table `(stone, row, col) → cumulative_score` tracks how often each
move causes cutoffs.  On every beta-cutoff, the move's score is increased by
`depth²` (deeper cutoffs are more valuable).  The history score is added to
the evaluation score during move sorting.

The history table is shared across iterative deepening iterations, so moves
that consistently cause cutoffs at multiple depths accumulate high scores.

---

## Principal Variation Search

PVS assumes the first move (after ordering) is the best.  It searches with
three strategies:

| Move index | Window           | Rationale                           |
| ---------- | ---------------- | ----------------------------------- |
| `i = 0`   | `[-β, -α]`      | Full window — find true score       |
| `i > 0`   | `[-α-1, -α]`    | Zero-window scout — prove ≤ α      |
| Re-search  | `[-β, -score]`  | Scout failed high → full re-search  |

The zero-window search is much cheaper because it only needs to prove the move
is *not* better than alpha.  Most of the time, the first move *is* the best,
and all scouts confirm this quickly.

---

## Quiescence Search

At leaf nodes (depth ≤ 1), instead of returning a static evaluation, the brain
extends the search for **forcing moves** — specifically four-threats (creating
or blocking four-in-a-row):

```
_quiesce(board, α, β, stone, qs_depth):
    stand_pat = evaluate_board(board, stone)
    if stand_pat ≥ β: return stand_pat     # Position already good enough
    α = max(α, stand_pat)

    if qs_depth ≤ 0: return stand_pat      # Maximum extension reached

    for each four-threat move:
        score = -_quiesce(child, -β, -α, opponent, qs_depth - 1)
        update α

    return α
```

### Properties

| Property       | Value                                        |
| -------------- | -------------------------------------------- |
| Max extra depth| 4 plies (`_MAX_QS_DEPTH`)                    |
| Move types     | Four-threats only (attack + defence)         |
| Stand-pat      | Static eval as lower bound                   |
| Purpose        | Avoid horizon effect for tactical sequences  |

This prevents the brain from blindly cutting off the search when an unresolved
four-in-a-row threat exists just beyond the horizon.

---

## Aspiration Windows

After the first iterative-deepening iteration, subsequent iterations search
with a narrow window `[prev_score - 50, prev_score + 50]` instead of
`[-∞, +∞]`.

| Outcome        | Action                                         |
| -------------- | ---------------------------------------------- |
| Score in window | Accept — search was efficient                  |
| Fail-low/high  | Re-search with full `[-∞, +∞]` window          |

The narrow window causes more cutoffs during the search.  The risk of
needing a re-search is low because scores rarely shift dramatically between
adjacent depths.

Aspiration windows are **skipped** when the previous score indicates a
near-terminal position (`|prev_score| ≥ 90,000`) to avoid misguided narrow
windows near wins/losses.

---

## Futility Pruning

At **depth 2** (one ply above leaf nodes), if the static evaluation plus a
margin cannot reach alpha, non-first moves are skipped:

```
if depth == 2 and i > 0:
    if static_eval + FUTILITY_MARGIN ≤ α:
        skip this move
```

| Property  | Value                                              |
| --------- | -------------------------------------------------- |
| Depth     | Applied at depth 2 only                            |
| Margin    | 1,500 (`_FUTILITY_MARGIN`)                         |
| Exception | The first move is always searched (safety)          |
| Risk      | May miss deep tactical shots with huge swings       |

The margin of 1,500 is conservative — it roughly covers a live-three plus a
fork bonus.  Moves with even larger evaluation swings are rare and typically
caught by the quiescence search.

---

## Evaluation Cache

A dict-based cache `zobrist_hash → score` avoids redundant calls to
`evaluate_board()`.  The cache is shared across all iterative deepening
iterations and used in:

1. **Quiescence search** — standing-pat evaluation.
2. **Futility pruning** — lazy static eval computation.

### Perspective correctness

The eval cache stores scores keyed by Zobrist hash, which encodes the
side-to-move.  This means `evaluate_board(board, BLACK)` and
`evaluate_board(board, WHITE)` produce different hashes and are stored
separately.

> **Note:** The eval cache is intentionally **not** used in `_order_moves()`
> to avoid a perspective mismatch: ordering evaluates positions from the
> current player's view but hashes there encode the *opponent's* turn (after
> an incremental update).  Using the cache in ordering would store
> current-player scores under opponent-to-move keys, poisoning later lookups
> in quiescence and futility pruning.

---

## Parameters

| Parameter | Default | Controls                     | Effect of increasing          |
| --------- | ------- | ---------------------------- | ----------------------------- |
| `depth`   | 4       | Maximum search depth (plies) | Stronger but exponentially slower |

### Internal constants

| Constant            | Value  | Purpose                                     |
| ------------------- | ------ | ------------------------------------------- |
| `_KILLER_SLOTS`     | 2      | Killer moves per depth level                |
| `_MAX_CANDIDATES`   | 20     | Branching factor cap                        |
| `_MAX_QS_DEPTH`     | 4      | Extra plies in quiescence                   |
| `_ASPIRATION_WINDOW`| 50     | Half-width of aspiration window             |
| `_FUTILITY_MARGIN`  | 1,500  | Futility pruning threshold                  |

### Tuning guidelines

| Scenario                  | Recommendation                                    |
| ------------------------- | ------------------------------------------------- |
| Fast play / low resources | `depth=2` — uses quiescence for tactical safety   |
| Default / balanced        | `depth=4` — good balance of speed and strength    |
| Maximum strength          | `depth=6` — significantly deeper, much slower     |

> **Note:** Each additional depth level roughly multiplies search time by the
> effective branching factor (~8–15 with pruning).  The improvements (TT,
> move ordering, PVS, futility) keep this manageable up to depth 5-6.

---

## Example Scenarios

### Example 1 — TT-seeded move ordering avoids tactical blunder

```
Iterative deepening d=1..4:

d=1:  best_move = (7,7),  score = +120    TT entry stored
d=2:  TT suggests (7,7) first → searched first → confirms
d=3:  TT suggests (7,7) first → deeper search finds (7,8) better
d=4:  TT suggests (7,8) first → all other moves pruned quickly
      → 73% of nodes pruned compared to plain alpha-beta
```

### Example 2 — Quiescence prevents horizon effect

```
Board (Black to move, depth=4 search ends here):

  Row 7:  . . X X X O .     Black has 3-in-a-row

Plain alpha-beta at depth 0:
  → static eval = +800 (looks good for Black)

With quiescence:
  → extends: White blocks at (7,5) → +200
  → or: Black plays (7,5) → five → +100,000
  → correct evaluation reveals the tactical opportunity
```

### Example 3 — Aspiration window saves time

```
d=3 score: +150

d=4 aspiration window: [100, 200]
  → Search returns +170 (within window) ✓
  → Saved ~40% of nodes vs. full-width search

d=4 aspiration window: [100, 200]
  → Search returns +80 (fail-low)
  → Re-search with [-∞, +∞] → true score is +80
  → Small cost for the re-search, but rare
```

### Example 4 — PVS scout confirms best move cheaply

```
Move ordering after TT + killers:
  Move 1: (7,8)  → full window search → score = +250
  Move 2: (6,7)  → scout [-251, -250] → returns -280 → ≤ α ✓ (pruned)
  Move 3: (8,8)  → scout [-251, -250] → returns -190 → > α → re-search!
                  → full window → score = +190 → doesn't beat +250
  Move 4-20: scouts all confirm ≤ α quickly

  Total: 1 full search + 1 re-search + 18 cheap scouts
```

### Example 5 — Futility pruning skips hopeless moves

```
Depth 2, α = +300, static eval = -1500

eval + FUTILITY_MARGIN = -1500 + 1500 = 0 < 300 (= α)

  Move 1: (4,4) → searched fully (i=0, always searched)
  Move 2-20: all skipped — position is too far behind
```

---

## Configuration & Usage

```python
from gomoku_brain.alphabeta import AlphaBetaBrain

brain = AlphaBetaBrain(depth=4)
move = brain.get_move(board, stone)
```

CLI usage:

```bash
uv run gomoku-cli play --black alphabeta --white threat --depth 4
```

---

## Brain Comparison

| Aspect              | Alpha-Beta             | Threat                       |
| ------------------- | ---------------------- | ---------------------------- |
| Search type         | General (all moves)    | Threat-focused (forcing only)|
| Depth               | 4 plies typical        | 20 plies (VCF), 12 (VCT)    |
| Branching factor    | ~8-15 (after pruning)  | ~2-5 (threats only)          |
| Strength: opening   | Good                   | Falls back to alpha-beta     |
| Strength: midgame   | Good                   | Good (threat sequences)      |
| Strength: endgame   | May miss forced wins   | Excellent (deep VCF/VCT)    |
| Key advantage       | Positional awareness   | Tactical depth               |
| Key weakness        | Limited depth           | Blind to non-forcing moves   |
| Used by threat brain| —                      | As Phase 4 fallback          |
