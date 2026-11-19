# Threat Model

How the Gomoku AI recognises, classifies, and exploits threats — moves the
opponent **must** answer or lose.

---

## Table of Contents

1. [What Is a Threat?](#what-is-a-threat)
2. [Threat Classification](#threat-classification)
3. [Pattern Scoring Table](#pattern-scoring-table)
4. [Four-Threats (Used by VCF)](#four-threats-used-by-vcf)
5. [Three-Threats (Used by VCT)](#three-threats-used-by-vct)
6. [Forks](#forks)
7. [Threat Detection Internals](#threat-detection-internals)
8. [Threat-Based Search Algorithms](#threat-based-search-algorithms)
9. [Defensive Considerations](#defensive-considerations)
10. [How Brains Use Threats](#how-brains-use-threats)

---

## What Is a Threat?

A **threat** is a move that forces the opponent to respond or face an
immediate loss.  By chaining threats, a player can maintain the initiative
through a sequence where the opponent has no choice but to keep blocking — 
eventually leading to an unblockable position.

Threats are the foundation of deep tactical play in Gomoku.  While general
search (alpha-beta) evaluates all candidate moves to a limited depth, threat
search explores only forcing lines and can therefore look 10–20 plies ahead.

---

## Threat Classification

### By severity

| Level | Name | Description | Defender options |
|-------|------|-------------|------------------|
| **5** | Five | 5-in-a-row — game over | None (game is won) |
| **4a** | Live-four | 4-in-a-row open at both ends | **None** — unblockable |
| **4b** | Dead-four | 4-in-a-row open at one end | **1** forced block |
| **3a** | Live-three | 3-in-a-row open at both ends | **2+** responses (block, counter) |
| **3b** | Dead-three | 3-in-a-row open at one end | Optional (not immediately fatal) |
| **2a** | Live-two | 2-in-a-row open at both ends | Low urgency |
| **2b** | Dead-two | 2-in-a-row open at one end | Minimal threat |

### By forcing power

```
         Forcing ──────────────────────────── Non-forcing

Five > Live-4 > Dead-4 > Live-3 > Dead-3 > Live-2 > Dead-2
 ↑       ↑        ↑        ↑
 Win   Unblock-  Must     Should
       able      block    respond
```

---

## Pattern Scoring Table

The evaluation function scores patterns by `(consecutive_count, open_ends)`:

| Pattern | Example | Score | Role |
|---------|---------|-------|------|
| `(5, *)` — five | `X X X X X` | 100,000 | Win |
| `(4, 2)` — live-four | `. X X X X .` | 10,000 | Unblockable |
| `(4, 1)` — dead-four | `O X X X X .` | 1,000 | Must block |
| `(3, 2)` — live-three | `. X X X .` | 1,000 | Urgent threat |
| `(3, 1)` — dead-three | `O X X X .` | 100 | Potential |
| `(2, 2)` — live-two | `. X X .` | 100 | Development |
| `(2, 1)` — dead-two | `O X X .` | 10 | Minor |
| `(1, 2)` — live-one | `. X .` | 10 | Seed |
| `(1, 1)` — dead-one | `O X .` | 1 | Negligible |

> `X` = own stone, `O` = opponent stone, `.` = empty

### Aggregation rules

- Patterns are scanned in **four directions** (horizontal, vertical, two
  diagonals) from each occupied cell.
- Own patterns are summed normally, opponent patterns are subtracted
  symmetrically (1.0×).
- A per-side **open-three bonus** (`_OPEN_THREE_BONUS = 1,500`) is added for
  each distinct open-three formation, and a per-cell **fork bonus** rewards
  empty cells where multiple threat lines converge.
- The total is **clamped** to ±99,000 to stay below the 100,000 win constant.

---

## Four-Threats (Used by VCF)

A four-threat is a move that creates **four** stones in a line.

### Dead four (1 open end)

```
Before:    . . X X X . .
After ★:   . . X X X ★ .        ← 4 consecutive, 1 open end
Defence:   . . X X X ★ ◆        ← opponent MUST block at ◆
```

The defender has exactly **one** forced reply.  VCF exploits this: since
the defender has no choice, the attacker can plan the next threat immediately.

### Live four (2 open ends)

```
Before:    . . X X X . .
After ★:   . ★ X X X . .        ← 4 consecutive, 2 open ends
```

**Unblockable** — the opponent cannot block both ends in one move.
This is an immediate win.

### Gap four

```
Before:    . X X . X . .
After ★:   . X X ★ X . .        ← fills the gap → 4 in a row
```

Detected by `_find_four_threats()`.  Note that a fully symmetric `X X _ X X`
gap, once filled, yields **five** in a row and is reported as an immediate win
(`total == 5`), not as a four-threat.

---

## Three-Threats (Used by VCT)

A three-threat is a move that creates an **open three** (both ends empty).

```
Before:    . . X X . . .
After ★:   . . X X ★ . .        ← 3 consecutive, 2 open ends (live-three)

Defences:  ◇ . X X ★ . ◇        ← opponent can block at either ◇
```

The defender has **at least two** reasonable responses:
1. Block at one end.
2. Block at the other end.
3. Play a **counter-threat** — create their own four-threat to seize the
   initiative.

This expanded defence set is what makes VCT broader (and slower) than VCF.

---

## Forks

A **fork** is a single move that creates two or more independent threat lines.

```
    . . . . .
    . . X . .         ★ creates:
    . . ★ . .         • horizontal threat: X ★ ...
    . X . . .         • diagonal threat:  X ★ ...
    . . . . .
```

The opponent can only block one line per turn, so a fork is often decisive.

### Fork detection

The evaluation function (`evaluate.py`) counts how many distinct threat lines
(scoring ≥ `_FORK_THRESHOLD = 100`, i.e. at least a live-two) converge on
each empty cell.  Each additional line beyond the first earns a
`_FORK_BONUS = 3,000` points.

---

## Threat Detection Internals

### `_find_four_threats(board, stone)`

Scans all candidate moves for the given stone and returns:

```python
list[tuple[Move, Move | None]]
#           ↑        ↑
#      attack    defence (None if unblockable)
```

- If `defence` is `None`, the four-threat is unblockable (live-four or
  immediate five).
- Detects straight fours and gap fours (`XX_XX`).
- Uses `get_candidate_moves()` with radius 2.

### `_find_three_threats(board, stone)`

Returns:

```python
list[tuple[Move, list[Move]]]
#           ↑       ↑
#      attack    defence list (typically 2 end cells)
```

- Only live-threes (open at both ends) — dead-threes are not forcing enough.
- Defence list contains the open-end cells.

### `_line_cells(board, row, col, dr, dc, stone)`

Low-level helper that counts consecutive stones in both directions from a
position:

```python
(count, open_ends, end_cells)
```

Used by both four-threat and three-threat detection.

---

## Threat-Based Search Algorithms

The Gomoku workspace implements three threat search algorithms:

| Algorithm | Threat types | Branching | Depth | Speed |
|-----------|-------------|-----------|-------|-------|
| **VCF** | Four-threats only | ~1 (defender), ~2–5 (attacker) | 20 plies | Very fast |
| **VCT** | Four + three-threats | ~2–5 (defender), ~5–10 (attacker) | 12 plies | Moderate |
| **PNS** | Four + three-threats | ~15 (defender), ~5–10 (attacker) | Budget-bounded | Moderate |

See [THREAT_BRAIN.md](../packages/gomoku-brain/docs/THREAT_BRAIN.md) for the
full algorithm descriptions, pseudocode, and worked examples.

---

## Defensive Considerations

### Direct blocks

The simplest defence: place a stone at the single forced-block cell of a
dead-four.

### Counter-threats

A more advanced defence: instead of blocking directly, play a **counter-four**
that forces the attacker to respond, disrupting their sequence.

```
Attacker plays three-threat at A5.
Defender ignores A5, plays a four-threat at K8.
→ Attacker MUST block K8 (or lose).
→ A5 is no longer part of a clean forcing sequence.
```

VCT and PNS both account for counter-threats in their defence enumeration.
The original VCT implementation only considered direct end-cell blocks and was
vulnerable to counter-threats (the "K8 bug" — see THREAT_BRAIN.md for the
full story).

### VCF-aware positional defence

When the opponent has a VCF (forced win through fours), the Threat Brain
tries every candidate move and re-checks whether the opponent's VCF still
exists on the resulting board.  Moves that break the VCF are called
**VCF-breakers**; those that stay breakers after any forced reply are
*durable* and preferred.  The breakers are then probed, best positional
evaluation first, for any other forced win of the opponent's (one that
uses threes too).  The first one that leaves the opponent no forced win is
played, which combines tactical safety with strategic value.  The same
probing answers an opponent's PNS/VCT win.

---

## How Brains Use Threats

| Brain | Threat usage |
|-------|-------------|
| **ThreatBrain** | Entire decision pipeline is threat-driven (own VCF → own PNS/VCT → defend opponent VCF → defend opponent PNS/VCT → block forks → AB fallback) |
| **AlphaBetaBrain** | Root VCF pre-check; threat-ordered moves; quiescence extends four-threats; threat extensions add depth on forcing lines |
| **MCTSBrain** | VCF safety gate restricts root moves when opponent has VCF; heuristic priors score threat patterns |
| **Evaluation** | All brains use pattern scores + fork detection for positional assessment |
