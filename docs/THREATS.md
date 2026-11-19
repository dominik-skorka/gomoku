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

In the diagrams below `X` is the attacker, `O` the defender, `.` empty,
`★` the move discussed and `◆` / `◇` the defender's replies.

---

## Threat Classification

### By severity

| Level | Name | What it threatens | Defender options |
|-------|------|-------------------|------------------|
| **5** | Five | 5 (or more) in a row — game over | None |
| **4a** | Live four | **Two** cells that each complete five (`. X X X X .`) | **None** — unblockable |
| **4a** | Double four | Two fours made by one move, on two lines | **None** — unblockable |
| **4b** | Four | **One** cell that completes five: straight (`O X X X X .`) or split (`X X X . X`, `X X . X X`) | **1** forced block |
| **3a** | Three | One more stone makes a live four (`. X X X .`; split: `. X X . X .`) | **2–3** blocks, or a counter-four |
| **3b** | Dead three | Three that can only become a blockable four | Optional |
| **2a** | Live two | `. X X .` | Low urgency |
| **2b** | Dead two | `O X X .` | Minimal |

The threat searches recognise fours by what they threaten — a cell that
completes five — so split fours count exactly like straight ones.  Split
threes count only where `split_threes` is on (see
[Three-Threats](#three-threats-used-by-vct)).  Which threats each search
plays and answers is configurable — see
[Configuring the threat spaces](#configuring-the-threat-spaces).

### By forcing power

```
         Forcing ──────────────────────────── Non-forcing

Five > Live-4 / double-4 > Four > Three > Dead-3 > Live-2 > Dead-2
 ↑            ↑              ↑       ↑
 Win     Unblockable       Must    Should
                           block   respond
```

---

## Pattern Scoring Table

The static evaluation (`evaluate.py`) scores **unbroken runs** by
`(consecutive_count, open_ends)`:

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

The evaluation is a heuristic and only sees runs: `X X . X X` scores as two
live twos, not as a four.  The threat searches below do not rely on it to
find threats.

### Aggregation rules

- Patterns are scanned in **four directions** (horizontal, vertical, two
  diagonals) from each occupied cell.
- Own patterns are summed, opponent patterns are subtracted with the same
  weight (1.0×).
- A per-side **open-three bonus** (`_OPEN_THREE_BONUS = 1,500`) is added for
  each distinct open-three formation, and a per-cell **fork bonus** rewards
  empty cells where multiple threat lines converge (see [Forks](#forks)).
- The total is **clamped** to ±99,000 to stay below the 100,000 win constant.

---

## Four-Threats (Used by VCF)

A **four-threat** is a move after which an empty cell would complete five
(or more) through it.  The first empty cell past each end of the unbroken
run through the move does when the run, that cell and the stones beyond it
add up to five.  One such cell is a blockable four; two (on one line or
two) make it unblockable.

### Straight four (one cell)

```
Before:    O X X X . .
After ★:   O X X X ★ .        ← four in a row, one end blocked
Defence:   O X X X ★ ◆        ← the only cell that stops five
```

The defender has exactly **one** forced reply.  VCF exploits this: since
the defender has no choice, the attacker can plan the next threat at once.

### Split four (one cell)

```
Before:    . X X . . X .
After ★:   . X X ★ . X .      ← X X X _ X: the gap completes five
Defence:   . X X ★ ◆ X .
```

Also `X X . X X` (the move makes either pair).  Split fours force exactly
like straight ones.  They are as much a defender's resource: a split
counter-four refutes an attack that ignores it.  (Before they were
detected, PNS "proved" wins that a split counter-four refutes.)

### Live four and double four (unblockable)

```
Before:    . . X X X . .
After ★:   . ★ X X X . .      ← both ends complete five
```

Two completion cells — both ends of `. X X X X .`, or fours on two lines
from the same move — cannot both be blocked: an immediate win.  Filling the
gap of `X X . X X` makes five outright.

---

## Three-Threats (Used by VCT)

A **three-threat** is a move after which one more stone makes a live four.

### Straight three (always detected)

```
Before:    . . X X . . .
After ★:   . . X X ★ . .        ← unbroken three, both ends empty
Defences:  . ◇ X X ★ ◇ .        ← block either end
```

### Split three (with `split_threes`)

```
Before:    . X X . . . .
After ★:   . X X . ★ . .        ← X X _ X: filling the gap makes a live four
Defences:  ◇ X X ◇ ★ ◇ .        ← the gap or either end
```

With `split_threes` a three is any move for which a window of four cells
through it holds three stones and one empty cell, with empty cells on both
sides of the window.  Its defences are the cells common to all such
windows — both ends of `. X X X .`; the gap and both ends of `. X X . X .`.
This definition also drops "threes" no move turns into a live four, such
as `O . X X X . O`.

`split_threes` is **off by default** in both `[brains.threat.threats.own]`
and `[brains.threat.threats.opponent]`.  On for both sides it widens
VCT/PNS a lot and cost 58 Elo at `pns_nodes = 30_000`; on for our attack
only it was neutral (+7 Elo, −16 to +30) at 1.2× the think time.  Proofs
are sound without it: a split three only threatens a live four, which the
defence enumeration already answers (see
[Defensive Considerations](#defensive-considerations)).

Besides blocking, the defender can answer any three with a
**counter-threat** — a four of their own that seizes the initiative.  This
expanded defence set is what makes VCT broader (and slower) than VCF.

---

## Forks

A **fork** is a single move that creates two or more threat lines at once.
The opponent can block only one line per turn, so a fork is often decisive.

```
    . . . . . .
    . . . X . .        ★ makes two threes at once:
    . . . X . .        • row:     X X ★
    . X X ★ . .        • column:  X X ★ (the two X above)
    . . . . . .
```

Forks are used in two places:

- **Evaluation fork bonus** (`_fork_score`): counts the lines through each
  empty cell whose unbroken neighbours score at least
  `_FORK_THRESHOLD = 100` (a live two); each line beyond the first earns
  `_FORK_BONUS = 3,000`.
- **Fork cells to block** (`find_fork_cells`, the threat brain's
  `block_fork` step): the opponent's cells where two or more such lines
  meet.  With `[brains.threat.threats.forks] split = true` a line also
  counts when the move makes a split three or four on it — two `X X _`
  shapes meeting at one cell, say.  Off by default: it cost 23 Elo (the
  brain blocked more and attacked less).

---

## Threat Detection Internals

All detectors run in Numba JIT kernels (`evaluate_jit.py`), with
pure-Python fallbacks for the JIT-disabled path.  The line rules live in
two shared helpers, mirrored in Python in `line_patterns.py`:

| JIT kernel / helper | Python | Answers |
|---------------------|--------|---------|
| `_five_cells_through` | `five_cells_through` | Along one line: the run through a cell and the cells (0–2) that would then complete five |
| `_three_defences` | `three_defences` | Along one line: the cells (0–3) that stop the three a move makes (split rules) |

### `_find_four_threats(board, stone, ctx=None, split=True)`

```python
list[tuple[Move, Move | None]]
#           ↑        ↑
#      attack    defence (None if unblockable)
```

- `defence` is the single completion cell; `None` means unblockable (live
  four, double four, or a five).
- Straight and split fours alike; `split=False` counts unbroken runs only
  (a `ThreatSpace` with `split_fours` / `split_counter_fours` off).
- Sorted strongest first (unblockable before blockable), so VCF tries wins
  and refutations early.
- Candidates are the empty cells within radius 2 of a stone; `ctx` is an
  optional precomputed scan context shared by several detectors.

Related checks: `has_unblockable_four_grid` (any unblockable four?) and
`makes_four_threat` / `has_any_four_threat_grid` (any four at all?) use
the same rules.  `is_four_threat` is different on purpose: an alpha-beta
heuristic for move ordering, threat extensions and quiescence that counts
unbroken runs of four, blocked or not.  Counting split fours there nearly
tripled quiescence's node count without a measurable gain.

### `_find_three_threats(board, stone, ctx=None, split=False)`

```python
list[tuple[Move, list[Move]]]
#           ↑       ↑
#      attack    defences (2 ends; with split, up to 3)
```

- One entry per `(attack, direction)`, so a cell that makes threes on two
  lines appears twice; entries are sorted by that fork degree.
- `split=False`: unbroken three with both ends empty; defences are the ends.
- `split=True`: the split rules above.

### `_line_cells(board, row, col, dr, dc, stone)`

Low-level Python helper that counts consecutive stones in both directions
from a cell: `(count, open_ends, end_cells)`.  Used by the straight-three
fallback.

---

## Threat-Based Search Algorithms

The Gomoku workspace implements three threat search algorithms:

| Algorithm | Threat types | Branching | Budget | Speed |
|-----------|-------------|-----------|--------|-------|
| **VCF** | Fours | ~1 (defender), ~2–5 (attacker) | 20 plies | Very fast |
| **VCT** | Fours + threes | ~2–5 (defender), ~5–10 (attacker) | 12 plies | Moderate |
| **PNS** | Fours + threes | ~15 (defender), ~5–10 (attacker) | 60,000 positions, 5 s (`gomoku.toml`) | Moderate |

See [THREAT_BRAIN.md](../packages/gomoku-brain/docs/THREAT_BRAIN.md) for the
full algorithm descriptions, pseudocode, and worked examples.

---

## Defensive Considerations

### Direct blocks

The simplest defence: place a stone on the single completion cell of a
four.

### Counter-threats

A more advanced defence: instead of blocking directly, play a **counter-four**
— straight or split — that forces the attacker to respond, disrupting their
sequence.

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

### What PNS lets the defender try

At every defender node PNS tries:

1. the block of each attacker four;
2. the defences of each attacker three;
3. the defender's own fours (counter-fours);
4. the defender's own threes (counter-threes);
5. every cell that defuses an attacker's threat to make an unblockable
   four next move — taking an end of a would-be four as well as its attack
   cell.

Point 5 also answers split threes when `split_threes` is off: such a three
threatens nothing but a live four.  A proof is only as good as this list —
each missing kind of defence (points 3 for split fours, and 5) once let PNS
prove wins the defender could refute.  Every point can be switched off in
`[brains.threat.threats.defence]`, at that price.

### Configuring the threat spaces

A `ThreatSpace` (`gomoku_brain/threat_space.py`) tells VCF, PNS and VCT
which threats the attacker plays and which replies the defender gets.  The
threat brain builds one for its own forced-win search and one for the
opponent's, from `gomoku.toml`:

| Table | Keys | Effect of turning fewer on |
|-------|------|----------------------------|
| `[brains.threat.threats.own]` | `fours`, `split_fours`, `threes`, `split_threes` | Our search finds fewer wins (never false ones) |
| `[brains.threat.threats.opponent]` | the same | The brain may miss a win the opponent has |
| `[brains.threat.threats.defence]` | `block_fours`, `block_threes`, `counter_fours`, `split_counter_fours`, `counter_threes`, `split_counter_threes`, `unblockable_four_defences` | **Unsound proofs**: wins that kind of reply refutes are "proved" (except `counter_threes` / `split_counter_threes`, heuristics) |
| `[brains.threat.threats.forks]` | `split` | Fork cells counted from unbroken runs only |

More attack threats find more wins but widen the search, so the same
`pns_nodes` proves less.  `gomoku.toml` documents each key; the defaults are
the measured configuration.

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
| **AlphaBetaBrain** | Root VCF pre-check: plays its own VCF, or breaks the opponent's — probing the breakers with PNS for other forced wins (`vcf_breaker_pns_nodes`, 10,000 in `gomoku.toml`); threat-ordered moves, threat extensions and quiescence use the unbroken-run four heuristic (`is_four_threat`) |
| **MCTSBrain** | VCF safety gate restricts root moves when opponent has VCF; heuristic priors score threat patterns |
| **Evaluation** | All brains use pattern scores + the fork bonus for positional assessment |
