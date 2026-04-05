# Threat Brain — VCF / VCT Search

A **threat-based** Gomoku brain that tries to prove a forced win before
resorting to general search.  It chains together *threats* — moves the opponent
**must** answer — until the sequence leads to five-in-a-row.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Threat Types](#threat-types)
4. [VCF — Victory by Continuous Fours](#vcf--victory-by-continuous-fours)
5. [VCT — Victory by Continuous Threats](#vct--victory-by-continuous-threats)
6. [Execution Pipeline](#execution-pipeline)
7. [Depth Parameters](#depth-parameters)
8. [Example Scenarios](#example-scenarios)
9. [Configuration & Usage](#configuration--usage)
10. [Brain Comparison](#brain-comparison)

---

## Overview

| Property      | Value                                      |
| ------------- | ------------------------------------------ |
| Brain name    | `threat`                                   |
| Algorithm     | VCF + VCT with alpha-beta fallback         |
| Default depth | `vcf_depth=20`, `vct_depth=8`, `depth=4`  |
| Strength      | Strongest brain in the workspace           |
| Best at       | Endgame / tactical positions               |

The threat brain runs three phases on every turn:

```
VCF  ──▶  VCT  ──▶  Opponent-VCF check  ──▶  Alpha-beta fallback
fast       moderate   fast                     general
```

---

## How It Works

Traditional brains (minimax, negamax, alpha-beta) evaluate **all** candidate
moves to a fixed depth.  The threat brain instead asks a narrower question:

> *"Is there a sequence of forcing moves that **guarantees** five-in-a-row?"*

Because only *threats* (moves the opponent must answer) are explored, the
search tree is extremely narrow and can be searched much deeper than general
alpha-beta — typically 20 plies for VCF vs. 3-4 plies for alpha-beta.

```
General search (alpha-beta depth=4)        Threat search (VCF depth=20)
┌─────────────────────────────┐            ┌──────────────┐
│  ╱╲    ╱╲    ╱╲    ╱╲       │            │  ╱╲          │
│ ╱╲╱╲  ╱╲╱╲  ╱╲╱╲  ╱╲╱╲    │            │ ╱  ╲         │
│╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲   │            │╱    ╲        │
│           ...               │            │╲    ╱        │
│  ~50 moves wide × 4 deep   │            │ ╲  ╱         │
│  = millions of nodes        │            │  ╲╱          │
└─────────────────────────────┘            │  ~2-5 wide   │
                                           │  × 20 deep   │
                                           │  = hundreds   │
                                           └──────────────┘
```

---

## Threat Types

### Four-Threat (used by VCF)

A move that creates **four** stones in a line.  The opponent has at most **one**
legal reply to prevent five-in-a-row.

```
  Before             After placing ★        Opponent MUST play ◆

  . . X X X . .      . . X X X ★ .         . . X X X ★ ◆
      3-in-a-row      Dead four (1 open)     Forced block

  . . X X X . .      . ★ X X X . .         Unblockable!
                       Live four (2 open)    (two ways to complete)
```

- **Dead four** (1 open end) → opponent has exactly 1 forced block.
- **Live four** (2 open ends) → **unblockable**, immediate win.
- **Five** → game over.

### Three-Threat (used by VCT)

A move that creates an **open three** (both ends empty).  The opponent has
**two or more** reasonable responses (block either end, counter-attack, etc.).

```
  Before             After placing ★

  . . X X . . .      . . X X ★ . .
      2-in-a-row      Live three (2 open ends)
                       Opponent can block at ◇ or ◇:
                      ◇ . X X ★ . ◇
```

---

## VCF — Victory by Continuous Fours

VCF explores **only** four-threats.  Because each four-threat forces a single
response, the branching factor is essentially **1** on the defender's side.

### Algorithm

```
function VCF(board, attacker, depth):
    if depth ≤ 0: return None

    for each four-threat (attack, defence) of attacker:
        board' = board + attack

        if attack creates five → return attack        # Win!
        if attack is live-four → return attack         # Unblockable!

        board'' = board' + defence                     # Forced block

        if VCF(board'', attacker, depth-2) ≠ None:
            return attack                              # Chain continues
    
    return None                                        # No forced win
```

### Properties

| Property          | Value                                |
| ----------------- | ------------------------------------ |
| Branching factor  | Attacker: # of four-threats (~2-5)   |
|                   | Defender: **1** (forced)             |
| Typical depth     | 20 plies (10 attack + 10 response)   |
| Speed             | Very fast                            |
| Completeness      | Only finds wins via four-chains      |

---

## VCT — Victory by Continuous Threats

VCT extends VCF by **also** considering live-three threats.  This finds more
forced wins but is slower because the defender has multiple responses to a
three-threat.

### Algorithm

```
function VCT(board, attacker, depth):
    if depth ≤ 0: return None

    # Phase 1 — try VCF first (stronger, faster)
    result = VCF(board, attacker, depth)
    if result ≠ None: return result

    # Phase 2 — try three-threats
    for each three-threat (attack, defences[]) of attacker:
        board' = board + attack

        if attack wins → return attack

        proven = true
        for each defence in defences[]:
            board'' = board' + defence

            if opponent has counter four-threat:
                proven = false; break            # Can't refute

            if VCT(board'', attacker, depth-2) = None:
                proven = false; break            # Can't continue

        if proven: return attack
    
    return None
```

### Properties

| Property          | Value                                  |
| ----------------- | -------------------------------------- |
| Branching factor  | Attacker: four-threats + three-threats |
|                   | Defender: **2+** per three-threat      |
| Typical depth     | 8 plies (4 attack + 4 response)        |
| Speed             | Moderate                               |
| Completeness      | Finds wins via four- and three-chains  |

---

## Execution Pipeline

On each turn, the brain runs four phases in order, stopping at the first hit:

```
┌──────────────────────────────────────────────────────────┐
│                      get_move()                          │
├──────────┬───────────┬──────────────────┬────────────────┤
│ Phase 1  │ Phase 2   │ Phase 3          │ Phase 4        │
│ VCF      │ VCT       │ Opponent VCF     │ Alpha-beta     │
│          │           │ (defensive)      │ (fallback)     │
├──────────┼───────────┼──────────────────┼────────────────┤
│ Find own │ Find own  │ Check if oppnt   │ General search │
│ forced   │ forced    │ has a VCF; if so │ when no threat │
│ win via  │ win via   │ block the first  │ sequence found │
│ fours    │ fours +   │ attack move      │                │
│ only     │ threes    │                  │                │
├──────────┼───────────┼──────────────────┼────────────────┤
│ depth=20 │ depth=8   │ depth=20         │ depth=4        │
│ ~μs      │ ~ms       │ ~μs              │ ~100ms         │
└──────────┴───────────┴──────────────────┴────────────────┘
     fast      moderate       fast              slow
```

---

## Depth Parameters

The brain exposes three depth knobs:

| Parameter    | Default | Controls                     | Effect of increasing          |
| ------------ | ------- | ---------------------------- | ----------------------------- |
| `vcf_depth`  | 20      | Max plies for VCF search     | Finds longer four-chains      |
| `vct_depth`  | 8       | Max plies for VCT search     | Finds longer three/four-chains|
| `depth`      | 4       | Alpha-beta fallback depth    | Better positional play        |

### How depth is consumed

Each attack + response pair consumes **2 plies** of depth:

```
depth=8 budget:

  Ply 1:  Attacker plays threat        ─┐
  Ply 2:  Defender blocks              ─┘  depth remaining: 6

  Ply 3:  Attacker plays threat        ─┐
  Ply 4:  Defender blocks              ─┘  depth remaining: 4

  Ply 5:  Attacker plays threat        ─┐
  Ply 6:  Defender blocks              ─┘  depth remaining: 2

  Ply 7:  Attacker plays threat        ─┐
  Ply 8:  Defender blocks              ─┘  depth remaining: 0 → stop
```

So `vcf_depth=20` means up to **10 consecutive four-threats** can be chained,
and `vct_depth=8` means up to **4 mixed threats**.

### Tuning guidelines

| Scenario                        | Recommendation                  |
| ------------------------------- | ------------------------------- |
| Fast play / low resources       | `vcf_depth=12, vct_depth=4`    |
| Default / balanced              | `vcf_depth=20, vct_depth=8`    |
| Maximum tactical strength       | `vcf_depth=30, vct_depth=12`   |
| Better fallback positional play | Increase `depth` (e.g. 5-6)    |

> **Note:** `vcf_depth` is cheap to increase (narrow tree). `vct_depth` is
> expensive (exponential branching). Increasing `depth` affects only the
> alpha-beta fallback and has the same cost profile as the standalone
> `alphabeta` brain.

---

## Example Scenarios

### Example 1 — VCF finds an immediate win

```
Board (row 4):      . . X X X X .
                               ↑
                         4 in a row, one open end

VCF detects:  placing at col=6 → five-in-a-row → WIN
Result:       VCF returns (row=4, col=6) instantly
```

### Example 2 — VCF chains two four-threats

```
Board state:
  Row 4:  . X X X . . .      (3 in a row, both ends open)
  Row 6:  . X X X . . .      (3 in a row, both ends open)

VCF sequence:
  1. Black plays (4,4) → dead four:  . X X X X .
     White forced to block at (4,5)
  2. Black plays (6,4) → live four:  . X X X X .  ← unblockable!
     Result: WIN

VCF depth consumed: 4 plies (2 attacks + 2 responses)
```

### Example 3 — VCT finds a win through three-threats

```
Board state:
  Row 4:  . . X X . . .      (2 in a row, open)

VCT sequence:
  1. Black plays (4,4) → live three:  . . X X X . .
     White blocks at (4,5)
  2. Black plays a four-threat elsewhere → VCF completes the win

VCT depth consumed: 4 plies
```

### Example 4 — No threats, fallback to alpha-beta

```
Board state:  early game, scattered stones, no forcing sequences

  VCF: returns None (no four-threats chain to a win)
  VCT: returns None (no three-threat chains either)
  Opponent VCF: returns None (opponent has no forced win)
  → Alpha-beta evaluates all candidate moves at depth=4
```

---

## Configuration & Usage

### CLI

```bash
# Default settings (vcf_depth=20, vct_depth=8, fallback depth=3)
uv run gomoku play --black threat --white alphabeta

# Increase fallback search depth
uv run gomoku play --black threat --white random --depth 5

# Custom board size
uv run gomoku play --black threat --white minimax --board-size 9
```

The `--depth` flag controls the alpha-beta fallback depth.  VCF and VCT depths
use their built-in defaults (20 and 8 respectively).

### Python API

```python
from gomoku_brain import get_brain
from gomoku_brain.threat import ThreatBrain

# Via registry (uses --depth from CLI kwargs)
brain = get_brain("threat", depth=4)

# Direct construction with full control
brain = ThreatBrain(
    depth=5,         # alpha-beta fallback depth
    vcf_depth=24,    # deeper four-threat search
    vct_depth=10,    # deeper three-threat search
)

# Use it
move = brain.get_move(board, stone)
```

### Constructor parameters

```python
ThreatBrain(
    depth: int = 4,         # Alpha-beta fallback depth
    vcf_depth: int = 20,    # VCF search depth (plies)
    vct_depth: int = 8,     # VCT search depth (plies)
)
```

---

## Brain Comparison

### Complexity

| Brain       | Algorithm             | Branching Factor | Typical Depth | Nodes Explored    |
| ----------- | --------------------- | ---------------- | ------------- | ----------------- |
| `random`    | Uniform random        | 1                | 0             | 1                 |
| `minimax`   | Classic minimax       | ~50              | 3             | ~125,000          |
| `negamax`   | Negamax               | ~50              | 3             | ~125,000          |
| `alphabeta` | Alpha-beta pruning    | ~7 (with cutoff) | 4             | ~2,400            |
| **`threat`**| **VCF/VCT + AB**      | **1–5**          | **20 / 8 / 4**| **~100–2,500**    |

> Node counts are approximate for a mid-game 15×15 board.

### Strength Ranking

```
Strongest ──────────────────────────────────── Weakest

  threat  >  alphabeta  >  negamax  ≈  minimax  >>  random
    ↑            ↑            ↑                        ↑
  Finds        Prunes       Same as                 Pure
  forced       bad          minimax,                luck
  wins         branches     cleaner code
```

### Strengths & Weaknesses

| Brain       | Strengths                              | Weaknesses                            |
| ----------- | -------------------------------------- | ------------------------------------- |
| `random`    | Instant, no computation                | No intelligence whatsoever            |
| `minimax`   | Simple, correct                        | Slow, explores redundant branches     |
| `negamax`   | Cleaner than minimax, same strength    | Same exponential cost                 |
| `alphabeta` | Prunes well, good move ordering        | Misses long tactical sequences        |
| **`threat`**| **Finds forced wins 10+ moves ahead**  | **Relies on fallback for positional play; VCT can be slow with high depth** |

### When to use `threat`

- **Best for:** mid-game and endgame positions where tactical sequences decide.
- **Overkill for:** very early game (few threats exist, falls back to alpha-beta anyway).
- **Pair with:** use `threat` for both players to see if a forced win exists from a given position.
