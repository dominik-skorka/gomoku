# Threat Brain — VCF / VCT / PNS Search

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
6. [PNS — Proof-Number Search](#pns--proof-number-search)
7. [Execution Pipeline](#execution-pipeline)
8. [Parameters](#parameters)
9. [Example Scenarios](#example-scenarios)
10. [Configuration & Usage](#configuration--usage)
11. [Brain Comparison](#brain-comparison)

---

## Overview

| Property        | Value                                             |
| --------------- | ------------------------------------------------- |
| Brain name      | `threat`                                          |
| Algorithm       | VCF + (PNS or VCT) with alpha-beta fallback       |
| Default search  | `threat_search=pns` (configurable: `pns` or `vct`)|
| Default params  | `vcf_depth=20`, `vct_depth=12`, `pns_nodes=10000`, `pns_time_s=5.0`, `depth=4` |
| Strength        | Strongest brain in the workspace                  |
| Best at         | Endgame / tactical positions                      |

The threat brain runs four phases on every turn, stopping at the first hit:

```
VCF  ──▶  PNS/VCT  ──▶  VCF-aware search  ──▶  Alpha-beta fallback
fast       moderate     smart defence          general
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

### Defence enumeration

When the attacker plays a three-threat, the defender's response set includes:

1. **End cells** — the two open ends of the live three (direct blocks).
2. **Counter-threat moves** — any candidate cell that, when played by the
   defender, creates a new four-threat.  This catches moves like creating a
   counter-four from a distant cell, which the original VCT missed.

The expanded defence set is computed by `_vct_defence_moves()`.

> **History:** The original VCT only considered the two end cells.  In a real
> game (threat vs alphabeta), VCT declared G11 a forced win at move 43, but
> White replied K8 — not an end cell — creating an unblockable four at K5.
> Black spent 22 moves defending and lost.  The fix adds counter-threat moves
> to the defence set, making VCT sound against this class of refutation.

### Algorithm

```
function VCT(board, attacker, depth):
    if depth <= 0: return None

    # Phase 1 — try VCF first (stronger, faster)
    result = VCF(board, attacker, depth)
    if result != None: return result

    # Phase 2 — try three-threats
    for each three-threat (attack) of attacker:
        board' = board + attack

        if attack wins: return attack

        # Compute full defence set (end cells + counter-threat moves)
        defences = end_cells(attack) ∪ counter_threat_moves(board', opponent)

        proven = true
        for each defence in defences:
            board'' = board' + defence

            if opponent has unblockable counter-four:
                proven = false; break

            if opponent has blockable counter-fours:
                block them all (costs extra depth)

            if VCT(board_after, attacker, depth - 2 - cost) = None:
                proven = false; break

        if proven: return attack
    
    return None
```

### Properties

| Property          | Value                                  |
| ----------------- | -------------------------------------- |
| Branching factor  | Attacker: four-threats + three-threats |
|                   | Defender: end cells + counter-threats  |
| Typical depth     | 12 plies (6 attack + 6 response)       |
| Speed             | Moderate                               |
| Completeness      | Finds wins via four- and three-chains  |
| Soundness         | Sound against counter-four refutations |

---

## PNS — Proof-Number Search

PNS (Allis 1994) is an alternative to VCT for the broader threat search
phase.  Instead of depth-limited DFS, it uses **proof numbers** to guide a
best-first search through the threat-space tree.

### Key advantage over VCT

At **AND nodes** (defender to move), PNS considers **all threat-relevant
defensive moves** — direct blocks of attacker threats plus counter-threats
(defender fours and threes).  This covers the same refutations as all-candidate
expansion in practice while keeping branching tractable (~15 moves instead
of ~100+).

### Node types

| Node type | Who moves | Children | Win condition |
|-----------|-----------|----------|---------------|
| **OR**  (attacker) | Attacker | Four-threats + three-threats | *Any* child proved |
| **AND** (defender) | Defender | Blocks + counter-threats | *Every* child proved |

### How proof numbers work

Each node has two numbers:

- **Proof number (PN)**: minimum leaves to examine to *prove* this node (attacker wins).
- **Disproof number (DN)**: minimum leaves to examine to *disprove* this node.

| Node state | PN | DN |
|------------|------|------|
| Proved (attacker wins) | 0 | ∞ |
| Disproved (no win) | ∞ | 0 |
| Unknown | ≥ 1 | ≥ 1 |

For OR nodes: `PN = min(children.PN)`, `DN = sum(children.DN)`.
For AND nodes: `PN = sum(children.PN)`, `DN = min(children.DN)`.

### Algorithm

```
function PNS(board, attacker, max_nodes):
    root = OR_node(board)
    evaluate(root)

    while root is neither proved nor disproved and nodes < max_nodes:
        mpn, path = select_most_proving_node(root)  # walk to best leaf
        expand(mpn)                                  # generate children
        backpropagate_path(path)                     # update PN/DN along path

    if root is proved:
        return root's proved child's move
    return None
```

### Optimisations

- **Path-based backpropagation**: Only nodes on the selection path are updated
  after each expansion — O(depth) per iteration instead of O(tree_size).
- **No-copy immediate-win check** (`_find_winning_move`): Counts neighbours
  in-place instead of deep-copying the board per candidate (~100× faster).
- **Smart defence moves**: AND nodes expand blocks + counter-threats (~15 moves)
  instead of all candidates (~100+), keeping the tree tractable.
- **Immediate-win pruning**: If any child is an immediate five-in-a-row,
  expansion stops early.
- **Defender-win pruning**: At AND nodes, if a defender move creates an
  immediate five for the defender, that single child disproves the subtree.

### Properties

| Property          | Value                                  |
| ----------------- | -------------------------------------- |
| Branching factor  | Attacker: four-threats + three-threats |
|                   | Defender: blocks + counter-threats     |
| Budget            | Node count (default 10,000)            |
| Speed             | Fast (budget-bounded, path backprop)   |
| Soundness         | Sound against blocks & counter-threats |
| Completeness      | Within budget; may not prove/disprove  |

### PNS vs VCT comparison

| Aspect | VCT | PNS |
|--------|-----|-----|
| Search strategy | Depth-limited DFS | Best-first (proof numbers) |
| Defender moves | End cells + counter-four moves | Blocks + counter-threats |
| Termination | Depth exhausted | Node budget exhausted |
| Soundness | Sound against counter-fours | Sound against blocks & counter-threats |
| Speed on simple positions | Faster | Slightly slower |
| Speed on complex positions | May miss refutations | More reliable |
| Memory | O(depth) stack | O(nodes) tree |

---

## Execution Pipeline

On each turn, the brain runs four phases in order, stopping at the first hit:

```
┌──────────────────────────────────────────────────────────┐
│                      get_move()                          │
├──────────┬───────────┬──────────────────┬────────────────┤
│ Phase 1  │ Phase 2   │ Phase 3          │ Phase 4        │
│ VCF      │ PNS / VCT │ VCF-aware search │ Alpha-beta     │
│          │           │ (defensive)      │ (fallback)     │
├──────────┼───────────┼──────────────────┼────────────────┤
│ Find own │ Find own  │ Opponent has VCF │ General search │
│ forced   │ forced    │ → find all      │ when no threat │
│ win via  │ win via   │ candidate moves  │ sequence found │
│ fours    │ all threat│ that break it;   │                │
│ only     │ types     │ pick the best    │                │
│          │           │ one positionally │                │
├──────────┼───────────┼──────────────────┼────────────────┤
│ depth=20 │ PNS:10k  │ N x VCF(depth=20)│ depth=4        │
│          │ VCT:12    │                  │                │
│ ~us      │ ~ms-s     │ ~30-60ms         │ ~100ms         │
└──────────┴───────────┴──────────────────┴────────────────┘
     fast      moderate      smart defence       slow
```

Before any of the four phases, the brain also checks for immediate wins and
forced blocks:

1. **Immediate win** — if the current player can complete five-in-a-row, play it.
2. **Forced block** — if the opponent can win next move, block that cell.

The `threat_search` parameter selects which algorithm runs in Phase 2:
- `"pns"` (default) — Proof-Number Search with a node budget.
- `"vct"` — depth-limited VCT with expanded defence enumeration.

### Phase 3 — VCF-aware search

When the opponent has a VCF (forced win via four-threats), the brain does
**not** blindly block at the VCF's first attack cell.  Instead it runs a
*VCF-aware search*:

1. For every candidate move, play it and re-check whether the opponent still
   has a VCF on the resulting board.
2. Collect all **VCF-breakers** — moves that eliminate the opponent's VCF.
3. Among the breakers, pick the one with the highest positional evaluation
   score (via `evaluate_board`).
4. If **no** candidate breaks the VCF, fall back to blocking at the VCF's
   first attack cell (same as the old behaviour).

This allows the brain to choose a move that both neutralises the opponent's
tactical threat *and* improves its own position — for example blocking a
growing row instead of a low-urgency deep VCF.

> **History:** In game 3 (threat-Black vs alphabeta-White), the old pipeline
> unconditionally blocked White's VCF at step 4, ignoring White's gradual
> build-up on row 7.  Because the VCF was deep (many plies), almost any
> positional move would have disrupted it.  The VCF-aware search would have
> chosen a move that both broke the VCF and blocked the row.

---

## Parameters

The brain exposes six knobs:

| Parameter       | Default  | Controls                        | Effect of increasing           |
| --------------- | -------- | ------------------------------- | ------------------------------ |
| `vcf_depth`     | 20       | Max plies for VCF search        | Finds longer four-chains       |
| `vct_depth`     | 12       | Max plies for VCT search        | Finds longer three/four-chains |
| `pns_nodes`     | 10,000   | Node budget for PNS             | Deeper/wider proof search      |
| `pns_time_s`    | 5.0      | Time budget for PNS (seconds)   | Allows longer proof search     |
| `threat_search` | `"pns"`  | Algorithm: `"pns"` or `"vct"`  | Selects Phase 2 algorithm      |
| `depth`         | 4        | Alpha-beta fallback depth       | Better positional play         |

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

| Scenario                        | Recommendation                                          |
| ------------------------------- | ------------------------------------------------------- |
| Fast play / low resources       | `vcf_depth=12, threat_search=vct, vct_depth=6`         |
| Default / balanced              | `threat_search=pns, pns_nodes=10000, pns_time_s=5.0, vcf_depth=20` |
| Maximum tactical strength       | `threat_search=pns, pns_nodes=50000, pns_time_s=15.0, vcf_depth=30` |
| Better fallback positional play | Increase `depth` (e.g. 5-6)                            |

> **Note:** `vcf_depth` is cheap to increase (narrow tree). `vct_depth` is
> expensive (exponential branching). `pns_nodes` scales linearly with memory
> but search time depends on position complexity; `pns_time_s` provides a hard
> wall-clock cap.  Increasing `depth` affects only the alpha-beta fallback and
> has the same cost profile as the standalone `alphabeta` brain.
>
> The fallback alpha-beta engine includes Zobrist transposition table,
> iterative deepening, principal variation search, quiescence search,
> history heuristic, aspiration windows, futility pruning, and evaluation
> caching.  See [ALPHABETA_BRAIN.md](ALPHABETA_BRAIN.md) for details.

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
  → Enhanced alpha-beta fallback (with TT, iterative deepening,
    PVS, quiescence, etc.) evaluates all candidate moves at depth=4
```

### Example 5 — VCF-aware search picks positional defence

```
Board state (Black to move):
  Row 7:  . . O O O . .      White has 3-in-a-row, building toward five
  White also has a deep VCF (needing 8+ plies to execute)

Old behaviour:
  VCF-aware search did not exist — brain blindly blocked at White's
  first VCF attack cell, ignoring the growing row-7 threat.

New behaviour:
  Phase 3 (VCF-aware search):
    1. Enumerate all candidate moves.
    2. For each, check if White still has a VCF on the resulting board.
    3. Most candidates near row 7 break the deep VCF (it was fragile).
    4. Among VCF-breakers, evaluate_board picks blocking row 7 as the
       positionally best move.
    → Brain blocks row 7 AND breaks the VCF in one move.
```

---

## Configuration & Usage

### CLI

```bash
# Default settings (PNS mode, pns_nodes=10000, vcf_depth=20, fallback depth=4)
uv run gomoku play --black threat --white alphabeta

# Custom board size
uv run gomoku play --black threat --white minimax --board-size 9
```

### gomoku.toml

```toml
[brains.threat]
depth = 4
vcf_depth = 20
vct_depth = 12
threat_search = "pns"   # "pns" (default) or "vct"
pns_nodes = 10_000      # node budget for PNS
pns_time_s = 5.0        # time budget for PNS (seconds)
```

### Python API

```python
from gomoku_brain import get_brain
from gomoku_brain.threat import ThreatBrain

# Via registry (uses settings from gomoku.toml)
brain = get_brain("threat", depth=4)

# Direct construction — PNS mode (default)
brain = ThreatBrain(
    depth=5,             # alpha-beta fallback depth
    vcf_depth=24,        # deeper four-threat search
    threat_search="pns", # use proof-number search
    pns_nodes=50_000,    # larger node budget
    pns_time_s=10.0,     # larger time budget
)

# Direct construction — VCT mode
brain = ThreatBrain(
    depth=4,
    vcf_depth=20,
    threat_search="vct",
    vct_depth=12,
)

# Use it
move = brain.get_move(board, stone)
```

### Constructor parameters

```python
ThreatBrain(
    depth: int = 4,              # Alpha-beta fallback depth
    vcf_depth: int = 20,         # VCF search depth (plies)
    vct_depth: int = 12,         # VCT search depth (plies)
    threat_search: str = "pns",  # "pns" or "vct"
    pns_nodes: int = 10_000,     # PNS node budget
    pns_time_s: float = 5.0,     # PNS time budget (seconds)
)
```

---

## Brain Comparison

### Complexity

| Brain       | Algorithm               | Branching Factor | Typical Depth/Budget | Nodes Explored    |
| ----------- | ----------------------- | ---------------- | -------------------- | ----------------- |
| `random`    | Uniform random          | 1                | 0                    | 1                 |
| `minimax`   | Classic minimax         | ~50              | 3                    | ~125,000          |
| `negamax`   | Negamax                 | ~50              | 3                    | ~125,000          |
| `alphabeta` | Alpha-beta pruning      | ~7 (with cutoff) | 4                    | ~2,400            |
| **`threat`**| **VCF + PNS + AB**      | **1-5 / all**    | **20 / 50k / 4**    | **~100-50,000**   |

> Node counts are approximate for a mid-game 15x15 board.

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
| **`threat`**| **Finds forced wins 10+ moves ahead; VCF-aware defence balances tactics and position**  | **Relies on fallback for positional play; VCT can be slow with high depth** |

### When to use `threat`

- **Best for:** mid-game and endgame positions where tactical sequences decide.
- **Overkill for:** very early game (few threats exist, falls back to alpha-beta anyway).
- **Pair with:** use `threat` for both players to see if a forced win exists from a given position.
