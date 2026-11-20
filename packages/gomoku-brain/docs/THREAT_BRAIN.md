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
| Built-in defaults | `vcf_depth=20`, `vct_depth=12`, `pns_nodes=10000`, `pns_time_s=5.0`, `depth=4` |
| Shipped `gomoku.toml` | `pns_nodes=120000`; fallback `depth=5`, `max_candidates=12`, `eval_ordering_depth=11` (see [Parameters](#parameters)) |
| Strength        | Strongest brain in the workspace                  |
| Best at         | Endgame / tactical positions                      |

On every turn the threat brain first looks for its own forced win, then
for the opponent's, and only then searches positionally, stopping at the
first step that produces a move:

```
win / block five ─▶ own VCF ─▶ own PNS/VCT ─▶ opp VCF ─▶ opp PNS/VCT ─▶ opp fork ─▶ alpha-beta
   instant           fast       moderate      defend      defend         block      general
```

See [Execution Pipeline](#execution-pipeline) for each step.

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

### Status: proved, disproved or unknown

`pns_threat_status()` reports how a search ended:

| Status | Meaning |
|--------|---------|
| `PROVED` | The attacker has a forced win; the first move is returned |
| `DISPROVED` | The attacker has **no** forced win in threat space (exact, whatever the budget) |
| `UNKNOWN` | The node or time budget ran out first |

`pns_threat_search()` returns only the winning move or `None`, folding the
last two together.  Callers that use "no forced win" as a safety guarantee,
such as the breaker probes in the defensive steps, need the difference.

### Key advantage over VCT

At **AND nodes** (defender to move), PNS considers **all threat-relevant
defensive moves** (`_pns_defence_moves()`):

1. Forced blocks of the attacker's fours — straight or split (`XXX_X`,
   `XX_XX`): a four is any move after which one empty cell completes five.
2. Both end cells of the attacker's live threes.
3. The defender's own fours (counter-fours).
4. The defender's own threes (counter-threes).
5. Every cell that defuses an attacker's threat to make an unblockable
   four (a live four or a double four) next move.  Occupying the attack
   cell is only one such defence: taking an end of the would-be four, for
   example, leaves the attacker a four that can be blocked.

This keeps branching tractable (~15 moves instead of ~100+) while covering
the refutations all-candidate expansion would find.

> **History:** without step 5, PNS "proved" wins the defender could refute
> by taking an end of the would-be four.  The threat brain then played
> those false wins and lost games it was ahead in.

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

### Graph search

Positions are stored in a table keyed by Zobrist hash, so a position
reached through different move orders is a **single node** with several
parents.  The search runs on a DAG instead of re-searching the same
subtree once per move order (30–65% of expansions in unproductive
searches used to be such repeats).  This is sound because a node's status
depends only on its position.  `max_nodes` counts distinct positions.

### Algorithm

```
function PNS(board, attacker, max_nodes):
    root = OR_node(board)
    evaluate(root)

    while root is neither proved nor disproved and nodes < max_nodes:
        mpn, path = select_most_proving_node(root)  # walk to best leaf
        expand(mpn)                                  # generate children
        backpropagate_path(path)                     # update PN/DN along path

    if root is proved:    return PROVED, root's proved child's move
    if root is disproved: return DISPROVED
    return UNKNOWN                                    # budget ran out
```

### Optimisations

- **Path-based backpropagation**: Only nodes on the selection path are updated
  after each expansion — O(depth) per iteration instead of O(graph size).
  A shared node can leave another parent's cached best child stale; the
  next walk then ends on an already-solved node, and updating that path
  repairs it.
- **One mutable board**: Nodes store no positions.  A single `SearchState`
  replays the edge moves along the selection path each iteration.
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
| Budget            | Distinct positions (`pns_nodes`) and seconds (`pns_time_s`) |
| Speed             | Fast (budget-bounded, path backprop)   |
| Soundness         | Sound against blocks & counter-threats |
| Completeness      | Within budget; otherwise `UNKNOWN`     |

### PNS vs VCT comparison

| Aspect | VCT | PNS |
|--------|-----|-----|
| Search strategy | Depth-limited DFS | Best-first (proof numbers) |
| Defender moves | End cells + counter-four moves | Blocks + counter-threats |
| Termination | Depth exhausted | Proved, disproved, or node/time budget exhausted |
| Soundness | Sound against counter-fours | Sound against blocks & counter-threats |
| Speed on simple positions | Faster | Slightly slower |
| Speed on complex positions | May miss refutations | More reliable |
| Memory | O(depth) stack | O(nodes) tree |

---

## Execution Pipeline

`get_move()` runs these steps in order and plays the first move one of
them produces.  The decision path recorded in the diagnostics (see
[DIAGNOSTICS.md](../../../docs/DIAGNOSTICS.md)) names the step:

| Step | Check | Move played | Decision path |
|------|-------|-------------|---------------|
| 1 | We can complete five | That cell | `immediate_win` |
| 2 | The opponent can complete five | Block it | `block_immediate_win` |
| 3 | Our VCF (`vcf_depth`) | First move of the win | `vcf_win` |
| 4 | Our PNS (`pns_nodes`, `pns_time_s`) or VCT (`vct_depth`) | First move of the win | `pns_win` / `vct_win` |
| 5 | Opponent's VCF | [VCF-aware search](#step-5--vcf-aware-search) | `vcf_aware_search` |
| 6 | Opponent's PNS/VCT win | [VCT-aware search](#step-6--vct-aware-search) | `vct_aware_search` |
| 7 | Opponent fork cells (≥ 2 threat lines meet) | Block the fork | `block_fork` |
| 8 | Nothing forcing | [Alpha-beta fallback](#step-8--alpha-beta-fallback) | the fallback's own path (`alphabeta`, ...) |

`threat_search` selects the algorithm of steps 4 and 6:
- `"pns"` (default): Proof-Number Search with a node and time budget.
- `"vct"`: depth-limited VCT with expanded defence enumeration.

All VCF searches of one decision share a transposition cache.

### Step 5 — VCF-aware search

When the opponent has a VCF (forced win via fours), the brain does
**not** simply block the VCF's first attack cell:

1. Every candidate (radius 2) is played, and the opponent's VCF is searched
   again.  Moves after which it is gone are **VCF-breakers**.
   `classify_vcf_breakers()` also marks which breakers are **durable**:
   the VCF stays broken after any forced reply.  A breaker that only
   works because it makes a four can be undone by the forced block.
   Durable breakers are preferred when any exist.
2. Breaking the VCF is not enough if the opponent still has a forced win
   that uses threes.  So the breakers are **probed**, best static
   evaluation first: a VCF search, then a PNS status search (VCT in
   `vct` mode) for the opponent.
3. The first breaker after which the opponent's win is `DISPROVED` is
   played.  Failing that, the best breaker whose probe ran out of budget
   (`UNKNOWN`) is played; failing that, the positionally best breaker.
4. With no breaker at all, block the VCF's first attack cell.

Each probe gets half of `pns_nodes` and `pns_time_s`.  The whole probe
loop is capped at `pns_time_s`.

> **History:** an earlier version blocked the VCF's first cell
> unconditionally, ignoring a growing row the opponent was building.  A
> later one picked the positionally best breaker without the probe in
> point 2, and walked into the opponent's three-based forced win.

### Step 6 — VCT-aware search

If the opponent has no VCF, the brain runs the same threat search as in
step 4 for the opponent.  That search is skipped when the opponent lacks
the material for a forced win: at least one four-threat or two live-three
cells.  Skipping it is the common case in quiet positions, and each search
can cost up to `pns_time_s`.  If the opponent has a forced win, every
candidate is probed as in step 5, points 2–3.  When no probe succeeds, the
brain blocks the first move of the opponent's win.

### Step 8 — Alpha-beta fallback

With nothing forcing on the board, an `AlphaBetaBrain` picks the move.
It is configured by the `fallback` table (same keys as `[brains.alphabeta]`)
or, without one, runs with default settings at the top-level `depth`.  Its
own VCF pre-check is left off: steps 3 and 5 have already shown that
neither side has a VCF.  The fallback includes a Zobrist transposition
table, iterative deepening, principal variation search, quiescence search,
history heuristic, aspiration windows, futility pruning and evaluation
caching.  See [ALPHABETA_BRAIN.md](ALPHABETA_BRAIN.md).

---

## Parameters

The brain has eight settings, one of them the `threats` tables.  "Default" is the built-in value, used
when a setting is left out; the shipped `gomoku.toml` changes some:

| Parameter       | Default  | `gomoku.toml` | Controls                        | Effect of increasing           |
| --------------- | -------- | ------------- | ------------------------------- | ------------------------------ |
| `vcf_depth`     | 20       | 20            | Max plies for VCF search        | Finds longer four-chains       |
| `vct_depth`     | 12       | 12            | Max plies for VCT search        | Finds longer three/four-chains |
| `pns_nodes`     | 10,000   | **120,000**   | Distinct positions PNS may create | Proves/disproves more positions |
| `pns_time_s`    | 5.0      | 5.0           | Time budget for PNS (seconds)   | Allows longer proof search     |
| `threat_search` | `"pns"`  | `"pns"`       | Algorithm: `"pns"` or `"vct"`   | Selects the algorithm of steps 4 and 6 |
| `depth`         | 4        | 5 (ignored)   | Fallback depth without a `fallback` table | Better positional play |
| `fallback`      | none     | **table**     | `AlphaBetaBrain` settings for the fallback (its `depth` replaces the top-level one) | — |
| `threats`       | defaults | defaults      | Threat spaces of the searches (tables below) | — |

The shipped fallback table is `depth = 5`, `eval_ordering_depth = 11`,
`max_candidates = 12`: a depth-5 search that orders moves by cheap
pattern priors and keeps 12 candidates per node.

### Threat spaces (`[brains.threat.threats.*]`)

Which threats VCF and PNS / VCT play and answer.  `own` is the attacker's
side of our forced-win search (steps 3–4), `opponent` of the search for the
opponent's win (steps 5–6 and the breaker probes); `defence` is what the
defender may reply inside both; `forks` is the fork-cell rule of step 7.
A *four* is a move after which one cell completes five, a *three* one after
which one more stone makes a live four.

| Table | Key | Default | Off / on |
|-------|-----|---------|----------|
| `own`, `opponent` | `fours` | `true` | off: no VCF; PNS / VCT attack with threes only |
| | `split_fours` | `true` | off: unbroken runs only (`XXX_X`, `XX_XX` not fours) |
| | `threes` | `true` | off: PNS / VCT attack with fours only |
| | `split_threes` | `false` | on: `_XX_X_` is a three (defended at gap or ends); wider search |
| `defence` | `block_fours` | `true` | off: every four "wins" — **false proofs** |
| | `block_threes` | `true` | off: threes answered by counter-threats only — **false proofs** |
| | `counter_fours` | `true` | off: the "K8" bug returns — **false proofs** |
| | `split_counter_fours` | `true` | off: the F5/E9 bug returns — **false proofs** |
| | `counter_threes` | `true` | off: smaller, faster PNS (a heuristic reply) |
| | `split_counter_threes` | `false` | on: wider PNS |
| | `unblockable_four_defences` | `true` | off: the #42 bug returns — **false proofs** |
| `forks` | `split` | `false` | on: split threes / fours count as fork lines |

Fewer **attack** threats only make a search less complete — fewer wins
found, never false ones — and more find more wins but widen the search.
Fewer **defence** replies make proofs unsound: the search proves wins that
kind of reply refutes, and the brain plays them.  `gomoku.toml` documents
each key; `gomoku_brain.threat_space.ThreatSpace` is the code side.

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
| Default / balanced              | the shipped `gomoku.toml` (`pns_nodes=120000`, ~1 s per move) |
| Faster play                     | `pns_nodes=60000` (about half the think time)            |
| Better fallback positional play | Raise `fallback.depth` and measure it (see below)       |

Measured with `gomoku match` (300 games each, colours swapped per opening):

| Change | Result | Think time per move |
|--------|--------|---------------------|
| `pns_nodes` 60,000 → 120,000, with the faster PNS of #55 | **+47 Elo** (95% interval +22 to +72) | 1027 → 954 ms |
| `pns_nodes` 30,000 → 60,000 | **+30 Elo** (95% interval +4 to +57) | 667 → 968 ms |
| `pns_nodes` 30,000 → 100,000 | +40 Elo (+11 to +69) | 651 → 1231 ms |
| split threes for both sides (`own` and `opponent` `split_threes = true`, at 30,000) | **−58 Elo** (−92 to −26) | 385 → 817 ms |
| split threes in our attack only (`own.split_threes = true`, at 60,000) | +7 Elo (−16 to +30): no measurable difference | 1040 → 1248 ms |
| `forks.split = true` (at 30,000) | −23 Elo (−40 to −7) | 643 → 657 ms |
| fallback depth 5 → 4 (at 30,000) | −24 Elo (−59 to +10): no measurable difference | ~6% faster |

Earlier measurements, before split fours were detected (PNS was then
blind to them; proofs were smaller and sometimes false):

| Change | Result | Think time per move |
|--------|--------|---------------------|
| `pns_nodes` 10,000 → 30,000 | +43 Elo (+15 to +72) | 270 → 436 ms |
| `pns_nodes` 10,000 → 100,000 | +40 Elo (+10 to +69): no more than 30,000 | 270 → ~730 ms |
| fallback depth 4 → 5 with the 12-candidate table (at 10,000) | −16 Elo (−49 to +16): no measurable difference | ~9% slower |

Before a long match, `gomoku review` (see the main README) tells whether
the brain's mistakes come from too small a budget or from bugs.  It
replays each blunder with larger settings and names the one that fixes
it.

> **Note:** `vcf_depth` is cheap to increase (narrow tree). `vct_depth` is
> expensive (exponential branching). `pns_nodes` scales linearly with memory
> but search time depends on position complexity; `pns_time_s` provides a hard
> wall-clock cap.  The fallback's depth affects only the alpha-beta fallback
> and has the same cost profile as the standalone `alphabeta` brain.

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

  Our VCF, our PNS: no forced win
  Opponent VCF: none; opponent PNS: skipped (not enough material)
  Opponent fork cells: none
  → Alpha-beta fallback (TT, iterative deepening, PVS, quiescence, ...)
    searches the candidates at the fallback's depth (5 in gomoku.toml)
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
  Step 5 (VCF-aware search):
    1. Enumerate all candidate moves.
    2. For each, check if White still has a VCF on the resulting board.
    3. Most candidates near row 7 break the deep VCF (it was fragile).
    4. Probe the breakers, best evaluation first, for any other forced
       win of White's; blocking row 7 evaluates best and leaves none.
    → Brain blocks row 7 AND breaks the VCF in one move.
```

---

## Configuration & Usage

### CLI

```bash
# The [brains.threat] alias from gomoku.toml (PNS, pns_nodes=120000, fallback depth 5)
uv run gomoku play --black threat --white alphabeta

# Custom board size
uv run gomoku play --black threat --white minimax --board-size 9
```

### gomoku.toml

The shipped alias:

```toml
[brains.threat]
type = "threat"
depth = 5               # fallback depth when there is no fallback table
vcf_depth = 20
vct_depth = 12
threat_search = "pns"   # "pns" (default) or "vct"
pns_nodes = 120_000     # distinct positions PNS may create
pns_time_s = 5.0        # time budget for PNS (seconds)

[brains.threat.fallback]  # AlphaBetaBrain settings for the fallback
depth = 5                 # replaces the top-level depth
eval_ordering_depth = 11
max_candidates = 12

[brains.threat.threats.own]       # see "Threat spaces" above; these are
split_threes = false              # the defaults, so all four tables may
[brains.threat.threats.opponent]  # be left out
split_threes = false
[brains.threat.threats.defence]
counter_threes = true
[brains.threat.threats.forks]
split = false
```

### Python API

```python
from gomoku.settings import GomokuSettings
from gomoku_brain import get_brain, get_brain_from_settings
from gomoku_brain.threat import ThreatBrain

# The [brains.threat] alias from gomoku.toml
brain = get_brain_from_settings("threat", GomokuSettings().brains)

# By type with explicit settings (gomoku.toml is not read)
brain = get_brain("threat", pns_nodes=30_000)

# Direct construction — PNS mode (default)
brain = ThreatBrain(
    vcf_depth=24,        # deeper four-threat search
    threat_search="pns", # use proof-number search
    pns_nodes=30_000,    # larger node budget
    pns_time_s=10.0,     # larger time budget
    fallback={"depth": 5, "max_candidates": 12, "eval_ordering_depth": 11},
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
    pns_nodes: int = 10_000,     # PNS budget (distinct positions)
    pns_time_s: float = 5.0,     # PNS time budget (seconds)
    fallback: dict | None = None,  # AlphaBetaBrain kwargs for the fallback;
                                   # its "depth" replaces the top-level depth
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
| **`threat`**| **VCF + PNS + AB**      | **1-5 / all**    | **20 / 120k / 5**   | **~100-120,000**  |

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
