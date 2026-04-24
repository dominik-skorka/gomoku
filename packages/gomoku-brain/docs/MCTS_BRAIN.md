# MCTS Brain — Monte Carlo Tree Search with PUCT and Heuristic Priors

An **AlphaGo Zero-inspired** Gomoku brain that replaces random rollouts
with heuristic leaf evaluation and replaces plain UCB1 with the PUCT
formula.  It has no neural network — pattern scoring stands in for the
AGZ policy and value heads.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [PUCT Selection](#puct-selection)
4. [Heuristic Priors](#heuristic-priors)
5. [Dirichlet Root Noise](#dirichlet-root-noise)
6. [Heuristic Leaf Evaluation](#heuristic-leaf-evaluation)
7. [Decision Shortcuts](#decision-shortcuts)
8. [VCF Safety Gate](#vcf-safety-gate)
9. [Iteration Loop](#iteration-loop)
10. [Diagnostics](#diagnostics)
11. [Parameters](#parameters)
12. [Example Scenarios](#example-scenarios)
13. [Configuration & Usage](#configuration--usage)
14. [Brain Comparison](#brain-comparison)

---

## Overview

| Property        | Value                                             |
| --------------- | ------------------------------------------------- |
| Brain name      | `mcts`                                            |
| Algorithm       | PUCT + heuristic priors + heuristic leaf eval    |
| Default budget  | 10 000 iterations                                |
| Strength        | Medium — exploratory, good at open positions     |
| Best at         | Broad positional play, middle-game branching      |

The brain runs many independent simulations from the root, building a
search tree where each node caches visit counts, accumulated values, and
per-move priors.  Each iteration does four things:

```
Select → Expand → Evaluate → Backpropagate
```

Unlike classical UCT (random rollouts to a terminal), this implementation
scores the leaf with a lightweight pattern heuristic — giving a ~30-60×
speed-up on sparse early-game boards.

---

## Architecture

```
get_move(board, stone)
│
├── Decision shortcuts (no MCTS needed)
│   ├── Only one legal candidate         → "single_candidate"
│   ├── We have an immediate win         → "instant_win"
│   └── Opponent wins next               → "instant_block"
│
├── VCF safety gate (opponent forced-win check)
│   └── Restrict candidates to the proven blocking cells
│
├── Build root node
│   ├── Generate candidates (radius-2 of occupied)
│   ├── Compute per-move priors (pattern score → normalised)
│   └── Mix in Dirichlet noise
│
├── For i = 1..iterations:
│   └── _run_iteration(root, grid, occupied, stone)
│       │
│       ├── 1. Selection       PUCT descent while untried is empty
│       │                      and the node has children
│       │
│       ├── 2. Expansion       Pop one untried child at random,
│       │                      apply its move, compute its priors
│       │
│       ├── 3. Evaluation      Heuristic eval of the leaf position
│       │                      → value in [0, 1] (root-relative)
│       │                      Shortcut: immediate win / decisive loss
│       │
│       └── 4. Backpropagation Walk up to root updating visits / wins
│                              (flip perspective on opponent nodes)
│
└── Return the child of root with the most visits (most robust choice)
```

All hot-path work uses **raw `int` grids** (`list[list[int]]`) and
`set[tuple[int, int]]` occupancy sets — the code deliberately avoids
constructing `Board` objects per iteration, because `Board.place_stone`
is an immutable-style deep-copy that would dominate the cost.

---

## PUCT Selection

Classical UCT uses UCB1:

```
UCB1 = Q + c · √( ln N / n )
```

That formula treats every unvisited move equally — fine for random
rollouts but wasteful when we have prior information about which moves
are tactically promising.  The PUCT formula from AlphaGo Zero multiplies
the exploration term by each move's **prior probability**:

```
PUCT = Q + c · P · √N / (n + 1)

Q = node.wins / node.visits        (mean value from root's view)
P = prior probability              (from pattern heuristic)
N = parent.visits                  (simulations reaching the parent)
n = node.visits                    (simulations through this node)
c = exploration constant           (default 2.0)
```

### Properties

- Moves with low priors still receive some exploration thanks to the
  `√N / (n + 1)` term, but moves with high priors are tried first.
- The `(n + 1)` denominator prevents a division-by-zero on unvisited
  children and smooths the exploration bonus.
- `best_child()` at selection time picks `argmax(PUCT)`; `get_move()`
  at the end of search picks `argmax(visits)` (the "most robust move"
  — more visits means the tree kept finding it convincing under
  exploration pressure).

---

## Heuristic Priors

Priors replace AGZ's neural-network policy head.  Each candidate move
is scored by a raw-grid version of the pattern evaluator
(`_score_move_prior`):

```python
# (consecutive_stones, open_ends) → raw score
_PRIOR_PATTERN_SCORES = {
    (4, 2): 10_000,   # live-4
    (4, 1):  1_000,   # dead-4
    (3, 2):  1_000,   # live-3
    (3, 1):    100,   # dead-3
    (2, 2):    100,   # live-2
    (2, 1):     10,   # dead-2
    (1, 2):     10,
    (1, 1):      1,
}
```

For each candidate, the scorer looks in all four directions twice:

1. **Offensive** — counts consecutive own stones that pass through the
   cell, plus open ends.
2. **Defensive** — same but for opponent stones, weighted **0.8×**.

The raw scores are then normalised so the priors sum to 1.0 across
the candidate set.  A constant `+1.0` baseline ensures no move receives
a zero prior (which would make PUCT ignore it entirely).

> **Why priors matter:** on a 15×15 board with ~30 candidates, a
> uniform prior gives each move 3.3% exploration weight.  Giving a
> live-3 cell 60% of the mass lets the tree concentrate on the move
> that actually matters — while still sampling the others often enough
> to detect surprises.

---

## Dirichlet Root Noise

Deterministic priors can develop blind spots: if the heuristic
consistently under-rates a move, the search will never visit it enough
to discover the mistake.  AGZ §14.4 fixes this by mixing **Dirichlet
noise** into the root priors only:

```
new_prior = (1 - weight) · prior + weight · noise
```

- **`dirichlet_alpha`** (default `0.03`) controls the noise shape.
  Small α → concentrated (one move dominates the noise vector); larger
  α → flat (uniform noise).
- **`dirichlet_weight`** (default `0.25`) controls how much the noise
  displaces the priors.  `0.0` disables noise entirely (deterministic
  play); `1.0` replaces priors with pure noise.

The implementation samples Dirichlet via the gamma-function trick
(`Gamma(α, 1)` per dimension, then normalise) so no NumPy dependency
is needed.

Noise is applied **only at the root** — deeper nodes use pure priors so
exploration doesn't bleed into every subtree.

---

## Heuristic Leaf Evaluation

Classical MCTS runs a random playout from the leaf to a terminal state
and backs up the winner as 0 or 1.  Random playouts are O(depth) per
iteration and terribly noisy on a 225-cell board.

This brain instead calls `_evaluate_position(grid, size, occupied,
root_stone_val)`:

1. Scan every run of consecutive same-colour stones on the board.
2. Score each run using `_PRIOR_PATTERN_SCORES`, with opponent runs
   weighted 0.8× (same defensive ratio as the prior).
3. Map the net advantage through a sigmoid, `k = 1000`:

```python
clamped = max(-10, min(10, score / 1000))
value = 1 / (1 + exp(-clamped))
```

- A zero-advantage position maps to `0.5` (maximally uncertain).
- A live-3 advantage (~1000 eval points) maps to `≈ 0.73`.
- A live-4 advantage (~10 000 points) saturates near `1.0`.

Because the returned value is already in `[0, 1]` it plugs directly
into the PUCT `Q` term without any rescaling.

### Decisive-leaf shortcuts

Before falling back to the sigmoid, expansion checks:

- **Expansion move wins** — `_fast_check_winner` finds 5-in-a-row for
  the stone we just placed → value is `1.0` from that player's view.
- **Opponent can win immediately** — `_find_winning_move` finds an
  opponent reply that ends the game → value is `0.0`.

Either way the node is backpropagated with an exact terminal value, so
the tree learns forced wins/losses with perfect accuracy.

---

## Decision Shortcuts

Before starting any MCTS iterations, `get_move` short-circuits three
trivial cases:

| Case | Decision path | Cost |
|------|--------------|------|
| Only one candidate in range | `"single_candidate"` | O(1) |
| We can win in 1 ply | `"instant_win"` | O(C) candidates |
| Opponent can win in 1 ply — we must block | `"instant_block"` | O(C) candidates |

These checks use raw-grid helpers (`_fast_check_winner`,
`_find_winning_move`) so they are effectively free.

---

## VCF Safety Gate

MCTS can be seduced by long exploratory lines and miss a forced loss
that a narrow threat search catches instantly.  To hedge against this,
when `vcf_depth > 0` the brain runs a **VCF probe as the opponent**
before spinning up the tree:

```python
opp_vcf = _vcf(board, opponent, vcf_depth, is_attacker=True)
if opp_vcf is not None:
    # Opponent has a forced win via continuous fours.
    # Restrict root candidates to the proven blocking cells.
```

Blocking candidates come from two sources:

1. The opponent's first VCF attack move itself.
2. All four-threat attack/defence pairs discovered by
   `_find_four_threats(board, opponent)`.

When the restricted set is non-empty, MCTS continues with only those
moves — ensuring the brain cannot wander away from a proven threat.
The decision path is stamped as `"mcts_vcf_gated"` so post-game
analysis can tell the gate fired.

The gate also writes `opp_threats.vcf_result = "win"` into the
diagnostics, which is one of the triggers for `gomoku analyze`'s
`opp_forced_win` flag.

---

## Iteration Loop

A single iteration (`_run_iteration`) mutates a per-iteration copy of
the board grid — creating and tearing it down costs a shallow list
comprehension (`[row[:] for row in grid]`), roughly 50× faster than
`copy.deepcopy` used by `Board.place_stone`.

```
1. Selection
   while node has no untried moves AND has children:
       node = argmax_{child} PUCT(child)
       apply node.move to grid

2. Expansion  (if node still has untried moves)
   move, prior = node.pop_untried()        # random choice
   apply move to grid
   child = _Node(move, stone, untried=candidates(grid))
   compute child.untried_priors
   node.children.append(child)
   node = child

   # terminal shortcuts
   if move wins:          value = 1.0 (or 0.0)
   if opp can win next:   value = 0.0 (or 1.0)

3. Evaluation
   value = _evaluate_position(grid, root_stone)    # sigmoid ∈ [0, 1]

4. Backpropagation
   walk up to root:
       node.visits += 1
       node.wins  += value     if node.stone == root_stone
                     (1-value) otherwise
```

### Expansion order

`pop_untried()` picks a **uniformly random** untried move, not the
highest-prior one.  Selection (step 1) already prefers high-prior
children via PUCT, so the *order* in which children are created does
not matter — randomising keeps the tree from becoming structurally
biased when multiple moves share a prior.

---

## Diagnostics

When a diagnostics context is active, MCTSBrain records:

| Field | Value |
|-------|-------|
| `decision_path` | `"single_candidate"`, `"instant_win"`, `"instant_block"`, `"mcts"`, `"mcts_vcf_gated"` |
| `chosen_move` | The most-visited root child (or the shortcut move) |
| `search_stats.iterations` | Simulations run |
| `search_stats.nodes_searched` | `root.visits` (total tree samples) |
| `search_stats.time_ms` | Wall-clock search time |
| `top_candidates` | Top-10 root children by visits, with **`score_type="winrate"`** — score is `wins / visits ∈ [0, 1]` |
| `alternative_best` | Set when the top-visits candidate differs from the chosen move (never, unless a shortcut fired first) |
| `opp_threats.vcf_result` | `"win"` when the VCF safety gate fired |
| `notes` | `"Best: {move} visits={n}/{N} wr={rate}"`; VCF-gate explanations |

Because MCTS candidate scores are winrates (not eval units), the
`gomoku analyze` tool intentionally does **not** trigger
`bypassed_top_candidate` on MCTS diagnostics — the heuristic gates on
`score_type == "eval"` so it doesn't compare apples to oranges.

---

## Parameters

### Constructor / CLI parameters

| Parameter | Default | Range | Controls | Effect of increasing |
|-----------|---------|-------|----------|----------------------|
| `iterations` | 10 000 | 100 – 1 000 000 | Simulation budget per move | Linearly stronger, linearly slower |
| `exploration` | 2.0 | 0.1 – 5.0 | PUCT constant `c` | More exploration, less exploitation |
| `dirichlet_alpha` | 0.03 | 0.0 – 1.0 | Dirichlet shape | Larger = flatter noise |
| `dirichlet_weight` | 0.25 | 0.0 – 1.0 | Noise mix weight | `0.0` = deterministic priors; `1.0` = pure noise |
| `vcf_depth` | 10 | 0 – 40 | Opponent VCF probe depth for the safety gate | Deeper probe, more blocks enforced; `0` disables the gate |

### Internal constants

| Constant | Value | Purpose |
|----------|-------|---------|
| Candidate radius | 2 | Cells-within-radius generation at root and per expansion |
| Sigmoid scale `k` | 1000 | Maps eval-point advantage to winrate; live-3 ≈ 0.73 |
| Defensive weight | 0.8× | Applied to opponent pattern scores in priors and leaf eval |
| Prior baseline | `+1.0` | Ensures no candidate receives zero prior |

### Tuning guidelines

| Scenario | Recommendation |
|----------|----------------|
| Weak baseline for comparison | `iterations=1000` |
| Default | `iterations=10_000, exploration=2.0, dirichlet_weight=0.25, vcf_depth=10` |
| Deterministic self-play | `dirichlet_weight=0.0` |
| Open-position exploration | Increase `exploration` to 3.0 – 4.0 |
| Tactical sharpness | Increase `vcf_depth` to 20 – 30 (cheap, catches more forced losses) |
| Strongest available | `iterations=100_000, vcf_depth=20` — time scales linearly |

---

## Example Scenarios

### Example 1 — PUCT focuses effort on live threats

```
Root candidates (15-cell radius around mid-board):
  A) live-3 cell          prior = 0.42
  B) live-2 cell          prior = 0.08
  C) random empty cell    prior = 0.004  (× 27 others)

First 100 simulations distribute ~80% of visits to A, ~15% to B,
~5% spread over the rest — vs ~6.7% uniform with plain UCT.
```

### Example 2 — Dirichlet noise reveals a missed move

```
Priors (deterministic):
  A: 0.80      B: 0.15      C: 0.05
With Dirichlet(α=0.03, weight=0.25):
  A: 0.62      B: 0.12      C: 0.26  ← noise vector spiked on C

C is now visited often enough to discover that it leads to a fork
win that the heuristic under-rated.  Visit-proportional final pick
still prefers whichever move has the highest simulated winrate.
```

### Example 3 — VCF gate saves the brain

```
Position: White has a 4-ply VCF threatening five-in-a-row.
Without gate:  MCTS explores broadly, picks a positional move,
                White wins next turn.
With gate:     _vcf(board, White, depth=10) = (H8)
                Candidates restricted to blocking cells only.
                MCTS picks the best block, game continues.
```

### Example 4 — Heuristic eval vs random rollout

```
Leaf position: Black has live-3, White has dead-2.

Random rollout (80 random plies to terminal):
  ~50% win for either side (high variance, ~10 ms per rollout)

Heuristic eval:
  score ≈ +800 → sigmoid(0.8) = 0.69
  Deterministic, ~0.1 ms per call, captures the positional edge.
```

### Example 5 — Decisive-move shortcut avoids wasted budget

```
Expansion places a stone creating 5-in-a-row.
Instead of heuristic eval, _fast_check_winner flags the terminal win:
    value = 1.0 (from placer's perspective)
Backpropagation records a perfect win on this branch — no sigmoid noise
masks the result, and PUCT will prefer this path deterministically.
```

---

## Configuration & Usage

### Python API

```python
from gomoku_brain.mcts import MCTSBrain

# Defaults
brain = MCTSBrain()
move = brain.get_move(board, stone)

# Deterministic play (no root noise)
brain = MCTSBrain(
    iterations=50_000,
    exploration=2.0,
    dirichlet_weight=0.0,
    vcf_depth=20,
)
```

### CLI

```bash
# Default settings
uv run gomoku-cli play --black mcts --white alphabeta

# Inspect MCTS decisions after the game
uv run gomoku-cli play --black mcts --white alphabeta --record --diagnostics
uv run gomoku-cli analyze recordings/gomoku_*.diag.jsonl
```

### gomoku.toml

```toml
[brains.mcts]
type = "mcts"
iterations = 10_000
exploration = 2.0
dirichlet_alpha = 0.03
dirichlet_weight = 0.25
vcf_depth = 10

# Stronger variant as a named alias
[brains.mcts_strong]
type = "mcts"
iterations = 100_000
exploration = 2.0
dirichlet_weight = 0.15
vcf_depth = 20
```

---

## Brain Comparison

| Aspect | MCTS | Alpha-Beta | Threat |
|--------|------|-----------|--------|
| Tree shape | Wide, shallow, visit-weighted | Narrow, deep, alpha-beta pruned | Very narrow, very deep (forcing) |
| Leaf value | Heuristic eval → sigmoid | Heuristic eval (with quiescence) | Terminal win / loss proof |
| Uses priors | ✅ pattern heuristic + Dirichlet | ❌ | ❌ |
| Forced-win detection | VCF safety gate at root | VCF pre-check + quiescence | Core strategy (VCF / VCT / PNS) |
| Strength: opening | Good (broad exploration) | Good | Falls back to alpha-beta |
| Strength: midgame | Medium | Good | Good |
| Strength: endgame | May drift without gate | May miss deep forced wins | Excellent |
| Budget knob | `iterations` (linear scaling) | `depth` (exponential scaling) | `vcf_depth`, `vct_depth`, `pns_nodes` |
| Key advantage | Explores broadly; no depth cliff | Positional accuracy with pruning | Deep tactical proofs |
| Key weakness | Slower per-move; noisy eval | Fixed depth horizon | Blind to quiet positional play |
