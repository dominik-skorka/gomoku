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
│   ├── Opponent wins next               → "instant_block"
│   └── We have a VCF (vcf_depth > 0)    → "vcf_win"
│
├── VCF safety gate (opponent forced-win check)
│   └── Restrict candidates to the durable VCF breakers
│
├── Build root node (or reuse the kept subtree, reuse_tree = true)
│   ├── Generate candidates (radius-2 of occupied)
│   ├── Compute per-move priors (pattern score → normalised)
│   └── Mix in Dirichlet noise (not on a gated or reused root)
│
├── Until `iterations` ran or `time_limit_s` passed:
│   └── _run_iteration(root, state, stone)
│       │
│       ├── 1. Selection       PUCT descent; the best untried move
│       │                      competes as an unvisited child (FPU)
│       │                      Moves of a node are generated at its
│       │                      first expansion (lazy)
│       │
│       ├── 2. Expansion       Pop the highest-prior untried move,
│       │                      classify the new node: terminal
│       │                      (decided), forced (one block) or open
│       │
│       ├── 3. Evaluation      Terminal: its stored 0 / 1 value
│       │                      Otherwise heuristic eval of the leaf
│       │                      (+ tempo for the side to move)
│       │                      → value in [0, 1] (root-relative)
│       │
│       └── 4. Backpropagation Walk up to root updating visits / wins
│                              (flip perspective on opponent nodes)
│
└── Return the child of root with the most visits (most robust choice)
```

The search runs on one mutable `SearchState`: each iteration `make`s
the moves on the way down and `unmake`s them on the way back, so no
`Board` is built per iteration.

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
is scored by the pattern scorer `gomoku_brain.priors.compute_priors`
(shared with alpha-beta):

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

- **`dirichlet_alpha`** (default `0.25`) controls the noise shape.
  Small α → concentrated (one move dominates the noise vector); larger
  α → flat (uniform noise).  AlphaZero scaled α inversely to the number
  of legal moves (0.3 chess, 0.15 shogi, 0.03 Go); `0.25` suits the ~50
  candidates of a gomoku root.
- **`dirichlet_weight`** (default `0.0`) controls how much the noise
  displaces the priors.  `0.0` disables noise entirely (deterministic
  play); `1.0` replaces priors with pure noise.

**Off by default.**  AlphaZero adds the noise to explore while
generating self-play training data; this brain does not learn, so the
noise only takes prior mass from the best moves.  Measured with
`gomoku match` (100 games, 2 000 iterations), weight `0.0` vs `0.25`
scored 50.0% (+0 Elo, 95% interval −54 to +54).  Turn it on only to vary
or weaken play.

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

This brain instead scores the leaf with the static evaluator shared
with alpha-beta (`gomoku_brain.evaluate.evaluate_board`, from the root
player's view) and maps it through a sigmoid, `k = 1000`:

```python
clamped = max(-10, min(10, score / 1000))
value = 1 / (1 + exp(-clamped))
```

- A zero-advantage position maps to `0.5` (maximally uncertain).
- A live-3 advantage (~1000 eval points) maps to `≈ 0.73`.
- A live-4 advantage (~10 000 points) saturates near `1.0`.

Because the returned value is already in `[0, 1]` it plugs directly
into the PUCT `Q` term without any rescaling.

### Tempo for the side to move

The static evaluation does not know whose turn it is, but an open
three is nearly a win for the side to move and only a threat to block
for the other side.  When the side to move at a leaf can make a live
four or a double four next move, and the other side has no five threat,
the score moves by `_TEMPO_WIN = 5000` towards the side to move before
the sigmoid.

### Decided and forced nodes

When a node is created, `_node_status` classifies the position after
its move:

| Position | Node |
|----------|------|
| The move made five | Terminal: win for the mover |
| The side to move can make five | Terminal: loss for the mover |
| The mover threatens five on two or more cells (e.g. open four) | Terminal: win for the mover |
| The mover threatens five on one cell | Forced: the block is its only move |
| Otherwise | Open: moves generated at its first expansion |

A terminal node stores its value (`_Node.terminal`, from the root
player's view), has no moves and is never expanded: every visit backs
up the same exact value, so a proven result is never averaged with
heuristic values.

### MCTS-Solver

Proofs propagate up the tree (Winands, Björnsson, Saito, *Monte-Carlo
Tree Search Solver*, CG 2008).  When a node is proven, its parent is:

- **lost for its mover** when that child wins for the player choosing
  at the parent;
- **won for its mover** when every move of the parent has been expanded
  and every child loses for the chooser.

Proven nodes use their exact value (1 or 0) as Q in PUCT; selection
never descends into a proven loss while another child is unproven, and
a node whose children are all proven losses expands its next move.  At
the root a proven win is played (`decision_path = "mcts_proven_win"`),
proven losses are avoided, and the search stops as soon as the root is
solved.

Measured with `gomoku match` (200 games, 2 000 iterations) against the
same brain without the solver: −2 Elo for the solver (95% interval −31
to +28), ~7% more think time.  It is kept for its guarantees (a proven
win is played, a proven loss avoided, a solved root ends the search),
not for strength.

Tried and removed (#97): implicit minimax backups (Lanctot et al., CIG
2014; weight 0.2 / 0.4 / 0.6: +7, −24, −49 Elo, 10–20% slower) and
MCTS–minimax hybrids (Baier, Winands; at equal time: shallow-search
root priors +3, the same at visited nodes −156, VCF at visited nodes
+3).  See #99 and #100 for the measurements.

---

## Decision Shortcuts

Before starting any MCTS iterations, `get_move` short-circuits three
trivial cases:

| Case | Decision path | Cost |
|------|--------------|------|
| Only one candidate in range | `"single_candidate"` | O(1) |
| We can win in 1 ply | `"instant_win"` | O(C) candidates |
| Opponent can win in 1 ply — we must block | `"instant_block"` | O(C) candidates |
| We have a VCF of at most `vcf_depth` (as AlphaBeta's root) | `"vcf_win"` | VCF search |

---

## VCF Safety Gate

MCTS can be seduced by long exploratory lines and miss a forced loss
that a narrow threat search catches instantly.  To hedge against this,
when `vcf_depth > 0` the brain runs a **VCF probe as the opponent**
before spinning up the tree:

```python
opp_vcf = _vcf(board, opponent, vcf_depth)
if opp_vcf is not None:
    # Opponent has a forced win via continuous fours.
    durable, breakers = classify_vcf_breakers(board, candidates, stone, vcf_depth)
    candidates = durable or breakers
```

A *breaker* is a candidate after which the opponent's VCF is gone; a
*durable* breaker also keeps it gone after the forced reply to any
four it makes (`classify_vcf_breakers`, shared with AlphaBeta and
ThreatBrain).

When the restricted set is non-empty, MCTS continues with only those
moves — ensuring the brain cannot wander away from a proven threat.
The decision path is stamped as `"mcts_vcf_gated"` so post-game
analysis can tell the gate fired.

The gate also writes `opp_threats.vcf_result = "win"` into the
diagnostics, which is one of the triggers for `gomoku analyze`'s
`opp_forced_win` flag.

---

## Iteration Loop

```
1. Selection
   loop:
       if node is terminal: stop
       if node's moves are not generated yet: generate them (+ priors)
       if node has no children, or its best untried move outscores
          every child (FPU, below): stop
       node = argmax_{child} PUCT(child); state.make(node.move)

   if node is terminal: backpropagate node.terminal, done

2. Expansion  (if node still has untried moves)
   move, prior = node.pop_untried()        # highest prior
   state.make(move)
   classify (terminal / forced / open), add the child
   if terminal: backpropagate its value, done

3. Evaluation
   value = _leaf_value(state, root_stone, side_to_move)   # ∈ [0, 1]

4. Backpropagation
   walk up to root:
       node.visits += 1
       node.wins  += value     if node.stone == root_stone
                     (1-value) otherwise
   unmake every move of the iteration
```

### Expansion order and First Play Urgency

Untried moves are kept sorted by prior; `pop_untried()` takes the
highest.  An untried move competes with the existing children as an
unvisited child:

```
PUCT_untried = FPU + c · P_best_untried · √N
FPU          = (value of the node for the player choosing) − fpu_reduction
```

A new child is expanded only when this beats the best child, so a node
does not have to expand all its ~50–140 moves before the search goes
deeper.  In a quiet 6-stone position at 10 000 iterations this took the
maximum depth from 7 to 16 plies.

---

## Diagnostics

When a diagnostics context is active, MCTSBrain records:

| Field | Value |
|-------|-------|
| `decision_path` | `"single_candidate"`, `"instant_win"`, `"instant_block"`, `"vcf_win"`, `"mcts"`, `"mcts_vcf_gated"`, `"mcts_proven_win"` |
| `chosen_move` | The most-visited root child (or the shortcut move) |
| `search_stats.iterations` | Simulations run (fewer than `iterations` when `time_limit_s` stopped the search) |
| `search_stats.nodes_searched` | `root.visits` (total tree samples) |
| `search_stats.time_ms` | Wall-clock search time |
| `top_candidates` | Top-10 root children by visits, with **`score_type="winrate"`** — score is `wins / visits ∈ [0, 1]` |
| `alternative_best` | Set when the top-visits candidate differs from the chosen move (never, unless a shortcut fired first) |
| `opp_threats.vcf_result` | `"win"` when the VCF safety gate fired |
| `our_threats.vcf_result` | `"win"` when our own VCF was played |
| `notes` | `"Best: {move} visits={n}/{N} wr={rate}"`; VCF-gate explanations; `"Reused tree: …"` |

Because MCTS candidate scores are winrates (not eval units), the
`gomoku analyze` tool intentionally does **not** trigger
`bypassed_top_candidate` on MCTS diagnostics — the heuristic gates on
`score_type == "eval"` so it doesn't compare apples to oranges.

---

## Parameters

### Constructor / CLI parameters

| Parameter | Default | Range | Controls | Effect of increasing |
|-----------|---------|-------|----------|----------------------|
| `iterations` | 10 000 | 1 – 1 000 000 | Simulation budget per move | Linearly stronger, linearly slower |
| `time_limit_s` | 0 | 0 – 600 | Hard time limit per move (`0` = unlimited) | The search stops at the deadline |
| `exploration` | 2.0 | 0.1 – 5.0 | PUCT constant `c` | More exploration, less exploitation |
| `fpu_reduction` | 0.2 | 0.0 – 1.0 | First Play Urgency: Q of an untried move = node value − this | Deeper, narrower tree |
| `reuse_tree` | false (`gomoku.toml`: true) | bool | Continue from the kept subtree of the opponent's reply | More visits per move when the same brain plays on |
| `dirichlet_alpha` | 0.25 | 0.0 – 1.0 | Dirichlet shape | Larger = flatter noise |
| `dirichlet_weight` | 0.0 | 0.0 – 1.0 | Noise mix weight | `0.0` = deterministic priors; `1.0` = pure noise |
| `vcf_depth` | 10 (`gomoku.toml`: 20) | 0 – 40 | Own VCF search and opponent VCF probe for the safety gate | Deeper probes; `0` disables both |

### Internal constants

| Constant | Value | Purpose |
|----------|-------|---------|
| Candidate radius | 2 | Cells-within-radius generation at root and per expansion |
| Sigmoid scale `k` | 1000 | Maps eval-point advantage to winrate; live-3 ≈ 0.73 |
| Defensive weight | 0.8× | Applied to opponent pattern scores in priors and leaf eval |
| Prior baseline | `+1.0` | Ensures no candidate receives zero prior |
| `_TEMPO_WIN` | 5000 | Leaf score shift for a side to move that can make a live four / double four |

### Tuning guidelines

| Scenario | Recommendation |
|----------|----------------|
| Weak baseline for comparison | `iterations=1000` |
| Default | `iterations=10_000, exploration=2.0, dirichlet_weight=0.0, vcf_depth=10` |
| Varied / weaker play | `dirichlet_weight=0.25` |
| Bounded think time | `time_limit_s=2.0` with a large `iterations` |
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
With Dirichlet(α=0.25, weight=0.25):
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
Expansion places a stone making an open four.
_node_status sees two five threats the opponent cannot both block:
    terminal = 1.0 (from the placer's perspective)
The node is never expanded; every visit backs up the same exact value,
so no sigmoid noise masks the result.
```

---

## Configuration & Usage

### Python API

```python
from gomoku_brain.mcts import MCTSBrain

# Defaults
brain = MCTSBrain()
move = brain.get_move(board, stone)

# Bigger budget, bounded time, tree kept between moves
brain = MCTSBrain(
    iterations=50_000,
    time_limit_s=5.0,
    vcf_depth=20,
    reuse_tree=True,
)
```

### CLI

```bash
# Default settings
uv run gomoku play --black mcts --white alphabeta

# Inspect MCTS decisions after the game
uv run gomoku play --black mcts --white alphabeta --record --diagnostics
uv run gomoku analyze recordings/gomoku_*.diag.jsonl
```

### gomoku.toml

```toml
[brains.mcts]
type = "mcts"
iterations = 10_000
time_limit_s = 0.0
exploration = 2.0
fpu_reduction = 0.2
vcf_depth = 20
reuse_tree = true
dirichlet_alpha = 0.25
dirichlet_weight = 0.0

# Stronger variant as a named alias
[brains.mcts_strong]
type = "mcts"
iterations = 100_000
time_limit_s = 10.0
vcf_depth = 20
reuse_tree = true
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
