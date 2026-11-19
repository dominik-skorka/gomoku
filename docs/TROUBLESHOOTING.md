# Troubleshooting Guide — Analysing Games for Blunders, Mistakes & Inefficiencies

A step-by-step manual for reviewing recorded Gomoku games to find where a
brain went wrong, why, and how to fix it.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick-Start Checklist](#quick-start-checklist)
3. [Step 1 — Replay the Game](#step-1--replay-the-game)
4. [Step 2 — Run `gomoku analyze`](#step-2--run-gomoku-analyze)
5. [Step 3 — Identify Blunders](#step-3--identify-blunders)
6. [Step 4 — Identify Mistakes](#step-4--identify-mistakes)
7. [Step 5 — Identify Inefficiencies](#step-5--identify-inefficiencies)
8. [Step 6 — Root-Cause Analysis](#step-6--root-cause-analysis)
9. [Step 7 — Remediation](#step-7--remediation)
10. [Common Patterns & Fixes](#common-patterns--fixes)
11. [Analysis Scripts](#analysis-scripts)
12. [Worked Example](#worked-example)

---

## Prerequisites

- A recorded game file (`recordings/gomoku_*.json`).
- Optionally, a diagnostics file (`recordings/gomoku_*.diag.jsonl`).
- Python environment with the gomoku workspace installed (`uv sync`).

If you don't have diagnostics, re-play the same matchup with `--diagnostics`:

```bash
uv run gomoku play --black threat --white alphabeta \
    --record --diagnostics --diagnostics-dir recordings/
```

---

## Quick-Start Checklist

```
□ Replay the game visually with playback (diagnostics auto-overlay)
□ Run `gomoku analyze <file>.diag.jsonl` — start from its flagged moves
□ For each flagged move, check:
   □ reasons      — which heuristic(s) fired?
   □ decision_path — which algorithm chose the move?
   □ top_candidates / alternative_best — was a better move available?
   □ our_threats / opp_threats — were threats missed?
   □ eval_before / eval_after — did the eval swing sharply?
   □ think_time_ms — was the brain under time pressure?
   □ notes — did the brain log any warnings?
□ Classify as blunder / mistake / inefficiency
□ Determine root cause
□ Apply fix (config tuning, depth increase, etc.)
```

---

## Step 1 — Replay the Game

Start by watching the game move-by-move to build intuition about where things
went wrong:

```bash
uv run gomoku playback recordings/gomoku_20260405_143012.json
```

**What to look for:**

| Signal | Meaning |
|--------|---------|
| A stone placed far from the action | Possible blunder — brain ignored an urgent threat |
| Opponent builds an uncontested line | Missed defensive opportunity |
| Sudden loss after a long advantage | Horizon effect or missed VCF |
| Very fast move in a complex position | Brain may have used a shallow fallback |
| Very slow move (>5s) | Search struggled — position may be ambiguous |

Use `←` / `→` to navigate.  Note the move numbers that look suspicious.

When a companion `.diag.jsonl` file sits next to the recording, playback
overlays per-move brain internals under each board (decision path, TT hit
rate, threat counts, top candidates, `alternative_best` warnings).

---

## Step 2 — Run `gomoku analyze`

`gomoku analyze` parses the diagnostics file and flags suspect moves for
you.  Start here before diving into raw JSON:

```bash
uv run gomoku analyze recordings/gomoku_20260405_143012.diag.jsonl
```

Output is two tables:

1. **Per-player statistics** — moves, total / avg think time, avg and max
   depth, total nodes, blunder count.
2. **Flagged moves** — every move that tripped one or more heuristics.

### Flagging heuristics

| Reason | Fires when | Default threshold |
|--------|-----------|-------------------|
| `eval_drop` | Your own `eval_after` dropped sharply vs your previous move | ≥ 500 eval units |
| `bypassed_top_candidate` | `alternative_best` outscored the chosen move (eval-typed candidates only) | ≥ 500 eval units |
| `time_spike` | Think time much larger than this player's running average | ≥ 2× average **and** ≥ 200 ms |
| `opp_double_four` | Opponent has ≥ 2 fours after your move and you did not pick a `block_*` decision path | — |
| `opp_forced_win` | Opponent's `vcf_result` / `vct_result` is `"win"` | — |

Thresholds are module-level constants in `gomoku_cli.analyze`
(`BLUNDER_EVAL_DROP`, `ALTERNATIVE_GAP`, `TIME_SPIKE_FACTOR`,
`TIME_SPIKE_MIN_MS`).

### Machine-readable output

```bash
uv run gomoku analyze recordings/gomoku_20260405_143012.diag.jsonl --json \
    | jq '.flagged[] | select(.reasons | contains(["eval_drop"]))'
```

### Manual deep-dive (optional)

When the analyzer flags a move you want to inspect by hand, load the JSONL
directly:

```python
import json
from pathlib import Path

lines = Path("recordings/gomoku_20260405_143012.diag.jsonl").read_text().splitlines()
moves = [json.loads(line) for line in lines if '"move_number"' in line]
```

### Quick overview script

```python
for m in moves:
    path = m.get("decision_path", "?")
    think = m.get("think_time_ms", 0)
    ev_before = m.get("eval_before")
    ev_after = m.get("eval_after")
    swing = (ev_after - ev_before) if ev_before is not None and ev_after is not None else None
    swing_str = f"Δ{swing:+.0f}" if swing is not None else ""
    print(f"  #{m['move_number']:3d}  {path:20s}  {think:6d}ms  {swing_str}")
```

Look for:
- **Large negative eval swings** (e.g., Δ-3000) — the chosen move made the
  position significantly worse.
- **Decision path changes** — a brain switching from `vcf_win` to
  `alphabeta` mid-game may signal it lost the tactical thread.

---

## Step 3 — Identify Blunders

A **blunder** is a move that loses a clearly winning or equal position.
Typical eval swing: **Δ ≤ −3,000** or transition from "winning" to "losing".

### Signs of a blunder

| Diagnostic field | Red flag |
|-----------------|----------|
| `eval_before` → `eval_after` | Large negative swing (≥ 3,000 points) |
| `opp_threats.four_threats` | > 0 and brain didn't block |
| `opp_threats.vcf_result` | `"win"` — opponent has a forced win the brain didn't see |
| `our_threats.vcf_result` | Was `"win"` last turn but now `None` — brain abandoned its own VCF |
| `top_candidates` / `alternative_best` | Top candidate ≫ chosen move's score (analyzer reports this as `bypassed_top_candidate`) |

### How to check

```python
for m in moves:
    eb = m.get("eval_before")
    ea = m.get("eval_after")
    if eb is not None and ea is not None and (ea - eb) < -3000:
        print(f"  BLUNDER at move #{m['move_number']}: eval {eb:.0f} → {ea:.0f}")
        print(f"    Path: {m['decision_path']}")
        print(f"    Opp threats: {m.get('opp_threats', {})}")
        print(f"    Notes: {m.get('notes', [])}")
```

---

## Step 4 — Identify Mistakes

A **mistake** is a suboptimal move that doesn't immediately lose but weakens
the position.  Typical eval swing: **Δ −500 to −3,000**.

### Signs of a mistake

| Diagnostic field | Warning |
|-----------------|---------|
| `eval_before` → `eval_after` | Moderate negative swing |
| `top_candidates` | #2 or #3 candidate scored significantly higher than chosen move |
| `decision_path` | `"alphabeta"` when a threat search might have found something |
| `our_threats.three_threats` | > 0 but brain didn't follow up on them |
| `opp_threats.three_threats` | > 0 and brain didn't block |

### Common mistake patterns

1. **Positional drift** — brain plays safe positional moves instead of
   pressing a tactical advantage.
2. **Premature fallback** — VCF/VCT didn't find a win (depth too shallow),
   so the brain fell back to alpha-beta which played a weaker move.
3. **Wrong block** — opponent has multiple threats, brain blocked the less
   urgent one.

---

## Step 5 — Identify Inefficiencies

An **inefficiency** doesn't lose material or position, but wastes time or
resources — leading to weaker play in subsequent moves.

### Signs of inefficiency

| Diagnostic field | Warning |
|-----------------|---------|
| `think_time_ms` | >> 5,000 ms for a non-critical move |
| `search_stats.nodes_searched` | Very high node count with no eval improvement |
| `search_stats.tt_hits` / `tt_probes` | Hit rate `tt_hits/tt_probes` near zero — TT not helping |
| `search_stats.aspiration_resets` | > 2 — windows keep failing |
| `decision_path` | Expensive algorithm used when position is simple |
| `search_stats.depth_reached` | < configured depth — search timed out |

### Common inefficiency patterns

1. **Time pressure collapse** — early moves consumed the time budget
   (`time_limit_s`), forcing shallower searches later.
2. **TT pollution** — transposition table grew too large or had low hit rate.
3. **Unnecessary PNS search** — PNS ran for 5 seconds on a position with no
   threats.
4. **Aspiration window thrashing** — repeated fail-low/fail-high re-searches
   wasted iterations.

---

## Step 6 — Root-Cause Analysis

For each identified issue, match it to a root cause:

### Blunder root causes

| Symptom | Likely cause | Evidence |
|---------|-------------|----------|
| Didn't block opponent's four | VCF depth too shallow | `opp_threats.vcf_result` is `None` but opponent wins next |
| Abandoned own winning sequence | VCT depth too shallow or PNS budget exhausted | Last move had `vcf_result: "win"`, this move has `None` |
| Played far from threats | Candidate radius too small or eval didn't see the threat | `top_candidates` list doesn't include the critical cell |
| Blocked wrong cell | VCF-aware defence picked wrong breaker | `notes` show VCF-defence was used |

### Mistake root causes

| Symptom | Likely cause |
|---------|-------------|
| Alpha-beta fallback missed tactical shot | `depth` too low; increase or use threat brain |
| Brain doesn't pursue live-threes | VCT/PNS not searching three-threats deeply enough |
| Ignored opponent build-up | Blocking is under-ranked: move priors weigh the opponent's lines 0.8× (the evaluation itself 1.0×) |

### Inefficiency root causes

| Symptom | Likely cause |
|---------|-------------|
| Move took > 10s but eval barely changed | PNS budget too high for this position type |
| Aspiration resets > 2 | Eval instability between depths — try wider initial window |
| Low TT hit rate | Different board representations or hash collisions |

---

## Step 7 — Remediation

### Quick fixes (configuration)

| Issue | Fix |
|-------|-----|
| VCF too shallow → missed forced win | Increase `vcf_depth` (e.g., 20 → 30) |
| VCT too shallow → missed threat chain | Increase `vct_depth` (e.g., 12 → 16) |
| PNS gives up too early | Increase `pns_nodes` (shipped: 120k, +47 Elo over 60k at the same think time once PNS was made faster) or `pns_time_s` |
| Alpha-beta too weak positionally | Increase `depth` (e.g., 4 → 5) |
| Opening phase wastes time | Add an opening book (`opening_book = "..."`) |
| Moves too slow | Set `time_limit_s` or reduce depths |
| Quiescence misses tactics | Increase `qs_vcf_depth` (alpha-beta only) |
| Late-game fumbles after good opening | Use threat brain instead of alpha-beta for endgame |

### Structural fixes (code)

| Issue | Fix |
|-------|-----|
| Defensive weight too low | Raise the 0.8× opponent weight of the move priors (`priors.py` and `compute_priors_grid` in `evaluate_jit.py`); the evaluation itself weighs both sides 1.0× |
| Fork detection misses complex forks | Lower `_FORK_THRESHOLD` or increase `_FORK_BONUS` |
| Candidate radius misses distant threats | Increase radius from 2 to 3 (expensive) |
| Counter-threats not caught | Verify `_vct_defence_moves()` covers the pattern |

---

## Common Patterns & Fixes

### Pattern 1: "The horizon blunder"

```
Move 20: eval +500 (alphabeta, depth 4)
Move 21: opponent plays live-four → eval -10000
```

**Cause:** Alpha-beta search ended at depth 4, just before the opponent's
winning threat materialized.

**Fix:** Enable quiescence search (`qs_vcf_depth = 12`) or use the threat
brain which naturally searches much deeper for forcing lines.

### Pattern 2: "The abandoned VCF"

```
Move 15: decision_path="vcf_win", eval +8000 (own VCF found)
Move 17: decision_path="alphabeta", eval +200 (VCF gone)
```

**Cause:** The brain found a VCF but the opponent's response disrupted it.
The brain didn't find the VCF again in later positions (different board state).

**Fix:** Check if the opponent's counter-move was a counter-threat that VCT
would have anticipated.  If so, switch `threat_search` to `"pns"` for deeper
analysis.  If the VCF was provably unsound, the detection was correct but the
brain's fallback should have preserved the tactical advantage — increase
alpha-beta `depth`.

### Pattern 3: "The slow opening"

```
Move 1: think_time_ms=12000, decision_path="alphabeta"
Move 2: think_time_ms=15000, decision_path="alphabeta"
```

**Cause:** No opening book configured.  Alpha-beta exhaustively searches an
almost-empty board where most moves are equivalent.

**Fix:** Use an opening book (`opening_book = "..."`) or increase
`early_game_max_candidates` to reduce branching.

### Pattern 4: "The PNS timeout"

```
Move 25: think_time_ms=5100, decision_path="alphabeta", notes=["No forced win found (VCF/PNS/VCT)", ...]
```

**Cause:** PNS ran for its full time budget (5s) without proving or disproving
a win, then the brain fell back to alpha-beta.

**Fix:** If this happens often in games that aren't tactically sharp, the
position may be too quiet for threat search.  Consider reducing `pns_time_s`
(e.g., 2.0) so the brain reaches alpha-beta faster, or increasing `pns_nodes`
if the position *should* have a provable win.

### Pattern 5: "The missed fork"

```
Move 30: eval +200, top_candidates show F5 at +200
         But F5 would create a double-three fork (→ unstoppable)
```

**Cause:** The evaluation didn't score the fork highly enough, or the search
didn't look deep enough to see that the fork leads to an unavoidable win.

**Fix:** Verify fork detection in `evaluate.py` — the cell should have
`_threat_lines_at() ≥ 2` and receive the `_FORK_BONUS`.  If the fork involves
a three-threat chain, increase `vct_depth` or `pns_nodes`.

---

## Analysis Scripts

The primary analyser is the built-in CLI command:

```bash
uv run gomoku analyze recordings/gomoku_*.diag.jsonl
uv run gomoku analyze recordings/gomoku_*.diag.jsonl --json | jq '.flagged'
```

For narrower follow-up queries, pipe `gomoku analyze --json` through
`jq` rather than reaching for one-off scripts:

```bash
# Moves where the brain bypassed a clearly better candidate
uv run gomoku analyze recordings/*.diag.jsonl --json \
    | jq '.flagged[] | select(.reasons | contains(["bypassed_top_candidate"]))'

# Per-player average think-time
uv run gomoku analyze recordings/*.diag.jsonl --json \
    | jq '.player_stats | to_entries[] | {stone:.key, avg_ms:.value.avg_think_ms}'

# All decision paths the brain took, sorted by frequency
jq -r '.decision_path' recordings/*.diag.jsonl 2>/dev/null \
    | sort | uniq -c | sort -rn
```

If you need an analysis that `gomoku analyze` can't express, the
`MoveDiagnostics` Pydantic model in `gomoku.diagnostics` deserialises
each line of the `.diag.jsonl` file directly, so a five-line Python
script can usually answer the question.

---

## Worked Example

### Scenario

Game: `gomoku_20260405_143012.json` — Threat (Black) vs AlphaBeta (White).
White wins at move 42 despite Black having the first-move advantage.

### Step 1: Replay

```bash
uv run gomoku playback recordings/gomoku_20260405_143012.json
```

Navigate to move 25 — Black seems to have the initiative. By move 30, White
has built a row on row 7 uncontested. By move 38, White has a live-four.

### Step 2: Run the analyzer

```bash
uv run gomoku analyze recordings/gomoku_20260405_143012.diag.jsonl
```

Suppose it reports:

```
Flagged moves (2)
#   Stone  Path               Reasons                  Details
27  Black  vcf_aware_search   eval_drop                eval_drop=820 eval_prev=+100 eval_after=-720
30  Black  alphabeta          eval_drop                eval_drop=540 eval_prev=-170 eval_after=-710
```

For additional context, load the JSONL directly and inspect neighbours of
the flagged moves:

```python
# Load diagnostics
lines = Path("recordings/gomoku_20260405_143012.diag.jsonl").read_text().splitlines()
moves = [json.loads(l) for l in lines if '"move_number"' in l]

for m in moves[24:35]:  # moves 25-35
    eb, ea = m.get("eval_before"), m.get("eval_after")
    swing = (ea - eb) if eb and ea else None
    print(f"#{m['move_number']:3d} {m['decision_path']:20s} "
          f"opp_fours={m['opp_threats']['four_threats']} "
          f"Δ={swing}")
```

Output:
```
# 25 vcf_aware_search       opp_fours=0  Δ=-200
# 26 alphabeta              opp_fours=0  Δ=+100
# 27 vcf_aware_search       opp_fours=1  Δ=-800     ← flagged
# 28 alphabeta              opp_fours=0  Δ=+50
# 29 alphabeta              opp_fours=0  Δ=-100
# 30 alphabeta              opp_fours=0  Δ=-500     ← flagged
```

### Step 3: Diagnosis

**Move 27** — Black ran `vcf_aware_search`, meaning White had a VCF that
Black was trying to break.  Black's response neutralised the VCF but let
White's row-7 grow uncontested (Δ=-800). The VCF-breaker wasn't the one
that also helped positionally.

**Move 30** — Black's plain alpha-beta played a general move while White
had been building on row 7 for three turns.

### Step 4: Root cause

The VCF-aware defence (Phase 3) correctly broke White's VCF at move 27, but
among multiple VCF-breakers it chose one far from row 7.  The positional
evaluation didn't rate blocking row 7 highly enough because only 3 stones
were in that row (dead-three = 100 points).

### Step 5: Fix

Increase `_OPEN_THREE_BONUS` from 1,500 to 2,000 in `evaluate.py`, or
deepen the alpha-beta fallback (`[brains.threat.fallback] depth`, 5 in the
shipped config) so the brain sees the row-7 threat materializing earlier.
Alternatively, configure a custom alias:

```toml
[brains.threat_deep]
type = "threat"
vcf_depth = 24       # deeper VCF
pns_nodes = 100_000  # larger PNS budget

[brains.threat_deep.fallback]
depth = 6            # deeper alpha-beta fallback
```

Measure such a change with `gomoku review` and `gomoku match` (see the
main README) before adopting it.
