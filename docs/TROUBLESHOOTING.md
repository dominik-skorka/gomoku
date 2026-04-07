# Troubleshooting Guide — Analysing Games for Blunders, Mistakes & Inefficiencies

A step-by-step manual for reviewing recorded Gomoku games to find where a
brain went wrong, why, and how to fix it.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick-Start Checklist](#quick-start-checklist)
3. [Step 1 — Replay the Game](#step-1--replay-the-game)
4. [Step 2 — Enable & Read Diagnostics](#step-2--enable--read-diagnostics)
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
uv run gomoku-cli play --black threat --white alphabeta \
    --record --diagnostics --diagnostics-dir recordings/
```

---

## Quick-Start Checklist

```
□ Replay the game visually with playback
□ Note the move where the losing side lost the initiative
□ Open the diagnostics file for that game
□ For each suspicious move, check:
   □ decision_path — which algorithm chose the move?
   □ top_candidates — was a better move available?
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
uv run gomoku-cli playback recordings/gomoku_20260405_143012.json
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

---

## Step 2 — Enable & Read Diagnostics

If you have a `.diag.jsonl` file, load it:

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
- **Decision path changes** — a brain switching from `VCF` to `Alpha-Beta`
  mid-game may signal it lost the tactical thread.

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
| `top_candidates` | Top candidate ≫ chosen move's score |

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
| `decision_path` | `"Alpha-Beta"` when a threat search might have found something |
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
| `search_stats.tt_hits` | 0 or very low — TT not helping |
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
| Ignored opponent build-up | Defensive weight (0.8×) too low for this position |

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
| PNS gives up too early | Increase `pns_nodes` (e.g., 10k → 50k) or `pns_time_s` |
| Alpha-beta too weak positionally | Increase `depth` (e.g., 4 → 5) |
| Opening phase wastes time | Add an opening book (`opening_book = "..."`) |
| Moves too slow | Set `time_limit_s` or reduce depths |
| Quiescence misses tactics | Increase `qs_vcf_depth` (alpha-beta only) |
| Late-game fumbles after good opening | Use threat brain instead of alpha-beta for endgame |

### Structural fixes (code)

| Issue | Fix |
|-------|-----|
| Defensive weight too low | Adjust the 0.8× enemy-weight in `evaluate.py` |
| Fork detection misses complex forks | Lower `_FORK_THRESHOLD` or increase `_FORK_BONUS` |
| Candidate radius misses distant threats | Increase radius from 2 to 3 (expensive) |
| Counter-threats not caught | Verify `_vct_defence_moves()` covers the pattern |

---

## Common Patterns & Fixes

### Pattern 1: "The horizon blunder"

```
Move 20: eval +500 (Alpha-Beta, depth 4)
Move 21: opponent plays live-four → eval -10000
```

**Cause:** Alpha-beta search ended at depth 4, just before the opponent's
winning threat materialized.

**Fix:** Enable quiescence search (`qs_vcf_depth = 12`) or use the threat
brain which naturally searches much deeper for forcing lines.

### Pattern 2: "The abandoned VCF"

```
Move 15: decision_path="VCF", eval +8000 (own VCF found)
Move 17: decision_path="Alpha-Beta", eval +200 (VCF gone)
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
Move 1: think_time_ms=12000, decision_path="Alpha-Beta"
Move 2: think_time_ms=15000, decision_path="Alpha-Beta"
```

**Cause:** No opening book configured.  Alpha-beta exhaustively searches an
almost-empty board where most moves are equivalent.

**Fix:** Use an opening book (`opening_book = "..."`) or increase
`early_game_max_candidates` to reduce branching.

### Pattern 4: "The PNS timeout"

```
Move 25: think_time_ms=5100, decision_path="Alpha-Beta", notes=["PNS: budget exhausted"]
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

The `scripts/` directory contains utilities for post-game analysis:

| Script | Purpose | Usage |
|--------|---------|-------|
| `analyze_threats.py` | Scan games for undetected threats | `uv run python scripts/analyze_threats.py recordings/*.json` |
| `analyze_vct_bailout.py` | Find positions where VCT gave up prematurely | `uv run python scripts/analyze_vct_bailout.py recordings/*.json` |
| `trace_vct_game2.py` | Trace VCT decisions in a specific game | `uv run python scripts/trace_vct_game2.py` |
| `test_vct_depths.py` | Benchmark VCT at multiple depths | `uv run python scripts/test_vct_depths.py` |

---

## Worked Example

### Scenario

Game: `gomoku_20260405_143012.json` — Threat (Black) vs AlphaBeta (White).
White wins at move 42 despite Black having the first-move advantage.

### Step 1: Replay

```bash
uv run gomoku-cli playback recordings/gomoku_20260405_143012.json
```

Navigate to move 25 — Black seems to have the initiative. By move 30, White
has built a row on row 7 uncontested. By move 38, White has a live-four.

### Step 2: Diagnostics overview

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
# 25 VCF-defence            opp_fours=0  Δ=-200
# 26 Alpha-Beta             opp_fours=0  Δ=+100
# 27 VCF-defence            opp_fours=1  Δ=-800     ← !!
# 28 Alpha-Beta             opp_fours=0  Δ=+50
# 29 Alpha-Beta             opp_fours=0  Δ=-100
# 30 Alpha-Beta             opp_fours=0  Δ=-500     ← !!
```

### Step 3: Diagnosis

**Move 27** — Black used VCF-defence, meaning White had a VCF. But Black's
response let White's row-7 grow uncontested (Δ=-800). The VCF-defence picked
a breaker that neutralised the VCF but didn't address the positional build-up.

**Move 30** — Black's alpha-beta played a general move while White had been
building on row 7 for three turns.

### Step 4: Root cause

The VCF-aware defence (Phase 3) correctly broke White's VCF at move 27, but
among multiple VCF-breakers it chose one far from row 7.  The positional
evaluation didn't rate blocking row 7 highly enough because only 3 stones
were in that row (dead-three = 100 points).

### Step 5: Fix

Increase `_OPEN_THREE_BONUS` from 1,500 to 2,000 in `evaluate.py`, or
increase alpha-beta fallback `depth` from 4 to 5 so the brain sees the
row-7 threat materializing earlier.  Alternatively, configure a custom alias:

```toml
[brains.threat_deep]
type = "threat"
depth = 5          # deeper alpha-beta fallback
vcf_depth = 24     # deeper VCF
pns_nodes = 20000  # larger PNS budget
```
