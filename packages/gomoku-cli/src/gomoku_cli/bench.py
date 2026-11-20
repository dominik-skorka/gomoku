"""Speed benchmark: is the candidate slower than the baseline?

``gomoku bench`` times engine A (the baseline) and engine B (the
candidate) on a fixed set of positions and reports how much slower or
faster B thinks.  Both run in worker processes (see
:mod:`gomoku_cli.match`), so A can be another git revision.

Shared CI runners are noisy, so the two engines are timed side by side
rather than against a stored number: every repeat asks A then B for each
position (B then A on odd repeats), and each engine's time for a position
is the median of its repeats.  The verdict compares total median time.

It also counts positions where A and B choose **different moves** — a
change of behaviour, not only of speed — and can profile B (cProfile, top
functions by self time) to show where the time goes.
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path

from gomoku.models import Move

from gomoku_cli.match import Engine, EngineSpec, find_repo, resolve_source

Cell = tuple[int, int]


@dataclass
class Position:
    phase: str
    moves: list[Cell]


@dataclass
class Timing:
    phase: str
    ply: int
    a_ms: float  # median over repeats
    b_ms: float
    a_move: Cell
    b_move: Cell

    @property
    def ratio(self) -> float:
        return self.b_ms / self.a_ms if self.a_ms > 0 else 1.0


@dataclass
class BenchResult:
    timings: list[Timing]
    repeat: int
    profile: str = ""
    by_phase: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def a_total(self) -> float:
        return sum(t.a_ms for t in self.timings)

    @property
    def b_total(self) -> float:
        return sum(t.b_ms for t in self.timings)

    @property
    def ratio(self) -> float:
        return self.b_total / self.a_total if self.a_total > 0 else 1.0

    @property
    def moves_differ(self) -> int:
        return sum(t.a_move != t.b_move for t in self.timings)


def default_positions() -> list[Position]:
    """The bundled 48-position set (16 each from opening, middle and endgame)."""
    data = json.loads(files("gomoku_cli").joinpath("bench_positions.json").read_text())
    return [Position(p["phase"], [tuple(m) for m in p["moves"]]) for p in data["positions"]]


def load_positions(path: Path) -> list[Position]:
    """Positions from a JSON file in the bundled format."""
    data = json.loads(path.read_text())
    return [Position(p.get("phase", "-"), [tuple(m) for m in p["moves"]]) for p in data["positions"]]


def run_bench(
    spec_a: EngineSpec,
    spec_b: EngineSpec,
    positions: list[Position],
    *,
    repeat: int = 3,
    size: int = 15,
    move_timeout_s: float = 600.0,
    profile_top: int = 0,
) -> BenchResult:
    """Time A and B on *positions*; optionally profile B (*profile_top* > 0)."""
    repo = find_repo()
    engine_a = Engine(spec_a, resolve_source(spec_a, repo), size, move_timeout_s)
    engine_b = Engine(spec_b, resolve_source(spec_b, repo), size, move_timeout_s)
    try:
        a_ms: list[list[float]] = [[] for _ in positions]
        b_ms: list[list[float]] = [[] for _ in positions]
        a_moves: list[Cell] = [(-1, -1)] * len(positions)
        b_moves: list[Cell] = [(-1, -1)] * len(positions)
        for r in range(max(1, repeat)):
            for i, pos in enumerate(positions):
                order = ((engine_a, a_ms, a_moves), (engine_b, b_ms, b_moves))
                for engine, times, moves in order if r % 2 == 0 else reversed(order):
                    (row, col), ms = engine.ask(pos.moves)
                    times[i].append(ms)
                    moves[i] = (row, col)
        timings = [
            Timing(pos.phase, len(pos.moves), statistics.median(a), statistics.median(b), am, bm)
            for pos, a, b, am, bm in zip(positions, a_ms, b_ms, a_moves, b_moves, strict=True)
        ]
        text = ""
        if profile_top > 0:
            reply = engine_b.query(
                {"op": "profile", "positions": [p.moves for p in positions], "top": profile_top},
                timeout_s=move_timeout_s * len(positions),
            )
            text = reply["text"]
    finally:
        engine_a.close()
        engine_b.close()
    result = BenchResult(timings=timings, repeat=repeat, profile=text)
    for phase in dict.fromkeys(t.phase for t in timings):
        sel = [t for t in timings if t.phase == phase]
        result.by_phase[phase] = (sum(t.a_ms for t in sel), sum(t.b_ms for t in sel))
    return result


def verdict(result: BenchResult, *, fail_slower: float, warn_slower: float) -> tuple[str, str]:
    """``(PASS|WARN|FAIL, reason)`` from B's total time relative to A's."""
    pct = (result.ratio - 1.0) * 100
    change = f"B thinks {abs(pct):.0f}% {'slower' if pct >= 0 else 'faster'} than A"
    if result.ratio > fail_slower:
        return "FAIL", f"{change} (limit {(fail_slower - 1) * 100:.0f}%)"
    if result.ratio > warn_slower:
        return "WARN", f"{change} (warning above {(warn_slower - 1) * 100:.0f}%)"
    return "PASS", change


def format_report(
    spec_a: EngineSpec, spec_b: EngineSpec, result: BenchResult, *, fail_slower: float, warn_slower: float
) -> str:
    n = len(result.timings)
    lines = [
        f"{n} positions, median of {result.repeat} run(s) each",
        f"A ({spec_a.label}): {result.a_total / 1000:.2f} s total, {result.a_total / n:.0f} ms/move",
        f"B ({spec_b.label}): {result.b_total / 1000:.2f} s total, {result.b_total / n:.0f} ms/move",
        f"B / A: {result.ratio:.2f}x",
        "",
        "by phase      A ms/move   B ms/move   B / A",
    ]
    for phase, (a, b) in result.by_phase.items():
        k = sum(t.phase == phase for t in result.timings)
        lines.append(f"  {phase:<10} {a / k:9.0f}   {b / k:9.0f}   {b / a if a else 1:5.2f}x")
    # Positions B is really slower on: at least 5% and 5 ms.
    slower = [t for t in result.timings if t.b_ms - t.a_ms >= 5 and t.ratio >= 1.05]
    slowest = sorted(slower, key=lambda t: t.b_ms - t.a_ms, reverse=True)[:5]
    lines += ["", "largest slowdowns (B - A):" if slowest else "largest slowdowns (B - A): none"]
    for t in slowest:
        lines.append(f"  {t.phase:<10} ply {t.ply:3d}: A {t.a_ms:7.0f} ms   B {t.b_ms:7.0f} ms   ({t.ratio:.2f}x)")
    if result.moves_differ:
        diff = [t for t in result.timings if t.a_move != t.b_move]
        names = ", ".join(f"ply {t.ply} {_name(t.a_move)}->{_name(t.b_move)}" for t in diff[:8]) + (
            " ..." if len(diff) > 8 else ""
        )
        lines += [
            "",
            f"different moves on {len(diff)} of {n} positions (a change of behaviour): {names}",
            "  (on a busy machine a time limit, e.g. pns_time_s, can also change a move or two)",
        ]
    else:
        lines += ["", f"same moves on all {n} positions"]
    status, reason = verdict(result, fail_slower=fail_slower, warn_slower=warn_slower)
    lines += ["", f"Verdict: {status} — {reason}"]
    return "\n".join(lines)


def _name(cell: Cell) -> str:
    return Move(row=cell[0], col=cell[1]).position


def save_bench(path: Path, spec_a: EngineSpec, spec_b: EngineSpec, result: BenchResult, status: str) -> None:
    payload = {
        "a": spec_a.label,
        "b": spec_b.label,
        "verdict": status,
        "ratio": result.ratio,
        "a_total_ms": result.a_total,
        "b_total_ms": result.b_total,
        "moves_differ": result.moves_differ,
        "timings": [asdict(t) for t in result.timings],
        "profile": result.profile,
    }
    path.write_text(json.dumps(payload, indent=1))
