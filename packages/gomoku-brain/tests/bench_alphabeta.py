"""Baseline benchmarks for :class:`AlphaBetaBrain`.

The goal of this module is to freeze reproducible numbers for the current
(pre-refactor) implementation so that subsequent phases of the search
architecture overhaul can be measured against it.

Usage
-----
As a standalone script (recommended for readable output):

    uv run python packages/gomoku-brain/tests/bench_alphabeta.py

As pytest (collected but only executed with ``-m bench``):

    uv run pytest packages/gomoku-brain/tests/bench_alphabeta.py -m bench -s

Metrics captured per scenario:

* wall-clock time (seconds)
* interior node count (``AlphaBetaBrain.node_count``)
* quiescence node count (``AlphaBetaBrain.qs_node_count``)

Fixed scenarios:

* ``early``      — 6 stones, past the opening book (small radius + symmetry)
* ``midgame``    — 15 stones around a contested centre fight
* ``endgame``    — 35 stones with several simultaneous tactics

The scenarios are expressed as move lists so they are reproducible across
machines and commits.  Add more if you want finer granularity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.alphabeta import AlphaBetaBrain

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    moves: tuple[tuple[int, int], ...]
    side_to_move: Stone
    board_size: int = 15


# 6 stones past the opening book (which covers <= 4 stones).  Exercises the
# early-game code path (small radius, symmetry dedup) through the full search.
_EARLY = Scenario(
    name="early",
    moves=(
        (7, 7),  # B
        (7, 8),  # W
        (8, 7),  # B
        (6, 8),  # W
        (8, 8),  # B
        (6, 7),  # W
    ),
    side_to_move=Stone.BLACK,
)

# 15 stones, Black to move.  A tangled centre fight with live threes for
# both sides — representative of where the search burns cycles in real play.
_MIDGAME = Scenario(
    name="midgame",
    moves=(
        (7, 7),  # B
        (7, 8),  # W
        (8, 7),  # B
        (6, 8),  # W
        (9, 6),  # B
        (8, 8),  # W
        (6, 6),  # B
        (5, 5),  # W
        (9, 9),  # B
        (6, 9),  # W
        (5, 7),  # B
        (7, 9),  # W
        (10, 5),  # B
        (4, 8),  # W
        (8, 5),  # B
    ),
    side_to_move=Stone.WHITE,
)

# 35 stones, several converging patterns.  Most alpha-beta cost happens at
# densities like this.
_ENDGAME = Scenario(
    name="endgame",
    moves=(
        (7, 7), (7, 8), (8, 7), (6, 8), (9, 6), (8, 8), (6, 6), (5, 5),
        (9, 9), (6, 9), (5, 7), (7, 9), (10, 5), (4, 8), (8, 5), (9, 4),
        (10, 6), (11, 5), (4, 7), (3, 7), (4, 9), (3, 9), (5, 8), (6, 7),
        (2, 10), (1, 11), (11, 4), (12, 3), (8, 10), (8, 11), (7, 10), (7, 11),
        (10, 10), (11, 11), (12, 12),
    ),
    side_to_move=Stone.WHITE,
)


SCENARIOS: tuple[Scenario, ...] = (_EARLY, _MIDGAME, _ENDGAME)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    scenario: str
    depth: int
    time_s: float
    node_count: int
    qs_node_count: int
    chosen_move: Move
    extra: dict[str, object] = field(default_factory=dict)

    def row(self) -> str:
        return (
            f"{self.scenario:<10} {self.depth:>5} {self.time_s:>8.3f}s "
            f"{self.node_count:>10,} {self.qs_node_count:>10,} "
            f"{self.chosen_move!s:>7}"
        )


def build_board(scenario: Scenario) -> Board:
    board = Board(size=scenario.board_size)
    stone = Stone.BLACK
    for r, c in scenario.moves:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return board


def run_scenario(scenario: Scenario, depth: int, **brain_kwargs: object) -> BenchResult:
    """Execute one ``get_move`` call under the scenario and return metrics.

    ``brain_kwargs`` are passed through to :class:`AlphaBetaBrain` so you can
    toggle features (e.g. ``lmr=True``) when comparing variants.
    """
    board = build_board(scenario)
    brain = AlphaBetaBrain(depth=depth, **brain_kwargs)

    t0 = time.monotonic()
    move = brain.get_move(board, scenario.side_to_move)
    elapsed = time.monotonic() - t0

    return BenchResult(
        scenario=scenario.name,
        depth=depth,
        time_s=elapsed,
        node_count=brain.node_count,
        qs_node_count=brain.qs_node_count,
        chosen_move=move,
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


_DEPTHS: tuple[int, ...] = (4,)
"""Depths to benchmark by default.  Add 6 locally when you can spare ~1 min/run."""


def run_baseline(depths: tuple[int, ...] = _DEPTHS) -> list[BenchResult]:
    """Run every scenario x depth combination and return the full result list."""
    # Warm up numba JIT once before measurement — first call compiles.
    warm = run_scenario(_EARLY, depth=1)
    assert warm.chosen_move is not None

    header = f"{'scenario':<10} {'depth':>5} {'time_s':>9} {'nodes':>10} {'qs_nodes':>10} {'move':>7}"
    print()
    print(header)
    print("-" * len(header))

    results: list[BenchResult] = []
    for depth in depths:
        for scenario in SCENARIOS:
            res = run_scenario(scenario, depth=depth)
            print(res.row())
            results.append(res)
    return results


if __name__ == "__main__":
    run_baseline(depths=(4,))


# ---------------------------------------------------------------------------
# pytest integration — opt-in via ``-m bench``
# ---------------------------------------------------------------------------


@pytest.mark.bench
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_bench_depth_4(scenario: Scenario) -> None:
    """Record baseline numbers at depth 4.  Only runs with ``-m bench``."""
    res = run_scenario(scenario, depth=4)
    # Sanity — never crash and always return *some* move.
    assert res.chosen_move is not None
    # Emit as a pytest-visible line.
    print(f"\n{res.row()}")
