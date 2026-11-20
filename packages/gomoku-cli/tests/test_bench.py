"""Speed benchmark: position set, verdict, report and the CLI."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from click.testing import CliRunner
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner
from gomoku_cli.bench import BenchResult, Timing, default_positions, format_report, run_bench, verdict
from gomoku_cli.main import cli
from gomoku_cli.match import EngineSpec


def test_bundled_positions_are_legal_unfinished_and_balanced():
    positions = default_positions()
    assert Counter(p.phase for p in positions) == {"opening": 16, "middle": 16, "endgame": 16}
    for p in positions:
        board, stone = Board(size=15), Stone.BLACK
        for r, c in p.moves:
            move = Move(row=r, col=c)
            assert board.is_valid_move(move)
            board = board.place_stone(move, stone)
            assert check_winner(board, move) is None
            stone = stone.opponent()


def _result(pairs: list[tuple[float, float]], moves_differ: int = 0) -> BenchResult:
    timings = [
        Timing("middle", 20 + i, a, b, (7, 7), (7, 8) if i < moves_differ else (7, 7)) for i, (a, b) in enumerate(pairs)
    ]
    result = BenchResult(timings=timings, repeat=3)
    result.by_phase["middle"] = (sum(a for a, _ in pairs), sum(b for _, b in pairs))
    return result


@pytest.mark.parametrize(
    ("pairs", "status"),
    [
        ([(100, 100), (200, 210)], "PASS"),  # 3% slower
        ([(100, 115), (200, 230)], "WARN"),  # 15% slower
        ([(100, 140), (200, 260)], "FAIL"),  # 33% slower
        ([(100, 50), (200, 100)], "PASS"),  # faster
    ],
)
def test_verdict_uses_total_time(pairs, status):
    assert verdict(_result(pairs), fail_slower=1.25, warn_slower=1.10)[0] == status


def test_report_names_slowdowns_and_changed_moves():
    result = _result([(100, 100), (100, 300), (100, 90)], moves_differ=1)
    text = format_report(EngineSpec("x"), EngineSpec("y"), result, fail_slower=1.25, warn_slower=1.10)
    assert "B / A: 1.63x" in text
    assert "ply  21: A     100 ms   B     300 ms   (3.00x)" in text
    assert "different moves on 1 of 3 positions" in text and "H8->I8" in text
    assert "Verdict: FAIL" in text


def test_run_bench_with_real_workers():
    result = run_bench(EngineSpec("random"), EngineSpec("random"), default_positions()[:4], repeat=2, profile_top=5)
    assert len(result.timings) == 4
    assert all(t.a_ms >= 0 and t.b_ms >= 0 for t in result.timings)
    assert "get_candidate_moves" in result.profile


def test_cli_writes_json_and_exits_on_fail(tmp_path: Path, monkeypatch):
    out = tmp_path / "bench.json"
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"positions": [{"phase": "x", "moves": [[7, 7], [7, 8]]}]}))
    args = ["bench", "--a", "random", "--b", "random", "--positions", str(positions), "--repeat", "1"]
    result = CliRunner().invoke(cli, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["verdict"] in {"PASS", "WARN"}

    import gomoku_cli.bench as bench_module

    monkeypatch.setattr(bench_module, "run_bench", lambda *a, **k: _result([(100, 200)]))
    failed = CliRunner().invoke(cli, args)
    assert failed.exit_code == 1 and "Verdict: FAIL" in failed.output
