"""Tests for the :mod:`gomoku_cli.analyze` post-game analyser."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from gomoku.diagnostics import (
    CandidateInfo,
    DiagnosticsHeader,
    MoveDiagnostics,
    SearchStats,
    ThreatInfo,
)
from gomoku.models import GameConfig, Move, Stone
from gomoku_cli.analyze import (
    BLUNDER_EVAL_DROP,
    TIME_SPIKE_FACTOR,
    TIME_SPIKE_MIN_MS,
    analyze_diagnostics,
    load_diagnostics,
)
from gomoku_cli.main import cli

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl(
    path: Path,
    header: DiagnosticsHeader,
    moves: list[MoveDiagnostics],
    summary: dict | None = None,
) -> Path:
    lines = [header.model_dump_json()]
    lines.extend(m.model_dump_json() for m in moves)
    if summary is not None:
        lines.append(json.dumps({"type": "summary", **summary}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _header() -> DiagnosticsHeader:
    return DiagnosticsHeader(
        started_at=datetime.now(tz=UTC),
        game_config=GameConfig(board_size=15, win_length=5),
        players=[
            {"stone": int(Stone.BLACK), "provider": "alphabeta", "settings": {}},
            {"stone": int(Stone.WHITE), "provider": "random", "settings": {}},
        ],
    )


def _move(
    n: int,
    stone: Stone,
    *,
    think_ms: int = 50,
    eval_after: float | None = 0.0,
    eval_before: float | None = 0.0,
    chosen: Move | None = None,
    top_candidates: list[CandidateInfo] | None = None,
    alt_best: CandidateInfo | None = None,
    our: ThreatInfo | None = None,
    opp: ThreatInfo | None = None,
    decision_path: str = "alphabeta",
    stats: SearchStats | None = None,
) -> MoveDiagnostics:
    return MoveDiagnostics(
        move_number=n,
        stone=stone,
        chosen_move=chosen or Move(row=7, col=7),
        decision_path=decision_path,
        think_time_ms=think_ms,
        eval_before=eval_before,
        eval_after=eval_after,
        search_stats=stats or SearchStats(depth_reached=4, nodes_searched=1000),
        top_candidates=top_candidates or [],
        alternative_best=alt_best,
        our_threats=our or ThreatInfo(),
        opp_threats=opp or ThreatInfo(),
    )


# ---------------------------------------------------------------------------
# load_diagnostics
# ---------------------------------------------------------------------------


class TestLoad:
    def test_parses_header_moves_summary(self, tmp_path: Path) -> None:
        p = tmp_path / "g.diag.jsonl"
        moves = [_move(1, Stone.BLACK), _move(2, Stone.WHITE)]
        _write_jsonl(p, _header(), moves, summary={"total_moves": 2})
        header, parsed, summary = load_diagnostics(p)
        assert header["type"] == "header"
        assert len(parsed) == 2
        assert parsed[0].move_number == 1 and parsed[0].stone is Stone.BLACK
        assert summary == {"type": "summary", "total_moves": 2}

    def test_rejects_missing_header(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.diag.jsonl"
        p.write_text(_move(1, Stone.BLACK).model_dump_json() + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a diagnostics header"):
            load_diagnostics(p)

    def test_rejects_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.diag.jsonl"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty diagnostics file"):
            load_diagnostics(p)

    def test_tolerates_no_summary(self, tmp_path: Path) -> None:
        p = tmp_path / "nosum.diag.jsonl"
        _write_jsonl(p, _header(), [_move(1, Stone.BLACK)], summary=None)
        _, parsed, summary = load_diagnostics(p)
        assert summary is None and len(parsed) == 1

    def test_annotates_bad_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.diag.jsonl"
        p.write_text(
            _header().model_dump_json() + "\n" + "{\"move_number\": \"nope\"}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="malformed move record"):
            load_diagnostics(p)


# ---------------------------------------------------------------------------
# Flagging heuristics
# ---------------------------------------------------------------------------


class TestFlagging:
    def test_eval_drop_flagged_once_threshold_crossed(self, tmp_path: Path) -> None:
        # Black plays two moves; second move's eval drops by more than the
        # blunder threshold relative to its own previous move.
        drop = BLUNDER_EVAL_DROP + 100
        moves = [
            _move(1, Stone.BLACK, eval_after=1000.0),
            _move(2, Stone.WHITE, eval_after=0.0),
            _move(3, Stone.BLACK, eval_after=1000.0 - drop),
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        assert [f.move_number for f in report.flagged] == [3]
        assert "eval_drop" in report.flagged[0].reasons
        assert report.player_stats[Stone.BLACK].blunders == 1

    def test_small_eval_drop_not_flagged(self, tmp_path: Path) -> None:
        moves = [
            _move(1, Stone.BLACK, eval_after=1000.0),
            _move(2, Stone.WHITE, eval_after=0.0),
            _move(3, Stone.BLACK, eval_after=1000.0 - (BLUNDER_EVAL_DROP - 50)),
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        assert report.flagged == []

    def test_bypassed_top_candidate_flagged(self, tmp_path: Path) -> None:
        chosen = Move(row=1, col=1)
        top = Move(row=2, col=2)
        cands = [
            CandidateInfo(move=top, score=2000, score_type="eval"),
            CandidateInfo(move=chosen, score=1000, score_type="eval"),
        ]
        alt = cands[0]
        moves = [
            _move(
                1,
                Stone.BLACK,
                chosen=chosen,
                top_candidates=cands,
                alt_best=alt,
            )
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        assert len(report.flagged) == 1
        assert "bypassed_top_candidate" in report.flagged[0].reasons

    def test_winrate_alternative_not_flagged(self, tmp_path: Path) -> None:
        # MCTS uses score_type="winrate" — the heuristic should NOT compare
        # those scores with the eval-tuned ALTERNATIVE_GAP threshold.
        chosen = Move(row=1, col=1)
        top = Move(row=2, col=2)
        cands = [
            CandidateInfo(move=top, score=0.9, score_type="winrate"),
            CandidateInfo(move=chosen, score=0.1, score_type="winrate"),
        ]
        moves = [
            _move(
                1,
                Stone.BLACK,
                chosen=chosen,
                top_candidates=cands,
                alt_best=cands[0],
            )
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        assert all("bypassed_top_candidate" not in f.reasons for f in report.flagged)

    def test_time_spike_flagged(self, tmp_path: Path) -> None:
        base = max(TIME_SPIKE_MIN_MS, 100)
        # Three baseline black moves at `base`, then a spike.  Uses
        # rolling-average so spike must exceed ``TIME_SPIKE_FACTOR × avg``.
        spike = int(base * TIME_SPIKE_FACTOR + 200)
        moves = [
            _move(1, Stone.BLACK, think_ms=base),
            _move(2, Stone.WHITE, think_ms=10),
            _move(3, Stone.BLACK, think_ms=base),
            _move(4, Stone.WHITE, think_ms=10),
            _move(5, Stone.BLACK, think_ms=base),
            _move(6, Stone.WHITE, think_ms=10),
            _move(7, Stone.BLACK, think_ms=spike),
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        flagged_ns = {f.move_number for f in report.flagged}
        assert 7 in flagged_ns

    def test_opp_double_four_flagged(self, tmp_path: Path) -> None:
        moves = [
            _move(
                1,
                Stone.BLACK,
                decision_path="alphabeta",
                opp=ThreatInfo(four_threats=2),
            )
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        assert "opp_double_four" in report.flagged[0].reasons

    def test_opp_double_four_not_flagged_when_blocking(self, tmp_path: Path) -> None:
        moves = [
            _move(
                1,
                Stone.BLACK,
                decision_path="block_fork",
                opp=ThreatInfo(four_threats=2),
            )
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        assert report.flagged == []

    def test_opp_forced_win_flagged(self, tmp_path: Path) -> None:
        moves = [
            _move(
                1,
                Stone.BLACK,
                opp=ThreatInfo(vcf_result="win"),
            )
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        assert "opp_forced_win" in report.flagged[0].reasons


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------


class TestAggregates:
    def test_player_stats_sums(self, tmp_path: Path) -> None:
        moves = [
            _move(
                1,
                Stone.BLACK,
                think_ms=100,
                stats=SearchStats(depth_reached=6, nodes_searched=500),
            ),
            _move(
                2,
                Stone.WHITE,
                think_ms=50,
                stats=SearchStats(depth_reached=4, nodes_searched=200),
            ),
            _move(
                3,
                Stone.BLACK,
                think_ms=200,
                stats=SearchStats(depth_reached=8, nodes_searched=1500),
            ),
        ]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves)
        report = analyze_diagnostics(p)
        black = report.player_stats[Stone.BLACK]
        white = report.player_stats[Stone.WHITE]

        assert black.moves == 2
        assert black.total_think_ms == 300
        assert black.total_nodes == 2000
        assert black.max_depth_reached == 8
        assert black.avg_depth == 7.0

        assert white.moves == 1
        assert white.total_think_ms == 50
        assert white.max_depth_reached == 4

    def test_to_json_shape(self, tmp_path: Path) -> None:
        moves = [_move(1, Stone.BLACK)]
        p = _write_jsonl(tmp_path / "x.diag.jsonl", _header(), moves, summary={"total_moves": 1})
        report = analyze_diagnostics(p)
        data = report.to_json()
        assert set(data) >= {"path", "header", "summary", "total_moves", "player_stats", "flagged"}
        assert data["total_moves"] == 1
        assert str(Stone.BLACK) in data["player_stats"]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    def test_analyze_runs_cleanly(self, tmp_path: Path) -> None:
        moves = [
            _move(1, Stone.BLACK, eval_after=1000.0),
            _move(2, Stone.WHITE, eval_after=0.0),
            _move(3, Stone.BLACK, eval_after=0.0),
        ]
        p = _write_jsonl(tmp_path / "g.diag.jsonl", _header(), moves)

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", str(p)])
        assert result.exit_code == 0, result.output
        assert "Per-player statistics" in result.output
        assert "Flagged moves" in result.output or "No suspect moves" in result.output

    def test_analyze_json_mode(self, tmp_path: Path) -> None:
        moves = [_move(1, Stone.BLACK)]
        p = _write_jsonl(tmp_path / "g.diag.jsonl", _header(), moves)

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", str(p), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_moves"] == 1
        assert "player_stats" in data
