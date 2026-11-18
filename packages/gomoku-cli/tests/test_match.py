"""Tests for the engine-versus-engine match harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from gomoku_cli.main import cli
from gomoku_cli.match import (
    EngineSpec,
    GameRecord,
    describe_rev,
    format_summary,
    generate_openings,
    run_match,
    summarize,
)


class TestEngineSpec:
    def test_bare_name_is_a_brain(self):
        assert EngineSpec.parse("threat") == EngineSpec(brain="threat")

    def test_full_spec(self):
        spec = EngineSpec.parse("brain=threat, rev=master, config=alt.toml")
        assert spec == EngineSpec(brain="threat", rev="master", config=Path("alt.toml"))
        assert spec.label == "threat@master[alt.toml]"

    @pytest.mark.parametrize("text", ["rev=master", "brain=", "brain=x,depth=3", ""])
    def test_invalid_specs_are_rejected(self, text: str):
        with pytest.raises(ValueError, match="Invalid engine spec"):
            EngineSpec.parse(text)


class TestOpenings:
    def test_deterministic_and_distinct(self):
        first = generate_openings(20, seed=7)
        assert first == generate_openings(20, seed=7)
        assert first != generate_openings(20, seed=8)
        assert len({tuple(o) for o in first}) == 20

    def test_stones_are_distinct_cells_near_the_centre(self):
        for opening in generate_openings(30, seed=3, size=15, stones=6, spread=2):
            assert len(opening) == len(set(opening)) == 6
            assert all(5 <= r <= 9 and 5 <= c <= 9 for r, c in opening)

    def test_quiet_openings_have_no_forced_win(self):
        from gomoku.board import Board
        from gomoku.models import Move, Stone
        from gomoku_brain.pns import pns_threat_search

        for opening in generate_openings(8, seed=5, quiet=True):
            board = Board(size=15)
            stone = Stone.BLACK
            for r, c in opening:
                board = board.place_stone(Move(row=r, col=c), stone)
                stone = stone.opponent()
            assert pns_threat_search(board, Stone.BLACK, max_nodes=3_000) is None
            assert pns_threat_search(board, Stone.WHITE, max_nodes=3_000) is None

    def test_too_many_stones_for_the_area(self):
        with pytest.raises(ValueError, match="Cannot place"):
            generate_openings(1, seed=1, stones=10, spread=1)


def _game(winner: str | None, a_is_black: bool = True, opening: int = 0) -> GameRecord:
    return GameRecord(
        opening=opening, a_is_black=a_is_black, winner=winner, reason="test", moves=[], a_ms=[10.0], b_ms=[30.0]
    )


def _pairs(*outcomes: tuple[str | None, str | None]) -> list[GameRecord]:
    """One opening per outcome tuple: (result with A as black, result with A as white)."""
    records = []
    for i, (as_black, as_white) in enumerate(outcomes):
        records += [_game(as_black, True, i), _game(as_white, False, i)]
    return records


class TestSummary:
    def test_counts_score_and_colours(self):
        records = [_game("a"), _game("a", a_is_black=False), _game("b"), _game(None)]
        s = summarize(records)
        assert (s.games, s.wins_a, s.wins_b, s.draws) == (4, 2, 1, 1)
        assert s.score == pytest.approx(0.625)
        assert (s.a_wins_as_black, s.a_wins_as_white, s.b_wins_as_white) == (1, 1, 1)
        assert s.score_low < s.score < s.score_high
        assert s.elo is not None and s.elo > 0
        assert s.los is not None and s.los > 0.5
        assert (s.a_ms_per_move, s.b_ms_per_move) == (10.0, 30.0)

    def test_even_match_is_zero_elo(self):
        s = summarize([_game("a"), _game("b")] * 10)
        assert s.score == 0.5
        assert s.elo == pytest.approx(0.0)
        assert s.los == pytest.approx(0.5)

    def test_interval_is_per_opening_pair(self):
        """Split pairs are pure colour noise; only swept pairs move the needle."""
        s = summarize(_pairs(*[("a", "b")] * 40, *[("a", "a")] * 10))
        assert s.pairs == 50
        assert s.score == pytest.approx(0.6)
        # As 100 independent games the half-width would be about 0.096;
        # with 40 of 50 pairs tied the paired interval is much tighter.
        assert s.score_high - s.score_low < 0.15
        assert s.score_low > 0.5
        assert s.los is not None and s.los > 0.99

    def test_uniform_small_sample_is_not_overconfident(self):
        """Two split pairs alone have zero variance; the interval must stay wide."""
        s = summarize(_pairs(("a", "b"), ("b", "a")))
        assert s.score == 0.5
        assert s.score_low < 0.2 and s.score_high > 0.8

    def test_whitewash_has_no_finite_elo(self):
        s = summarize([_game("a")] * 6)
        assert s.score == 1.0
        assert s.elo is None
        text = format_summary(EngineSpec("x"), EngineSpec("y"), s)
        assert "n/a" in text and "100.0%" in text


class TestMatchEndToEnd:
    """Real engine processes; the random brain keeps it fast."""

    def test_random_vs_random(self):
        seen: list[GameRecord] = []
        records = run_match(EngineSpec("random"), EngineSpec("random"), games=4, seed=2, size=9, on_game=seen.append)
        assert len(records) == len(seen) == 4
        assert [r.a_is_black for r in records] == [True, False, True, False]
        assert [r.opening for r in records] == [0, 0, 1, 1]
        for r in records:
            assert r.winner in {"a", "b", None}
            assert len(r.moves) == len(set(r.moves)) >= 7
            assert len(r.a_ms) + len(r.b_ms) == len(r.moves) - 7

    def test_cli_writes_report_and_json(self, tmp_path: Path):
        out = tmp_path / "games.json"
        result = CliRunner().invoke(
            cli,
            ["match", "--a", "random", "--b", "brain=random", "--games", "2", "--board-size", "9", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert "A score:" in result.output
        payload = json.loads(out.read_text())
        assert len(payload["games"]) == 2
        assert payload["summary"]["games"] == 2

    def test_cli_rejects_bad_spec(self):
        result = CliRunner().invoke(cli, ["match", "--a", "rev=master", "--b", "random"])
        assert result.exit_code != 0
        assert "Invalid engine spec" in result.output


class TestDescribeRev:
    @staticmethod
    def _git(cwd: Path, *args: str) -> str:
        identity = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "init.defaultBranch=main"]
        return subprocess.run(
            ["git", *identity, *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()

    def test_stale_branch_is_flagged(self, tmp_path: Path):
        origin, clone = tmp_path / "origin", tmp_path / "clone"
        origin.mkdir()
        self._git(origin, "init", "-q")
        self._git(origin, "commit", "-q", "--allow-empty", "-m", "first")
        self._git(tmp_path, "clone", "-q", str(origin), str(clone))
        assert describe_rev(EngineSpec("threat", rev="main"), clone).endswith(" first")

        self._git(origin, "commit", "-q", "--allow-empty", "-m", "second")
        self._git(clone, "fetch", "-q")
        line = describe_rev(EngineSpec("threat", rev="main"), clone)
        assert line.endswith("[WARNING: main is 1 commit(s) behind its upstream; fetch and update it]")
        # A remote branch or a commit has no upstream to fall behind.
        assert "WARNING" not in describe_rev(EngineSpec("threat", rev="origin/main"), clone)
        assert describe_rev(EngineSpec("threat"), clone) == "working tree"
