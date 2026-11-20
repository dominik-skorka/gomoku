"""Blunder review: detection rules, escalation and the end-to-end command."""

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from gomoku.models import Stone
from gomoku_cli.main import cli
from gomoku_cli.match import Engine, EngineError, EngineSpec, GameRecord, find_repo, resolve_source
from gomoku_cli.match_worker import merge
from gomoku_cli.review import (
    WIN_PATHS,
    Blunder,
    EngineTally,
    GameReview,
    OracleConfig,
    _EngineSlot,
    _escalate,
    _Oracle,
    escalations,
    five_cells,
    format_report,
    review_game,
    run_review,
    tally,
    verdict,
)

SIZE = 15


class FakeOracle(_Oracle):
    """Answers from a table keyed by the position; anything else is DISPROVED."""

    def __init__(self, table=None, *, screen_default: str = "DISPROVED") -> None:
        super().__init__(engine=None, config=OracleConfig(screen_moves=400))  # type: ignore[arg-type]
        self.table = table or {}
        self.screen_default = screen_default

    def status(self, moves, stone=None, *, screen=False):
        key = (tuple(moves), None if stone is None else int(stone))
        return self.table.get(key, (self.screen_default if screen else "DISPROVED", None))


class FakeSlot:
    def __init__(self, replies: dict[str, tuple[int, int]], brain_type: str = "threat", ms: float = 100.0) -> None:
        self.replies = replies
        self.brain_type = brain_type
        self.ms = ms
        self.seen: list[dict[str, Any]] = []

    def settings(self):
        return self.brain_type, {
            "depth": 5,
            "pns_nodes": 10_000,
            "pns_time_s": 5.0,
            "threat_search": "pns",
            "vct_depth": 12,
            "fallback": {"depth": 5, "max_candidates": 12},
        }

    def retry(self, moves, overrides, timeout_s):
        self.seen.append(overrides)
        name = next((n for n, o in [("repeat", {}), *escalations(*self.settings())] if o == overrides), "?")
        if name not in self.replies:
            return None  # a timeout
        return {"move": list(self.replies[name]), "ms": self.ms, "path": None}


def _record(moves: list[tuple[int, int]], *, reason: str = "move limit", paths=None) -> GameRecord:
    engine_moves = len(moves) - 2
    return GameRecord(
        opening=0,
        a_is_black=True,
        winner=None,
        reason=reason,
        moves=moves,
        a_ms=[100.0] * ((engine_moves + 1) // 2),
        b_ms=[100.0] * (engine_moves // 2),
        paths=paths or [None] * engine_moves,
    )


# Black has four in row 7 (cols 3-6, open at 2 and 7); White's stones are scattered.
FOUR = [(7, 3), (0, 0), (7, 4), (0, 2), (7, 5), (0, 4), (7, 6), (0, 6)]
# The same four with White blocking col 7: only (7, 2) completes it.
HALF_FOUR = [*FOUR[:-1], (7, 7)]


class TestFiveCells:
    def test_open_four_has_two_completions(self):
        assert sorted(five_cells(FOUR, Stone.BLACK, SIZE)) == [(7, 2), (7, 7)]
        assert five_cells(FOUR, Stone.WHITE, SIZE) == []

    def test_gap_and_overline_count(self):
        gapped = [(3, 3), (0, 14), (3, 4), (2, 14), (3, 6), (4, 14), (3, 7)]
        assert five_cells(gapped, Stone.BLACK, SIZE) == [(3, 5)]


class TestDetection:
    def test_missed_win_and_missed_block(self):
        # Black (to move at ply 8) ignores its own five; White then ignores the block.
        moves = [*HALF_FOUR, (10, 10), (10, 11)]
        review = review_game(0, _record(moves), 2, FakeOracle(), {}, SIZE, escalate=False)
        found = [(b.ply, b.kind, b.engine, b.better) for b in review.blunders]
        assert found == [(8, "missed_win", "a", (7, 2)), (9, "missed_block", "b", (7, 2))]

    def test_open_four_cannot_be_blocked(self):
        moves = [*FOUR, (7, 2)]  # White (ply 7) faced two completions: no blunder
        review = review_game(0, _record(moves, reason="five in a row"), 2, FakeOracle(), {}, SIZE, escalate=False)
        assert review.blunders == []
        assert review.statuses[-1] is None

    def test_losing_move_names_a_saving_move(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14)]
        # After Black's 3rd move (ply 2) White still has nothing, but after
        # White's ply-3 move Black is proven to win; a different ply-3 move held.
        table = {
            (tuple(moves[:4]), None): ("PROVED", (7, 9)),
            (tuple(moves[:3]), int(Stone.BLACK)): ("PROVED", (7, 9)),
        }
        for cell in [(r, c) for r in range(SIZE) for c in range(SIZE)]:
            if cell != (7, 6) and cell not in moves:
                table[((*moves[:3], cell), None)] = ("PROVED", (7, 9))
        review = review_game(0, _record(moves), 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert [(b.ply, b.kind, b.better) for b in review.blunders] == [(3, "losing_move", (7, 6))]

    def test_already_lost_position_is_not_a_blunder(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14)]
        cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        table = {((*moves[:3], cell), None): ("PROVED", None) for cell in cells}
        table[(tuple(moves[:4]), None)] = ("PROVED", None)
        review = review_game(0, _record(moves), 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert review.blunders == []

    def test_brain_claim_disproved_is_unsound(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14)]
        review = review_game(0, _record(moves, paths=["pns_win", None]), 2, FakeOracle(), {}, SIZE, escalate=True)
        assert [(b.ply, b.kind, b.verdict) for b in review.blunders] == [(2, "unsound_proof", "bug")]

    def test_oracle_proof_lost_after_its_own_move_is_unsound(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14), (3, 3), (4, 4)]
        table = {(tuple(moves[:2]), None): ("PROVED", (7, 8))}  # ply 2 plays the proof's move
        review = review_game(0, _record(moves), 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert [(b.ply, b.kind) for b in review.blunders] == [(2, "unsound_proof")]

    def test_win_kept_after_a_different_move_is_fine(self):
        moves = [(7, 7), (0, 0), (7, 8), (14, 14), (3, 3), (4, 4)]
        table = {
            (tuple(moves[:2]), None): ("PROVED", (7, 9)),
            (tuple(moves[:4]), None): ("PROVED", (7, 9)),
            (tuple(moves[:6]), None): ("PROVED", (7, 9)),
        }
        # White's position stayed lost whatever it played.
        oracle = FakeOracle(table, screen_default="PROVED")
        review = review_game(0, _record(moves), 2, oracle, {}, SIZE, escalate=False)
        assert review.blunders == []


class TestUnconvertedWin:
    """A claimed forced win must hold to the end of the game.

    Black (engine A) plays plies 2, 4 and 6; White plays 3, 5 and 7.  No
    five is ever on the board: the games end on the move limit unless a
    test says otherwise.
    """

    MOVES = ((7, 7), (0, 0), (7, 8), (14, 14), (3, 3), (4, 4), (5, 5), (6, 6))
    WIN = (7, 9)  # the oracle's winning move, when it proves one

    def _oracle(self, status: str) -> FakeOracle:
        """*status* for Black before each of its moves; White has no win."""
        table = {
            (tuple(self.MOVES[:p]), None): (status, self.WIN if status == "PROVED" else None) for p in (2, 4, 6, 8)
        }
        return FakeOracle(table, screen_default="PROVED")

    def _review(self, paths, oracle, *, reason="move limit", winner=None, escalate=False, slots=None):
        record = _record(list(self.MOVES), reason=reason, paths=paths)
        record.winner = winner
        return review_game(0, record, 2, oracle, slots or {}, SIZE, escalate=escalate)

    def test_claim_dropped_while_the_win_is_still_there_is_tuning(self):
        slot = FakeSlot({"repeat": self.WIN})
        review = self._review(
            ["pns_win", None, None, None, None, None], self._oracle("PROVED"), escalate=True, slots={"a": slot}
        )
        [blunder] = review.blunders
        assert (blunder.kind, blunder.ply, blunder.claim_ply, blunder.better) == ("unconverted_win", 4, 2, self.WIN)
        assert blunder.verdict == "tuning"  # asked again, the brain finds the win
        assert "move 3" in blunder.detail and "move 5 was no win" in blunder.detail

    def test_claim_dropped_and_no_setting_finds_the_win_is_a_bug(self):
        slot = FakeSlot({"repeat": (1, 1)})
        review = self._review(
            ["pns_win", None, None, None, None, None], self._oracle("PROVED"), escalate=True, slots={"a": slot}
        )
        [blunder] = review.blunders
        assert (blunder.kind, blunder.verdict) == ("unconverted_win", "bug")

    def test_oracle_cannot_tell_is_unclassified(self):
        review = self._review(["pns_win", None, None, None, None, None], self._oracle("UNKNOWN"))
        [blunder] = review.blunders
        assert (blunder.kind, blunder.ply, blunder.verdict) == ("unconverted_win", 4, "unclassified")

    def test_claims_kept_to_the_end_of_an_unfinished_game_are_reported(self):
        paths = ["pns_win", None, "vcf_win", None, "pns_win", None]
        review = self._review(paths, self._oracle("UNKNOWN"))
        [blunder] = review.blunders
        assert (blunder.kind, blunder.ply) == ("unconverted_win", 6)  # the last own move
        assert "ended without its five (move limit)" in blunder.detail

    def test_opponent_five_after_a_claim(self):
        paths = ["pns_win", None, "pns_win", None, "pns_win", None]
        review = self._review(paths, self._oracle("UNKNOWN"), reason="five in a row", winner="b")
        [blunder] = review.blunders
        assert blunder.kind == "unconverted_win" and "opponent completed five" in blunder.detail

    def test_a_claim_ending_in_its_own_five_is_fine(self):
        moves = [(7, 7), (0, 0), (7, 3), (0, 2), (7, 4), (0, 4), (7, 5), (0, 6), (7, 6)]  # Black's five at ply 8
        table = {(tuple(moves[:p]), None): ("UNKNOWN", None) for p in range(2, 9)}
        record = _record(moves, reason="five in a row", paths=["pns_win", None, "pns_win", None, "vcf_win", None, None])
        record.winner = "a"
        review = review_game(0, record, 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert [b.kind for b in review.blunders if b.engine == "a"] == []

    def test_forced_block_inside_the_chain_is_fine(self):
        # After Black's claim White makes a four (row 0, open only at (0, 4));
        # Black's block there is part of the proof, then Black wins.
        moves = [(7, 7), (9, 9), (7, 3), (0, 0), (7, 4), (0, 1), (8, 8), (0, 2), (6, 6), (0, 3)]
        moves += [(0, 4), (12, 12), (7, 5), (13, 13), (7, 6)]  # Black blocks, then completes five
        table = {(tuple(moves[:p]), None): ("UNKNOWN", None) for p in range(2, len(moves) + 1)}
        paths = [None] * (len(moves) - 2)
        paths[2 - 2] = "pns_win"  # Black's first engine move claims the win
        for ply in (4, 6, 8, 12):
            paths[ply - 2] = "pns_win"
        record = _record(moves, reason="five in a row", paths=paths)
        record.winner = "a"
        review = review_game(0, record, 2, FakeOracle(table), {}, SIZE, escalate=False)
        assert [b.kind for b in review.blunders if b.engine == "a"] == []

    def test_no_claim_no_report(self):
        review = self._review([None] * 6, self._oracle("UNKNOWN"))
        assert review.blunders == []

    def test_a_disproved_claim_is_reported_once(self):
        # The oracle disproves the claim where it is made: unsound_proof
        # explains the chain, so no second blunder for it.
        review = self._review(["pns_win", None, None, None, None, None], FakeOracle())
        assert [b.kind for b in review.blunders if b.engine == "a"] == ["unsound_proof"]

    def test_report_draws_the_claim_and_the_break(self):
        review = self._review(["pns_win", None, None, None, None, None], self._oracle("UNKNOWN"))
        text = format_report(EngineSpec("threat"), EngineSpec("threat"), [review])
        assert "unconverted win 1" in text
        assert "when the win was claimed (move 3, ! played)" in text
        assert "where the claim broke (move 5)" in text


class TestEscalation:
    @staticmethod
    def _blunder(kind: str = "missed_block", better=(7, 2)) -> Blunder:
        return Blunder(0, 9, "b", "white", kind, (10, 11), better, "", None, 100.0)

    def test_setting_that_fixes_it_is_named_with_its_cost(self):
        slot = FakeSlot({"repeat": (10, 11), "more_pns": (10, 11), "deeper": (7, 2)}, ms=300.0)
        blunder = self._blunder()
        blunder.ms = 100.0
        _escalate(blunder, [*HALF_FOUR, (10, 10)], slot, FakeOracle(), SIZE, 10.0)
        assert blunder.verdict == "tuning"
        assert blunder.remedy == "deeper: fallback.depth=6 (3.0x think time)"
        assert [a.good for a in blunder.attempts] == [False, False, True]  # stops at the first fix

    def test_nothing_helps_means_a_bug(self):
        slot = FakeSlot(dict.fromkeys(["repeat", "more_pns", "deeper", "wider", "all"], (10, 11)))
        blunder = self._blunder()
        _escalate(blunder, [*HALF_FOUR, (10, 10)], slot, FakeOracle(), SIZE, 10.0)
        assert blunder.verdict == "bug"
        assert len(blunder.attempts) == 5

    def test_flaky_move_found_on_repeat(self):
        slot = FakeSlot({"repeat": (7, 2)})
        blunder = self._blunder()
        _escalate(blunder, [*HALF_FOUR, (10, 10)], slot, FakeOracle(), SIZE, 10.0)
        assert blunder.verdict == "tuning" and "timing" in blunder.remedy

    def test_losing_move_retry_is_judged_by_the_oracle(self):
        moves = [(7, 7), (0, 0), (7, 8)]
        oracle = FakeOracle({((*moves, (5, 5)), None): ("PROVED", None)})
        slot = FakeSlot({"repeat": (5, 5), "more_pns": (6, 6)})
        blunder = Blunder(0, 3, "b", "white", "losing_move", (14, 14), (7, 6), "", None, 100.0)
        _escalate(blunder, [*moves, (14, 14)], slot, oracle, SIZE, 10.0)
        assert [a.good for a in blunder.attempts[:2]] == [False, True]
        assert blunder.remedy.startswith("more_pns: pns_nodes=100000, pns_time_s=20.0")


class TestEscalations:
    def test_threat_with_fallback_table(self):
        settings = {
            "depth": 5,
            "pns_nodes": 10_000,
            "pns_time_s": 5.0,
            "threat_search": "pns",
            "vct_depth": 12,
            "fallback": {"depth": 5, "max_candidates": 12},
        }
        assert escalations("threat", settings) == [
            ("more_pns", {"pns_nodes": 100_000, "pns_time_s": 20.0}),
            ("deeper", {"fallback": {"depth": 6}}),
            ("wider", {"fallback": {"max_candidates": 24}}),
            ("all", {"pns_nodes": 100_000, "pns_time_s": 20.0, "fallback": {"depth": 6, "max_candidates": 24}}),
        ]

    def test_threat_without_fallback_keeps_its_depth(self):
        settings = {"depth": 4, "pns_nodes": 10_000, "pns_time_s": 30.0, "threat_search": "vct", "vct_depth": 12}
        variants = dict(escalations("threat", {**settings, "fallback": None}))
        assert variants["more_pns"] == {"pns_nodes": 100_000, "pns_time_s": 60.0, "vct_depth": 16}
        assert variants["deeper"] == {"depth": 5}
        assert variants["wider"] == {"fallback": {"depth": 4, "max_candidates": 40}}
        assert variants["all"]["fallback"] == {"depth": 4, "max_candidates": 40}

    def test_other_brains(self):
        assert [n for n, _ in escalations("alphabeta", {"depth": 4, "max_candidates": 20})] == [
            "deeper",
            "wider",
            "all",
        ]
        assert escalations("mcts", {"iterations": 1000}) == [("more_iterations", {"iterations": 4000})]
        assert escalations("random", {}) == []

    def test_variants_validate_against_the_settings_models(self):
        from gomoku.settings import GomokuSettings
        from gomoku_brain import BRAINS, resolve_brain

        for name in ("threat", "alphabeta", "mcts"):
            brain_type, settings = resolve_brain(name, GomokuSettings().brains)
            for _, overrides in escalations(brain_type, settings):
                BRAINS[brain_type].settings_cls(**merge(settings, overrides))


class TestMerge:
    def test_nested_tables_merge(self):
        base = {"depth": 5, "fallback": {"depth": 5, "max_candidates": 12}}
        assert merge(base, {"fallback": {"depth": 6}}) == {"depth": 5, "fallback": {"depth": 6, "max_candidates": 12}}
        assert merge({"fallback": None}, {"fallback": {"depth": 4}}) == {"fallback": {"depth": 4}}
        assert base["fallback"]["depth"] == 5


class TestVerdict:
    @staticmethod
    def _totals(a_bugs: int, b_bugs: int, b_blunders: int) -> dict[str, EngineTally]:
        return {
            "a": EngineTally(moves=10, blunders=a_bugs, bugs=a_bugs),
            "b": EngineTally(moves=10, blunders=b_blunders, bugs=b_bugs),
        }

    @pytest.mark.parametrize(
        ("a_bugs", "b_bugs", "b_blunders", "status"),
        [(0, 0, 0, "PASS"), (0, 0, 2, "WARN"), (1, 1, 1, "WARN"), (0, 1, 1, "FAIL"), (1, 2, 2, "FAIL")],
    )
    def test_candidate_fails_only_on_extra_bugs(self, a_bugs, b_bugs, b_blunders, status):
        assert verdict(self._totals(a_bugs, b_bugs, b_blunders))[0] == status

    def test_report_lists_candidate_blunders_first(self):
        record = _record([*FOUR, (10, 10), (10, 11)])
        blunders = [
            Blunder(0, 8, "a", "black", "missed_win", (10, 10), (7, 2), "x", None, 50.0, verdict="bug"),
            Blunder(0, 9, "b", "white", "missed_block", (10, 11), (7, 2), "y", "alphabeta", 50.0, verdict="bug"),
        ]
        text = format_report(EngineSpec("x"), EngineSpec("y"), [GameReview(0, record, [], blunders)])
        assert text.index("move 10: B") < text.index("move 9: A")
        assert "B (y): 1 blunder(s) in 4 moves (missed block 1); 0 fixable by tuning the brain's settings, " in text
        assert "1 need a fix in the brain's code" in text
        assert "Verdict: WARN" in text


class TestReviewEndToEnd:
    """Real engine and oracle processes; the random brain keeps it fast."""

    def test_random_vs_random(self):
        seen: list[GameReview] = []
        reviews = run_review(
            EngineSpec("random"),
            EngineSpec("random"),
            games=2,
            seed=3,
            size=9,
            oracle=OracleConfig(nodes=2_000, time_s=1.0, screen_nodes=1_000, screen_time_s=0.5, screen_moves=5),
            on_game=seen.append,
        )
        assert len(reviews) == len(seen) == 2
        for review in reviews:
            record = review.record
            assert len(record.paths) == len(record.moves) - 7
            assert len(review.statuses) == len(record.moves) - 7 + 1
            assert all(s is None or s[0] in {"PROVED", "DISPROVED", "UNKNOWN"} for s in review.statuses)
            for blunder in review.blunders:
                assert blunder.verdict in {"tuning", "bug"}
                assert blunder.attempts or blunder.kind == "unsound_proof"

    def test_cli_report_json_and_exit_code(self, tmp_path: Path):
        out = tmp_path / "review.json"
        args = ["review", "--a", "random", "--b", "random", "--games", "2", "--board-size", "9"]
        args += ["--oracle-nodes", "2000", "--oracle-time", "1", "--no-escalate", "--out", str(out)]
        result = CliRunner().invoke(cli, args)
        payload = json.loads(out.read_text())
        assert "Verdict:" in result.output
        assert result.exit_code == (1 if payload["verdict"]["status"] == "FAIL" else 0), result.output
        assert len(payload["games"]) == 2
        assert {"oracle", "blunders", "record"} <= set(payload["games"][0])


# ---------------------------------------------------------------------------
# Real worker processes
# ---------------------------------------------------------------------------

# Black's open three on row 7 (cols 5-7); White's stones are far away.
OPEN_THREE = [(7, 5), (0, 0), (7, 6), (0, 14), (7, 7)]


@pytest.fixture(scope="module")
def engine():
    """One alpha-beta worker on a 15x15 board, shared by the protocol tests."""
    spec = EngineSpec("alphabeta")
    worker = Engine(spec, resolve_source(spec, find_repo()), SIZE, move_timeout_s=120.0)
    yield worker
    worker.close()


class TestWorkerProtocol:
    def test_settings_are_the_resolved_alias(self, engine):
        from gomoku.settings import GomokuSettings
        from gomoku_brain import resolve_brain

        reply = engine.query({"op": "settings"})
        assert (reply["type"], reply["settings"]) == resolve_brain("alphabeta", GomokuSettings().brains)

    def test_diagnostics_report_the_decision_path(self, engine):
        moves = [(7, 3), (0, 0), (7, 4), (0, 2), (7, 5), (0, 4), (7, 6), (0, 6)]
        reply = engine.query({"moves": moves, "diagnostics": True, "overrides": {"depth": 1}})
        assert tuple(reply["move"]) in {(7, 2), (7, 7)}
        assert reply["path"] in WIN_PATHS  # instant win, or the VCF pre-check finding it
        assert "path" not in engine.query({"moves": moves, "overrides": {"depth": 1}})

    def test_bad_requests_get_an_error_and_the_worker_survives(self, engine):
        with pytest.raises(EngineError, match="ValidationError"):
            engine.query({"moves": OPEN_THREE, "overrides": {"depth": 99}})
        with pytest.raises(EngineError, match="unknown op"):
            engine.query({"op": "nonsense"})
        assert engine.alive
        assert engine.query({"op": "settings"})["type"] == "alphabeta"

    def test_oracle_status_for_either_side(self, engine):
        oracle = _Oracle(engine, OracleConfig(nodes=20_000, time_s=10.0))
        status, move = oracle.status([*OPEN_THREE, (14, 0)])  # Black to move: open three becomes an open four
        assert status == "PROVED"
        assert move is not None and move[0] == 7
        assert oracle.status(OPEN_THREE, Stone.BLACK)[0] == "PROVED"  # same question, asked for the side not to move
        assert oracle.status([*OPEN_THREE, (14, 0)], Stone.WHITE)[0] != "PROVED"

    def test_status_in_the_brain_space_falls_back_without_one(self, engine):
        # alpha-beta has no threat space of its own: the default search runs.
        plain = engine.query({"op": "status", "moves": OPEN_THREE, "nodes": 5_000, "time_s": 5.0})
        brain = engine.query({"op": "status", "moves": OPEN_THREE, "nodes": 5_000, "time_s": 5.0, "space": "brain"})
        assert brain == plain

    def test_losing_move_found_with_the_real_oracle(self, engine):
        # White (ply 5) ignores the open three; blocking an end would have held.
        moves = [*OPEN_THREE, (14, 14)]
        record = GameRecord(
            opening=0,
            a_is_black=True,
            winner=None,
            reason="move limit",
            moves=moves,
            a_ms=[10.0],
            b_ms=[10.0],
            paths=[None, None],
        )
        oracle = _Oracle(engine, OracleConfig(nodes=20_000, time_s=10.0, screen_nodes=5_000, screen_time_s=2.0))
        review = review_game(0, record, 4, oracle, {}, SIZE, escalate=False)
        assert [(b.ply, b.kind, b.engine) for b in review.blunders] == [(5, "losing_move", "b")]
        assert review.blunders[0].better in {(7, 4), (7, 8)}
        assert review.statuses[-1][0] == "PROVED"


class TestEngineSlot:
    def test_timed_out_retry_restarts_the_engine(self):
        spec = EngineSpec("alphabeta")
        source = resolve_source(spec, find_repo())
        slot = _EngineSlot(lambda: Engine(spec, source, SIZE, move_timeout_s=120.0))
        try:
            first = slot.engine
            quiet = [(7, 7), (8, 8), (6, 9), (9, 5)]  # nothing to shortcut: a full search runs
            assert slot.retry(quiet, {"depth": 8, "max_candidates": 60, "vcf_depth": 0}, timeout_s=0.01) is None
            assert not first.alive and slot.engine is not first
            reply = slot.retry(OPEN_THREE, {"depth": 1}, timeout_s=60.0)
            assert reply is not None and len(reply["move"]) == 2
        finally:
            slot.close()


class TestCliVerdict:
    def test_fail_exits_with_status_one(self, tmp_path: Path, monkeypatch):
        import gomoku_cli.review as review_module

        record = _record([*HALF_FOUR, (10, 10), (10, 11)])
        bug = Blunder(0, 9, "b", "white", "missed_block", (10, 11), (7, 2), "y", None, 50.0, verdict="bug")
        monkeypatch.setattr(review_module, "run_review", lambda *a, **k: [GameReview(0, record, [], [bug])])
        out = tmp_path / "review.json"
        result = CliRunner().invoke(cli, ["review", "--a", "random", "--b", "random", "--out", str(out)])
        assert result.exit_code == 1, result.output
        assert "Verdict: FAIL" in result.output
        payload = json.loads(out.read_text())
        assert payload["verdict"]["status"] == "FAIL"
        assert payload["totals"]["b"]["bugs"] == 1
        assert payload["games"][0]["blunders"][0]["kind"] == "missed_block"


class TestOracleSource:
    def test_worktree_oracle_runs_from_the_working_tree(self, monkeypatch):
        import gomoku_cli.review as review_module

        class Resolved(Exception):
            pass

        seen: list[EngineSpec] = []

        def fake_resolve(spec, repo):
            seen.append(spec)
            if len(seen) == 3:  # A, B, then the oracle: stop before starting engines
                raise Resolved
            return None

        monkeypatch.setattr(review_module, "resolve_source", fake_resolve)
        monkeypatch.setattr(review_module, "generate_openings", lambda *a, **k: [])
        for oracle_rev, expected in (("worktree", None), (None, "base"), ("other", "other")):
            seen.clear()
            with pytest.raises(Resolved):
                review_module.run_review(
                    EngineSpec("random", rev="base"), EngineSpec("random"), games=2, oracle_rev=oracle_rev
                )
            assert seen[2].rev == expected


class TestSecondOracle:
    MOVES = ((7, 7), (0, 0), (7, 8), (14, 14), (3, 3), (4, 4))

    def _review(self, second):
        # The brain claims a forced win at ply 2 that the first oracle disproves.
        record = _record(list(self.MOVES), paths=["pns_win", None, None, None])
        return review_game(0, record, 2, FakeOracle(), {}, SIZE, escalate=False, second_oracle=second)

    def test_agreeing_oracles_confirm_the_blunder(self):
        review = self._review(FakeOracle())
        assert [(b.kind, b.verdict) for b in review.blunders] == [("unsound_proof", "bug")]

    def test_disagreeing_oracles_dispute_it(self):
        # The candidate's own search proves the claimed win.
        second = FakeOracle({(tuple(self.MOVES[:2]), None): ("PROVED", (7, 8))})
        second.table[(tuple(self.MOVES[:4]), None)] = ("PROVED", (7, 9))
        review = self._review(second)
        (blunder,) = review.blunders
        assert blunder.verdict == "disputed" and "no blunder" in blunder.remedy
        totals = tally([review])
        assert (totals["b"].bugs, totals["a"].bugs, totals["a"].disputed) == (0, 0, 1)

    def test_disputed_blunders_warn_but_do_not_fail(self):
        totals = {"a": EngineTally(moves=10), "b": EngineTally(moves=10, blunders=1, disputed=1)}
        assert verdict(totals)[0] == "WARN"
        record = _record(list(self.MOVES))
        b = Blunder(0, 3, "b", "white", "losing_move", (14, 14), None, "x", None, 1.0, verdict="disputed", remedy="r")
        text = format_report(EngineSpec("x"), EngineSpec("y"), [GameReview(0, record, [], [b])])
        assert "=> DISPUTED: r" in text and "1 disputed by the second oracle" in text


class TestBoards:
    def test_render_board_crops_and_marks(self):
        from gomoku_cli.review import render_board

        text = render_board([(7, 7), (7, 8), (8, 7)], {(9, 7): "!", (6, 7): "*"}, size=15, pad=1)
        assert text.splitlines() == [
            "    G H I J",
            " 11 . . . .",
            " 10 . ! . .",
            "  9 . x . .",
            "  8 . X O .",
            "  7 . * . .",
            "  6 . . . .",
        ]

    def test_report_draws_each_blunder_unless_asked_not_to(self):
        record = _record([*HALF_FOUR, (10, 10), (10, 11)])
        blunder = Blunder(0, 9, "b", "white", "missed_block", (10, 11), (7, 2), "y", None, 50.0, verdict="bug")
        review = GameReview(0, record, [], [blunder])
        text = format_report(EngineSpec("x"), EngineSpec("y"), [review])
        assert "board: X black, O white" in text
        assert "  8 . . * X X X X O" in text  # the blocked four, completed at C8
        assert "!" in text.split("=> BUG")[1] and "*" in text.split("=> BUG")[1]
        plain = format_report(EngineSpec("x"), EngineSpec("y"), [review], boards=False)
        assert "board:" not in plain and "  8 " not in plain
