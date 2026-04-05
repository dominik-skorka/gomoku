"""Tests for the Proof-Number Search (PNS) threat-space algorithm."""

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.pns import pns_threat_search

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _place(board: Board, moves: list[tuple[int, int]], stone: Stone) -> Board:
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
    return board


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


class TestPnsBasic:
    def test_finds_immediate_five(self):
        """Four-in-a-row → PNS should find the winning move."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        move = pns_threat_search(b, Stone.BLACK, max_nodes=1_000)
        assert move is not None
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_returns_none_when_no_threats(self):
        """Single stone on board → no threats → PNS returns None."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)
        move = pns_threat_search(b, Stone.BLACK, max_nodes=1_000)
        assert move is None

    def test_live_four_wins(self):
        """Open three → placing at either end creates a live four (unblockable)."""
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (8, 8)], Stone.WHITE)
        move = pns_threat_search(b, Stone.BLACK, max_nodes=5_000)
        assert move is not None
        # A move extending the three to a live four
        trial = b.place_stone(move, Stone.BLACK)
        # The move should create a four-in-a-row
        from gomoku_brain.threat import _find_four_threats

        fours = _find_four_threats(trial, Stone.BLACK)
        assert len(fours) > 0


# ---------------------------------------------------------------------------
# Counter-threat soundness (the K8 bug scenario)
# ---------------------------------------------------------------------------


class TestPnsCounterThreats:
    def test_does_not_prove_false_win(self):
        """Reproduce the game-2 position where old VCT falsely declared G11 a win.

        After Black plays G11, White can play K8 creating an unblockable
        four at K5. PNS must NOT declare G11 as a winning move.

        Board at move 42 (Black to play) from game 2 (threat vs alphabeta).
        """
        import json
        from pathlib import Path

        recording = Path(__file__).resolve().parents[2] / "../../recordings/gomoku_20260405_120630.json"
        if not recording.exists():
            # Skip if recording not available
            return

        with open(recording) as f:
            data = json.load(f)

        b = Board(size=15)
        stone_map = {1: Stone.BLACK, 2: Stone.WHITE}
        for m in data["moves"]:
            if m["number"] >= 43:
                break
            s = stone_map[m["stone"]]
            b = b.place_stone(Move(row=m["move"]["row"], col=m["move"]["col"]), s)

        # PNS should NOT claim G11 as a forced win (it's not — White has K8 counter)
        move = pns_threat_search(b, Stone.BLACK, max_nodes=50_000)
        if move is not None:
            # If PNS claims a win, it must not be G11 (which is unsound)
            # — or if it IS G11, PNS must have actually proved it
            # through all defender responses including K8.
            # Let's verify: after G11 + K8, PNS shouldn't find a continuation.
            g11 = Move(row=10, col=6)
            if move == g11:
                board_after = b.place_stone(g11, Stone.BLACK)
                k8 = Move(row=7, col=10)
                board_after = board_after.place_stone(k8, Stone.WHITE)
                followup = pns_threat_search(board_after, Stone.BLACK, max_nodes=50_000)
                # If PNS proves G11 works, it must also work after K8
                # (which we know it doesn't from analysis)
                assert followup is not None, "PNS claimed G11 wins but can't continue after K8 — unsound proof"


class TestPnsChainedThreats:
    def test_two_connected_live_threes(self):
        """Two open threes that can be chained into a forced win."""
        b = Board(size=9)
        # Row 4: open three
        b = _place(b, [(4, 2), (4, 3), (4, 4)], Stone.BLACK)
        # Row 6: open three
        b = _place(b, [(6, 2), (6, 3), (6, 4)], Stone.BLACK)
        # White far away
        b = _place(b, [(0, 0), (0, 8), (8, 0), (8, 8)], Stone.WHITE)

        move = pns_threat_search(b, Stone.BLACK, max_nodes=10_000)
        # Should find a forced win (live-four is unblockable)
        assert move is not None

    def test_does_not_find_win_when_blocked(self):
        """A position where all attack lines are blocked."""
        b = Board(size=9)
        # Black has one stone
        b = _place(b, [(4, 4)], Stone.BLACK)
        # White surrounds it
        b = _place(b, [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)], Stone.WHITE)
        move = pns_threat_search(b, Stone.BLACK, max_nodes=5_000)
        assert move is None


# ---------------------------------------------------------------------------
# ThreatBrain integration with PNS
# ---------------------------------------------------------------------------


class TestThreatBrainPns:
    def test_pns_mode_finds_win(self):
        """ThreatBrain with threat_search='pns' finds a forced win."""
        from gomoku_brain.threat import ThreatBrain

        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)

        brain = ThreatBrain(threat_search="pns", pns_nodes=5_000)
        move = brain.get_move(b, Stone.BLACK)
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_vct_mode_still_works(self):
        """ThreatBrain with threat_search='vct' uses VCT."""
        from gomoku_brain.threat import ThreatBrain

        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)

        brain = ThreatBrain(threat_search="vct", vct_depth=12)
        move = brain.get_move(b, Stone.BLACK)
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_pns_fallback_on_no_threats(self):
        """PNS mode falls back to alpha-beta when no threats exist."""
        from gomoku_brain.threat import ThreatBrain

        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)

        brain = ThreatBrain(threat_search="pns", pns_nodes=1_000)
        move = brain.get_move(b, Stone.BLACK)
        assert b.is_valid_move(move)
