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
    """Guard against the "K8 bug" class of unsound proofs.

    The original regression — game-2 move 42, Black G11 declared a
    winning VCT — happened because the old defence-move generator did
    not include the opponent's counter-four cells.  Black's attack
    looked forced because only natural blocks were considered; the
    lethal counter-four at K8 was invisible, so PNS would falsely
    declare Black's live-three attack a forced win.

    These tests encode the bug's *structure* rather than the specific
    board: any position where the attacker has only a live-three while
    the defender has a live-three that extends to a live-four must be
    proved unwinnable.
    """

    def _fork_setup(self) -> Board:
        """Black double-three fork at (7, 7) — playing it creates a
        horizontal AND a vertical live-three simultaneously.

        Horizontal live-two: (7, 5), (7, 6) → + (7, 7) = live-three.
        Vertical live-two:   (5, 7), (6, 7) → + (7, 7) = live-three.
        """
        b = Board(size=15)
        return _place(b, [(7, 5), (7, 6), (5, 7), (6, 7)], Stone.BLACK)

    def test_fork_wins_without_counter_threat(self):
        """Sanity check: a double-three fork is a classical forced win
        — White can only block one of the two live-threes, the other
        extends to a live-four on Black's next move.  PNS must find it.

        If this test fails, ``test_counter_four_refutes_fork_attack``
        below is proving the wrong thing (PNS simply can't solve forks
        rather than correctly handling the counter-threat)."""
        b = self._fork_setup()
        # White material is cosmetic / unrelated to the fork
        b = _place(b, [(0, 0), (0, 1), (14, 14)], Stone.WHITE)

        move = pns_threat_search(b, Stone.BLACK, max_nodes=50_000)
        assert move is not None, (
            "PNS failed to solve the baseline double-three fork — the counter-threat test below is moot."
        )
        # The fork cell (7, 7) should be the proven move.
        assert move == Move(row=7, col=7)

    def test_counter_four_refutes_fork_attack(self):
        """Guard against the K8 bug: same Black fork as above, but now
        White has a live-three that becomes an unblockable live-four.

        After Black plays the fork at (7, 7), White's defence set
        (``_pns_defence_moves``) must include the counter-four cells
        (3, 3) / (3, 7).  When White plays one, the resulting live-four
        forces Black to chase blocks instead of converting the fork —
        and since Black can only block one end of a live-four, White
        wins the race.  PNS must therefore *not* declare the fork a
        forced win.

        This is the synthetic analogue of the original game-2 K8
        position: the attacker's threat sequence is disrupted by a
        defender counter-four that arrives first on the board.
        """
        b = self._fork_setup()
        # White live-three — playing (3, 3) or (3, 7) creates a live-four
        b = _place(b, [(3, 4), (3, 5), (3, 6)], Stone.WHITE)

        move = pns_threat_search(b, Stone.BLACK, max_nodes=50_000)
        assert move is None, (
            f"PNS falsely proved a forced win at {move} — White's "
            "counter-four at (3,3)/(3,7) refutes the double-three fork."
        )

    def test_pns_result_is_sound_when_positive(self):
        """Whenever PNS returns a move, that move must be legal and must
        either complete five-in-a-row immediately or produce at least
        one four-threat for the attacker.  A proof violating either
        invariant is unsound."""
        from gomoku.rules import check_winner
        from gomoku_brain.threat import _find_four_threats

        b = Board(size=15)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        move = pns_threat_search(b, Stone.BLACK, max_nodes=5_000)
        assert move is not None
        assert b.is_valid_move(move)
        trial = b.place_stone(move, Stone.BLACK)
        won = check_winner(trial, move) is not None
        fours = _find_four_threats(trial, Stone.BLACK)
        assert won or len(fours) > 0, (
            f"PNS returned {move} but it neither wins nor creates a four-threat — unsound proof"
        )


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


# ---------------------------------------------------------------------------
# Shared positions (search graph instead of a tree)
# ---------------------------------------------------------------------------


class TestPnsTranspositions:
    @staticmethod
    def _midgame() -> Board:
        """A 19-stone position from a self-played game; Black to move."""
        moves = [
            (7, 7), (7, 8), (8, 7), (6, 8), (8, 8), (6, 7), (6, 9), (8, 9), (5, 6), (9, 9),
            (9, 10), (5, 8), (4, 8), (10, 9), (7, 9), (4, 9), (7, 6), (2, 11), (3, 10),
        ]  # fmt: skip
        board = Board(size=15)
        stone = Stone.BLACK
        for r, c in moves:
            board = board.place_stone(Move(row=r, col=c), stone)
            stone = stone.opponent()
        return board

    def test_finds_win_the_tree_search_missed(self):
        """With one node per position a 100k budget reaches this forced
        win; the tree search re-searched move-order permutations and needed
        a far larger budget.  (10k sufficed while split fours were invisible
        to the defence enumeration; seeing the defender's split counter-fours
        widens the proof.  With split_threes it takes 300k.)"""
        move = pns_threat_search(self._midgame(), Stone.BLACK, max_nodes=100_000, max_time_s=60.0)
        assert move == Move(row=6, col=6)

    def test_move_orders_share_one_node(self):
        """Two attack orders reaching the same position link to one node."""
        from gomoku_brain.pns import _child_at, _NodeType
        from gomoku_brain.search_state import SearchState

        board = _place(Board(size=15), [(7, 7)], Stone.BLACK)
        board = _place(board, [(0, 0)], Stone.WHITE)
        state = SearchState.from_board(board, Stone.BLACK)
        table: dict = {}

        def reach(first: tuple[int, int], second: tuple[int, int]):
            state.make(*first, Stone.BLACK)
            state.make(0, 14, Stone.WHITE)
            state.make(*second, Stone.BLACK)
            try:
                return _child_at(state, _NodeType.AND, table)
            finally:
                for _ in range(3):
                    state.unmake()

        node_a, new_a = reach((7, 8), (7, 9))
        node_b, new_b = reach((7, 9), (7, 8))
        assert new_a and not new_b
        assert node_a is node_b

    def test_caller_state_is_restored(self):
        from gomoku_brain.search_state import SearchState

        state = SearchState.from_board(self._midgame(), Stone.BLACK)
        ply, zobrist = state.ply, state.zobrist
        pns_threat_search(state, Stone.BLACK, max_nodes=2_000)
        assert (state.ply, state.zobrist) == (ply, zobrist)


# ---------------------------------------------------------------------------
# Search status: proved / disproved / budget exhausted
# ---------------------------------------------------------------------------


class TestPnsStatus:
    def test_proved_carries_the_winning_move(self):
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        b = _place(Board(size=15), [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        status, move = pns_threat_status(b, Stone.BLACK, max_nodes=1_000)
        assert status is PNSStatus.PROVED
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_no_threats_is_disproved(self):
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        b = _place(Board(size=15), [(7, 7)], Stone.BLACK)
        b = _place(b, [(3, 3)], Stone.WHITE)
        assert pns_threat_status(b, Stone.BLACK, max_nodes=1_000) == (PNSStatus.DISPROVED, None)

    def test_exhausted_budget_is_unknown_not_disproved(self):
        """The same position: a tiny budget proves nothing, a full one proves the win."""
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        board = TestPnsTranspositions._midgame()
        assert pns_threat_status(board, Stone.BLACK, max_nodes=20) == (PNSStatus.UNKNOWN, None)
        assert pns_threat_search(board, Stone.BLACK, max_nodes=20) is None
        status, _ = pns_threat_status(board, Stone.BLACK, max_nodes=100_000, max_time_s=60.0)
        assert status is PNSStatus.PROVED


# ---------------------------------------------------------------------------
# Regression: a recorded game the threat brain lost
# ---------------------------------------------------------------------------

# Moves alternate Black, White from move 1 (row, col).
_LOST_GAME = [
    (7, 7), (6, 6), (8, 8), (5, 7), (7, 5), (4, 8), (7, 8), (3, 9), (2, 10), (7, 6),
    (6, 8), (5, 8), (5, 9), (8, 6), (5, 6), (6, 7), (8, 5), (4, 10),
]  # fmt: skip


def _replay(moves: list[tuple[int, int]]) -> Board:
    board = Board(size=15)
    stone = Stone.BLACK
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return board


class TestUnblockableFourDefences:
    """Before move 11 PNS "proved" that I7 wins for Black.  After I7 Black
    threatens I10, a four open at both ends; PNS let White answer only at
    I10, but White can also take an end of the would-be four (I6, as White
    played, or I11), after which Black has no forced win."""

    def test_defence_set_includes_the_ends_of_the_would_be_four(self):
        from gomoku_brain.pns import _pns_defence_moves
        from gomoku_brain.search_state import SearchState

        board = _replay(_LOST_GAME[:10]).place_stone(Move(row=6, col=8), Stone.BLACK)  # I7
        defences = {(m.row, m.col) for m in _pns_defence_moves(SearchState.from_board(board, Stone.WHITE), Stone.BLACK)}
        assert {(9, 8), (5, 8), (10, 8)} <= defences  # I10, I6, I11

    def test_no_false_win_before_move_11(self):
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        status, _ = pns_threat_status(_replay(_LOST_GAME[:10]), Stone.BLACK, max_nodes=50_000, max_time_s=60.0)
        assert status is not PNSStatus.PROVED

    def test_reply_i6_does_not_save_white(self):
        """I6 looked like White's defence while split fours (XXX_X, XX_XX)
        were invisible.  Black still wins with them; the proof was checked
        against every White reply, two attacking moves deep."""
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        board = _replay([*_LOST_GAME[:10], (6, 8), (5, 8)])  # I7, I6
        status, _ = pns_threat_status(board, Stone.BLACK, max_nodes=200_000, max_time_s=60.0)
        assert status is PNSStatus.PROVED
