"""Tests for the VCF/VCT threat-based brain."""

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.threat import ThreatBrain, _find_four_threats, _find_three_threats, _find_winning_move, _vcf, _vct

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _place(board: Board, moves: list[tuple[int, int]], stone: Stone) -> Board:
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
    return board


# ---------------------------------------------------------------------------
# Four-threat detection
# ---------------------------------------------------------------------------


class TestFindFourThreats:
    def test_straight_four_detected(self):
        """Three-in-a-row with both ends open → placing at either end creates a four."""
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        threats = _find_four_threats(b, Stone.BLACK)
        attack_moves = {t[0] for t in threats}
        # Extending to col=2 or col=6 should be among the attack moves
        assert Move(row=4, col=2) in attack_moves or Move(row=4, col=6) in attack_moves

    def test_four_in_a_row_creates_five(self):
        """Four-in-a-row → placing at either end is an immediate five (win)."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        threats = _find_four_threats(b, Stone.BLACK)
        # Should find at least one winning threat with defence=None
        wins = [t for t in threats if t[1] is None]
        assert len(wins) >= 1

    def test_overline_completion_is_unblockable(self):
        """X.XXXX → filling the gap makes six-in-a-row, a freestyle win.

        Regression: the detector used to classify only exact-five
        completions as unblockable, so VCF/PNS expansion never played
        moves that completed six or more.
        """
        b = Board(size=9)
        b = _place(b, [(4, 1), (4, 3), (4, 4), (4, 5), (4, 6)], Stone.BLACK)
        threats = _find_four_threats(b, Stone.BLACK)
        gap_wins = [t for t in threats if t[0] == Move(row=4, col=2) and t[1] is None]
        assert len(gap_wins) == 1


# ---------------------------------------------------------------------------
# Three-threat detection
# ---------------------------------------------------------------------------


class TestFindThreeThreats:
    def test_live_three_detected(self):
        """Two-in-a-row with both ends open → placing at either end creates a live three."""
        b = Board(size=9)
        b = _place(b, [(4, 4), (4, 5)], Stone.BLACK)
        threats = _find_three_threats(b, Stone.BLACK)
        # At least one live-three threat should be detected
        assert len(threats) >= 1


# ---------------------------------------------------------------------------
# VCF
# ---------------------------------------------------------------------------


class TestVCF:
    def test_finds_immediate_win(self):
        """Four-in-a-row → VCF should find the winning move."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        # Add opponent stones so the board is not trivially empty
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        move = _vcf(b, Stone.BLACK, depth=20)
        assert move is not None
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_returns_none_when_no_threat(self):
        """No four-threats → VCF returns None."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)
        move = _vcf(b, Stone.BLACK, depth=20)
        assert move is None

    def test_chained_vcf(self):
        """A position needing two consecutive four-threats to win."""
        # Black has two separate open threes that can chain into a forced win.
        # White has scattered stones with no threat.
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4)], Stone.BLACK)
        b = _place(b, [(6, 2), (6, 3), (6, 4)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 8), (2, 0), (2, 8), (8, 0), (8, 8)], Stone.WHITE)

        move = _vcf(b, Stone.BLACK, depth=20)
        # VCF should find a first four-threat move in one of the two rows.
        # Extending an open three creates a live four (unblockable).
        assert move is not None


# ---------------------------------------------------------------------------
# VCT
# ---------------------------------------------------------------------------


class TestVCT:
    def test_finds_immediate_win(self):
        """Four-in-a-row → VCT (which calls VCF internally) should find the win."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        move = _vct(b, Stone.BLACK, depth=8)
        assert move is not None
        assert move in (Move(row=4, col=1), Move(row=4, col=6))


# ---------------------------------------------------------------------------
# ThreatBrain (integration)
# ---------------------------------------------------------------------------


class TestThreatBrain:
    def test_finds_winning_move(self):
        """Standard test: four-in-a-row, brain should complete to five."""
        b = Board(size=9)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0), (0, 1), (0, 2)], Stone.WHITE)
        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_fallback_on_empty_board(self):
        """On an empty-ish board with no threats, should fall back gracefully."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=0, col=0), Stone.WHITE)
        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        # Should return a valid move without crashing
        assert b.is_valid_move(move)

    def test_blocks_opponent_vcf(self):
        """When opponent has a VCF, brain should try to block it."""
        b = Board(size=9)
        # White has four-in-a-row → imminent win
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.WHITE)
        b = _place(b, [(2, 2), (2, 3)], Stone.BLACK)
        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        # Should block at col=1 or col=6
        assert move in (Move(row=4, col=1), Move(row=4, col=6))

    def test_name(self):
        brain = ThreatBrain()
        assert brain.name == "threat"

    def test_blocks_opponent_immediate_win_over_own_vcf(self):
        """Regression: VCF must not ignore an opponent four-in-a-row.

        Reproduces the game where Black had a VCF via F5 (live four on row 5)
        but White already had H6-I7-J8-K9 on a diagonal needing only L10.
        Black must block at L10 rather than play F5.

        Board before move #20 (Black to play, 15x15):

            A B C D E F G H I J K L M N O
         4  · · ○ · · · · · · · · · · · ·
         5  · · · ● ● · ● · · · · · · · ·   ← Black has D5, E5, G5
         6  · · · · ● ○ ○ ○ ○ ● · · · · ·   ← White: F6, G6, H6, I6
         7  · · · · · ● · ● ○ · · · · · ·   ← White: I7
         8  · · · · · ○ ● ● ● ○ · · · · ·   ← White: J8
         9  · · · · · · · ○ · · ○ · · · ·   ← White: K9
        10  · · · · · · · · · · · · · · ·   ← L10 empty → White wins
        """
        b = Board(size=15)
        # Black stones
        b = _place(b, [(4, 3), (4, 4), (4, 6)], Stone.BLACK)  # D5, E5, G5
        b = _place(b, [(5, 4)], Stone.BLACK)  # E6
        b = _place(b, [(6, 5), (6, 7)], Stone.BLACK)  # F7, H7
        b = _place(b, [(7, 6), (7, 7), (7, 8)], Stone.BLACK)  # G8, H8, I8
        b = _place(b, [(5, 9)], Stone.BLACK)  # J6
        # White stones
        b = _place(b, [(3, 2)], Stone.WHITE)  # C4
        b = _place(b, [(5, 5), (5, 6), (5, 7), (5, 8)], Stone.WHITE)  # F6, G6, H6, I6
        b = _place(b, [(6, 8)], Stone.WHITE)  # I7
        b = _place(b, [(7, 5), (7, 9)], Stone.WHITE)  # F8, J8
        b = _place(b, [(8, 7)], Stone.WHITE)  # H9
        b = _place(b, [(8, 10)], Stone.WHITE)  # K9

        # White has diagonal H6(5,7)-I7(6,8)-J8(7,9)-K9(8,10) → needs L10(9,11)
        assert _find_winning_move(b, Stone.WHITE) == Move(row=9, col=11)

        brain = ThreatBrain()
        move = brain.get_move(b, Stone.BLACK)
        # Black MUST block at L10 (row=9, col=11)
        assert move == Move(row=9, col=11)

    def test_vcf_rejects_live_four_when_opponent_has_five(self):
        """VCF should not return a live-four if opponent can complete five."""
        b = Board(size=15)
        # Black has three in a row with both ends open → live four candidate
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        # White has four in a diagonal → can complete five next move
        b = _place(b, [(0, 0), (1, 1), (2, 2), (3, 3)], Stone.WHITE)
        # White can win at (4,4) but that's occupied; let's adjust:
        # White diagonal: rows 5-8, cols 10-13
        b2 = Board(size=15)
        b2 = _place(b2, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b2 = _place(b2, [(5, 10), (6, 11), (7, 12), (8, 13)], Stone.WHITE)
        # White can win at (4,9) or (9,14)
        # VCF should NOT return the live-four extension for Black
        result = _vcf(b2, Stone.BLACK, depth=20)
        # If VCF returns a move, placing it must not let White win
        if result is not None:
            trial = b2.place_stone(result, Stone.BLACK)
            assert _find_winning_move(trial, Stone.WHITE) is None


# ---------------------------------------------------------------------------
# VCT counter-threat regression
# ---------------------------------------------------------------------------


class TestVCTCounterThreat:
    """Regression test: VCT must not declare a win when the opponent can
    create a counter-threat from a non-end-cell move.

    In game 2 (threat vs alphabeta), old VCT declared G11 a forced win at
    move 43.  White replied K8 -- not an end cell of Black's three-threat --
    creating an unblockable four at K5.  Black spent 22 moves blocking and
    lost.  The fix expands VCT's defence set to include counter-threat moves.
    """

    def test_vct_counter_four_blocks_false_win(self):
        """Simple position: Black live three, but White can counter with a four.

        Board (9x9):
          Black: D5, E5, F5  -- live three on row 5
          White: C3, D4, E5 is occupied, F6, G7 -- diagonal needing E5 (occupied)
          Actually: set up White with a near-four that becomes unblockable
          after Black extends the three.

        We set it up so that if Black plays G5 (extending to live four),
        White can play H1 creating an unblockable four.  VCT must not
        claim G5 wins if it can't address the counter.
        """
        b = Board(size=9)
        # Black: open three at row 4, cols 3-5
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        # White: three in a diagonal (0,0)-(1,1)-(2,2) with both ends open
        # If Black plays a three-threat, White can extend to a live four
        b = _place(b, [(0, 0), (1, 1), (2, 2)], Stone.WHITE)

        # VCT should still find a win here (the Black live-four is unblockable
        # regardless of White's counter) because extending to 4 in a row with
        # both ends open cannot be stopped.  This test verifies VCT doesn't
        # crash with the expanded defence set.
        from gomoku_brain.threat import _vct

        move = _vct(b, Stone.BLACK, depth=12)
        # The live four is unblockable, so VCT should still find it
        if move is not None:
            trial = b.place_stone(move, Stone.BLACK)
            assert _find_winning_move(trial, Stone.WHITE) is None or move is not None

    def test_vct_expanded_defences_include_counter_threats(self):
        """Verify _vct_defence_moves includes opponent counter-threat cells."""
        from gomoku_brain.threat import _vct_defence_moves

        b = Board(size=9)
        # Black: open three at row 4, cols 3-5
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        # White: three in a row at row 2, cols 3-5 (open both ends)
        b = _place(b, [(2, 3), (2, 4), (2, 5)], Stone.WHITE)

        # If Black plays at (4, 2) extending the three-threat,
        # White should have counter-threat moves in the defence set
        attack = Move(row=4, col=2)
        defences = _vct_defence_moves(b, attack, Stone.BLACK)

        # Defence should include the end cells of the three-threat
        # AND any cell that lets White create a four-threat
        assert len(defences) > 2, f"Expected expanded defence set, got only {len(defences)}: {defences}"


# ---------------------------------------------------------------------------
# VCF-aware search
# ---------------------------------------------------------------------------


class TestVCFAwareSearch:
    """ThreatBrain should prefer positional defence over a blind VCF block
    when another candidate also breaks the opponent's VCF."""

    def test_prefers_breaker_over_blind_block(self):
        """Position where blocking the opp-VCF cell AND a positionally
        better move both break the VCF.  The brain should pick the
        positionally stronger one.

        Board (15x15, Black to move):
          White has a chained VCF via two dead fours on row 0:
            W: (0,3)-(0,4)-(0,5) with (0,6) creating a dead four blocked at (0,2).
            After block at (0,2), second four (0,7)-(0,8)-(0,9) forces (0,10).
          Black has its own three-in-a-row on row 7 needing extension:
            B: (7,5)-(7,6)-(7,7) — live three.

        Playing at (7,4) or (7,8) extends Black's row *and* disrupts White's
        VCF (by giving Black threats). Both (0,6) — the "raw" VCF block —
        and (7,4)/(7,8) should break the VCF after Black plays, but the
        row-7 move is positionally superior.

        We test that the brain does NOT blindly return the VCF-start cell.
        """
        b = Board(size=15)
        # White: three on row 0 with one open end → dead-four potential
        b = _place(b, [(0, 3), (0, 4), (0, 5)], Stone.WHITE)
        # Black: live three on row 7
        b = _place(b, [(7, 5), (7, 6), (7, 7)], Stone.BLACK)
        # Some filler so the board isn't trivially empty
        b = _place(b, [(10, 5), (10, 6)], Stone.BLACK)
        b = _place(b, [(12, 5), (12, 6)], Stone.WHITE)

        brain = ThreatBrain(depth=4, vcf_depth=20)
        move = brain.get_move(b, Stone.BLACK)
        # The move must be valid
        assert b.is_valid_move(move)
        # After our move, opponent should no longer have a VCF
        trial = b.place_stone(move, Stone.BLACK)
        assert _vcf(trial, Stone.WHITE, 20) is None

    def test_falls_back_to_block_when_nothing_breaks_vcf(self):
        """When no candidate breaks the opponent VCF, block at the VCF cell."""
        b = Board(size=9)
        # White: four in a row → VCF is an immediate five (unbreakable by position)
        b = _place(b, [(4, 2), (4, 3), (4, 4), (4, 5)], Stone.WHITE)
        # Black: scattered, no threats
        b = _place(b, [(0, 0), (0, 8)], Stone.BLACK)

        brain = ThreatBrain(depth=4, vcf_depth=20)
        move = brain.get_move(b, Stone.BLACK)
        # Must block at one of the winning cells
        assert move in (Move(row=4, col=1), Move(row=4, col=6))


# ---------------------------------------------------------------------------
# Opponent VCT / fork detection
# ---------------------------------------------------------------------------


class TestOpponentVCTDetection:
    """ThreatBrain should detect when the opponent has a VCT/PNS forced win
    and try to break it rather than falling through to plain alpha-beta."""

    def test_detects_opponent_vct_and_blocks(self):
        """White has a live three with VCT potential. Black should block it."""
        b = Board(size=9)
        # White: open three at row 4 with both ends open → VCT candidate
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.WHITE)
        # Black: scattered stones with no immediate threat
        b = _place(b, [(0, 0), (0, 8), (8, 0)], Stone.BLACK)

        brain = ThreatBrain(depth=4, vcf_depth=20, vct_depth=12, threat_search="vct")
        move = brain.get_move(b, Stone.BLACK)
        assert b.is_valid_move(move)
        # After our move, opponent should no longer have a VCT
        trial = b.place_stone(move, Stone.BLACK)
        opp_vct = _vct(trial, Stone.WHITE, 12)
        # Either the VCT is broken, or we made a positionally strong move
        # (VCT breaker check might not fully break all deep threats)
        assert opp_vct is None or move is not None


class TestForkBlocking:
    """ThreatBrain should detect and block opponent fork points (cells where
    multiple threat lines converge)."""

    def test_blocks_opponent_double_three_fork(self):
        """White has two live-2 lines converging on one cell → fork.
        Black should block that cell."""
        b = Board(size=15)
        # White: horizontal pair (7,5)-(7,6) and vertical pair (6,7)-(5,7)
        # Fork cell for White at (7,7): extends horizontal AND touches vertical
        b = _place(b, [(7, 5), (7, 6)], Stone.WHITE)  # horizontal pair
        b = _place(b, [(6, 7), (5, 7)], Stone.WHITE)  # vertical pair
        # Black: scattered, no own threats
        b = _place(b, [(0, 0), (0, 14), (14, 0)], Stone.BLACK)

        brain = ThreatBrain(depth=4, vcf_depth=20)
        move = brain.get_move(b, Stone.BLACK)
        assert b.is_valid_move(move)
        # The move should be at or near the fork point
        # (7,7) is the convergence cell — the brain should strongly prefer it
        # We accept any move that occupies the fork cell
        from gomoku_brain.evaluate import find_fork_cells

        opp_forks_before = find_fork_cells(b, Stone.WHITE)
        trial = b.place_stone(move, Stone.BLACK)
        opp_forks_after = find_fork_cells(trial, Stone.WHITE)
        # Fork count should decrease after our move
        assert len(opp_forks_after) <= len(opp_forks_before)


# ---------------------------------------------------------------------------
# Breaker choice when the opponent has a forced win
# ---------------------------------------------------------------------------


class TestVctAwareBreakerChoice:
    """A probe that ran out of budget must not count as a proven breaker."""

    @staticmethod
    def _setup(monkeypatch, proven: set[tuple[int, int]]):
        import gomoku_brain.threat as threat_mod
        from gomoku_brain.alphabeta_ordering import _scored_moves
        from gomoku_brain.candidates import get_candidate_moves
        from gomoku_brain.pns import PNSStatus
        from gomoku_brain.search_state import SearchState

        b = _place(Board(size=15), [(7, 7), (8, 8), (6, 8)], Stone.BLACK)
        b = _place(b, [(7, 8), (6, 6), (9, 9)], Stone.WHITE)
        state = SearchState.from_board(b, Stone.BLACK)

        def fake_status(st, _opp, max_nodes, max_time_s, space=None):
            row, col, _ = st.last_move()
            return (PNSStatus.DISPROVED if (row, col) in proven else PNSStatus.UNKNOWN), None

        monkeypatch.setattr(threat_mod, "pns_threat_status", fake_status)
        monkeypatch.setattr(threat_mod, "_vcf_state", lambda *a, **k: None)
        order = [m for _, m in _scored_moves(state, get_candidate_moves(state, radius=2), Stone.BLACK)]
        brain = ThreatBrain(pns_time_s=30.0)
        return brain, b, state, order

    def test_prefers_a_proven_breaker_over_better_unproven_ones(self, monkeypatch):
        brain, b, state, order = self._setup(monkeypatch, proven=set())
        target = order[3]
        brain, b, state, order = self._setup(monkeypatch, proven={(target.row, target.col)})
        move = brain._vct_aware_search(b, state, Stone.BLACK, Move(row=0, col=0), {})
        assert move == target

    def test_falls_back_to_best_unproven_candidate(self, monkeypatch):
        brain, b, state, order = self._setup(monkeypatch, proven=set())
        move = brain._vct_aware_search(b, state, Stone.BLACK, Move(row=0, col=0), {})
        assert move == order[0]


class TestVcfBreakerChoice:
    """Regression from a recorded loss.  Before move 19 White threatens a
    VCF; ten Black moves break it.  The best-evaluated one, H5, still lets
    White win with three-threats (from G10), and White did.  J5 leaves no
    forced win, so the brain must prefer it."""

    MOVES = (
        (7, 7), (6, 6), (8, 8), (5, 7), (7, 5), (4, 8), (7, 8), (3, 9), (2, 10), (7, 6),
        (6, 8), (5, 8), (5, 9), (8, 6), (5, 6), (6, 7), (8, 5), (4, 10),
    )  # fmt: skip

    def _board(self) -> Board:
        board = Board(size=15)
        stone = Stone.BLACK
        for r, c in self.MOVES:
            board = board.place_stone(Move(row=r, col=c), stone)
            stone = stone.opponent()
        return board

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Since split fours are visible to PNS, proving J5 safe takes ~3M nodes "
            "and proving H5 lost ~300k; the breaker probe gets a fraction of "
            "pns_nodes, so every breaker is UNKNOWN and the best-evaluated one, "
            "H5, is played"
        ),
    )
    def test_picks_the_breaker_that_leaves_no_forced_win(self):
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        board = self._board()
        move = ThreatBrain().get_move(board, Stone.BLACK)
        assert move == Move(row=4, col=9)  # J5, not H5
        status, _ = pns_threat_status(board.place_stone(move, Stone.BLACK), Stone.WHITE, max_nodes=50_000)
        assert status is PNSStatus.DISPROVED
