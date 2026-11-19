"""AlphaBetaBrain reports what its root VCF pre-check actually decided."""

from __future__ import annotations

import pytest
from gomoku.board import Board
from gomoku.diagnostics import MoveDiagnostics, set_move_diagnostics
from gomoku.models import Move, Stone
from gomoku_brain.alphabeta import AlphaBetaBrain

# Black's open three on row 7 (cols 4-6); White's stones are in the corners.
OPEN_THREE = [(7, 4), (0, 0), (7, 5), (0, 14), (7, 6)]


def _decide(moves: list[tuple[int, int]]) -> tuple[Move, str]:
    board = Board(size=15)
    stone = Stone.BLACK
    for row, col in moves:
        board = board.place_stone(Move(row=row, col=col), stone)
        stone = stone.opponent()
    diag = MoveDiagnostics(move_number=len(moves) + 1, stone=stone)
    set_move_diagnostics(diag)
    try:
        move = AlphaBetaBrain(depth=2, vcf_depth=20).get_move(board, stone)
    finally:
        set_move_diagnostics(None)
    return move, diag.decision_path


@pytest.mark.parametrize(
    ("moves", "path", "cells"),
    [
        # Black to move turns the open three into an open four.
        ([*OPEN_THREE, (14, 0)], "vcf_win", {(7, 3), (7, 7)}),
        # White to move must break Black's VCF: a defence, not a win.
        (OPEN_THREE, "vcf_breaker", {(7, 2), (7, 3), (7, 7), (7, 8)}),
        # Black can complete five.
        ([(7, 3), (0, 0), (7, 4), (0, 2), (7, 5), (0, 4), (7, 6), (0, 6)], "instant_win", {(7, 2), (7, 7)}),
    ],
)
def test_root_vcf_decision_path_is_kept(moves, path, cells):
    move, decision = _decide(moves)
    assert decision == path
    assert (move.row, move.col) in cells


class TestVcfBreakerProbe:
    """From `gomoku review` of the alpha-beta brain: White must break
    Black's VCF.  The best-evaluated breaker, I3, still lets Black win with
    three-threats; probing the breakers with PNS finds one that holds."""

    MOVES = (
        (6, 10), (5, 6), (7, 8), (9, 10), (4, 7), (4, 8), (8, 10), (6, 6), (5, 7),
        (4, 6), (3, 6), (5, 8), (3, 7), (6, 7), (7, 6), (3, 8), (6, 8),
    )  # fmt: skip

    def _board(self) -> Board:
        board = Board(size=15)
        stone = Stone.BLACK
        for r, c in self.MOVES:
            board = board.place_stone(Move(row=r, col=c), stone)
            stone = stone.opponent()
        return board

    def _move(self, nodes: int) -> Move:
        # A generous time cap keeps the test independent of machine speed.
        brain = AlphaBetaBrain(depth=4, vcf_depth=20, vcf_breaker_pns_nodes=nodes, vcf_breaker_time_s=60.0)
        return brain.get_move(self._board(), Stone.WHITE)

    def test_without_the_probe_the_evaluation_picks_a_losing_breaker(self):
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        move = self._move(0)
        assert move == Move(row=2, col=8)  # I3
        status, _ = pns_threat_status(self._board().place_stone(move, Stone.WHITE), Stone.BLACK, max_nodes=200_000)
        assert status is PNSStatus.PROVED

    def test_the_probe_picks_a_breaker_that_holds(self):
        from gomoku_brain.pns import PNSStatus, pns_threat_status

        move = self._move(10_000)
        status, _ = pns_threat_status(self._board().place_stone(move, Stone.WHITE), Stone.BLACK, max_nodes=200_000)
        assert status is PNSStatus.DISPROVED
