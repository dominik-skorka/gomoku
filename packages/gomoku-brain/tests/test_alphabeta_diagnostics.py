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
