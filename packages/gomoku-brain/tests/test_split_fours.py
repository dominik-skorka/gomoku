"""Split fours: a move that leaves XXX_X or XX_XX threatens five like a straight four.

The four-threat detectors used to count only the unbroken run through the
move, so these threats were invisible to VCF, PNS and alpha-beta, as
attacks and as the defender's counter-fours.  PNS then "proved" wins a
split counter-four refutes.
"""

from __future__ import annotations

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.evaluate_jit import get_dirs, is_four_threat
from gomoku_brain.pns import PNSStatus, pns_threat_status
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import (
    _find_four_threats,
    _find_four_threats_python,
    _has_unblockable_four,
)

FAR_WHITE = [(0, 0), (0, 14), (14, 0)]


def _board(black: list[tuple[int, int]], white: list[tuple[int, int]]) -> Board:
    board = Board(size=15)
    for r, c in black:
        board = board.place_stone(Move(row=r, col=c), Stone.BLACK)
    for r, c in white:
        board = board.place_stone(Move(row=r, col=c), Stone.WHITE)
    return board


def _threats(board: Board) -> tuple[set, set]:
    def cells(threats):
        return {((a.row, a.col), None if d is None else (d.row, d.col)) for a, d in threats}

    jit = cells(_find_four_threats(SearchState.from_board(board, Stone.BLACK), Stone.BLACK))
    return jit, cells(_find_four_threats_python(board, Stone.BLACK))


@pytest.mark.parametrize(
    ("black", "white", "expected"),
    [
        # XX__X: either gap makes a split four; the other gap completes it.
        ([(7, 4), (7, 5), (7, 8)], FAR_WHITE, {((7, 6), (7, 7)), ((7, 7), (7, 6))}),
        # The same with the near end blocked, in a column (the game's E7 E8 _ _ E11).
        ([(6, 4), (7, 4), (10, 4)], [(5, 4), (0, 14), (14, 0)], {((8, 4), (9, 4)), ((9, 4), (8, 4))}),
        # XXX with one end blocked: the straight four, and now also XXX_X.
        ([(6, 4), (7, 4), (8, 4)], [(5, 4), (0, 14), (14, 0)], {((9, 4), (10, 4)), ((10, 4), (9, 4))}),
        # X_X_X on a diagonal: filling a gap leaves one gap that completes five.
        ([(3, 3), (5, 5), (7, 7)], FAR_WHITE, {((4, 4), (6, 6)), ((6, 6), (4, 4))}),
    ],
)
def test_split_fours_are_found_by_both_detectors(black, white, expected):
    jit, python = _threats(_board(black, white))
    assert jit == python == expected


def test_split_and_straight_four_on_one_line_is_unblockable():
    # X_XX_X: playing between the left pair leaves XXXX_X, which completes
    # at both ends of the four.
    board = _board([(7, 3), (7, 5), (7, 6), (7, 8)], FAR_WHITE)
    jit, python = _threats(board)
    assert ((7, 4), None) in jit and ((7, 4), None) in python
    assert _has_unblockable_four(SearchState.from_board(board, Stone.BLACK), Stone.BLACK)


def test_split_four_counts_for_move_ordering():
    board = _board([(7, 4), (7, 5), (7, 8)], FAR_WHITE)
    grid = SearchState.from_board(board, Stone.BLACK).grid
    assert is_four_threat(grid, 15, 7, 6, int(Stone.BLACK), get_dirs())
    assert not is_four_threat(grid, 15, 6, 6, int(Stone.BLACK), get_dirs())


def test_split_counter_four_refutes_a_false_proof():
    """From `gomoku review`: White to move "proved" a win starting F5 at
    any budget, yet after Black's split counter-four E9 (E7 E8 E9 _ E11)
    there was none."""
    moves = [
        (7, 10), (4, 7), (9, 5), (5, 4), (6, 4), (9, 9), (10, 5), (6, 5),
        (8, 5), (7, 5), (7, 4), (7, 6), (8, 7), (6, 3), (10, 4),
    ]  # fmt: skip
    board = Board(size=15)
    stone = Stone.BLACK
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    status, move = pns_threat_status(board, Stone.WHITE, max_nodes=30_000, max_time_s=60.0)
    assert status is not PNSStatus.PROVED, move
