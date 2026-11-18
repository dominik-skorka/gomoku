"""Split fours: a move that leaves XXX_X or XX_XX threatens five like a straight four.

The four-threat detectors used to count only the unbroken run through the
move, so these threats were invisible to VCF, PNS and alpha-beta, as
attacks and as the defender's counter-fours.  PNS then "proved" wins a
split counter-four refutes.
"""

from __future__ import annotations

import numpy as np
import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.evaluate_jit import get_dirs, has_any_four_threat_grid, is_four_threat, makes_four_threat
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


def test_threat_search_sees_split_fours_but_alpha_beta_heuristics_do_not():
    """Quiescence and move ordering stay on unbroken runs (cost); the
    threat searches' four test and its any-four gate see split fours."""
    board = _board([(7, 4), (7, 5), (7, 8)], FAR_WHITE)
    state = SearchState.from_board(board, Stone.BLACK)
    grid, dirs = state.grid, get_dirs()
    assert makes_four_threat(grid, 15, 7, 6, int(Stone.BLACK), dirs)
    assert not makes_four_threat(grid, 15, 6, 6, int(Stone.BLACK), dirs)
    assert not is_four_threat(grid, 15, 7, 6, int(Stone.BLACK), dirs)
    candidates = np.array([[7, 6]], dtype=np.int32)
    assert has_any_four_threat_grid(grid, 15, candidates, int(Stone.BLACK), dirs)


def test_blocked_straight_four_is_not_a_threat():
    # OXXXX_ with the far end blocked too: four in a row, but no five possible.
    board = _board([(7, 1), (7, 2), (7, 3)], [(7, 0), (7, 5), (14, 0)])
    grid = SearchState.from_board(board, Stone.BLACK).grid
    assert not makes_four_threat(grid, 15, 7, 4, int(Stone.BLACK), get_dirs())
    assert is_four_threat(grid, 15, 7, 4, int(Stone.BLACK), get_dirs())  # the heuristic counts the run


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


# ---------------------------------------------------------------------------
# Split threes: XX_X with open ends threatens a live four like XXX does
# ---------------------------------------------------------------------------

# Cell G8 = (7, 6).  Each case puts two shapes on the row and the
# anti-diagonal through G8; White's stones are in far corners.


def _threes_at_g8(black, white=None):
    from gomoku_brain.threat_detect import _find_three_threats, _find_three_threats_python

    board = _board(black, white or [(0, 0), (0, 14), (14, 0), (14, 14)][: len(black) - 1])

    def at_g8(threats):
        return sorted(tuple(sorted((d.row, d.col) for d in defs)) for a, defs in threats if (a.row, a.col) == (7, 6))

    jit = at_g8(_find_three_threats(SearchState.from_board(board, Stone.BLACK), Stone.BLACK))
    return board, jit, at_g8(_find_three_threats_python(board, Stone.BLACK))


@pytest.mark.parametrize(
    ("label", "black", "defences"),
    [
        # XX + G8 on both lines: two straight threes, defended at their ends.
        ("adjacent pairs", [(7, 4), (7, 5), (5, 8), (6, 7)], [((4, 9), (8, 5)), ((7, 3), (7, 7))]),
        # XX _ G8 on both lines: two split threes, defended at the gap or an end.
        ("pairs with a gap", [(7, 3), (7, 4), (4, 9), (5, 8)], [((3, 10), (6, 7), (8, 5)), ((7, 2), (7, 5), (7, 7))]),
        # X G8 X on both lines.
        ("G8 fills the middle", [(7, 5), (7, 7), (6, 7), (8, 5)], [((5, 8), (9, 4)), ((7, 4), (7, 8))]),
    ],
)
def test_double_threes_are_found_and_are_fork_cells(label, black, defences):
    from gomoku_brain.evaluate import find_fork_cells

    board, jit, python = _threes_at_g8(black)
    assert jit == python == defences, label
    forks = find_fork_cells(SearchState.from_board(board, Stone.BLACK), Stone.BLACK)
    assert Move(row=7, col=6) in forks, label


def test_three_that_cannot_become_a_live_four_is_no_threat():
    # O _ X X G8 _ O: every four made from it has a blocked end.
    _, jit, python = _threes_at_g8([(7, 4), (7, 5)], [(7, 2), (7, 8)])
    assert jit == python == []
