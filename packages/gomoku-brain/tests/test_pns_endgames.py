"""Compare PNS claims with exhaustive legal play on small endgames."""

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner
from gomoku_brain import evaluate, threat_detect
from gomoku_brain.pns import PNSStatus, pns_threat_status


def _move_value(board: Board, move: Move, stone: Stone) -> int:
    """Value for the mover: win (1), draw (0), or loss (-1)."""
    after = board.place_stone(move, stone)
    if check_winner(after, move):
        return 1
    replies = after.get_valid_moves()
    return -max((_move_value(after, reply, stone.opponent()) for reply in replies), default=0)


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(("XXOXO", "XX..O", "OXXXO", "O..OO", "O.XOX"), 0, id="quiet-defence"),
        pytest.param(("OOXOO", "XXXOO", "XOXOO", "OXOX.", "XXXXO"), 0, id="last-cell-block"),
        pytest.param(("XXOXO", "XXO.O", "OXXXO", "O..OO", "OXXOX"), 1, id="winning-endgame"),
    ],
)
def test_proved_move_wins_against_every_reply(rows, expected, use_jit, monkeypatch):
    monkeypatch.setattr(evaluate, "_use_jit", use_jit)
    monkeypatch.setattr(threat_detect, "_SCAN_KIT", None)
    stones = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}
    board = Board(size=5, grid=[[stones[c] for c in row] for row in rows])
    values = {move: _move_value(board, move, Stone.BLACK) for move in board.get_valid_moves()}
    assert max(values.values()) == expected

    status, move = pns_threat_status(board, Stone.BLACK)
    if expected == 1:
        assert status is PNSStatus.PROVED
    if status is PNSStatus.PROVED:
        assert values[move] == 1
    else:
        assert status is PNSStatus.DISPROVED
