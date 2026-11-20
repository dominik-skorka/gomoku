"""Claimed PNS wins must be converted when played out.

"PNS found winning move" promises a forced win.  These positions, from
self-play games of the threat brain, are ones where it claimed one
with PNS (pns_win; a later, stronger VCF may claim it first, which is
just as binding).  Each is played out against a defender brain to the end:
the attacker must complete five, and every attacker move on the way must
be a win (a claimed win or the five itself) or a forced block of the
defender's five (a counter-four the proof allows for).  A claim that
does not hold to the end is either an unsound proof or a win the brain
failed to find again; gomoku review reports the same thing in games
as unconverted_win.

Settings come from the repository's gomoku.toml, as in real play.
"""

from pathlib import Path

import pytest
from gomoku.board import Board
from gomoku.diagnostics import MoveDiagnostics, set_move_diagnostics
from gomoku.models import Move, Stone
from gomoku.rules import check_winner
from gomoku.settings import GomokuSettings
from gomoku_brain import BRAINS, resolve_brain
from gomoku_brain.threat_detect import _find_winning_move

_CONFIG = Path(__file__).resolve().parents[3] / "gomoku.toml"
_CLAIMS = {"pns_win", "vcf_win", "vct_win"}
_WIN_PATHS = _CLAIMS | {"immediate_win", "instant_win"}
_MAX_MOVES = 40  # per side; the proofs here finish in far fewer

# Move lists in move-name notation (column letter, row number), Black first;
# the side to move after them claims the win.
POSITIONS = [
    pytest.param(
        "F5 K10 K9 E5 H8 F11 K6 F10 J9 F12 F9 E10 J10 G10 H10 E11 G13 I9 E12 F13 F14 H11 G11",
        id="game6-move24",
    ),
    pytest.param(
        "K8 I11 F5 J9 E7 E9 H9 H10 F6 J10 G5 D8 F4 F3 D6 F8 J13 G9 J12",
        id="game9-move20",
    ),
    pytest.param("I9 I8 G10 G5 G9 F7 J11 F6 H9 F9 F10 E5 F8 E7 H4", id="game14-move16"),
    pytest.param(
        "F6 J9 E10 K6 K8 H5 G9 I4 G6 J5 L7 K5 G5 I5 L5 H3 G2 I6 I3 H4 H6 I7 I8 G7 H8 H2 H1 J8 J7",
        id="game28-move30",
    ),
    pytest.param("F8 J8 K11 E5 K9 G11 I10 K8 J10 L8 I8 I9 H10 K10 I11 G9", id="game30-move17"),
    pytest.param(
        "G9 K10 H8 K6 K5 H9 F5 G8 G7 F7 E6 I10 J11 J10 H10 I11 "
        "G4 H3 I7 J6 F6 I9 I8 I12 I13 J9 K9 J8 J7 K7 L6 L10 "
        "M10 H11 K8 I6 E5 D4 F10 E11 D7 C8 L9 N11 G11 G6 H6 H12 "
        "F13",
        id="game2-move50",
    ),
    pytest.param("H11 I10 G10 I11 I8 K10 H6 I12 I9 H13 J11", id="game20-move12"),
    pytest.param(
        "I10 E8 F9 H10 F5 E7 J10 F8 I11 G8 H8 C8 D8 E9 E6 D10 G7",
        id="game33-move18",
    ),
    pytest.param(
        "H9 G9 E6 F5 J7 H8 E8 H10 F7 G8 E7 E5 G7 H7 E9 E10 F9 D7",
        id="game40-move19",
    ),
    pytest.param("G5 I11 H6 E9 I9 F11 K8 F10 F4 I7 E3 D2 G11 G9 H8", id="game46-move16"),
]


def _brain(name: str):
    brain_type, settings = resolve_brain(name, GomokuSettings().brains)
    spec = BRAINS[brain_type]
    return spec.brain_cls(**spec.settings_cls(**settings).model_dump())


def _move(token: str) -> Move:
    return Move(row=int(token[1:]) - 1, col=ord(token[0]) - ord("A"))


def _think(brain, board: Board, stone: Stone) -> tuple[Move, str | None]:
    diag = MoveDiagnostics(move_number=1, stone=stone)
    set_move_diagnostics(diag)
    try:
        return brain.get_move(board, stone), diag.decision_path
    finally:
        set_move_diagnostics(None)


@pytest.fixture(scope="module")
def brains():
    mp = pytest.MonkeyPatch()
    mp.setenv("GOMOKU_CONFIG", str(_CONFIG))
    try:
        yield {name: _brain(name) for name in ("threat", "alphabeta")}
    finally:
        mp.undo()


@pytest.mark.parametrize("defender", ["alphabeta", "threat"])
@pytest.mark.parametrize("moves", POSITIONS)
def test_claimed_pns_win_is_converted(brains, moves, defender):
    board = Board(size=15)
    stone = Stone.BLACK
    for token in moves.split():
        board = board.place_stone(_move(token), stone)
        stone = stone.opponent()
    attacker = stone

    move, path = _think(brains["threat"], board, attacker)
    # Found as pns_win in self-play; a stronger VCF may now claim it first.
    assert path in _CLAIMS, f"the position no longer starts with a claimed win ({path})"

    trail = []
    for _ in range(2 * _MAX_MOVES):
        if stone is attacker:
            if trail:  # the first move was thought above
                forced = _find_winning_move(board, attacker.opponent())
                move, path = _think(brains["threat"], board, attacker)
                assert path in _WIN_PATHS or move == forced, f"claim dropped after {' '.join(trail)}: {move} ({path})"
        else:
            move, _ = _think(brains[defender], board, stone)
        trail.append(str(move))
        board = board.place_stone(move, stone)
        if check_winner(board, move) is not None:
            assert stone is attacker, f"the defender completed five: {' '.join(trail)}"
            return
        stone = stone.opponent()
    pytest.fail(f"no five within {_MAX_MOVES} moves: {' '.join(trail)}")
