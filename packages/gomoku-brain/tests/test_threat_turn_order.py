"""Tempo regressions in the continuous-threat searches."""

from dataclasses import replace

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner
from gomoku_brain import evaluate, threat_detect, threat_search
from gomoku_brain.threat_space import DEFAULT_THREAT_SPACE


def _cell(name):
    return Move(row=int(name[1:]) - 1, col=ord(name[0]) - ord("A"))


def _board(black, white):
    board = Board(size=9)
    for names, stone in [(black, Stone.BLACK), (white, Stone.WHITE)]:
        for name in names.split():
            board = board.place_stone(_cell(name), stone)
    return board


@pytest.fixture(params=[False, True], ids=["python", "jit"])
def backend(request, monkeypatch):
    monkeypatch.setattr(evaluate, "_use_jit", request.param)
    monkeypatch.setattr(threat_detect, "_SCAN_KIT", None)


@pytest.mark.parametrize("blocking_four", [False, True])
def test_vcf_counter_four_tempo(backend, blocking_four):
    white = "A5 F2 F3 F4 A1" if blocking_four else "A5 D8 E8 F8 A1"
    board = _board("B5 C5 D5 E3 E4", white)
    result = threat_search._vcf(board, Stone.BLACK, depth=20)
    after = board.place_stone(_cell("E5"), Stone.BLACK).place_stone(_cell("F5"), Stone.WHITE)
    if blocking_four:
        # F5 defends and threatens a white five at either end.
        assert result is None
        winner, ends = Stone.WHITE, ("F1", "F6")
    else:
        assert result == _cell("E5")
        after = after.place_stone(_cell("E6"), Stone.BLACK)
        winner, ends = Stone.BLACK, ("E2", "E7")
    for name in ends:
        move = _cell(name)
        assert check_winner(after.place_stone(move, winner), move)


def _check_turns(monkeypatch, name):
    search = getattr(threat_search, name)

    def checked(state, stone, *args, **kwargs):
        if state.ply:
            assert state.last_move()[2] == stone.opponent()
        return search(state, stone, *args, **kwargs)

    monkeypatch.setattr(threat_search, name, checked)


def test_vct_search_resumes_on_attackers_turn(backend, monkeypatch):
    _check_turns(monkeypatch, "_vct_state")
    board = _board("D5 E5", "A1 B1")
    space = replace(DEFAULT_THREAT_SPACE, split_counter_fours=False)
    assert threat_search._vct(board, Stone.BLACK, depth=4, space=space) is None


@pytest.mark.parametrize("black", ["B5 C5", "B5 C5 E5"])
def test_breaker_probes_follow_legal_turns(backend, monkeypatch, black):
    _check_turns(monkeypatch, "_vcf_state")
    board = _board(black, "A5 H8")
    move = _cell("D5")
    assert threat_search.classify_vcf_breakers(board, [move], Stone.BLACK, 4) == ([move], [move])
