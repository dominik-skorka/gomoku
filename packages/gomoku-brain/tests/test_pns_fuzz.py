"""Fuzz PNS soundness on small, nearly full boards.

Every ``PROVED`` claim must be a real win: the claimed move wins against
every legal reply, checked by exhaustive search.  Small boards with a few
empty cells keep the exhaustive search cheap while still producing forced
blocks, counter-fours and quiet endgames, where threat-space search can
go wrong.  Before the fix in ``_expand`` (a node whose defender had no
tactical reply counted as proved), a third of the proofs on such boards
were false.

``DISPROVED`` is not checked against the exhaustive value: it means no
forced win *by threats*, and a quiet win may still exist.
"""

import random
from functools import cache

import pytest
from gomoku.board import Board
from gomoku.models import Stone
from gomoku_brain import evaluate, threat_detect
from gomoku_brain.pns import PNSStatus, pns_threat_status

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _five_at(cells: tuple[int, ...], size: int, row: int, col: int, stone: int) -> bool:
    """Does the stone at (row, col) complete five or more (freestyle)?"""
    for dr, dc in _DIRS:
        run = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size and cells[r * size + c] == stone:
                run += 1
                r += sign * dr
                c += sign * dc
        if run >= 5:
            return True
    return False


@cache
def _value(cells: tuple[int, ...], size: int, stone: int) -> int:
    """Exhaustive value for *stone* to move: 1 win, 0 draw, -1 loss."""
    best = -1
    has_move = False
    for i, cell in enumerate(cells):
        if cell:
            continue
        has_move = True
        after = (*cells[:i], stone, *cells[i + 1 :])
        if _five_at(after, size, i // size, i % size, stone):
            return 1
        best = max(best, -_value(after, size, 3 - stone))
        if best == 1:
            return 1
    return best if has_move else 0


def _positions(seed: int, count: int):
    """Random 5×5 / 6×6 positions with 3–8 empty cells and no five yet."""
    rng = random.Random(seed)
    while count:
        size = rng.choice((5, 6))
        empties = rng.randint(3, 8 if size == 5 else 7)
        stones = size * size - empties
        black = (stones + 1) // 2
        cells = [1] * black + [2] * (stones - black) + [0] * empties
        rng.shuffle(cells)
        cells = tuple(cells)
        if any(cells[i] and _five_at(cells, size, i // size, i % size, cells[i]) for i in range(len(cells))):
            continue
        to_move = 1 if black == stones - black else 2
        count -= 1
        yield size, cells, to_move


@pytest.mark.parametrize("use_jit", [True, False])
def test_every_pns_proof_is_a_real_win(use_jit, monkeypatch):
    monkeypatch.setattr(evaluate, "_use_jit", use_jit)
    monkeypatch.setattr(threat_detect, "_SCAN_KIT", None)
    proofs = 0
    false_proofs = []
    for size, cells, to_move in _positions(seed=7, count=200):
        board = Board(size=size, grid=[[Stone(cells[r * size + c]) for c in range(size)] for r in range(size)])
        status, move = pns_threat_status(board, Stone(to_move), max_nodes=20_000, max_time_s=10.0)
        if status is not PNSStatus.PROVED:
            continue
        proofs += 1
        i = move.row * size + move.col
        after = (*cells[:i], to_move, *cells[i + 1 :])
        wins = _five_at(after, size, move.row, move.col, to_move) or _value(after, size, 3 - to_move) == -1
        if not wins:
            rows = ["".join(".XO"[cells[r * size + c]] for c in range(size)) for r in range(size)]
            false_proofs.append((rows, "XO"[to_move - 1], (move.row, move.col)))
    assert proofs >= 30  # the sample must exercise proofs, not only disproofs
    assert not false_proofs, f"{len(false_proofs)} false proofs of {proofs}, e.g. {false_proofs[:3]}"
