"""Bounded, two-generation transposition table."""

import pytest
from gomoku_brain.zobrist import BoundType, TranspositionTable


def _store(tt: TranspositionTable, key: int, depth: int = 1, score: float = 0.0) -> None:
    tt.store(key, depth, score, BoundType.EXACT, None)


def test_basic_store_probe():
    tt = TranspositionTable()
    _store(tt, 7, depth=3, score=1.5)
    entry = tt.probe(7)
    assert entry is not None
    assert (entry.depth, entry.score) == (3, 1.5)
    assert tt.probe(8) is None
    assert (tt.hits, tt.probes) == (1, 2)


def test_depth_preferred_replacement():
    tt = TranspositionTable()
    _store(tt, 1, depth=5, score=1.0)
    _store(tt, 1, depth=2, score=9.0)  # shallower: ignored
    assert tt.probe(1).score == 1.0
    _store(tt, 1, depth=5, score=2.0)  # equal depth: replaces
    assert tt.probe(1).score == 2.0


def test_memory_is_bounded():
    tt = TranspositionTable(max_entries=100)
    for k in range(10_000):
        _store(tt, k)
    assert len(tt._current) <= 100
    assert len(tt._old) <= 100
    assert len(tt) <= 200


def test_old_generation_survives_one_rollover_and_is_promoted():
    tt = TranspositionTable(max_entries=10)
    for k in range(10):
        _store(tt, k)  # fills current
    _store(tt, 100)  # rollover: 0..9 become old
    assert tt.probe(3) is not None  # still reachable, promoted to current
    for k in range(200, 209):
        _store(tt, k)  # fills current again -> next rollover drops old
    _store(tt, 300)
    assert tt.probe(3) is not None  # promoted entry survived
    assert tt.probe(4) is None  # never touched: aged out


def test_len_does_not_double_count_promoted_entries():
    tt = TranspositionTable(max_entries=2)
    _store(tt, 1)
    _store(tt, 2)
    _store(tt, 3)  # rollover
    tt.probe(1)  # promote
    assert len(tt) == 3


def test_invalid_capacity():
    with pytest.raises(ValueError):
        TranspositionTable(max_entries=0)
