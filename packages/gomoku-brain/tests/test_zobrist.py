"""Tests for Zobrist hashing and the transposition table.

These tests cover the two most collision-sensitive parts of the search:

* :func:`hash_board` vs :func:`incremental_hash` must stay in lock-step.  A
  divergence would silently poison every transposition-table lookup.
* :class:`TranspositionTable` must honour its replacement policy and bound
  semantics, otherwise alpha-beta returns the wrong score.
"""

from __future__ import annotations

import random

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.zobrist import (
    _TABLE,
    SIDE_KEY,
    BoundType,
    TranspositionTable,
    TTEntry,
    _stone_index,
    hash_board,
    incremental_hash,
)

# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------


class TestHashBoard:
    def test_empty_board_hash_is_side_dependent(self):
        b = Board(size=15)
        assert hash_board(b, Stone.BLACK) == 0
        # White to move flips the side key, giving a distinct non-zero hash.
        assert hash_board(b, Stone.WHITE) == SIDE_KEY
        assert hash_board(b, Stone.BLACK) != hash_board(b, Stone.WHITE)

    def test_is_deterministic(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        h1 = hash_board(b, Stone.WHITE)
        h2 = hash_board(b, Stone.WHITE)
        assert h1 == h2

    def test_order_independent(self):
        """``hash_board`` XORs stones into the hash, so the placement order
        must not matter — only the final position does."""
        b1 = Board(size=9)
        b1 = b1.place_stone(Move(row=4, col=4), Stone.BLACK)
        b1 = b1.place_stone(Move(row=3, col=3), Stone.WHITE)
        b1 = b1.place_stone(Move(row=5, col=5), Stone.BLACK)

        b2 = Board(size=9)
        b2 = b2.place_stone(Move(row=5, col=5), Stone.BLACK)
        b2 = b2.place_stone(Move(row=3, col=3), Stone.WHITE)
        b2 = b2.place_stone(Move(row=4, col=4), Stone.BLACK)

        assert hash_board(b1, Stone.WHITE) == hash_board(b2, Stone.WHITE)

    def test_different_stones_give_different_hashes(self):
        b = Board(size=9)
        b_black = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b_white = b.place_stone(Move(row=4, col=4), Stone.WHITE)
        assert hash_board(b_black, Stone.WHITE) != hash_board(b_white, Stone.BLACK)


class TestStoneIndex:
    def test_maps_stones_to_expected_indices(self):
        assert _stone_index(Stone.BLACK) == 0
        assert _stone_index(Stone.WHITE) == 1


class TestIncrementalHash:
    def test_matches_full_rehash_after_move(self):
        """``incremental_hash(h, r, c, stone)`` must equal
        ``hash_board(board_after, side_after)``."""
        b = Board(size=9)
        h = hash_board(b, Stone.BLACK)

        updated = incremental_hash(h, 4, 4, Stone.BLACK)
        b_after = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        assert updated == hash_board(b_after, Stone.WHITE)

    def test_matches_full_rehash_over_random_walk(self):
        """At every ply of a random 30-move walk the incremental hash
        must equal :func:`hash_board` on the equivalent board."""
        rng = random.Random(0xC0FFEE)
        b = Board(size=11)
        h = hash_board(b, Stone.BLACK)
        side = Stone.BLACK
        for _ in range(30):
            while True:
                r, c = rng.randrange(11), rng.randrange(11)
                if b.at(r, c) is Stone.EMPTY:
                    break
            h = incremental_hash(h, r, c, side)
            b = b.place_stone(Move(row=r, col=c), side)
            side = side.opponent()
            assert h == hash_board(b, side)

    def test_is_its_own_inverse(self):
        """XOR is involutive — applying the same move twice returns the
        original hash.  Used during ``SearchState.unmake``."""
        b = Board(size=9)
        h = hash_board(b, Stone.BLACK)
        h1 = incremental_hash(h, 4, 4, Stone.BLACK)
        h2 = incremental_hash(h1, 4, 4, Stone.BLACK)
        assert h2 == h

    def test_side_key_flips_every_call(self):
        """Each call XORs SIDE_KEY; two calls restore the side parity."""
        h = 0
        h1 = incremental_hash(h, 4, 4, Stone.BLACK)
        h2 = incremental_hash(h1, 4, 5, Stone.WHITE)
        # SIDE_KEY was flipped twice → back to black-to-move parity.
        expected = _TABLE[4][4][0] ^ _TABLE[4][5][1]
        assert h2 == expected


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------


class TestTranspositionTable:
    def test_empty_probe_returns_none(self):
        tt = TranspositionTable()
        assert tt.probe(12345) is None
        assert len(tt) == 0

    def test_store_then_probe(self):
        tt = TranspositionTable()
        move = Move(row=7, col=7)
        tt.store(42, depth=3, score=1.5, flag=BoundType.EXACT, best_move=move)
        entry = tt.probe(42)
        assert entry is not None
        assert entry.key == 42
        assert entry.depth == 3
        assert entry.score == 1.5
        assert entry.flag is BoundType.EXACT
        assert entry.best_move == move

    def test_depth_preferred_replacement_keeps_deeper(self):
        """Storing a shallower entry must not overwrite a deeper one."""
        tt = TranspositionTable()
        tt.store(1, depth=5, score=10.0, flag=BoundType.EXACT, best_move=None)
        tt.store(1, depth=2, score=-99.0, flag=BoundType.LOWER, best_move=None)
        entry = tt.probe(1)
        assert entry is not None
        assert entry.depth == 5
        assert entry.score == 10.0

    def test_depth_preferred_replacement_accepts_equal_or_deeper(self):
        """A same-or-deeper entry should overwrite the existing one."""
        tt = TranspositionTable()
        tt.store(1, depth=3, score=1.0, flag=BoundType.EXACT, best_move=None)
        tt.store(1, depth=3, score=2.0, flag=BoundType.LOWER, best_move=None)
        assert tt.probe(1).score == 2.0

        tt.store(1, depth=7, score=3.0, flag=BoundType.UPPER, best_move=None)
        assert tt.probe(1).score == 3.0
        assert tt.probe(1).flag is BoundType.UPPER

    def test_distinct_keys_do_not_collide(self):
        tt = TranspositionTable()
        tt.store(1, depth=1, score=1.0, flag=BoundType.EXACT, best_move=None)
        tt.store(2, depth=1, score=2.0, flag=BoundType.EXACT, best_move=None)
        assert tt.probe(1).score == 1.0
        assert tt.probe(2).score == 2.0
        assert len(tt) == 2

    def test_len_tracks_stored_entries(self):
        tt = TranspositionTable()
        assert len(tt) == 0
        tt.store(1, 1, 0.0, BoundType.EXACT, None)
        tt.store(2, 1, 0.0, BoundType.EXACT, None)
        tt.store(3, 1, 0.0, BoundType.EXACT, None)
        assert len(tt) == 3
        # Re-storing the same key (with equal depth) doesn't grow the table.
        tt.store(1, 1, 0.0, BoundType.EXACT, None)
        assert len(tt) == 3

    def test_hit_counters_start_zero(self):
        tt = TranspositionTable()
        assert tt.hits == 0
        assert tt.probes == 0

    def test_probe_counts_hits_and_probes(self):
        """``probe`` must increment ``probes`` always and ``hits`` only on match."""
        tt = TranspositionTable()
        tt.store(1, 1, 0.0, BoundType.EXACT, None)
        tt.probe(1)  # hit
        tt.probe(2)  # miss
        tt.probe(1)  # hit
        assert tt.probes == 3
        assert tt.hits == 2

    def test_store_does_not_affect_counters(self):
        tt = TranspositionTable()
        tt.store(1, 1, 0.0, BoundType.EXACT, None)
        tt.store(2, 1, 0.0, BoundType.EXACT, None)
        assert tt.probes == 0 and tt.hits == 0


class TestBoundType:
    @pytest.mark.parametrize(
        "flag",
        [BoundType.EXACT, BoundType.LOWER, BoundType.UPPER],
    )
    def test_round_trips_through_tt(self, flag):
        tt = TranspositionTable()
        tt.store(123, depth=1, score=0.0, flag=flag, best_move=None)
        assert tt.probe(123).flag is flag

    def test_values_are_distinct(self):
        values = {BoundType.EXACT, BoundType.LOWER, BoundType.UPPER}
        assert len(values) == 3


class TestTTEntry:
    def test_is_a_namedtuple(self):
        """Downstream code unpacks TTEntry by position — guard that
        signature."""
        entry = TTEntry(1, 2, 3.0, BoundType.EXACT, None)
        key, depth, score, flag, best_move = entry
        assert (key, depth, score, flag, best_move) == (1, 2, 3.0, BoundType.EXACT, None)
