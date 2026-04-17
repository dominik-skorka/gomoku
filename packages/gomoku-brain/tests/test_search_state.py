"""Tests for SearchState — mutable bitboard with make/unmake."""

from __future__ import annotations

import random

import numpy as np
import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.search_state import SearchState
from gomoku_brain.zobrist import hash_board

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equivalent_board(state: SearchState, moves: list[tuple[int, int, Stone]]) -> Board:
    """Build a Board by replaying *moves* from empty."""
    b = Board(size=state.size)
    for r, c, stone in moves:
        b = b.place_stone(Move(row=r, col=c), stone)
    return b


def _side_after(moves: list[tuple[int, int, Stone]], initial: Stone) -> Stone:
    """Return the side to move after *moves* applied, given *initial* side."""
    side = initial
    for _ in moves:
        side = side.opponent()
    return side


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_empty_board(self):
        b = Board(size=15)
        s = SearchState.from_board(b, Stone.BLACK)
        assert s.size == 15
        assert s.ply == 0
        assert s.occ_count == 0
        assert (s.black == 0).all()
        assert (s.white == 0).all()
        assert s.zobrist == hash_board(b, Stone.BLACK)

    def test_from_board_with_stones(self):
        b = Board(size=15)
        b = b.place_stone(Move(row=7, col=7), Stone.BLACK)
        b = b.place_stone(Move(row=3, col=10), Stone.WHITE)
        b = b.place_stone(Move(row=14, col=0), Stone.BLACK)

        s = SearchState.from_board(b, Stone.WHITE)
        assert s.size == 15
        assert s.ply == 0
        assert s.occ_count == 3
        assert s.at(7, 7) is Stone.BLACK
        assert s.at(3, 10) is Stone.WHITE
        assert s.at(14, 0) is Stone.BLACK
        assert s.at(0, 0) is Stone.EMPTY
        assert s.zobrist == hash_board(b, Stone.WHITE)

    def test_from_bitboards(self):
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        ref = SearchState.from_board(b, Stone.WHITE)

        alt = SearchState.from_bitboards(ref.black, ref.white, 9, Stone.WHITE)
        assert alt.zobrist == ref.zobrist
        assert alt.occ_count == ref.occ_count
        # Bitboards are copied — verify independence
        alt.make(0, 0, Stone.WHITE)
        assert ref.black[0] == 0 and ref.white[0] == 0

    def test_side_to_move_in_zobrist(self):
        b = Board(size=15)
        s_black = SearchState.from_board(b, Stone.BLACK)
        s_white = SearchState.from_board(b, Stone.WHITE)
        assert s_black.zobrist != s_white.zobrist


# ---------------------------------------------------------------------------
# Make / unmake basics
# ---------------------------------------------------------------------------


class TestMakeUnmake:
    def test_make_updates_bitboards(self):
        s = SearchState.from_board(Board(size=15), Stone.BLACK)
        s.make(7, 7, Stone.BLACK)
        assert s.at(7, 7) is Stone.BLACK
        assert s.occ_count == 1
        assert s.ply == 1

        s.make(3, 3, Stone.WHITE)
        assert s.at(3, 3) is Stone.WHITE
        assert s.occ_count == 2
        assert s.ply == 2

    def test_unmake_restores_bitboards(self):
        s = SearchState.from_board(Board(size=15), Stone.BLACK)
        orig_black = s.black.copy()
        orig_white = s.white.copy()

        s.make(7, 7, Stone.BLACK)
        s.make(3, 3, Stone.WHITE)
        s.unmake()
        s.unmake()

        np.testing.assert_array_equal(s.black, orig_black)
        np.testing.assert_array_equal(s.white, orig_white)
        assert s.ply == 0
        assert s.occ_count == 0

    def test_unmake_restores_zobrist(self):
        s = SearchState.from_board(Board(size=15), Stone.BLACK)
        orig_hash = s.zobrist

        s.make(7, 7, Stone.BLACK)
        assert s.zobrist != orig_hash
        s.make(3, 3, Stone.WHITE)
        assert s.zobrist != orig_hash

        s.unmake()
        s.unmake()
        assert s.zobrist == orig_hash

    def test_zobrist_flips_side(self):
        b = Board(size=15)
        s = SearchState.from_board(b, Stone.BLACK)  # Black to move
        s.make(7, 7, Stone.BLACK)
        # After Black moves, hash should match a board with that move, White-to-move
        b2 = b.place_stone(Move(row=7, col=7), Stone.BLACK)
        assert s.zobrist == hash_board(b2, Stone.WHITE)

    def test_last_move(self):
        s = SearchState.from_board(Board(size=15), Stone.BLACK)
        assert s.last_move() is None

        s.make(7, 7, Stone.BLACK)
        assert s.last_move() == (7, 7, Stone.BLACK)

        s.make(5, 8, Stone.WHITE)
        assert s.last_move() == (5, 8, Stone.WHITE)

        s.unmake()
        assert s.last_move() == (7, 7, Stone.BLACK)

    def test_is_empty(self):
        s = SearchState.from_board(Board(size=15), Stone.BLACK)
        assert s.is_empty(7, 7)
        s.make(7, 7, Stone.BLACK)
        assert not s.is_empty(7, 7)
        assert s.is_empty(7, 8)


# ---------------------------------------------------------------------------
# Zobrist parity: SearchState hash equals hash_board on equivalent Board
# ---------------------------------------------------------------------------


class TestZobristParity:
    def test_parity_after_each_ply(self):
        """At every ply along a 30-move sequence, state.zobrist must equal
        hash_board(equivalent_board, side_to_move)."""
        rng = random.Random(0xABC)
        b = Board(size=15)
        s = SearchState.from_board(b, Stone.BLACK)
        moves: list[tuple[int, int, Stone]] = []
        stone = Stone.BLACK

        for _ in range(30):
            while True:
                r, c = rng.randrange(15), rng.randrange(15)
                if s.is_empty(r, c):
                    break
            s.make(r, c, stone)
            moves.append((r, c, stone))
            stone = stone.opponent()

            equiv = _equivalent_board(s, moves)
            side_now = _side_after(moves, Stone.BLACK)
            assert s.zobrist == hash_board(equiv, side_now)

    def test_unmake_all_returns_to_start(self):
        rng = random.Random(0x123)
        b = Board(size=15)
        s = SearchState.from_board(b, Stone.BLACK)
        orig_hash = s.zobrist
        orig_black = s.black.copy()
        orig_white = s.white.copy()

        stone = Stone.BLACK
        for _ in range(50):
            while True:
                r, c = rng.randrange(15), rng.randrange(15)
                if s.is_empty(r, c):
                    break
            s.make(r, c, stone)
            stone = stone.opponent()

        for _ in range(50):
            s.unmake()

        assert s.zobrist == orig_hash
        assert s.ply == 0
        assert s.occ_count == 0
        np.testing.assert_array_equal(s.black, orig_black)
        np.testing.assert_array_equal(s.white, orig_white)


# ---------------------------------------------------------------------------
# Stress: many random make/unmake pairs must leave state identical
# ---------------------------------------------------------------------------


class TestStress:
    @pytest.mark.parametrize("seed", [1, 7, 42, 2026])
    def test_random_make_unmake_pairs(self, seed):
        rng = random.Random(seed)
        b = Board(size=9)
        # Pre-populate so non-trivial state exists
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=3, col=3), Stone.WHITE)
        b = b.place_stone(Move(row=5, col=5), Stone.BLACK)

        s = SearchState.from_board(b, Stone.WHITE)
        orig_hash = s.zobrist
        orig_black = s.black.copy()
        orig_white = s.white.copy()
        orig_occ = s.occ_count

        for _ in range(2000):
            r, c = rng.randrange(9), rng.randrange(9)
            if not s.is_empty(r, c):
                continue
            stone = Stone.BLACK if rng.random() < 0.5 else Stone.WHITE
            s.make(r, c, stone)
            s.unmake()

        assert s.zobrist == orig_hash
        assert s.ply == 0
        assert s.occ_count == orig_occ
        np.testing.assert_array_equal(s.black, orig_black)
        np.testing.assert_array_equal(s.white, orig_white)

    @pytest.mark.parametrize("seed", [3, 11, 99])
    def test_deep_push_pop(self, seed):
        """Push 60 moves, pop all, verify full restoration."""
        rng = random.Random(seed)
        s = SearchState.from_board(Board(size=15), Stone.BLACK)
        orig_hash = s.zobrist

        stone = Stone.BLACK
        pushed = 0
        for _ in range(200):
            if pushed >= 60:
                break
            r, c = rng.randrange(15), rng.randrange(15)
            if not s.is_empty(r, c):
                continue
            s.make(r, c, stone)
            stone = stone.opponent()
            pushed += 1

        assert s.ply == 60
        for _ in range(60):
            s.unmake()

        assert s.zobrist == orig_hash
        assert s.ply == 0
        assert s.occ_count == 0
        assert (s.black == 0).all()
        assert (s.white == 0).all()


# ---------------------------------------------------------------------------
# Bitboard layout parity with existing BitboardState
# ---------------------------------------------------------------------------


class TestBitboardLayoutParity:
    """SearchState.black / .white must have the same layout as BitboardState
    so existing JIT kernels (bitboard_jit.*) can be called on them directly."""

    def test_layout_matches_bitboard_state(self):
        from gomoku_brain.bitboard import BitboardState

        b = Board(size=15)
        b = b.place_stone(Move(row=7, col=7), Stone.BLACK)
        b = b.place_stone(Move(row=8, col=8), Stone.WHITE)

        bb = BitboardState.from_board(b)
        s = SearchState.from_board(b, Stone.BLACK)

        np.testing.assert_array_equal(s.black, bb.black)
        np.testing.assert_array_equal(s.white, bb.white)

    def test_layout_persists_through_make(self):
        from gomoku_brain.bitboard import BitboardState

        b = Board(size=15)
        s = SearchState.from_board(b, Stone.BLACK)
        s.make(7, 7, Stone.BLACK)
        s.make(3, 3, Stone.WHITE)

        b2 = b.place_stone(Move(row=7, col=7), Stone.BLACK)
        b2 = b2.place_stone(Move(row=3, col=3), Stone.WHITE)
        bb = BitboardState.from_board(b2)

        np.testing.assert_array_equal(s.black, bb.black)
        np.testing.assert_array_equal(s.white, bb.white)
