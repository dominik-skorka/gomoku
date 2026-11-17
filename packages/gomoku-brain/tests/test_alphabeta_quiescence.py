"""Tests for ``alphabeta_quiescence``.

The quiescence search core was lifted out of ``AlphaBetaBrain._quiesce``
(class method) into a free :func:`quiesce` function in
:mod:`gomoku_brain.alphabeta_quiescence`.  The brain now keeps a thin
wrapper that bundles the bits of instance state quiescence touches —
``qs_vcf_depth`` and the running ``qs_node_count`` counter — into a
:class:`QuiesceCtx` and delegates.

These tests pin three things:

1. The free function is importable, the context dataclass round-trips,
   and ``ctx.nodes`` is incremented per call (so the brain's running
   counter stays accurate).
2. Stand-pat behaviour: on an empty board with no threats, the search
   returns the static evaluation directly.
3. End-to-end: ``AlphaBetaBrain.get_move`` still routes through the
   extracted function and finds the same instant winning move it would
   have found with the in-line implementation.
"""

from __future__ import annotations

import math

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.alphabeta_quiescence import QuiesceCtx, get_threat_moves, quiesce
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.search_state import SearchState


def _state(board: Board, side_to_move: Stone) -> SearchState:
    return SearchState.from_board(board, side_to_move)


class TestQuiesceCtx:
    def test_dataclass_round_trip(self):
        ctx = QuiesceCtx(qs_vcf_depth=4, eval_cache={}, nodes=0)
        assert ctx.qs_vcf_depth == 4
        assert ctx.nodes == 0
        ctx.nodes += 7
        assert ctx.nodes == 7


class TestQuiesceDirect:
    def test_increments_node_counter(self):
        b = Board(size=9)
        state = _state(b, Stone.BLACK)
        ctx = QuiesceCtx(qs_vcf_depth=0, eval_cache={}, nodes=0)
        quiesce(state, alpha=-1e9, beta=1e9, stone=Stone.BLACK, ctx=ctx, qs_depth=2)
        assert ctx.nodes >= 1

    def test_returns_stand_pat_on_quiet_position(self):
        # Empty board — no threats, so quiescence has nothing to extend
        # and must return the static evaluation directly.
        b = Board(size=9)
        state = _state(b, Stone.BLACK)
        ctx = QuiesceCtx(qs_vcf_depth=0, eval_cache={}, nodes=0)
        score = quiesce(state, alpha=-1e9, beta=1e9, stone=Stone.BLACK, ctx=ctx, qs_depth=2)
        expected = evaluate_board(state, Stone.BLACK)
        assert math.isclose(score, expected)

    def test_eval_cache_is_populated(self):
        # The cache is shared across calls — quiescence should write the
        # stand-pat result so the next probe of the same position skips
        # the re-evaluation.
        b = Board(size=9)
        state = _state(b, Stone.BLACK)
        cache: dict[int, float] = {}
        ctx = QuiesceCtx(qs_vcf_depth=0, eval_cache=cache, nodes=0)
        quiesce(state, alpha=-1e9, beta=1e9, stone=Stone.BLACK, ctx=ctx, qs_depth=2)
        assert state.zobrist in cache

    def test_finds_winning_threat_at_depth_one(self):
        # Black has 4-in-a-row on row 4 cols 2-5; placing at (4, 1) or (4, 6)
        # completes 5.  The quiescence threat loop should return ≥ 100_000.
        b = Board(size=9)
        for c in range(2, 6):
            b = b.place_stone(Move(row=4, col=c), Stone.BLACK)
        state = _state(b, Stone.BLACK)
        ctx = QuiesceCtx(qs_vcf_depth=0, eval_cache={}, nodes=0)
        score = quiesce(state, alpha=-1e9, beta=1e9, stone=Stone.BLACK, ctx=ctx, qs_depth=2)
        assert score >= 100_000

    def test_get_threat_moves_finds_block(self):
        # White has 4-in-a-row, Black to move — the only sensible
        # threats are blocking moves at (4, 1) or (4, 6).
        b = Board(size=9)
        for c in range(2, 6):
            b = b.place_stone(Move(row=4, col=c), Stone.WHITE)
        state = _state(b, Stone.BLACK)
        threats = get_threat_moves(state, Stone.BLACK)
        threat_cells = {(t.row, t.col) for t in threats}
        assert {(4, 1), (4, 6)} <= threat_cells


class TestAlphaBetaWrapperIntegration:
    """End-to-end check that ``AlphaBetaBrain.get_move`` still routes
    through the extracted ``quiesce`` correctly — the brain plays the
    instant winning move that quiescence detection underwrites."""

    def test_brain_finds_winning_move_via_quiescence(self):
        b = Board(size=9)
        for c in range(2, 6):
            b = b.place_stone(Move(row=4, col=c), Stone.BLACK)
        # depth=1 means the brain falls into quiescence at the leaf.
        brain = AlphaBetaBrain(depth=1)
        move = brain.get_move(b, Stone.BLACK)
        assert (move.row, move.col) in {(4, 1), (4, 6)}

    def test_qs_node_counter_advances(self):
        """The wrapper must thread the ctx counter back into
        ``self.qs_node_count`` — otherwise diagnostics report zero."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=3, col=3), Stone.WHITE)
        brain = AlphaBetaBrain(depth=1)
        brain.get_move(b, Stone.BLACK)
        assert brain.qs_node_count > 0


class TestQuiesceVcfProbe:
    """The leaf VCF probe is gated on a four-threat and shares one proof cache."""

    @staticmethod
    def _state(black: list[tuple[int, int]], white: list[tuple[int, int]]) -> SearchState:
        board = Board(size=15)
        for r, c in black:
            board = board.place_stone(Move(row=r, col=c), Stone.BLACK)
        for r, c in white:
            board = board.place_stone(Move(row=r, col=c), Stone.WHITE)
        return SearchState.from_board(board, Stone.BLACK)

    def test_no_four_threat_skips_the_probe(self):
        state = self._state([(7, 7), (7, 8)], [(3, 3), (11, 11)])
        ctx = QuiesceCtx(qs_vcf_depth=12, eval_cache={})
        quiesce(state, float("-inf"), float("inf"), Stone.BLACK, ctx, 0)
        assert ctx.vcf_cache == {}

    def test_probe_result_is_cached_and_reused(self):
        # Two open threes for Black: a forced win by continuous fours.
        state = self._state(
            [(4, 2), (4, 3), (4, 4), (6, 2), (6, 3), (6, 4)],
            [(0, 0), (0, 14), (14, 0), (14, 14), (2, 0), (2, 14)],
        )
        ctx = QuiesceCtx(qs_vcf_depth=12, eval_cache={})
        first = quiesce(state, float("-inf"), float("inf"), Stone.BLACK, ctx, 0)
        assert first == 100_000.0
        assert state.zobrist in ctx.vcf_cache
        size = len(ctx.vcf_cache)
        assert quiesce(state, float("-inf"), float("inf"), Stone.BLACK, ctx, 0) == first
        assert len(ctx.vcf_cache) == size
