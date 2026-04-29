"""Tests for ``alphabeta_root.root_vcf``.

The root VCF pre-check used to live as a class method on
``AlphaBetaBrain`` (``_root_vcf``).  It was lifted into its own module
because (a) it's ~90 lines of self-contained pre-search analysis, and
(b) extracting it shrinks ``alphabeta.py`` below the 1k-line threshold.

These tests pin three things:

1. The function returns the instant winning move when one exists,
   bypassing the expensive VCF probe entirely.
2. It returns ``None`` when neither side has a VCF — the caller will
   then run the normal alpha-beta search.
3. The brain still uses the function via ``get_move`` (end-to-end smoke).
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.alphabeta_root import root_vcf


class TestRootVcfDirect:
    def test_returns_instant_win(self):
        # Black has 4-in-a-row on row 7 cols 3-6 — placing at (7, 2) or (7, 7)
        # completes 5.  ``root_vcf`` short-circuits and returns the win.
        b = Board(size=15)
        for c in range(3, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        result = root_vcf(b, Stone.BLACK, vcf_depth=4)
        assert result is not None
        assert (result.row, result.col) in {(7, 2), (7, 7)}

    def test_returns_none_when_no_vcf(self):
        # Empty-ish board with no four-threat, so neither side has a VCF.
        b = Board(size=15)
        b = b.place_stone(Move(row=7, col=7), Stone.BLACK)
        b = b.place_stone(Move(row=8, col=8), Stone.WHITE)
        assert root_vcf(b, Stone.BLACK, vcf_depth=4) is None

    def test_finds_own_vcf_win(self):
        # Black open-three on row 7 cols 4-6: placing at (7, 3) or (7, 7)
        # creates a live four → forced win in 1 ply.  VCF probe should
        # find this even when there's no immediate 5-in-a-row available.
        b = Board(size=15)
        for c in range(4, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        result = root_vcf(b, Stone.BLACK, vcf_depth=4)
        assert result is not None


class TestAlphaBetaUsesRootVcf:
    """End-to-end check that ``AlphaBetaBrain.get_move`` still goes
    through the extracted ``root_vcf`` function — the brain plays the
    instant winning move, which only the root pre-check returns
    (regular alpha-beta would also find it but takes longer)."""

    def test_brain_finds_instant_win_via_root_vcf(self):
        b = Board(size=15)
        for c in range(3, 7):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        # Default alphabeta has vcf_depth=0, so explicitly enable it.
        brain = AlphaBetaBrain(depth=2, vcf_depth=4)
        move = brain.get_move(b, Stone.BLACK)
        assert (move.row, move.col) in {(7, 2), (7, 7)}
