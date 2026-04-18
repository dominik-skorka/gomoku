"""Shared direction vectors for line scanning on the Gomoku board.

The four directions are horizontal, vertical, and the two diagonals.  Many
brain modules (`alphabeta`, `threat`, `evaluate`, `mcts`) need the same
``list[tuple[int, int]]`` and the numba-compiled numpy variant from
:mod:`gomoku_brain.evaluate_jit`.  Having them in one place prevents silent
drift when the set is extended.
"""

from __future__ import annotations

# (dr, dc) pairs.  Order matches ``evaluate_jit._DIRS`` row-for-row so a
# scan-by-direction produces identical results in both paths.
DIRECTIONS: list[tuple[int, int]] = [(0, 1), (1, 0), (1, 1), (1, -1)]
