"""The pattern evaluation with other weights, as an :class:`~gomoku.interfaces.Evaluator`.

:func:`~gomoku_brain.evaluate.evaluate_board` scores runs of stones with
fixed weights (``_PATTERN_SCORES`` and the fork / open-three bonuses).
:func:`pattern_evaluator` builds the same evaluation with the weights of a
:class:`PatternWeights`, on the same Numba kernels and with the same
incremental ``SearchState`` cache, so it is as fast::

    register_evaluator("pattern_x", pattern_evaluator(PatternWeights(dead_four=3_000)))

Unlike ``evaluate_board`` it always runs the grid kernels: it ignores
``[evaluation] use_jit`` and ``[bitboard] enabled`` (Numba is a
dependency of ``gomoku-brain``).  Scores are clamped to ±``_EVAL_CLAMP``.
"""

from dataclasses import dataclass, fields

import numpy as np
from gomoku.interfaces import BoardView, Evaluator
from gomoku.models import Stone

from gomoku_brain.evaluate import _EVAL_CLAMP, _eval_state_black
from gomoku_brain.evaluate_jit import evaluate_grid, get_dirs
from gomoku_brain.grid_utils import board_to_grid, occupied_coords
from gomoku_brain.search_state import SearchState

# Scored for any run of five, as in the built-in table: never tuned, the
# searches detect wins themselves and the clamp keeps it below their scores.
_FIVE = 100_000


@dataclass(frozen=True)
class PatternWeights:
    """Weights of the pattern evaluation; the defaults are the built-in ones.

    ``live_*`` runs have both ends open, ``dead_*`` one.  Each stone of a
    run scores it in each direction, the opponent's runs count against.
    """

    live_four: int = 10_000
    dead_four: int = 1_000
    live_three: int = 1_000
    dead_three: int = 100
    live_two: int = 100
    dead_two: int = 10
    live_one: int = 10
    dead_one: int = 1
    #: Smallest run score that counts as a threat line for forks.
    fork_threshold: int = 100
    #: Per extra threat line converging on an empty cell.
    fork_bonus: int = 3_000
    #: Per open three (three in a row, both ends empty).
    open_three_bonus: int = 1_500

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not 0 <= value < _FIVE:
                msg = f"{f.name} must be in [0, {_FIVE}), got {value}"
                raise ValueError(msg)

    def kernel_constants(self) -> tuple:  # type: ignore[type-arg]
        """``(dirs, table, fork_threshold, fork_bonus, open_three_bonus)`` for the kernels."""
        table = np.zeros((6, 3), dtype=np.int32)
        table[5, :] = _FIVE
        table[4, 2], table[4, 1] = self.live_four, self.dead_four
        table[3, 2], table[3, 1] = self.live_three, self.dead_three
        table[2, 2], table[2, 1] = self.live_two, self.dead_two
        table[1, 2], table[1, 1] = self.live_one, self.dead_one
        return (get_dirs(), table, self.fork_threshold, self.fork_bonus, self.open_three_bonus)


def pattern_evaluator(weights: PatternWeights) -> Evaluator:
    """The pattern evaluation with *weights*.

    The kernel constants are built once here; the returned function holds
    nothing else, so threads can share it (each search has its own
    ``SearchState``, whose cache it uses).
    """
    consts = weights.kernel_constants()
    dirs, table, fork_threshold, fork_bonus, open_three_bonus = consts

    def evaluate(board: BoardView, stone: Stone, /) -> float:
        if isinstance(board, SearchState):
            e = _eval_state_black(board, consts)
            raw = e if stone is Stone.BLACK else -e
        else:
            raw = evaluate_grid(
                board_to_grid(board),
                board.size,
                occupied_coords(board),
                int(stone),
                dirs,
                table,
                fork_threshold,
                fork_bonus,
                open_three_bonus,
            )
        return max(-_EVAL_CLAMP, min(_EVAL_CLAMP, raw))

    return evaluate
