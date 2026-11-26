"""Registers the built-in search strategies in :mod:`gomoku.strategies`.

* ``"pattern"`` evaluator: :func:`~gomoku_brain.evaluate.evaluate_board`,
  the pattern evaluation with its incremental ``SearchState`` cache.
* ``"radius"`` move generator: :func:`~gomoku_brain.candidates.radius_generator`,
  the empty cells within the radius of a stone.

Imported by :mod:`gomoku_brain`, so the names exist whenever a brain does.
"""

from gomoku.strategies import register_evaluator, register_move_generator

from gomoku_brain.candidates import radius_generator
from gomoku_brain.evaluate import evaluate_board

DEFAULT_EVALUATOR = "pattern"
DEFAULT_MOVE_GENERATOR = "radius"

register_evaluator(DEFAULT_EVALUATOR, evaluate_board)
register_move_generator(DEFAULT_MOVE_GENERATOR, radius_generator)
