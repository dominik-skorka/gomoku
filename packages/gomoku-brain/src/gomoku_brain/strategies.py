"""Registers the built-in search strategies in :mod:`gomoku.strategies`.

* ``"pattern"`` evaluator: :func:`~gomoku_brain.evaluate.evaluate_board`,
  the pattern evaluation with its incremental ``SearchState`` cache.
* ``"pattern_open3"`` evaluator: the same patterns with the open-three
  bonus doubled (:mod:`~gomoku_brain.pattern_weights`), measured stronger
  in #105; chosen in ``gomoku.toml`` for alpha-beta and the threat
  brain's fallback, the code default stays ``"pattern"``.
* ``"radius"`` move generator: :func:`~gomoku_brain.candidates.radius_generator`,
  the empty cells within the radius of a stone.

Imported by :mod:`gomoku_brain`, so the names exist whenever a brain does.
"""

from gomoku.strategies import register_evaluator, register_move_generator

from gomoku_brain.candidates import radius_generator
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.pattern_weights import PatternWeights, pattern_evaluator

DEFAULT_EVALUATOR = "pattern"
DEFAULT_MOVE_GENERATOR = "radius"

register_evaluator(DEFAULT_EVALUATOR, evaluate_board)
register_move_generator(DEFAULT_MOVE_GENERATOR, radius_generator)

# Measured in #105: open_three_bonus 3_000 (built-in 1_500) beat "pattern"
# by +28 Elo (95% interval +1 to +55, 300 games, alpha-beta at 0.4 s a move)
# and by +50 (+16 to +85) on other openings (--seed 2).
OPEN_THREE_EVALUATOR = "pattern_open3"
register_evaluator(OPEN_THREE_EVALUATOR, pattern_evaluator(PatternWeights(open_three_bonus=3_000)))
