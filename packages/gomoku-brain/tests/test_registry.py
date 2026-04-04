import pytest
from gomoku_brain import BRAINS, get_brain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.random import RandomBrain


class TestRegistry:
    def test_all_brains_registered(self):
        assert set(BRAINS) == {"random", "minimax", "negamax", "alphabeta"}

    def test_get_brain_random(self):
        brain = get_brain("random")
        assert isinstance(brain, RandomBrain)

    def test_get_brain_with_kwargs(self):
        brain = get_brain("minimax", depth=2)
        assert isinstance(brain, MinimaxBrain)
        assert brain.depth == 2

    def test_get_brain_unknown(self):
        with pytest.raises(ValueError, match="Unknown brain"):
            get_brain("nonexistent")
