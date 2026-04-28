"""Tests for the single-source brain registry.

The registry pairs each brain class with its pydantic settings model so
adding a new brain only requires editing :data:`gomoku_brain.BRAINS` —
no parallel dict in :mod:`gomoku.settings` to keep in sync.
"""

from __future__ import annotations

import pytest
from gomoku.settings import (
    AlphaBetaBrainSettings,
    BrainSettings,
    MCTSBrainSettings,
    MinimaxBrainSettings,
    RandomBrainSettings,
    ThreatBrainSettings,
)
from gomoku_brain import (
    BRAINS,
    BrainSpec,
    get_brain,
    get_brain_from_settings,
    register_brain,
    resolve_brain,
)
from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.mcts import MCTSBrain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.random import RandomBrain
from gomoku_brain.threat import ThreatBrain


class TestRegistryShape:
    def test_all_brains_registered(self):
        expected = {"random", "minimax", "negamax", "alphabeta", "threat", "mcts"}
        assert expected <= set(BRAINS)

    def test_each_entry_is_a_brainspec(self):
        for name, spec in BRAINS.items():
            assert isinstance(spec, BrainSpec)
            assert spec.name == name

    def test_registry_pairs_brain_with_settings(self):
        # Lock in the canonical pairings so a future drift fails fast.
        expected = {
            "random": (RandomBrain, RandomBrainSettings),
            "minimax": (MinimaxBrain, MinimaxBrainSettings),
            "alphabeta": (AlphaBetaBrain, AlphaBetaBrainSettings),
            "threat": (ThreatBrain, ThreatBrainSettings),
            "mcts": (MCTSBrain, MCTSBrainSettings),
        }
        for name, (brain_cls, settings_cls) in expected.items():
            assert BRAINS[name].brain_cls is brain_cls
            assert BRAINS[name].settings_cls is settings_cls


class TestGetBrain:
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


class TestResolveBrain:
    def test_resolves_builtin_type_with_defaults(self):
        brain_type, kwargs = resolve_brain("random", BrainSettings())
        assert brain_type == "random"
        assert kwargs == {}

    def test_resolves_alias_to_underlying_type(self):
        bs = BrainSettings.model_validate({"strong_ab": {"type": "alphabeta", "depth": 6}})
        brain_type, kwargs = resolve_brain("strong_ab", bs)
        assert brain_type == "alphabeta"
        assert kwargs["depth"] == 6

    def test_alias_kwargs_validated_against_settings_model(self):
        # depth=999 violates the AlphaBetaBrainSettings 1..10 range.
        bs = BrainSettings.model_validate({"too_deep": {"type": "alphabeta", "depth": 999}})
        with pytest.raises(ValueError):
            resolve_brain("too_deep", bs)

    def test_alias_without_type_key_raises(self):
        bs = BrainSettings.model_validate({"broken": {"depth": 4}})
        with pytest.raises(ValueError, match="must be a table with a 'type' key"):
            resolve_brain("broken", bs)

    def test_unknown_alias_type_raises(self):
        bs = BrainSettings.model_validate({"weird": {"type": "nonexistent"}})
        with pytest.raises(ValueError, match="Unknown brain type"):
            resolve_brain("weird", bs)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown brain"):
            resolve_brain("nonexistent", BrainSettings())


class TestGetBrainFromSettings:
    def test_builtin_with_default_settings(self):
        brain = get_brain_from_settings("random")
        assert isinstance(brain, RandomBrain)

    def test_alias_uses_alias_kwargs(self):
        bs = BrainSettings.model_validate({"deep_mm": {"type": "minimax", "depth": 1}})
        brain = get_brain_from_settings("deep_mm", bs)
        assert isinstance(brain, MinimaxBrain)
        assert brain.depth == 1


class TestRegisterBrain:
    def test_register_round_trips(self):
        from gomoku_brain.random import RandomBrain as _RB

        try:
            register_brain("custom_alias", _RB, RandomBrainSettings)
            assert "custom_alias" in BRAINS
            assert BRAINS["custom_alias"].brain_cls is _RB
            assert BRAINS["custom_alias"].settings_cls is RandomBrainSettings
            # And it resolves like any other built-in.
            assert resolve_brain("custom_alias", BrainSettings()) == ("custom_alias", {})
        finally:
            BRAINS.pop("custom_alias", None)
