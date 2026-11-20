"""Tests for the single-source brain registry.

The registry pairs each brain class with its pydantic settings model so
adding a new brain only requires editing :data:`gomoku_brain.BRAINS` —
no parallel dict in :mod:`gomoku.settings` to keep in sync.
"""

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
    unregister_brain,
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
        assert brain.settings.depth == 2

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
        assert brain.settings.depth == 1


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
            unregister_brain("custom_alias")


class TestThreatFallbackTuning:
    def test_default_threat_brain_has_untuned_fallback(self):
        """Without a fallback table, the alpha-beta fallback keeps bare
        defaults at the threat brain's depth — behaviour unchanged."""
        brain = get_brain("threat", depth=5)
        fb = brain._fallback
        assert isinstance(fb, AlphaBetaBrain)
        assert fb.settings.depth == 5
        assert fb.settings.qs_vcf_depth == 0
        assert fb.settings.lmr is False
        assert fb.settings.null_move_pruning is False

    def test_fallback_table_tunes_the_fallback(self):
        bs = BrainSettings.model_validate(
            {
                "tuned_threat": {
                    "type": "threat",
                    "depth": 4,
                    "fallback": {
                        "qs_vcf_depth": 12,
                        "threat_extensions": 2,
                        "null_move_pruning": True,
                        "lmr": True,
                        "time_limit_s": 120.0,
                    },
                }
            }
        )
        brain = get_brain_from_settings("tuned_threat", bs)
        fb = brain._fallback
        assert fb.settings.qs_vcf_depth == 12
        assert fb.settings.threat_extensions == 2
        assert fb.settings.null_move_pruning is True
        assert fb.settings.lmr is True
        assert fb.settings.time_limit_s == 120.0
        # vcf_depth stays off — the threat brain already ran VCF itself.
        assert fb.settings.vcf_depth == 0

    def test_alphabeta_alias_keeps_early_game_settings(self):
        """Regression: early_game_* keys used to be silently dropped by
        AlphaBetaBrainSettings (pydantic ignores extras), so the TOML
        values never reached the brain."""
        bs = BrainSettings.model_validate(
            {"ab": {"type": "alphabeta", "early_game_max_candidates": 8, "early_game_futility_margin": 600}}
        )
        brain = get_brain_from_settings("ab", bs)
        assert brain.settings.early_game_max_candidates == 8
        assert brain.settings.early_game_futility_margin == 600

    def test_fallback_search_shape_settings_reach_the_brain(self):
        bs = BrainSettings.model_validate(
            {
                "t": {
                    "type": "threat",
                    "fallback": {"depth": 5, "eval_ordering_depth": 11, "max_candidates": 12, "futility_margin": 900},
                }
            }
        )
        fb = get_brain_from_settings("t", bs)._fallback.settings
        assert (fb.depth, fb.eval_ordering_depth, fb.max_candidates, fb.futility_margin) == (5, 11, 12, 900)

    def test_alphabeta_search_shape_defaults_are_unchanged(self):
        s = get_brain("alphabeta").settings
        assert (s.max_candidates, s.futility_margin, s.eval_ordering_depth) == (20, 1500, 3)
