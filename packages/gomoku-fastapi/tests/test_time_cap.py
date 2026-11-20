"""API-side cap on per-request search time (``api.max_time_limit_s``)."""

from gomoku.settings import GomokuSettings
from gomoku_fastapi.deps import _cap_time_budget, build_brain


def _settings(cap: float) -> GomokuSettings:
    return GomokuSettings(api={"max_time_limit_s": cap})


class TestCapTimeBudget:
    def test_unlimited_alphabeta_gets_cap(self):
        assert _cap_time_budget("alphabeta", {"time_limit_s": 0.0}, 5.0)["time_limit_s"] == 5.0

    def test_long_alphabeta_limit_is_clamped(self):
        assert _cap_time_budget("alphabeta", {"time_limit_s": 120.0}, 5.0)["time_limit_s"] == 5.0

    def test_short_alphabeta_limit_is_kept(self):
        assert _cap_time_budget("alphabeta", {"time_limit_s": 2.0}, 5.0)["time_limit_s"] == 2.0

    def test_threat_without_fallback_gets_capped_fallback_and_pns(self):
        out = _cap_time_budget("threat", {"depth": 6, "pns_time_s": 30.0, "fallback": None}, 5.0)
        assert out["fallback"] == {"time_limit_s": 5.0}  # no depth key: keeps top-level depth
        assert out["pns_time_s"] == 5.0

    def test_threat_existing_fallback_is_clamped(self):
        out = _cap_time_budget("threat", {"pns_time_s": 1.0, "fallback": {"depth": 3, "time_limit_s": 99.0}}, 5.0)
        assert out["fallback"] == {"depth": 3, "time_limit_s": 5.0}
        assert out["pns_time_s"] == 1.0

    def test_other_brains_untouched(self):
        kwargs = {"iterations": 100}
        assert _cap_time_budget("mcts", kwargs, 5.0) == kwargs

    def test_input_not_mutated(self):
        kwargs = {"time_limit_s": 0.0}
        _cap_time_budget("alphabeta", kwargs, 5.0)
        assert kwargs == {"time_limit_s": 0.0}


class TestBuildBrain:
    def test_alphabeta_is_capped(self):
        brain = build_brain("alphabeta", None, _settings(3.0))
        assert brain.settings.time_limit_s == 3.0

    def test_override_cannot_exceed_cap(self):
        brain = build_brain("alphabeta", {"time_limit_s": 500.0}, _settings(3.0))
        assert brain.settings.time_limit_s == 3.0

    def test_threat_fallback_is_capped_and_keeps_depth(self):
        brain = build_brain("threat", None, _settings(3.0))
        assert brain._fallback.settings.time_limit_s == 3.0
        assert brain._fallback.settings.depth == brain.settings.depth

    def test_zero_disables_cap(self):
        brain = build_brain("alphabeta", {"time_limit_s": 100.0}, _settings(0.0))
        assert brain.settings.time_limit_s == 100.0
