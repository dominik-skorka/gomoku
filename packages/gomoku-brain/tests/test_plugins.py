"""Plug-ins: modules declared under the ``gomoku.plugins`` entry points."""

import sys
import textwrap
from importlib.metadata import EntryPoint
from pathlib import Path

import gomoku_brain
import pytest
from gomoku.settings import BrainSettings
from gomoku.strategies import EVALUATORS, unregister_strategy
from gomoku_brain import (
    BRAINS,
    PLUGIN_GROUP,
    PluginWarning,
    get_brain,
    get_brain_from_settings,
    load_plugins,
    unregister_brain,
)

_PLUGIN = """
from typing import override

from gomoku_brain import Brain
from gomoku_brain.candidates import get_candidate_moves


class FirstBrain(Brain, name="{name}"):
    @override
    def choose_move(self, board, stone, session):
        return get_candidate_moves(board)[0]
"""


@pytest.fixture
def plugins(monkeypatch, tmp_path: Path):
    """Install fake entry points: ``plugins(name=module_source, ...)``.

    Each module is written to *tmp_path* (on ``sys.path``) and declared
    under :data:`PLUGIN_GROUP`; the registry forgets that plug-ins were
    loaded, so its next use loads these.  Brains and modules the test
    created are removed afterwards.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    created: list[str] = []

    def install(**modules: str) -> None:
        eps = []
        for module, source in modules.items():
            (tmp_path / f"{module}.py").write_text(textwrap.dedent(source))
            eps.append(EntryPoint(name=module, value=module, group=PLUGIN_GROUP))
            created.append(module)

        def fake_entry_points(*, group: str) -> list[EntryPoint]:
            return eps if group == PLUGIN_GROUP else []

        monkeypatch.setattr(gomoku_brain, "entry_points", fake_entry_points)
        monkeypatch.setattr(gomoku_brain, "_plugins_loaded", False)

    yield install
    for module in created:
        unregister_brain(module)
        unregister_strategy(module)
        sys.modules.pop(module, None)


_STRATEGY_PLUGIN = """
from gomoku.strategies import register_evaluator


def material(board, stone, /):
    own = sum(1 for r, c in board.occupied if board.at(r, c) is stone)
    return float(2 * own - len(board.occupied))


register_evaluator("{name}", material)
"""


def test_registry_loads_plugins_on_first_use(plugins):
    plugins(plug_first=_PLUGIN.format(name="plug_first"))
    assert "plug_first" in BRAINS
    assert get_brain("plug_first").name == "plug_first"


def test_load_plugins_runs_once(plugins):
    plugins(plug_once=_PLUGIN.format(name="plug_once"))
    assert load_plugins() == ["plug_once"]
    assert load_plugins() == []


def test_broken_plugin_is_skipped_with_a_warning(plugins):
    plugins(plug_broken="raise ImportError('missing dependency')", plug_ok=_PLUGIN.format(name="plug_ok"))
    with pytest.warns(PluginWarning, match="plug_broken.*missing dependency"):
        assert load_plugins() == ["plug_ok"]
    assert "plug_ok" in BRAINS
    assert "plug_broken" not in BRAINS


def test_plugin_cannot_take_a_built_in_name(plugins):
    plugins(plug_threat=_PLUGIN.format(name="threat"))
    with pytest.warns(PluginWarning, match="'threat' is taken"):
        load_plugins()
    assert BRAINS["threat"].brain_cls is gomoku_brain.ThreatBrain


def test_plugin_strategy_is_known_before_settings_are_validated(plugins):
    """A plug-in evaluator works in an alias, as on the CLI and in the API:
    resolving the brain loads plug-ins before its settings are checked."""
    plugins(plug_eval=_STRATEGY_PLUGIN.format(name="plug_eval"))
    assert "plug_eval" not in EVALUATORS  # not loaded yet
    aliases = BrainSettings.model_validate({"ab_plug": {"type": "alphabeta", "evaluator": "plug_eval", "depth": 1}})
    brain = get_brain_from_settings("ab_plug", aliases)
    assert "plug_eval" in EVALUATORS
    assert brain.settings.evaluator == "plug_eval"
