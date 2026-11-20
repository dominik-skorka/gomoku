"""Brain settings given on the command line (NAME[,KEY=VALUE...])."""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from gomoku.settings import AlphaBetaBrainSettings, ThreatBrainSettings
from gomoku_cli.brain_spec import (
    apply_overrides,
    check_overrides,
    describe,
    parse_brain_spec,
    parse_value,
    split_parts,
)
from gomoku_cli.main import cli
from gomoku_cli.match import Engine, EngineError, EngineSpec


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "value"),
        [("8", 8), ("2.5", 2.5), ("true", True), ('"a b"', "a b"), ("vct", "vct"), ("[1, 2]", [1, 2])],
    )
    def test_values_are_toml_or_plain_words(self, text, value):
        assert parse_value(text) == value

    def test_commas_inside_brackets_and_quotes_do_not_split(self):
        assert split_parts('threat,x=[1, 2],y="a,b",z=3') == ["threat", "x=[1, 2]", 'y="a,b"', "z=3"]

    def test_dotted_keys_build_nested_tables(self):
        name, overrides = parse_brain_spec("threat,depth=8,fallback.depth=8,threats.own.split_threes=true")
        assert name == "threat"
        assert overrides == {"depth": 8, "fallback": {"depth": 8}, "threats": {"own": {"split_threes": True}}}

    def test_a_bare_name_has_no_overrides(self):
        assert parse_brain_spec("alphabeta") == ("alphabeta", {})

    @pytest.mark.parametrize("text", ["threat,depth", "threat,=3", "depth=3", "threat,fallback=1,fallback.depth=2"])
    def test_malformed_specs_are_errors(self, text):
        with pytest.raises(ValueError):
            parse_brain_spec(text)

    def test_describe_is_the_inverse(self):
        overrides = {"depth": 8, "fallback": {"depth": 8, "lmr": False}}
        assert describe(overrides) == "depth=8,fallback.depth=8,fallback.lmr=false"
        assert parse_brain_spec("threat," + describe(overrides))[1] == overrides


class TestChecking:
    def test_unknown_key_names_the_valid_ones(self):
        with pytest.raises(ValueError, match=r"unknown setting 'dpeth'; valid here: depth, fallback"):
            check_overrides(ThreatBrainSettings, {"dpeth": 8})

    def test_unknown_nested_key(self):
        with pytest.raises(ValueError, match=r"unknown setting 'fallback.dpeth'"):
            check_overrides(ThreatBrainSettings, {"fallback": {"dpeth": 8}})
        with pytest.raises(ValueError, match=r"unknown setting 'threats.own.splitthrees'"):
            check_overrides(ThreatBrainSettings, {"threats": {"own": {"splitthrees": True}}})

    def test_a_single_value_is_not_a_table(self):
        with pytest.raises(ValueError, match=r"'depth' is a single value"):
            check_overrides(ThreatBrainSettings, {"depth": {"x": 1}})

    def test_overrides_go_on_top_of_the_alias_settings(self):
        base = ThreatBrainSettings(pns_nodes=120_000, fallback=AlphaBetaBrainSettings(depth=5, max_candidates=12))
        merged = apply_overrides(ThreatBrainSettings, base.model_dump(), {"depth": 8, "fallback": {"depth": 7}})
        assert merged["depth"] == 8 and merged["pns_nodes"] == 120_000
        assert merged["fallback"]["depth"] == 7 and merged["fallback"]["max_candidates"] == 12

    def test_invalid_values_are_reported_in_one_line(self):
        with pytest.raises(ValueError, match=r"^depth: Input should be less than or equal to 10$"):
            apply_overrides(ThreatBrainSettings, ThreatBrainSettings().model_dump(), {"depth": 99})


class TestEngineSpec:
    def test_settings_parts_become_overrides(self):
        spec = EngineSpec.parse("brain=threat,rev=master,pns_nodes=60000,fallback.depth=4")
        assert (spec.brain, spec.rev) == ("threat", "master")
        assert spec.overrides == {"pns_nodes": 60000, "fallback": {"depth": 4}}
        assert spec.label == "threat(pns_nodes=60000,fallback.depth=4)@master"

    def test_plain_spec_is_unchanged(self):
        assert EngineSpec.parse("alphabeta,rev=HEAD") == EngineSpec("alphabeta", rev="HEAD")

    def test_worker_rejects_an_unknown_setting_at_start(self):
        with pytest.raises(EngineError, match=r"unknown setting 'dpeth'"):
            Engine(EngineSpec("threat", overrides={"dpeth": 3}), None, 15, 60.0)


class TestPlayCommand:
    def test_unknown_setting_is_a_usage_error(self):
        result = CliRunner().invoke(cli, ["play", "--black", "threat,dpeth=8", "--white", "random", "--no-record"])
        assert result.exit_code == 2
        assert "unknown setting 'dpeth'" in result.output

    def test_bad_syntax_is_a_usage_error(self):
        result = CliRunner().invoke(cli, ["play", "--black", "threat,depth", "--white", "random", "--no-record"])
        assert result.exit_code == 2
        assert "expected KEY=VALUE" in result.output
