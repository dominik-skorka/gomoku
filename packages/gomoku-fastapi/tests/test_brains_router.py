"""``GET /v1/brains`` lists registered brains and their settings schemas;
``GET /v1/brains/{name}/settings`` returns the settings a brain plays with."""

from gomoku_fastapi.deps import reset_settings_cache


class TestListBrains:
    def test_returns_all_built_in_brains(self, client):
        response = client.get("/v1/brains")
        assert response.status_code == 200
        names = {b["name"] for b in response.json()["brains"]}
        # The built-in registry must include all six brains.  Tests don't
        # need to lock in ordering because the endpoint sorts by name.
        assert {"random", "minimax", "negamax", "alphabeta", "threat", "mcts"} <= names

    def test_settings_schema_is_valid_json_schema(self, client):
        """Each brain entry must include a JSON Schema that a UI can use to
        render a form.  We don't validate the whole schema spec here —
        just check the basic shape so a regression in pydantic's
        ``model_json_schema`` would be caught."""
        response = client.get("/v1/brains")
        for entry in response.json()["brains"]:
            schema = entry["settings_schema"]
            assert isinstance(schema, dict)
            # Pydantic-generated schemas always have one of these top-level keys.
            assert "type" in schema or "properties" in schema or "$ref" in schema

    def test_alphabeta_schema_includes_depth(self, client):
        """Spot-check: a known brain knob should appear in the schema —
        otherwise a UI rendering this would silently miss settings."""
        response = client.get("/v1/brains")
        ab = next(b for b in response.json()["brains"] if b["name"] == "alphabeta")
        assert "depth" in ab["settings_schema"]["properties"]

    def test_response_sorted_by_name(self, client):
        """Stable ordering matters for clients that diff the list — they
        shouldn't see spurious changes when the dict iteration order
        of ``BRAINS`` shifts."""
        response = client.get("/v1/brains")
        names = [b["name"] for b in response.json()["brains"]]
        assert names == sorted(names)


class TestBrainSettings:
    def test_built_in_brain_has_its_defaults(self, client):
        response = client.get("/v1/brains/negamax/settings")
        assert response.status_code == 200
        body = response.json()
        assert body == {
            "name": "negamax",
            "type": "negamax",
            "settings": {"depth": 3, "evaluator": "pattern", "move_generator": "radius"},
        }

    def test_alias_values_win_over_defaults(self, client, tmp_path):
        (tmp_path / "gomoku.toml").write_text('[brains.alphabeta]\ntype = "alphabeta"\ndepth = 2\nvcf_depth = 20\n')
        reset_settings_cache()
        body = client.get("/v1/brains/alphabeta/settings").json()
        assert body["type"] == "alphabeta"
        assert body["settings"]["depth"] == 2
        assert body["settings"]["vcf_depth"] == 20
        # What the alias leaves out is the brain's default.
        assert body["settings"]["lmr"] is False

    def test_named_alias_resolves_to_its_type(self, client, tmp_path):
        (tmp_path / "gomoku.toml").write_text('[brains.fast_ab]\ntype = "alphabeta"\ndepth = 1\n')
        reset_settings_cache()
        body = client.get("/v1/brains/fast_ab/settings").json()
        assert body["name"] == "fast_ab"
        assert body["type"] == "alphabeta"
        assert body["settings"]["depth"] == 1

    def test_server_time_cap_is_applied(self, client, monkeypatch):
        monkeypatch.setenv("GOMOKU_API__MAX_TIME_LIMIT_S", "3")
        reset_settings_cache()
        settings = client.get("/v1/brains/alphabeta/settings").json()["settings"]
        # 0 means unlimited for alpha-beta, so the server's cap replaces it.
        assert settings["time_limit_s"] == 3.0
        threat = client.get("/v1/brains/threat/settings").json()["settings"]
        assert threat["pns_time_s"] == 3.0

    def test_keys_match_the_settings_schema(self, client):
        schema = next(b for b in client.get("/v1/brains").json()["brains"] if b["name"] == "threat")["settings_schema"]
        settings = client.get("/v1/brains/threat/settings").json()["settings"]
        assert set(settings) == set(schema["properties"])

    def test_unknown_brain_is_404(self, client):
        response = client.get("/v1/brains/nope/settings")
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "unknown_brain"
