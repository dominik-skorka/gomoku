"""``GET /v1/brains`` lists registered brains and their settings schemas."""

from __future__ import annotations


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
