"""Server-only brain settings (``gomoku.settings.SERVER_ONLY``, #113).

``opening_book`` names a file on the server.  Clients must not choose it:
``brain_overrides`` with another value are rejected before the brain (and
its file access) is built, and the setting is not published.
"""

from pathlib import Path

import pytest
from gomoku.board import Board
from gomoku.models import Stone
from gomoku.settings import AlphaBetaBrainSettings, ThreatBrainSettings
from gomoku_brain.opening_book import OpeningBookFile
from gomoku_fastapi.deps import reset_settings_cache
from gomoku_fastapi.server_only import pin_server_only, public_schema, strip_server_only


@pytest.fixture
def book_loads(monkeypatch) -> list[Path]:
    """Every path an opening book is loaded from during the test."""
    loaded: list[Path] = []
    original = OpeningBookFile.load.__func__

    def spy(cls, path: Path) -> OpeningBookFile:
        loaded.append(Path(path))
        return original(cls, path)

    monkeypatch.setattr(OpeningBookFile, "load", classmethod(spy))
    return loaded


@pytest.fixture
def server_book(tmp_path) -> Path:
    path = tmp_path / "server_book.json"
    OpeningBookFile().save(path)
    return path


def _move(client, brain: str, overrides: dict | None):
    payload = {
        "board": Board(size=9).model_dump(mode="json"),
        "side_to_move": int(Stone.BLACK),
        "brain": brain,
        "brain_overrides": overrides,
    }
    return client.post("/v1/move", json=payload)


def _toml(tmp_path, text: str) -> None:
    (tmp_path / "gomoku.toml").write_text(text)
    reset_settings_cache()


class TestOverridesRejected:
    def test_top_level_path_is_rejected_without_touching_the_file(self, client, tmp_path, book_loads):
        # A file that exists and is a valid book, so only the policy can stop it.
        target = tmp_path / "elsewhere.json"
        OpeningBookFile().save(target)
        response = _move(client, "alphabeta", {"depth": 1, "opening_book": str(target)})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "invalid_brain_overrides"
        assert detail["server_only"] == ["opening_book"]
        assert book_loads == []

    def test_path_inside_the_threat_fallback_is_rejected(self, client, book_loads):
        response = _move(client, "threat", {"fallback": {"depth": 1, "opening_book": "/etc/passwd"}})
        assert response.status_code == 422
        assert response.json()["detail"]["server_only"] == ["fallback.opening_book"]
        assert book_loads == []

    def test_sessions_reject_it_too(self, client):
        response = client.post(
            "/v1/games",
            json={"black": {"brain": "alphabeta", "brain_overrides": {"opening_book": "/etc/passwd"}}},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["server_only"] == ["opening_book"]


class TestOverridesAccepted:
    def test_repeating_the_servers_value_is_fine(self, client):
        # The web sends whole nested groups (the server's values with a change).
        response = _move(client, "threat", {"depth": 1, "fallback": {"depth": 1, "opening_book": ""}})
        assert response.status_code == 200

    def test_toml_book_still_loads(self, client, tmp_path, server_book, book_loads):
        _toml(tmp_path, f'[brains.ab]\ntype = "alphabeta"\ndepth = 1\nopening_book = "{server_book}"\n')
        assert _move(client, "ab", None).status_code == 200
        assert _move(client, "ab", {"depth": 2}).status_code == 200
        assert _move(client, "ab", {"opening_book": str(server_book)}).status_code == 200
        assert book_loads == [server_book] * 3

    def test_a_group_sent_without_the_book_keeps_the_servers_book(self, client, tmp_path, server_book, book_loads):
        _toml(
            tmp_path,
            f'[brains.th]\ntype = "threat"\ndepth = 1\n'
            f'[brains.th.fallback]\ndepth = 1\nopening_book = "{server_book}"\n',
        )
        assert _move(client, "th", {"fallback": {"depth": 2}}).status_code == 200
        assert book_loads == [server_book]

    def test_a_different_value_against_a_toml_book_is_rejected(self, client, tmp_path, server_book, book_loads):
        _toml(tmp_path, f'[brains.ab]\ntype = "alphabeta"\nopening_book = "{server_book}"\n')
        response = _move(client, "ab", {"opening_book": ""})
        assert response.status_code == 422
        assert book_loads == []


class TestNotPublished:
    def test_schemas_have_no_server_only_settings(self, client):
        for entry in client.get("/v1/brains").json()["brains"]:
            schema = entry["settings_schema"]
            for node in [schema, *schema.get("$defs", {}).values()]:
                assert "opening_book" not in node.get("properties", {}), entry["name"]

    def test_effective_settings_hide_the_servers_path(self, client, tmp_path, server_book):
        _toml(tmp_path, f'[brains.th]\ntype = "threat"\n[brains.th.fallback]\nopening_book = "{server_book}"\n')
        settings = client.get("/v1/brains/th/settings").json()["settings"]
        assert "opening_book" not in settings["fallback"]
        assert str(server_book) not in str(settings)
        ab = client.get("/v1/brains/alphabeta/settings").json()["settings"]
        assert "opening_book" not in ab


class TestHelpers:
    def test_pin_reports_paths_and_restores_values(self):
        resolved = ThreatBrainSettings(fallback={"opening_book": "server.json"}).model_dump()
        requested = ThreatBrainSettings(fallback={"opening_book": "client.json", "depth": 2}).model_dump()
        rejected = pin_server_only(
            ThreatBrainSettings, requested, resolved, {"fallback": {"opening_book": "client.json", "depth": 2}}
        )
        assert rejected == ["fallback.opening_book"]
        assert requested["fallback"] == {**requested["fallback"], "opening_book": "server.json", "depth": 2}

    def test_unset_nested_group_compares_against_defaults(self):
        resolved = ThreatBrainSettings().model_dump()
        assert resolved["fallback"] is None
        requested = ThreatBrainSettings(fallback={"opening_book": ""}).model_dump()
        assert pin_server_only(ThreatBrainSettings, requested, resolved, {"fallback": {"opening_book": ""}}) == []

    def test_strip_and_schema(self):
        values = ThreatBrainSettings(fallback={}).model_dump()
        stripped = strip_server_only(ThreatBrainSettings, values)
        assert "opening_book" not in stripped["fallback"]
        assert "opening_book" in values["fallback"]  # the input is not changed
        assert "opening_book" not in public_schema(AlphaBetaBrainSettings)["properties"]
        assert "depth" in public_schema(AlphaBetaBrainSettings)["properties"]
