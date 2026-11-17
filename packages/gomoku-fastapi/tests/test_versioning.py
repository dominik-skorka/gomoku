"""API versioning: discovery endpoint, version headers and deprecation signals."""

from __future__ import annotations

from datetime import date
from importlib.metadata import version as package_version

import pytest
from fastapi.testclient import TestClient
from gomoku_fastapi.deps import reset_settings_cache
from gomoku_fastapi.main import create_app
from gomoku_fastapi.versioning import LATEST, V1, VERSIONS, ApiVersion, VersionHeadersMiddleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route


def _move_payload() -> dict:
    board = {"size": 15, "grid": [[0] * 15 for _ in range(15)], "occupied": []}
    board["grid"][7][7] = 1
    board["occupied"] = [[7, 7]]
    return {"board": board, "side_to_move": 2, "brain": "random"}


class TestVersionsEndpoint:
    def test_lists_served_versions(self, client):
        r = client.get("/versions")
        assert r.status_code == 200
        body = r.json()
        assert body["latest"] == LATEST.name == "v1"
        assert body["versions"] == [
            {
                "version": "v1",
                "path": "/v1",
                "status": "stable",
                "released": "2026-05-01",
                "deprecated": None,
                "sunset": None,
                "successor": None,
            }
        ]

    def test_is_unversioned(self, client):
        assert client.get("/v1/versions").status_code == 404
        assert "api-version" not in client.get("/versions").headers


class TestVersionHeader:
    def test_success_response_carries_version(self, client):
        r = client.get("/v1/brains")
        assert r.status_code == 200
        assert r.headers["API-Version"] == "v1"
        assert "deprecation" not in r.headers and "sunset" not in r.headers

    @pytest.mark.parametrize(
        ("method", "path", "kwargs", "status"),
        [
            ("get", "/v1/no-such-endpoint", {}, 404),
            ("post", "/v1/move", {"json": {"brain": "random"}}, 422),
            ("get", "/v1/games/00000000-0000-0000-0000-000000000000", {}, 404),
        ],
    )
    def test_error_responses_carry_version(self, client, method, path, kwargs, status):
        r = getattr(client, method)(path, **kwargs)
        assert r.status_code == status
        assert r.headers["API-Version"] == "v1"

    def test_rate_limited_response_carries_version(self, monkeypatch):
        monkeypatch.setenv("GOMOKU_API__RATE_LIMIT_COMPUTE", "1/minute")
        reset_settings_cache()
        tight = TestClient(create_app())
        assert tight.post("/v1/move", json=_move_payload()).status_code == 200
        r = tight.post("/v1/move", json=_move_payload())
        assert r.status_code == 429
        assert r.headers["API-Version"] == "v1"

    def test_unversioned_endpoints_have_no_version_header(self, client):
        assert "api-version" not in client.get("/healthz").headers
        assert "api-version" not in client.get("/openapi.json").headers


class TestOpenApi:
    def test_document_version_is_the_package_version(self, client):
        assert client.get("/openapi.json").json()["info"]["version"] == package_version("gomoku-fastapi")

    def test_every_operation_is_versioned_or_infrastructure(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        prefixes = tuple(f"{v.prefix}/" for v in VERSIONS)
        stray = [p for p in paths if not p.startswith(prefixes) and p not in {"/healthz", "/versions"}]
        assert stray == []


class TestCorsExposesVersionHeaders:
    def test_browser_can_read_version_and_request_id(self, monkeypatch):
        monkeypatch.setenv("GOMOKU_API__CORS_ORIGINS", '["http://example.test"]')
        reset_settings_cache()
        r = TestClient(create_app()).get("/v1/brains", headers={"Origin": "http://example.test"})
        exposed = {h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")}
        assert {"api-version", "deprecation", "sunset", "link", "x-request-id"} <= exposed


class TestDeprecationSignals:
    """``v1`` is not deprecated yet, so the lifecycle path is exercised on a stand-in."""

    OLD = ApiVersion(
        name="v0",
        released=date(2026, 1, 1),
        deprecated=date(2026, 6, 1),
        sunset=date(2026, 12, 31),
        successor="v1",
    )

    def test_header_values_follow_the_rfcs(self):
        headers = {k.decode(): v.decode() for k, v in self.OLD.response_headers()}
        assert headers == {
            "api-version": "v0",
            # RFC 9745: "@" + Unix seconds of 2026-06-01T00:00:00Z.
            "deprecation": "@1780272000",
            # RFC 8594: an HTTP-date.
            "sunset": "Thu, 31 Dec 2026 00:00:00 GMT",
            "link": '</v1>; rel="successor-version"',
        }
        assert self.OLD.status == "deprecated"
        assert V1.status == "stable"

    def test_middleware_marks_only_its_own_prefix(self):
        app = Starlette(
            routes=[
                Route("/v0/ping", lambda request: PlainTextResponse("pong")),
                Route("/v0x/ping", lambda request: PlainTextResponse("pong")),
                Route("/healthz", lambda request: PlainTextResponse("ok")),
            ]
        )
        client = TestClient(VersionHeadersMiddleware(app, versions=(self.OLD, V1)))

        old = client.get("/v0/ping")
        assert old.headers["API-Version"] == "v0"
        assert old.headers["Deprecation"] == "@1780272000"
        assert old.headers["Link"] == '</v1>; rel="successor-version"'
        # Matching is by whole path segment, not string prefix.
        assert "api-version" not in client.get("/v0x/ping").headers
        assert "api-version" not in client.get("/healthz").headers
