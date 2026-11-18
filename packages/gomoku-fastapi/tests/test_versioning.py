"""API versioning: discovery endpoint, version headers and deprecation signals."""

from __future__ import annotations

from datetime import date, timedelta
from importlib.metadata import version as package_version

import pytest
from fastapi.testclient import TestClient
from gomoku_fastapi.deps import reset_settings_cache
from gomoku_fastapi.main import create_app
from gomoku_fastapi.versioning import (
    LATEST,
    MIN_DEPRECATION_PERIOD,
    V1,
    VERSIONS,
    ApiVersion,
    VersionHeadersMiddleware,
    validate_registry,
)
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
                "openapi": "/v1/openapi.json",
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
        homes = {v.prefix for v in VERSIONS}
        stray = [p for p in paths if not p.startswith(prefixes) and p not in {"/healthz", "/versions", *homes}]
        assert stray == []

    def test_each_version_has_its_own_document(self, client):
        r = client.get("/v1/openapi.json")
        assert r.status_code == 200
        assert r.headers["API-Version"] == "v1"
        doc = r.json()
        assert doc["info"]["title"].endswith("(v1)")
        assert doc["info"]["version"] == package_version("gomoku-fastapi")
        assert all(p == "/v1" or p.startswith("/v1/") for p in doc["paths"])
        assert "/v1/move" in doc["paths"]
        assert "VersionsResponse" not in doc["components"]["schemas"]


class TestVersionHome:
    def test_describes_the_version(self, client):
        r = client.get("/v1")
        assert r.status_code == 200
        assert r.headers["API-Version"] == "v1"
        assert r.json() == client.get("/versions").json()["versions"][0]


class TestUnknownVersion:
    def test_unserved_version_lists_supported_ones(self, client):
        r = client.get("/v2/brains")
        assert r.status_code == 404
        assert r.json() == {
            "detail": "API version v2 is not served. Supported versions: v1.",
            "versions": "/versions",
        }

    def test_other_unknown_paths_keep_the_plain_404(self, client):
        assert client.get("/version2/brains").json() == {"detail": "Not Found"}
        assert client.get("/v0/brains").json() == {"detail": "Not Found"}  # v0 is not a valid name


class TestCorsExposesVersionHeaders:
    def test_browser_can_read_version_and_request_id(self, monkeypatch):
        monkeypatch.setenv("GOMOKU_API__CORS_ORIGINS", '["http://example.test"]')
        reset_settings_cache()
        r = TestClient(create_app()).get("/v1/brains", headers={"Origin": "http://example.test"})
        exposed = {h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")}
        assert {"api-version", "deprecation", "sunset", "link", "x-request-id"} <= exposed


class TestDeprecationSignals:
    """``v1`` is not deprecated yet, so the lifecycle path is exercised on a stand-in."""

    # A stand-in "v1" being phased out in favour of a "v2".
    OLD = ApiVersion(
        name="v1",
        released=date(2026, 1, 1),
        deprecated=date(2026, 6, 1),
        sunset=date(2026, 12, 31),
        successor="v2",
    )
    NEW = ApiVersion(name="v2", released=date(2026, 6, 1))
    LINK = '</versions>; rel="deprecation", </versions>; rel="sunset", </v2>; rel="successor-version"'

    def test_header_values_follow_the_rfcs(self):
        headers = {k.decode(): v.decode() for k, v in self.OLD.response_headers()}
        assert headers == {
            "api-version": "v1",
            # RFC 9745: "@" + Unix seconds of 2026-06-01T00:00:00Z.
            "deprecation": "@1780272000",
            # RFC 8594: an HTTP-date.
            "sunset": "Thu, 31 Dec 2026 00:00:00 GMT",
            # RFC 9745 / RFC 8594 link relations, plus the successor.
            "link": self.LINK,
        }

    def test_status_follows_the_calendar(self):
        """RFC 9745 allows announcing a future deprecation; it isn't deprecated until then."""
        assert self.OLD.status(on=date(2026, 5, 31)) == "stable"
        assert self.OLD.status(on=date(2026, 6, 1)) == "deprecated"
        assert V1.status() == "stable"

    def _app(self) -> TestClient:
        app = Starlette(
            routes=[
                Route("/v1/ping", lambda request: PlainTextResponse("pong")),
                Route("/v1x/ping", lambda request: PlainTextResponse("pong")),
                Route("/healthz", lambda request: PlainTextResponse("ok")),
            ]
        )
        return TestClient(VersionHeadersMiddleware(app, versions=(self.OLD, self.NEW)))

    def test_middleware_marks_only_its_own_prefix(self):
        client = self._app()
        old = client.get("/v1/ping")
        assert old.text == "pong"
        assert old.headers["API-Version"] == "v1"
        assert old.headers["Deprecation"] == "@1780272000"
        assert old.headers["Link"] == self.LINK
        # Matching is by whole path segment, not string prefix.
        assert "api-version" not in client.get("/v1x/ping").headers
        assert "api-version" not in client.get("/healthz").headers

    def test_retired_version_is_gone(self):
        """RFC 8594 section 9: after the sunset, 410 Gone instead of a bare 404."""
        app = Starlette(routes=[Route("/v2/ping", lambda request: PlainTextResponse("pong"))])
        client = TestClient(VersionHeadersMiddleware(app, versions=(self.NEW,), retired=(self.OLD,)))
        r = client.get("/v1/ping")
        assert r.status_code == 410
        assert r.json() == {
            "detail": "API version v1 was retired on 2026-12-31; use v2.",
            "successor": "/v2",
            "versions": "/versions",
        }
        assert r.headers["Sunset"] == "Thu, 31 Dec 2026 00:00:00 GMT"
        assert client.get("/v2/ping").text == "pong"


class TestRegistryRules:
    V2 = ApiVersion(name="v2", released=date(2026, 6, 1))

    @staticmethod
    def _deprecated(**overrides) -> ApiVersion:
        fields = {
            "name": "v1",
            "released": date(2026, 1, 1),
            "deprecated": date(2026, 6, 1),
            "sunset": date(2026, 6, 1) + MIN_DEPRECATION_PERIOD,
            "successor": "v2",
        }
        return ApiVersion(**{**fields, **overrides})

    def test_valid_lifecycle_passes(self):
        validate_registry((self._deprecated(), self.V2), ())
        validate_registry((self.V2,), (self._deprecated(),))
        validate_registry(VERSIONS, ())

    @pytest.mark.parametrize(
        ("served", "message"),
        [
            # AIP-185: major version only, as "v<number>".
            ((ApiVersion("1", date(2026, 1, 1)),), "must be 'v' followed by a major number"),
            ((ApiVersion("v1.1", date(2026, 1, 1)),), "must be 'v' followed by a major number"),
            ((ApiVersion("v0", date(2026, 1, 1)),), "must be 'v' followed by a major number"),
            ((ApiVersion("v2", date(2026, 1, 1)), ApiVersion("v1", date(2026, 1, 1))), "ordered oldest first"),
            ((), "at least one API version"),
        ],
    )
    def test_names_and_order(self, served, message):
        with pytest.raises(ValueError, match=message):
            validate_registry(served, ())

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"sunset": date(2026, 6, 1) + MIN_DEPRECATION_PERIOD - timedelta(days=1)}, "at least 180 days"),
            ({"sunset": date(2026, 5, 1)}, "at least 180 days"),
            ({"sunset": None}, "needs a sunset date and a successor"),
            ({"successor": None}, "needs a sunset date and a successor"),
            ({"successor": "v3"}, "successor must be a newer served version"),
            ({"deprecated": date(2025, 12, 1)}, "deprecated before it was released"),
        ],
    )
    def test_deprecation_rules(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            validate_registry((self._deprecated(**overrides), self.V2), ())

    def test_latest_version_cannot_be_deprecated(self):
        with pytest.raises(ValueError, match="latest version v1 cannot be deprecated"):
            validate_registry((self._deprecated(successor=None),), ())

    def test_sunset_needs_a_deprecation(self):
        undated = ApiVersion("v1", date(2026, 1, 1), sunset=date(2027, 1, 1), successor="v2")
        with pytest.raises(ValueError, match="sunset needs a deprecation date first"):
            validate_registry((undated, self.V2), ())

    def test_served_and_retired_must_not_overlap(self):
        with pytest.raises(ValueError, match="both served and retired"):
            validate_registry((self.V2,), (self._deprecated(name="v2", successor=None),))
