"""Tests for the slowapi-backed per-IP rate limit on compute endpoints.

The conftest sets the limits to ``""`` (permissive) so the rest of the
suite isn't affected; these tests write a per-test ``gomoku.toml`` that
*does* configure tight limits, then fire enough requests to trip them.

What we lock in:

1. The compute endpoints (``/v1/move`` and ``/v1/games/{id}/think``)
   honour the ``rate_limit_compute`` setting.
2. A 429 response carries the structured ``detail.error`` envelope
   every other 4xx in this API uses (``rate_limit_exceeded``).
3. A non-compute endpoint (``/healthz``) is governed by the default
   limit, not the compute one — so it tolerates more requests.
4. Empty-string limits truly disable the tier (the default in tests).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gomoku.board import Board
from gomoku.models import Stone
from gomoku_fastapi import limits
from gomoku_fastapi.deps import reset_settings_cache
from gomoku_fastapi.main import create_app


def _move_payload() -> dict:
    return {
        "board": Board(size=9).model_dump(mode="json"),
        "side_to_move": int(Stone.BLACK),
        "brain": "random",
    }


@pytest.fixture
def tight_compute_client(tmp_path, monkeypatch) -> TestClient:
    """A client whose compute endpoints are limited to 2/minute.

    Two requests should succeed; the third must hit a 429.  Using a
    very low cap keeps the test fast and deterministic.
    """
    # Override the env vars set by the autouse conftest fixture so
    # the limit is *enforced*, not permissive.
    monkeypatch.setenv("GOMOKU_API__RATE_LIMIT_COMPUTE", "2/minute")
    monkeypatch.setenv("GOMOKU_API__RATE_LIMIT_DEFAULT", "")
    reset_settings_cache()
    limits.limiter.reset()
    return TestClient(create_app())


@pytest.fixture
def tight_default_client(tmp_path, monkeypatch) -> TestClient:
    """A client whose *default* limit is 2/minute (so /healthz trips)."""
    monkeypatch.setenv("GOMOKU_API__RATE_LIMIT_DEFAULT", "2/minute")
    monkeypatch.setenv("GOMOKU_API__RATE_LIMIT_COMPUTE", "")
    reset_settings_cache()
    limits.limiter.reset()
    return TestClient(create_app())


class TestComputeLimitTrips:
    def test_third_call_to_move_is_429(self, tight_compute_client):
        # Two allowed...
        for _ in range(2):
            r = tight_compute_client.post("/v1/move", json=_move_payload())
            assert r.status_code == 200
        # ...third blocked.
        r = tight_compute_client.post("/v1/move", json=_move_payload())
        assert r.status_code == 429
        body = r.json()
        assert body["detail"]["error"] == "rate_limit_exceeded"
        # Retry-After header lets a polite client back off.
        assert "Retry-After" in r.headers

    def test_think_endpoint_also_rate_limited(self, tight_compute_client):
        """``/v1/games/{id}/think`` shares the compute tier with /v1/move,
        so they must count against the same per-IP bucket."""
        # Burn the budget on /v1/move...
        for _ in range(2):
            tight_compute_client.post("/v1/move", json=_move_payload())
        # ...then confirm /think is also blocked even though we never
        # called it directly.  Need a session first.
        created = tight_compute_client.post(
            "/v1/games", json={"board_size": 9, "black": {"brain": "random"}}
        )
        assert created.status_code == 201
        gid = created.json()["game_id"]
        r = tight_compute_client.post(f"/v1/games/{gid}/think", json={})
        assert r.status_code == 429


class TestDefaultLimitTrips:
    def test_healthz_under_tight_default_limit(self, tight_default_client):
        for _ in range(2):
            assert tight_default_client.get("/healthz").status_code == 200
        assert tight_default_client.get("/healthz").status_code == 429

    def test_versioned_endpoint_under_tight_default_limit(self, tight_default_client):
        """The default limit must reach routes nested in included routers
        (v1 includes a router per resource), not only top-level ones."""
        for _ in range(2):
            assert tight_default_client.get("/v1/brains").status_code == 200
        assert tight_default_client.get("/v1/brains").status_code == 429


class TestPermissiveByDefault:
    def test_many_calls_succeed_when_limits_disabled(self, client):
        """The conftest sets both limits to ``""`` — confirm that
        actually disables enforcement (no 429s in a tight loop)."""
        for _ in range(10):
            assert client.post("/v1/move", json=_move_payload()).status_code == 200
