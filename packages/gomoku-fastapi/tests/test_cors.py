"""Tests for the CORS middleware (mounted only when configured).

The conftest's autouse fixture leaves ``cors_origins`` empty, so a
default test client has *no* CORS middleware mounted.  These tests use
per-test ``GOMOKU_API__CORS_ORIGINS`` env-var overrides to assert
both branches of the mount-or-skip decision.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gomoku_fastapi import limits
from gomoku_fastapi.deps import reset_settings_cache
from gomoku_fastapi.main import create_app


@pytest.fixture
def cors_client(monkeypatch) -> TestClient:
    """Client whose CORS allowlist is exactly ``http://example.test``."""
    # pydantic-settings parses JSON-style lists from env vars.
    monkeypatch.setenv("GOMOKU_API__CORS_ORIGINS", '["http://example.test"]')
    reset_settings_cache()
    limits.limiter.reset()
    return TestClient(create_app())


class TestCorsConfigured:
    def test_preflight_returns_allowed_origin(self, cors_client):
        response = cors_client.options(
            "/healthz",
            headers={
                "Origin": "http://example.test",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Preflight returns 200 with the allow-origin header echoed back.
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://example.test"

    def test_actual_request_carries_allow_origin(self, cors_client):
        response = cors_client.get("/healthz", headers={"Origin": "http://example.test"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://example.test"

    def test_disallowed_origin_omits_header(self, cors_client):
        """Browsers enforce CORS — the server just declines to set the
        allow-origin header for a non-allowlisted Origin.  The request
        itself still goes through (CORS is a *response* contract)."""
        response = cors_client.get("/healthz", headers={"Origin": "http://evil.test"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


class TestCorsDisabledByDefault:
    def test_no_cors_headers_when_origins_empty(self, client):
        """Default test fixture: cors_origins=[].  Any Origin should be
        ignored — no allow-origin header at all."""
        response = client.get("/healthz", headers={"Origin": "http://anything.test"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
