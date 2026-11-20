"""Tests for the request-ID middleware and logfire wiring.

Logfire itself is hard to assert on without monkeypatching the SDK
internals; these tests focus on the *observable contract* —
``X-Request-ID`` echo behaviour, ContextVar propagation, and the
fact that ``configure_logfire`` is callable repeatedly without
crashing.
"""

from gomoku_fastapi.observability import (
    RequestIDMiddleware,
    configure_logfire,
    get_request_id,
)


class TestRequestIdHeader:
    def test_generated_when_absent(self, client):
        """No incoming X-Request-ID → server mints one and returns it."""
        response = client.get("/healthz")
        assert response.status_code == 200
        rid = response.headers.get(RequestIDMiddleware.HEADER)
        # UUID4 hex is 32 chars — keep this loose enough that a future
        # change to the ID format (e.g. shorter prefix) doesn't break
        # the assertion, but tight enough to catch "header missing
        # entirely" regressions.
        assert rid and len(rid) >= 8

    def test_echoes_client_supplied_header(self, client):
        rid = "test-correlation-id-123"
        response = client.get("/healthz", headers={RequestIDMiddleware.HEADER: rid})
        assert response.headers.get(RequestIDMiddleware.HEADER) == rid

    def test_truncates_oversized_header(self, client):
        """A pathologically long incoming header must be capped — otherwise
        a malicious client could blow up downstream log lines."""
        oversized = "x" * 500
        response = client.get("/healthz", headers={RequestIDMiddleware.HEADER: oversized})
        echoed = response.headers.get(RequestIDMiddleware.HEADER)
        assert echoed is not None
        assert len(echoed) <= RequestIDMiddleware.MAX_LEN

    def test_blank_header_is_ignored(self, client):
        """An empty (whitespace-only) header should fall back to a fresh UUID
        rather than echo the empty value back."""
        response = client.get("/healthz", headers={RequestIDMiddleware.HEADER: "   "})
        echoed = response.headers.get(RequestIDMiddleware.HEADER)
        assert echoed and echoed.strip() == echoed and len(echoed) >= 8


class TestRequestIdContextVar:
    def test_outside_request_returns_empty(self):
        """Reading the ContextVar with no active request must not raise."""
        # Module-import context — no request ever set the var.
        assert get_request_id() == ""


class TestConfigureLogfire:
    def test_safe_to_call_repeatedly(self, client):
        """Each test builds a fresh app — confirm the configure call
        is idempotent (no warnings, no exceptions on re-invocation)."""
        from gomoku.settings import ApiSettings

        # Two consecutive calls with both networking + console disabled.
        configure_logfire(client.app, ApiSettings(logfire_send=False, logfire_console=False))
        configure_logfire(client.app, ApiSettings(logfire_send=False, logfire_console=False))
