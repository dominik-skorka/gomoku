"""Tests for the FastAPI lifespan handler (graceful shutdown)."""

from fastapi.testclient import TestClient
from gomoku_fastapi.deps import get_session_store
from gomoku_fastapi.main import create_app


class TestLifespan:
    def test_shutdown_clears_session_store(self):
        """Sessions live across requests within an app's lifetime, but
        the lifespan handler must drop them on shutdown so a restart
        doesn't carry stale state forward.

        TestClient's context-manager form (``with TestClient(app) as
        client:``) is what triggers startup + shutdown — outside the
        ``with`` block it's just a request maker."""
        app = create_app()
        with TestClient(app) as client:
            # Create a session inside the app's lifetime.
            r = client.post("/v1/games", json={"board_size": 9})
            assert r.status_code == 201
            assert len(get_session_store()) == 1
        # Lifespan shutdown ran on context exit.
        assert len(get_session_store()) == 0
