"""Shared fixtures for ``gomoku-fastapi`` tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gomoku_fastapi import limits
from gomoku_fastapi.deps import reset_session_store, reset_settings_cache
from gomoku_fastapi.main import create_app


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Each test gets its own CWD, settings cache, session store, and limiter state.

    The settings module walks up from CWD looking for ``gomoku.toml``;
    without this fixture we'd pick up the workspace root's TOML and
    test results would depend on the dev's working directory.  Also
    drops any inherited ``GOMOKU_CONFIG`` env var so tests don't see a
    config the user set in their shell.

    Resetting the session store between tests prevents one test's
    sessions from being visible to the next — important because the
    store is process-cached at module level.

    Resetting the slowapi limiter is critical: its in-memory backend
    counts hits across the whole process lifetime, so without a reset
    a test that fires 30 requests would push the next test over the
    per-IP cap and produce sporadic 429s that have nothing to do with
    what's being tested.

    Disables logfire console output by default — individual tests can
    re-enable via ``GOMOKU_API__LOGFIRE_CONSOLE=true`` env var when
    they're specifically exercising the observability path.
    """
    monkeypatch.delenv("GOMOKU_CONFIG", raising=False)
    # Permissive rate limits so the suite isn't impacted by
    # accidental triggering — individual tests that need to *exercise*
    # the limiter set their own gomoku.toml with tighter numbers.
    monkeypatch.setenv("GOMOKU_API__RATE_LIMIT_DEFAULT", "")
    monkeypatch.setenv("GOMOKU_API__RATE_LIMIT_COMPUTE", "")
    monkeypatch.setenv("GOMOKU_API__LOGFIRE_CONSOLE", "false")
    monkeypatch.setenv("GOMOKU_API__LOGFIRE_SEND", "false")
    # ``pyproject.toml`` marks the temp dir as a project root so the
    # walk-up doesn't escape into the dev's filesystem.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()
    reset_session_store()
    limits.limiter.reset()
    yield
    reset_settings_cache()
    reset_session_store()
    limits.limiter.reset()


@pytest.fixture
def client() -> TestClient:
    """A fresh ``TestClient`` per test, with a fresh app underneath."""
    return TestClient(create_app())
