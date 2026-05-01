"""Shared fixtures for ``gomoku-fastapi`` tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gomoku_fastapi.deps import reset_settings_cache
from gomoku_fastapi.main import create_app


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Each test gets its own CWD and a fresh settings cache.

    The settings module walks up from CWD looking for ``gomoku.toml``;
    without this fixture we'd pick up the workspace root's TOML and
    test results would depend on the dev's working directory.  Also
    drops any inherited ``GOMOKU_CONFIG`` env var so tests don't see a
    config the user set in their shell.
    """
    monkeypatch.delenv("GOMOKU_CONFIG", raising=False)
    # ``pyproject.toml`` marks the temp dir as a project root so the
    # walk-up doesn't escape into the dev's filesystem.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client() -> TestClient:
    """A fresh ``TestClient`` per test, with a fresh app underneath."""
    return TestClient(create_app())
