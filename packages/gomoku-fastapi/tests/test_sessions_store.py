"""Tests for the in-memory :class:`SessionStore`.

The store sits between the API router and :class:`GameRunner`; these
tests pin the contract the router relies on:

1. UUIDs are unique across creates.
2. ``get`` bumps last-accessed time so an active session doesn't expire.
3. Lazy TTL eviction kicks in on ``get`` and ``create``.
4. ``delete`` is idempotent.
"""

import time

import pytest
from gomoku.models import GameConfig
from gomoku.runner import GameRunner
from gomoku_fastapi.sessions.store import (
    SessionNotFoundError,
    SessionStore,
)


def _runner() -> GameRunner:
    return GameRunner(GameConfig(board_size=9, win_length=5))


class TestCreate:
    def test_creates_unique_ids(self):
        store = SessionStore()
        a = store.create(_runner(), black_brain=None, white_brain=None)
        b = store.create(_runner(), black_brain=None, white_brain=None)
        assert a.id != b.id
        assert len(store) == 2

    def test_records_brain_names(self):
        store = SessionStore()
        s = store.create(_runner(), black_brain="alphabeta", white_brain=None)
        assert s.black_brain == "alphabeta"
        assert s.white_brain is None


class TestGet:
    def test_returns_existing(self):
        store = SessionStore()
        s = store.create(_runner(), black_brain=None, white_brain=None)
        assert store.get(s.id) is s

    def test_unknown_id_raises(self):
        store = SessionStore()
        with pytest.raises(SessionNotFoundError):
            store.get("does-not-exist")

    def test_bumps_last_accessed(self):
        store = SessionStore(ttl_seconds=10_000)
        s = store.create(_runner(), black_brain=None, white_brain=None)
        original = s.last_accessed_at
        time.sleep(0.01)
        store.get(s.id)
        assert s.last_accessed_at > original


class TestEviction:
    def test_expired_session_is_evicted_on_get(self, monkeypatch):
        """A session that's older than the TTL must NOT be returned —
        locking this in guards against the eviction sweep silently
        skipping a long-idle session."""
        store = SessionStore(ttl_seconds=1)
        s = store.create(_runner(), black_brain=None, white_brain=None)
        # Force the session's last-accessed past the TTL window.
        s.last_accessed_at -= 5
        with pytest.raises(SessionNotFoundError):
            store.get(s.id)
        # And the entry is gone — re-querying doesn't find it either.
        assert len(store) == 0

    def test_create_sweeps_stale_entries(self):
        store = SessionStore(ttl_seconds=1)
        old = store.create(_runner(), black_brain=None, white_brain=None)
        old.last_accessed_at -= 5
        # Creating a second session should evict the stale one.
        store.create(_runner(), black_brain=None, white_brain=None)
        assert old.id not in {s.id for s in store._sessions.values()}


class TestDelete:
    def test_returns_true_on_existing(self):
        store = SessionStore()
        s = store.create(_runner(), black_brain=None, white_brain=None)
        assert store.delete(s.id) is True
        assert len(store) == 0

    def test_returns_false_on_missing(self):
        store = SessionStore()
        assert store.delete("never-existed") is False


class TestTempDirCleanup:
    """Per-session recording / diagnostics dirs must not outlive the session."""

    def _session_with_dirs(self, store, tmp_path):
        rec = tmp_path / "rec"
        diag = tmp_path / "diag"
        rec.mkdir()
        diag.mkdir()
        (rec / "game.json").write_text("{}")
        session = store.create(
            _runner(), black_brain=None, white_brain=None, record_dir=str(rec), diag_dir=str(diag)
        )
        return session, rec, diag

    def test_delete_removes_dirs(self, tmp_path):
        store = SessionStore()
        session, rec, diag = self._session_with_dirs(store, tmp_path)
        assert store.delete(session.id) is True
        assert not rec.exists()
        assert not diag.exists()

    def test_expired_get_removes_dirs(self, tmp_path):
        store = SessionStore(ttl_seconds=1)
        session, rec, diag = self._session_with_dirs(store, tmp_path)
        session.last_accessed_at = time.monotonic() - 10
        with pytest.raises(SessionNotFoundError):
            store.get(session.id)
        assert not rec.exists()
        assert not diag.exists()

    def test_create_sweep_removes_dirs_of_stale_sessions(self, tmp_path):
        store = SessionStore(ttl_seconds=1)
        session, rec, diag = self._session_with_dirs(store, tmp_path)
        session.last_accessed_at = time.monotonic() - 10
        store.create(_runner(), black_brain=None, white_brain=None)
        assert not rec.exists()
        assert not diag.exists()

    def test_clear_removes_dirs(self, tmp_path):
        store = SessionStore()
        _, rec, diag = self._session_with_dirs(store, tmp_path)
        store.clear()
        assert len(store) == 0
        assert not rec.exists()
        assert not diag.exists()

    def test_missing_dir_is_ignored(self, tmp_path):
        store = SessionStore()
        session = store.create(
            _runner(), black_brain=None, white_brain=None, record_dir=str(tmp_path / "gone")
        )
        assert store.delete(session.id) is True
