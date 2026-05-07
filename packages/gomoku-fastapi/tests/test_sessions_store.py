"""Tests for the in-memory :class:`SessionStore`.

The store sits between the API router and :class:`GameRunner`; these
tests pin the contract the router relies on:

1. UUIDs are unique across creates.
2. ``get`` bumps last-accessed time so an active session doesn't expire.
3. Lazy TTL eviction kicks in on ``get`` and ``create``.
4. ``delete`` is idempotent.
"""

from __future__ import annotations

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
