"""In-memory session store with TTL-based eviction.

A :class:`SessionStore` is a thread-safe ``dict[str, Session]`` —
process-local, no persistence.  Each :class:`Session` owns:

* a :class:`gomoku.runner.GameRunner` (the actual game state);
* per-session metadata (creation / last-access timestamps, brain
  names) so ``GET /v1/games/{id}`` can answer without inspecting the
  runner's internals;
* per-session locking so concurrent ``submit_move`` / ``think`` calls
  for the *same* session can't interleave half-applied state.

TTL eviction is lazy — the store doesn't run a background task.
:meth:`SessionStore.get` and :meth:`SessionStore.create` opportunistically
sweep stale entries.  That keeps the module dependency-free (no anyio
task lifecycle to manage) and is plenty for the API's scale.  When
the in-memory store is replaced with a Redis-backed alternative,
expiration moves into Redis and this concern disappears.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field

from gomoku.runner import GameRunner

# Default TTL: an hour of inactivity.  Loose because long-form games
# routinely take longer than a typical request timeout, and we'd
# rather lose session memory to a forgotten browser tab than to
# someone thinking too long about a move.
DEFAULT_TTL_SECONDS = 60 * 60


@dataclass
class Session:
    """One game session with its own lock and TTL bookkeeping.

    Don't construct directly — use :meth:`SessionStore.create`, which
    generates the ID and registers the session atomically.
    """

    id: str
    runner: GameRunner
    black_brain: str | None
    white_brain: str | None
    created_at: float
    last_accessed_at: float
    record_dir: str | None = None
    diag_dir: str | None = None
    # Per-session lock — prevents concurrent ``submit_move`` /
    # ``think`` calls from corrupting the runner's state machine.
    # Held only for the duration of a single state transition; reads
    # don't take it.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def touch(self) -> None:
        """Bump ``last_accessed_at`` so the next sweep doesn't evict us."""
        self.last_accessed_at = time.monotonic()

    def cleanup(self) -> None:
        """Remove the per-session recording / diagnostics temp dirs.

        Best-effort: a missing or undeletable directory is ignored so
        eviction never fails because of the filesystem.
        """
        for path in (self.record_dir, self.diag_dir):
            if path:
                shutil.rmtree(path, ignore_errors=True)


class SessionNotFoundError(LookupError):
    """Raised by :meth:`SessionStore.get` when an ID isn't in the store."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id!r} not found or expired")


class SessionStore:
    """Thread-safe ``dict[str, Session]`` with lazy TTL eviction."""

    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_seconds

    def __len__(self) -> int:
        # No lock — callers that care about exact counts should hold
        # the store lock themselves; tests use this for sanity checks.
        return len(self._sessions)

    def _new_id(self) -> str:
        """Sessions are identified by URL-safe UUIDs.  Hex form, no
        hyphens — keeps URLs short and easy to copy-paste."""
        return uuid.uuid4().hex

    def _sweep_locked(self, now: float) -> list[Session]:
        """Evict sessions whose last access is older than the TTL.

        Caller MUST hold ``self._lock`` and, after releasing it, call
        :meth:`Session.cleanup` on the returned sessions (filesystem work
        stays outside the lock).
        """
        cutoff = now - self.ttl_seconds
        stale = [sid for sid, sess in self._sessions.items() if sess.last_accessed_at < cutoff]
        return [self._sessions.pop(sid) for sid in stale]

    def create(
        self,
        runner: GameRunner,
        *,
        black_brain: str | None,
        white_brain: str | None,
        record_dir: str | None = None,
        diag_dir: str | None = None,
    ) -> Session:
        """Register *runner* under a fresh ID and return the :class:`Session`."""
        now = time.monotonic()
        with self._lock:
            evicted = self._sweep_locked(now)
            session = Session(
                id=self._new_id(),
                runner=runner,
                black_brain=black_brain,
                white_brain=white_brain,
                created_at=now,
                last_accessed_at=now,
                record_dir=record_dir,
                diag_dir=diag_dir,
            )
            self._sessions[session.id] = session
        for old in evicted:
            old.cleanup()
        return session

    def get(self, session_id: str) -> Session:
        """Look up *session_id* and bump its TTL.

        Raises :class:`SessionNotFoundError` when the ID is unknown
        OR when it was found but is past TTL (the lookup eagerly
        evicts stale entries before returning).
        """
        now = time.monotonic()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            expired = session.last_accessed_at < now - self.ttl_seconds
            if expired:
                # Expired between the last access and this lookup.  Evict
                # before raising so the next call doesn't see it either.
                del self._sessions[session_id]
            else:
                session.touch()
        if expired:
            session.cleanup()
            raise SessionNotFoundError(session_id)
        return session

    def delete(self, session_id: str) -> bool:
        """Remove *session_id* from the store.  Returns ``True`` when
        the ID was present, ``False`` for an unknown ID — the caller
        decides whether unknown == 204 or 404.  The session's recording /
        diagnostics temp dirs are removed with it."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.cleanup()
        return True

    def clear(self) -> None:
        """Drop every session and its temp dirs (used on shutdown)."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.cleanup()


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "Session",
    "SessionNotFoundError",
    "SessionStore",
]
