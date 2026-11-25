"""FastAPI dependencies — settings, brain registry, session store.

Wires :class:`gomoku.settings.GomokuSettings` (with its walk-up TOML
discovery), :func:`gomoku_brain.get_brain_from_settings`, and the
in-memory :class:`SessionStore` into the request lifecycle.  Endpoints
declare an ``Annotated[..., Depends(...)]`` parameter and get a
per-request value injected.

The settings instance and the session store are both cached at module
level so we don't re-walk the filesystem or lose stateful sessions on
every request.  Tests can clear the cache via :func:`reset_settings_cache`
and :func:`reset_session_store`.
"""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from gomoku.interfaces import Player
from gomoku.settings import GomokuSettings
from gomoku_brain import (
    BRAINS,
    get_brain_from_settings,
    resolve_brain,
)

from gomoku_fastapi.sessions import SessionStore

_settings: GomokuSettings | None = None
_session_store: SessionStore | None = None


def get_settings() -> GomokuSettings:
    """Return the process-wide :class:`GomokuSettings` (cached)."""
    global _settings
    if _settings is None:
        _settings = GomokuSettings()
    return _settings


def reset_settings_cache() -> None:
    """Drop the cached settings — used by tests that monkey-patch env vars."""
    global _settings
    _settings = None


def get_session_store() -> SessionStore:
    """Return the process-wide :class:`SessionStore` (cached)."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


def reset_session_store() -> None:
    """Drop the cached session store — used by tests so each test
    starts with an empty store and no inherited session state."""
    global _session_store
    _session_store = None


SettingsDep = Annotated[GomokuSettings, Depends(get_settings)]
SessionStoreDep = Annotated[SessionStore, Depends(get_session_store)]


def _cap_time_budget(brain_type: str, kwargs: dict[str, Any], cap: float) -> dict[str, Any]:
    """Clamp the search time budget in *kwargs* to *cap* seconds (``cap > 0``).

    ``time_limit_s == 0`` means "unlimited" for alpha-beta, so it is
    replaced by the cap rather than min()-ed.  The threat brain carries
    its alpha-beta tuning under ``fallback``; when none is configured an
    entry holding only the cap is added (the fallback keeps the
    top-level depth in that case).
    """
    kwargs = dict(kwargs)

    def capped(limit: float) -> float:
        return min(limit, cap) if limit > 0 else cap

    if brain_type == "alphabeta":
        kwargs["time_limit_s"] = capped(kwargs.get("time_limit_s", 0.0))
    elif brain_type == "threat":
        fallback = dict(kwargs.get("fallback") or {})
        fallback["time_limit_s"] = capped(fallback.get("time_limit_s", 0.0))
        kwargs["fallback"] = fallback
        if "pns_time_s" in kwargs:
            kwargs["pns_time_s"] = min(kwargs["pns_time_s"], cap)
    return kwargs


def effective_brain_settings(name: str, settings: GomokuSettings) -> tuple[str, dict[str, Any]]:
    """The brain type and the full settings a move by *name* uses without overrides.

    Resolves *name* (a registry key or a ``[brains.<name>]`` alias) the way
    :func:`build_brain` does, fills every setting the alias leaves out with
    the brain's default, and applies the server's time cap
    (``[api].max_time_limit_s``), so the values are the ones a move really
    runs with.  Raises the same 404 / 422 as :func:`build_brain`.
    """
    if name not in BRAINS and name not in settings.brains.aliases():
        available = sorted(set(BRAINS) | set(settings.brains.aliases()))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "unknown_brain", "name": name, "available": available},
        )
    try:
        brain_type, kwargs = resolve_brain(name, settings.brains)
        kwargs = BRAINS[brain_type].settings_cls(**kwargs).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "brain_resolution_failed", "message": str(exc)},
        ) from exc
    cap = settings.api.max_time_limit_s
    if cap > 0:
        kwargs = _cap_time_budget(brain_type, kwargs, cap)
    return brain_type, kwargs


def build_brain(
    name: str,
    overrides: dict[str, Any] | None,
    settings: GomokuSettings,
) -> Player:
    """Resolve ``name`` against the registry and instantiate the brain.

    *name* is either a built-in brain key (``"alphabeta"``, ``"threat"``,
    …) or a TOML alias from ``[brains.<name>]``.  *overrides* are
    request-time tweaks merged on top of the resolved settings — they
    are validated against the brain's pydantic settings model so a
    malformed knob produces a clean 422 instead of a 500.

    Raises :class:`HTTPException` (404 / 422) on any resolution failure
    so the FastAPI error handler renders it as a normal API error
    rather than a stack trace.
    """
    if name not in BRAINS and name not in settings.brains.aliases():
        available = sorted(set(BRAINS) | set(settings.brains.aliases()))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "unknown_brain",
                "name": name,
                "available": available,
            },
        )

    try:
        brain_type, kwargs = resolve_brain(name, settings.brains)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "brain_resolution_failed", "message": str(exc)},
        ) from exc

    if overrides:
        # Merge + revalidate via the brain's settings model so out-of-range
        # values (e.g. depth=999) fail at the API edge with a 422.
        spec = BRAINS[brain_type]
        try:
            kwargs = spec.settings_cls(**{**kwargs, **overrides}).model_dump()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"error": "invalid_brain_overrides", "message": str(exc)},
            ) from exc

    cap = settings.api.max_time_limit_s
    if cap > 0:
        kwargs = _cap_time_budget(brain_type, kwargs, cap)

    # ``get_brain_from_settings`` would re-resolve the alias — but we already
    # have the validated kwargs, so instantiate the class directly.
    return BRAINS[brain_type].brain_cls(**kwargs)


__all__ = [
    "SessionStoreDep",
    "SettingsDep",
    "build_brain",
    "effective_brain_settings",
    # Re-exported for convenience so endpoint code only imports from one place.
    "get_brain_from_settings",
    "get_session_store",
    "get_settings",
    "reset_session_store",
    "reset_settings_cache",
]
