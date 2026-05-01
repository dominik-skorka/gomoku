"""FastAPI dependencies — settings + brain registry resolver.

Wires :class:`gomoku.settings.GomokuSettings` (with its walk-up TOML
discovery) and :func:`gomoku_brain.get_brain_from_settings` into the
request lifecycle.  Endpoints that need either declare a
``Depends(get_settings)`` / ``Depends(get_brain_factory)`` parameter and
get a per-request value injected.

The settings instance is cached at module level so we don't re-walk the
filesystem on every request.  Tests can clear the cache via
:func:`reset_settings_cache`.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from gomoku.protocol import MoveProvider
from gomoku.settings import GomokuSettings
from gomoku_brain import (
    BRAINS,
    get_brain_from_settings,
    resolve_brain,
)

_settings: GomokuSettings | None = None


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


SettingsDep = Annotated[GomokuSettings, Depends(get_settings)]


def build_brain(
    name: str,
    overrides: dict[str, Any] | None,
    settings: GomokuSettings,
) -> MoveProvider:
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

    # ``get_brain_from_settings`` would re-resolve the alias — but we already
    # have the validated kwargs, so instantiate the class directly.
    return BRAINS[brain_type].brain_cls(**kwargs)


__all__ = [
    "SettingsDep",
    "build_brain",
    # Re-exported for convenience so endpoint code only imports from one place.
    "get_brain_from_settings",
    "get_settings",
    "reset_settings_cache",
]
