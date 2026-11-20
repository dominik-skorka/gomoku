"""REST API for the Gomoku engine.

The package exposes ``Player`` brains and the static evaluator over
HTTP.  The app is constructed by :func:`gomoku_fastapi.main.create_app`;
deployments use the ``gomoku-api`` console script (see
:func:`gomoku_fastapi.main.run`).

This iteration ships the *stateless* surface only: ``GET /healthz``,
``GET /v1/brains``, ``POST /v1/move``.  Stateful sessions, diagnostics
streaming, and analyze upload arrive in later iterations — see
``docs/ARCHITECTURE.md`` for the roadmap.
"""

from gomoku_fastapi.main import create_app

__all__ = ["create_app"]
