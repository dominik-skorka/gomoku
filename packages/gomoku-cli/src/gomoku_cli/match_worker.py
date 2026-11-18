"""Worker process for ``gomoku match`` and ``gomoku review``: one brain over JSON lines.

Run by :mod:`gomoku_cli.match` with ``PYTHONPATH`` pointing at the source
tree of the revision under test, so every import below resolves to that
revision's code.  Protocol, one JSON object per line:

* start-up: the worker prints ``{"ready": true, "brain": NAME}``
* ``{"moves": [[row, col], ...]}`` asks for a move; optional keys
  ``"overrides"`` (settings merged into the brain's own, nested tables
  included) and ``"diagnostics": true`` (also return ``"path"``, the
  brain's decision path).  Reply: ``{"move": [row, col], "ms": ...}``.
* ``{"op": "settings"}`` returns ``{"type": TYPE, "settings": {...}}``.
* ``{"op": "status", "moves": [...], "nodes": N, "time_s": T}`` runs a
  threat-space proof-number search for the side to move (or for
  ``"stone"``, 1 = black, 2 = white, when given) and returns
  ``{"status": "PROVED"|"DISPROVED"|"UNKNOWN", "move": [r, c] | null}``.

A request that fails gets ``{"error": MESSAGE}`` and the worker
carries on.  An empty line ends the session.  Everything a brain or its logging
prints goes to stderr, so stdout carries only the protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return *base* with *overrides* applied; nested tables merge key by key."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict):
            current = merged.get(key)
            merged[key] = merge(current if isinstance(current, dict) else {}, value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", required=True)
    parser.add_argument("--size", type=int, default=15)
    args = parser.parse_args()

    # Keep the protocol on the real stdout and send everything else a
    # brain or its logging might print to stderr.
    pipe = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    from gomoku.board import Board
    from gomoku.models import Move, Stone
    from gomoku.settings import GomokuSettings
    from gomoku_brain import BRAINS, resolve_brain

    brain_type, base_settings = resolve_brain(args.brain, GomokuSettings().brains)
    brains: dict[str, Any] = {}

    def brain_for(overrides: dict[str, Any]) -> Any:
        key = json.dumps(overrides, sort_keys=True)
        if key not in brains:
            spec = BRAINS[brain_type]
            settings = spec.settings_cls(**merge(base_settings, overrides)).model_dump()
            brains[key] = spec.brain_cls(**settings)
        return brains[key]

    def position(moves: list[list[int]]) -> tuple[Board, Stone]:
        board = Board(size=args.size)
        stone = Stone.BLACK
        for row, col in moves:
            board = board.place_stone(Move(row=row, col=col), stone)
            stone = stone.opponent()
        return board, stone

    def think(request: dict[str, Any]) -> dict[str, Any]:
        brain = brain_for(request.get("overrides") or {})
        board, stone = position(request["moves"])
        diag = _start_diagnostics(stone) if request.get("diagnostics") else None
        started = time.perf_counter()
        try:
            move = brain.get_move(board, stone)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if diag is not None:
                _stop_diagnostics()
        reply: dict[str, Any] = {"move": [move.row, move.col], "ms": elapsed_ms}
        if diag is not None:
            reply["path"] = diag.decision_path or None
        return reply

    def status(request: dict[str, Any]) -> dict[str, Any]:
        board, stone = position(request["moves"])
        if request.get("stone") is not None:  # ask about the side *not* to move
            stone = Stone(request["stone"])
        nodes, time_s = int(request.get("nodes", 200_000)), float(request.get("time_s", 30.0))
        try:
            from gomoku_brain.pns import pns_threat_status
        except ImportError:  # revisions before PNSStatus: a miss is only "unknown"
            from gomoku_brain.pns import pns_threat_search

            found = pns_threat_search(board, stone, max_nodes=nodes, max_time_s=time_s)
            name = "PROVED" if found is not None else "UNKNOWN"
        else:
            state, found = pns_threat_status(board, stone, max_nodes=nodes, max_time_s=time_s)
            name = state.name
        return {"status": name, "move": None if found is None else [found.row, found.col]}

    # Warm up (JIT compilation, lazy imports) so think times exclude it.
    centre = args.size // 2
    warm = [[centre, centre], [centre, centre + 1], [centre + 1, centre], [centre - 1, centre + 1]]
    warm += [[centre + 1, centre + 1], [centre - 1, centre]]
    brain_for({}).get_move(*position(warm))

    pipe.write(json.dumps({"ready": True, "brain": args.brain}) + "\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        try:
            request = json.loads(line)
            op = request.get("op", "move")
            if op == "move":
                reply = think(request)
            elif op == "settings":
                reply = {"type": brain_type, "settings": base_settings}
            elif op == "status":
                reply = status(request)
            else:
                reply = {"error": f"unknown op {op!r}"}
        except Exception as exc:  # a bad request must not take the engine down
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        pipe.write(json.dumps(reply) + "\n")


def _start_diagnostics(stone: Any) -> Any:
    """Activate per-move diagnostics if this revision has them."""
    try:
        from gomoku.diagnostics import MoveDiagnostics, set_move_diagnostics
    except ImportError:
        return None
    diag = MoveDiagnostics(move_number=1, stone=stone)
    set_move_diagnostics(diag)
    return diag


def _stop_diagnostics() -> None:
    from gomoku.diagnostics import set_move_diagnostics

    set_move_diagnostics(None)


if __name__ == "__main__":
    main()
