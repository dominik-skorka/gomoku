"""Engine process for ``gomoku match`` — one brain behind a JSON-lines pipe.

The match runner starts this file *by path* with ``PYTHONPATH`` pointing
at the source tree of the build under test, so it must import nothing
from ``gomoku_cli`` and only the long-stable public API of ``gomoku``
and ``gomoku_brain``.

Protocol (one JSON object per line):

* on start the worker writes ``{"ready": true, "brain": <name>}``;
* request ``{"moves": [[row, col], ...]}`` — the game so far, Black first;
* reply ``{"move": [row, col], "ms": <think time>}``;
* an empty line or EOF ends the process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


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
    from gomoku_brain import get_brain_from_settings

    brain = get_brain_from_settings(args.brain, GomokuSettings().brains)

    def position(moves: list[list[int]]) -> tuple[Board, Stone]:
        board = Board(size=args.size)
        stone = Stone.BLACK
        for row, col in moves:
            board = board.place_stone(Move(row=row, col=col), stone)
            stone = stone.opponent()
        return board, stone

    # Warm up (JIT compilation, lazy imports) so think times exclude it.
    centre = args.size // 2
    warm = [[centre, centre], [centre, centre + 1], [centre + 1, centre], [centre - 1, centre + 1]]
    warm += [[centre + 1, centre + 1], [centre - 1, centre]]
    brain.get_move(*position(warm))

    pipe.write(json.dumps({"ready": True, "brain": args.brain}) + "\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        board, stone = position(json.loads(line)["moves"])
        started = time.perf_counter()
        move = brain.get_move(board, stone)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        pipe.write(json.dumps({"move": [move.row, move.col], "ms": elapsed_ms}) + "\n")


if __name__ == "__main__":
    main()
