"""Playback navigator — replay a recorded game with arrow-key controls."""

from __future__ import annotations

import sys
import termios
import tty

from gomoku.board import Board
from gomoku.models import Stone
from gomoku.recorder import GameRecord

from gomoku_cli.display import render_board, render_playback_info, render_result


def _read_key() -> str:
    """Read a single keypress from stdin and return a key identifier.

    Returns one of: ``"right"``, ``"left"``, ``"q"``, or ``"other"``.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[C":
                return "right"
            if seq == "[D":
                return "left"
            return "other"
        if ch in ("q", "Q"):
            return "q"
        return "other"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_playback(record: GameRecord, from_move: int = 1) -> None:
    """Interactive playback of a recorded game.

    Navigation:
        → (right arrow) — advance to next move
        ← (left arrow)  — go back to previous move
        q               — quit
    """
    moves = record.moves
    total = len(moves)

    # Clamp from_move to valid range (0 = show empty board before move 1)
    current = max(0, min(from_move, total))

    def _build_board(up_to: int) -> Board:
        """Replay the first *up_to* moves onto a fresh board."""
        board = Board(size=record.game_config.board_size)
        for m in moves[:up_to]:
            board = board.place_stone(m.move, m.stone)
        return board

    render_playback_info(record)

    while True:
        board = _build_board(current)

        last_move = moves[current - 1].move if current > 0 else None
        current_stone: Stone | None = None
        if current < total:
            current_stone = moves[current].stone
        elif current > 0:
            current_stone = moves[current - 1].stone.opponent()

        render_board(board, last_move=last_move, turn=current, current_stone=current_stone)

        # Move info line
        if current > 0:
            m = moves[current - 1]
            print(f"  Move {m.number}/{total}: {m.stone} → {m.move}  (think: {m.think_time_ms}ms)")
        else:
            print(f"  Start — move 0/{total}")

        if current == total:
            render_result(record.result, total)

        print("\n  ← previous  |  → next  |  q quit")

        key = _read_key()
        if key == "q":
            break
        elif key == "right" and current < total:
            current += 1
        elif key == "left" and current > 0:
            current -= 1
