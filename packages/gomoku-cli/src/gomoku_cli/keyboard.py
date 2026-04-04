"""Keyboard input MoveProvider — reads moves from the terminal."""

from __future__ import annotations

import re

from gomoku.board import Board
from gomoku.models import Move, Stone
from rich.console import Console
from rich.prompt import Prompt

_MOVE_PATTERN = re.compile(r"^([A-Za-z])(\d{1,2})$")

console = Console()


class KeyboardProvider:
    """MoveProvider that reads moves from the keyboard using Rich prompts."""

    @property
    def name(self) -> str:
        return "keyboard"

    def get_move(self, board: Board, stone: Stone) -> Move:
        while True:
            raw = Prompt.ask(f"[bold]{stone}[/bold] — enter move (e.g. H8)")
            raw = raw.strip()

            match = _MOVE_PATTERN.match(raw)
            if not match:
                console.print("[red]Invalid format.[/red] Use letter + number, e.g. H8")
                continue

            col = ord(match.group(1).upper()) - ord("A")
            row = int(match.group(2)) - 1

            move = Move(row=row, col=col)
            if not board.is_valid_move(move):
                if not (0 <= move.row < board.size and 0 <= move.col < board.size):
                    console.print(f"[red]Out of bounds.[/red] Board is {board.size}x{board.size}")
                else:
                    console.print("[red]Cell already occupied.[/red] Choose another.")
                continue

            return move
