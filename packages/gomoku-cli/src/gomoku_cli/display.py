"""Rich board display for the Gomoku CLI."""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import GameResult, Move, Stone
from gomoku.recorder import GameRecord
from rich.console import Console
from rich.text import Text

_STONE_CHARS = {
    Stone.EMPTY: "·",
    Stone.BLACK: "●",
    Stone.WHITE: "○",
}

console = Console()


def render_board(
    board: Board,
    last_move: Move | None = None,
    turn: int = 0,
    current_stone: Stone | None = None,
    winning_moves: list[Move] | None = None,
) -> None:
    """Print the board to the console with labels and highlighting."""
    size = board.size
    win_cells: set[tuple[int, int]] = {(m.row, m.col) for m in winning_moves} if winning_moves else set()

    # Header
    header = Text("   ")
    for c in range(size):
        header.append(f" {chr(ord('A') + c)}", style="bold cyan")
    console.print(header)

    for r in range(size):
        row_label = f"{r + 1:2d} "
        line = Text(row_label, style="bold cyan")

        for c in range(size):
            stone = board.at(r, c)
            char = _STONE_CHARS[stone]

            if (r, c) in win_cells:
                line.append(f" {char}", style="bold bright_yellow")
            elif last_move and last_move.row == r and last_move.col == c:
                if stone is Stone.BLACK:
                    line.append(f" {char}", style="bold red")
                else:
                    line.append(f" {char}", style="bold green")
            elif stone is Stone.BLACK:
                line.append(f" {char}", style="white")
            elif stone is Stone.WHITE:
                line.append(f" {char}", style="bright_white")
            else:
                line.append(f" {char}", style="dim")

        line.append(f"  {r + 1}", style="bold cyan")
        console.print(line)

    # Footer column labels
    footer = Text("   ")
    for c in range(size):
        footer.append(f" {chr(ord('A') + c)}", style="bold cyan")
    console.print(footer)


def render_result(result: GameResult, total_turns: int) -> None:
    """Print the game result."""

    if result.is_draw:
        console.print(f"\n[bold yellow]Draw![/bold yellow] Board filled after {total_turns} moves.")
    else:
        console.print(f"\n[bold green]{result.winner} wins[/bold green] after {total_turns} moves!")
        if result.winning_moves:
            moves_str = ", ".join(str(m) for m in result.winning_moves)
            console.print(f"Winning line: {moves_str}", style="dim")


def render_playback_info(record: GameRecord) -> None:
    """Print a header summarising the recorded game."""
    cfg = record.game_config
    console.print(f"\n[bold]Gomoku {cfg.board_size}x{cfg.board_size}[/bold] \u2014 Playback")

    for p in record.players:
        settings_str = ", ".join(f"{k}={v}" for k, v in p.settings.items()) if p.settings else ""
        label = f"{p.provider}" + (f" ({settings_str})" if settings_str else "")
        console.print(f"  {p.stone}: {label}")

    started = record.started_at.strftime("%Y-%m-%d %H:%M:%S")
    ended = record.ended_at.strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"  Played: {started} → {ended}")

    total = len(record.moves)
    outcome = "Draw" if record.result.is_draw else f"{record.result.winner} wins"
    console.print(f"  Moves: {total}  |  Result: {outcome}\n")
