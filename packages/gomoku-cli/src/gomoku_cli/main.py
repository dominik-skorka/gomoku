"""Gomoku CLI entry point."""

from __future__ import annotations

import click
import logfire
from gomoku.game import Game, InvalidMoveError
from gomoku.models import GameConfig, Stone
from gomoku_brain import BRAINS, get_brain
from rich.console import Console

from gomoku_cli.display import render_board, render_result
from gomoku_cli.keyboard import KeyboardProvider

console = Console()

_ALL_PROVIDERS = sorted([*BRAINS, "keyboard"])


def _make_provider(name: str, depth: int) -> object:
    if name == "keyboard":
        return KeyboardProvider()
    return get_brain(name, **_brain_kwargs(name, depth))


def _brain_kwargs(name: str, depth: int) -> dict[str, object]:
    if name == "random":
        return {}
    return {"depth": depth}


@click.group()
def cli() -> None:
    """Gomoku (Five in a Row) — play from the terminal."""
    logfire.configure(send_to_logfire=False)


@cli.command()
@click.option(
    "--black",
    "black_name",
    type=click.Choice(_ALL_PROVIDERS, case_sensitive=False),
    default="keyboard",
    show_default=True,
    help="Move provider for Black.",
)
@click.option(
    "--white",
    "white_name",
    type=click.Choice(_ALL_PROVIDERS, case_sensitive=False),
    default="random",
    show_default=True,
    help="Move provider for White.",
)
@click.option("--board-size", type=click.IntRange(5, 25), default=15, show_default=True, help="Board size (NxN).")
@click.option("--depth", type=click.IntRange(1, 10), default=3, show_default=True, help="Search depth for AI brains.")
def play(black_name: str, white_name: str, board_size: int, depth: int) -> None:
    """Start a Gomoku game."""
    config = GameConfig(board_size=board_size)
    black = _make_provider(black_name, depth)
    white = _make_provider(white_name, depth)

    console.print(f"\n[bold]Gomoku {board_size}x{board_size}[/bold]")
    console.print(f"  Black: {black_name}  |  White: {white_name}\n")

    game = Game(config, black, white)  # type: ignore[arg-type]
    render_board(game.board, current_stone=Stone.BLACK)

    while True:
        try:
            result = game.play_turn()
        except InvalidMoveError as e:
            console.print(f"[red]Error:[/red] {e}")
            break

        _last_stone, last_move = game.move_history[-1]
        turn = len(game.move_history)

        render_board(game.board, last_move=last_move, turn=turn, current_stone=game.current_stone)

        if result is not None:
            render_result(result, turn)
            break


if __name__ == "__main__":
    cli()
