"""Gomoku CLI entry point."""

from __future__ import annotations

from pathlib import Path

import click
import logfire
from gomoku.game import Game, InvalidMoveError
from gomoku.models import GameConfig, Stone
from gomoku.settings import GomokuSettings
from gomoku_brain import BRAINS, get_brain_from_settings
from rich.console import Console

from gomoku_cli.display import render_board, render_result
from gomoku_cli.keyboard import KeyboardProvider

console = Console()

_ALL_PROVIDERS = sorted([*BRAINS, "keyboard"])


def _make_provider(name: str, settings: GomokuSettings) -> object:
    if name == "keyboard":
        return KeyboardProvider()
    return get_brain_from_settings(name, settings.brains)


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
@click.option("--board-size", type=click.IntRange(5, 25), default=None, help="Board size (NxN). [default: from config]")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to gomoku.toml config file.",
)
def play(black_name: str, white_name: str, board_size: int | None, config_path: Path | None) -> None:
    """Start a Gomoku game."""
    settings = _load_settings(config_path)

    if board_size is not None:
        settings.game.board_size = board_size

    config = GameConfig(board_size=settings.game.board_size, win_length=settings.game.win_length)
    black = _make_provider(black_name, settings)
    white = _make_provider(white_name, settings)

    console.print(f"\n[bold]Gomoku {config.board_size}x{config.board_size}[/bold]")
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


def _load_settings(config_path: Path | None) -> GomokuSettings:
    """Load settings, optionally from an explicit TOML path."""
    if config_path is not None:
        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        return GomokuSettings(**data)
    return GomokuSettings()


if __name__ == "__main__":
    cli()
