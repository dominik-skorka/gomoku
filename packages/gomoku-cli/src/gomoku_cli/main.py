"""Gomoku CLI entry point."""

from __future__ import annotations

from pathlib import Path

import click
import logfire
from gomoku.game import Game, InvalidMoveError
from gomoku.models import GameConfig, Stone
from gomoku.recorder import GameRecorder, PlayerInfo, load_record
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


def _player_settings(name: str, settings: GomokuSettings) -> dict:
    """Extract the brain settings dict for a given provider name."""
    if name == "keyboard":
        return {}
    section = getattr(settings.brains, name, None)
    if section is None:
        return {}
    return section.model_dump()


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
@click.option("--record/--no-record", default=None, help="Record the game to a JSON file.")
@click.option(
    "--record-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for recordings. [default: from config]",
)
def play(
    black_name: str,
    white_name: str,
    board_size: int | None,
    config_path: Path | None,
    record: bool | None,
    record_dir: Path | None,
) -> None:
    """Start a Gomoku game."""
    settings = _load_settings(config_path)

    if board_size is not None:
        settings.game.board_size = board_size

    config = GameConfig(board_size=settings.game.board_size, win_length=settings.game.win_length)
    black = _make_provider(black_name, settings)
    white = _make_provider(white_name, settings)

    # Determine whether to record
    do_record = record if record is not None else settings.recording.enabled
    rec_dir = record_dir if record_dir is not None else settings.recording.directory

    recorder: GameRecorder | None = None
    on_move = None
    if do_record:
        players = [
            PlayerInfo(stone=Stone.BLACK, provider=black_name, settings=_player_settings(black_name, settings)),
            PlayerInfo(stone=Stone.WHITE, provider=white_name, settings=_player_settings(white_name, settings)),
        ]
        recorder = GameRecorder(game_config=config, players=players, directory=rec_dir)
        on_move = recorder.record_move

    console.print(f"\n[bold]Gomoku {config.board_size}x{config.board_size}[/bold]")
    console.print(f"  Black: {black_name}  |  White: {white_name}")
    if do_record:
        console.print(f"  Recording to: {rec_dir}/")
    console.print()

    game = Game(config, black, white, on_move=on_move)  # type: ignore[arg-type]
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
            if recorder is not None:
                recorder.finalize(result)
                console.print(f"\n[bold cyan]Game recorded:[/bold cyan] {recorder.path}")
            break


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--from-move",
    "from_move",
    type=int,
    default=1,
    show_default=True,
    help="Start playback from this move number.",
)
def playback(file: Path, from_move: int) -> None:
    """Replay a recorded game from a JSON file.

    Navigate with arrow keys: ← previous, → next, q quit.
    """
    from gomoku_cli.playback import run_playback

    record = load_record(file)
    run_playback(record, from_move=from_move)


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
