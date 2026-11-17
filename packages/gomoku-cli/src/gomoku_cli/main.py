"""Gomoku CLI entry point."""

from __future__ import annotations

import time
from pathlib import Path

import click
import logfire
from gomoku.game import Game, InvalidMoveError
from gomoku.models import GameConfig, Move, Stone
from gomoku.recorder import GameRecorder, PlayerInfo, load_record
from gomoku.settings import GomokuSettings, KeyboardBrainSettings
from gomoku_brain import available_brains, get_brain_from_settings, register_brain, resolve_brain
from rich.console import Console

from gomoku_cli.display import render_board, render_result
from gomoku_cli.keyboard import KeyboardProvider

console = Console()

# Register the keyboard provider so it's resolved like any other brain.
register_brain("keyboard", KeyboardProvider, KeyboardBrainSettings)


def _make_provider(name: str, settings: GomokuSettings) -> object:
    return get_brain_from_settings(name, settings.brains)


def _player_settings(name: str, settings: GomokuSettings) -> dict:
    """Extract the brain settings dict for a given provider name."""
    _brain_type, kwargs = resolve_brain(name, settings.brains)
    return kwargs


@click.group()
def cli() -> None:
    """Gomoku (Five in a Row) — play from the terminal."""
    logfire.configure(send_to_logfire=False, inspect_arguments=False)


@cli.command()
@click.option(
    "--black",
    "black_name",
    type=str,
    default="keyboard",
    show_default=True,
    help="Move provider for Black (brain name or alias from gomoku.toml).",
)
@click.option(
    "--white",
    "white_name",
    type=str,
    default="random",
    show_default=True,
    help="Move provider for White (brain name or alias from gomoku.toml).",
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
@click.option("--diagnostics/--no-diagnostics", default=None, help="Record per-move diagnostics to a .diag.jsonl file.")
@click.option(
    "--diagnostics-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for diagnostics files. [default: from config]",
)
@click.option(
    "--opening-book",
    "opening_book_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to an opening book JSON file (overrides brain config).",
)
def play(
    black_name: str,
    white_name: str,
    board_size: int | None,
    config_path: Path | None,
    record: bool | None,
    record_dir: Path | None,
    diagnostics: bool | None,
    diagnostics_dir: Path | None,
    opening_book_path: Path | None,
) -> None:
    """Start a Gomoku game."""
    settings = _load_settings(config_path)

    # If --opening-book is given, inject the path into all alpha-beta brain aliases.
    if opening_book_path is not None:
        book_str = str(opening_book_path)
        for name in [black_name, white_name]:
            try:
                brain_type, _kwargs = resolve_brain(name, settings.brains)
            except ValueError:
                continue
            if brain_type == "alphabeta":
                alias = (settings.brains.model_extra or {}).get(name)
                if alias is not None and isinstance(alias, dict):
                    alias["opening_book"] = book_str
                elif name == "alphabeta":
                    # Built-in name without an alias — create one.
                    if settings.brains.model_extra is None:
                        object.__setattr__(settings.brains, "__pydantic_extra__", {})
                    settings.brains.model_extra[name] = {"type": "alphabeta", "opening_book": book_str}

    # Validate provider names early.
    available = available_brains(settings.brains)
    for name, label in [(black_name, "--black"), (white_name, "--white")]:
        if name not in available:
            raise click.BadParameter(
                f"Unknown provider {name!r}. Available: {', '.join(available)}",
                param_hint=f"'{label}'",
            )

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
    console.print(f"  Black: {black_name}")
    console.print(f"  White: {white_name}")
    if do_record and recorder is not None and recorder.path is not None:
        console.print(f"  Recording to: {recorder.path}")

    # Determine whether to produce diagnostics
    from gomoku.diagnostics import DiagnosticsRecorder

    do_diag = diagnostics if diagnostics is not None else settings.diagnostics.enabled
    diag_dir = diagnostics_dir if diagnostics_dir is not None else settings.diagnostics.directory
    diag_recorder: DiagnosticsRecorder | None = None
    if do_diag:
        players_info = [
            {"stone": int(Stone.BLACK), "provider": black_name, "settings": _player_settings(black_name, settings)},
            {"stone": int(Stone.WHITE), "provider": white_name, "settings": _player_settings(white_name, settings)},
        ]
        diag_recorder = DiagnosticsRecorder(
            game_config=config,
            players=players_info,
            directory=diag_dir,
            started_at=recorder._started_at if recorder is not None else None,
        )
        console.print(f"  Diagnostics to: {diag_recorder.path}")

    console.print()

    game = Game(config, black, white, on_move=on_move)  # type: ignore[arg-type]

    while True:
        move_number = len(game.move_history) + 1
        console.print(f"\nMove #{move_number} \u2014 {game.current_stone} to play", style="bold")

        # Set up diagnostics context for this move.
        diag_entry = None
        move_start: float | None = None
        if diag_recorder is not None:
            board_stones = [
                f"{'B' if game.board.at(r, c) == Stone.BLACK else 'W'} {Move(row=r, col=c)}"
                for r, c in sorted(game.board.occupied)
            ]
            diag_entry = diag_recorder.begin_move(move_number, game.current_stone, board_stones)
            move_start = time.monotonic()

        try:
            result = game.play_turn()
        except InvalidMoveError as e:
            console.print(f"[red]Error:[/red] {e}")
            if diag_entry is not None:
                diag_entry.notes.append(f"InvalidMoveError: {e}")
                if move_start is not None:
                    diag_entry.think_time_ms = int((time.monotonic() - move_start) * 1000)
                diag_recorder.end_move(diag_entry)  # type: ignore[union-attr]
            break

        _last_stone, last_move = game.move_history[-1]
        turn = len(game.move_history)

        # Finalise diagnostics for this move.
        if diag_entry is not None:
            if move_start is not None:
                diag_entry.think_time_ms = int((time.monotonic() - move_start) * 1000)
            if diag_entry.chosen_move is None:
                diag_entry.chosen_move = last_move
            diag_recorder.end_move(diag_entry)  # type: ignore[union-attr]

        winning_moves = result.winning_moves if result is not None else None
        render_board(
            game.board,
            last_move=last_move,
            turn=turn,
            current_stone=game.current_stone,
            winning_moves=winning_moves,
        )

        if result is not None:
            render_result(result, turn)
            if recorder is not None:
                recorder.finalize(result)
                console.print(f"\n[bold cyan]Game recorded:[/bold cyan] {recorder.path}")
            if diag_recorder is not None:
                diag_recorder.write_summary(
                    {
                        "ended_at": str(result),
                        "total_moves": turn,
                        "winner": int(result.winner) if result.winner else None,
                        "is_draw": result.is_draw,
                    }
                )
                console.print(f"[bold cyan]Diagnostics:[/bold cyan] {diag_recorder.path}")
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
    If a companion ``.diag.jsonl`` file is found next to *file* it is
    loaded automatically and per-move diagnostics are overlaid.
    """
    from gomoku_cli.playback import run_playback

    record = load_record(file)
    run_playback(record, from_move=from_move, diagnostics_path=_find_diagnostics(file))


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit the analysis report as a single JSON document (for scripting).",
)
def analyze(file: Path, as_json: bool) -> None:
    """Analyse a ``.diag.jsonl`` diagnostics file for blunders and perf issues."""
    import json as _json

    from gomoku_cli.analyze import analyze_diagnostics, render_report

    report = analyze_diagnostics(file)
    if as_json:
        click.echo(_json.dumps(report.to_json(), indent=2, default=str))
        return
    render_report(report, console)


def _find_diagnostics(record_path: Path) -> Path | None:
    """Locate the companion ``.diag.jsonl`` file next to a ``.json`` recording.

    Game recordings are named ``gomoku_YYYYMMDD_HHMMSS.json`` and their
    diagnostics twin is ``gomoku_YYYYMMDD_HHMMSS.diag.jsonl`` in the same
    directory.  We also try ``<stem>.diag.jsonl`` as a fallback when the
    user uses a custom filename.
    """
    candidates: list[Path] = []
    if record_path.suffix == ".json":
        candidates.append(record_path.with_suffix(".diag.jsonl"))
    stem = record_path.name.removesuffix(".json")
    candidates.append(record_path.with_name(f"{stem}.diag.jsonl"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_settings(config_path: Path | None) -> GomokuSettings:
    """Load settings, optionally from an explicit TOML path."""
    if config_path is not None:
        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        return GomokuSettings(**data)
    return GomokuSettings()


@cli.command("generate-book")
@click.option(
    "--brain",
    "brain_name",
    type=str,
    default="alphabeta",
    show_default=True,
    help="Brain to use for position evaluation.",
)
@click.option(
    "--max-positions",
    type=click.IntRange(1, 100_000),
    default=100,
    show_default=True,
    help="Maximum number of positions to expand into the book.",
)
@click.option(
    "--error-threshold",
    type=float,
    default=500.0,
    show_default=True,
    help="Max eval gap for including opponent moves (wider = broader book).",
)
@click.option(
    "--depth-penalty",
    type=float,
    default=50.0,
    show_default=True,
    help="Weight of path length in priority function (higher = shallower/wider).",
)
@click.option(
    "--max-ply",
    type=click.IntRange(2, 30),
    default=10,
    show_default=True,
    help="Maximum book depth in half-moves.",
)
@click.option(
    "--board-size",
    type=click.IntRange(5, 25),
    default=None,
    help="Board size (NxN). [default: from config]",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("opening_book.json"),
    show_default=True,
    help="Output path for the generated book.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to gomoku.toml config file.",
)
def generate_book(
    brain_name: str,
    max_positions: int,
    error_threshold: float,
    depth_penalty: float,
    max_ply: int,
    board_size: int | None,
    output_path: Path,
    config_path: Path | None,
) -> None:
    """Generate an opening book using Lincke's Drop-Out Expansion algorithm."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = _load_settings(config_path)

    available = available_brains(settings.brains)
    if brain_name not in available:
        raise click.BadParameter(
            f"Unknown brain {brain_name!r}. Available: {', '.join(available)}",
            param_hint="'--brain'",
        )

    if board_size is not None:
        settings.game.board_size = board_size

    brain = _make_provider(brain_name, settings)

    console.print("\n[bold]Opening Book Generator (DOE)[/bold]")
    console.print(f"  Brain: {brain_name}")
    console.print(f"  Board: {settings.game.board_size}x{settings.game.board_size}")
    console.print(f"  Max positions: {max_positions}")
    console.print(f"  Error threshold: {error_threshold}")
    console.print(f"  Depth penalty: {depth_penalty}")
    console.print(f"  Max ply: {max_ply}")
    console.print(f"  Output: {output_path}")
    console.print()

    from gomoku_brain.opening_book_gen import generate_opening_book

    def _progress(expanded: int, total: int, key: str) -> None:
        console.print(f"  [{expanded}/{total}] expanded", end="\r")

    book = generate_opening_book(
        brain,
        board_size=settings.game.board_size,
        max_positions=max_positions,
        error_threshold=error_threshold,
        depth_penalty=depth_penalty,
        max_book_ply=max_ply,
        progress_callback=_progress,
    )

    book.generator_brain = brain_name
    try:
        _brain_type, kwargs = resolve_brain(brain_name, settings.brains)
        book.generator_settings = kwargs
    except ValueError:
        pass

    book.save(output_path)
    console.print(f"\n[bold green]Book saved:[/bold green] {output_path} ({len(book.positions)} positions)")


@cli.command()
@click.option("--a", "spec_a", required=True, help="Engine A: brain=NAME[,rev=GIT_REV][,config=PATH].")
@click.option("--b", "spec_b", required=True, help="Engine B, same format. Without rev= the working tree is used.")
@click.option("--games", type=click.IntRange(2, 10_000), default=20, show_default=True, help="Games to play (even).")
@click.option("--seed", type=int, default=1, show_default=True, help="Seed for the generated openings.")
@click.option("--jobs", type=click.IntRange(1, 64), default=1, show_default=True, help="Games played in parallel.")
@click.option("--board-size", type=click.IntRange(5, 25), default=15, show_default=True)
@click.option("--max-moves", type=click.IntRange(10, 625), default=150, show_default=True, help="Draw after this many.")
@click.option("--opening-stones", type=click.IntRange(0, 12), default=6, show_default=True)
@click.option("--opening-spread", type=click.IntRange(1, 6), default=2, show_default=True, help="Radius around centre.")
@click.option(
    "--quiet-openings/--any-openings",
    default=True,
    show_default=True,
    help="Reject openings where either colour already has a forced win.",
)
@click.option(
    "--move-timeout", type=float, default=600.0, show_default=True, help="Seconds before an engine is failed."
)
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None, help="Write all games as JSON.")
def match(
    spec_a: str,
    spec_b: str,
    games: int,
    seed: int,
    jobs: int,
    board_size: int,
    max_moves: int,
    opening_stones: int,
    opening_spread: int,
    quiet_openings: bool,
    move_timeout: float,
    out_path: Path | None,
) -> None:
    """Play engine A against engine B and report who is stronger.

    Every opening is played twice with colours swapped.  Use rev= to pit
    a git revision against the working tree, or config= to compare
    settings:

      gomoku match --a brain=threat,rev=master --b brain=threat --games 40
    """
    from gomoku_cli.match import (
        EngineError,
        EngineSpec,
        format_summary,
        progress_printer,
        run_match,
        save_records,
        summarize,
    )

    try:
        engine_a = EngineSpec.parse(spec_a)
        engine_b = EngineSpec.parse(spec_b)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    total = 2 * ((games + 1) // 2)
    click.echo(f"A: {engine_a.label}\nB: {engine_b.label}\nStarting engines (first start of a revision compiles)...")
    try:
        with progress_printer(click.echo, total) as on_game:
            records = run_match(
                engine_a,
                engine_b,
                games=games,
                seed=seed,
                size=board_size,
                max_moves=max_moves,
                jobs=jobs,
                move_timeout_s=move_timeout,
                opening_stones=opening_stones,
                opening_spread=opening_spread,
                quiet_openings=quiet_openings,
                on_game=on_game,
            )
    except (EngineError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    summary = summarize(records)
    click.echo()
    click.echo(format_summary(engine_a, engine_b, summary))
    if out_path is not None:
        save_records(out_path, engine_a, engine_b, records, summary)
        click.echo(f"Games written to {out_path}")


if __name__ == "__main__":
    cli()
