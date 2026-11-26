"""Playback navigator — replay a recorded game with arrow-key controls."""

import sys
import termios
import tty
from pathlib import Path

from gomoku.board import Board
from gomoku.diagnostics import MoveDiagnostics
from gomoku.models import Stone
from gomoku.recorder import GameRecord
from rich.console import Console

from gomoku_cli.display import render_board, render_playback_info, render_result

_console = Console()


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


def _load_diag_by_move(path: Path | None) -> dict[int, MoveDiagnostics]:
    """Return ``{move_number: MoveDiagnostics}`` from a ``.diag.jsonl`` file."""
    if path is None or not path.is_file():
        return {}
    import json

    out: dict[int, MoveDiagnostics] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("type") in {"header", "summary"}:
            continue
        try:
            diag = MoveDiagnostics.model_validate(data)
        except Exception:
            continue
        out[diag.move_number] = diag
    return out


def _render_move_diag(diag: MoveDiagnostics) -> None:
    """Print a compact diagnostics overlay for one move."""
    stats = diag.search_stats
    bits: list[str] = []
    if diag.decision_path:
        bits.append(f"path={diag.decision_path}")
    if stats.depth_reached is not None:
        bits.append(f"depth={stats.depth_reached}")
    if stats.nodes_searched is not None:
        bits.append(f"nodes={stats.nodes_searched}")
    if stats.tt_hits is not None and stats.tt_probes:
        rate = stats.tt_hits / stats.tt_probes * 100
        bits.append(f"tt={stats.tt_hits}/{stats.tt_probes} ({rate:.0f}%)")
    if stats.iterations is not None:
        bits.append(f"iters={stats.iterations}")
    if stats.time_ms is not None:
        bits.append(f"search={stats.time_ms}ms")
    if diag.eval_before is not None and diag.eval_after is not None:
        bits.append(f"eval {diag.eval_before:+.0f} → {diag.eval_after:+.0f}")

    if bits:
        _console.print("  [cyan]" + "  ".join(bits) + "[/cyan]")

    # Threat summary (only print when non-trivial to avoid noise).
    our = diag.our_threats
    opp = diag.opp_threats
    threat_line_parts: list[str] = []
    if our.four_threats or our.three_threats:
        threat_line_parts.append(f"ours 4s={our.four_threats} 3s={our.three_threats}")
    if opp.four_threats or opp.three_threats:
        threat_line_parts.append(f"opp 4s={opp.four_threats} 3s={opp.three_threats}")
    if threat_line_parts:
        _console.print("  [magenta]threats:[/magenta] " + "  |  ".join(threat_line_parts))

    # Top candidates (first 5).
    if diag.top_candidates:
        cand_str = ", ".join(f"{c.move}={c.score:g}" for c in diag.top_candidates[:5])
        _console.print(f"  [dim]top:[/dim] {cand_str}")

    # Alternative-best warning when the chosen move was not top.
    if diag.alternative_best is not None and diag.chosen_move is not None:
        alt = diag.alternative_best
        _console.print(f"  [yellow]alt-best:[/yellow] {alt.move} ({alt.score:g}) — chose {diag.chosen_move} instead")


def run_playback(
    record: GameRecord,
    from_move: int = 1,
    diagnostics_path: Path | None = None,
) -> None:
    """Interactive playback of a recorded game.

    Navigation:
        → (right arrow) — advance to next move
        ← (left arrow)  — go back to previous move
        q               — quit

    When *diagnostics_path* points to a ``.diag.jsonl`` file (typically the
    companion produced by ``gomoku play --diagnostics``), per-move brain
    internals are rendered beneath each board state.
    """
    moves = record.moves
    total = len(moves)
    diag_by_move = _load_diag_by_move(diagnostics_path)

    # Clamp from_move to valid range (0 = show empty board before move 1)
    current = max(0, min(from_move, total))

    def _build_board(up_to: int) -> Board:
        """Replay the first *up_to* moves onto a fresh board."""
        board = Board(size=record.game_config.board_size)
        for m in moves[:up_to]:
            board = board.place_stone(m.move, m.stone)
        return board

    render_playback_info(record)
    if diag_by_move:
        _console.print(f"  [dim]Diagnostics overlay: {diagnostics_path}[/dim]\n")

    while True:
        board = _build_board(current)

        last_move = moves[current - 1].move if current > 0 else None
        current_stone: Stone | None = None
        if current < total:
            current_stone = moves[current].stone
        elif current > 0:
            current_stone = moves[current - 1].stone.opponent()

        winning_moves = record.result.winning_moves if record.result is not None and current == total else None
        render_board(
            board,
            last_move=last_move,
            turn=current,
            current_stone=current_stone,
            winning_moves=winning_moves,
        )

        # Move info line
        if current > 0:
            m = moves[current - 1]
            print(f"  Move {m.number}/{total}: {m.stone} → {m.move}  (think: {m.think_time_ms}ms)")
            diag = diag_by_move.get(m.number)
            if diag is not None:
                _render_move_diag(diag)
        else:
            print(f"  Start — move 0/{total}")

        if current == total and record.result is not None:
            render_result(record.result, total)

        print("\n  ← previous  |  → next  |  q quit")

        key = _read_key()
        if key == "q":
            break
        elif key == "right" and current < total:
            current += 1
        elif key == "left" and current > 0:
            current -= 1
