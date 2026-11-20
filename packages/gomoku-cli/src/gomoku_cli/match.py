"""Engine-versus-engine matches for measuring playing strength.

A match pits two *engines* against each other over a set of generated
openings.  Every opening is played twice with colours swapped, so an
unbalanced opening cannot favour one side, and deterministic brains
still produce distinct games.

An engine is a brain alias plus, optionally, the git revision its code
comes from and the config file it reads — so the same command compares
two builds, two aliases, or two settings of one alias::

    gomoku match --a brain=threat,rev=master --b brain=threat --games 40

Each engine runs in its own process (see :mod:`gomoku_cli.match_worker`)
with ``PYTHONPATH`` pointing at its source tree, which is how two
revisions of the same packages coexist.
"""

from __future__ import annotations

import json
import math
import os
import random
import select
import subprocess
import sys
import tarfile
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_cli.brain_spec import describe, parse_overrides, split_parts

_WORKER = Path(__file__).with_name("match_worker.py")
_SOURCE_PACKAGES = ("gomoku", "gomoku-brain")


# ---------------------------------------------------------------------------
# Engine specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineSpec:
    """Which brain to run, from which code, with which config and settings."""

    brain: str
    rev: str | None = None
    config: Path | None = None
    #: Brain settings changed on top of the config (``KEY=VALUE`` parts).
    overrides: dict[str, Any] = field(default_factory=dict, hash=False)

    @property
    def label(self) -> str:
        parts = [self.brain]
        if self.overrides:
            parts.append(f"({describe(self.overrides)})")
        if self.rev:
            parts.append(f"@{self.rev}")
        if self.config:
            parts.append(f"[{self.config.name}]")
        return "".join(parts)

    @classmethod
    def parse(cls, text: str) -> EngineSpec:
        """Parse ``brain=NAME[,rev=REV][,config=PATH][,KEY=VALUE...]`` (a bare name is a brain).

        Other ``KEY=VALUE`` parts change brain settings, as in ``gomoku play``
        (see :mod:`gomoku_cli.brain_spec`); the worker checks the keys.
        """
        fields: dict[str, str] = {}
        settings: list[str] = []
        for i, part in enumerate(split_parts(text)):
            key, sep, value = part.partition("=")
            if not sep and i == 0:
                key, value = "brain", part  # a bare first part names the brain
            key = key.strip()
            if (not sep and i > 0) or not value.strip() or key in fields:
                msg = f"Invalid engine spec {text!r}: expected brain=NAME[,rev=REV][,config=PATH][,KEY=VALUE...]"
                raise ValueError(msg)
            if key in {"brain", "rev", "config"}:
                fields[key] = value.strip()
            else:
                settings.append(part)
        try:
            overrides = parse_overrides(settings)
        except ValueError as exc:
            msg = f"Invalid engine spec {text!r}: {exc}"
            raise ValueError(msg) from exc
        if "brain" not in fields:
            msg = f"Invalid engine spec {text!r}: no brain given"
            raise ValueError(msg)
        config = Path(fields["config"]) if "config" in fields else None
        if config is not None and not config.is_file():
            # The settings loader would quietly fall back to another config.
            msg = f"Invalid engine spec {text!r}: config file {str(config)!r} does not exist"
            raise ValueError(msg)
        return cls(brain=fields["brain"], rev=fields.get("rev"), config=config, overrides=overrides)


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------


def generate_openings(
    count: int, seed: int, size: int = 15, stones: int = 7, spread: int = 3, quiet: bool = True
) -> list[list[tuple[int, int]]]:
    """Return *count* distinct openings of *stones* alternating moves.

    Stones are drawn uniformly from the ``(2 * spread + 1)`` square around
    the centre.  The colour swap already cancels any bias in an opening,
    but an opening that is tactically decided teaches nothing: both
    engines win it as the favoured colour.  With *quiet* set, openings
    in which either colour has a forced win by threats are rejected.

    The defaults (7 stones, so White moves first after the opening, drawn
    from a 7x7 area) were picked by same-engine self-play: over 40 games
    the threat brain won 19 as Black and 21 as White, against 26 and 14
    with 6 stones from a 5x5 area.
    """
    rng = random.Random(seed)
    centre = size // 2
    cells = [
        (r, c) for r in range(centre - spread, centre + spread + 1) for c in range(centre - spread, centre + spread + 1)
    ]
    if stones > len(cells):
        msg = f"Cannot place {stones} stones in a {2 * spread + 1}x{2 * spread + 1} area"
        raise ValueError(msg)

    openings: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    while len(openings) < count:
        attempts += 1
        if attempts > count * 1000:
            msg = f"Could not generate {count} distinct openings; raise --opening-spread"
            raise ValueError(msg)
        opening = rng.sample(cells, stones)
        key = tuple(opening)
        if key in seen or _has_winner(opening, size) or (quiet and _is_tactical(opening, size)):
            continue
        seen.add(key)
        openings.append(opening)
    return openings


def _is_tactical(moves: list[tuple[int, int]], size: int) -> bool:
    """True if either colour, given the move, has a forced win by threats."""
    from gomoku_brain.pns import pns_threat_search

    board = Board(size=size)
    stone = Stone.BLACK
    for row, col in moves:
        board = board.place_stone(Move(row=row, col=col), stone)
        stone = stone.opponent()
    return any(
        pns_threat_search(board, s, max_nodes=3_000, max_time_s=5.0) is not None for s in (Stone.BLACK, Stone.WHITE)
    )


def _has_winner(moves: list[tuple[int, int]], size: int) -> bool:
    board = Board(size=size)
    stone = Stone.BLACK
    for row, col in moves:
        move = Move(row=row, col=col)
        board = board.place_stone(move, stone)
        if check_winner(board, move) is not None:
            return True
        stone = stone.opponent()
    return False


# ---------------------------------------------------------------------------
# Engine processes
# ---------------------------------------------------------------------------


class EngineError(RuntimeError):
    """An engine process died, timed out, or broke the protocol."""


class Engine:
    """A brain running in a worker process."""

    def __init__(self, spec: EngineSpec, source: Path | None, size: int, move_timeout_s: float) -> None:
        self.spec = spec
        self._timeout = move_timeout_s
        env = dict(os.environ)
        env.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
        cwd: Path | None = None
        if source is not None:
            paths = [str(source / "packages" / pkg / "src") for pkg in _SOURCE_PACKAGES]
            if env.get("PYTHONPATH"):
                paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(paths)
            cwd = source
            default_config = source / "gomoku.toml"
            if default_config.exists():
                env["GOMOKU_CONFIG"] = str(default_config)
        if spec.config is not None:
            env["GOMOKU_CONFIG"] = str(spec.config.resolve())
        # -P keeps the worker's own directory (this package) off sys.path.
        command = [sys.executable, "-P", str(_WORKER), "--brain", spec.brain, "--size", str(size)]
        if spec.overrides:
            command += ["--overrides", json.dumps(spec.overrides)]
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=cwd,
            text=True,
            bufsize=1,
        )
        ready = self._read(timeout_s=600.0)  # start-up includes JIT compilation
        if "error" in ready:
            self.close()
            msg = f"Engine {spec.label}: {ready['error']}"
            raise EngineError(msg)

    def ask(self, moves: list[tuple[int, int]]) -> tuple[tuple[int, int], float]:
        """Return ``((row, col), think_ms)`` for the position after *moves*."""
        reply = self.query({"moves": moves})
        row, col = reply["move"]
        return (int(row), int(col)), float(reply["ms"])

    def query(self, request: dict, timeout_s: float | None = None) -> dict:
        """Send one protocol request (see :mod:`gomoku_cli.match_worker`) and return the reply."""
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            msg = f"Engine {self.spec.label} is not accepting input"
            raise EngineError(msg) from exc
        reply = self._read(self._timeout if timeout_s is None else timeout_s)
        if "error" in reply:
            msg = f"Engine {self.spec.label}: {reply['error']}"
            raise EngineError(msg)
        return reply

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def _read(self, timeout_s: float) -> dict:
        assert self._proc.stdout is not None
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout_s)
        if not ready:
            self.close()
            msg = f"Engine {self.spec.label} did not answer within {timeout_s:.0f}s"
            raise EngineError(msg)
        line = self._proc.stdout.readline()
        if not line:
            msg = f"Engine {self.spec.label} exited (code {self._proc.poll()})"
            raise EngineError(msg)
        return json.loads(line)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                stream.close()


# ---------------------------------------------------------------------------
# Source trees
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def resolve_source(spec: EngineSpec, repo: Path | None) -> Path | None:
    """Return the source tree an engine should import from.

    ``None`` means "whatever is installed in this environment".  A spec
    without ``rev`` uses the working tree (uncommitted changes included).
    A ``rev`` is exported once into a per-commit cache directory, so its
    compiled-kernel cache is reused by later matches.
    """
    if spec.rev is None:
        return repo
    if repo is None:
        msg = f"Engine {spec.label}: rev= needs to be run inside the git repository"
        raise ValueError(msg)
    sha = _git(["rev-parse", "--verify", f"{spec.rev}^{{commit}}"], repo)
    target = Path(tempfile.gettempdir()) / "gomoku-match" / sha
    if not (target / ".complete").exists():
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "src.tar"
        subprocess.run(["git", "archive", "-o", str(archive), sha, "packages", "gomoku.toml"], cwd=repo, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
        archive.unlink()
        (target / ".complete").touch()
    return target


def describe_rev(spec: EngineSpec, repo: Path | None) -> str:
    """One line naming the commit an engine runs, with a warning if it looks stale.

    A local branch that is behind its upstream (``master`` not pulled
    after a merge, say) silently compares against old code; the warning
    makes that visible before a long run.
    """
    if spec.rev is None or repo is None:
        return "working tree"
    line = _git(["log", "-1", "--format=%h %s", f"{spec.rev}^{{commit}}"], repo)
    try:
        upstream = _git(["rev-parse", "--verify", "--quiet", f"{spec.rev}@{{upstream}}"], repo)
    except subprocess.CalledProcessError:
        return line  # not a branch with an upstream
    behind = _git(["rev-list", "--count", f"{spec.rev}..{upstream}"], repo)
    if behind != "0":
        line += f"  [WARNING: {spec.rev} is {behind} commit(s) behind its upstream; fetch and update it]"
    return line


def find_repo() -> Path | None:
    try:
        return Path(_git(["rev-parse", "--show-toplevel"], Path.cwd()))
    except subprocess.CalledProcessError, FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """One finished game.  ``winner`` is ``"a"``, ``"b"`` or ``None`` (draw)."""

    opening: int
    a_is_black: bool
    winner: str | None
    reason: str
    moves: list[tuple[int, int]]
    a_ms: list[float] = field(default_factory=list)
    b_ms: list[float] = field(default_factory=list)
    #: Decision path of every engine move (``play_game(..., diagnostics=True)`` only).
    paths: list[str | None] = field(default_factory=list)


def play_game(
    engine_a: Engine,
    engine_b: Engine,
    opening: list[tuple[int, int]],
    index: int,
    a_is_black: bool,
    size: int,
    max_moves: int,
    *,
    diagnostics: bool = False,
) -> GameRecord:
    """Play one game from *opening*; an illegal move loses on the spot.

    With *diagnostics*, each engine move's decision path is kept in
    ``record.paths``.
    """
    board = Board(size=size)
    stone = Stone.BLACK
    moves: list[tuple[int, int]] = []
    for row, col in opening:
        board = board.place_stone(Move(row=row, col=col), stone)
        moves.append((row, col))
        stone = stone.opponent()

    record = GameRecord(opening=index, a_is_black=a_is_black, winner=None, reason="move limit", moves=moves)
    while len(moves) < min(max_moves, size * size):
        a_to_move = (stone is Stone.BLACK) == a_is_black
        engine = engine_a if a_to_move else engine_b
        if diagnostics:
            reply = engine.query({"moves": moves, "diagnostics": True})
            (row, col), think_ms = reply["move"], float(reply["ms"])
            record.paths.append(reply.get("path"))
        else:
            (row, col), think_ms = engine.ask(moves)
        (record.a_ms if a_to_move else record.b_ms).append(think_ms)
        move = Move(row=row, col=col)
        if not board.is_valid_move(move):
            record.winner = "b" if a_to_move else "a"
            record.reason = f"illegal move {move}"
            return record
        board = board.place_stone(move, stone)
        moves.append((row, col))
        if check_winner(board, move) is not None:
            record.winner = "a" if a_to_move else "b"
            record.reason = "five in a row"
            return record
        stone = stone.opponent()
    if len(moves) >= size * size:
        record.reason = "board full"
    return record


def run_match(
    spec_a: EngineSpec,
    spec_b: EngineSpec,
    *,
    games: int,
    seed: int = 1,
    size: int = 15,
    max_moves: int = 150,
    jobs: int = 1,
    move_timeout_s: float = 600.0,
    opening_stones: int = 7,
    opening_spread: int = 3,
    quiet_openings: bool = True,
    on_game: Callable[[GameRecord], None] | None = None,
) -> list[GameRecord]:
    """Play *games* games (rounded up to an even number) and return them."""
    pairs = (games + 1) // 2
    openings = generate_openings(pairs, seed, size, opening_stones, opening_spread, quiet_openings)
    repo = find_repo()
    source_a = resolve_source(spec_a, repo)
    source_b = resolve_source(spec_b, repo)

    schedule = [(i, a_black) for i in range(pairs) for a_black in (True, False)]
    local = threading.local()
    engines: list[Engine] = []
    lock = threading.Lock()

    def pair() -> tuple[Engine, Engine]:
        if not hasattr(local, "pair"):
            created = (Engine(spec_a, source_a, size, move_timeout_s), Engine(spec_b, source_b, size, move_timeout_s))
            with lock:
                engines.extend(created)
            local.pair = created
        return local.pair

    def play(item: tuple[int, bool]) -> GameRecord:
        index, a_black = item
        engine_a, engine_b = pair()
        record = play_game(engine_a, engine_b, openings[index], index, a_black, size, max_moves)
        if on_game is not None:
            with lock:
                on_game(record)
        return record

    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            return list(pool.map(play, schedule))
    finally:
        for engine in engines:
            engine.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchSummary:
    """Aggregate result from engine A's point of view."""

    games: int
    wins_a: int
    wins_b: int
    draws: int
    a_wins_as_black: int
    a_wins_as_white: int
    b_wins_as_black: int
    b_wins_as_white: int
    score: float
    score_low: float
    score_high: float
    elo: float | None
    elo_low: float | None
    elo_high: float | None
    los: float | None
    pairs: int
    a_ms_per_move: float
    b_ms_per_move: float


def _elo(score: float) -> float | None:
    """Elo difference implied by an expected score; ``None`` at 0 or 1."""
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def _game_score(record: GameRecord) -> float:
    """A's score in one game: 1 win, 0.5 draw, 0 loss."""
    return {"a": 1.0, "b": 0.0, None: 0.5}[record.winner]


def summarize(records: list[GameRecord]) -> MatchSummary:
    """Summarise *records*: score with a 95% interval, Elo, and LOS.

    The interval is computed over opening pairs, not single games.  The
    two games of an opening share its colour bias, so their results are
    correlated; a pair's average score (0, 0.5 or 1 without draws) is the
    independent unit, and it removes the colour noise that dominates
    per-game results.

    To keep small or uniform samples from producing a falsely narrow
    interval (two split pairs alone would give zero variance), the
    variance estimate includes one virtual pair lost and one won by A.
    Their effect fades as real pairs accumulate.  ``los`` is the
    likelihood of superiority, the probability that A is the stronger
    engine, from the same paired estimate.
    """
    n = len(records)
    wins_a = sum(r.winner == "a" for r in records)
    wins_b = sum(r.winner == "b" for r in records)
    draws = n - wins_a - wins_b
    score = (wins_a + 0.5 * draws) / n if n else 0.5

    by_opening: dict[int, list[float]] = {}
    for record in records:
        by_opening.setdefault(record.opening, []).append(_game_score(record))
    pair_scores = [sum(v) / len(v) for v in by_opening.values()]
    samples = [*pair_scores, 0.0, 1.0]
    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    se = math.sqrt(variance / max(1, len(pair_scores)))
    margin = 1.96 * se
    low, high = max(0.0, score - margin), min(1.0, score + margin)
    los = 0.5 * (1.0 + math.erf((score - 0.5) / (se * math.sqrt(2.0)))) if n else None

    a_ms = [ms for r in records for ms in r.a_ms]
    b_ms = [ms for r in records for ms in r.b_ms]
    return MatchSummary(
        games=n,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        a_wins_as_black=sum(r.winner == "a" and r.a_is_black for r in records),
        a_wins_as_white=sum(r.winner == "a" and not r.a_is_black for r in records),
        b_wins_as_black=sum(r.winner == "b" and not r.a_is_black for r in records),
        b_wins_as_white=sum(r.winner == "b" and r.a_is_black for r in records),
        score=score,
        score_low=low,
        score_high=high,
        elo=_elo(score),
        elo_low=_elo(low),
        elo_high=_elo(high),
        los=los,
        pairs=len(pair_scores),
        a_ms_per_move=sum(a_ms) / len(a_ms) if a_ms else 0.0,
        b_ms_per_move=sum(b_ms) / len(b_ms) if b_ms else 0.0,
    )


def format_summary(spec_a: EngineSpec, spec_b: EngineSpec, summary: MatchSummary) -> str:
    """Render *summary* as a short plain-text report."""

    def elo(value: float | None, bound: str) -> str:
        return f"{round(value) + 0:+d}" if value is not None else bound

    s = summary
    colours_a = f"{s.a_wins_as_black}/{s.a_wins_as_white}"
    colours_b = f"{s.b_wins_as_black}/{s.b_wins_as_white}"
    elo_range = f"{elo(s.elo_low, '-inf')} to {elo(s.elo_high, '+inf')}"
    lines = [
        f"A: {spec_a.label}",
        f"B: {spec_b.label}",
        "",
        f"Games: {s.games}   A wins {s.wins_a}   B wins {s.wins_b}   draws {s.draws}",
        f"A wins as black/white: {colours_a}   B wins as black/white: {colours_b}",
        f"A score: {s.score:.1%}  (95% interval {s.score_low:.1%} to {s.score_high:.1%}, from {s.pairs} opening pairs)",
        f"Elo difference (A - B): {elo(s.elo, 'n/a')}  (95% interval {elo_range})",
    ]
    if s.los is not None:
        los = "<0.1%" if s.los < 0.001 else ">99.9%" if s.los > 0.999 else f"{s.los:.1%}"
        lines.append(f"Likelihood A is stronger: {los}")
    lines.append(f"Think time per move: A {s.a_ms_per_move:.0f} ms   B {s.b_ms_per_move:.0f} ms")
    return "\n".join(lines)


def save_records(
    path: Path, spec_a: EngineSpec, spec_b: EngineSpec, records: list[GameRecord], summary: MatchSummary
) -> None:
    """Write every game (moves and think times) plus the summary as JSON."""
    payload = {
        "a": spec_a.label,
        "b": spec_b.label,
        "summary": asdict(summary),
        "games": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=1))


@contextmanager
def progress_printer(echo: Callable[[str], None], total: int) -> Iterator[Callable[[GameRecord], None]]:
    """Yield an ``on_game`` callback that prints one line per finished game."""
    done = 0
    tally = {"a": 0, "b": 0, None: 0}

    def on_game(record: GameRecord) -> None:
        nonlocal done
        done += 1
        tally[record.winner] += 1
        colour = "black" if record.a_is_black else "white"
        outcome = {"a": "A wins", "b": "B wins", None: "draw"}[record.winner]
        echo(
            f"[{done}/{total}] opening {record.opening} (A {colour}): {outcome} in {len(record.moves)} moves"
            f" ({record.reason})  — A {tally['a']}  B {tally['b']}  = {tally[None]}"
        )

    yield on_game
