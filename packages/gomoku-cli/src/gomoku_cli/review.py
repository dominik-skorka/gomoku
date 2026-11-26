"""Blunder review: a quick tactical check to run before a long match.

``gomoku review`` plays a few games between a baseline (A) and a
candidate (B), the same way ``gomoku match`` does, then checks every
engine move against an *oracle*: a threat-space proof-number search with
a much larger budget than the brains get.  The oracle runs in its own
worker process from a separate source tree (the baseline's revision by
default), so a bug in the candidate's search cannot also hide its own
mistakes.

Each position before an engine move gets one oracle verdict for the side
to move — ``PROVED`` (it has a forced win), ``DISPROVED`` (no forced
win in threat space) or ``UNKNOWN`` (budget ran out).  Comparing
neighbouring verdicts finds four kinds of blunder, and following the
brain's own claims through the game a fifth:

``missed_win``
    The mover could complete five, or had a proven forced win, and after
    its move and the reply the win is disproved.
``missed_block``
    The opponent threatened to complete five on a single cell and the
    mover played elsewhere.
``losing_move``
    After the move the opponent has a proven forced win, and another
    move (shown) would have left it with none.
``unsound_proof``
    The brain claimed a forced win the oracle disproves, or the oracle
    proved a win, its own move was played, and after the reply the win
    was disproved.  Always a bug in a proof search.
``unconverted_win``
    The brain claimed a forced win (``vcf_win``, ``pns_win``,
    ``vct_win``), but a later move of its own was no win (neither a
    claimed win nor a forced block of the opponent's five), or the game
    ended without its five.  A claimed win must hold to the end of the
    game, so this needs no oracle to spot.  The oracle only tells the two
    causes apart: if the win was still there when the chain broke, the
    brain failed to find it again (escalated like a missed win); if the
    oracle disproves it there, the claim was unsound (a bug).

For every blunder except ``unsound_proof`` (and a disproved
``unconverted_win``) the engine is asked the same
position again, first unchanged and then with larger settings (a bigger
PNS budget, one more ply, a wider fallback, all of them together).  If
one of those finds a good move, the blunder is a *tuning* matter and the
report names the setting and its think-time cost; if none does, the
brain's logic needs fixing.

The review passes when the candidate has no more logic-bug blunders
than the baseline (both play the same openings, with colours swapped).
"""

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_cli.match import (
    Engine,
    EngineError,
    EngineSpec,
    GameRecord,
    find_repo,
    generate_openings,
    play_game,
    resolve_source,
)

# Decision paths that claim a forced win found by a proof search.  Not
# "vcf_precheck": revisions before it was split into vcf_win / vcf_breaker /
# vcf_block_no_breaker / instant_win used it for defences too.
CLAIMED_WIN_PATHS = frozenset({"vcf_win", "pns_win", "vct_win"})
# Decision paths that play a win (claimed or immediate).
WIN_PATHS = CLAIMED_WIN_PATHS | {"immediate_win", "instant_win"}

# --oracle-rev value that runs the oracle from the working tree.
WORKTREE = "worktree"

KINDS = ("missed_win", "missed_block", "losing_move", "unsound_proof", "unconverted_win")
_KIND_TEXT = {
    "missed_win": "missed win",
    "missed_block": "missed block",
    "losing_move": "losing move",
    "unsound_proof": "unsound proof",
    "unconverted_win": "unconverted win",
}

Cell = tuple[int, int]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Attempt:
    """The engine re-asked a blunder position under *variant* settings."""

    variant: str
    overrides: dict[str, Any]
    move: Cell | None
    ms: float | None
    good: bool | None  # None: timed out or could not be judged
    path: str | None = None


@dataclass
class Blunder:
    game: int
    ply: int  # index into the game's moves; the move number is ply + 1
    engine: str  # "a" or "b"
    stone: str  # "black" or "white"
    kind: str
    played: Cell
    better: Cell | None
    detail: str
    path: str | None
    ms: float
    attempts: list[Attempt] = field(default_factory=list)
    verdict: str = "bug"  # "tuning", "bug" or "disputed"
    remedy: str = ""
    # unconverted_win: the ply whose move first claimed the win.
    claim_ply: int | None = None


@dataclass
class GameReview:
    index: int
    record: GameRecord
    # Oracle verdict for the side to move before each ply, from the end of
    # the opening; None for the final position of a won game.
    statuses: list[tuple[str, Cell | None] | None]
    blunders: list[Blunder]
    size: int = 15


@dataclass
class OracleConfig:
    nodes: int = 200_000
    time_s: float = 30.0
    # Saving-move search: smaller budget per candidate (a DISPROVED result
    # is exact whatever the budget) and a cap on candidates tried.
    screen_nodes: int = 20_000
    screen_time_s: float = 3.0
    screen_moves: int = 40


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------


def _board(moves: list[Cell], size: int) -> Board:
    board = Board(size=size)
    stone = Stone.BLACK
    for row, col in moves:
        board = board.place_stone(Move(row=row, col=col), stone)
        stone = stone.opponent()
    return board


def five_cells(moves: list[Cell], stone: Stone, size: int) -> list[Cell]:
    """Empty cells where *stone* would complete five or more in a row."""
    grid: dict[Cell, Stone] = {}
    turn = Stone.BLACK
    for cell in moves:
        grid[cell] = turn
        turn = turn.opponent()
    found = []
    for row in range(size):
        for col in range(size):
            if (row, col) in grid:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                run = 1
                for sign in (1, -1):
                    r, c = row + sign * dr, col + sign * dc
                    while 0 <= r < size and 0 <= c < size and grid.get((r, c)) is stone:
                        run += 1
                        r, c = r + sign * dr, c + sign * dc
                if run >= 5:
                    found.append((row, col))
                    break
    return found


def _nearby(moves: list[Cell], size: int, radius: int = 2) -> list[Cell]:
    taken = set(moves)
    cells = {
        (r, c)
        for mr, mc in moves
        for r in range(max(0, mr - radius), min(size, mr + radius + 1))
        for c in range(max(0, mc - radius), min(size, mc + radius + 1))
    }
    return sorted(cells - taken)


def cell_name(cell: Cell | None) -> str:
    return "-" if cell is None else Move(row=cell[0], col=cell[1]).position


def _stone_at(ply: int) -> Stone:
    return Stone.BLACK if ply % 2 == 0 else Stone.WHITE


# ---------------------------------------------------------------------------
# Escalation variants
# ---------------------------------------------------------------------------


def escalations(brain_type: str, settings: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Larger-budget settings to retry a blunder with, cheapest first.

    Each entry is ``(name, overrides)``; the overrides merge into the
    brain's resolved settings (see :func:`gomoku_cli.match_worker.merge`).
    """
    variants: list[tuple[str, dict[str, Any]]] = []
    if brain_type == "threat":
        threat = {
            "pns_nodes": min(settings["pns_nodes"] * 10, 10_000_000),
            "pns_time_s": min(settings["pns_time_s"] * 4, 60.0),
        }
        if settings.get("threat_search") == "vct":
            threat["vct_depth"] = min(settings["vct_depth"] + 4, 30)
        fallback = settings.get("fallback")
        if fallback is not None:
            deeper = {"fallback": {"depth": min(fallback["depth"] + 1, 10)}}
            wider = {"fallback": {"max_candidates": min(fallback["max_candidates"] * 2, 60)}}
        else:
            # A fallback table replaces the top-level depth, so keep it.
            deeper = {"depth": min(settings["depth"] + 1, 10)}
            wider = {"fallback": {"depth": settings["depth"], "max_candidates": 40}}
        variants = [("more_pns", threat), ("deeper", deeper), ("wider", wider)]
        combined = {**threat, **deeper}
        combined["fallback"] = {**deeper.get("fallback", {}), **wider["fallback"]}
        variants.append(("all", combined))
    elif brain_type in {"alphabeta", "negamax", "minimax"}:
        deeper = {"depth": min(settings["depth"] + 1, 10)}
        variants = [("deeper", deeper)]
        if "max_candidates" in settings:
            wider = {"max_candidates": min(settings["max_candidates"] * 2, 60)}
            variants += [("wider", wider), ("all", {**deeper, **wider})]
    elif brain_type == "mcts":
        variants = [("more_iterations", {"iterations": min(settings["iterations"] * 4, 1_000_000)})]
    return variants


def _describe(overrides: dict[str, Any], prefix: str = "") -> str:
    parts = []
    for key, value in overrides.items():
        if isinstance(value, dict):
            parts.append(_describe(value, f"{prefix}{key}."))
        else:
            parts.append(f"{prefix}{key}={value}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Review of one game
# ---------------------------------------------------------------------------


class _Oracle:
    def __init__(self, engine: Engine, config: OracleConfig, *, brain_space: bool = False) -> None:
        self.engine = engine
        self.config = config
        # Search in the engine's own threat space (the second oracle: judge
        # B in the threat space B plays in).
        self.brain_space = brain_space

    def status(self, moves: list[Cell], stone: Stone | None = None, *, screen: bool = False) -> tuple[str, Cell | None]:
        nodes = self.config.screen_nodes if screen else self.config.nodes
        time_s = self.config.screen_time_s if screen else self.config.time_s
        request: dict[str, Any] = {"op": "status", "moves": moves, "nodes": nodes, "time_s": time_s}
        if stone is not None:
            request["stone"] = int(stone)
        if self.brain_space:
            request["space"] = "brain"
        reply = self.engine.query(request, timeout_s=time_s * 4 + 60)
        move = reply["move"]
        return reply["status"], None if move is None else (int(move[0]), int(move[1]))


class _EngineSlot:
    """An engine that is restarted if an escalation times out."""

    def __init__(self, factory: Callable[[], Engine]) -> None:
        self._factory = factory
        self.engine = factory()
        self._settings: tuple[str, dict[str, Any]] | None = None

    def settings(self) -> tuple[str, dict[str, Any]]:
        if self._settings is None:
            reply = self.engine.query({"op": "settings"})
            self._settings = (reply["type"], reply["settings"])
        return self._settings

    def retry(self, moves: list[Cell], overrides: dict[str, Any], timeout_s: float) -> dict[str, Any] | None:
        try:
            return self.engine.query({"moves": moves, "overrides": overrides, "diagnostics": True}, timeout_s=timeout_s)
        except EngineError:
            self.engine.close()
            self.engine = self._factory()
            return None

    def close(self) -> None:
        self.engine.close()


def review_game(
    index: int,
    record: GameRecord,
    opening_len: int,
    oracle: _Oracle,
    slots: dict[str, _EngineSlot],
    size: int,
    *,
    escalate: bool = True,
    escalate_timeout_s: float = 300.0,
    second_oracle: _Oracle | None = None,
) -> GameReview:
    """Find the blunders in *record* and classify each one.

    With *second_oracle*, each blunder is re-judged by it (the candidate's
    own search); one the two oracles disagree on is ``disputed``: listed,
    not escalated, and left out of the bug count.  It catches an oracle
    that cannot see a threat the candidate now uses.
    """
    moves = record.moves
    won = record.reason == "five in a row"
    statuses: list[tuple[str, Cell | None] | None] = []
    for ply in range(opening_len, len(moves) + 1):
        statuses.append(None if won and ply == len(moves) else oracle.status(moves[:ply]))

    def status_at(ply: int) -> str | None:
        k = ply - opening_len
        entry = statuses[k] if 0 <= k < len(statuses) else None
        return None if entry is None else entry[0]

    blunders = []
    for ply in range(opening_len, len(moves)):
        k = ply - opening_len
        stone = _stone_at(ply)
        engine = "a" if (stone is Stone.BLACK) == record.a_is_black else "b"
        path = record.paths[k] if k < len(record.paths) else None
        ms_list = record.a_ms if engine == "a" else record.b_ms
        own_index = k // 2  # the engines alternate, each with its own think-time list
        ms = ms_list[own_index] if own_index < len(ms_list) else 0.0
        found = _classify(ply, moves, statuses[k], status_at, oracle, size, path, won)
        if found is None:
            continue
        kind, better, detail = found
        blunders.append(
            Blunder(
                game=index,
                ply=ply,
                engine=engine,
                stone=stone.name.lower(),
                kind=kind,
                played=moves[ply],
                better=better,
                detail=detail,
                path=path,
                ms=ms,
            )
        )

    if second_oracle is not None:
        second: dict[int, tuple[str, Cell | None] | None] = {}

        def second_at(ply: int) -> tuple[str, Cell | None] | None:
            if ply not in second:
                over = won and ply == len(moves)
                second[ply] = None if over or ply > len(moves) else second_oracle.status(moves[:ply])
            return second[ply]

        for blunder in blunders:
            ply = blunder.ply
            path = record.paths[ply - opening_len] if ply - opening_len < len(record.paths) else None
            again = _classify(
                ply,
                moves,
                second_at(ply),
                lambda p: (lambda r: None if r is None else r[0])(second_at(p)),
                second_oracle,
                size,
                path,
                won,
            )
            if again is None or again[0] != blunder.kind:
                view = "no blunder" if again is None else _KIND_TEXT[again[0]]
                blunder.verdict = "disputed"
                blunder.remedy = (
                    f"the second oracle (the candidate's own search) sees {view} here; "
                    "check by hand which search is wrong"
                )

    blunders += _unconverted_wins(index, record, opening_len, statuses, blunders, size)

    for blunder in blunders:
        if blunder.verdict == "disputed":
            continue
        if blunder.kind == "unconverted_win" and blunder.better is None:
            if blunder.verdict == "bug":
                blunder.remedy = "the claimed win did not exist (the oracle disproves it): a proof search is unsound"
            else:
                blunder.remedy = "the oracle cannot tell whether the win was still there; check by hand"
            continue
        if blunder.kind == "unsound_proof":
            blunder.verdict, blunder.remedy = "bug", "a proof search is unsound; fix it before tuning anything"
        elif escalate:
            _escalate(blunder, moves, slots[blunder.engine], oracle, size, escalate_timeout_s)
        else:
            blunder.verdict, blunder.remedy = "unclassified", "escalation disabled"
    return GameReview(index=index, record=record, statuses=statuses, blunders=blunders, size=size)


def _unconverted_wins(
    index: int,
    record: GameRecord,
    opening_len: int,
    statuses: list[tuple[str, Cell | None] | None],
    found: list[Blunder],
    size: int,
) -> list[Blunder]:
    """Claimed forced wins the game did not bear out, one per engine.

    From an engine's first move with a claimed-win decision path, every
    later move of its own must also be a win (a claim, or completing
    five) or block the opponent's only five (a counter-four the proof
    allows for: the attacker blocks it and goes on), and the game must
    end with its five.  The first own move that
    breaks the chain, or the last one if the game simply ended, is
    reported.  A chain already explained by another blunder of the same
    engine (say a ``missed_win`` at the break) is not reported twice.
    """
    moves = record.moves
    paths = record.paths
    if not paths:
        return []
    won = record.reason == "five in a row"

    def path_at(ply: int) -> str | None:
        k = ply - opening_len
        return paths[k] if 0 <= k < len(paths) else None

    def engine_at(ply: int) -> str:
        return "a" if (_stone_at(ply) is Stone.BLACK) == record.a_is_black else "b"

    out = []
    for engine in ("a", "b"):
        claim = next(
            (p for p in range(opening_len, len(moves)) if engine_at(p) == engine and path_at(p) in CLAIMED_WIN_PATHS),
            None,
        )
        if claim is None:
            continue
        breaks = None
        detail = ""
        ply = claim + 2
        while ply < len(moves):
            if won and ply == len(moves) - 1:
                break  # this move completes five: the claim held
            forced_block = five_cells(list(moves[:ply]), _stone_at(ply).opponent(), size) == [moves[ply]]
            if path_at(ply) not in WIN_PATHS and not forced_block:
                breaks = ply
                detail = f"its move {ply + 1} was no win ({path_at(ply) or 'no decision path'})"
                break
            ply += 2
        if breaks is None and not (won and record.winner == engine):
            last = ply - 2 if ply - 2 >= claim else claim
            breaks = last
            detail = "the opponent completed five" if won else f"the game ended without its five ({record.reason})"
        if breaks is None:
            continue
        if any(b.engine == engine and claim <= b.ply <= breaks for b in found):
            continue
        k = breaks - opening_len
        entry = statuses[k] if 0 <= k < len(statuses) else None
        status, oracle_move = entry if entry is not None else ("UNKNOWN", None)
        ms_list = record.a_ms if engine == "a" else record.b_ms
        own_index = k // 2
        blunder = Blunder(
            game=index,
            ply=breaks,
            engine=engine,
            stone=_stone_at(breaks).name.lower(),
            kind="unconverted_win",
            played=moves[breaks],
            better=oracle_move if status == "PROVED" else None,
            detail=f"claimed a forced win at move {claim + 1} ({path_at(claim)}); {detail}",
            path=path_at(breaks),
            ms=ms_list[own_index] if own_index < len(ms_list) else 0.0,
            claim_ply=claim,
        )
        if status == "PROVED":
            blunder.detail += f"; the oracle still wins with {cell_name(oracle_move)}"
        elif status == "DISPROVED":
            blunder.verdict = "bug"
        else:
            blunder.verdict = "unclassified"
        out.append(blunder)
    return out


def _classify(
    ply: int,
    moves: list[Cell],
    before: tuple[str, Cell | None] | None,
    status_at: Callable[[int], str | None],
    oracle: _Oracle,
    size: int,
    path: str | None,
    won: bool,
) -> tuple[str, Cell | None, str] | None:
    """Return ``(kind, better_move, detail)`` if the move at *ply* is a blunder."""
    stone = _stone_at(ply)
    opponent = stone.opponent()
    played = moves[ply]
    winning_move = won and ply == len(moves) - 1
    if winning_move:
        return None
    history = moves[:ply]
    status, oracle_move = before if before is not None else ("UNKNOWN", None)

    ours = five_cells(history, stone, size)
    if ours:
        return "missed_win", ours[0], f"{cell_name(ours[0])} completed five"
    theirs = five_cells(history, opponent, size)
    if len(theirs) == 1 and played != theirs[0]:
        return "missed_block", theirs[0], f"the opponent completes five at {cell_name(theirs[0])}"
    if theirs:
        return None  # two or more fives threatened: already lost

    after_reply = status_at(ply + 2)
    opponent_wins_next = won and ply + 1 == len(moves) - 1
    if status == "PROVED":
        if played == oracle_move and after_reply == "DISPROVED":
            return (
                "unsound_proof",
                None,
                (
                    f"the oracle proved a win starting {cell_name(oracle_move)}, it was played, "
                    "and after the reply the oracle disproves the win"
                ),
            )
        if opponent_wins_next or after_reply == "DISPROVED":
            return "missed_win", oracle_move, f"forced win starting {cell_name(oracle_move)}; gone after the reply"
        return None

    if path in CLAIMED_WIN_PATHS and status == "DISPROVED":
        return "unsound_proof", None, f"the brain claimed a forced win ({path}); the oracle disproves it"

    if status_at(ply + 1) == "PROVED":
        saving = _find_saving_move(history, played, stone, oracle, size)
        if saving is not None:
            return "losing_move", saving, f"the opponent then wins by force; {cell_name(saving)} held"
    return None


def _find_saving_move(history: list[Cell], played: Cell, stone: Stone, oracle: _Oracle, size: int) -> Cell | None:
    """A move after which the opponent has no forced win, or None."""
    _, threat_move = oracle.status(history, stone.opponent())
    candidates = [c for c in _nearby(history, size) if c != played]
    if threat_move is not None:
        tr, tc = threat_move
        candidates.sort(key=lambda c: (max(abs(c[0] - tr), abs(c[1] - tc)), c))
    for cell in candidates[: oracle.config.screen_moves]:
        if oracle.status([*history, cell], screen=True)[0] == "DISPROVED":
            return cell
    return None


def _escalate(
    blunder: Blunder,
    moves: list[Cell],
    slot: _EngineSlot,
    oracle: _Oracle,
    size: int,
    timeout_s: float,
) -> None:
    history = moves[: blunder.ply]
    stone = _stone_at(blunder.ply)
    brain_type, settings = slot.settings()
    variants = [("repeat", {}), *escalations(brain_type, settings)]

    def is_good(move: Cell, path: str | None) -> bool | None:
        if blunder.kind == "missed_block":
            return move == blunder.better
        if blunder.kind in ("missed_win", "unconverted_win"):
            if move == blunder.better or move in five_cells(history, stone, size):
                return True
            return path in WIN_PATHS
        status, _ = oracle.status([*history, move])
        return {"DISPROVED": True, "PROVED": False}.get(status)

    for name, overrides in variants:
        reply = slot.retry(history, overrides, timeout_s)
        if reply is None:
            blunder.attempts.append(Attempt(name, overrides, None, None, None))
            continue
        move = (int(reply["move"][0]), int(reply["move"][1]))
        path = reply.get("path")
        good = True if move == blunder.better else is_good(move, path)
        blunder.attempts.append(Attempt(name, overrides, move, float(reply["ms"]), good, path))
        if good:
            break  # variants run cheapest first

    fixed = [a for a in blunder.attempts if a.good]
    if not fixed:
        blunder.verdict = "bug"
        blunder.remedy = "no setting tried finds a good move; the brain's logic needs a fix"
        return
    first = fixed[0]
    cost = f"{first.ms / blunder.ms:.1f}x think time" if first.ms and blunder.ms else "unknown cost"
    blunder.verdict = "tuning"
    if first.variant == "repeat":
        blunder.remedy = (
            "the same settings found it on a second try: the move depends on timing or chance (raise time limits)"
        )
    else:
        blunder.remedy = f"{first.variant}: {_describe(first.overrides)} ({cost})"


# ---------------------------------------------------------------------------
# Running a review
# ---------------------------------------------------------------------------


def run_review(
    spec_a: EngineSpec,
    spec_b: EngineSpec,
    *,
    games: int = 6,
    seed: int = 1,
    size: int = 15,
    max_moves: int = 150,
    jobs: int = 1,
    move_timeout_s: float = 600.0,
    opening_stones: int = 7,
    opening_spread: int = 3,
    quiet_openings: bool = True,
    oracle_rev: str | None = None,
    oracle: OracleConfig | None = None,
    second_oracle: bool = True,
    escalate: bool = True,
    escalate_timeout_s: float = 300.0,
    on_game: Callable[[GameReview], None] | None = None,
) -> list[GameReview]:
    """Play *games* games (rounded up to an even number) and review every move.

    With *second_oracle* (the default), blunders are re-judged by B's own
    search when it comes from a different source tree than the oracle.
    """
    oracle = oracle or OracleConfig()
    pairs = (games + 1) // 2
    openings = generate_openings(pairs, seed, size, opening_stones, opening_spread, quiet_openings)
    repo = find_repo()
    source_a = resolve_source(spec_a, repo)
    source_b = resolve_source(spec_b, repo)
    if oracle_rev == WORKTREE:
        oracle_spec = EngineSpec("random")
    else:
        oracle_spec = EngineSpec("random", rev=oracle_rev if oracle_rev is not None else spec_a.rev)
    source_oracle = resolve_source(oracle_spec, repo)
    # The second oracle also runs when B is the oracle's code configured to
    # search differently (config= with its own threat space).
    use_second = second_oracle and (source_b != source_oracle or spec_b.config is not None)

    schedule = [(i, a_black) for i in range(pairs) for a_black in (True, False)]
    local = threading.local()
    slots: list[_EngineSlot] = []
    lock = threading.Lock()

    def resources() -> tuple[_EngineSlot, _EngineSlot, _EngineSlot]:
        if not hasattr(local, "slots"):
            created = (
                _EngineSlot(lambda: Engine(spec_a, source_a, size, move_timeout_s)),
                _EngineSlot(lambda: Engine(spec_b, source_b, size, move_timeout_s)),
                _EngineSlot(lambda: Engine(oracle_spec, source_oracle, size, move_timeout_s)),
            )
            with lock:
                slots.extend(created)
            local.slots = created
        return local.slots

    def play(item: tuple[int, int, bool]) -> GameReview:
        number, index, a_black = item
        slot_a, slot_b, slot_oracle = resources()
        record = play_game(
            slot_a.engine, slot_b.engine, openings[index], index, a_black, size, max_moves, diagnostics=True
        )
        review = review_game(
            number,
            record,
            len(openings[index]),
            _Oracle(slot_oracle.engine, oracle),
            {"a": slot_a, "b": slot_b},
            size,
            escalate=escalate,
            escalate_timeout_s=escalate_timeout_s,
            second_oracle=_Oracle(slot_b.engine, oracle, brain_space=True) if use_second else None,
        )
        if on_game is not None:
            with lock:
                on_game(review)
        return review

    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            return list(pool.map(play, [(n, i, a) for n, (i, a) in enumerate(schedule)]))
    finally:
        for slot in slots:
            slot.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class EngineTally:
    moves: int = 0
    blunders: int = 0
    tuning: int = 0
    bugs: int = 0
    disputed: int = 0
    by_kind: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))


def tally(reviews: list[GameReview]) -> dict[str, EngineTally]:
    totals = {"a": EngineTally(), "b": EngineTally()}
    for review in reviews:
        totals["a"].moves += len(review.record.a_ms)
        totals["b"].moves += len(review.record.b_ms)
        for blunder in review.blunders:
            t = totals[blunder.engine]
            t.blunders += 1
            t.by_kind[blunder.kind] += 1
            if blunder.verdict == "bug":
                t.bugs += 1
            elif blunder.verdict == "tuning":
                t.tuning += 1
            elif blunder.verdict == "disputed":
                t.disputed += 1
    return totals


def verdict(totals: dict[str, EngineTally]) -> tuple[str, str]:
    """``(PASS|WARN|FAIL, reason)`` for candidate B against baseline A."""
    a, b = totals["a"], totals["b"]
    if b.bugs > a.bugs:
        reason = f"B has {b.bugs} blunder(s) needing a fix in the brain's code, A has {a.bugs}"
        return "FAIL", f"{reason}; fix them before a long match"
    if b.disputed:
        return "WARN", f"the oracles disagree on {b.disputed} of B's blunder(s); check them by hand"
    if b.blunders:
        return "WARN", f"B has {b.blunders} blunder(s), A has {a.blunders}; no more of them need a code fix than A's"
    return "PASS", "no blunders by B"


BOARD_LEGEND = "board: X black, O white (lower case: the move just before), ! played, * a better move"


def render_board(
    moves: list[Cell],
    marks: dict[Cell, str],
    size: int = 15,
    pad: int = 2,
    indent: str = "",
) -> str:
    """Draw the position after *moves* as text, cropped to the stones and marks.

    Columns are lettered and rows numbered as in move names (``H8``), with
    row 1 at the bottom.  *marks* override what is drawn on a cell.
    """
    grid = {cell: "X" if i % 2 == 0 else "O" for i, cell in enumerate(moves)}
    if moves:
        last = moves[-1]
        grid[last] = grid[last].lower()
    cells = [*grid, *marks] or [(size // 2, size // 2)]
    r0, r1 = max(0, min(r for r, _ in cells) - pad), min(size - 1, max(r for r, _ in cells) + pad)
    c0, c1 = max(0, min(c for _, c in cells) - pad), min(size - 1, max(c for _, c in cells) + pad)
    lines = [indent + "    " + " ".join(chr(ord("A") + c) for c in range(c0, c1 + 1))]
    for r in range(r1, r0 - 1, -1):
        row = " ".join(marks.get((r, c)) or grid.get((r, c), ".") for c in range(c0, c1 + 1))
        lines.append(f"{indent}{r + 1:3d} {row}")
    return "\n".join(lines)


def format_blunder(blunder: Blunder, labels: dict[str, str], moves: list[Cell] | None = None, size: int = 15) -> str:
    """One blunder; with the game's *moves*, also the position before it."""
    who = f"{blunder.engine.upper()} ({labels[blunder.engine]}, {blunder.stone})"
    lines = [
        (
            f"game {blunder.game + 1}, move {blunder.ply + 1}: {who} played {cell_name(blunder.played)} — "
            f"{_KIND_TEXT[blunder.kind]}: {blunder.detail}"
        ),
        f"    decision: {blunder.path or 'n/a'}, {blunder.ms / 1000:.1f} s",
    ]
    if blunder.attempts:
        tried = []
        for a in blunder.attempts:
            mark = {True: "ok", False: "no", None: "?"}[a.good]
            took = "timeout" if a.ms is None else f"{a.ms / 1000:.1f} s"
            tried.append(f"{a.variant} {mark} {cell_name(a.move)} ({took})")
        lines.append("    retried: " + " | ".join(tried))
    label = {"tuning": "TUNING", "bug": "BUG", "disputed": "DISPUTED", "unclassified": "?"}[blunder.verdict]
    lines.append(f"    => {label}: {blunder.remedy}")
    if moves is not None:
        marks: dict[Cell, str] = {(blunder.played[0], blunder.played[1]): "!"}
        if blunder.better is not None:
            marks[(blunder.better[0], blunder.better[1])] = "*"
        if blunder.claim_ply is not None and blunder.claim_ply != blunder.ply:
            claim = blunder.claim_ply
            lines.append(f"    when the win was claimed (move {claim + 1}, ! played):")
            claimed: list[Cell] = [(m[0], m[1]) for m in moves[:claim]]
            claim_mark: dict[Cell, str] = {(moves[claim][0], moves[claim][1]): "!"}
            lines.append(render_board(claimed, claim_mark, size, indent="    "))
            lines.append(f"    where the claim broke (move {blunder.ply + 1}):")
        history: list[Cell] = [(m[0], m[1]) for m in moves[: blunder.ply]]
        lines.append(render_board(history, marks, size, indent="    "))
    return "\n".join(lines)


def format_report(spec_a: EngineSpec, spec_b: EngineSpec, reviews: list[GameReview], *, boards: bool = True) -> str:
    """The review report; with *boards*, each blunder's position is drawn."""
    labels = {"a": spec_a.label, "b": spec_b.label}
    totals = tally(reviews)
    blunders = sorted((b for r in reviews for b in r.blunders), key=lambda b: (b.engine != "b", b.game, b.ply))
    lines = [f"Reviewed {len(reviews)} games, {totals['a'].moves + totals['b'].moves} engine moves."]
    for key in ("a", "b"):
        t = totals[key]
        kinds = ", ".join(f"{_KIND_TEXT[k]} {n}" for k, n in t.by_kind.items() if n) or "none"
        lines.append(
            f"{key.upper()} ({labels[key]}): {t.blunders} blunder(s) in {t.moves} moves ({kinds}); "
            f"{t.tuning} fixable by tuning the brain's settings, {t.bugs} need a fix in the brain's code"
            + (f", {t.disputed} disputed by the second oracle" if t.disputed else "")
        )
    games = {r.index: r for r in reviews}
    if blunders:
        lines.append("")
        if boards:
            lines.append(BOARD_LEGEND)
        for b in blunders:
            game = games.get(b.game)
            moves = game.record.moves if boards and game is not None else None
            lines.append(format_blunder(b, labels, moves, game.size if game is not None else 15))
    status, reason = verdict(totals)
    lines += ["", f"Verdict: {status} — {reason}"]
    return "\n".join(lines)


def save_review(path: Path, spec_a: EngineSpec, spec_b: EngineSpec, reviews: list[GameReview]) -> None:
    totals = tally(reviews)
    status, reason = verdict(totals)
    payload = {
        "a": spec_a.label,
        "b": spec_b.label,
        "verdict": {"status": status, "reason": reason},
        "totals": {k: asdict(v) for k, v in totals.items()},
        "games": [
            {
                "game": r.index,
                "record": asdict(r.record),
                "oracle": [None if s is None else {"status": s[0], "move": s[1]} for s in r.statuses],
                "blunders": [asdict(b) for b in r.blunders],
            }
            for r in reviews
        ],
    }
    path.write_text(json.dumps(payload, indent=1))


def game_line(review: GameReview, done: int, total: int) -> str:
    r = review.record
    outcome = {"a": "A wins", "b": "B wins", None: "draw"}[r.winner]
    colour = "black" if r.a_is_black else "white"
    count = {k: sum(1 for b in review.blunders if b.engine == k) for k in ("a", "b")}
    return (
        f"[{done}/{total}] opening {r.opening} (A {colour}): {outcome} in {len(r.moves)} moves — "
        f"blunders A {count['a']}  B {count['b']}"
    )
