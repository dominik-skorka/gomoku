"""Probing candidate moves for the opponent's forced win, on one or more threads.

Both brains end their forced-win defence the same way: candidate replies,
best evaluation first, are each checked with a threat search for the
opponent, and the first one after which the opponent provably has no
forced win is played.  :func:`probe_in_order` runs that loop.

With ``threads`` > 1 the candidates are checked in parallel, each thread
on its own copy of the position (:meth:`SearchState.fork`).  The answer is
the one the serial loop would give from the same probe results: the
*first* safe candidate in evaluation order, not the first to finish, and
the first unproven one before it.  So more threads change the move only
through time: within the same deadline they probe more candidates (and a
probe finishing sooner or later can still change its own result).

Threads pay off on a free-threaded Python (``python3.14t``), where the
searches run truly in parallel.  With the GIL they take turns, so the
probes only interleave and ``threads`` > 1 brings no speed-up, just a
little overhead.
"""

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

from gomoku.models import Move, Stone

from gomoku_brain.search_state import SearchState


class Verdict(Enum):
    """What a probe found out about the opponent's forced win."""

    SAFE = auto()  # the opponent provably has none: play this candidate
    LOST = auto()  # the opponent has one: skip this candidate
    UNPROVEN = auto()  # the probe ran out of budget


# ``probe(state, stop)`` runs on *state* with the candidate already played
# and returns its verdict.  It should give up (``UNPROVEN``) once *stop* is
# set; ``stop`` is ``None`` on the serial path.
Probe = Callable[[SearchState, threading.Event | None], Verdict]


class ProbeResult:
    """Outcome of :func:`probe_in_order`."""

    __slots__ = ("probed", "safe", "timed_out", "unproven")

    def __init__(self) -> None:
        self.safe: Move | None = None  # first safe candidate
        self.unproven: Move | None = None  # first unproven candidate before it
        self.probed = 0  # candidates whose probe started
        self.timed_out = False  # the deadline stopped the loop


def probe_in_order(
    state: SearchState,
    stone: Stone,
    candidates: Sequence[tuple[float, Move]],
    probe: Probe,
    deadline: float,
    threads: int = 1,
) -> ProbeResult:
    """Probe *candidates* (``(score, move)``, best first) for *stone*.

    Each candidate is played on *state* for *stone* and handed to *probe*.
    Stops at the first ``SAFE`` verdict, or when *deadline*
    (``time.monotonic()``) passes before a probe starts.  *state* is left
    as it was.  See the module docstring for ``threads``.
    """
    if threads <= 1 or len(candidates) <= 1:
        return _serial(state, stone, candidates, probe, deadline)
    return _parallel(state, stone, candidates, probe, deadline, threads)


def _serial(
    state: SearchState,
    stone: Stone,
    candidates: Sequence[tuple[float, Move]],
    probe: Probe,
    deadline: float,
) -> ProbeResult:
    result = ProbeResult()
    for _score, move in candidates:
        if time.monotonic() > deadline:
            result.timed_out = True
            break
        result.probed += 1
        state.make(move.row, move.col, stone)
        try:
            verdict = probe(state, None)
        finally:
            state.unmake()
        if verdict is Verdict.SAFE:
            result.safe = move
            return result
        if verdict is Verdict.UNPROVEN and result.unproven is None:
            result.unproven = move
    return result


def _parallel(
    state: SearchState,
    stone: Stone,
    candidates: Sequence[tuple[float, Move]],
    probe: Probe,
    deadline: float,
    threads: int,
) -> ProbeResult:
    workers = min(threads, len(candidates))
    # One private copy of the position per worker, handed out per probe.
    states: queue.SimpleQueue[SearchState] = queue.SimpleQueue()
    for _ in range(workers):
        states.put(state.fork())
    stop = threading.Event()

    def run(move: Move) -> Verdict | None:
        # None: not started (deadline passed, or an earlier candidate
        # already proved safe).
        if stop.is_set() or time.monotonic() > deadline:
            return None
        own = states.get()
        try:
            own.make(move.row, move.col, stone)
            try:
                return probe(own, stop)
            finally:
                own.unmake()
        finally:
            states.put(own)

    result = ProbeResult()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="probe") as pool:
        # The pool starts tasks in submission order, so earlier
        # (better-evaluated) candidates are always probed first.
        futures = [pool.submit(run, move) for _score, move in candidates]
        try:
            # Read verdicts in candidate order: a safe candidate counts
            # only once every better one has been ruled out, exactly as
            # in the serial loop.
            for future, (_score, move) in zip(futures, candidates, strict=True):
                verdict = future.result()
                if verdict is None:
                    result.timed_out = not stop.is_set()
                    break
                result.probed += 1
                if verdict is Verdict.SAFE:
                    result.safe = move
                    break
                if verdict is Verdict.UNPROVEN and result.unproven is None:
                    result.unproven = move
        finally:
            # Later probes are no longer needed: cancel the queued ones and
            # tell the running ones to give up.
            stop.set()
            for future in futures:
                future.cancel()
    return result
