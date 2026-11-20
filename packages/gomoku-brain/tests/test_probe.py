"""Tests for gomoku_brain.probe: parallel probes give the serial answer."""

import random
import threading
import time

import pytest
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.probe import Verdict, probe_in_order
from gomoku_brain.search_state import SearchState


def _state() -> SearchState:
    board = Board(size=15)
    for i, (r, c) in enumerate([(7, 7), (7, 8), (8, 7), (6, 9)]):
        board = board.place_stone(Move(row=r, col=c), Stone.BLACK if i % 2 == 0 else Stone.WHITE)
    return SearchState.from_board(board, Stone.BLACK)


def _candidates(n: int = 12) -> list[tuple[float, Move]]:
    cells = [(r, c) for r in range(3, 12) for c in range(3, 12) if (r, c) not in {(7, 7), (7, 8), (8, 7), (6, 9)}]
    return [(float(-i), Move(row=r, col=c)) for i, (r, c) in enumerate(cells[:n])]


def _probe_from(verdicts: dict[tuple[int, int], Verdict], jitter: random.Random | None = None):
    """A probe that reads the verdict of the move just played, after an
    optional random delay (so threads finish out of order)."""
    lock = threading.Lock()

    def probe(state: SearchState, stop: threading.Event | None) -> Verdict:
        row, col, stone = state.last_move()
        assert stone is Stone.BLACK
        if jitter is not None:
            with lock:
                delay = jitter.random() * 0.01
            time.sleep(delay)
        return verdicts[(row, col)]

    return probe


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("threads", [2, 4, 8])
def test_parallel_answer_matches_serial(seed, threads):
    rng = random.Random(seed)
    candidates = _candidates()
    verdicts = {(m.row, m.col): rng.choice(list(Verdict)) for _, m in candidates}
    state = _state()
    zobrist = state.zobrist
    deadline = time.monotonic() + 60.0

    serial = probe_in_order(state, Stone.BLACK, candidates, _probe_from(verdicts), deadline, 1)
    parallel = probe_in_order(state, Stone.BLACK, candidates, _probe_from(verdicts, rng), deadline, threads)

    assert (parallel.safe, parallel.unproven, parallel.probed) == (serial.safe, serial.unproven, serial.probed)
    assert state.zobrist == zobrist and state.ply == 0


def test_safe_candidate_stops_later_probes():
    candidates = _candidates(8)
    started: list[Move] = []
    stopped = threading.Event()

    def probe(state: SearchState, stop: threading.Event | None) -> Verdict:
        row, col, _ = state.last_move()
        started.append(Move(row=row, col=col))
        if (row, col) == (candidates[0][1].row, candidates[0][1].col):
            time.sleep(0.05)
            return Verdict.SAFE
        # Later candidates run "forever" until told to stop.
        assert stop is not None
        if stop.wait(5.0):
            stopped.set()
        return Verdict.UNPROVEN

    result = probe_in_order(_state(), Stone.BLACK, candidates, probe, time.monotonic() + 60.0, 2)
    assert result.safe == candidates[0][1]
    assert result.unproven is None
    assert stopped.is_set()  # the probe running alongside was cancelled
    # The thread that finished the safe candidate may start one more before
    # the verdict is read; the rest of the queue never starts.
    assert len(started) <= 3


def test_deadline_stops_the_loop():
    candidates = _candidates(8)

    def probe(state: SearchState, stop: threading.Event | None) -> Verdict:
        time.sleep(0.05)
        return Verdict.UNPROVEN

    result = probe_in_order(_state(), Stone.BLACK, candidates, probe, time.monotonic() + 0.08, 2)
    assert result.timed_out
    assert result.safe is None
    assert result.unproven == candidates[0][1]
    assert 2 <= result.probed < len(candidates)


def test_threads_get_their_own_position():
    state = _state()
    seen: set[int] = set()
    lock = threading.Lock()

    def probe(own: SearchState, stop: threading.Event | None) -> Verdict:
        assert own is not state
        with lock:
            seen.add(own.zobrist)
        time.sleep(0.005)
        return Verdict.LOST

    candidates = _candidates(10)
    probe_in_order(state, Stone.BLACK, candidates, probe, time.monotonic() + 60.0, 4)
    expected = set()
    for _, move in candidates:
        state.make(move.row, move.col, Stone.BLACK)
        expected.add(state.zobrist)
        state.unmake()
    assert seen == expected


def test_fork_copies_the_position():
    state = _state()
    state.make(3, 3, Stone.BLACK)
    copy = state.fork()
    assert copy.zobrist == state.zobrist
    assert copy.occupied == state.occupied
    copy.make(4, 4, Stone.WHITE)
    assert state.is_empty(4, 4)
    copy.unmake()
    assert copy.zobrist == state.zobrist
