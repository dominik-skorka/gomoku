"""Tests for :class:`gomoku.runner.GameRunner`.

The runner is what the FastAPI sessions endpoint sits on top of.  It's
deliberately a stripped-down, externally-driven sibling of
:class:`gomoku.game.Game` — these tests pin the contract that the API
relies on:

1. External-only sessions: both sides accept ``submit_move`` only.
2. Brain-only sessions: ``think`` runs the brain, applies the move,
   produces a diagnostics record.
3. Mixed sessions: each side uses its own path.
4. State machine: wrong-turn / game-over / illegal moves raise
   typed exceptions the API can convert into clean 4xx responses.
"""

import pytest
from gomoku.game import InvalidMoveError
from gomoku.models import GameConfig, GameResult, Move, Stone
from gomoku.protocol import MoveProvider
from gomoku.runner import (
    GameAlreadyOverError,
    GameRunner,
    NoBrainConfiguredError,
    WrongTurnError,
)


class _FixedBrain(MoveProvider):
    """Always returns ``move``; lets tests assert against a known move."""

    def __init__(self, name: str, move: Move) -> None:
        self._name = name
        self._move = move

    @property
    def name(self) -> str:
        return self._name

    def get_move(self, board, stone) -> Move:
        return self._move


# ---------------------------------------------------------------------------
# Construction + introspection
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_external_only_starts_at_black(self):
        r = GameRunner(GameConfig(board_size=9, win_length=5))
        assert r.current_stone is Stone.BLACK
        assert r.is_external(Stone.BLACK)
        assert r.is_external(Stone.WHITE)

    def test_brain_name_returns_none_when_external(self):
        r = GameRunner(GameConfig(board_size=9, win_length=5))
        assert r.brain_name(Stone.BLACK) is None

    def test_brain_name_returns_provider_name(self):
        b = _FixedBrain("alphabeta", Move(row=4, col=4))
        r = GameRunner(GameConfig(board_size=9, win_length=5), black=b)
        assert r.brain_name(Stone.BLACK) == "alphabeta"


# ---------------------------------------------------------------------------
# submit_move (external player path)
# ---------------------------------------------------------------------------


class TestSubmitMove:
    def test_advances_state(self):
        r = GameRunner(GameConfig(board_size=9, win_length=5))
        r.submit_move(Stone.BLACK, Move(row=4, col=4))
        assert r.current_stone is Stone.WHITE
        assert r.move_history == [(Stone.BLACK, Move(row=4, col=4))]
        assert r.result is None

    def test_rejects_wrong_turn(self):
        r = GameRunner(GameConfig(board_size=9, win_length=5))
        with pytest.raises(WrongTurnError) as exc:
            r.submit_move(Stone.WHITE, Move(row=4, col=4))
        assert exc.value.expected is Stone.BLACK
        assert exc.value.got is Stone.WHITE

    def test_rejects_illegal_move(self):
        r = GameRunner(GameConfig(board_size=9, win_length=5))
        r.submit_move(Stone.BLACK, Move(row=4, col=4))
        # White tries to play on the cell Black just took.
        with pytest.raises(InvalidMoveError):
            r.submit_move(Stone.WHITE, Move(row=4, col=4))

    def test_returns_result_on_winning_move(self):
        r = GameRunner(GameConfig(board_size=9, win_length=5))
        # Black plays four-in-a-row on row 4, then completes the line.
        sequence = [
            (Stone.BLACK, Move(row=4, col=2)),
            (Stone.WHITE, Move(row=0, col=0)),
            (Stone.BLACK, Move(row=4, col=3)),
            (Stone.WHITE, Move(row=0, col=1)),
            (Stone.BLACK, Move(row=4, col=4)),
            (Stone.WHITE, Move(row=0, col=2)),
            (Stone.BLACK, Move(row=4, col=5)),
            (Stone.WHITE, Move(row=0, col=3)),
        ]
        for stone, move in sequence:
            r.submit_move(stone, move)
        # Final winning move.
        result = r.submit_move(Stone.BLACK, Move(row=4, col=6))
        assert result is not None
        assert result.winner is Stone.BLACK
        assert r.result is result

    def test_rejects_moves_after_game_over(self):
        r = GameRunner(GameConfig(board_size=9, win_length=5))
        # Win for Black in 5 moves on row 4 (skipping the alternating pattern).
        for c in range(2, 6):
            r.submit_move(Stone.BLACK, Move(row=4, col=c))
            # White plays away each time.
            r.submit_move(Stone.WHITE, Move(row=0, col=c))
        r.submit_move(Stone.BLACK, Move(row=4, col=6))  # winning move
        assert r.result is not None
        # Any further move must raise.
        with pytest.raises(GameAlreadyOverError):
            r.submit_move(Stone.WHITE, Move(row=1, col=1))


# ---------------------------------------------------------------------------
# think (brain-driven path)
# ---------------------------------------------------------------------------


class TestThink:
    def test_runs_brain_and_applies_move(self):
        b = _FixedBrain("fixed", Move(row=4, col=4))
        r = GameRunner(GameConfig(board_size=9, win_length=5), black=b)
        move, result, diag = r.think(Stone.BLACK)
        assert move == Move(row=4, col=4)
        assert result is None
        assert r.current_stone is Stone.WHITE
        # Diagnostics must be populated even when no recorder is
        # configured — the API needs to return them on the response.
        assert diag is not None
        assert diag.move_number == 1
        assert diag.stone is Stone.BLACK

    def test_raises_when_side_is_external(self):
        # Only BLACK has a brain.  Calling think(WHITE) must surface as a
        # NoBrainConfiguredError so the API can return a clean 422.
        b = _FixedBrain("fixed", Move(row=4, col=4))
        r = GameRunner(GameConfig(board_size=9, win_length=5), black=b)
        r.think(Stone.BLACK)  # advance to WHITE's turn
        with pytest.raises(NoBrainConfiguredError) as exc:
            r.think(Stone.WHITE)
        assert exc.value.stone is Stone.WHITE

    def test_raises_when_called_for_wrong_side(self):
        b_black = _FixedBrain("b", Move(row=4, col=4))
        b_white = _FixedBrain("w", Move(row=3, col=3))
        r = GameRunner(GameConfig(board_size=9, win_length=5), black=b_black, white=b_white)
        # WHITE's brain shouldn't be invoked when it's BLACK's turn.
        with pytest.raises(WrongTurnError):
            r.think(Stone.WHITE)

    def test_invalid_move_from_brain_raises_invalidmove(self):
        # Brain returns a cell that's already occupied — runner must
        # catch this BEFORE corrupting the board.
        bad_brain = _FixedBrain("bad", Move(row=4, col=4))
        good_brain = _FixedBrain("good", Move(row=4, col=4))
        r = GameRunner(GameConfig(board_size=9, win_length=5), black=good_brain, white=bad_brain)
        r.think(Stone.BLACK)  # places at (4, 4)
        with pytest.raises(InvalidMoveError):
            r.think(Stone.WHITE)
        # Board state must still be coherent — only Black's move applied.
        assert r.move_history == [(Stone.BLACK, Move(row=4, col=4))]


# ---------------------------------------------------------------------------
# Mixed sessions (one brain + one external)
# ---------------------------------------------------------------------------


class TestMixed:
    def test_external_then_brain(self):
        # The classic API shape: human plays Black via submit_move,
        # then the server's brain plays White via think.
        ai = _FixedBrain("ai", Move(row=3, col=3))
        r = GameRunner(GameConfig(board_size=9, win_length=5), white=ai)
        r.submit_move(Stone.BLACK, Move(row=4, col=4))
        move, result, diag = r.think(Stone.WHITE)
        assert move == Move(row=3, col=3)
        assert result is None
        assert diag is not None
        assert r.current_stone is Stone.BLACK
        assert len(r.move_history) == 2


# ---------------------------------------------------------------------------
# Game observers (recorder hook)
# ---------------------------------------------------------------------------


class _Log:
    """A GameObserver that logs what it is told."""

    def __init__(self) -> None:
        self.moves: list[tuple[Stone, Move]] = []
        self.ends: list[GameResult] = []

    def on_move(self, stone: Stone, move: Move) -> None:
        self.moves.append((stone, move))

    def on_game_end(self, result: GameResult) -> None:
        self.ends.append(result)


class TestObservers:
    def test_told_each_move(self):
        log = _Log()
        r = GameRunner(GameConfig(board_size=9, win_length=5), observers=[log])
        r.submit_move(Stone.BLACK, Move(row=4, col=4))
        r.submit_move(Stone.WHITE, Move(row=3, col=3))
        assert log.moves == [(Stone.BLACK, Move(row=4, col=4)), (Stone.WHITE, Move(row=3, col=3))]
        assert log.ends == []

    def test_not_told_a_rejected_move(self):
        log = _Log()
        r = GameRunner(GameConfig(board_size=9, win_length=5), observers=[log])
        with pytest.raises(WrongTurnError):
            r.submit_move(Stone.WHITE, Move(row=4, col=4))
        assert log.moves == []

    def test_told_the_end_once(self):
        log = _Log()
        r = GameRunner(GameConfig(board_size=9, win_length=5), observers=[log])
        for col in range(4):
            r.submit_move(Stone.BLACK, Move(row=0, col=col))
            r.submit_move(Stone.WHITE, Move(row=8, col=col))
        result = r.submit_move(Stone.BLACK, Move(row=0, col=4))
        assert result is not None and log.ends == [result]
        assert len(log.moves) == 9
