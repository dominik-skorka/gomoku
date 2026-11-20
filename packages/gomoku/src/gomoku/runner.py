"""Stateful game runner — supports external players, brains, or a mix.

:class:`gomoku.game.Game` requires a ``Player`` for both colours
and drives them itself.  That's the right shape for the CLI's ``play``
command (two configured brains taking turns) but doesn't fit the API:
a stateful session may have a *human* player on one side whose moves
arrive over HTTP, while the brain on the other side is invoked via a
separate ``POST .../think`` call.

:class:`GameRunner` is the alternative loop — neither side is
required to have a brain (``None`` means "external"), and applying a
move is the unit of work the caller drives.  The state machine is
identical to ``Game.play_turn`` (validate → place stone → record →
check win → toggle side); we just expose it as two public methods
:meth:`submit_move` and :meth:`think` so the API can mix-and-match.

Recording and diagnostics are wired the same way as in ``Game`` (game
observers and a ``DiagnosticsRecorder``), so a session that opts into
``record=True`` produces files that ``gomoku playback`` and ``gomoku
analyze`` consume without modification.
"""

from collections.abc import Iterable

from gomoku.board import Board
from gomoku.diagnostics import DiagnosticsRecorder, MoveDiagnostics, move_diagnostics
from gomoku.game import InvalidMoveError
from gomoku.interfaces import GameObserver, Player
from gomoku.models import GameConfig, GameResult, Move, Stone
from gomoku.rules import get_game_status


class GameAlreadyOverError(Exception):
    """Raised when a move is submitted to a session whose game has finished."""

    def __init__(self, result: GameResult) -> None:
        self.result = result
        winner = "draw" if result.is_draw else f"won by {result.winner}"
        super().__init__(f"Game already over ({winner})")


class WrongTurnError(Exception):
    """Raised when a move is submitted for the wrong side."""

    def __init__(self, expected: Stone, got: Stone) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"Expected move from {expected}, got {got}")


class NoBrainConfiguredError(Exception):
    """Raised when ``think()`` is called for a side configured as external."""

    def __init__(self, stone: Stone) -> None:
        self.stone = stone
        super().__init__(f"No brain configured for {stone}; submit the move externally")


class GameRunner:
    """A :class:`Game`-like loop where each side may be brain-driven OR external.

    Construction takes the game config and an optional brain per side.
    A ``None`` brain marks that side as external — the only way to
    advance the game on that side is :meth:`submit_move`.

    *observers* and an optional :class:`DiagnosticsRecorder` plug in the
    same recording and diagnostics machinery as :class:`Game`.
    """

    def __init__(
        self,
        config: GameConfig,
        *,
        black: Player | None = None,
        white: Player | None = None,
        diagnostics_recorder: DiagnosticsRecorder | None = None,
        observers: Iterable[GameObserver] = (),
    ) -> None:
        self.config = config
        self.board = Board(size=config.board_size)
        self.brains: dict[Stone, Player | None] = {Stone.BLACK: black, Stone.WHITE: white}
        self.current_stone: Stone = Stone.BLACK
        self.move_history: list[tuple[Stone, Move]] = []
        self.result: GameResult | None = None
        self._diag_recorder = diagnostics_recorder
        self._observers: list[GameObserver] = [
            *observers,
            *([diagnostics_recorder] if diagnostics_recorder is not None else []),
        ]

    # -- introspection --------------------------------------------------------

    def is_external(self, stone: Stone) -> bool:
        """``True`` when *stone*'s moves come from outside the runner."""
        return self.brains[stone] is None

    def brain_name(self, stone: Stone) -> str | None:
        brain = self.brains[stone]
        return None if brain is None else brain.name

    # -- core ----------------------------------------------------------------

    def submit_move(self, stone: Stone, move: Move) -> GameResult | None:
        """Apply an externally-provided move.

        Returns the :class:`GameResult` when the move ends the game,
        ``None`` otherwise.  Raises :class:`GameAlreadyOverError`,
        :class:`WrongTurnError`, or :class:`InvalidMoveError` when the
        submission is rejected — each exception carries enough detail
        for the API to surface a clean 4xx.
        """
        if self.result is not None:
            raise GameAlreadyOverError(self.result)
        if stone is not self.current_stone:
            raise WrongTurnError(self.current_stone, stone)
        if not self.board.is_valid_move(move):
            raise InvalidMoveError(move, f"cell occupied or out of bounds (board_size={self.config.board_size})")
        return self._apply(stone, move)

    def think(self, stone: Stone) -> tuple[Move, GameResult | None, MoveDiagnostics | None]:
        """Run the configured brain for *stone* and apply its move.

        Activates a :class:`MoveDiagnostics` ContextVar around the
        brain call so the brain populates it during search; returned
        as the third tuple element.  When the runner has a
        :class:`DiagnosticsRecorder`, the diagnostics record is also
        appended to the ``.diag.jsonl`` file.

        Raises :class:`NoBrainConfiguredError` when *stone* is external,
        :class:`GameAlreadyOverError` when the game has finished, or
        :class:`WrongTurnError` when *stone* isn't the side to move.
        """
        if self.result is not None:
            raise GameAlreadyOverError(self.result)
        if stone is not self.current_stone:
            raise WrongTurnError(self.current_stone, stone)
        brain = self.brains[stone]
        if brain is None:
            raise NoBrainConfiguredError(stone)

        # Diagnostics are always collected, so the API can return them;
        # with a recorder they are also written to its file.
        with move_diagnostics(self._diag_recorder, self.board, stone, len(self.move_history) + 1) as diag:
            move = brain.get_move(self.board, stone)
            if diag.chosen_move is None:
                diag.chosen_move = move

        # ``submit_move`` would re-validate; we already know whose
        # turn it is and we've already produced a candidate, so go
        # straight to ``_apply`` for the state mutation.  We DO check
        # legality — a buggy brain returning an occupied cell should
        # surface as InvalidMoveError, not corrupt the board.
        if not self.board.is_valid_move(move):
            raise InvalidMoveError(move, "brain returned an illegal move")
        result = self._apply(stone, move)
        return move, result, diag

    # -- internals ------------------------------------------------------------

    def _apply(self, stone: Stone, move: Move) -> GameResult | None:
        self.board = self.board.place_stone(move, stone)
        self.move_history.append((stone, move))
        for observer in self._observers:
            observer.on_move(stone, move)
        result = get_game_status(self.board, move, self.config.win_length)
        if result is not None:
            self.result = result
            for observer in self._observers:
                observer.on_game_end(result)
            return result
        self.current_stone = self.current_stone.opponent()
        return None


__all__ = [
    "GameAlreadyOverError",
    "GameRunner",
    "NoBrainConfiguredError",
    "WrongTurnError",
]
