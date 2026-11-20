"""Game engine — orchestrates a full Gomoku game."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext

import logfire

from gomoku.board import Board
from gomoku.diagnostics import DiagnosticsRecorder, move_diagnostics
from gomoku.interfaces import GameObserver, Player
from gomoku.models import GameConfig, GameResult, Move, Stone
from gomoku.rules import get_game_status


class InvalidMoveError(Exception):
    """Raised when a Player returns an illegal move."""

    def __init__(self, move: Move, reason: str) -> None:
        self.move = move
        self.reason = reason
        super().__init__(f"Invalid move {move}: {reason}")


class Game:
    """Runs a Gomoku game between two Players.

    *observers* (``GameObserver``) are told about each move and the end
    of the game, for example a :class:`~gomoku.recorder.GameRecorder`.
    With *diagnostics*, each move's diagnostics are collected around the
    player's decision and written to that recorder's file, which also
    observes the game (it writes the summary line at the end).
    """

    def __init__(
        self,
        config: GameConfig,
        black: Player,
        white: Player,
        observers: Iterable[GameObserver] = (),
        *,
        diagnostics: DiagnosticsRecorder | None = None,
    ) -> None:
        self.config = config
        self.board = Board(size=config.board_size)
        self.players: dict[Stone, Player] = {Stone.BLACK: black, Stone.WHITE: white}
        self.current_stone: Stone = Stone.BLACK
        self.move_history: list[tuple[Stone, Move]] = []
        self.result: GameResult | None = None
        self._diagnostics = diagnostics
        self._observers: list[GameObserver] = [*observers, *([diagnostics] if diagnostics is not None else [])]

    def play_turn(self) -> GameResult | None:
        """Execute one turn. Returns GameResult if the game ends, else None."""
        provider = self.players[self.current_stone]

        with logfire.span(
            "gomoku.turn",
            turn=len(self.move_history) + 1,
            stone=str(self.current_stone),
            provider=provider.name,
        ):
            number = len(self.move_history) + 1
            diag_session = (
                move_diagnostics(self._diagnostics, self.board, self.current_stone, number)
                if self._diagnostics is not None
                else nullcontext(None)
            )
            with diag_session as diag:
                move = provider.get_move(self.board, self.current_stone)
                if not self.board.is_valid_move(move):
                    msg = f"cell occupied or out of bounds (board_size={self.config.board_size})"
                    error = InvalidMoveError(move, msg)
                    if diag is not None:
                        diag.notes.append(f"InvalidMoveError: {error}")
                    raise error
                if diag is not None and diag.chosen_move is None:
                    diag.chosen_move = move

            self.board = self.board.place_stone(move, self.current_stone)
            self.move_history.append((self.current_stone, move))
            for observer in self._observers:
                observer.on_move(self.current_stone, move)

            logfire.info("Move played: {stone} -> {move}", stone=str(self.current_stone), move=str(move))

            result = get_game_status(self.board, move, self.config.win_length)
            if result is not None:
                self.result = result
                for observer in self._observers:
                    observer.on_game_end(result)
                return result

            self.current_stone = self.current_stone.opponent()
            return None

    def play(self) -> GameResult:
        """Play the full game until completion."""
        with logfire.span("gomoku.game", board_size=self.config.board_size, win_length=self.config.win_length):
            while True:
                result = self.play_turn()
                if result is not None:
                    if result.is_draw:
                        logfire.info("Game ended in a draw after {turns} turns", turns=len(self.move_history))
                    else:
                        logfire.info(
                            "Game won by {winner} after {turns} turns",
                            winner=str(result.winner),
                            turns=len(self.move_history),
                        )
                    return result
