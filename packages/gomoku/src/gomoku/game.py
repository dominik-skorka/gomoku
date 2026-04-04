"""Game engine — orchestrates a full Gomoku game."""

from __future__ import annotations

import logfire

from gomoku.board import Board
from gomoku.models import GameConfig, GameResult, Move, Stone
from gomoku.protocol import MoveProvider
from gomoku.rules import get_game_status


class InvalidMoveError(Exception):
    """Raised when a MoveProvider returns an illegal move."""

    def __init__(self, move: Move, reason: str) -> None:
        self.move = move
        self.reason = reason
        super().__init__(f"Invalid move {move}: {reason}")


class Game:
    """Runs a Gomoku game between two MoveProviders."""

    def __init__(self, config: GameConfig, black: MoveProvider, white: MoveProvider) -> None:
        self.config = config
        self.board = Board(size=config.board_size)
        self.players: dict[Stone, MoveProvider] = {Stone.BLACK: black, Stone.WHITE: white}
        self.current_stone: Stone = Stone.BLACK
        self.move_history: list[tuple[Stone, Move]] = []

    def play_turn(self) -> GameResult | None:
        """Execute one turn. Returns GameResult if the game ends, else None."""
        provider = self.players[self.current_stone]

        with logfire.span(
            "gomoku.turn",
            turn=len(self.move_history) + 1,
            stone=str(self.current_stone),
            provider=provider.name,
        ):
            move = provider.get_move(self.board, self.current_stone)

            if not self.board.is_valid_move(move):
                msg = f"cell occupied or out of bounds (board_size={self.config.board_size})"
                raise InvalidMoveError(move, msg)

            self.board = self.board.place_stone(move, self.current_stone)
            self.move_history.append((self.current_stone, move))

            logfire.info("Move played: {stone} -> {move}", stone=str(self.current_stone), move=str(move))

            result = get_game_status(self.board, move, self.config.win_length)
            if result is not None:
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
