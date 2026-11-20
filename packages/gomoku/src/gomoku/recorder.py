"""Game recording and loading — captures a full game for later playback."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gomoku.models import GameConfig, GameResult, Move, Stone


class RecordedMove(BaseModel, frozen=True):
    """A single move with timing metadata."""

    number: int = Field(ge=1)
    stone: Stone
    move: Move
    timestamp: datetime
    think_time_ms: int = Field(ge=0)


class PlayerInfo(BaseModel, frozen=True):
    """Identity and configuration of one player."""

    stone: Stone
    provider: str
    settings: dict[str, Any] = {}


class GameRecord(BaseModel, frozen=True):
    """Complete recording of a finished game."""

    format_version: int = 1
    started_at: datetime
    ended_at: datetime | None = None
    game_config: GameConfig
    players: list[PlayerInfo]
    moves: list[RecordedMove] = []
    result: GameResult | None = None


# ── File I/O ────────────────────────────────────────────────────────────────


def save_record(record: GameRecord, directory: Path) -> Path:
    """Save a game record as JSON. Returns the written file path."""
    directory.mkdir(parents=True, exist_ok=True)
    ts = record.started_at.strftime("%Y%m%d_%H%M%S")
    filename = f"gomoku_{ts}.json"
    path = directory / filename
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_record(path: Path) -> GameRecord:
    """Load a game record from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return GameRecord.model_validate(data)


# ── Recorder helper ─────────────────────────────────────────────────────────


class GameRecorder:
    """Accumulates moves during a live game and produces a GameRecord.

    When *directory* is provided the recorder creates the JSON file
    immediately on construction and rewrites it after every move, so
    that partial recordings survive crashes.  On :meth:`finalize` it
    writes the completed file and a companion ``.md5sum`` checksum.
    """

    def __init__(
        self,
        game_config: GameConfig,
        players: list[PlayerInfo],
        directory: Path | None = None,
    ) -> None:
        self._game_config = game_config
        self._players = players
        self._moves: list[RecordedMove] = []
        self._started_at: datetime = datetime.now(tz=UTC)
        self._last_timestamp: datetime = self._started_at

        # Incremental file I/O
        self._path: Path | None = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            ts = self._started_at.strftime("%Y%m%d_%H%M%S")
            self._path = directory / f"gomoku_{ts}.json"
            self._write()

    @property
    def path(self) -> Path | None:
        """Path to the recording file, or *None* if no directory was given."""
        return self._path

    def record_move(self, stone: Stone, move: Move) -> None:
        """Record a move with its timestamp and think time."""
        now = datetime.now(tz=UTC)
        think_ms = int((now - self._last_timestamp).total_seconds() * 1000)
        self._moves.append(
            RecordedMove(
                number=len(self._moves) + 1,
                stone=stone,
                move=move,
                timestamp=now,
                think_time_ms=think_ms,
            )
        )
        self._last_timestamp = now
        self._write()

    # -- GameObserver ---------------------------------------------------------

    def on_move(self, stone: Stone, move: Move) -> None:
        """Record each move (``GameObserver``)."""
        self.record_move(stone, move)

    def on_game_end(self, result: GameResult) -> None:
        """Finalize the recording (``GameObserver``)."""
        self.finalize(result)

    def finalize(self, result: GameResult) -> GameRecord:
        """Build the completed GameRecord, write it, and produce an MD5 checksum."""
        record = GameRecord(
            started_at=self._started_at,
            ended_at=datetime.now(tz=UTC),
            game_config=self._game_config,
            players=self._players,
            moves=self._moves,
            result=result,
        )
        if self._path is not None:
            self._path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            md5 = hashlib.md5(self._path.read_bytes()).hexdigest()
            self._path.with_suffix(".md5sum").write_text(f"{md5}  {self._path.name}\n", encoding="utf-8")
        return record

    # -- private helpers -----------------------------------------------------

    def _write(self) -> None:
        """Rewrite the recording file with the current state."""
        if self._path is None:
            return
        snapshot = GameRecord(
            started_at=self._started_at,
            game_config=self._game_config,
            players=self._players,
            moves=self._moves,
        )
        self._path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
