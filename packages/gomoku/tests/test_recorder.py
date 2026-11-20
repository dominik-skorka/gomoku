"""Tests for game recording and loading."""

import hashlib
from datetime import UTC, datetime

from gomoku.models import GameConfig, GameResult, Move, Stone
from gomoku.recorder import GameRecord, GameRecorder, PlayerInfo, RecordedMove, load_record, save_record

# ── Model tests ─────────────────────────────────────────────────────────────


class TestRecordedMove:
    def test_create(self):
        m = RecordedMove(
            number=1,
            stone=Stone.BLACK,
            move=Move(row=7, col=7),
            timestamp=datetime(2026, 4, 5, 14, 30, 0, tzinfo=UTC),
            think_time_ms=123,
        )
        assert m.number == 1
        assert m.stone is Stone.BLACK
        assert m.think_time_ms == 123

    def test_move_position_in_json(self):
        m = RecordedMove(
            number=1,
            stone=Stone.BLACK,
            move=Move(row=7, col=7),
            timestamp=datetime(2026, 4, 5, 14, 30, 0, tzinfo=UTC),
            think_time_ms=0,
        )
        data = m.model_dump()
        assert data["move"]["position"] == "H8"


class TestPlayerInfo:
    def test_with_settings(self):
        p = PlayerInfo(stone=Stone.BLACK, provider="threat", settings={"depth": 4, "vcf_depth": 20})
        assert p.provider == "threat"
        assert p.settings["depth"] == 4

    def test_without_settings(self):
        p = PlayerInfo(stone=Stone.WHITE, provider="keyboard")
        assert p.settings == {}


# ── Round-trip save / load ──────────────────────────────────────────────────


def _sample_record() -> GameRecord:
    """Create a minimal GameRecord for testing."""
    now = datetime(2026, 4, 5, 14, 30, 0, tzinfo=UTC)
    return GameRecord(
        started_at=now,
        ended_at=datetime(2026, 4, 5, 14, 31, 0, tzinfo=UTC),
        game_config=GameConfig(board_size=9, win_length=5),
        players=[
            PlayerInfo(stone=Stone.BLACK, provider="threat", settings={"depth": 4}),
            PlayerInfo(stone=Stone.WHITE, provider="random"),
        ],
        moves=[
            RecordedMove(
                number=1,
                stone=Stone.BLACK,
                move=Move(row=4, col=4),
                timestamp=now,
                think_time_ms=0,
            ),
            RecordedMove(
                number=2,
                stone=Stone.WHITE,
                move=Move(row=3, col=3),
                timestamp=datetime(2026, 4, 5, 14, 30, 1, tzinfo=UTC),
                think_time_ms=1000,
            ),
        ],
        result=GameResult(winner=Stone.BLACK, winning_moves=[Move(row=4, col=4)]),
    )


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        record = _sample_record()
        path = save_record(record, tmp_path)

        assert path.suffix == ".json"
        assert path.exists()
        assert "gomoku_20260405_143000" in path.name

        loaded = load_record(path)
        assert loaded == record

    def test_creates_directory(self, tmp_path):
        dest = tmp_path / "sub" / "dir"
        record = _sample_record()
        path = save_record(record, dest)
        assert path.exists()

    def test_filename_contains_timestamp(self, tmp_path):
        record = _sample_record()
        path = save_record(record, tmp_path)
        assert "20260405_143000" in path.stem

    def test_load_preserves_game_config(self, tmp_path):
        record = _sample_record()
        path = save_record(record, tmp_path)
        loaded = load_record(path)
        assert loaded.game_config.board_size == 9
        assert loaded.game_config.win_length == 5

    def test_load_preserves_players(self, tmp_path):
        record = _sample_record()
        path = save_record(record, tmp_path)
        loaded = load_record(path)
        assert len(loaded.players) == 2
        assert loaded.players[0].provider == "threat"
        assert loaded.players[0].settings == {"depth": 4}
        assert loaded.players[1].provider == "random"

    def test_load_preserves_moves(self, tmp_path):
        record = _sample_record()
        path = save_record(record, tmp_path)
        loaded = load_record(path)
        assert len(loaded.moves) == 2
        assert loaded.moves[0].think_time_ms == 0
        assert loaded.moves[1].think_time_ms == 1000

    def test_load_preserves_result(self, tmp_path):
        record = _sample_record()
        path = save_record(record, tmp_path)
        loaded = load_record(path)
        assert loaded.result.winner is Stone.BLACK
        assert not loaded.result.is_draw


# ── GameRecorder ────────────────────────────────────────────────────────────


class TestGameRecorder:
    def test_record_and_finalize(self):
        config = GameConfig(board_size=9, win_length=5)
        players = [
            PlayerInfo(stone=Stone.BLACK, provider="alphabeta", settings={"depth": 4}),
            PlayerInfo(stone=Stone.WHITE, provider="random"),
        ]
        recorder = GameRecorder(game_config=config, players=players)

        recorder.record_move(Stone.BLACK, Move(row=4, col=4))
        recorder.record_move(Stone.WHITE, Move(row=3, col=3))

        result = GameResult(winner=Stone.BLACK)
        record = recorder.finalize(result)

        assert record.game_config == config
        assert len(record.moves) == 2
        assert record.moves[0].number == 1
        assert record.moves[0].stone is Stone.BLACK
        assert record.moves[1].number == 2
        assert record.moves[1].stone is Stone.WHITE
        assert record.result.winner is Stone.BLACK
        assert record.started_at <= record.moves[0].timestamp
        assert record.moves[0].timestamp <= record.moves[1].timestamp
        assert record.moves[1].timestamp <= record.ended_at

    def test_think_time_ms_non_negative(self):
        config = GameConfig(board_size=9, win_length=5)
        players = [PlayerInfo(stone=Stone.BLACK, provider="random")]
        recorder = GameRecorder(game_config=config, players=players)

        recorder.record_move(Stone.BLACK, Move(row=0, col=0))
        result = GameResult(winner=Stone.BLACK)
        record = recorder.finalize(result)

        assert record.moves[0].think_time_ms >= 0


# ── Incremental recording ──────────────────────────────────────────────────


def _make_recorder(tmp_path) -> tuple[GameRecorder, GameConfig, list[PlayerInfo]]:
    config = GameConfig(board_size=9, win_length=5)
    players = [
        PlayerInfo(stone=Stone.BLACK, provider="alphabeta", settings={"depth": 4}),
        PlayerInfo(stone=Stone.WHITE, provider="random"),
    ]
    recorder = GameRecorder(game_config=config, players=players, directory=tmp_path)
    return recorder, config, players


class TestIncrementalRecording:
    def test_file_created_on_init(self, tmp_path):
        recorder, *_ = _make_recorder(tmp_path)
        assert recorder.path is not None
        assert recorder.path.exists()
        assert recorder.path.suffix == ".json"

    def test_initial_file_is_loadable(self, tmp_path):
        recorder, config, _players = _make_recorder(tmp_path)
        loaded = load_record(recorder.path)
        assert loaded.game_config == config
        assert loaded.moves == []
        assert loaded.result is None
        assert loaded.ended_at is None

    def test_file_updated_after_each_move(self, tmp_path):
        recorder, *_ = _make_recorder(tmp_path)

        recorder.record_move(Stone.BLACK, Move(row=4, col=4))
        loaded = load_record(recorder.path)
        assert len(loaded.moves) == 1

        recorder.record_move(Stone.WHITE, Move(row=3, col=3))
        loaded = load_record(recorder.path)
        assert len(loaded.moves) == 2

    def test_finalize_writes_result(self, tmp_path):
        recorder, *_ = _make_recorder(tmp_path)
        recorder.record_move(Stone.BLACK, Move(row=4, col=4))

        result = GameResult(winner=Stone.BLACK)
        recorder.finalize(result)

        loaded = load_record(recorder.path)
        assert loaded.result is not None
        assert loaded.result.winner is Stone.BLACK
        assert loaded.ended_at is not None

    def test_finalize_creates_md5sum(self, tmp_path):
        recorder, *_ = _make_recorder(tmp_path)
        recorder.record_move(Stone.BLACK, Move(row=4, col=4))
        recorder.finalize(GameResult(winner=Stone.BLACK))

        md5_path = recorder.path.with_suffix(".md5sum")
        assert md5_path.exists()

        content = md5_path.read_text(encoding="utf-8")
        expected_hash = hashlib.md5(recorder.path.read_bytes()).hexdigest()
        assert content == f"{expected_hash}  {recorder.path.name}\n"

    def test_no_directory_means_no_file(self):
        config = GameConfig(board_size=9, win_length=5)
        players = [PlayerInfo(stone=Stone.BLACK, provider="random")]
        recorder = GameRecorder(game_config=config, players=players)
        assert recorder.path is None

    def test_creates_directory_if_missing(self, tmp_path):
        dest = tmp_path / "sub" / "dir"
        config = GameConfig(board_size=9, win_length=5)
        players = [PlayerInfo(stone=Stone.BLACK, provider="random")]
        recorder = GameRecorder(game_config=config, players=players, directory=dest)
        assert recorder.path.exists()
