from gomoku.models import GameConfig, GameResult, Move, Stone


class TestStone:
    def test_values(self):
        assert Stone.EMPTY == 0
        assert Stone.BLACK == 1
        assert Stone.WHITE == 2

    def test_opponent(self):
        assert Stone.BLACK.opponent() is Stone.WHITE
        assert Stone.WHITE.opponent() is Stone.BLACK

    def test_str(self):
        assert str(Stone.BLACK) == "Black"
        assert str(Stone.WHITE) == "White"


class TestMove:
    def test_creation(self):
        m = Move(row=3, col=7)
        assert m.row == 3
        assert m.col == 7

    def test_str_notation(self):
        assert str(Move(row=0, col=0)) == "A1"
        assert str(Move(row=14, col=14)) == "O15"

    def test_position_field(self):
        m = Move(row=7, col=7)
        assert m.position == "H8"

    def test_position_in_serialized_output(self):
        data = Move(row=0, col=0).model_dump()
        assert data["position"] == "A1"

    def test_frozen(self):
        m = Move(row=0, col=0)
        try:
            m.row = 1  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass

    def test_negative_row_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            Move(row=-1, col=0)


class TestGameConfig:
    def test_defaults(self):
        cfg = GameConfig()
        assert cfg.board_size == 15
        assert cfg.win_length == 5

    def test_custom(self):
        cfg = GameConfig(board_size=19, win_length=5)
        assert cfg.board_size == 19

    def test_win_length_too_large(self):
        import pytest

        with pytest.raises(ValueError, match="cannot exceed"):
            GameConfig(board_size=5, win_length=6)


class TestGameResult:
    def test_draw(self):
        r = GameResult()
        assert r.is_draw
        assert r.winner is None

    def test_winner(self):
        r = GameResult(winner=Stone.BLACK, winning_moves=[Move(row=0, col=i) for i in range(5)])
        assert not r.is_draw
        assert r.winner is Stone.BLACK
        assert len(r.winning_moves) == 5
