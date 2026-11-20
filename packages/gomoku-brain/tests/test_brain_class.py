"""The Brain base class: what a new brain gets and what it must provide."""

from typing import override

import pytest
from gomoku.board import Board
from gomoku.diagnostics import MoveDiagnostics, set_move_diagnostics
from gomoku.interfaces import Player
from gomoku.models import Move, Stone
from gomoku.settings import BaseBrainSettings, NoBrainSettings
from gomoku_brain import BRAINS, AlphaBetaBrain, MinimaxBrain, ThreatBrain, get_brain, unregister_brain
from gomoku_brain.base import Brain
from gomoku_brain.brain_base import BrainSession
from pydantic import Field, ValidationError


class _CornerSettings(BaseBrainSettings):
    corner: int = Field(default=0, ge=0, le=3)


@pytest.fixture
def corner_brain():
    """A minimal brain, registered for the test and removed after it."""

    class CornerBrain(Brain[_CornerSettings], name="test_corner"):
        @override
        def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
            last = board.size - 1
            row, col = [(0, 0), (0, last), (last, 0), (last, last)][self.settings.corner]
            if session.diag is not None:
                session.diag.decision_path = "corner"
            return Move(row=row, col=col)

    yield CornerBrain
    unregister_brain("test_corner")


class TestNewBrain:
    def test_declaring_the_class_registers_it(self, corner_brain):
        assert BRAINS["test_corner"].brain_cls is corner_brain
        assert BRAINS["test_corner"].settings_cls is _CornerSettings
        assert isinstance(get_brain("test_corner", corner=3), corner_brain)

    def test_settings_type_comes_from_the_class_statement(self, corner_brain):
        assert corner_brain.settings_type is _CornerSettings
        brain = corner_brain(corner=1)
        assert brain.settings == _CornerSettings(corner=1)
        assert brain.name == "test_corner"

    def test_it_is_a_player_and_plays(self, corner_brain):
        brain = corner_brain(corner=3)
        assert isinstance(brain, Player)
        assert brain.get_move(Board(size=9), Stone.BLACK) == Move(row=8, col=8)

    def test_get_move_hands_the_session_to_choose_move(self, corner_brain):
        diag = MoveDiagnostics(move_number=1, stone=Stone.BLACK)
        set_move_diagnostics(diag)
        try:
            corner_brain().get_move(Board(size=9), Stone.BLACK)
        finally:
            set_move_diagnostics(None)
        assert diag.decision_path == "corner"

    def test_unknown_and_invalid_settings_are_errors(self, corner_brain):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            corner_brain(cornr=1)
        with pytest.raises(ValidationError, match="less than or equal to 3"):
            corner_brain(corner=4)

    def test_a_settings_model_or_keywords_not_both(self, corner_brain):
        assert corner_brain(_CornerSettings(corner=2)).settings.corner == 2
        with pytest.raises(TypeError, match="not both"):
            corner_brain(_CornerSettings(), corner=2)

    def test_a_brain_without_choose_move_cannot_be_made(self):
        class Unfinished(Brain[_CornerSettings]):  # no name: not registered
            pass

        with pytest.raises(TypeError, match="choose_move"):
            Unfinished()

    def test_a_brain_without_settings_gets_none(self):
        class Plain(Brain):
            @override
            def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
                return Move(row=0, col=0)

        assert Plain.settings_type is NoBrainSettings
        with pytest.raises(ValidationError):
            Plain(depth=1)

    def test_names_are_unique(self, corner_brain):
        with pytest.raises(ValueError, match="taken"):

            class Other(Brain[_CornerSettings], name="test_corner"):
                @override
                def choose_move(self, board: Board, stone: Stone, session: BrainSession) -> Move:
                    return Move(row=0, col=0)


class TestBuiltInBrains:
    @pytest.mark.parametrize("name", ["random", "minimax", "negamax", "alphabeta", "threat", "mcts"])
    def test_each_is_a_registered_brain(self, name):
        cls = BRAINS[name].brain_cls
        assert issubclass(cls, Brain)
        assert cls.brain_name == name
        assert BRAINS[name].settings_cls is cls.settings_type

    def test_settings_are_the_single_source_of_defaults(self):
        assert MinimaxBrain().settings.depth == 3
        assert AlphaBetaBrain(depth=6).settings.depth == 6
        assert not hasattr(AlphaBetaBrain(), "depth")  # no copies beside the settings

    def test_threat_fallback_keeps_the_top_level_depth_unless_set(self):
        assert ThreatBrain(depth=6)._fallback.settings.depth == 6
        assert ThreatBrain(depth=6, fallback={"lmr": True})._fallback.settings.depth == 6
        assert ThreatBrain(depth=6, fallback={"depth": 3})._fallback.settings.depth == 3
