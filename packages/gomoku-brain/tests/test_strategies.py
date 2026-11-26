"""Pluggable search strategies (#77): the Evaluator / MoveGenerator protocols."""

import random
from collections.abc import Iterator

import pytest
from gomoku.board import Board
from gomoku.interfaces import BoardView
from gomoku.models import Move, Stone
from gomoku.settings import (
    AlphaBetaBrainSettings,
    BaseBrainSettings,
    BrainSettings,
    MCTSBrainSettings,
    MinimaxBrainSettings,
    NegamaxBrainSettings,
    ThreatBrainSettings,
)
from gomoku.strategies import (
    EVALUATORS,
    MOVE_GENERATORS,
    get_evaluator,
    get_move_generator,
    register_evaluator,
    register_move_generator,
    unregister_strategy,
)
from gomoku_brain import resolve_brain
from gomoku_brain.candidates import get_candidate_moves, radius_generator
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.search_state import SearchState
from pydantic import ValidationError


def _random_board(seed: int, stones: int, size: int = 15) -> Board:
    rng = random.Random(seed)
    board = Board(size=size)
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)
    stone = Stone.BLACK
    for r, c in cells[:stones]:
        board = board.place_stone(Move(row=r, col=c), stone)
        stone = stone.opponent()
    return board


@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_radius_generator_is_get_candidate_moves(radius: int, seed: int) -> None:
    board = _random_board(seed, stones=8)
    generate = radius_generator(radius)
    for stone in (Stone.BLACK, Stone.WHITE):
        assert list(generate(board, stone)) == get_candidate_moves(board, radius)


@pytest.mark.parametrize("seed", range(3))
def test_radius_generator_on_search_state(seed: int) -> None:
    board = _random_board(seed, stones=10)
    state = SearchState.from_board(board, Stone.BLACK)
    assert list(radius_generator()(state, Stone.BLACK)) == get_candidate_moves(state)


def test_radius_generator_empty_board_is_centre() -> None:
    assert list(radius_generator()(Board(size=15), Stone.BLACK)) == [Move(row=7, col=7)]


# ── Registries and settings (#102) ─────────────────────────────────────────


def _stones_only(board: BoardView, stone: Stone, /) -> float:
    """Toy evaluator: own stones minus the opponent's."""
    own = sum(1 for r, c in board.occupied if board.at(r, c) is stone)
    return float(2 * own - len(board.occupied))


@pytest.fixture
def toy_strategies() -> Iterator[None]:
    register_evaluator("stones", _stones_only)
    register_move_generator("near", lambda radius: radius_generator(1))
    yield
    unregister_strategy("stones")
    unregister_strategy("near")


def test_builtins_are_registered() -> None:
    assert EVALUATORS["pattern"] is evaluate_board
    assert get_move_generator("radius", 3)(_random_board(0, 6), Stone.BLACK) == get_candidate_moves(
        _random_board(0, 6), 3
    )


SETTINGS_WITH_BOTH = [MinimaxBrainSettings, NegamaxBrainSettings, AlphaBetaBrainSettings]


@pytest.mark.parametrize("model", [*SETTINGS_WITH_BOTH, MCTSBrainSettings])
def test_defaults_are_the_builtins(model: type[BaseBrainSettings]) -> None:
    settings = model()
    assert settings.evaluator == "pattern"  # type: ignore[attr-defined]
    if model is not MCTSBrainSettings:
        assert settings.move_generator == "radius"  # type: ignore[attr-defined]


@pytest.mark.parametrize("model", [*SETTINGS_WITH_BOTH, MCTSBrainSettings])
def test_unknown_evaluator_is_a_settings_error(model: type[BaseBrainSettings]) -> None:
    with pytest.raises(ValidationError, match=r"unknown evaluator 'nope'; registered: .*pattern"):
        model(evaluator="nope")


@pytest.mark.parametrize("model", SETTINGS_WITH_BOTH)
def test_unknown_move_generator_is_a_settings_error(model: type[BaseBrainSettings]) -> None:
    with pytest.raises(ValidationError, match=r"unknown move generator 'nope'; registered: .*radius"):
        model(move_generator="nope")


def test_mcts_has_no_move_generator_setting() -> None:
    with pytest.raises(ValidationError, match="move_generator"):
        MCTSBrainSettings(move_generator="radius")  # type: ignore[call-arg]


@pytest.mark.usefixtures("toy_strategies")
def test_registered_names_are_accepted() -> None:
    settings = AlphaBetaBrainSettings(evaluator="stones", move_generator="near")
    assert (settings.evaluator, settings.move_generator) == ("stones", "near")
    assert get_evaluator("stones") is _stones_only


def test_threat_fallback_validates_its_evaluator() -> None:
    with pytest.raises(ValidationError, match=r"fallback\.evaluator"):
        ThreatBrainSettings.model_validate({"fallback": {"evaluator": "nope"}})


@pytest.mark.usefixtures("toy_strategies")
def test_toml_alias_resolves_with_a_registered_evaluator() -> None:
    brains = BrainSettings.model_validate({"ab_stones": {"type": "alphabeta", "evaluator": "stones"}})
    brain_type, kwargs = resolve_brain("ab_stones", brains)
    assert (brain_type, kwargs["evaluator"], kwargs["move_generator"]) == ("alphabeta", "stones", "radius")


def test_toml_alias_with_unknown_evaluator_fails() -> None:
    brains = BrainSettings.model_validate({"ab_bad": {"type": "alphabeta", "evaluator": "nope"}})
    with pytest.raises(ValidationError, match="unknown evaluator 'nope'"):
        resolve_brain("ab_bad", brains)


def test_unregister_removes_both() -> None:
    register_evaluator("tmp", _stones_only)
    register_move_generator("tmp", radius_generator)
    unregister_strategy("tmp")
    assert "tmp" not in EVALUATORS and "tmp" not in MOVE_GENERATORS
