"""Pluggable search strategies (#77): the Evaluator / MoveGenerator protocols."""

import random
from collections.abc import Iterator

import pytest
from gomoku.board import Board
from gomoku.interfaces import BoardView, MoveGenerator
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
from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.candidates import get_candidate_moves, radius_generator
from gomoku_brain.evaluate import _EVAL_CLAMP, evaluate_board
from gomoku_brain.mcts import MCTSBrain
from gomoku_brain.minimax import MinimaxBrain
from gomoku_brain.negamax import NegamaxBrain
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat import ThreatBrain
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


# ── Minimax / negamax use the chosen strategies (#103) ─────────────────────

SIMPLE_BRAINS = [MinimaxBrain, NegamaxBrain]


@pytest.mark.parametrize("brain_cls", SIMPLE_BRAINS)
def test_defaults_resolve_to_the_builtins(brain_cls: type[MinimaxBrain | NegamaxBrain]) -> None:
    brain = brain_cls()
    assert brain._evaluate is evaluate_board  # the same function: no behaviour change
    assert brain._candidates.__qualname__ == "radius_generator(2)"


def _corner_only(radius: int) -> MoveGenerator:
    """Toy generator: the first empty cell in reading order."""

    def generate(board: BoardView, stone: Stone, /) -> list[Move]:
        size = board.size
        return [next(Move(row=r, col=c) for r in range(size) for c in range(size) if board.at(r, c) is Stone.EMPTY)]

    return generate


@pytest.fixture
def corner_generator() -> Iterator[None]:
    register_move_generator("corner", _corner_only)
    yield
    unregister_strategy("corner")


@pytest.mark.usefixtures("corner_generator")
@pytest.mark.parametrize("brain_cls", SIMPLE_BRAINS)
def test_generator_by_name_sets_the_moves_searched(brain_cls: type[MinimaxBrain | NegamaxBrain]) -> None:
    board = _random_board(1, stones=6)
    move = brain_cls(depth=2, move_generator="corner").get_move(board, Stone.BLACK)
    assert move == _corner_only(2)(board, Stone.BLACK)[0]


def _prefer_centre(board: BoardView, stone: Stone, /) -> float:
    """Toy evaluator: own stones near the centre are good."""
    centre = board.size // 2
    return -float(sum(abs(r - centre) + abs(c - centre) for r, c in board.occupied if board.at(r, c) is stone))


@pytest.fixture
def centre_evaluator() -> Iterator[None]:
    register_evaluator("centre", _prefer_centre)
    yield
    unregister_strategy("centre")


@pytest.mark.usefixtures("centre_evaluator")
@pytest.mark.parametrize("brain_cls", SIMPLE_BRAINS)
def test_evaluator_by_name_sets_the_choice(brain_cls: type[MinimaxBrain | NegamaxBrain]) -> None:
    # Black on the 5,5-6,6 diagonal: "pattern" blocks it at (8, 8), the toy
    # evaluator goes to the centre (7, 7).
    board = Board(size=15).place_stone(Move(row=5, col=5), Stone.BLACK)
    board = board.place_stone(Move(row=9, col=9), Stone.WHITE)
    board = board.place_stone(Move(row=6, col=6), Stone.BLACK)
    assert brain_cls(depth=1).get_move(board, Stone.WHITE) == Move(row=8, col=8)
    assert brain_cls(depth=1, evaluator="centre").get_move(board, Stone.WHITE) == Move(row=7, col=7)


@pytest.mark.usefixtures("toy_strategies", "corner_generator", "centre_evaluator")
@pytest.mark.parametrize("brain_cls", SIMPLE_BRAINS)
def test_every_registered_strategy_plays_legal_moves(brain_cls: type[MinimaxBrain | NegamaxBrain]) -> None:
    for evaluator in sorted(EVALUATORS):
        for generator in sorted(MOVE_GENERATORS):
            brain = brain_cls(depth=2, evaluator=evaluator, move_generator=generator)
            for seed in range(3):
                board = _random_board(seed, stones=7)
                move = brain.get_move(board, Stone.BLACK)
                assert board.at(move.row, move.col) is Stone.EMPTY, (evaluator, generator, seed)


# ── Alpha-beta, threat fallback, MCTS (#104) ───────────────────────────────


@pytest.mark.usefixtures("toy_strategies", "centre_evaluator")
@pytest.mark.parametrize("seed", range(20))
def test_every_registered_evaluator_is_clamped(seed: int) -> None:
    board = _random_board(seed, stones=10 + 4 * seed)
    state = SearchState.from_board(board, Stone.BLACK)
    for name, evaluate in EVALUATORS.items():
        for position in (board, state):
            for stone in (Stone.BLACK, Stone.WHITE):
                assert abs(evaluate(position, stone)) <= _EVAL_CLAMP, (name, seed)


def test_alphabeta_defaults_resolve_to_the_builtins() -> None:
    brain = AlphaBetaBrain()
    assert brain._evaluate is evaluate_board
    assert brain._candidates.__qualname__ == "radius_generator(2)"
    assert brain._early_candidates.__qualname__ == f"radius_generator({brain.settings.early_game_radius})"


def test_threat_fallback_gets_the_strategy() -> None:
    register_evaluator("stones", _stones_only)
    try:
        brain = ThreatBrain(fallback=AlphaBetaBrainSettings(depth=2, evaluator="stones"))
        assert brain._fallback._evaluate is _stones_only
    finally:
        unregister_strategy("stones")


def test_mcts_uses_the_evaluator_for_leaves() -> None:
    calls: list[int] = []

    def counting(board: BoardView, stone: Stone, /) -> float:
        calls.append(1)
        return evaluate_board(board, stone)

    register_evaluator("counting", counting)
    try:
        board = _random_board(2, stones=6)
        MCTSBrain(iterations=30, vcf_depth=0, evaluator="counting").get_move(board, Stone.BLACK)
    finally:
        unregister_strategy("counting")
    assert calls


def test_alphabeta_quiescence_uses_the_evaluator() -> None:
    calls: list[int] = []

    def counting(board: BoardView, stone: Stone, /) -> float:
        calls.append(1)
        return evaluate_board(board, stone)

    register_evaluator("counting", counting)
    try:
        brain = AlphaBetaBrain(depth=2, evaluator="counting")
        brain.get_move(_random_board(3, stones=9), Stone.BLACK)
        assert brain._qs_ctx is not None and brain._qs_ctx.evaluate is counting
    finally:
        unregister_strategy("counting")
    assert calls


@pytest.mark.usefixtures("toy_strategies", "corner_generator", "centre_evaluator")
def test_alphabeta_and_mcts_play_legal_moves_with_every_strategy() -> None:
    boards = [_random_board(seed, stones=3 + 5 * seed) for seed in range(3)]
    for evaluator in sorted(EVALUATORS):
        mcts = MCTSBrain(iterations=40, vcf_depth=0, evaluator=evaluator)
        for generator in sorted(MOVE_GENERATORS):
            brain = AlphaBetaBrain(depth=2, evaluator=evaluator, move_generator=generator)
            for board in boards:
                for player in (brain, mcts):
                    move = player.get_move(board, Stone.BLACK)
                    assert board.at(move.row, move.col) is Stone.EMPTY, (player.name, evaluator, generator)
