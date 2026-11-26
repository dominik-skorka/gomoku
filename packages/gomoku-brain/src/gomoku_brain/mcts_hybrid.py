"""Shallow minimax searches used as MCTS node priors (MCTS–minimax hybrids).

Baier and Winands embed shallow minimax with an evaluation function in
MCTS; as a *node prior* it gave their strongest hybrid, with its cost
kept down by move ordering and k-best pruning ("Monte-Carlo Tree Search
and Minimax Hybrids with Heuristic Evaluation Functions", CIG 2013;
"MCTS-Minimax Hybrids with State Evaluations", JAIR 2018).

This module scores moves with a small negamax over the shared static
evaluation (:func:`gomoku_brain.evaluate.evaluate_board`): at every
inner node only the *k* moves with the best pattern priors are searched,
a five threat of the side to move ends the search as a win, and a five
threat of the other side leaves only the block.  The scores become a
softmax distribution that :class:`~gomoku_brain.mcts.MCTSBrain` blends
into its priors.
"""

import math

from gomoku.models import Stone

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.priors import compute_priors
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat import _find_winning_move

#: Score of a won position (above any static evaluation, which is clamped
#: to ±99 000).
WIN = 100_000.0


def _best_moves(state: SearchState, side: Stone, k: int) -> list[tuple[int, int]]:
    """The *k* candidates with the best pattern priors for *side*."""
    moves = [(m.row, m.col) for m in get_candidate_moves(state, radius=2)]
    if len(moves) <= k:
        return moves
    priors = compute_priors(state, moves, int(side), int(side.opponent()))
    order = sorted(range(len(moves)), key=priors.__getitem__, reverse=True)
    return [moves[i] for i in order[:k]]


def _negamax(state: SearchState, side: Stone, depth: int, alpha: float, beta: float, k: int) -> float:
    """Value of the position for *side* (to move), searched *depth* plies."""
    if _find_winning_move(state, side) is not None:
        return WIN
    if depth <= 0:
        return evaluate_board(state, side)
    other = side.opponent()
    block = _find_winning_move(state, other)
    moves = [(block.row, block.col)] if block is not None else _best_moves(state, side, k)
    if not moves:
        return 0.0
    best = -math.inf
    for r, c in moves:
        state.make(r, c, side)
        value = -_negamax(state, other, depth - 1, -beta, -alpha, k)
        state.unmake()
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def move_scores(
    state: SearchState,
    moves: list[tuple[int, int]],
    side: Stone,
    depth: int,
    k: int,
) -> list[float]:
    """Score each of *moves* for *side* (to move) with a *depth*-ply search.

    ``depth = 1`` is the static evaluation after the move; each extra ply
    lets the opponent (then us, ...) reply with its *k* best moves.
    *state* is returned unchanged.

    Precondition: *side* cannot make five at once (MCTS calls this only
    for undecided nodes), so no move here completes five; inside the
    search a side that can make five scores :data:`WIN` before moving.
    """
    other = side.opponent()
    scores: list[float] = []
    for r, c in moves:
        state.make(r, c, side)
        # Full window: every move needs its own score, not just the best.
        value = -_negamax(state, other, depth - 1, -math.inf, math.inf, k)
        state.unmake()
        scores.append(value)
    return scores


def search_priors(
    state: SearchState,
    moves: list[tuple[int, int]],
    side: Stone,
    depth: int,
    k: int,
    temperature: float = 1000.0,
) -> list[float]:
    """Softmax over :func:`move_scores` with the given *temperature*.

    *temperature* is in evaluation points; 1000 matches the scale of the
    MCTS leaf sigmoid (a live-three advantage).
    """
    if not moves:
        return []
    scores = move_scores(state, moves, side, depth, k)
    top = max(scores)
    weights = [math.exp(max(-50.0, (s - top) / temperature)) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def blend(priors: list[float], other: list[float], weight: float) -> list[float]:
    """``(1 − weight) · priors + weight · other``, element by element."""
    return [(1.0 - weight) * p + weight * o for p, o in zip(priors, other, strict=True)]
