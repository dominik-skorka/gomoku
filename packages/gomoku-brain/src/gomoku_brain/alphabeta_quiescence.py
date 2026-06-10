"""Quiescence search for ``AlphaBetaBrain``.

At leaf nodes (``depth ≤ 1``) the alpha-beta brain extends the search
along *forcing* lines — four-threats and immediate wins — so it does
not return a static evaluation in the middle of a tactical sequence.
The recursion is bounded by ``_MAX_QS_DEPTH`` and uses a stand-pat
lower bound (``evaluate_board(state, stone)``) so each level can
cut off as soon as the position is already good enough.

This module owns the recursion; the brain keeps a thin
``AlphaBetaBrain._quiesce`` wrapper that bundles instance state
(``qs_vcf_depth`` and the running ``qs_node_count`` counter) into a
:class:`QuiesceCtx` and delegates to :func:`quiesce`.

Why a context-object rather than ``self``: extracting a recursive
search out of a class means it can no longer reach instance state
directly.  Bundling the few bits it needs — the VCF probe depth and a
mutable node counter — into one tiny dataclass keeps the call
signatures clean and lets the search stay a pure free function.
"""

from __future__ import annotations

from dataclasses import dataclass

from gomoku.models import Move, Stone

from gomoku_brain.alphabeta_ordering import _check_win, _split_threats
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.evaluate import evaluate_board
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import _find_winning_move, _scan_context
from gomoku_brain.threat_search import _vcf

EvalCache = dict[int, float]


@dataclass
class QuiesceCtx:
    """Per-search context bundling the mutable state quiescence touches.

    * ``qs_vcf_depth`` — read-only configuration; if non-zero, leaf
      positions trigger a VCF probe to detect multi-move forced wins
      that the one-ply threat loop would miss.
    * ``eval_cache`` — shared Zobrist-hash → static-eval cache.  The
      function reads and writes this cache so successive calls don't
      re-evaluate identical positions.
    * ``nodes`` — incremented once per :func:`quiesce` invocation; the
      brain reads it back into ``self.qs_node_count`` after the search
      so the running total survives across the multiple entry points
      in :meth:`AlphaBetaBrain._alphabeta`.
    """

    qs_vcf_depth: int
    eval_cache: EvalCache
    nodes: int = 0


def get_threat_moves(state: SearchState, stone: Stone) -> list[Move]:
    """Return moves that create a four-in-a-row for *stone* or block one
    for the opponent.

    Only considers candidates within radius 2 of occupied stones — this
    is a lightweight detector for quiescence, not an exhaustive threat
    enumeration.

    On the JIT path the whole scan happens array-side via
    ``get_threat_moves_grid``: ``Move`` objects are constructed only for
    the handful of threat cells, not for every candidate.  (Building a
    pydantic ``Move`` per candidate per quiescence node used to dominate
    the entire alpha-beta search cost on quiet positions.)
    """
    ctx = _scan_context(state)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import get_threat_moves_grid

        grid, size, candidates, dirs = ctx
        result = get_threat_moves_grid(grid, size, candidates, int(stone), int(stone.opponent()), dirs)
        return [Move(row=int(result[i, 0]), col=int(result[i, 1])) for i in range(result.shape[0])]

    candidates = get_candidate_moves(state)
    threats, _ = _split_threats(state, candidates, stone)
    return threats


def quiesce(
    state: SearchState,
    alpha: float,
    beta: float,
    stone: Stone,
    ctx: QuiesceCtx,
    qs_depth: int,
) -> float:
    """Quiescence search: extend leaves for forcing (four-threat) moves.

    Mutates ``ctx.nodes`` (per-call increment) and ``ctx.eval_cache``
    (memoised stand-pat evaluations).  ``state`` is mutated via
    ``make`` / ``unmake`` and restored before each return.  Returns the
    score from *stone*'s perspective; the caller negates as usual for
    the recursive negamax framing.
    """
    ctx.nodes += 1
    zh = state.zobrist

    # -- VCF probe: detect multi-move forced wins that the one-ply
    #    threat loop would miss.
    if ctx.qs_vcf_depth > 0:
        # Guard: if the opponent already has a winning move (four/five
        # on the board), we must block — we cannot freely start our own
        # VCF.  Skipping this check caused the engine to return a false
        # +100 000 even when the opponent had an unstoppable live-4.
        # _vcf accepts the SearchState directly and leaves it at the
        # entry ply (make/unmake balanced).
        opp = stone.opponent()
        if _find_winning_move(state, opp) is None and _vcf(state, stone, ctx.qs_vcf_depth) is not None:
            return 100_000.0

    # Standing pat — static evaluation as a lower bound.
    if zh in ctx.eval_cache:
        stand_pat = ctx.eval_cache[zh]
    else:
        stand_pat = evaluate_board(state, stone)
        ctx.eval_cache[zh] = stand_pat
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    if qs_depth <= 0:
        return stand_pat

    # Only search forcing moves.
    threats = get_threat_moves(state, stone)
    if not threats:
        return stand_pat

    for move in threats:
        state.make(move.row, move.col, stone)
        try:
            if _check_win(state, move):
                return 100_000.0

            score = -quiesce(state, -beta, -alpha, stone.opponent(), ctx, qs_depth - 1)
        finally:
            state.unmake()

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    return alpha
