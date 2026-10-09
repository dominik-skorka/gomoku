"""Root-level VCF pre-check for ``AlphaBetaBrain``.

Before spinning up the main alpha-beta search, the brain runs a quick
VCF probe from the root:

* If we have a forced VCF win, return it immediately — no main search
  needed.
* If the *opponent* has a forced VCF win, restrict the root candidate
  set to moves that genuinely break the threat (durable breakers) and
  pick the positionally best one.
* Otherwise, return ``None`` and let the caller run normal search.

This logic is significant enough (≈90 lines) that bundling it into the
massive ``AlphaBetaBrain.get_move`` would obscure both halves.  Pulling
it into its own module also lets the diagnostics and decision-path
tags travel with the function rather than being woven through the
search code.
"""

from __future__ import annotations

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.alphabeta_ordering import _scored_moves
from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import _find_winning_move
from gomoku_brain.threat_search import _vcf, classify_vcf_breakers


def root_vcf(board: Board, stone: Stone, vcf_depth: int) -> Move | None:
    """Run VCF at the root for *stone* and its opponent.

    Returns:

    * The instant winning move if *stone* has one (five-in-a-row).
    * The first attack of *stone*'s forced VCF, if one exists.
    * The positionally best **durable breaker** of the opponent's VCF
      when the opponent has a forced win.  When no breaker exists, the
      opponent's instant-win cell is returned (so we at least block the
      five-in-a-row).
    * ``None`` when neither side has a VCF — the caller should run its
      normal search.

    The function emits the same ``decision_path`` / ``notes`` / threat
    annotations on the active diagnostics record as the original
    method did, so post-game analysis sees no behavioural change.
    """
    from gomoku.diagnostics import current_move_diagnostics

    diag = current_move_diagnostics()

    # 0) Instant win — check before the expensive VCF search.
    instant = _find_winning_move(board, stone)
    if instant is not None:
        logfire.info("AlphaBeta instant win: {move}", move=str(instant))
        if diag is not None:
            diag.decision_path = "instant_win"
        return instant

    # 1) Do we have a forced win?
    our_vcf = _vcf(board, stone, vcf_depth)
    if our_vcf is not None:
        logfire.info("AlphaBeta VCF win: {move}", move=str(our_vcf))
        if diag is not None:
            diag.decision_path = "vcf_win"
            diag.our_threats = diag.our_threats.model_copy(update={"vcf_result": "win"})
        return our_vcf

    # 2) Does the opponent have a forced win we must disrupt?
    opp = stone.opponent()
    opp_vcf = _vcf(board, opp, vcf_depth)
    if opp_vcf is None:
        if diag is not None:
            diag.notes.append("No VCF for either side")
        return None  # No VCF concern — proceed with normal search.

    logfire.info("Opponent has VCF starting at {move}, searching breakers", move=str(opp_vcf))
    if diag is not None:
        diag.opp_threats = diag.opp_threats.model_copy(update={"vcf_result": "win"})
        diag.notes.append(f"Opponent VCF at {opp_vcf}, searching breakers")

    candidates = get_candidate_moves(board, radius=2)
    durable, breakers = classify_vcf_breakers(board, candidates, stone, vcf_depth)

    if not breakers:
        # Nothing breaks the VCF — position is lost.  Prefer blocking an
        # instant winning move (five-in-a-row) over the multi-step VCF
        # start, since the instant win ends the game immediately.
        block = _find_winning_move(board, opp)
        if block is None:
            block = opp_vcf
        logfire.info(
            "No VCF breaker found, blocking opponent VCF start: {move}",
            move=str(block),
        )
        if diag is not None:
            diag.decision_path = "vcf_block_no_breaker"
            diag.chosen_move = block
            diag.notes.append(f"No VCF breaker, blocking opponent at {block}")
        return block

    effective = durable if durable else breakers
    logfire.info(
        "VCF breakers: {total} total, {durable} durable, using {n}",
        total=len(breakers),
        durable=len(durable),
        n=len(effective),
    )
    if diag is not None:
        diag.decision_path = "vcf_breaker"
        diag.notes.append(f"VCF breakers: {len(breakers)} total, {len(durable)} durable, using {len(effective)}")

    if len(effective) == 1:
        logfire.info("Single VCF breaker: {move}", move=str(effective[0]))
        return effective[0]

    # Multiple breakers — pick via evaluation (make/unmake on one state
    # instead of a Pydantic board copy per breaker).
    state = SearchState.from_board(board, stone)
    return _scored_moves(state, effective, stone)[0][1]
