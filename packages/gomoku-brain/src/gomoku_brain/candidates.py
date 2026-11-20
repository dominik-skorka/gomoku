"""Smart move candidate generation for Gomoku AI.

Instead of evaluating all empty cells (up to 225 on a 15x15 board),
restrict candidates to cells within a radius of existing stones.
"""

from __future__ import annotations

from gomoku.interfaces import BoardView
from gomoku.models import Move

_RADIUS = 2


def _get_candidates_python(board: BoardView, radius: int) -> list[Move]:
    """Pure-Python candidate generation."""
    size = board.size
    candidates: set[tuple[int, int]] = set()
    for r, c in board.occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in board.occupied:
                    candidates.add((nr, nc))
    return [Move(row=r, col=c) for r, c in sorted(candidates)]


def get_candidate_moves(board: BoardView, radius: int = _RADIUS) -> list[Move]:
    """Return empty cells within ``radius`` of any placed stone.

    On an empty board, returns the center cell.
    Dispatches to bitboard JIT, numpy-grid JIT, or pure Python.
    """
    # ``occ_count`` is tracked incrementally on SearchState — checking it
    # avoids materialising the lazy ``occupied`` set on every call.
    occ_count = getattr(board, "occ_count", None)
    if occ_count == 0 or (occ_count is None and not board.occupied):
        center = board.size // 2
        return [Move(row=center, col=center)]

    from gomoku_brain.search_state import SearchState

    if radius == 2 and isinstance(board, SearchState):
        # SearchState at the default radius: reuse the zobrist-memoised
        # scan context, and memoise the Move list alongside it — building
        # a pydantic Move per candidate is the dominant cost of this
        # function.  A copy is returned because callers slice / re-order.
        from gomoku_brain.threat_detect import _scan_context

        ctx = _scan_context(board)
        if ctx is not None:
            ply = board.ply
            moves = board._ctx_moves[ply]
            if moves is None:
                _grid, _size, cand_arr, _dirs = ctx
                moves = [Move(row=int(cand_arr[i, 0]), col=int(cand_arr[i, 1])) for i in range(cand_arr.shape[0])]
                board._ctx_moves[ply] = moves
            return list(moves)

    from gomoku_brain.bitboard import _should_use_bitboard

    if _should_use_bitboard():
        from gomoku_brain.bitboard_jit import get_candidates_bb
        from gomoku_brain.grid_utils import board_to_bitboard

        black, white, size = board_to_bitboard(board)
        result = get_candidates_bb(black, white, size, radius)
        return [Move(row=int(result[i, 0]), col=int(result[i, 1])) for i in range(result.shape[0])]

    from gomoku_brain.evaluate import _should_use_jit

    if _should_use_jit():
        from gomoku_brain.evaluate_jit import get_candidates_grid
        from gomoku_brain.grid_utils import board_to_grid, occupied_coords

        grid = board_to_grid(board)
        occ = occupied_coords(board)
        result = get_candidates_grid(grid, board.size, occ, radius)
        return [Move(row=int(result[i, 0]), col=int(result[i, 1])) for i in range(result.shape[0])]

    return _get_candidates_python(board, radius)
