"""Opening book and board symmetry reduction for Gomoku alpha-beta search."""

from typing import TYPE_CHECKING

from gomoku.models import Move, Stone

if TYPE_CHECKING:
    from gomoku.interfaces import BoardView


# ---------------------------------------------------------------------------
# 8-fold symmetry of the square board
# ---------------------------------------------------------------------------
# Transform indices:
#   0 = identity
#   1 = rotate  90° CW
#   2 = rotate 180°
#   3 = rotate 270° CW
#   4 = reflect horizontal (flip columns)
#   5 = reflect vertical   (flip rows)
#   6 = reflect main diagonal       (transpose)
#   7 = reflect anti-diagonal


def _transform(r: int, c: int, s: int, t: int) -> tuple[int, int]:
    """Apply symmetry transform *t* to coordinate *(r, c)* on a board of side *s*."""
    m = s - 1
    if t == 0:
        return (r, c)
    if t == 1:
        return (c, m - r)
    if t == 2:
        return (m - r, m - c)
    if t == 3:
        return (m - c, r)
    if t == 4:
        return (r, m - c)
    if t == 5:
        return (m - r, c)
    if t == 6:
        return (c, r)
    # t == 7
    return (m - c, m - r)


# Inverse of each transform (so we can map a book move back to the
# original frame of reference).
_INVERSE: list[int] = [0, 3, 2, 1, 4, 5, 6, 7]


def _board_symmetry_group(board: BoardView) -> list[int]:
    """Return transform indices that leave *board* invariant."""
    size = board.size
    occupied = board.occupied
    if not occupied:
        return list(range(8))  # empty board has full symmetry

    stones: set[tuple[int, int, int]] = set()
    for r, c in occupied:
        stones.add((int(board.at(r, c)), r, c))

    group: list[int] = []
    for t in range(8):
        transformed = {(sv, *_transform(r, c, size, t)) for sv, r, c in stones}
        if transformed == stones:
            group.append(t)
    return group


def deduplicate_by_symmetry(board: BoardView, candidates: list[Move]) -> list[Move]:
    """Remove symmetry-equivalent candidates, keeping one per equivalence class.

    Returns *candidates* unchanged when the board has no non-trivial symmetry.
    """
    sym_group = _board_symmetry_group(board)
    if len(sym_group) <= 1:
        return candidates

    size = board.size
    seen: set[tuple[int, int]] = set()
    result: list[Move] = []
    for move in candidates:
        if (move.row, move.col) in seen:
            continue
        result.append(move)
        for t in sym_group:
            tr, tc = _transform(move.row, move.col, size, t)
            seen.add((tr, tc))
    return result


# ---------------------------------------------------------------------------
# Opening book
# ---------------------------------------------------------------------------
# Positions are stored in *canonical* form: the lexicographically smallest
# representation over all 8 symmetry transforms.  Each entry maps to the
# recommended move in that canonical frame.
#
# Format: frozenset of (stone_int, row, col) → Move (in canonical frame)


def _canonicalize(
    stones: set[tuple[int, int, int]],
    size: int,
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    """Canonical form of *stones* over the 8-fold symmetry, plus the transform index used."""
    best_key: tuple[tuple[int, int, int], ...] | None = None
    best_t = 0
    for t in range(8):
        key = tuple(sorted((sv, *_transform(r, c, size, t)) for sv, r, c in stones))
        if best_key is None or key < best_key:
            best_key = key
            best_t = t
    assert best_key is not None
    return best_key, best_t


# Book entries for standard 15x15 freestyle Gomoku.
_BOOK_15: dict[tuple[tuple[int, int, int], ...], tuple[int, int]] = {}


def _add_book(stones: list[tuple[int, int, int]], move: tuple[int, int], size: int = 15) -> None:
    """Register a book entry (auto-canonicalised)."""
    key, t = _canonicalize(set(stones), size)
    # Transform the recommended move into the canonical frame.
    canon_move = _transform(move[0], move[1], size, t)
    _BOOK_15[key] = canon_move


# -- Move 2: Black at center, White to play ---------------------------------
# Diagonal adjacency is the main-line opening.
_add_book([(1, 7, 7)], (6, 6))

# -- Move 3: Center + diagonal, Black to play --------------------------------
# Extend diagonally on the opposite side — creates a connected line.
_add_book([(1, 7, 7), (2, 6, 6)], (8, 8))

# -- Move 3 variant: Center + orthogonal adjacent, Black to play --------------
# After White plays adjacent orthogonally, Black extends the other way.
_add_book([(1, 7, 7), (2, 6, 7)], (8, 7))


def lookup_opening_book(board: BoardView, stone: Stone) -> Move | None:
    """Return a book move for *board* + *stone* to play, or ``None``."""
    if board.size != 15:
        return None

    occupied = board.occupied
    if len(occupied) > 4:
        return None  # book only covers the first few plies

    stones: set[tuple[int, int, int]] = set()
    for r, c in occupied:
        stones.add((int(board.at(r, c)), r, c))

    key, t = _canonicalize(stones, board.size)
    canon_move = _BOOK_15.get(key)
    if canon_move is None:
        return None

    # Map the canonical move back to the original frame.
    inv_t = _INVERSE[t]
    actual = _transform(canon_move[0], canon_move[1], board.size, inv_t)
    move = Move(row=actual[0], col=actual[1])

    # Sanity: the cell must be empty.
    if (move.row, move.col) in occupied:
        return None

    return move
