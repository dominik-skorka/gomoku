"""Opening book generator using Lincke's Drop-Out Expansion (DOE) algorithm.

The algorithm builds an opening book by best-first expansion:
- For the book side (engine): only the **best move** is followed.
- For the opponent: **all plausible moves** within an evaluation threshold
  are included, since the opponent may play anything.
- A priority function selects which leaf to expand next, favouring lines
  where the opponent plays well (low cumulative error) and short paths.
- Each leaf position is evaluated using a brain's alpha-beta search.
- Minimax values are back-propagated through the book graph.

Reference:
  Lincke, T.R. (2000). "Strategies for the Automatic Construction of
  Opening Books", CG 2000, Springer.
"""

import heapq
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gomoku.board import Board
from gomoku.models import GameConfig, Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.opening import _canonicalize, _transform, deduplicate_by_symmetry
from gomoku_brain.opening_book import BookEntry, OpeningBookFile, _canonical_key

if TYPE_CHECKING:
    from gomoku.interfaces import Player

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Book graph node
# ---------------------------------------------------------------------------


@dataclass
class _BookNode:
    """A node in the opening-book graph."""

    # Board position (stones only, not the Board object — for memory).
    stones: set[tuple[int, int, int]]
    # Canonical key (JSON string).
    key: str
    # Stone to move at this position.
    stone_to_move: Stone
    # Evaluated child moves: move → (score, child_key | None).
    # child_key is None for leaves not yet expanded.
    children: dict[tuple[int, int], tuple[float, str | None]] = field(default_factory=dict)
    # The minimax-backed score for this node (from the perspective of
    # stone_to_move).
    score: float = 0.0
    # Whether this node is terminal (win/draw/loss).
    terminal: bool = False
    # Whether this node has been fully evaluated (children populated).
    expanded: bool = False
    # Number of occupied cells (ply count).
    ply: int = 0

    @property
    def best_move(self) -> tuple[int, int] | None:
        """Return the child move with the highest score."""
        if not self.children:
            return None
        return max(self.children, key=lambda m: self.children[m][0])


# ---------------------------------------------------------------------------
# Priority queue entry
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _PriorityEntry:
    """Entry in the expansion priority queue.

    Lower ``priority`` value = higher expansion priority.
    We negate the priority so that heapq (min-heap) pops the highest-priority
    entry first.
    """

    priority: float
    # Path from root as list of (move_row, move_col) tuples.
    path: list[tuple[int, int]] = field(compare=False)
    # Key of the leaf node to expand.
    leaf_key: str = field(compare=False)
    # Cumulative opponent error along this path.
    cum_error: float = field(compare=False, default=0.0)


# ---------------------------------------------------------------------------
# DOE Generator
# ---------------------------------------------------------------------------

# Depth penalty constant: controls breadth vs depth trade-off.
_DEFAULT_DEPTH_PENALTY = 50.0


def _board_from_stones(
    stones: set[tuple[int, int, int]],
    size: int,
) -> Board:
    """Reconstruct a Board from a set of (stone_int, row, col) tuples."""
    board = Board(size=size)
    grid = [row[:] for row in board.grid]
    occupied: set[tuple[int, int]] = set()
    for sv, r, c in stones:
        grid[r][c] = Stone(sv)
        occupied.add((r, c))
    return Board.model_construct(size=size, grid=grid, occupied=occupied)


def _apply_move(
    stones: set[tuple[int, int, int]],
    move: tuple[int, int],
    stone: Stone,
) -> set[tuple[int, int, int]]:
    """Return a new stone set with *move* applied."""
    return stones | {(int(stone), move[0], move[1])}


def _is_terminal(board: Board, last_move: Move, config: GameConfig) -> bool:
    """Check if the position is terminal (win or draw)."""
    result = check_winner(board, last_move, config.win_length)
    if result is not None:
        return True
    return board.is_full()


def _evaluate_position(
    brain: Player,
    board: Board,
    stone: Stone,
) -> list[tuple[Move, float]]:
    """Evaluate all candidate moves from *board* for *stone* using *brain*.

    Returns a list of (move, score) sorted by score descending.
    The score is from the perspective of *stone* (higher = better for stone).
    """
    from gomoku_brain.evaluate import evaluate_board

    candidates = get_candidate_moves(board, radius=2)
    # In early game, deduplicate by symmetry to avoid redundant work.
    if len(board.occupied) <= 6:
        candidates = deduplicate_by_symmetry(board, candidates)

    scored: list[tuple[Move, float]] = []
    for move in candidates:
        new_board = board.place_stone(move, stone)
        # Check for immediate win — assign maximum score.
        result = check_winner(new_board, move)
        if result is not None:
            scored.append((move, 100_000.0))
            continue
        # Evaluate from the perspective of the moving side.
        score = evaluate_board(new_board, stone)
        scored.append((move, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _evaluate_with_brain(
    brain: Player,
    board: Board,
    stone: Stone,
) -> tuple[Move, float]:
    """Ask the brain for its best move and evaluate the resulting position.

    Returns (best_move, score) where score is from stone's perspective.
    """
    from gomoku_brain.evaluate import evaluate_board

    best_move = brain.get_move(board, stone)
    new_board = board.place_stone(best_move, stone)
    result = check_winner(new_board, best_move)
    if result is not None:
        return best_move, 100_000.0
    score = evaluate_board(new_board, stone)
    return best_move, score


def generate_opening_book(
    brain: Player,
    *,
    board_size: int = 15,
    max_positions: int = 100,
    error_threshold: float = 500.0,
    depth_penalty: float = _DEFAULT_DEPTH_PENALTY,
    max_book_ply: int = 10,
    progress_callback: Callable[[int, int, str], object] | None = None,
) -> OpeningBookFile:
    """Generate an opening book using Lincke's DOE algorithm.

    Parameters
    ----------
    brain:
        The Player (e.g. AlphaBetaBrain) used to evaluate positions.
    board_size:
        Board side length.
    max_positions:
        Maximum number of positions to expand into the book.
    error_threshold:
        Maximum evaluation gap (from best opponent move) for an opponent
        move to be included in the book.  Wider = broader coverage of
        sub-optimal opponent lines.
    depth_penalty:
        Weight of path length in the priority function.  Higher values
        produce shallower but wider books.
    max_book_ply:
        Maximum number of moves (half-moves) deep the book will go.
    progress_callback:
        If provided, called as ``callback(expanded, max_positions, node_key)``
        after each expansion.

    Returns
    -------
    OpeningBookFile ready to be saved.
    """
    config = GameConfig(board_size=board_size)
    nodes: dict[str, _BookNode] = {}
    pq: list[_PriorityEntry] = []

    # -- Helper: get or create a node for a position -------------------------
    def _get_or_create_node(
        stones: set[tuple[int, int, int]],
        stone_to_move: Stone,
    ) -> _BookNode:
        key = _canonical_key(stones, board_size)
        if key not in nodes:
            nodes[key] = _BookNode(
                stones=stones,
                key=key,
                stone_to_move=stone_to_move,
                ply=len(stones),
            )
        return nodes[key]

    # -- Helper: compute priority for a path ---------------------------------
    def _compute_priority(cum_error: float, depth: int) -> float:
        # Lower = higher priority.  Negate for min-heap.
        # Lines with low cumulative error (strong opponent play) and
        # short paths are expanded first.
        return -((-cum_error) - depth_penalty * depth)

    # -- Step 1: Initialise with the starting position -----------------------
    # On an empty board, the first move is always the centre.
    empty_stones: set[tuple[int, int, int]] = set()
    root = _get_or_create_node(empty_stones, Stone.BLACK)

    # Evaluate root: the first move is always centre.
    root_board = _board_from_stones(empty_stones, board_size)
    logger.info("Evaluating root position...")

    # Use the brain to pick the first move.
    centre = board_size // 2
    centre_move = Move(row=centre, col=centre)
    root_board_after = root_board.place_stone(centre_move, Stone.BLACK)
    from gomoku_brain.evaluate import evaluate_board

    root_score = evaluate_board(root_board_after, Stone.BLACK)
    child_stones = _apply_move(empty_stones, (centre, centre), Stone.BLACK)
    child_key = _canonical_key(child_stones, board_size)
    root.children[(centre, centre)] = (root_score, child_key)
    root.score = root_score
    root.expanded = True

    # Create the child node (White to move after Black's centre).
    child_node = _get_or_create_node(child_stones, Stone.WHITE)

    # Seed the priority queue with this child.
    pq_entry = _PriorityEntry(
        priority=_compute_priority(0.0, 1),
        path=[(centre, centre)],
        leaf_key=child_key,
        cum_error=0.0,
    )
    heapq.heappush(pq, pq_entry)

    expanded_count = 1  # root counts as 1

    start_time = time.monotonic()

    # -- Step 2: Best-first expansion loop -----------------------------------
    while pq and expanded_count < max_positions:
        entry = heapq.heappop(pq)
        leaf_key = entry.leaf_key

        # Skip if already expanded (can occur with duplicate queue entries).
        if leaf_key not in nodes:
            continue
        leaf = nodes[leaf_key]
        if leaf.expanded or leaf.terminal:
            continue
        if leaf.ply >= max_book_ply:
            continue

        # Reconstruct the board for this position.
        board = _board_from_stones(leaf.stones, board_size)
        stone = leaf.stone_to_move

        elapsed = time.monotonic() - start_time
        logger.info(
            "Expanding position %d/%d (ply=%d, stone=%s, elapsed=%.1fs)",
            expanded_count + 1,
            max_positions,
            leaf.ply,
            stone,
            elapsed,
        )

        # Evaluate all candidate moves using the brain.
        # We use the brain's get_move for the "best" move, but evaluate
        # all candidates with the static evaluator for breadth.
        scored_moves = _evaluate_position(brain, board, stone)

        if not scored_moves:
            leaf.terminal = True
            leaf.expanded = True
            continue

        # Use the brain to get its preferred move — this runs the full
        # search (alpha-beta with TT, etc.) and gives a higher-quality
        # best-move selection than the static evaluator alone.
        try:
            brain_move, brain_score = _evaluate_with_brain(brain, board, stone)
            # Ensure the brain's move is among the scored moves and has
            # the brain's evaluation.
            brain_idx = None
            for i, (m, _s) in enumerate(scored_moves):
                if m.row == brain_move.row and m.col == brain_move.col:
                    brain_idx = i
                    break
            if brain_idx is not None:
                # Promote the brain's move to the top with the brain's score.
                scored_moves[brain_idx] = (brain_move, brain_score)
                scored_moves.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            logger.warning("Brain evaluation failed, using static evaluation only", exc_info=True)

        best_score = scored_moves[0][1]
        leaf.score = best_score

        for move, score in scored_moves:
            move_tuple = (move.row, move.col)
            new_board = board.place_stone(move, stone)
            result = check_winner(new_board, move, config.win_length)
            is_term = result is not None or new_board.is_full()

            child_stones = _apply_move(leaf.stones, move_tuple, stone)
            child_key = _canonical_key(child_stones, board_size)
            leaf.children[move_tuple] = (score, child_key)

            if is_term:
                # Terminal: create node but don't queue for expansion.
                child_node = _get_or_create_node(child_stones, stone.opponent())
                child_node.terminal = True
                child_node.expanded = True
                child_node.score = score
                continue

            # For opponent nodes: include all moves within error_threshold.
            # For our nodes: only the best move gets queued.
            error_from_best = best_score - score
            if error_from_best > error_threshold:
                continue  # Skip this move — too far from best.

            child_node = _get_or_create_node(child_stones, stone.opponent())

            if not child_node.expanded:
                cum_error = entry.cum_error + error_from_best
                depth = len(entry.path) + 1
                if depth <= max_book_ply:
                    priority = _compute_priority(cum_error, depth)
                    new_entry = _PriorityEntry(
                        priority=priority,
                        path=[*entry.path, move_tuple],
                        leaf_key=child_key,
                        cum_error=cum_error,
                    )
                    heapq.heappush(pq, new_entry)

        leaf.expanded = True
        expanded_count += 1

        if progress_callback is not None:
            progress_callback(expanded_count, max_positions, leaf_key)

    # -- Step 3: Back-propagate minimax scores --------------------------------
    _backpropagate(nodes, root.key)

    # -- Step 4: Build the output book ----------------------------------------
    elapsed = time.monotonic() - start_time
    logger.info(
        "Book generation complete: %d positions in %.1fs",
        expanded_count,
        elapsed,
    )

    book = OpeningBookFile(
        board_size=board_size,
        generator_brain=getattr(brain, "name", "unknown"),
        search_depth=getattr(brain, "depth", 0),
        max_positions=max_positions,
        error_threshold=error_threshold,
    )

    for _key, node in nodes.items():
        if not node.expanded:
            continue
        best = node.best_move
        if best is None:
            continue
        # Store the best move in the canonical frame.
        # The key is already canonical, and the move coordinates are in the
        # frame of the canonical position (since we canonicalise when creating
        # the node's stones).
        # However, the node's stones might not be in canonical form — they're
        # the actual stones.  We need to transform the best move.
        canon, t = _canonicalize(node.stones, board_size)
        canon_key = json.dumps(canon)
        canon_move = _transform(best[0], best[1], board_size, t)
        book.positions[canon_key] = BookEntry(
            move=canon_move,
            score=node.score,
            depth=getattr(brain, "depth", 0),
        )

    return book


def _backpropagate(nodes: dict[str, _BookNode], root_key: str) -> float:
    """Recursive minimax back-propagation.  Returns score from root's perspective."""
    node = nodes[root_key]
    if not node.children or node.terminal:
        return node.score

    best_score = float("-inf")
    for move, (child_score, child_key) in node.children.items():
        if child_key is not None and child_key in nodes and nodes[child_key].expanded:
            # Recurse into expanded child; negate because the child is
            # from the opponent's perspective.
            propagated = -_backpropagate(nodes, child_key)
        else:
            propagated = child_score
        # Update the child score in the parent.
        node.children[move] = (propagated, node.children[move][1])
        if propagated > best_score:
            best_score = propagated

    if best_score > float("-inf"):
        node.score = best_score
    return node.score
