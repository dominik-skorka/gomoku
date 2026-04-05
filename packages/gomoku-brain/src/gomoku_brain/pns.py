"""Proof-Number Search (PNS) in threat-space for Gomoku.

Implements the PN1 algorithm (Allis 1994) restricted to threat moves:

- **OR nodes (attacker)**: children are four-threats and live-three threats.
  The attacker wins if *any* child is proved.
- **AND nodes (defender)**: children are threat-relevant moves -- direct
  blocks and counter-threats.  The attacker wins only if *every* child is
  proved.

Proof numbers guide the search toward the *most-provable* subtree,
disproof numbers toward the *most-disprovable*.  The search terminates
when the root is proved (forced win found), disproved (no forced win),
or the node budget is exhausted.
"""

from __future__ import annotations

import time
from enum import Enum, auto

from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.threat import (
    _find_four_threats,
    _find_three_threats,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF = 1_000_000_000  # Practical infinity for proof/disproof numbers

_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class _NodeType(Enum):
    OR = auto()  # attacker to move
    AND = auto()  # defender to move


# ---------------------------------------------------------------------------
# PNS Node
# ---------------------------------------------------------------------------


class _PNSNode:
    """A single node in the proof-number search tree."""

    __slots__ = (
        "board",
        "children",
        "disproof",
        "expanded",
        "move",
        "node_type",
        "proof",
        "stone",
    )

    def __init__(
        self,
        board: Board,
        stone: Stone,
        node_type: _NodeType,
        move: Move | None = None,
    ) -> None:
        self.board = board
        self.stone = stone  # the *attacker* colour (constant through tree)
        self.node_type = node_type
        self.move = move  # the move that led to this node
        self.expanded = False
        self.children: list[_PNSNode] = []
        # Initial proof/disproof: 1/1 for unknown nodes
        self.proof = 1
        self.disproof = 1

    @property
    def proved(self) -> bool:
        return self.proof == 0

    @property
    def disproved(self) -> bool:
        return self.disproof == 0


# ---------------------------------------------------------------------------
# Cheap immediate-win check (no board copies)
# ---------------------------------------------------------------------------


def _can_win_immediately(board: Board, stone: Stone) -> Move | None:
    """Check if *stone* can complete five-in-a-row in one move.

    Counts neighbours in-place without copying the board, making it
    ~100x faster than ``_find_winning_move`` (which deep-copies per
    candidate).
    """
    size = board.size
    for move in get_candidate_moves(board, radius=2):
        r, c = move.row, move.col
        for dr, dc in _DIRECTIONS:
            count = 1  # the stone we would place
            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and board.at(nr, nc) is stone:
                count += 1
                nr += dr
                nc += dc
            nr, nc = r - dr, c - dc
            while 0 <= nr < size and 0 <= nc < size and board.at(nr, nc) is stone:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 5:
                return move
    return None


# ---------------------------------------------------------------------------
# Evaluation of leaf nodes
# ---------------------------------------------------------------------------


def _evaluate_leaf(node: _PNSNode) -> None:
    """Lightweight leaf evaluation: check immediate wins only.

    Full threat-existence detection is deferred to expansion time --
    if an OR node has no threats, expansion produces no children and
    the node is disproved then.
    """
    attacker = node.stone
    defender = attacker.opponent()
    board = node.board

    if node.node_type is _NodeType.OR:
        if _can_win_immediately(board, attacker) is not None:
            node.proof = 0
            node.disproof = INF
            return
    else:
        if _can_win_immediately(board, defender) is not None:
            node.proof = INF
            node.disproof = 0
            return

    node.proof = 1
    node.disproof = 1


# ---------------------------------------------------------------------------
# Smart defence generation for AND nodes
# ---------------------------------------------------------------------------


def _pns_defence_moves(board: Board, attacker: Stone) -> list[Move]:
    """Compute defender moves for an AND node: blocks + counter-threats.

    Includes:
    1. Forced blocks of attacker four-threats.
    2. End cells of attacker live-three threats.
    3. Defender's own four-threat moves (counter-fours).
    4. Defender's own three-threat moves (counter-threes).
    """
    defender = attacker.opponent()
    cells: set[tuple[int, int]] = set()

    # 1) Forced blocks of attacker fours
    for attack, block in _find_four_threats(board, attacker):
        if block is not None:
            cells.add((block.row, block.col))
        else:
            # Immediate five or live four — occupy the attack cell to prevent it
            cells.add((attack.row, attack.col))

    # 2) End cells of attacker live threes
    for _attack, defs in _find_three_threats(board, attacker):
        for d in defs:
            cells.add((d.row, d.col))

    # 3) Defender's own four-threats (counter-fours)
    for attack, _block in _find_four_threats(board, defender):
        cells.add((attack.row, attack.col))

    # 4) Defender's own three-threats (counter-threes)
    for attack, _defs in _find_three_threats(board, defender):
        cells.add((attack.row, attack.col))

    return [
        Move(row=r, col=c) for r, c in sorted(cells) if board.at(r, c) is Stone.EMPTY
    ]


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _expand(node: _PNSNode) -> None:
    """Generate children for a node."""
    node.expanded = True
    attacker = node.stone
    defender = attacker.opponent()
    board = node.board

    if node.node_type is _NodeType.OR:
        # Attacker moves: four-threats and three-threats only

        # Immediate win
        win_move = _can_win_immediately(board, attacker)
        if win_move is not None:
            child = _PNSNode(
                board.place_stone(win_move, attacker),
                attacker,
                _NodeType.AND,
                move=win_move,
            )
            child.proof = 0
            child.disproof = INF
            node.children.append(child)
            _update_ancestors(node)
            return

        # If defender has an immediate win, attacker must block it
        def_win = _can_win_immediately(board, defender)
        if def_win is not None:
            if board.is_valid_move(def_win):
                new_board = board.place_stone(def_win, attacker)
                child = _PNSNode(new_board, attacker, _NodeType.AND, move=def_win)
                _evaluate_leaf(child)
                node.children.append(child)
            else:
                node.proof = INF
                node.disproof = 0
            _update_ancestors(node)
            return

        # Four-threats
        seen_moves: set[tuple[int, int]] = set()
        for attack, _defence in _find_four_threats(board, attacker):
            key = (attack.row, attack.col)
            if key in seen_moves:
                continue
            seen_moves.add(key)

            new_board = board.place_stone(attack, attacker)
            r = check_winner(new_board, attack)
            if r is not None and r.winner is attacker:
                child = _PNSNode(new_board, attacker, _NodeType.AND, move=attack)
                child.proof = 0
                child.disproof = INF
                node.children.append(child)
                _update_ancestors(node)
                return

            if _can_win_immediately(new_board, defender) is not None:
                continue

            child = _PNSNode(new_board, attacker, _NodeType.AND, move=attack)
            _evaluate_leaf(child)
            node.children.append(child)

        # Three-threats
        for attack, _defs in _find_three_threats(board, attacker):
            key = (attack.row, attack.col)
            if key in seen_moves:
                continue
            seen_moves.add(key)

            new_board = board.place_stone(attack, attacker)
            r = check_winner(new_board, attack)
            if r is not None and r.winner is attacker:
                child = _PNSNode(new_board, attacker, _NodeType.AND, move=attack)
                child.proof = 0
                child.disproof = INF
                node.children.append(child)
                _update_ancestors(node)
                return

            if _can_win_immediately(new_board, defender) is not None:
                continue

            child = _PNSNode(new_board, attacker, _NodeType.AND, move=attack)
            _evaluate_leaf(child)
            node.children.append(child)

        if not node.children:
            # No viable attack moves -> disproved
            node.proof = INF
            node.disproof = 0

    else:
        # AND node: defender responds with blocks + counter-threats

        # If attacker can win immediately, defender MUST block at that cell.
        # If attacker has multiple winning cells (e.g. live four), the
        # node is effectively proved — defender can only block one.
        atk_win = _can_win_immediately(board, attacker)
        if atk_win is not None:
            # Try blocking at the winning cell
            new_board = board.place_stone(atk_win, defender)
            # Check if attacker still has an immediate win after the block
            if _can_win_immediately(new_board, attacker) is not None:
                # Multiple winning cells — defender can't block all → proved
                node.proof = 0
                node.disproof = INF
                _update_ancestors(node)
                return
            # Single winning cell — defender blocks, continue from there
            r = check_winner(new_board, atk_win)
            if r is not None and r.winner is defender:
                child = _PNSNode(new_board, attacker, _NodeType.OR, move=atk_win)
                child.proof = INF
                child.disproof = 0
                node.children.append(child)
                _update_ancestors(node)
                return
            child = _PNSNode(new_board, attacker, _NodeType.OR, move=atk_win)
            _evaluate_leaf(child)
            node.children.append(child)
            _update_ancestors(node)
            return

        defences = _pns_defence_moves(board, attacker)

        if not defences:
            # No blocks or counter-threats -> attacker's threat is unanswerable
            node.proof = 0
            node.disproof = INF
            _update_ancestors(node)
            return

        for cand in defences:
            new_board = board.place_stone(cand, defender)

            r = check_winner(new_board, cand)
            if r is not None and r.winner is defender:
                child = _PNSNode(new_board, attacker, _NodeType.OR, move=cand)
                child.proof = INF
                child.disproof = 0
                node.children.append(child)
                _update_ancestors(node)
                return

            child = _PNSNode(new_board, attacker, _NodeType.OR, move=cand)
            _evaluate_leaf(child)
            node.children.append(child)

    _update_ancestors(node)


# ---------------------------------------------------------------------------
# Proof/disproof number update
# ---------------------------------------------------------------------------


def _update_ancestors(node: _PNSNode) -> None:
    """Recompute proof/disproof from children (non-recursive; called bottom-up)."""
    if not node.children:
        return

    if node.node_type is _NodeType.OR:
        node.proof = min(c.proof for c in node.children)
        dp_sum = sum(c.disproof for c in node.children)
        node.disproof = min(dp_sum, INF)
    else:
        pn_sum = sum(c.proof for c in node.children)
        node.proof = min(pn_sum, INF)
        node.disproof = min(c.disproof for c in node.children)


# ---------------------------------------------------------------------------
# MPN selection with path tracking
# ---------------------------------------------------------------------------


def _select_mpn(node: _PNSNode) -> tuple[_PNSNode, list[_PNSNode]]:
    """Walk from root to the most-proving leaf, returning ``(leaf, path)``."""
    path = [node]
    current = node
    while current.expanded and current.children:
        if current.node_type is _NodeType.OR:
            current = min(current.children, key=lambda c: c.proof)
        else:
            current = min(current.children, key=lambda c: c.disproof)
        path.append(current)
    return current, path


# ---------------------------------------------------------------------------
# Path-based backpropagation
# ---------------------------------------------------------------------------


def _backpropagate_path(path: list[_PNSNode]) -> None:
    """Update proof/disproof numbers along the selection path only (bottom-up).

    O(path_length) per call instead of O(tree_size).
    """
    for node in reversed(path):
        if node.expanded and node.children:
            _update_ancestors(node)


# ---------------------------------------------------------------------------
# Main PNS entry point
# ---------------------------------------------------------------------------


def pns_threat_search(
    board: Board,
    stone: Stone,
    max_nodes: int = 10_000,
    max_time_s: float = 5.0,
) -> Move | None:
    """Run Proof-Number Search in threat-space.

    Returns the first move of a forced winning sequence for *stone*,
    or ``None`` if no win is proved within the node/time budget.
    """
    root = _PNSNode(board, stone, _NodeType.OR)
    _evaluate_leaf(root)

    if root.proved:
        return _can_win_immediately(board, stone)
    if root.disproved:
        return None

    nodes_created = 1
    deadline = time.monotonic() + max_time_s

    while root.proof != 0 and root.disproof != 0 and nodes_created < max_nodes:
        if time.monotonic() > deadline:
            break

        # 1) Select the most-proving node
        mpn, path = _select_mpn(root)

        # 2) Expand it
        children_before = len(mpn.children)
        _expand(mpn)
        nodes_created += len(mpn.children) - children_before

        # 3) Backpropagate along selection path only
        _backpropagate_path(path)

    if root.proved and root.children:
        # Root is OR -> find the proved child
        for child in root.children:
            if child.proved:
                return child.move
    return None
