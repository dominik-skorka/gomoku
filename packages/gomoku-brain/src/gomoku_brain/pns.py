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

Positions are represented implicitly: each node stores only its move and
parent pointer.  A single mutable :class:`SearchState` is replayed along
the selection path each iteration (O(depth) make/unmake per iteration),
so no Pydantic ``Board`` is allocated per node.
"""

from __future__ import annotations

import time
from enum import Enum, auto

from gomoku.board import Board
from gomoku.models import Move, Stone

from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import (
    _find_four_threats,
    _find_three_threats,
    _find_winning_move,
    _scan_context,
    _ScanCtx,
)
from gomoku_brain.threat_search import _state_is_win

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF = 1_000_000_000  # Practical infinity for proof/disproof numbers


class _NodeType(Enum):
    OR = auto()  # attacker to move
    AND = auto()  # defender to move


# ---------------------------------------------------------------------------
# PNS Node
# ---------------------------------------------------------------------------


class _PNSNode:
    """A single node in the proof-number search tree.

    Stores only the move that led here and the parent pointer — the
    position itself is materialised on demand by replaying moves into
    the shared :class:`SearchState` owned by :func:`pns_threat_search`.
    """

    __slots__ = (
        "best_child",
        "children",
        "disproof",
        "expanded",
        "move",
        "node_type",
        "parent",
        "proof",
        "stone",
    )

    def __init__(
        self,
        stone: Stone,
        node_type: _NodeType,
        move: Move | None = None,
        parent: _PNSNode | None = None,
    ) -> None:
        self.stone = stone  # the *attacker* colour (constant through tree)
        self.node_type = node_type
        self.move = move  # the move that led to this node
        self.parent = parent
        self.expanded = False
        self.children: list[_PNSNode] = []
        # Initial proof/disproof: 1/1 for unknown nodes
        self.proof = 1
        self.disproof = 1
        # MPN selection cache — set during ``_update_ancestors``.  Reading
        # this is O(1); the previous ``min(... key=...)`` walk inside
        # ``_select_mpn`` was O(branching) per tree level.
        self.best_child: _PNSNode | None = None

    @property
    def proved(self) -> bool:
        return self.proof == 0

    @property
    def disproved(self) -> bool:
        return self.disproof == 0


# ---------------------------------------------------------------------------
# Evaluation of leaf nodes
# ---------------------------------------------------------------------------


def _evaluate_leaf(
    node: _PNSNode,
    state: SearchState,
    attacker: Stone,
    ctx: _ScanCtx | None = None,
) -> None:
    """Lightweight leaf evaluation: check immediate wins only.

    ``state`` must already reflect the position at ``node``.  Full
    threat-existence detection is deferred to expansion time — if an OR
    node has no threats, expansion produces no children and the node is
    disproved then.

    ``ctx`` is the expanding parent's scan context.  Leaf evaluation
    always probes the side that did *not* play the move leading to the
    leaf, so the parent's candidate snapshot is still valid for it (see
    the staleness contract on ``_ScanCtx``).
    """
    defender = attacker.opponent()

    if node.node_type is _NodeType.OR:
        if _find_winning_move(state, attacker, ctx) is not None:
            node.proof = 0
            node.disproof = INF
            return
    else:
        if _find_winning_move(state, defender, ctx) is not None:
            node.proof = INF
            node.disproof = 0
            return

    node.proof = 1
    node.disproof = 1


# ---------------------------------------------------------------------------
# Smart defence generation for AND nodes
# ---------------------------------------------------------------------------


def _pns_defence_moves(
    state: SearchState,
    attacker: Stone,
    ctx: _ScanCtx | None = None,
) -> list[Move]:
    """Compute defender moves for an AND node: blocks + counter-threats.

    Includes:
    1. Forced blocks of attacker four-threats.
    2. End cells of attacker live-three threats.
    3. Defender's own four-threat moves (counter-fours).
    4. Defender's own three-threat moves (counter-threes).

    On the JIT path the four detection routines share one scan context —
    ``ctx`` (the expanding node's context, built at the node's own
    position) when provided, else one derived here.
    """
    defender = attacker.opponent()
    cells: set[tuple[int, int]] = set()

    if ctx is None:
        ctx = _scan_context(state)
    if ctx is not None:
        from gomoku_brain.evaluate_jit import find_four_threats_grid, find_three_threats_grid

        grid, size, candidates, dirs = ctx

        atk_int = int(attacker)
        def_int = int(defender)

        # 1) Attacker fours — forced blocks (or attack cell if unblockable).
        atk_fours = find_four_threats_grid(grid, size, candidates, atk_int, dirs)
        for i in range(atk_fours.shape[0]):
            ar = int(atk_fours[i, 0])
            ac = int(atk_fours[i, 1])
            dr_end = int(atk_fours[i, 2])
            if dr_end < 0:
                cells.add((ar, ac))
            else:
                cells.add((dr_end, int(atk_fours[i, 3])))

        # 2) Attacker live-threes — end-cell blocks.
        atk_threes = find_three_threats_grid(grid, size, candidates, atk_int, dirs)
        for i in range(atk_threes.shape[0]):
            cells.add((int(atk_threes[i, 2]), int(atk_threes[i, 3])))
            cells.add((int(atk_threes[i, 4]), int(atk_threes[i, 5])))

        # 3) Defender's own four-threats (counter-fours).
        def_fours = find_four_threats_grid(grid, size, candidates, def_int, dirs)
        for i in range(def_fours.shape[0]):
            cells.add((int(def_fours[i, 0]), int(def_fours[i, 1])))

        # 4) Defender's own three-threats (counter-threes).
        def_threes = find_three_threats_grid(grid, size, candidates, def_int, dirs)
        for i in range(def_threes.shape[0]):
            cells.add((int(def_threes[i, 0]), int(def_threes[i, 1])))
    else:
        # 1) Forced blocks of attacker fours
        for attack, block in _find_four_threats(state, attacker):
            if block is not None:
                cells.add((block.row, block.col))
            else:
                cells.add((attack.row, attack.col))

        # 2) End cells of attacker live threes
        for _attack, defs in _find_three_threats(state, attacker):
            for d in defs:
                cells.add((d.row, d.col))

        # 3) Defender's own four-threats (counter-fours)
        for attack, _block in _find_four_threats(state, defender):
            cells.add((attack.row, attack.col))

        # 4) Defender's own three-threats (counter-threes)
        for attack, _defs in _find_three_threats(state, defender):
            cells.add((attack.row, attack.col))

    return [Move(row=r, col=c) for r, c in sorted(cells) if state.is_empty(r, c)]


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _expand(node: _PNSNode, state: SearchState) -> None:
    """Generate children for ``node``.

    Precondition: ``state`` is at ``node``'s position.  Postcondition:
    ``state`` is restored to ``node``'s position (intermediate
    make/unmake cycles are balanced by ``try`` / ``finally`` blocks).

    One scan context is built per expansion and shared across every
    detection call — node-level enumeration runs at the context's own
    position, and all post-make probes target the side that did not
    just move, so the staleness contract on ``_ScanCtx`` holds.
    """
    node.expanded = True
    attacker = node.stone
    defender = attacker.opponent()
    ctx = _scan_context(state)

    if node.node_type is _NodeType.OR:
        # Attacker moves: four-threats and three-threats only.

        # Immediate win — no need to materialise the child position.
        win_move = _find_winning_move(state, attacker, ctx)
        if win_move is not None:
            child = _PNSNode(attacker, _NodeType.AND, move=win_move, parent=node)
            child.proof = 0
            child.disproof = INF
            node.children.append(child)
            _update_ancestors(node)
            return

        # If defender has an immediate win, attacker must block it.
        def_win = _find_winning_move(state, defender, ctx)
        if def_win is not None:
            if state.is_empty(def_win.row, def_win.col):
                state.make(def_win.row, def_win.col, attacker)
                try:
                    child = _PNSNode(attacker, _NodeType.AND, move=def_win, parent=node)
                    _evaluate_leaf(child, state, attacker, ctx)
                    node.children.append(child)
                finally:
                    state.unmake()
            else:
                node.proof = INF
                node.disproof = 0
            _update_ancestors(node)
            return

        # Four-threats
        seen_moves: set[tuple[int, int]] = set()
        for attack, _defence in _find_four_threats(state, attacker, ctx):
            key = (attack.row, attack.col)
            if key in seen_moves:
                continue
            seen_moves.add(key)

            state.make(attack.row, attack.col, attacker)
            try:
                if _state_is_win(state, attack.row, attack.col):
                    child = _PNSNode(attacker, _NodeType.AND, move=attack, parent=node)
                    child.proof = 0
                    child.disproof = INF
                    node.children.append(child)
                    _update_ancestors(node)
                    return

                if _find_winning_move(state, defender, ctx) is not None:
                    continue

                child = _PNSNode(attacker, _NodeType.AND, move=attack, parent=node)
                _evaluate_leaf(child, state, attacker, ctx)
                node.children.append(child)
            finally:
                state.unmake()

        # Three-threats
        for attack, _defs in _find_three_threats(state, attacker, ctx):
            key = (attack.row, attack.col)
            if key in seen_moves:
                continue
            seen_moves.add(key)

            state.make(attack.row, attack.col, attacker)
            try:
                if _state_is_win(state, attack.row, attack.col):
                    child = _PNSNode(attacker, _NodeType.AND, move=attack, parent=node)
                    child.proof = 0
                    child.disproof = INF
                    node.children.append(child)
                    _update_ancestors(node)
                    return

                if _find_winning_move(state, defender, ctx) is not None:
                    continue

                child = _PNSNode(attacker, _NodeType.AND, move=attack, parent=node)
                _evaluate_leaf(child, state, attacker, ctx)
                node.children.append(child)
            finally:
                state.unmake()

        if not node.children:
            # No viable attack moves -> disproved
            node.proof = INF
            node.disproof = 0

    else:
        # AND node: defender responds with blocks + counter-threats.

        # If attacker can win immediately, defender MUST block at that cell.
        # If attacker has multiple winning cells (e.g. live four), the
        # node is effectively proved — defender can only block one.
        atk_win = _find_winning_move(state, attacker, ctx)
        if atk_win is not None:
            state.make(atk_win.row, atk_win.col, defender)
            try:
                if _find_winning_move(state, attacker, ctx) is not None:
                    # Multiple winning cells — defender can't block all → proved
                    node.proof = 0
                    node.disproof = INF
                    _update_ancestors(node)
                    return
                if _state_is_win(state, atk_win.row, atk_win.col):
                    # Defender's block itself completes five for defender
                    # (rare but possible: defender had own 4-in-a-row through
                    # the block cell).  Disprove this OR-child.
                    child = _PNSNode(attacker, _NodeType.OR, move=atk_win, parent=node)
                    child.proof = INF
                    child.disproof = 0
                    node.children.append(child)
                    _update_ancestors(node)
                    return
                child = _PNSNode(attacker, _NodeType.OR, move=atk_win, parent=node)
                _evaluate_leaf(child, state, attacker, ctx)
                node.children.append(child)
                _update_ancestors(node)
                return
            finally:
                state.unmake()

        defences = _pns_defence_moves(state, attacker, ctx)

        if not defences:
            # No blocks or counter-threats -> attacker's threat is unanswerable
            node.proof = 0
            node.disproof = INF
            _update_ancestors(node)
            return

        for cand in defences:
            state.make(cand.row, cand.col, defender)
            try:
                if _state_is_win(state, cand.row, cand.col):
                    child = _PNSNode(attacker, _NodeType.OR, move=cand, parent=node)
                    child.proof = INF
                    child.disproof = 0
                    node.children.append(child)
                    _update_ancestors(node)
                    return

                child = _PNSNode(attacker, _NodeType.OR, move=cand, parent=node)
                _evaluate_leaf(child, state, attacker, ctx)
                node.children.append(child)
            finally:
                state.unmake()

    _update_ancestors(node)


# ---------------------------------------------------------------------------
# Proof/disproof number update
# ---------------------------------------------------------------------------


def _update_ancestors(node: _PNSNode) -> None:
    """Recompute proof/disproof from children (non-recursive; called bottom-up).

    Also caches the most-proving / most-disproving child in
    ``node.best_child`` so :func:`_select_mpn` can walk the tree in
    O(depth) rather than O(depth × branching).
    """
    if not node.children:
        return

    children = node.children
    if node.node_type is _NodeType.OR:
        best = children[0]
        best_proof = best.proof
        dp_sum = best.disproof
        for c in children[1:]:
            if c.proof < best_proof:
                best = c
                best_proof = c.proof
            dp_sum += c.disproof
        node.proof = best_proof
        node.disproof = min(dp_sum, INF)
        node.best_child = best
    else:
        best = children[0]
        best_disproof = best.disproof
        pn_sum = best.proof
        for c in children[1:]:
            if c.disproof < best_disproof:
                best = c
                best_disproof = c.disproof
            pn_sum += c.proof
        node.proof = min(pn_sum, INF)
        node.disproof = best_disproof
        node.best_child = best


# ---------------------------------------------------------------------------
# MPN selection with path tracking
# ---------------------------------------------------------------------------


def _select_mpn(node: _PNSNode) -> tuple[_PNSNode, list[_PNSNode]]:
    """Walk from root to the most-proving leaf, returning ``(leaf, path)``.

    Uses ``node.best_child`` cached during :func:`_update_ancestors` so the
    walk is O(depth).  Falls back to a per-level ``min`` only if a node
    was expanded without subsequent backprop (shouldn't happen in
    well-formed trees but kept for safety).
    """
    path = [node]
    current = node
    while current.expanded and current.children:
        nxt = current.best_child
        if nxt is None:
            if current.node_type is _NodeType.OR:
                nxt = min(current.children, key=lambda c: c.proof)
            else:
                nxt = min(current.children, key=lambda c: c.disproof)
        current = nxt
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
    board: Board | SearchState,
    stone: Stone,
    max_nodes: int = 10_000,
    max_time_s: float = 5.0,
) -> Move | None:
    """Run Proof-Number Search in threat-space.

    Returns the first move of a forced winning sequence for *stone*,
    or ``None`` if no win is proved within the node/time budget.

    Accepts an existing :class:`SearchState` to skip the per-call
    Pydantic→bitboard round-trip; the search restores the state to
    its starting ply on every iteration boundary, so the caller's
    state is unchanged on return.
    """
    attacker = stone
    defender = attacker.opponent()

    state = board if isinstance(board, SearchState) else SearchState.from_board(board, stone)
    root = _PNSNode(stone, _NodeType.OR)
    _evaluate_leaf(root, state, attacker)

    if root.proved:
        return _find_winning_move(state, stone)
    if root.disproved:
        return None

    nodes_created = 1
    deadline = time.monotonic() + max_time_s

    while root.proof != 0 and root.disproof != 0 and nodes_created < max_nodes:
        if time.monotonic() > deadline:
            break

        # 1) Select the most-proving node.
        mpn, path = _select_mpn(root)

        # 2) Replay path moves into the shared state.  The move at path[i]
        #    was played by the side that was to move at path[i-1].
        for i in range(1, len(path)):
            parent = path[i - 1]
            nd = path[i]
            who = attacker if parent.node_type is _NodeType.OR else defender
            assert nd.move is not None  # only root has move=None
            state.make(nd.move.row, nd.move.col, who)

        try:
            # 3) Expand the MPN leaf (state is already at its position).
            children_before = len(mpn.children)
            _expand(mpn, state)
            nodes_created += len(mpn.children) - children_before

            # 4) Backpropagate along selection path only.
            _backpropagate_path(path)
        finally:
            # 5) Unwind state back to root for the next iteration.
            for _ in range(len(path) - 1):
                state.unmake()

    if root.proved and root.children:
        # Root is OR -> find the proved child.
        for child in root.children:
            if child.proved:
                return child.move
    return None
