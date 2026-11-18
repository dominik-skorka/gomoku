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

from gomoku_brain.candidates import get_candidate_moves
from gomoku_brain.search_state import SearchState
from gomoku_brain.threat_detect import (
    _find_four_threats,
    _find_three_threats,
    _find_winning_move,
    _has_unblockable_four,
    _scan_context,
    _ScanCtx,
)
from gomoku_brain.threat_search import _state_is_win

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF = 1_000_000_000  # Practical infinity for proof/disproof numbers

# Position table for one search: zobrist hash -> the node for that
# position.  A position reached through a different move order links to
# the node that already exists, so the search runs on a DAG instead of
# re-searching the same subtree per move order — 30-65% of all
# expansions in unproductive searches used to be such repeats.
#
# Sharing is sound because a node's status is a pure property of its
# position: stones are only ever added (no repetition, so the graph is
# acyclic), and the node/time budget never marks anything solved.  The
# hash encodes the side to move, and attacker/defender strictly
# alternate, so one key never maps to both an OR and an AND node.
_NodeTable = dict[int, "_PNSNode"]


class _NodeType(Enum):
    OR = auto()  # attacker to move
    AND = auto()  # defender to move


# ---------------------------------------------------------------------------
# PNS Node
# ---------------------------------------------------------------------------


class _PNSNode:
    """A single position in the proof-number search graph.

    The position itself is materialised on demand by replaying edge
    moves into the shared :class:`SearchState` owned by
    :func:`pns_threat_search`.  Moves live on the edges (``moves[i]``
    leads to ``children[i]``) because a node can have several parents.
    """

    __slots__ = (
        "best_index",
        "children",
        "disproof",
        "expanded",
        "moves",
        "node_type",
        "proof",
    )

    def __init__(self, node_type: _NodeType) -> None:
        self.node_type = node_type
        self.expanded = False
        self.children: list[_PNSNode] = []
        self.moves: list[Move] = []
        # Initial proof/disproof: 1/1 for unknown nodes
        self.proof = 1
        self.disproof = 1
        # Index of the most-proving child, cached by ``_update_ancestors``
        # so ``_select_mpn`` walks the graph in O(depth).
        self.best_index = -1


# ---------------------------------------------------------------------------
# Evaluation of leaf nodes
# ---------------------------------------------------------------------------


def _mark(node: _PNSNode, proved: bool) -> None:
    """Mark ``node`` as solved."""
    if proved:
        node.proof = 0
        node.disproof = INF
    else:
        node.proof = INF
        node.disproof = 0


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
    5. Every cell that defuses an attacker's unblockable-four threat
       (see :func:`_unblockable_four_defences`).

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

    # 5) Attacker threats of an unblockable four (a live four, or a
    #    double four, next move).  Occupying the attack cell is only one
    #    defence: taking either end of the would-be four, for one, leaves
    #    the attacker a four that can be blocked.  Missing those let the
    #    search "prove" wins the defender can refute.
    cells |= _unblockable_four_defences(state, attacker, ctx)

    return [Move(row=r, col=c) for r, c in sorted(cells) if state.is_empty(r, c)]


def _unblockable_four_defences(state: SearchState, attacker: Stone, ctx: _ScanCtx | None) -> set[tuple[int, int]]:
    """Cells after which the attacker no longer has an unblockable four.

    Empty when the attacker has no such threat.  Tried over the node's
    candidate cells: any cell that can affect a four lies within the
    candidate radius of the attacker's stones.  The detection reuses
    *ctx*, which stays valid because only the defender moves (see the
    staleness contract on ``_ScanCtx``).
    """
    if not _has_unblockable_four(state, attacker, ctx):
        return set()
    if ctx is not None:
        _grid, _size, candidates, _dirs = ctx
        cells = [(int(candidates[i, 0]), int(candidates[i, 1])) for i in range(candidates.shape[0])]
    else:
        cells = [(m.row, m.col) for m in get_candidate_moves(state)]
    defender = attacker.opponent()
    defences: set[tuple[int, int]] = set()
    for r, c in cells:
        if not state.is_empty(r, c):
            continue
        state.make(r, c, defender)
        try:
            if not _has_unblockable_four(state, attacker, ctx):
                defences.add((r, c))
        finally:
            state.unmake()
    return defences


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _child_at(state: SearchState, node_type: _NodeType, table: _NodeTable) -> tuple[_PNSNode, bool]:
    """Return ``(node, is_new)`` for the position ``state`` is currently at."""
    key = state.zobrist
    child = table.get(key)
    if child is not None:
        return child, False
    child = _PNSNode(node_type)
    table[key] = child
    return child, True


def _link(node: _PNSNode, child: _PNSNode, move: Move) -> None:
    node.children.append(child)
    node.moves.append(move)


def _expand(node: _PNSNode, state: SearchState, attacker: Stone, table: _NodeTable) -> int:
    """Generate children for ``node``; return how many new nodes were created.

    Precondition: ``state`` is at ``node``'s position.  Postcondition:
    ``state`` is restored to ``node``'s position (intermediate
    make/unmake cycles are balanced by ``try`` / ``finally`` blocks).

    A child whose position already has a node (reached earlier through
    another move order) is linked rather than created, and keeps
    whatever that node has already learned.

    One scan context is built per expansion and shared across every
    detection call — node-level enumeration runs at the context's own
    position, and all post-make probes target the side that did not
    just move, so the staleness contract on ``_ScanCtx`` holds.
    """
    node.expanded = True
    defender = attacker.opponent()
    ctx = _scan_context(state)
    created = 0

    if node.node_type is _NodeType.OR:
        # Attacker moves: four-threats and three-threats only.

        # Immediate win — the node is proved without materialising a child.
        if _find_winning_move(state, attacker, ctx) is not None:
            _mark(node, True)
            return 0

        # If defender has an immediate win, attacker must block it.
        def_win = _find_winning_move(state, defender, ctx)
        if def_win is not None:
            if state.is_empty(def_win.row, def_win.col):
                state.make(def_win.row, def_win.col, attacker)
                try:
                    child, is_new = _child_at(state, _NodeType.AND, table)
                    if is_new:
                        created += 1
                        _evaluate_leaf(child, state, attacker, ctx)
                    _link(node, child, def_win)
                finally:
                    state.unmake()
            else:
                _mark(node, False)
            _update_ancestors(node)
            return created

        # Four-threats first (stronger), then live-three threats; a cell
        # that is both is tried once.
        attacks = [atk for atk, _defence in _find_four_threats(state, attacker, ctx)]
        attacks.extend(atk for atk, _defs in _find_three_threats(state, attacker, ctx))
        seen_moves: set[tuple[int, int]] = set()
        for attack in attacks:
            key = (attack.row, attack.col)
            if key in seen_moves:
                continue
            seen_moves.add(key)

            state.make(attack.row, attack.col, attacker)
            try:
                if _state_is_win(state, attack.row, attack.col):
                    child, is_new = _child_at(state, _NodeType.AND, table)
                    created += is_new
                    _mark(child, True)
                    _link(node, child, attack)
                    _update_ancestors(node)
                    return created

                # The defender wins first — this attack is not playable.
                if _find_winning_move(state, defender, ctx) is not None:
                    continue

                # The check above is all ``_evaluate_leaf`` would do for
                # an AND child, so a new node keeps its 1/1 defaults.
                child, is_new = _child_at(state, _NodeType.AND, table)
                created += is_new
                _link(node, child, attack)
            finally:
                state.unmake()

        if not node.children:
            # No viable attack moves -> disproved
            _mark(node, False)

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
                    _mark(node, True)
                    return 0
                child, is_new = _child_at(state, _NodeType.OR, table)
                created += is_new
                if _state_is_win(state, atk_win.row, atk_win.col):
                    # Defender's block itself completes five for defender
                    # (rare but possible: defender had own 4-in-a-row through
                    # the block cell).  Disprove this OR-child.
                    _mark(child, False)
                elif is_new:
                    _evaluate_leaf(child, state, attacker, ctx)
                _link(node, child, atk_win)
                _update_ancestors(node)
                return created
            finally:
                state.unmake()

        defences = _pns_defence_moves(state, attacker, ctx)

        if not defences:
            # No blocks or counter-threats -> attacker's threat is unanswerable
            _mark(node, True)
            return 0

        for cand in defences:
            state.make(cand.row, cand.col, defender)
            try:
                child, is_new = _child_at(state, _NodeType.OR, table)
                created += is_new
                if _state_is_win(state, cand.row, cand.col):
                    _mark(child, False)
                    _link(node, child, cand)
                    _update_ancestors(node)
                    return created
                if is_new:
                    _evaluate_leaf(child, state, attacker, ctx)
                _link(node, child, cand)
            finally:
                state.unmake()

    _update_ancestors(node)
    return created


# ---------------------------------------------------------------------------
# Proof/disproof number update
# ---------------------------------------------------------------------------


def _update_ancestors(node: _PNSNode) -> None:
    """Recompute proof/disproof from children (non-recursive; called bottom-up).

    Also caches the index of the most-proving / most-disproving child in
    ``node.best_index`` so :func:`_select_mpn` can walk the graph in
    O(depth) rather than O(depth × branching).
    """
    children = node.children
    if not children:
        return

    best_index = 0
    if node.node_type is _NodeType.OR:
        best_proof = children[0].proof
        dp_sum = 0
        for i, c in enumerate(children):
            if c.proof < best_proof:
                best_index = i
                best_proof = c.proof
            dp_sum += c.disproof
        node.proof = best_proof
        node.disproof = min(dp_sum, INF)
    else:
        best_disproof = children[0].disproof
        pn_sum = 0
        for i, c in enumerate(children):
            if c.disproof < best_disproof:
                best_index = i
                best_disproof = c.disproof
            pn_sum += c.proof
        node.proof = min(pn_sum, INF)
        node.disproof = best_disproof
    node.best_index = best_index


# ---------------------------------------------------------------------------
# MPN selection with path tracking
# ---------------------------------------------------------------------------


def _select_mpn(node: _PNSNode) -> tuple[_PNSNode, list[_PNSNode], list[Move]]:
    """Walk from root to the most-proving leaf.

    Returns ``(leaf, path, moves)`` where ``moves[i]`` is the edge from
    ``path[i]`` to ``path[i + 1]``.  Uses the ``best_index`` cached by
    :func:`_update_ancestors`.  With shared nodes a cached index can be
    stale (the child was updated through another parent); the walk then
    ends on an already-solved node and the caller's backpropagation
    refreshes the whole path, so the next walk is exact.
    """
    path = [node]
    moves: list[Move] = []
    current = node
    while current.expanded and current.children:
        i = current.best_index
        moves.append(current.moves[i])
        current = current.children[i]
        path.append(current)
    return current, path, moves


# ---------------------------------------------------------------------------
# Path-based backpropagation
# ---------------------------------------------------------------------------


def _backpropagate_path(path: list[_PNSNode]) -> None:
    """Update proof/disproof numbers along the selection path only (bottom-up).

    O(path_length) per call instead of O(graph_size).
    """
    for node in reversed(path):
        if node.expanded and node.children:
            _update_ancestors(node)


# ---------------------------------------------------------------------------
# Main PNS entry point
# ---------------------------------------------------------------------------


class PNSStatus(Enum):
    """Outcome of a threat-space proof-number search."""

    PROVED = auto()  # the attacker has a forced win
    DISPROVED = auto()  # no forced win exists in threat-space
    UNKNOWN = auto()  # the node or time budget ran out first


def pns_threat_status(
    board: Board | SearchState,
    stone: Stone,
    max_nodes: int = 10_000,
    max_time_s: float = 5.0,
) -> tuple[PNSStatus, Move | None]:
    """Run Proof-Number Search in threat-space and report how it ended.

    Returns ``(PROVED, first winning move)``, ``(DISPROVED, None)`` when
    the search established that *stone* has no forced win by threats,
    or ``(UNKNOWN, None)`` when the budget ran out first.  Telling the
    last two apart matters to callers that use "no forced win" as a
    safety guarantee.

    ``max_nodes`` counts distinct positions.  Accepts an existing
    :class:`SearchState` to skip the per-call Pydantic→bitboard
    round-trip; the search restores the state to its starting ply on
    every iteration boundary, so the caller's state is unchanged on
    return.
    """
    attacker = stone
    defender = attacker.opponent()

    state = board if isinstance(board, SearchState) else SearchState.from_board(board, stone)
    root = _PNSNode(_NodeType.OR)
    table: _NodeTable = {state.zobrist: root}
    _evaluate_leaf(root, state, attacker)

    if root.proof == 0:
        return PNSStatus.PROVED, _find_winning_move(state, stone)
    if root.disproof == 0:
        return PNSStatus.DISPROVED, None
    nodes_created = 1
    deadline = time.monotonic() + max_time_s
    # Every iteration either creates a node or repairs a stale path, so
    # this bound is never reached in practice; it guarantees termination.
    iterations_left = max_nodes * 4

    while root.proof != 0 and root.disproof != 0 and nodes_created < max_nodes and iterations_left > 0:
        iterations_left -= 1
        if time.monotonic() > deadline:
            break

        # 1) Select the most-proving node.
        mpn, path, moves = _select_mpn(root)

        # 2) Replay the edge moves into the shared state.  The move out of
        #    path[i] is played by the side to move at path[i].
        for parent, move in zip(path, moves, strict=False):
            who = attacker if parent.node_type is _NodeType.OR else defender
            state.make(move.row, move.col, who)

        try:
            # 3) Expand the leaf (state is already at its position).  A
            #    stale walk can end on an expanded, solved node; it then
            #    only needs the backpropagation below.
            if not mpn.expanded:
                nodes_created += _expand(mpn, state, attacker, table)

            # 4) Backpropagate along the selection path only.
            _backpropagate_path(path)
        finally:
            # 5) Unwind state back to root for the next iteration.
            for _ in moves:
                state.unmake()

    if root.proof == 0:
        # Root is OR -> return the move to a proved child.
        for child, move in zip(root.children, root.moves, strict=True):
            if child.proof == 0:
                return PNSStatus.PROVED, move
    if root.disproof == 0:
        return PNSStatus.DISPROVED, None
    return PNSStatus.UNKNOWN, None


def pns_threat_search(
    board: Board | SearchState,
    stone: Stone,
    max_nodes: int = 10_000,
    max_time_s: float = 5.0,
) -> Move | None:
    """Return the first move of a forced win for *stone*, or ``None``.

    ``None`` covers both "no forced win" and "budget ran out"; use
    :func:`pns_threat_status` when the difference matters.
    """
    status, move = pns_threat_status(board, stone, max_nodes, max_time_s)
    return move if status is PNSStatus.PROVED else None
