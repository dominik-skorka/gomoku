"""Threat-based brain — VCF (Victory by Continuous Fours) and VCT (Victory by Continuous Threats).

VCF searches only four-threats (forcing single-cell responses), enabling very
deep searches.  VCT extends this to also consider live-three threats, where the
defender has more responses but a forced win can still be proven.

When neither finds a forced win the brain falls back to alpha-beta search.
"""

from __future__ import annotations

import logfire
from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku.rules import check_winner

from gomoku_brain.alphabeta import AlphaBetaBrain
from gomoku_brain.candidates import get_candidate_moves

# ---------------------------------------------------------------------------
# Direction vectors (horizontal, vertical, two diagonals)
# ---------------------------------------------------------------------------
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# ---------------------------------------------------------------------------
# Threat detection helpers
# ---------------------------------------------------------------------------


def _line_cells(
    board: Board,
    row: int,
    col: int,
    dr: int,
    dc: int,
    stone: Stone,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Count consecutive ``stone`` cells in both directions from (row, col).

    Returns (count, open_ends, list_of_empty_neighbours_at_open_ends).
    The stone at (row, col) is *not* included in count (caller decides context).
    """
    size = board.size
    count = 0
    open_ends = 0
    end_cells: list[tuple[int, int]] = []

    for sign in (1, -1):
        r, c = row + dr * sign, col + dc * sign
        while 0 <= r < size and 0 <= c < size and board.at(r, c) is stone:
            count += 1
            r += dr * sign
            c += dc * sign
        if 0 <= r < size and 0 <= c < size and board.at(r, c) is Stone.EMPTY:
            open_ends += 1
            end_cells.append((r, c))

    return count, open_ends, end_cells


def _find_four_threats(board: Board, stone: Stone) -> list[tuple[Move, Move | None]]:
    """Find all moves that create a four-in-a-row for ``stone``.

    Returns a list of ``(attack_move, defence_move | None)``.
    ``defence_move`` is the single cell the opponent *must* play to block, or
    ``None`` when the four is a *double-four* (two open ends → unblockable,
    i.e. immediate win).

    Detects:
    - Straight fours: placing a stone extends three-in-a-row to four.
    - Gap fours: placing a stone fills a gap inside a would-be five
      (e.g. XX_XX pattern).
    """
    threats: list[tuple[Move, Move | None]] = []
    seen_attacks: set[tuple[int, int, int, int]] = set()  # (row, col, dr, dc)

    candidates = get_candidate_moves(board, radius=2)

    for move in candidates:
        r, c = move.row, move.col
        trial = board.place_stone(move, stone)

        for dr, dc in _DIRECTIONS:
            key = (r, c, dr, dc)
            if key in seen_attacks:
                continue

            # Count the line through the placed stone
            count, open_ends, end_cells = _line_cells(trial, r, c, dr, dc, stone)
            total = count + 1  # +1 for the placed stone itself

            if total == 5:
                # Immediate five — not a threat, it's a win
                threats.append((move, None))
                seen_attacks.add(key)
            elif total == 4:
                if open_ends == 2:
                    # Live four → two ways to complete → unblockable
                    threats.append((move, None))
                elif open_ends == 1 and end_cells:
                    # Dead four → exactly one cell the opponent must block
                    dr2, dc2 = end_cells[0]
                    threats.append((move, Move(row=dr2, col=dc2)))
                seen_attacks.add(key)

    return threats


def _find_three_threats(board: Board, stone: Stone) -> list[tuple[Move, list[Move]]]:
    """Find moves that create a *live* three (open at both ends).

    Returns ``(attack_move, defence_moves)`` — the defence moves are the cells
    the opponent could play to neutralise the threat (the two open ends plus
    any adjacent cells that break the line).
    """
    threats: list[tuple[Move, list[Move]]] = []
    candidates = get_candidate_moves(board, radius=2)

    for move in candidates:
        r, c = move.row, move.col
        trial = board.place_stone(move, stone)

        for dr, dc in _DIRECTIONS:
            count, open_ends, end_cells = _line_cells(trial, r, c, dr, dc, stone)
            total = count + 1

            if total == 3 and open_ends == 2:
                # Live three — both ends open ⇒ two natural defence cells
                defence = [Move(row=er, col=ec) for er, ec in end_cells]
                threats.append((move, defence))

    return threats


def _find_winning_move(board: Board, stone: Stone) -> Move | None:
    """Return a move that immediately completes five-in-a-row, or ``None``."""
    for move in get_candidate_moves(board, radius=2):
        trial = board.place_stone(move, stone)
        result = check_winner(trial, move)
        if result is not None and result.winner is stone:
            return move
    return None


# ---------------------------------------------------------------------------
# VCF search
# ---------------------------------------------------------------------------


def _vcf(
    board: Board,
    stone: Stone,
    depth: int,
    is_attacker: bool,
) -> Move | None:
    """Recursive VCF search.

    *Attacker* can only play moves that create four-threats.
    *Defender* must respond to each four-threat with the forced block.
    Returns the first attacking move of a winning sequence, or ``None``.
    """
    if depth <= 0:
        return None

    if is_attacker:
        threats = _find_four_threats(board, stone)
        for attack, defence in threats:
            # Check immediate win
            trial = board.place_stone(attack, stone)
            result = check_winner(trial, attack)
            if result is not None and result.winner is stone:
                return attack

            # Opponent gets to respond — can they win first?
            if _find_winning_move(trial, stone.opponent()) is not None:
                continue

            if defence is None:
                # Unblockable four (live-four or five) → win
                return attack

            # Defender is forced to block
            trial2 = trial.place_stone(defence, stone.opponent())

            # Recurse: attacker tries another four-threat
            if _vcf(trial2, stone, depth - 2, is_attacker=True) is not None:
                return attack

    return None


# ---------------------------------------------------------------------------
# VCT search
# ---------------------------------------------------------------------------


def _vct(
    board: Board,
    stone: Stone,
    depth: int,
    is_attacker: bool,
) -> Move | None:
    """Recursive VCT search.

    Attacker plays four-threats *and* live-three threats.
    Defender must block four-threats (forced), and for three-threats tries all
    reasonable defence moves (including own counter four-threats).
    """
    if depth <= 0:
        return None

    if is_attacker:
        # 1) Try VCF first (four-threats are stronger)
        vcf_result = _vcf(board, stone, depth, is_attacker=True)
        if vcf_result is not None:
            return vcf_result

        # 2) Try three-threats
        three_threats = _find_three_threats(board, stone)
        for attack, defence_moves in three_threats:
            trial = board.place_stone(attack, stone)

            # Check if attack itself wins
            result = check_winner(trial, attack)
            if result is not None and result.winner is stone:
                return attack

            # Opponent gets to respond — can they win first?
            if _find_winning_move(trial, stone.opponent()) is not None:
                continue

            # Defender responds: must refute ALL defence moves to prove win
            all_refuted = True
            if not defence_moves:
                all_refuted = False
            else:
                for defence in defence_moves:
                    if not trial.is_valid_move(defence):
                        continue
                    trial2 = trial.place_stone(defence, stone.opponent())

                    # Defender may also play a counter four-threat
                    opponent_fours = _find_four_threats(trial2, stone.opponent())
                    if opponent_fours:
                        # Opponent has a counter-threat — this defence line
                        # is not simply refutable; abandon this attack.
                        all_refuted = False
                        break

                    if _vct(trial2, stone, depth - 2, is_attacker=True) is None:
                        all_refuted = False
                        break

            if all_refuted:
                return attack

    return None


# ---------------------------------------------------------------------------
# ThreatBrain
# ---------------------------------------------------------------------------


class ThreatBrain:
    """MoveProvider using VCF / VCT with alpha-beta fallback."""

    def __init__(
        self,
        depth: int = 4,
        vcf_depth: int = 20,
        vct_depth: int = 8,
    ) -> None:
        self.depth = depth
        self.vcf_depth = vcf_depth
        self.vct_depth = vct_depth
        self._fallback = AlphaBetaBrain(depth=depth)

    @property
    def name(self) -> str:
        return "threat"

    def get_move(self, board: Board, stone: Stone) -> Move:
        with logfire.span(
            "brain.threat",
            vcf_depth=self.vcf_depth,
            vct_depth=self.vct_depth,
            stone=str(stone),
        ):
            # 0) If we can win immediately, do it
            win = _find_winning_move(board, stone)
            if win is not None:
                logfire.info("Immediate win: {move}", move=str(win))
                return win

            # 1) If opponent can win next turn, we must block
            opp = stone.opponent()
            opp_win = _find_winning_move(board, opp)
            if opp_win is not None:
                logfire.info("Blocking opponent immediate win at: {move}", move=str(opp_win))
                return opp_win

            # 2) Try VCF — very fast, deep search
            move = _vcf(board, stone, self.vcf_depth, is_attacker=True)
            if move is not None:
                logfire.info("VCF found winning move: {move}", move=str(move))
                return move

            # 3) Try VCT — broader threat search
            move = _vct(board, stone, self.vct_depth, is_attacker=True)
            if move is not None:
                logfire.info("VCT found winning move: {move}", move=str(move))
                return move

            # 4) Check if opponent has a VCF threat; if so, try to block
            opp_vcf = _vcf(board, opp, self.vcf_depth, is_attacker=True)
            if opp_vcf is not None:
                logfire.info("Blocking opponent VCF at: {move}", move=str(opp_vcf))
                if board.is_valid_move(opp_vcf):
                    return opp_vcf

            # 5) Fallback to alpha-beta
            logfire.info("No threat sequence found, falling back to alpha-beta")
            return self._fallback.get_move(board, stone)
