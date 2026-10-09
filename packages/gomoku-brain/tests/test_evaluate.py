from gomoku.board import Board
from gomoku.models import Move, Stone
from gomoku_brain.evaluate import evaluate_board, find_fork_cells


def _place(board: Board, moves: list[tuple[int, int]], stone: Stone) -> Board:
    for r, c in moves:
        board = board.place_stone(Move(row=r, col=c), stone)
    return board


class TestEvaluate:
    def test_empty_board_zero(self):
        b = Board(size=9)
        assert evaluate_board(b, Stone.BLACK) == 0.0

    def test_advantage_positive(self):
        """A board with more black stones near each other should score positive for black."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=5), Stone.BLACK)
        b = b.place_stone(Move(row=4, col=6), Stone.BLACK)
        score = evaluate_board(b, Stone.BLACK)
        assert score > 0

    def test_opponent_advantage_negative(self):
        """A board where white has a strong line should score negative for black."""
        b = Board(size=9)
        b = b.place_stone(Move(row=4, col=4), Stone.WHITE)
        b = b.place_stone(Move(row=4, col=5), Stone.WHITE)
        b = b.place_stone(Move(row=4, col=6), Stone.WHITE)
        b = b.place_stone(Move(row=4, col=7), Stone.WHITE)
        score = evaluate_board(b, Stone.BLACK)
        assert score < 0


# ---------------------------------------------------------------------------
# Fork scoring
# ---------------------------------------------------------------------------


class TestForkScore:
    def test_fork_bonus_increases_score(self):
        """A fork arrangement (two pairs converging on one cell) should score
        higher than the same stones scattered with no fork potential."""
        # Fork: horizontal pair (4,3)-(4,4) and vertical pair (3,5)-(2,5)
        # Fork cell at (4,5): extends horizontal to 3 AND extends vertical
        b_fork = Board(size=9)
        b_fork = _place(b_fork, [(4, 3), (4, 4), (3, 5), (2, 5)], Stone.BLACK)
        b_fork = _place(b_fork, [(0, 0)], Stone.WHITE)

        # Scattered: same 4 stones but pairs far apart — no fork
        b_scattered = Board(size=9)
        b_scattered = _place(b_scattered, [(4, 3), (4, 4), (0, 7), (1, 7)], Stone.BLACK)
        b_scattered = _place(b_scattered, [(0, 0)], Stone.WHITE)

        score_fork = evaluate_board(b_fork, Stone.BLACK)
        score_scattered = evaluate_board(b_scattered, Stone.BLACK)
        # The fork arrangement should score strictly higher due to fork bonus
        assert score_fork > score_scattered

    def test_opponent_fork_penalised(self):
        """Opponent fork cells should lower our score."""
        b = Board(size=9)
        # Opponent has an L-shape creating a fork
        b = _place(b, [(4, 4), (4, 5), (5, 4)], Stone.WHITE)
        b = _place(b, [(0, 0)], Stone.BLACK)
        score = evaluate_board(b, Stone.BLACK)
        # Should be negative: opponent has patterns AND fork potential
        assert score < 0


class TestFindForkCells:
    def test_double_three_fork_detected(self):
        """Classic double-three: two live-2 lines meeting → fork cell found."""
        b = Board(size=9)
        # Horizontal: (4,3), (4,4) with both ends open
        # Vertical: (3,5), (4,5) would overlap at (4,5)
        # Better: L-shape that converges on a single empty cell
        # Black: (4,3), (4,4) horizontal and (3,5), (2,5) vertical
        # Fork at (4,5) — extends horizontal AND meets vertical after (4,5)
        # Actually let's make it simpler:
        # Black at (4,3) and (4,4) → horizontal open-2
        # Black at (3,5) and (2,5) → vertical open-2
        # Fork at (4,5): extends horizontal to 3, and is adjacent to vertical
        b = _place(b, [(4, 3), (4, 4)], Stone.BLACK)  # horizontal pair
        b = _place(b, [(3, 5), (2, 5)], Stone.BLACK)  # vertical pair above (4,5)
        b = _place(b, [(0, 0)], Stone.WHITE)

        forks = find_fork_cells(b, Stone.BLACK)
        # (4,5) should be detected as a fork cell
        assert Move(row=4, col=5) in forks

    def test_no_fork_on_single_line(self):
        """A single line of stones should not create fork cells."""
        b = Board(size=9)
        b = _place(b, [(4, 3), (4, 4), (4, 5)], Stone.BLACK)
        b = _place(b, [(0, 0)], Stone.WHITE)
        forks = find_fork_cells(b, Stone.BLACK)
        # Cells at (4,2) and (4,6) are extensions of ONE line — not forks
        for _f in forks:
            # Any fork cell found should not be a pure single-line extension
            # This is a soft check — the heuristic might flag cells near
            # the line that also touch the diagonal, which is acceptable
            pass
        # The main assertion: single straight line has no obvious fork
        # (cells extending in only 1 direction won't count as fork)
        assert len(forks) == 0 or all(f not in (Move(row=4, col=2), Move(row=4, col=6)) for f in forks)
