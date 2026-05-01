"""``POST /v1/evaluate`` — static heuristic eval.

The endpoint is a thin wrapper over
:func:`gomoku_brain.evaluate.evaluate_board`; these tests pin the
wire shape and a few signed-score invariants (own pieces help, the
empty board is neutral, perspective flips the sign).
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone


def _payload(board: Board, side: Stone) -> dict:
    return {
        "board": board.model_dump(mode="json"),
        "side_to_move": int(side),
    }


class TestEvaluateHappyPath:
    def test_empty_board_is_close_to_zero(self, client):
        """The evaluator returns roughly zero on a board with no stones —
        nobody has any pattern advantage."""
        response = client.post("/v1/evaluate", json=_payload(Board(size=15), Stone.BLACK))
        assert response.status_code == 200
        score = response.json()["score"]
        # Allow some slack for rounding / clamp behaviour, but the score
        # absolutely must not be near a win (±100 000).
        assert abs(score) < 100

    def test_three_in_a_row_favours_owner(self, client):
        """A live-three for BLACK should score positive from BLACK's
        perspective.  This is the single sharpest sanity check on the
        wire format — a sign flip here would break every UI feature
        downstream."""
        b = Board(size=15)
        for c in range(3, 6):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        response = client.post("/v1/evaluate", json=_payload(b, Stone.BLACK))
        assert response.status_code == 200
        assert response.json()["score"] > 0

    def test_perspective_flip_reverses_sign(self, client):
        """Same board, opposite side: scores must have opposite signs.
        Locks in that the API doesn't accidentally hard-code BLACK's
        perspective."""
        b = Board(size=15)
        for c in range(3, 6):
            b = b.place_stone(Move(row=7, col=c), Stone.BLACK)
        black_score = client.post("/v1/evaluate", json=_payload(b, Stone.BLACK)).json()["score"]
        white_score = client.post("/v1/evaluate", json=_payload(b, Stone.WHITE)).json()["score"]
        # The eval applies a 0.8x defensive weight, so the magnitudes
        # don't match exactly.  But the signs must.
        assert black_score > 0 > white_score


class TestEvaluateValidation:
    def test_rejects_empty_side_to_move(self, client):
        response = client.post("/v1/evaluate", json=_payload(Board(size=9), Stone.EMPTY))
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_side_to_move"

    def test_rejects_extra_fields(self, client):
        """``EvaluateRequest`` is ``extra='forbid'`` so typoed fields
        produce a clean 422 instead of being silently dropped."""
        payload = _payload(Board(size=9), Stone.BLACK)
        payload["depth"] = 4  # not a valid evaluate field
        response = client.post("/v1/evaluate", json=payload)
        assert response.status_code == 422
