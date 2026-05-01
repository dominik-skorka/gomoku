"""``POST /v1/move`` — stateless brain query.

The tests use the ``random`` brain by default because it has zero
configuration and runs in microseconds — every assertion is about the
*API surface*, not the brain's tactical strength.  A separate
end-to-end smoke test uses ``alphabeta`` with ``depth=1`` so we cover
the thread-pool path with a brain that actually evaluates positions.
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone


def _empty_board_payload(size: int = 9) -> dict:
    """A minimal-but-valid ``MoveRequest`` JSON dict on an empty board."""
    return {
        "board": Board(size=size).model_dump(mode="json"),
        "side_to_move": int(Stone.BLACK),
        "brain": "random",
    }


class TestPostMoveHappyPath:
    def test_random_brain_returns_a_legal_move(self, client):
        response = client.post("/v1/move", json=_empty_board_payload())
        assert response.status_code == 200
        body = response.json()
        # The response must echo which brain was used (post-alias resolution).
        assert body["brain"] == "random"
        # On a blank board ``get_candidate_moves`` returns the centre cell
        # only, so the random brain has no choice — pinning the move
        # keeps the test deterministic.
        assert body["move"] == Move(row=4, col=4).model_dump(mode="json")

    def test_alphabeta_with_overrides_runs_via_thread_pool(self, client):
        """End-to-end check that the ``anyio.to_thread.run_sync`` path
        works for a brain that does real CPU work.  ``depth=1`` is fast
        enough to keep the test under a second but still exercises the
        full search pipeline."""
        payload = _empty_board_payload()
        payload["brain"] = "alphabeta"
        payload["brain_overrides"] = {"depth": 1}
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["brain"] == "alphabeta"
        # On the empty board the only candidate is the centre.
        assert body["move"] == Move(row=4, col=4).model_dump(mode="json")

    def test_advances_an_existing_position(self, client):
        """Make sure the API doesn't ignore the input board — pre-place
        a stone and confirm the brain plays somewhere different."""
        b = Board(size=9).place_stone(Move(row=4, col=4), Stone.BLACK)
        payload = {
            "board": b.model_dump(mode="json"),
            "side_to_move": int(Stone.WHITE),
            "brain": "random",
        }
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 200
        move = response.json()["move"]
        # The picked cell must not be the one we already occupied.
        assert (move["row"], move["col"]) != (4, 4)


class TestPostMoveValidation:
    def test_rejects_unknown_brain_with_404(self, client):
        payload = _empty_board_payload()
        payload["brain"] = "no_such_brain"
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail["error"] == "unknown_brain"
        # The error must include the list of available brains so a
        # client doesn't have to make a second call to ``GET /v1/brains``.
        assert "alphabeta" in detail["available"]

    def test_rejects_empty_side_to_move(self, client):
        payload = _empty_board_payload()
        payload["side_to_move"] = int(Stone.EMPTY)
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_side_to_move"

    def test_rejects_full_board(self, client):
        # Manually mark every cell occupied so ``Board.is_full()`` returns
        # True without forcing us to play an actual full game.
        full_grid = [[int(Stone.BLACK)] * 9 for _ in range(9)]
        payload = {
            "board": {
                "size": 9,
                "grid": full_grid,
                "occupied": [[r, c] for r in range(9) for c in range(9)],
            },
            "side_to_move": int(Stone.WHITE),
            "brain": "random",
        }
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "board_full"

    def test_rejects_invalid_brain_overrides_with_422(self, client):
        # ``depth=999`` violates ``AlphaBetaBrainSettings`` (1..10).  This
        # must surface as a 422 — *not* a 500 from a pydantic
        # ValidationError leaking through.
        payload = _empty_board_payload()
        payload["brain"] = "alphabeta"
        payload["brain_overrides"] = {"depth": 999}
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_brain_overrides"

    def test_rejects_extra_fields_in_request(self, client):
        """``MoveRequest`` is ``extra='forbid'`` — typoed knobs must fail
        loudly rather than be silently dropped."""
        payload = _empty_board_payload()
        payload["depth"] = 4  # belongs in ``brain_overrides``, not the top level
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 422


class TestBrainAliasResolution:
    """When a TOML alias is configured, ``brain="<alias>"`` should
    resolve through it.  We synthesise a settings file in the temp
    CWD (the conftest fixture chdirs there) so the walk-up picks it
    up automatically."""

    def test_alias_with_overrides_resolves_to_underlying_type(self, client, tmp_path):
        # Write the alias to the CWD (set by the conftest fixture).
        (tmp_path / "gomoku.toml").write_text('[brains.fast_ab]\ntype = "alphabeta"\ndepth = 1\n')
        # Reset cache so the next request picks up the new TOML.
        from gomoku_fastapi.deps import reset_settings_cache

        reset_settings_cache()
        payload = _empty_board_payload()
        payload["brain"] = "fast_ab"
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 200
        # The response echoes the *underlying* brain type, not the alias.
        assert response.json()["brain"] == "alphabeta"
