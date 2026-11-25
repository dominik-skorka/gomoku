"""``POST /v1/move`` — stateless brain query.

The tests use the ``random`` brain by default because it has zero
configuration and runs in microseconds — every assertion is about the
*API surface*, not the brain's tactical strength.  A separate
end-to-end smoke test uses ``alphabeta`` with ``depth=1`` so we cover
the thread-pool path with a brain that actually evaluates positions.
"""

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


class TestServerOnlyOverrides:
    """Settings that name server-side resources (``opening_book``) must
    not be settable by a client — otherwise a request could make the
    server open any file it can read."""

    def _payload(self, brain: str, overrides: dict) -> dict:
        payload = _empty_board_payload()
        payload["brain"] = brain
        payload["brain_overrides"] = overrides
        return payload

    def test_opening_book_override_rejected_without_touching_file(self, client, monkeypatch):
        from gomoku_brain import opening_book

        def fail_load(*_args, **_kwargs):
            raise AssertionError("opening book must not be loaded")

        monkeypatch.setattr(opening_book.OpeningBookFile, "load", classmethod(fail_load))
        response = client.post(
            "/v1/move",
            json=self._payload("alphabeta", {"opening_book": "/etc/passwd", "depth": 1}),
        )
        assert response.status_code == 422
        assert response.json()["detail"] == {"error": "override_not_allowed", "field": "opening_book"}

    def test_empty_opening_book_override_also_rejected(self, client):
        response = client.post("/v1/move", json=self._payload("alphabeta", {"opening_book": ""}))
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "override_not_allowed"

    def test_nested_opening_book_override_rejected(self, client):
        """The threat brain's ``fallback`` table is an alpha-beta settings
        model — the marker must be enforced one level down too."""
        response = client.post(
            "/v1/move",
            json=self._payload("threat", {"fallback": {"depth": 1, "opening_book": "/etc/passwd"}}),
        )
        assert response.status_code == 422
        assert response.json()["detail"] == {"error": "override_not_allowed", "field": "fallback.opening_book"}

    def test_session_create_rejects_opening_book_override(self, client):
        response = client.post(
            "/v1/games",
            json={
                "board_size": 9,
                "black": {"brain": "alphabeta", "brain_overrides": {"opening_book": "/etc/passwd"}},
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"] == {"error": "override_not_allowed", "field": "opening_book"}

    def test_regular_overrides_still_work(self, client):
        response = client.post(
            "/v1/move",
            json=self._payload("alphabeta", {"depth": 1, "threat_ordering": False, "max_candidates": 10}),
        )
        assert response.status_code == 200
        assert response.json()["move"] == Move(row=4, col=4).model_dump(mode="json")

    def test_nested_regular_override_still_works(self, client):
        response = client.post(
            "/v1/move",
            json=self._payload("threat", {"depth": 1, "fallback": {"depth": 1}}),
        )
        assert response.status_code == 200

    def test_unknown_override_key_is_clean_422(self, client):
        response = client.post("/v1/move", json=self._payload("alphabeta", {"no_such_knob": 1}))
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_brain_overrides"


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
