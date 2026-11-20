"""End-to-end tests for the ``/v1/games`` session endpoints.

These tests exercise the FastAPI router via :class:`TestClient` —
the goal is to cover the *API contract* (status codes, response
shapes, error envelopes), not the underlying ``GameRunner`` mechanics
(those live in ``gomoku/tests/test_runner.py``).

A typical UI workflow is rehearsed in :class:`TestEndToEnd` to make
sure the five endpoints compose into a complete game.
"""

from gomoku.models import Move, Stone


def _create(client, **kwargs) -> dict:
    """Create a session and return the parsed JSON body.

    Defaults to a 9x9 board (smaller than the 15x15 default) because
    every assertion is about API plumbing, not strategy — the smaller
    board keeps any incidental brain calls fast.
    """
    payload = {"board_size": 9, "win_length": 5, **kwargs}
    response = client.post("/v1/games", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# POST /v1/games
# ---------------------------------------------------------------------------


class TestCreateGame:
    def test_creates_external_session(self, client):
        body = _create(client)
        assert body["game_id"]
        assert body["side_to_move"] == int(Stone.BLACK)
        assert body["move_number"] == 1
        assert body["history"] == []
        assert body["black_brain"] is None
        assert body["white_brain"] is None
        assert body["result"] is None
        assert body["record_dir"] is None
        assert body["diag_dir"] is None

    def test_creates_brain_session(self, client):
        body = _create(client, black={"brain": "random"})
        assert body["black_brain"] == "random"
        assert body["white_brain"] is None

    def test_records_directory_when_record_true(self, client):
        body = _create(client, record=True)
        assert body["record_dir"] is not None
        assert body["diag_dir"] is None

    def test_diag_directory_when_diagnostics_true(self, client):
        body = _create(client, diagnostics=True)
        assert body["diag_dir"] is not None

    def test_unknown_brain_returns_404(self, client):
        response = client.post(
            "/v1/games",
            json={"board_size": 9, "black": {"brain": "no_such_brain"}},
        )
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "unknown_brain"

    def test_invalid_brain_overrides_returns_422(self, client):
        # ``depth=999`` is outside ``AlphaBetaBrainSettings``'s range —
        # a bad knob at create-time should surface as 422, not 500.
        response = client.post(
            "/v1/games",
            json={
                "board_size": 9,
                "black": {"brain": "alphabeta", "brain_overrides": {"depth": 999}},
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_brain_overrides"

    def test_rejects_extra_fields(self, client):
        response = client.post(
            "/v1/games",
            json={"board_size": 9, "secret_knob": 1},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/games/{id}
# ---------------------------------------------------------------------------


class TestGetGame:
    def test_returns_state_for_existing_session(self, client):
        created = _create(client)
        response = client.get(f"/v1/games/{created['game_id']}")
        assert response.status_code == 200
        assert response.json()["game_id"] == created["game_id"]

    def test_unknown_id_returns_404(self, client):
        response = client.get("/v1/games/does-not-exist")
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "session_not_found"


# ---------------------------------------------------------------------------
# DELETE /v1/games/{id}
# ---------------------------------------------------------------------------


class TestDeleteGame:
    def test_deletes_existing_session(self, client):
        created = _create(client)
        response = client.delete(f"/v1/games/{created['game_id']}")
        assert response.status_code == 204
        # And it really is gone.
        assert client.get(f"/v1/games/{created['game_id']}").status_code == 404

    def test_idempotent_for_missing_session(self, client):
        # Deleting a session that never existed must still 204 — the
        # contract is "afterwards, the ID is gone", which is true
        # regardless of whether it was there to begin with.
        response = client.delete("/v1/games/never-existed")
        assert response.status_code == 204


# ---------------------------------------------------------------------------
# POST /v1/games/{id}/moves
# ---------------------------------------------------------------------------


class TestRecordingIsFinished:
    def test_recording_has_the_result_and_checksum_when_the_game_ends(self, client):
        """The session's recorder observes the game: at the end the recording
        gets the result and its .md5sum.  (It used to stay partial.)"""
        from pathlib import Path

        from gomoku.recorder import load_record

        created = _create(client, record=True)
        url = f"/v1/games/{created['game_id']}/moves"
        for col in range(4):
            for stone, row in ((Stone.BLACK, 0), (Stone.WHITE, 8)):
                response = client.post(url, json={"stone": int(stone), "move": {"row": row, "col": col}})
                assert response.status_code == 200, response.text
        response = client.post(url, json={"stone": int(Stone.BLACK), "move": {"row": 0, "col": 4}})
        assert response.status_code == 200, response.text
        [path] = Path(created["record_dir"]).glob("*.json")
        record = load_record(path)
        assert record.result is not None and record.result.winner is Stone.BLACK
        assert len(record.moves) == 9
        assert path.with_suffix(".md5sum").exists()


class TestSubmitMove:
    def test_applies_an_external_move(self, client):
        created = _create(client)
        response = client.post(
            f"/v1/games/{created['game_id']}/moves",
            json={"stone": int(Stone.BLACK), "move": {"row": 4, "col": 4}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["side_to_move"] == int(Stone.WHITE)
        assert body["move_number"] == 2
        assert body["history"] == [
            {"stone": int(Stone.BLACK), "move": Move(row=4, col=4).model_dump(mode="json")},
        ]

    def test_wrong_turn_returns_422(self, client):
        created = _create(client)
        # Black to move; submit White instead.
        response = client.post(
            f"/v1/games/{created['game_id']}/moves",
            json={"stone": int(Stone.WHITE), "move": {"row": 4, "col": 4}},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "wrong_turn"
        assert detail["expected"] == int(Stone.BLACK)
        assert detail["got"] == int(Stone.WHITE)

    def test_invalid_move_returns_422(self, client):
        created = _create(client)
        # First a legal move so cell (4,4) is occupied...
        client.post(
            f"/v1/games/{created['game_id']}/moves",
            json={"stone": int(Stone.BLACK), "move": {"row": 4, "col": 4}},
        )
        # ...then White tries to play on the same cell.
        response = client.post(
            f"/v1/games/{created['game_id']}/moves",
            json={"stone": int(Stone.WHITE), "move": {"row": 4, "col": 4}},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_move"

    def test_game_already_over_returns_422(self, client):
        """After a winning sequence, further moves must be rejected."""
        created = _create(client, board_size=9)
        gid = created["game_id"]
        # Black plays five-in-a-row on row 0; White plays on row 8 to keep
        # the alternation legal without interfering.
        sequences = [
            (Stone.BLACK, Move(row=0, col=0)),
            (Stone.WHITE, Move(row=8, col=0)),
            (Stone.BLACK, Move(row=0, col=1)),
            (Stone.WHITE, Move(row=8, col=1)),
            (Stone.BLACK, Move(row=0, col=2)),
            (Stone.WHITE, Move(row=8, col=2)),
            (Stone.BLACK, Move(row=0, col=3)),
            (Stone.WHITE, Move(row=8, col=3)),
            (Stone.BLACK, Move(row=0, col=4)),  # five in a row → Black wins
        ]
        for stone, move in sequences:
            r = client.post(
                f"/v1/games/{gid}/moves",
                json={"stone": int(stone), "move": move.model_dump(mode="json")},
            )
            assert r.status_code == 200, r.text
        # Game is over; the next submission must be rejected.
        response = client.post(
            f"/v1/games/{gid}/moves",
            json={"stone": int(Stone.WHITE), "move": {"row": 8, "col": 4}},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "game_already_over"

    def test_unknown_session_returns_404(self, client):
        response = client.post(
            "/v1/games/no-such-id/moves",
            json={"stone": int(Stone.BLACK), "move": {"row": 0, "col": 0}},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/games/{id}/think
# ---------------------------------------------------------------------------


class TestThink:
    def test_brain_plays_on_its_turn(self, client):
        created = _create(client, black={"brain": "random"})
        response = client.post(f"/v1/games/{created['game_id']}/think", json={})
        assert response.status_code == 200
        body = response.json()
        assert "move" in body
        assert body["state"]["move_number"] == 2
        assert body["state"]["side_to_move"] == int(Stone.WHITE)
        # Without ``diagnostics: true`` the field is null.
        assert body["diagnostics"] is None

    def test_diagnostics_returned_when_requested(self, client):
        created = _create(client, black={"brain": "random"})
        response = client.post(
            f"/v1/games/{created['game_id']}/think",
            json={"diagnostics": True},
        )
        assert response.status_code == 200
        diag = response.json()["diagnostics"]
        assert diag is not None
        # The runner stamps the move number / stone before the brain runs.
        assert diag["move_number"] == 1
        assert diag["stone"] == int(Stone.BLACK)

    def test_external_side_returns_422(self, client):
        # Both sides external — calling /think has no brain to dispatch to.
        created = _create(client)
        response = client.post(f"/v1/games/{created['game_id']}/think", json={})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "external_side"
        assert detail["stone"] == int(Stone.BLACK)

    def test_unknown_session_returns_404(self, client):
        response = client.post("/v1/games/no-such-id/think", json={})
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# End-to-end: a complete mini-game through the API
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_external_vs_brain_full_workflow(self, client):
        """One create + alternating external-move / think calls + delete.

        Pins down the *whole* session lifecycle so a regression in any
        single endpoint that breaks a real UI flow gets caught here
        even if the per-endpoint tests still pass."""
        # Black is external (us); White is the random brain.
        created = _create(client, white={"brain": "random"})
        gid = created["game_id"]

        # Move 1: external Black plays the centre.
        r = client.post(
            f"/v1/games/{gid}/moves",
            json={"stone": int(Stone.BLACK), "move": {"row": 4, "col": 4}},
        )
        assert r.status_code == 200
        assert r.json()["side_to_move"] == int(Stone.WHITE)

        # Move 2: White's brain plays via /think.
        r = client.post(f"/v1/games/{gid}/think", json={})
        assert r.status_code == 200
        state = r.json()["state"]
        assert state["side_to_move"] == int(Stone.BLACK)
        assert state["move_number"] == 3
        assert len(state["history"]) == 2

        # Read back via GET to confirm persistence between calls.
        r = client.get(f"/v1/games/{gid}")
        assert r.status_code == 200
        assert len(r.json()["history"]) == 2

        # Tear down.
        assert client.delete(f"/v1/games/{gid}").status_code == 204
        assert client.get(f"/v1/games/{gid}").status_code == 404
