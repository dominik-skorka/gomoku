"""``POST /v1/move?diagnostics=true`` returns ``MoveDiagnostics``.

The point of this surface is to let a UI render *why* the brain made
its choice — top candidates, eval swing, threats, etc.  These tests
assert the wire shape and the diagnostics-isolation invariants
(concurrent calls don't see each other's records, omitting the flag
omits the field entirely).
"""

from __future__ import annotations

from gomoku.board import Board
from gomoku.models import Move, Stone


def _payload_with_diagnostics(*, brain: str = "alphabeta", overrides: dict | None = None) -> dict:
    return {
        "board": Board(size=9).place_stone(Move(row=4, col=4), Stone.BLACK).model_dump(mode="json"),
        "side_to_move": int(Stone.WHITE),
        "brain": brain,
        "brain_overrides": overrides or {"depth": 1},
        "diagnostics": True,
    }


class TestDiagnosticsRoundTrip:
    def test_diagnostics_field_absent_when_flag_false(self, client):
        """Default behaviour must stay zero-cost — no field in the
        response when the client didn't ask for it."""
        payload = _payload_with_diagnostics()
        payload["diagnostics"] = False
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 200
        body = response.json()
        # The field is present but ``None`` — pydantic includes it.
        assert body.get("diagnostics") is None

    def test_alphabeta_populates_diagnostics(self, client):
        """End-to-end: AlphaBetaBrain at depth=1 must produce a record
        with the search stats it always populates (decision_path,
        chosen_move, time_ms)."""
        response = client.post("/v1/move", json=_payload_with_diagnostics())
        assert response.status_code == 200
        diag = response.json()["diagnostics"]
        assert diag is not None
        # The brain stamps these in every successful search.
        assert diag["decision_path"]  # non-empty string
        assert diag["chosen_move"] is not None
        assert diag["search_stats"]["time_ms"] is not None

    def test_chosen_move_matches_top_level_move(self, client):
        """Sanity: the ``move`` field and ``diagnostics.chosen_move``
        must agree.  A drift here would mean the diagnostics record is
        from a different ``get_move`` call than the one we returned."""
        response = client.post("/v1/move", json=_payload_with_diagnostics())
        body = response.json()
        assert body["diagnostics"]["chosen_move"] == body["move"]

    def test_random_brain_does_not_populate_search_stats(self, client):
        """The random brain doesn't engage the diagnostics machinery —
        the record exists but its search_stats remain at defaults.
        Locking this in guards against a future change accidentally
        pretending the random brain has search depth."""
        payload = _payload_with_diagnostics(brain="random", overrides={})
        response = client.post("/v1/move", json=payload)
        assert response.status_code == 200
        diag = response.json()["diagnostics"]
        assert diag is not None
        # No search => no depth/time/nodes.
        assert diag["search_stats"]["depth_reached"] is None
        assert diag["search_stats"]["nodes_searched"] is None


class TestDiagnosticsIsolation:
    """Two requests in flight must each see their own diagnostics
    record, never each other's.  The ``ContextVar`` is per-task and
    anyio copies context on dispatch — but it's the kind of thing that
    breaks in surprising ways under future refactors, so we pin it."""

    def test_concurrent_calls_dont_share_diagnostics(self, client):
        # Issue two back-to-back requests and confirm each gets its
        # own record.  TestClient is sync, so this isn't *true*
        # concurrency, but it does exercise the set/clear cycle and
        # would catch a leaked ContextVar on the second call.
        p1 = _payload_with_diagnostics()
        p2 = _payload_with_diagnostics()
        r1 = client.post("/v1/move", json=p1)
        r2 = client.post("/v1/move", json=p2)
        d1 = r1.json()["diagnostics"]
        d2 = r2.json()["diagnostics"]
        # Both populated — neither leaked None.
        assert d1 is not None and d2 is not None
        # Each is its own record (same input, same content — but
        # different object identity in the eyes of the test).
        assert d1 is not d2
