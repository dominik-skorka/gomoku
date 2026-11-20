"""``POST /v1/analyze`` — multipart upload of a ``.diag.jsonl``.

The endpoint is a wire-level wrapper over
:func:`gomoku.analyze.analyze_diagnostics_text`.  These tests cover:

1. Happy-path round trip with a fixture file.
2. The response dict matches the canonical ``AnalysisReport.to_json()``
   shape (so a client can drop in the API for a local analyse
   pipeline).
3. The expected error responses on malformed / oversize / non-UTF-8
   input.
"""

import json

from gomoku.diagnostics import DiagnosticsHeader
from gomoku.models import GameConfig, Stone


def _minimal_diag_jsonl(tmp_path) -> str:
    """Build the smallest valid ``.diag.jsonl`` body the parser accepts:
    a header plus one move record."""
    header = DiagnosticsHeader(
        started_at=__import__("datetime").datetime.now(tz=__import__("datetime").UTC),
        game_config=GameConfig(board_size=9, win_length=5),
        players=[
            {"stone": int(Stone.BLACK), "provider": "alphabeta", "settings": {"depth": 1}},
            {"stone": int(Stone.WHITE), "provider": "random", "settings": {}},
        ],
    )
    move = {
        "move_number": 1,
        "stone": int(Stone.BLACK),
        "decision_path": "instant_win",
        "think_time_ms": 12,
    }
    summary = {"type": "summary", "total_moves": 1, "winner": int(Stone.BLACK)}
    return "\n".join(
        [
            header.model_dump_json(),
            json.dumps(move),
            json.dumps(summary),
        ]
    )


class TestAnalyzeHappyPath:
    def test_round_trip(self, client, tmp_path):
        body = _minimal_diag_jsonl(tmp_path)
        response = client.post(
            "/v1/analyze",
            files={"file": ("game.diag.jsonl", body, "application/x-ndjson")},
        )
        assert response.status_code == 200
        report = response.json()
        # Same top-level shape as ``AnalysisReport.to_json()``.
        assert set(report) == {"path", "header", "summary", "total_moves", "player_stats", "flagged"}
        assert report["total_moves"] == 1
        # ``path`` echoes the upload's filename so logs / client UIs can
        # show "analysed game.diag.jsonl" without making the user
        # remember which file they uploaded.
        assert report["path"] == "game.diag.jsonl"

    def test_player_stats_are_keyed_by_stone_string(self, client, tmp_path):
        """The ``player_stats`` dict uses ``str(Stone)`` keys
        (``"Black"`` / ``"White"``) so JSON consumers don't have to map
        integer codes back to names.  Pin this — it's the kind of
        change that would silently break a UI that reads
        ``report.player_stats.Black``."""
        body = _minimal_diag_jsonl(tmp_path)
        response = client.post(
            "/v1/analyze",
            files={"file": ("game.diag.jsonl", body, "application/x-ndjson")},
        )
        keys = set(response.json()["player_stats"])
        assert {"Black", "White"} <= keys


class TestAnalyzeValidation:
    def test_rejects_malformed_jsonl_with_400(self, client):
        body = "this is not jsonl\n"
        response = client.post(
            "/v1/analyze",
            files={"file": ("broken.diag.jsonl", body, "application/x-ndjson")},
        )
        assert response.status_code == 400
        # The detail must surface the parser's error so the user can fix
        # the file rather than guess.
        assert response.json()["detail"]["error"] == "malformed_diagnostics"

    def test_rejects_empty_upload_with_400(self, client):
        response = client.post(
            "/v1/analyze",
            files={"file": ("empty.diag.jsonl", b"", "application/x-ndjson")},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "malformed_diagnostics"

    def test_rejects_non_utf8_with_400(self, client):
        # 0x80 isn't a valid UTF-8 start byte.
        response = client.post(
            "/v1/analyze",
            files={"file": ("bin.diag.jsonl", b"\x80\x81\x82", "application/x-ndjson")},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "not_utf8"

    def test_missing_file_field_is_a_validation_error(self, client):
        """No ``file`` field at all → FastAPI's own 422.  Locked in so
        we don't accidentally accept empty uploads as a degenerate
        success."""
        response = client.post("/v1/analyze")
        assert response.status_code == 422
