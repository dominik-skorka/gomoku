"""Guard the ``v1`` contract against breaking changes.

``contracts/v1.openapi.json`` is a snapshot of every ``/v1`` operation
and every schema, with documentation text stripped.  Two tests compare
the live OpenAPI document against it:

* **No breaking changes.**  Removing an endpoint, a parameter or a field,
  making a parameter, body or field required, or changing a type fails
  with a list of what broke.  Such a change belongs in a new version
  (see :mod:`gomoku_fastapi.versioning`).
* **Snapshot is current.**  Backwards-compatible changes (new endpoints,
  new optional fields) fail too until the snapshot is refreshed, so
  every change to the public surface is reviewed in a diff:

      uv run python packages/gomoku-fastapi/tests/test_api_contract.py --update
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

SNAPSHOT = Path(__file__).parent / "contracts" / "v1.openapi.json"
PREFIX = "/v1/"

# Keys that document an API without changing what goes over the wire.
_DOC_KEYS = frozenset({"description", "title", "summary", "examples", "example", "operationId", "tags"})


def _shape(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _shape(v) for k, v in node.items() if k not in _DOC_KEYS}
    if isinstance(node, list):
        return [_shape(item) for item in node]
    return node


def contract(openapi: dict[str, Any]) -> dict[str, Any]:
    """The wire-relevant part of the v1 surface of an OpenAPI document."""
    paths = {p: ops for p, ops in openapi["paths"].items() if p.startswith(PREFIX)}
    return _shape({"paths": paths, "schemas": openapi.get("components", {}).get("schemas", {})})


def breaking_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Every change from *old* to *new* that could break an existing client."""
    problems: list[str] = []

    for path, old_ops in old["paths"].items():
        new_ops = new["paths"].get(path)
        if new_ops is None:
            problems.append(f"endpoint removed: {path}")
            continue
        for method, old_op in old_ops.items():
            new_op = new_ops.get(method)
            if new_op is None:
                problems.append(f"operation removed: {method.upper()} {path}")
                continue
            problems += _operation_changes(f"{method.upper()} {path}", old_op, new_op)

    for name, old_schema in old["schemas"].items():
        new_schema = new["schemas"].get(name)
        if new_schema is None:
            problems.append(f"schema removed: {name}")
            continue
        problems += _schema_changes(name, old_schema, new_schema)
    return problems


def _operation_changes(op: str, old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    problems = []
    old_params = {(p["in"], p["name"]): p for p in old.get("parameters", [])}
    new_params = {(p["in"], p["name"]): p for p in new.get("parameters", [])}
    for key, param in old_params.items():
        if key not in new_params:
            problems.append(f"{op}: {key[0]} parameter removed: {key[1]}")
        elif param.get("schema") != new_params[key].get("schema"):
            problems.append(f"{op}: {key[0]} parameter type changed: {key[1]}")
    for key, param in new_params.items():
        was_required = old_params.get(key, {}).get("required", False)
        if param.get("required", False) and not was_required:
            problems.append(f"{op}: {key[0]} parameter now required: {key[1]}")
    old_body, new_body = old.get("requestBody", {}), new.get("requestBody", {})
    if new_body.get("required") and not old_body.get("required"):
        problems.append(f"{op}: request body now required")
    if "content" in old_body and "content" in new_body and old_body["content"] != new_body["content"]:
        problems.append(f"{op}: request body type changed")
    for status, response in old.get("responses", {}).items():
        if status.startswith("2") and status not in new.get("responses", {}):
            problems.append(f"{op}: success response removed: {status}")
        elif status.startswith("2") and response.get("content") != new["responses"][status].get("content"):
            problems.append(f"{op}: {status} response type changed")
    return problems


def _schema_changes(name: str, old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    problems = []
    old_props, new_props = old.get("properties", {}), new.get("properties", {})
    for field, field_schema in old_props.items():
        if field not in new_props:
            problems.append(f"{name}: field removed: {field}")
        elif field_schema != new_props[field]:
            problems.append(f"{name}: field type changed: {field}")
    for field in sorted(set(new.get("required", [])) - set(old.get("required", []))):
        problems.append(f"{name}: field now required: {field}")
    rest_old = {k: v for k, v in old.items() if k not in {"properties", "required"}}
    rest_new = {k: v for k, v in new.items() if k not in {"properties", "required"}}
    if rest_old != rest_new:
        problems.append(f"{name}: schema type changed")
    return problems


def _live_contract() -> dict[str, Any]:
    from gomoku_fastapi.main import create_app

    return contract(create_app().openapi())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_v1_has_no_breaking_changes():
    problems = breaking_changes(json.loads(SNAPSHOT.read_text()), _live_contract())
    assert not problems, (
        "Breaking changes to the v1 API:\n  "
        + "\n  ".join(problems)
        + "\nKeep v1 backwards compatible and put these changes in a new version "
        "(see gomoku_fastapi/versioning.py)."
    )


def test_v1_snapshot_is_current():
    assert _live_contract() == json.loads(SNAPSHOT.read_text()), (
        "The v1 API changed in a backwards-compatible way.  Review the change, then refresh the "
        "snapshot:\n  uv run python packages/gomoku-fastapi/tests/test_api_contract.py --update"
    )


class TestBreakingChangeDetection:
    """The checker itself: breaking edits are caught, additive ones are not."""

    @staticmethod
    def _base() -> dict[str, Any]:
        return json.loads(SNAPSHOT.read_text())

    def test_identical_contract_is_compatible(self):
        assert breaking_changes(self._base(), self._base()) == []

    def test_additive_changes_are_compatible(self):
        old = self._base()
        new = copy.deepcopy(old)
        new["paths"]["/v1/new-endpoint"] = {"get": {"responses": {"200": {}}}}
        new["schemas"]["MoveRequest"]["properties"]["new_optional"] = {"type": "string"}
        new["schemas"]["BrandNew"] = {"type": "object"}
        assert breaking_changes(old, new) == []

    def test_removed_endpoint(self):
        old = self._base()
        new = copy.deepcopy(old)
        del new["paths"]["/v1/move"]
        assert breaking_changes(old, new) == ["endpoint removed: /v1/move"]

    def test_removed_and_retyped_and_required_fields(self):
        old = self._base()
        new = copy.deepcopy(old)
        props = new["schemas"]["MoveRequest"]["properties"]
        del props["diagnostics"]
        props["brain"] = {"type": "integer"}
        props["extra"] = {"type": "string"}
        new["schemas"]["MoveRequest"]["required"] = [*new["schemas"]["MoveRequest"]["required"], "extra"]
        assert sorted(breaking_changes(old, new)) == [
            "MoveRequest: field now required: extra",
            "MoveRequest: field removed: diagnostics",
            "MoveRequest: field type changed: brain",
        ]

    def test_removed_success_response(self):
        old = self._base()
        new = copy.deepcopy(old)
        del new["paths"]["/v1/brains"]["get"]["responses"]["200"]
        assert breaking_changes(old, new) == ["GET /v1/brains: success response removed: 200"]


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit(f"usage: {sys.argv[0]} --update")
    os.environ.setdefault("GOMOKU_API__LOGFIRE_CONSOLE", "false")
    os.environ.setdefault("GOMOKU_API__LOGFIRE_SEND", "false")
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps(_live_contract(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {SNAPSHOT}")
