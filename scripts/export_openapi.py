"""Write the full OpenAPI document of one API version to a file.

The web client's TypeScript types are generated from this file, so the
frontend never needs a running server to build, and CI can fail when the
committed copy drifts from what the API serves.

    uv run python scripts/export_openapi.py            # writes web/openapi/v1.json
    uv run python scripts/export_openapi.py --check    # exit 1 if it is stale
"""

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from gomoku_fastapi.main import create_app

ROOT = Path(__file__).resolve().parent.parent


def render(version: str) -> str:
    with TestClient(create_app()) as client:
        response = client.get(f"/{version}/openapi.json")
        response.raise_for_status()
        doc = response.json()
    # info.version is the server package version; it changes on every
    # release without changing the contract, so it is left out of the file.
    doc.get("info", {}).pop("version", None)
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args()
    out: Path = args.out or ROOT / "web" / "openapi" / f"{args.version}.json"
    text = render(args.version)
    if args.check:
        if not out.exists() or out.read_text() != text:
            print(f"{out.relative_to(ROOT)} is stale: run `uv run python scripts/export_openapi.py`", file=sys.stderr)
            return 1
        print(f"{out.relative_to(ROOT)} is up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
