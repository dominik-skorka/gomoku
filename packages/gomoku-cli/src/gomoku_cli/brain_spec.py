"""Brain settings on the command line: ``NAME[,KEY=VALUE...]``.

``gomoku play --black threat,depth=8,fallback.depth=8`` plays the
``threat`` brain (a built-in name or a ``[brains.<alias>]`` from
``gomoku.toml``) with those settings changed.  The same ``KEY=VALUE``
parts work in the engine specs of ``gomoku match``, ``review`` and
``bench`` (``brain=threat,rev=master,pns_nodes=60000``).

* A dotted key reaches into a nested table: ``fallback.depth=8``,
  ``threats.own.split_threes=true``.
* A value is read as a TOML value (``8``, ``2.5``, ``true``, ``"text"``);
  anything that is not one is taken as a plain string (``threat_search=vct``).
* Keys are checked against the brain's settings, nested tables included:
  a misspelt key is an error, not silently ignored.
* Overrides apply on top of the alias's settings from ``gomoku.toml``.
"""

from __future__ import annotations

import tomllib
import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError


def split_parts(text: str) -> list[str]:
    """Split *text* on commas outside brackets and quotes."""
    parts: list[str] = []
    depth = 0
    quote = ""
    current: list[str] = []
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts]


def parse_value(text: str) -> Any:
    """*text* as a TOML value, or as a plain string when it is not one."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def set_dotted(target: dict[str, Any], key: str, value: Any) -> None:
    """Set ``a.b.c`` in *target* as nested tables."""
    *tables, last = key.split(".")
    node = target
    for name in tables:
        child = node.setdefault(name, {})
        if not isinstance(child, dict):
            msg = f"{key!r}: {name!r} is set to a value and also used as a table"
            raise ValueError(msg)
        node = child
    if isinstance(node.get(last), dict):
        msg = f"{key!r} is a table; set one of its keys instead"
        raise ValueError(msg)
    node[last] = value


def parse_overrides(parts: list[str]) -> dict[str, Any]:
    """``["depth=8", "fallback.depth=8"]`` as nested settings."""
    overrides: dict[str, Any] = {}
    for part in parts:
        key, sep, value = part.partition("=")
        key = key.strip()
        if not sep or not key or not value.strip():
            msg = f"expected KEY=VALUE, got {part!r}"
            raise ValueError(msg)
        set_dotted(overrides, key, parse_value(value.strip()))
    return overrides


def parse_brain_spec(text: str) -> tuple[str, dict[str, Any]]:
    """``"threat,depth=8"`` as ``("threat", {"depth": 8})``."""
    name, *rest = split_parts(text)
    if not name or "=" in name:
        msg = f"Invalid brain {text!r}: expected NAME[,KEY=VALUE...]"
        raise ValueError(msg)
    try:
        return name, parse_overrides(rest)
    except ValueError as exc:
        msg = f"Invalid brain {text!r}: {exc}"
        raise ValueError(msg) from exc


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return *base* with *overrides* applied; nested tables merge key by key."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        merged[key] = merge(current if isinstance(current, dict) else {}, value) if isinstance(value, dict) else value
    return merged


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """The settings model a field holds (``Model`` or ``Model | None``), if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        for arg in get_args(annotation):
            model = _model_of(arg)
            if model is not None:
                return model
    return None


def check_overrides(model: type[BaseModel], overrides: dict[str, Any], prefix: str = "") -> None:
    """Raise ``ValueError`` for a key *model* (or a nested table) does not have."""
    for key, value in overrides.items():
        field = model.model_fields.get(key)
        if field is None:
            valid = ", ".join(sorted(model.model_fields))
            msg = f"unknown setting {prefix + key!r}; valid here: {valid}"
            raise ValueError(msg)
        if isinstance(value, dict):
            nested = _model_of(field.annotation)
            if nested is not None:
                check_overrides(nested, value, f"{prefix}{key}.")
            elif get_origin(field.annotation) is not dict and field.annotation is not dict:
                msg = f"{prefix + key!r} is a single value, not a table"
                raise ValueError(msg)


def apply_overrides(model: type[BaseModel], settings: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """*settings* with *overrides* applied, checked and validated by *model*."""
    if not overrides:
        return settings
    check_overrides(model, overrides)
    try:
        return model(**merge(settings, overrides)).model_dump()
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ValueError(problems) from None


def describe(overrides: dict[str, Any], prefix: str = "") -> str:
    """``{"depth": 8, "fallback": {"depth": 8}}`` as ``depth=8,fallback.depth=8``."""
    parts = []
    for key, value in overrides.items():
        if isinstance(value, dict):
            parts.append(describe(value, f"{prefix}{key}."))
        else:
            parts.append(f"{prefix}{key}={value if not isinstance(value, bool) else str(value).lower()}")
    return ",".join(p for p in parts if p)
