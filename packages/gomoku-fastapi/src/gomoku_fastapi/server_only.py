"""Settings only the server's configuration may set (``gomoku.settings.SERVER_ONLY``).

Brain settings such as ``opening_book`` name files on the server.  A
client must not choose them: ``brain_overrides`` are merged into the
settings the brain is built from, so a free-form path there would let any
client make the server probe and parse arbitrary files.

The helpers here walk a brain's settings model, nested models included
(``fallback: AlphaBetaBrainSettings | None`` of the threat brain), and

* :func:`pin_server_only` keeps every server-only setting at the value the
  server resolved and reports the ones a client tried to change;
* :func:`strip_server_only` / :func:`public_schema` leave them out of what
  the API publishes, so clients never see (or send back) server paths.
"""

import types
from typing import Any, Union, get_args, get_origin

from gomoku.settings import is_server_only
from pydantic import BaseModel


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """The pydantic model behind ``Model`` or ``Model | None``, if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        for arg in get_args(annotation):
            model = _model_of(arg)
            if model is not None:
                return model
    return None


def pin_server_only(
    model: type[BaseModel],
    requested: dict[str, Any],
    resolved: dict[str, Any],
    overrides: Any,
    path: str = "",
) -> list[str]:
    """Put the server's values back into *requested* for every server-only setting.

    *requested* is the validated dump of the settings with the overrides
    merged (changed in place); *resolved* is the dump of the server's own
    settings; *overrides* is what the client sent at this level.  Returns
    the dotted paths of server-only settings the client sent with a value
    other than the server's.  Leaving such a setting out is fine (the
    server's value applies), and so is repeating the server's value.
    """
    rejected: list[str] = []
    for name, field in model.model_fields.items():
        sent = isinstance(overrides, dict) and name in overrides
        if is_server_only(field):
            server_value = resolved.get(name, field.get_default(call_default_factory=True))
            if sent and requested.get(name) != server_value:
                rejected.append(path + name)
            requested[name] = server_value
            continue
        sub = _model_of(field.annotation)
        if sub is None or not isinstance(requested.get(name), dict):
            continue
        # A nested group the server leaves unset (None) plays with its defaults.
        sub_resolved = resolved.get(name)
        if not isinstance(sub_resolved, dict):
            sub_resolved = sub().model_dump()
        rejected += pin_server_only(
            sub, requested[name], sub_resolved, overrides.get(name) if sent else None, f"{path}{name}."
        )
    return rejected


def strip_server_only(model: type[BaseModel], values: dict[str, Any]) -> dict[str, Any]:
    """A copy of the settings *values* without the server-only settings, at every depth."""
    out: dict[str, Any] = {}
    for name, value in values.items():
        field = model.model_fields.get(name)
        if field is not None and is_server_only(field):
            continue
        sub = _model_of(field.annotation) if field is not None else None
        out[name] = strip_server_only(sub, value) if sub is not None and isinstance(value, dict) else value
    return out


def public_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON Schema without the server-only settings (top level and ``$defs``)."""
    schema = model.model_json_schema()

    def prune(node: dict[str, Any]) -> None:
        props = node.get("properties")
        if isinstance(props, dict):
            hidden = {k for k, p in props.items() if isinstance(p, dict) and p.get("server_only") is True}
            for key in hidden:
                del props[key]
            if isinstance(node.get("required"), list):
                node["required"] = [k for k in node["required"] if k not in hidden]

    prune(schema)
    for definition in schema.get("$defs", {}).values():
        prune(definition)
    return schema


__all__ = ["pin_server_only", "public_schema", "strip_server_only"]
