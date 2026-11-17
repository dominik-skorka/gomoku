"""API versions: what exists, what each one's status is, and how responses say so.

The API is versioned by major version in the URL path (``/v1``), with no
minor versions in the path (Google AIP-185).  The rules, also in the
package README:

* **Within a version, only backwards-compatible changes** (AIP-180).
  New endpoints, new optional request fields, new response fields and
  new enum values are fine.  Removing or renaming an endpoint, field or
  enum value, making a request field required, changing a type, a
  default or a validation constraint, or changing a field's meaning is
  breaking.  ``tests/test_api_contract.py`` enforces the mechanical part.
* **A breaking change goes into a new version**, served alongside the
  old one: add an :class:`ApiVersion` to :data:`VERSIONS` and a router
  for it, reusing every unchanged resource router.
* **Retiring a version takes two steps.**  First it is deprecated, at
  least :data:`MIN_DEPRECATION_PERIOD` before its sunset: its responses
  carry ``Deprecation`` (RFC 9745) and ``Sunset`` (RFC 8594) headers and
  ``Link``s to the deprecation information and the successor.  After the
  sunset it moves to :data:`RETIRED`, whose requests get ``410 Gone``.

Deprecation and sunset headers describe the *whole version*, not only
the resource that returned them (the increased scope RFC 8594 section 5
asks to be documented).  Infrastructure endpoints (``/healthz``,
``/versions``) are not versioned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

VERSION_HEADER = "API-Version"

#: Response headers a browser client may need to read; CORS must expose them.
EXPOSED_HEADERS = (VERSION_HEADER, "Deprecation", "Sunset", "Link")

#: Where clients find the version list, with deprecation and sunset dates.
VERSIONS_PATH = "/versions"

#: Shortest allowed gap between deprecating a version and its sunset
#: (the period Google AIP-185 recommends).
MIN_DEPRECATION_PERIOD = timedelta(days=180)

_NAME = re.compile(r"v([1-9][0-9]*)")


def _major(name: str) -> int:
    match = _NAME.fullmatch(name)
    if match is None:
        msg = f"API version name {name!r} must be 'v' followed by a major number, like 'v1'"
        raise ValueError(msg)
    return int(match.group(1))


@dataclass(frozen=True)
class ApiVersion:
    """One URL version of the API and where it is in its lifecycle."""

    name: str
    released: date
    deprecated: date | None = None
    sunset: date | None = None
    successor: str | None = None

    @property
    def prefix(self) -> str:
        return f"/{self.name}"

    @property
    def major(self) -> int:
        return _major(self.name)

    def status(self, on: date | None = None) -> Literal["stable", "deprecated"]:
        """``deprecated`` once the deprecation date has passed.

        A future deprecation date is announced in the ``Deprecation``
        header ahead of time (RFC 9745 allows that), but the version is
        still stable until the day comes.
        """
        today = on or datetime.now(UTC).date()
        return "deprecated" if self.deprecated is not None and self.deprecated <= today else "stable"

    def response_headers(self) -> list[tuple[bytes, bytes]]:
        """Headers added to every response under this version's prefix."""
        headers = [(VERSION_HEADER, self.name)]
        links = []
        if self.deprecated is not None:
            # RFC 9745: a structured-field date, i.e. "@" + Unix seconds.
            headers.append(("Deprecation", f"@{_midnight_utc(self.deprecated)}"))
            links.append(f'<{VERSIONS_PATH}>; rel="deprecation"')
        if self.sunset is not None:
            headers.append(("Sunset", _http_date(self.sunset)))
            links.append(f'<{VERSIONS_PATH}>; rel="sunset"')
        if self.successor is not None:
            links.append(f'</{self.successor}>; rel="successor-version"')
        if links:
            headers.append(("Link", ", ".join(links)))
        return [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers]


def _midnight_utc(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def _http_date(day: date) -> str:
    """RFC 8594: an HTTP-date (``Sun, 06 Nov 1994 08:49:37 GMT``)."""
    return format_datetime(datetime.fromtimestamp(_midnight_utc(day), tz=UTC), usegmt=True)


def validate_registry(served: tuple[ApiVersion, ...], retired: tuple[ApiVersion, ...]) -> None:
    """Raise ``ValueError`` if the version registry breaks the lifecycle rules."""
    if not served:
        raise ValueError("at least one API version must be served")
    majors = [v.major for v in served]  # also validates every name
    if majors != sorted(set(majors)):
        raise ValueError("served versions must be unique and ordered oldest first")
    names = {v.name for v in served}
    if names & {v.name for v in retired}:
        raise ValueError("a version cannot be both served and retired")
    if served[-1].deprecated is not None:
        raise ValueError(f"the latest version {served[-1].name} cannot be deprecated")
    for v in (*served, *retired):
        if v.sunset is not None and v.deprecated is None:
            raise ValueError(f"{v.name}: a sunset needs a deprecation date first")
        if v.deprecated is not None:
            if v.deprecated < v.released:
                raise ValueError(f"{v.name}: deprecated before it was released")
            if v.sunset is None or v.successor is None:
                raise ValueError(f"{v.name}: a deprecated version needs a sunset date and a successor")
            # RFC 9745: the sunset MUST NOT be earlier than the deprecation.
            if v.sunset - v.deprecated < MIN_DEPRECATION_PERIOD:
                raise ValueError(
                    f"{v.name}: sunset must be at least {MIN_DEPRECATION_PERIOD.days} days after deprecation"
                )
        if v.successor is not None and (v.successor not in names or _major(v.successor) <= v.major):
            raise ValueError(f"{v.name}: successor must be a newer served version")


V1 = ApiVersion(name="v1", released=date(2026, 5, 1))

#: Every served version, oldest first.  The last entry is the latest.
VERSIONS: tuple[ApiVersion, ...] = (V1,)

#: Versions past their sunset: no longer served, answered with 410 Gone.
RETIRED: tuple[ApiVersion, ...] = ()

LATEST = VERSIONS[-1]

validate_registry(VERSIONS, RETIRED)


class VersionHeadersMiddleware:
    """Apply the version lifecycle to every request under a version prefix.

    * Served version: add its headers to the response.  A middleware
      rather than a router dependency, so error responses (404, 422,
      429) carry them too — a client on a deprecated version must see
      the warning whatever it receives.
    * Retired version: ``410 Gone``, naming the successor (RFC 8594
      section 9 suggests 410 for resources past their sunset).
    * Any other ``v<number>`` segment: ``404`` that lists the served
      versions, instead of a bare "Not Found".
    """

    def __init__(
        self,
        app: ASGIApp,
        versions: tuple[ApiVersion, ...] = VERSIONS,
        retired: tuple[ApiVersion, ...] = RETIRED,
    ) -> None:
        self.app = app
        self._served = {v.name: v.response_headers() for v in versions}
        self._retired = {v.name: v for v in retired}
        self._supported = ", ".join(v.name for v in versions)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        segment = scope["path"].lstrip("/").split("/", 1)[0]
        extra = self._served.get(segment)
        if extra is not None:

            async def send_with_headers(message: Message) -> None:
                if message["type"] == "http.response.start":
                    message["headers"] = [*message.get("headers", []), *extra]
                await send(message)

            await self.app(scope, receive, send_with_headers)
            return

        old = self._retired.get(segment)
        if old is not None:
            detail = f"API version {old.name} was retired on {old.sunset}; use {old.successor}."
            body = {"detail": detail, "successor": f"/{old.successor}", "versions": VERSIONS_PATH}
            await _send_json(send, 410, body, old.response_headers())
            return
        if _NAME.fullmatch(segment):
            detail = f"API version {segment} is not served. Supported versions: {self._supported}."
            await _send_json(send, 404, {"detail": detail, "versions": VERSIONS_PATH}, [])
            return
        await self.app(scope, receive, send)


async def _send_json(send: Send, status: int, body: dict[str, str], headers: list[tuple[bytes, bytes]]) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                *headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


__all__ = [
    "EXPOSED_HEADERS",
    "LATEST",
    "MIN_DEPRECATION_PERIOD",
    "RETIRED",
    "V1",
    "VERSIONS",
    "VERSIONS_PATH",
    "VERSION_HEADER",
    "ApiVersion",
    "VersionHeadersMiddleware",
    "validate_registry",
]
