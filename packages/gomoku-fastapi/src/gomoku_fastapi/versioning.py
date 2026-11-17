"""API versions: what exists, what each one's status is, and how responses say so.

The API is versioned by URL prefix (``/v1``).  The rules, also in the
package README:

* **Within a version, only backwards-compatible changes.**  New
  endpoints, new optional request fields and new response fields are
  fine.  Removing or renaming an endpoint or field, making a request
  field required, or changing a field's type or meaning is breaking.
* **A breaking change goes into a new version**, served alongside the
  old one: add an :class:`ApiVersion` to :data:`VERSIONS` and a router
  for it, reusing every unchanged resource router.
* **Retiring a version takes two steps.**  First it is marked
  deprecated: its responses carry ``Deprecation`` (RFC 9745) and
  ``Sunset`` (RFC 8594) headers and a ``Link`` to the successor.  It is
  removed only after the sunset date.

Every versioned response carries an ``API-Version`` header, and the
unversioned ``GET /versions`` endpoint lists all versions with their
status.  Infrastructure endpoints (``/healthz``, ``/versions``) are not
versioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import format_datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

VERSION_HEADER = "API-Version"

#: Response headers a browser client may need to read; CORS must expose them.
EXPOSED_HEADERS = (VERSION_HEADER, "Deprecation", "Sunset", "Link")


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
    def status(self) -> Literal["stable", "deprecated"]:
        return "deprecated" if self.deprecated is not None else "stable"

    def response_headers(self) -> list[tuple[bytes, bytes]]:
        """Headers added to every response under this version's prefix."""
        headers = [(VERSION_HEADER, self.name)]
        if self.deprecated is not None:
            # RFC 9745: a structured-field date, i.e. "@" + Unix seconds.
            headers.append(("Deprecation", f"@{_midnight_utc(self.deprecated)}"))
        if self.sunset is not None:
            # RFC 8594: an HTTP-date.
            moment = datetime.fromtimestamp(_midnight_utc(self.sunset), tz=UTC)
            headers.append(("Sunset", format_datetime(moment, usegmt=True)))
        if self.successor is not None:
            headers.append(("Link", f'</{self.successor}>; rel="successor-version"'))
        return [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers]


def _midnight_utc(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


V1 = ApiVersion(name="v1", released=date(2026, 5, 1))

#: Every served version, oldest first.  The last entry is the latest.
VERSIONS: tuple[ApiVersion, ...] = (V1,)
LATEST = VERSIONS[-1]


class VersionHeadersMiddleware:
    """Add the version's headers to every response under its URL prefix.

    A middleware rather than a router dependency, so error responses
    (404, 422, 429) and responses a handler returns directly get the
    headers too — a client on a deprecated version must see the warning
    whatever it receives.
    """

    def __init__(self, app: ASGIApp, versions: tuple[ApiVersion, ...] = VERSIONS) -> None:
        self.app = app
        self._by_segment = {v.name: v.response_headers() for v in versions}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        segment = scope["path"].lstrip("/").split("/", 1)[0]
        extra = self._by_segment.get(segment)
        if extra is None:
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        await self.app(scope, receive, send_with_headers)


__all__ = [
    "EXPOSED_HEADERS",
    "LATEST",
    "V1",
    "VERSIONS",
    "VERSION_HEADER",
    "ApiVersion",
    "VersionHeadersMiddleware",
]
