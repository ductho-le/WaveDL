# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Header-only credentials and strict same-origin request validation."""

from __future__ import annotations

import hmac
import ipaddress
import re
from urllib.parse import urlsplit

from starlette.exceptions import HTTPException


DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]")
_TOKEN = re.compile(rb"[A-Za-z0-9._~+/=-]+\Z")
_DNS = re.compile(
    r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\Z"
)


def credential_bytes(token: str) -> bytes:
    """Validate an explicitly supplied credential without creating one."""
    if not isinstance(token, str):
        raise ValueError("Invalid daemon credential")
    try:
        value = token.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("Invalid daemon credential") from None
    if not _TOKEN.fullmatch(value) or len(value) > 4096:
        raise ValueError("Invalid daemon credential")
    return value


def _bearer_candidate(request) -> bytes:
    """Return the single syntactically valid bearer value or an empty value."""
    values = request.headers.getlist("authorization")
    candidate = b""
    if len(values) == 1:
        try:
            scheme, candidate = values[0].encode("ascii").split(b" ", 1)
            if scheme.lower() != b"bearer" or not _TOKEN.fullmatch(candidate):
                candidate = b""
        except (UnicodeEncodeError, ValueError):
            candidate = b""
    return candidate


def bearer_token(request) -> str | None:
    """Return a syntactically valid bearer for a route already checked by auth."""
    candidate = _bearer_candidate(request)
    return candidate.decode("ascii") if candidate else None


def require_auth(request, token: str | bytes | None = None, *, sessions=None) -> bool:
    """Accept one master or live session bearer, never URLs or cookies.

    Return ``True`` for master authority and ``False`` for session authority so
    the boundary can apply narrower route scopes without a second auth path.
    """
    expected = request.app.state.auth_token if token is None else token
    if isinstance(expected, str):
        expected = credential_bytes(expected)
    candidate = _bearer_candidate(request)
    if sessions is not None:
        try:
            presented = candidate.decode("ascii")
        except UnicodeDecodeError:
            presented = ""
        kind = sessions.authorization_kind(presented)
        if kind == "master":
            return True
        if kind == "session":
            return False
    elif hmac.compare_digest(candidate, expected):
        return True
    raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})


def authority(value: str) -> tuple[str, int | None]:
    """Parse a single ASCII authority without browser/parser ambiguity."""
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("Invalid authority")
    value = value.lower()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            raise ValueError("Invalid authority")
        host, rest = value[: end + 1], value[end + 1 :]
        ipaddress.IPv6Address(host[1:-1])
    else:
        host, separator, port = value.partition(":")
        rest = separator + port
        if not _DNS.fullmatch(host) or len(host) > 253:
            raise ValueError("Invalid authority")
    if not rest:
        return host, None
    if not re.fullmatch(r":[1-9][0-9]{0,4}", rest) or int(rest[1:]) > 65535:
        raise ValueError("Invalid authority")
    return host, int(rest[1:])


def validate_origin_host(request, allowed_hosts: tuple[str, ...]) -> None:
    """Reject DNS rebinding and foreign browser origins on every HTTP path."""
    hosts = request.headers.getlist("host")
    try:
        if len(hosts) != 1:
            raise ValueError("Invalid authority")
        host, port = authority(hosts[0])
        allowed = [authority(item) for item in allowed_hosts]
        if not any(
            host == name and (bound is None or bound == port) for name, bound in allowed
        ):
            raise ValueError("Invalid authority")
    except ValueError:
        raise HTTPException(400, "Invalid Host") from None
    origins = request.headers.getlist("origin")
    if not origins:
        return
    try:
        if len(origins) != 1:
            raise ValueError("Invalid origin")
        if any(ord(char) <= 32 or ord(char) >= 127 for char in origins[0]):
            raise ValueError("Invalid origin")
        origin = urlsplit(origins[0])
        if (
            origin.scheme not in {"http", "https"}
            or origin.path
            or origin.query
            or origin.fragment
        ):
            raise ValueError("Invalid origin")
        origin_host, origin_port = authority(origin.netloc)
        scheme = request.scope["scheme"]
        default = 443 if scheme == "https" else 80
        if (
            origin.scheme != scheme
            or origin_host != host
            or (origin_port or default) != (port or default)
        ):
            raise ValueError("Invalid origin")
    except ValueError:
        raise HTTPException(403, "Forbidden Origin") from None
