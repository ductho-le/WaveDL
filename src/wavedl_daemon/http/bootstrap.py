# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Ephemeral browser credentials and credential-safe access logging.

Bootstrap codes and session tokens are 256-bit random capabilities retained
only as digests. Short-lived bootstrap codes are single-use; the process launch
code is reusable until shutdown. Sessions have sliding and
absolute expirations. Access-log filtering is explicit opt-in.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import unquote_plus

from wavedl_daemon.http.auth import credential_bytes


if TYPE_CHECKING:
    from collections.abc import Callable


_REDACTED = "[REDACTED]"
_CODE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_CODE_RUN = re.compile(r"[A-Za-z0-9_-]{43,}\Z")
_SENSITIVE_NAMES = frozenset(
    {"token", "access_token", "bootstrap", "bootstrap_code", "code", "authorization"}
)
_QUERY_FIELD = re.compile(r"([?&;])([^\s?&;=\"'<>]+)=([^\s&;#\"'<>]*)")
_AUTH_HEADER = re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,\"'<>&;#]+")
_CODE_CANDIDATE = re.compile(r"(?:[A-Za-z0-9_-]|%[0-9a-fA-F]{2}){43,}")
_SESSION_IDLE_SECONDS = 12 * 60 * 60
_SESSION_ABSOLUTE_SECONDS = 7 * 24 * 60 * 60
_MAX_SESSIONS = 16


@dataclass(frozen=True)
class SessionGrant:
    """A newly minted session token and its current Unix-millisecond expiry."""

    token: str
    expires_at: int


class BootstrapExchangeError(ValueError):
    """An invalid, expired, or already consumed bootstrap capability."""


class BootstrapCapacityError(RuntimeError):
    """The bounded pool is full; minting fails closed."""


class BootstrapCredentials:
    """Thread-safe pool for bounded bootstrap codes and revocable sessions.

    The supplied master token is immediately reduced to a SHA-256 digest. The
    pool never retains plaintext bootstrap or session credentials.
    """

    def __init__(
        self,
        access_token: str,
        *,
        ttl_seconds: float = 120,
        max_pending: int = 128,
        session_idle_seconds: float = _SESSION_IDLE_SECONDS,
        session_absolute_seconds: float = _SESSION_ABSOLUTE_SECONDS,
        max_sessions: int = _MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        master = credential_bytes(access_token)
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("Invalid bootstrap lifetime")
        if (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending < 1
        ):
            raise ValueError("Invalid bootstrap capacity")
        if not callable(clock):
            raise ValueError("Invalid bootstrap clock")
        for value, label in (
            (session_idle_seconds, "session idle lifetime"),
            (session_absolute_seconds, "session absolute lifetime"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid {label}")
        if (
            isinstance(max_sessions, bool)
            or not isinstance(max_sessions, int)
            or max_sessions < 1
        ):
            raise ValueError("Invalid session capacity")
        if not callable(wall_clock):
            raise ValueError("Invalid session wall clock")
        self._master_digest = hashlib.sha256(master).digest()
        self._ttl_seconds = ttl_seconds
        self._max_pending = max_pending
        self._session_idle_seconds = session_idle_seconds
        self._session_absolute_seconds = session_absolute_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._pending: dict[bytes, float] = {}
        self._launch_digest: bytes | None = None
        self._sessions: OrderedDict[bytes, tuple[float, float]] = OrderedDict()

    def mint_launch(self) -> str:
        """Mint the sole process-lifetime launch capability, retaining its digest."""
        with self._lock:
            if self._launch_digest is not None:
                raise BootstrapCapacityError("Launch code already minted")
            code = secrets.token_urlsafe(32)
            digest = hashlib.sha256(code.encode("ascii")).digest()
            if (
                self._contains_digest(digest, self._pending)
                or self._contains_digest(digest, self._sessions)
                or hmac.compare_digest(digest, self._master_digest)
            ):
                raise BootstrapCapacityError("Bootstrap unavailable")
            self._launch_digest = digest
            return code

    def mint(self) -> str:
        """Return a fresh short-lived code; never retain its plaintext."""
        with self._lock:
            now = self._clock()
            if not math.isfinite(now):
                raise BootstrapCapacityError("Bootstrap unavailable")
            self._expire(now)
            if len(self._pending) >= self._max_pending:
                raise BootstrapCapacityError("Bootstrap capacity reached")
            # No retry loop: even a hypothetical random collision fails closed.
            code = secrets.token_urlsafe(32)
            digest = hashlib.sha256(code.encode("ascii")).digest()
            if (
                self._contains_digest(digest, self._pending)
                or self._contains_digest(digest, self._sessions)
                or hmac.compare_digest(digest, self._master_digest)
            ):
                raise BootstrapCapacityError("Bootstrap unavailable")
            deadline = now + self._ttl_seconds
            if not math.isfinite(deadline):
                raise BootstrapCapacityError("Bootstrap unavailable")
            self._pending[digest] = deadline
            return code

    def exchange(self, code: object) -> SessionGrant:
        """Exchange a valid code and atomically mint its browser session."""
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise BootstrapExchangeError("Invalid or expired bootstrap code")
        candidate = hashlib.sha256(code.encode("ascii")).digest()
        with self._lock:
            now = self._clock()
            if not math.isfinite(now):
                raise BootstrapExchangeError("Invalid or expired bootstrap code")
            self._expire(now)
            matched = None
            for digest in self._pending:
                if hmac.compare_digest(candidate, digest):
                    matched = digest
            launch = self._launch_digest is not None and hmac.compare_digest(
                candidate, self._launch_digest
            )
            if matched is None and not launch:
                raise BootstrapExchangeError("Invalid or expired bootstrap code")
            grant, session_digest = self._new_session_locked(now)
            if matched is not None:
                del self._pending[matched]
            if len(self._sessions) >= self._max_sessions:
                self._evict_lru_locked()
            self._sessions[session_digest] = (
                min(
                    now + self._session_idle_seconds,
                    now + self._session_absolute_seconds,
                ),
                now + self._session_absolute_seconds,
            )
            return grant

    def _contains_digest(self, candidate: bytes, values: dict) -> bool:
        found = self._launch_digest is not None and hmac.compare_digest(
            candidate, self._launch_digest
        )
        for digest in values:
            found = hmac.compare_digest(candidate, digest) or found
        return found

    def _expire(self, now: float) -> None:
        self._pending = {
            digest: deadline
            for digest, deadline in self._pending.items()
            if deadline > now
        }
        self._sessions = OrderedDict(
            (digest, record)
            for digest, record in self._sessions.items()
            if min(record) > now
        )

    def _evict_lru_locked(self) -> None:
        if self._sessions:
            self._sessions.popitem(last=False)

    def _new_session_locked(
        self, now: float, absolute_deadline: float | None = None
    ) -> tuple[SessionGrant, bytes]:
        absolute = (
            now + self._session_absolute_seconds
            if absolute_deadline is None
            else absolute_deadline
        )
        idle = min(now + self._session_idle_seconds, absolute)
        if not math.isfinite(idle) or not math.isfinite(absolute) or idle <= now:
            raise BootstrapCapacityError("Session unavailable")
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("ascii")).digest()
        if (
            hmac.compare_digest(digest, self._master_digest)
            or self._contains_digest(digest, self._pending)
            or self._contains_digest(digest, self._sessions)
        ):
            raise BootstrapCapacityError("Session unavailable")
        wall_now = self._wall_clock()
        if not math.isfinite(wall_now):
            raise BootstrapCapacityError("Session unavailable")
        expiry_ms = (wall_now + idle - now) * 1000
        if not math.isfinite(expiry_ms) or not 0 < expiry_ms <= 2**53 - 1:
            raise BootstrapCapacityError("Session unavailable")
        expires_at = int(expiry_ms)
        return SessionGrant(token, expires_at), digest

    def mint_session(self) -> SessionGrant:
        """Mint a session from the master credential, subject to the pool bound."""
        with self._lock:
            now = self._clock()
            if not math.isfinite(now):
                raise BootstrapCapacityError("Session unavailable")
            self._expire(now)
            grant, digest = self._new_session_locked(now)
            if len(self._sessions) >= self._max_sessions:
                self._evict_lru_locked()
            self._sessions[digest] = (
                min(
                    now + self._session_idle_seconds,
                    now + self._session_absolute_seconds,
                ),
                now + self._session_absolute_seconds,
            )
            return grant

    def _candidate_digest(self, token: object) -> bytes:
        if isinstance(token, str):
            try:
                return hashlib.sha256(credential_bytes(token)).digest()
            except ValueError:
                pass
        return hashlib.sha256(b"").digest()

    def is_master(self, token: object) -> bool:
        """Compare a presented bearer to the stored master digest."""
        candidate = self._candidate_digest(token)
        return hmac.compare_digest(candidate, self._master_digest)

    def authorization_kind(self, token: object) -> str | None:
        """Return master/session authority and slide a live session's idle expiry."""
        candidate = self._candidate_digest(token)
        with self._lock:
            now = self._clock()
            if not math.isfinite(now):
                return None
            self._expire(now)
            master = hmac.compare_digest(candidate, self._master_digest)
            matched = None
            for digest in self._sessions:
                if hmac.compare_digest(candidate, digest):
                    matched = digest
            if master:
                return "master"
            if matched is None:
                return None
            _, absolute_deadline = self._sessions.pop(matched)
            self._sessions[matched] = (
                min(now + self._session_idle_seconds, absolute_deadline),
                absolute_deadline,
            )
            return "session"

    def authenticate(self, token: object) -> bool:
        """Accept master or live session; valid sessions slide their idle expiry."""
        return self.authorization_kind(token) is not None

    def rotate(self, token: object) -> SessionGrant:
        """Mint a replacement; rotating a session preserves its absolute bound."""
        candidate = self._candidate_digest(token)
        with self._lock:
            now = self._clock()
            if not math.isfinite(now):
                raise BootstrapExchangeError("Invalid or expired session")
            self._expire(now)
            is_master = hmac.compare_digest(candidate, self._master_digest)
            matched = None
            for digest in self._sessions:
                if hmac.compare_digest(candidate, digest):
                    matched = digest
            if not is_master and matched is None:
                raise BootstrapExchangeError("Invalid or expired session")
            absolute_deadline = None
            if matched is not None:
                absolute_deadline = self._sessions.pop(matched)[1]
            if len(self._sessions) >= self._max_sessions:
                self._evict_lru_locked()
            grant, digest = self._new_session_locked(now, absolute_deadline)
            self._sessions[digest] = (
                min(
                    now + self._session_idle_seconds,
                    absolute_deadline or now + self._session_absolute_seconds,
                ),
                absolute_deadline or now + self._session_absolute_seconds,
            )
            return grant

    def revoke(self, token: object) -> bool:
        """Revoke a presented session token; master credentials are not revocable."""
        candidate = self._candidate_digest(token)
        with self._lock:
            now = self._clock()
            if not math.isfinite(now):
                return False
            self._expire(now)
            matched = None
            for digest in self._sessions:
                if hmac.compare_digest(candidate, digest):
                    matched = digest
            if matched is None:
                return False
            del self._sessions[matched]
            return True

    def matches_token(self, token: str) -> bool:
        """Allow app assembly to reject a mismatched injected credential pool."""
        return hmac.compare_digest(
            hashlib.sha256(credential_bytes(token)).digest(), self._master_digest
        )


def _decode(value: str) -> str:
    # Bounded repeated decoding also covers nested URL encoding without parsing
    # an attacker-controlled URL (which may contain malformed IPv6 authorities).
    for _ in range(3):
        decoded = unquote_plus(value, errors="replace")
        if decoded == value:
            break
        value = decoded
    return value


class AccessLogRedactionFilter(logging.Filter):
    """Redact rendered messages, including uvicorn's five-argument format.

    All bootstrap-shaped strings are redacted, including expired/consumed
    codes. This avoids retaining a second plaintext or unbounded replay list.
    Non-sensitive query fields retain their original encoding and spelling.
    """

    def __init__(
        self,
        credentials: BootstrapCredentials | str,
        *,
        master_token: str | None = None,
    ):
        super().__init__()
        token = (
            master_token
            if isinstance(credentials, BootstrapCredentials)
            else credentials
        )
        if token is None:
            raise ValueError("Access-log redaction requires the master credential")
        credential_bytes(token)
        # Match either literal or percent-encoded characters without decoding
        # the whole message and altering non-sensitive pagination fields.
        pattern = "".join(
            f"(?:{re.escape(char)}|%{ord(char):02x}|%25{ord(char):02x}|%2525{ord(char):02x})"
            for char in token
        )
        self._token_pattern = re.compile(pattern, re.IGNORECASE)

    def redact(self, message: str) -> str:
        def query_field(match):
            separator, key, value = match.groups()
            if _decode(key).lower() in _SENSITIVE_NAMES:
                value = _REDACTED
            return f"{separator}{key}={value}"

        message = _QUERY_FIELD.sub(query_field, message)
        message = _AUTH_HEADER.sub(lambda match: match[1] + _REDACTED, message)
        message = self._token_pattern.sub(_REDACTED, message)
        return _CODE_CANDIDATE.sub(
            lambda match: (
                _REDACTED if _CODE_RUN.fullmatch(_decode(match[0])) else match[0]
            ),
            message,
        )

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn's AccessFormatter unpacks these arguments itself. Preserve
        # its contract rather than only sanitizing the rendered message.
        if (
            record.msg == '%s - "%s %s HTTP/%s" %d'
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            record.args = tuple(
                self.redact(value) if isinstance(value, str) else value
                for value in record.args
            )
            record.exc_info, record.exc_text, record.stack_info = None, None, None
            return True
        try:
            message = record.getMessage()
        except Exception:
            # Formatting failures must not fall back to repr(raw secret args).
            message = "[unformattable access log]"
        record.msg, record.args = self.redact(message), ()
        # Access logs should not carry exception tracebacks containing secrets.
        record.exc_info, record.exc_text, record.stack_info = None, None, None
        return True


def install_access_log_redaction(
    credentials: BootstrapCredentials | str,
    logger: logging.Logger | None = None,
    *,
    master_token: str | None = None,
) -> AccessLogRedactionFilter:
    """Explicitly protect a logger; importing/constructing the app is inert."""
    redactor = AccessLogRedactionFilter(credentials, master_token=master_token)
    (logger if logger is not None else logging.getLogger("uvicorn.access")).addFilter(
        redactor
    )
    return redactor
