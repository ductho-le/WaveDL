"""Optional bounded live delivery; the durable journal remains authoritative.

One daemon thread owns one asyncio loop and at most one network operation.
An absolute coroutine deadline covers resolution, connection, TLS and all IO.
Cancellation closes the transport. OS DNS resolution may finish in an executor
after cancellation; we never retry or wait for that executor during bounded
close. Any failure disables live delivery; file replay repairs missing events.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from wavedl.core.protocol import encode, strict_loads


if TYPE_CHECKING:
    from collections.abc import Mapping

    from wavedl.core.protocol import Envelope


@dataclass(frozen=True, repr=False)
class LivePushConfig:
    url: str = field(repr=False)
    token: str = field(repr=False)
    deadline: float = 2.0
    max_record_bytes: int = 64 * 1024
    max_pending_bytes: int = 1024 * 1024

    def __post_init__(self):
        try:
            if not isinstance(self.url, str) or any(
                ord(c) <= 32 or ord(c) >= 127 for c in self.url
            ):
                raise ValueError
            parsed = urlsplit(self.url)
            host = parsed.hostname
            port = parsed.port
            if (
                parsed.scheme not in {"http", "https"}
                or not host
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
                or "%" in host
                or port == 0
            ):
                raise ValueError
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
                if re.fullmatch(r"[0-9.]+", host):
                    raise ValueError
                if not re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host
                ):
                    raise ValueError
                if any(
                    not label
                    or len(label) > 63
                    or label.startswith("-")
                    or label.endswith("-")
                    for label in host.split(".")
                ):
                    raise ValueError
            if parsed.scheme == "http" and not (
                host.lower() == "localhost"
                or (address is not None and address.is_loopback)
            ):
                raise ValueError
            if (
                not self.token
                or len(self.token) > 4096
                or any(not 33 <= ord(c) <= 126 for c in self.token)
            ):
                raise ValueError
            if type(self.deadline) not in (int, float) or not 0 < self.deadline <= 30:
                raise ValueError
            if (
                type(self.max_record_bytes) is not int
                or type(self.max_pending_bytes) is not int
                or not 1 <= self.max_record_bytes <= 64 * 1024
                or not self.max_record_bytes <= self.max_pending_bytes <= 1024 * 1024
            ):
                raise ValueError
        except (TypeError, ValueError, AttributeError):
            raise ValueError("Invalid live-push configuration") from None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]):
        url = environment.get("WAVEDL_DAEMON_URL")
        token = environment.get("WAVEDL_DAEMON_TOKEN")
        if url is None and token is None:
            return None
        if url is None or token is None:
            raise ValueError("Incomplete live-push configuration")
        return cls(url, token)


class LivePushClient:
    """Nonblocking ordered producer, with bounded capacity including in-flight IO."""

    def __init__(self, config: LivePushConfig):
        self._config = config
        self._queue = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._disabled = False
        self._closing = False
        self._pending_bytes = 0
        self._next = None
        self._counters = {"queued": 0, "acknowledged": 0, "failures": 0, "dropped": 0}
        self._loop = None
        self._task = None
        self._thread = threading.Thread(
            target=self._work, name="wavedl-live-push", daemon=True
        )
        self._thread.start()

    @property
    def counters(self):
        with self._lock:
            return self._counters | {
                "pending_bytes": self._pending_bytes,
                "disabled": self._disabled,
            }

    def _disable_locked(self):
        self._disabled = True
        self._increment("failures")
        self._increment("dropped", len(self._queue))
        self._queue.clear()
        self._pending_bytes = 0
        self._wake.set()
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._cancel_task)
            except RuntimeError:
                pass

    def _increment(self, name, amount=1):
        self._counters[name] = min(2**63 - 1, self._counters[name] + amount)

    def submit(self, envelope: Envelope) -> bool:
        with self._lock:
            if self._disabled or self._closing:
                self._increment("dropped")
                return False
        try:
            record = encode(envelope).rstrip(b"\n")
            body = (
                b'{"attempt":'
                + str(envelope.attempt).encode()
                + b',"events":['
                + record
                + b"]}"
            )
        except Exception:
            with self._lock:
                self._disable_locked()
            return False
        with self._lock:
            if self._disabled or self._closing:
                self._increment("dropped")
                return False
            identity = (envelope.run_id, envelope.attempt, envelope.seq)
            expected = self._next or (envelope.run_id, envelope.attempt, 0)
            if (
                identity != expected
                or (envelope.seq == 0 and envelope.type != "hello")
                or len(record) > self._config.max_record_bytes
                or self._pending_bytes + len(body) > self._config.max_pending_bytes
            ):
                self._disable_locked()
                return False
            self._next = (envelope.run_id, envelope.attempt, envelope.seq + 1)
            self._pending_bytes += len(body)
            self._queue.append((envelope, body))
            self._increment("queued")
            self._wake.set()
        return True

    async def _send(self, envelope, body):
        parsed = urlsplit(self._config.url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        writer = None
        try:
            reader, writer = await asyncio.open_connection(
                host,
                port,
                ssl=ssl.create_default_context() if parsed.scheme == "https" else None,
                limit=8192,
            )
            if parsed.scheme == "http":
                peer = writer.get_extra_info("peername")
                if not peer or not ipaddress.ip_address(peer[0]).is_loopback:
                    raise ValueError("HTTP peer must be loopback")
            host_header = f"[{host}]" if ":" in host else host
            request = (
                f"POST /api/runs/{envelope.run_id}/events HTTP/1.1\r\n"
                f"Host: {host_header}:{port}\r\n"
                f"Authorization: Bearer {self._config.token}\r\n"
                "Content-Type: application/json\r\nConnection: close\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode("ascii") + body
            writer.write(request)
            await writer.drain()
            header = await reader.readuntil(b"\r\n\r\n")
            if len(header) > 8192:
                raise ValueError("Response header exceeds bound")
            lines = header.decode("ascii").split("\r\n")
            status = lines[0].split(" ")
            if (
                len(status) < 2
                or status[0] not in {"HTTP/1.0", "HTTP/1.1"}
                or status[1] != "200"
            ):
                raise ValueError("Delivery rejected")
            headers = {}
            for line in lines[1:]:
                if line:
                    name, value = line.split(":", 1)
                    name = name.lower()
                    if name in headers:
                        raise ValueError("Duplicate response header")
                    headers[name] = value.strip()
            length = headers.get("content-length", "")
            if (
                "transfer-encoding" in headers
                or not length.isdecimal()
                or not 1 <= int(length) <= 4096
            ):
                raise ValueError("Invalid acknowledgement framing")
            ack = strict_loads(await reader.readexactly(int(length)))
            if (
                not isinstance(ack, dict)
                or set(ack) != {"attempt", "seq"}
                or type(ack["attempt"]) is not int
                or type(ack["seq"]) is not int
                or ack["attempt"] != envelope.attempt
                or ack["seq"] < envelope.seq
            ):
                raise ValueError("Invalid acknowledgement")
        finally:
            if writer is not None:
                writer.close()
                # Do not let TLS shutdown extend the absolute operation deadline.
                if writer.transport is not None:
                    writer.transport.abort()

    def _work(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            while True:
                self._wake.wait()
                with self._lock:
                    if self._disabled or (self._closing and not self._queue):
                        return
                    if not self._queue:
                        self._wake.clear()
                        continue
                    envelope, body = self._queue.popleft()
                try:
                    self._task = loop.create_task(
                        asyncio.wait_for(
                            self._send(envelope, body), timeout=self._config.deadline
                        )
                    )
                    loop.run_until_complete(self._task)
                except BaseException:
                    with self._lock:
                        if not self._disabled:
                            self._disable_locked()
                    return
                finally:
                    self._task = None
                with self._lock:
                    if not self._disabled:
                        self._pending_bytes -= len(body)
                        self._increment("acknowledged")
        finally:
            loop.close()

    def close(self, timeout: float = 0.25):
        """Try bounded drain, then cancel the actual network task without waiting."""
        timeout = max(0.0, min(float(timeout), 1.0))
        deadline = time.monotonic() + timeout
        with self._lock:
            self._closing = True
            self._wake.set()
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            with self._lock:
                if not self._disabled:
                    self._disable_locked()
            loop = self._loop
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(self._cancel_task)
                except RuntimeError:
                    pass

    def _cancel_task(self):
        if self._task is not None:
            self._task.cancel()
