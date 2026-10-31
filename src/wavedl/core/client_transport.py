"""Stdlib-only authenticated HTTP, incremental events and verified downloads."""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import ipaddress
import json
import os
import re
import ssl
import stat
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit
from uuid import UUID, uuid4

from wavedl.core.credentials import credential_bytes
from wavedl.core.live_push import LivePushConfig
from wavedl.core.protocol import decode, strict_loads
from wavedl.core.safe_files import open_directory


class ClientTransportError(RuntimeError):
    """Public transport errors never contain credentials or response bodies."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ClientTransportError("Ambiguous daemon JSON")
        result[name] = value
    return result


def _object(raw):
    result = strict_loads(raw)
    json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(result, dict):
        raise ClientTransportError("Invalid daemon JSON")
    return result


def _cursor(value):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int for item in value)
        or not 0 <= value[0] <= 2**63 - 1
        or not -1 <= value[1] <= 2**63 - 1
        or (value[0] == 0 and value[1] != -1)
    ):
        raise ClientTransportError("Invalid event cursor")
    return tuple(value)


def _uuid(value):
    try:
        canonical = str(UUID(value)) if isinstance(value, str) else None
    except ValueError:
        canonical = None
    if canonical != value or canonical is None:
        raise ClientTransportError("Invalid run identity")
    return value


@dataclass(frozen=True)
class StreamControl:
    kind: str
    cursor: tuple[int, int]
    recoverable: bool = True


@dataclass(frozen=True)
class ArtifactRef:
    run_id: str
    generation: str
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        _uuid(self.run_id)
        if (
            not isinstance(self.generation, str)
            or not re.fullmatch(r"gen-\d{6,}", self.generation)
            or not isinstance(self.path, str)
            or any(c in self.path for c in "\\:%")
            or any(ord(c) < 32 or ord(c) == 127 for c in self.path)
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
            or type(self.size) is not int
            or self.size < 0
            or not isinstance(self.sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
        ):
            raise ClientTransportError("Invalid artifact reference")


class HttpTransport:
    def __init__(self, url, token, *, deadline=30):
        credential_bytes(token)
        self._config = LivePushConfig(url, token, deadline=deadline)
        self._daemon_id = None

    def bind_daemon_id(self, daemon_id):
        """Pin subsequent requests before a reused forwarding port can mutate."""
        if (
            type(daemon_id) is not str
            or not 1 <= len(daemon_id) <= 256
            or any(not 33 <= ord(char) <= 126 for char in daemon_id)
        ):
            raise ClientTransportError("Invalid daemon identity")
        if self._daemon_id is not None and self._daemon_id != daemon_id:
            raise ClientTransportError("Daemon identity changed")
        self._daemon_id = daemon_id

    def _target(self, path, query):
        if (
            type(path) is not str
            or not path.startswith("/api/")
            or len(path) > 8192
            or any(not 33 <= ord(c) <= 126 for c in path)
            or any(c in path for c in "?#\\")
            or any(part in {".", ".."} for part in path.split("/"))
        ):
            raise ClientTransportError("Invalid daemon request path")
        if query is not None:
            if type(query) is not dict or len(query) > 32:
                raise ClientTransportError("Invalid query parameters")
            for name, value in query.items():
                if (
                    not isinstance(name, str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
                    or type(value) not in {str, int}
                    or len(str(value)) > 4096
                    or any(ord(c) < 32 or ord(c) == 127 for c in str(value))
                ):
                    raise ClientTransportError("Invalid query parameters")
            path += "?" + urlencode(query)
        if len(path) > 16384:
            raise ClientTransportError("Request target exceeds bound")
        return path

    async def _close(self, writer):
        writer.close()
        try:
            async with asyncio.timeout(0.1):
                await writer.wait_closed()
        except (OSError, TimeoutError):
            pass

    async def _open(self, method, target, body, extra):
        parsed = urlsplit(self._config.url)
        host, port = (
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
        writer = None
        try:
            reader, writer = await asyncio.open_connection(
                host,
                port,
                ssl=ssl.create_default_context() if parsed.scheme == "https" else None,
                limit=16384,
            )
            if parsed.scheme == "http":
                peer = writer.get_extra_info("peername")
                if not peer or not ipaddress.ip_address(peer[0]).is_loopback:
                    raise ClientTransportError("Plain HTTP peer must be loopback")
            authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            headers = {
                "Host": authority,
                "Connection": "close",
                "Content-Length": str(len(body)),
                "Authorization": "Bearer " + self._config.token,
            } | extra
            if self._daemon_id is not None:
                headers["X-WaveDL-Daemon-ID"] = self._daemon_id
            writer.write(
                (
                    f"{method} {target} HTTP/1.1\r\n"
                    + "".join(f"{key}: {value}\r\n" for key, value in headers.items())
                    + "\r\n"
                ).encode("ascii")
                + body
            )
            await writer.drain()
            raw = await reader.readuntil(b"\r\n\r\n")
            if len(raw) > 16384:
                raise ClientTransportError("Response header exceeds bound")
            lines = raw.decode("ascii").split("\r\n")
            status = re.fullmatch(
                r"HTTP/1\.[01] ([1-5][0-9]{2})(?: [\x20-\x7e]*)?", lines[0]
            )
            if status is None:
                raise ClientTransportError("Invalid daemon HTTP status")
            code = int(status[1])
            if code not in {200, 202}:
                raise ClientTransportError("Daemon request refused", status_code=code)
            fields = {}
            for line in lines[1:-2]:
                name, separator, value = line.partition(":")
                if (
                    not separator
                    or not re.fullmatch(r"[A-Za-z0-9-]+", name)
                    or name.lower() in fields
                    or any(ord(c) < 32 and c != "\t" for c in value)
                ):
                    raise ClientTransportError("Invalid response headers")
                fields[name.lower()] = value.strip()
            length, transfer = (
                fields.get("content-length"),
                fields.get("transfer-encoding"),
            )
            if (
                (
                    length is not None
                    and (
                        not re.fullmatch(r"[0-9]{1,19}", length)
                        or int(length) > 2**63 - 1
                    )
                )
                or (transfer is not None and transfer.lower() != "chunked")
                or (length is not None and transfer is not None)
                or fields.get("content-encoding", "identity").lower() != "identity"
            ):
                raise ClientTransportError("Invalid HTTP framing")
            return reader, writer, fields
        except BaseException:
            if writer is not None:
                await self._close(writer)
            raise

    async def _body(self, reader, fields):
        async def io(coroutine):
            async with asyncio.timeout(self._config.deadline):
                return await coroutine

        if "content-length" in fields:
            remaining = int(fields["content-length"])
            while remaining:
                chunk = await io(reader.read(min(65536, remaining)))
                if not chunk:
                    raise ClientTransportError("Truncated daemon response")
                remaining -= len(chunk)
                yield chunk
        elif fields.get("transfer-encoding", "").lower() == "chunked":
            while True:
                line = await io(reader.readuntil(b"\r\n"))
                if not re.fullmatch(rb"[0-9a-fA-F]{1,16}\r\n", line):
                    raise ClientTransportError("Invalid chunk framing")
                remaining = int(line[:-2], 16)
                if remaining > 16 * 1024 * 1024:
                    raise ClientTransportError("HTTP chunk exceeds bound")
                if remaining == 0:
                    if await io(reader.readexactly(2)) != b"\r\n":
                        raise ClientTransportError("Unsupported HTTP trailers")
                    return
                while remaining:
                    chunk = await io(reader.read(min(65536, remaining)))
                    if not chunk:
                        raise ClientTransportError("Truncated HTTP chunk")
                    remaining -= len(chunk)
                    yield chunk
                if await io(reader.readexactly(2)) != b"\r\n":
                    raise ClientTransportError("Invalid HTTP chunk boundary")
        else:
            raise ClientTransportError("Unbounded daemon response framing")

    async def request(
        self,
        method,
        path,
        body=None,
        *,
        query=None,
        idempotency_key=None,
        last_event_id=None,
    ):
        if method not in {"GET", "POST"}:
            raise ClientTransportError("Invalid request method")
        headers = {"Accept": "application/json"}
        for name, value, bound in (
            ("Idempotency-Key", idempotency_key, 128),
            ("Last-Event-ID", last_event_id, 64),
        ):
            if value is not None:
                if (
                    type(value) is not str
                    or not 1 <= len(value) <= bound
                    or any(not 33 <= ord(c) <= 126 for c in value)
                ):
                    raise ClientTransportError("Invalid request header")
                headers[name] = value
        if last_event_id is not None:
            _parse_id(last_event_id)
        writer = None
        try:
            data = b"" if body is None else json.dumps(body, allow_nan=False).encode()
            if len(data) > 65536:
                raise ClientTransportError("Request exceeds bound")
            if body is not None:
                headers["Content-Type"] = "application/json"
            async with asyncio.timeout(self._config.deadline):
                reader, writer, fields = await self._open(
                    method, self._target(path, query), data, headers
                )
                if (
                    fields.get("content-type", "").split(";", 1)[0].lower()
                    != "application/json"
                ):
                    raise ClientTransportError("Expected daemon JSON")
                raw = bytearray()
                async for chunk in self._body(reader, fields):
                    if len(raw) + len(chunk) > 1048576:
                        raise ClientTransportError("Daemon JSON exceeds bound")
                    raw.extend(chunk)
                return _object(raw)
        except (
            OSError,
            ValueError,
            TypeError,
            RecursionError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            raise ClientTransportError(
                "Daemon request unavailable or invalid"
            ) from None
        finally:
            if writer is not None:
                await self._close(writer)

    async def events(self, run_id, *, cursor=(0, -1), last_event_id=None):
        writer = None
        try:
            _uuid(run_id)
            position = _cursor(cursor)
            headers = {"Accept": "text/event-stream"}
            if last_event_id is not None:
                position = _parse_id(last_event_id)
                headers["Last-Event-ID"] = last_event_id
                query = None
            else:
                query = {"attempt": position[0], "seq": position[1]}
            async with asyncio.timeout(self._config.deadline):
                reader, writer, fields = await self._open(
                    "GET",
                    self._target(f"/api/runs/{run_id}/events", query),
                    b"",
                    headers,
                )
            if (
                fields.get("content-type", "").split(";", 1)[0].lower()
                != "text/event-stream"
            ):
                raise ClientTransportError("Expected event stream")
            decoder, pending, frame, frame_bytes = (
                codecs.getincrementaldecoder("utf-8")("strict"),
                "",
                {},
                0,
            )
            async for chunk in self._body(reader, fields):
                pending += decoder.decode(chunk)
                if len(pending.encode()) > 262144:
                    raise ClientTransportError("Event frame exceeds bound")
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    line = line.removesuffix("\r")
                    frame_bytes += len(line.encode()) + 1
                    if frame_bytes > 262144:
                        raise ClientTransportError("Event frame exceeds bound")
                    if not line:
                        if "data" in frame:
                            kind = frame.get("event", "message")
                            data = "\n".join(frame["data"])
                            if kind in {"overflow", "error"}:
                                value = _object(data)
                                resume = value.get("resumePosition")
                                if not isinstance(resume, dict) or set(resume) != {
                                    "attempt",
                                    "seq",
                                }:
                                    raise ClientTransportError("Invalid event control")
                                advertised = _cursor((resume["attempt"], resume["seq"]))
                                if advertised > position:
                                    raise ClientTransportError(
                                        "Control cannot advance cursor"
                                    )
                                yield StreamControl(kind, advertised)
                            elif kind == "message":
                                envelope = decode(data)
                                _object(data)
                                if (
                                    envelope.run_id != run_id
                                    or frame.get("id")
                                    != f"{envelope.attempt}:{envelope.seq}"
                                ):
                                    raise ClientTransportError(
                                        "Event identity mismatch"
                                    )
                                next_position = _cursor(
                                    (envelope.attempt, envelope.seq)
                                )
                                if next_position > position:
                                    position = next_position
                                    yield envelope
                            else:
                                raise ClientTransportError("Unsupported event frame")
                        frame, frame_bytes = {}, 0
                    elif not line.startswith(":"):
                        name, separator, value = line.partition(":")
                        value = value.removeprefix(" ") if separator else ""
                        if name == "data":
                            frame.setdefault("data", []).append(value)
                        elif name in {"id", "event"} and name not in frame:
                            frame[name] = value
                        else:
                            raise ClientTransportError("Invalid event framing")
            pending += decoder.decode(b"", final=True)
            if pending or frame:
                raise ClientTransportError("Incomplete event frame")
        except (
            OSError,
            ValueError,
            TypeError,
            RecursionError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            raise ClientTransportError("Event stream unavailable or invalid") from None
        finally:
            if writer is not None:
                await self._close(writer)

    async def download(
        self, ref: ArtifactRef, destination: Path, *, max_bytes: int
    ) -> Path:
        writer, temporary, parent, temporary_identity = None, None, None, None
        try:
            if (
                not isinstance(ref, ArtifactRef)
                or type(max_bytes) is not int
                or not 0 <= ref.size <= max_bytes
            ):
                raise ClientTransportError("Download exceeds user budget")
            destination = Path(destination).absolute()
            name = destination.name
            if not name or name in {".", ".."} or any(c in name for c in "/\\:\0"):
                raise ClientTransportError("Invalid download destination member")
            parent = open_directory(destination.parent, writable=True)
            parent.verify_private_owner()
            parent.guard()
            try:
                if os.name == "posix":
                    os.stat(name, dir_fd=parent.handle, follow_symlinks=False)
                else:
                    (parent.path / name).lstat()
            except FileNotFoundError:
                pass
            else:
                raise ClientTransportError("Download destination already exists")
            target = f"/api/runs/{ref.run_id}/artifacts/{ref.generation}/" + quote(
                ref.path, safe="/"
            )
            async with asyncio.timeout(self._config.deadline):
                reader, writer, fields = await self._open(
                    "GET", target, b"", {"Accept": "application/octet-stream"}
                )
            if fields.get("content-type") != "application/octet-stream" or fields.get(
                "content-length"
            ) != str(ref.size):
                raise ClientTransportError("Invalid artifact response")
            parent.guard()
            candidate = ".wavedl-download-" + uuid4().hex
            if os.name == "posix":
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent.handle,
                )
            else:
                # Native directory ancestry remains held without write/delete
                # sharing, preventing pathname redirection on Windows.
                descriptor = os.open(
                    parent.path / candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
            temporary = candidate
            opened = os.fstat(descriptor)
            temporary_identity = opened.st_dev, opened.st_ino
            digest, size = hashlib.sha256(), 0
            with os.fdopen(descriptor, "wb") as output:
                async for chunk in self._body(reader, fields):
                    size += len(chunk)
                    if size > ref.size:
                        raise ClientTransportError("Artifact exceeds inventory")
                    parent.guard()
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
                identity = os.fstat(output.fileno())
            if size != ref.size or digest.hexdigest() != ref.sha256:
                raise ClientTransportError("Artifact integrity mismatch")
            # Hard-link publication is atomic and refuses destination replacement.
            parent.verify_private_owner()
            parent.guard()
            current = (
                os.stat(temporary, dir_fd=parent.handle, follow_symlinks=False)
                if os.name == "posix"
                else (parent.path / temporary).lstat()
            )
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev,
                current.st_ino,
            ) != (identity.st_dev, identity.st_ino):
                raise ClientTransportError("Download temporary file changed")
            if os.name == "posix":
                os.link(
                    temporary,
                    name,
                    src_dir_fd=parent.handle,
                    dst_dir_fd=parent.handle,
                    follow_symlinks=False,
                )
                parent.sync()
                parent.guard()
                os.unlink(temporary, dir_fd=parent.handle)
                parent.sync()
            else:
                parent.link_file(temporary, name)
                parent.sync()
                parent.guard()
                (parent.path / temporary).unlink()
                parent.sync()
            temporary = None
            parent.guard()
            return destination
        except (
            OSError,
            ValueError,
            TypeError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            raise ClientTransportError(
                "Artifact download unavailable or invalid"
            ) from None
        finally:
            try:
                if writer is not None:
                    await self._close(writer)
            finally:
                if temporary is not None:
                    try:
                        current = (
                            os.stat(
                                temporary, dir_fd=parent.handle, follow_symlinks=False
                            )
                            if os.name == "posix"
                            else (parent.path / temporary).lstat()
                        )
                        if (current.st_dev, current.st_ino) == temporary_identity:
                            if os.name == "posix":
                                os.unlink(temporary, dir_fd=parent.handle)
                            else:
                                (parent.path / temporary).unlink(missing_ok=True)
                    except FileNotFoundError:
                        pass
                if parent is not None:
                    parent.close()


def _parse_id(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{1,19}:-?[0-9]{1,19}", value
    ):
        raise ClientTransportError("Invalid Last-Event-ID")
    return _cursor(tuple(int(item) for item in value.split(":")))
