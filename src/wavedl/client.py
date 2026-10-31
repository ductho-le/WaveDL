"""Additive daemon client; standalone training never imports this module."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import ssl
import sys
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from wavedl.core.credentials import credential_bytes, read_credential
from wavedl.core.live_push import LivePushConfig
from wavedl.core.protocol import strict_loads
from wavedl.core.safe_files import read_regular_file


class ClientError(RuntimeError):
    """Secret-free public network/request failure."""


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ClientError("Ambiguous daemon JSON")
        result[key] = value
    return result


class DaemonClient:
    """Bounded header-authenticated requests, with no proxies or redirects."""

    def __init__(self, url, token, *, timeout=30):
        credential_bytes(token)
        self._config = LivePushConfig(url, token, deadline=timeout)

    async def _exchange(self, method, path, body, headers):
        parsed = urlsplit(self._config.url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
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
                    raise ClientError("Plain HTTP peer must be loopback")
            authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            body = body or b""
            request = (
                f"{method} {path} HTTP/1.1\r\nHost: {authority}\r\n"
                "Connection: close\r\n"
                f"Content-Length: {len(body)}\r\n"
                + "".join(f"{key}: {value}\r\n" for key, value in headers.items())
                + "\r\n"
            ).encode("ascii")
            writer.write(request + body)
            await writer.drain()
            header = await reader.readuntil(b"\r\n\r\n")
            if len(header) > 16384:
                raise ClientError("Response header exceeds byte bound")
            lines = header.decode("ascii").split("\r\n")
            version, status, *_ = lines[0].split(" ")
            if version not in {"HTTP/1.0", "HTTP/1.1"} or status not in {"200", "202"}:
                raise ClientError("Daemon request refused")
            fields = {}
            for line in lines[1:]:
                if line:
                    key, value = line.split(":", 1)
                    key = key.lower()
                    if key in fields:
                        raise ClientError("Ambiguous daemon response")
                    fields[key] = value.strip()
            length = fields.get("content-length", "")
            if (
                "transfer-encoding" in fields
                or not length.isdecimal()
                or not 1 <= int(length) <= 1048576
                or fields.get("content-type", "").split(";", 1)[0] != "application/json"
            ):
                raise ClientError("Invalid daemon response framing")
            return await reader.readexactly(int(length))
        finally:
            if writer is not None:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), 0.1)
                except (OSError, TimeoutError):
                    pass

    async def request_async(self, method, path, body=None, *, idempotency_key=None):
        """Absolute deadline includes DNS, TLS, headers and response body."""
        data, headers = self._request_parts(method, path, body, idempotency_key)
        try:
            raw = await asyncio.wait_for(
                self._exchange(method, path, data, headers), self._config.deadline
            )
            result = strict_loads(raw)
            # Check duplicate identity keys without relaxing the finite-number
            # validation already performed by the shared protocol decoder.
            json.loads(raw, object_pairs_hook=_unique_pairs)
            if not isinstance(result, dict):
                raise ClientError("Invalid daemon response")
            return result
        except (
            OSError,
            ValueError,
            RecursionError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            raise ClientError("Daemon request unavailable or invalid") from None

    def request(self, method, path, body=None, *, idempotency_key=None):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.request_async(method, path, body, idempotency_key=idempotency_key)
            )
        finally:
            # OS DNS resolution may finish after cancellation; do not falsely
            # make that noncooperative executor part of the caller IO deadline.
            loop.close()

    def _request_parts(self, method, path, body, idempotency_key):
        if (
            method not in {"GET", "POST"}
            or type(path) is not str
            or len(path) > 8192
            or not path.startswith("/api/")
            or any(not 33 <= ord(char) <= 126 for char in path)
            or any(char in path for char in "\r\n?#")
        ):
            raise ClientError("Invalid daemon request")
        headers = {
            "Authorization": f"Bearer {self._config.token}",
            "Accept": "application/json",
        }
        try:
            data = (
                None
                if body is None
                else json.dumps(body, allow_nan=False).encode("utf-8")
            )
            if data is not None:
                if len(data) > 65536:
                    raise ClientError("Request exceeds byte bound")
                headers["Content-Type"] = "application/json"
            if idempotency_key is not None:
                if (
                    not isinstance(idempotency_key, str)
                    or not 1 <= len(idempotency_key) <= 128
                    or any(not 33 <= ord(char) <= 126 for char in idempotency_key)
                ):
                    raise ClientError("Invalid idempotency key")
                headers["Idempotency-Key"] = idempotency_key
            return data, headers
        except (ValueError, TypeError, RecursionError):
            raise ClientError("Daemon request unavailable or invalid") from None

    def run(self, payload, idempotency_key, *, submit=True):
        health = self.request("GET", "/api/health")
        low, high = health.get("protocol_min"), health.get("protocol_max")
        if type(low) is not int or type(high) is not int or not low <= 1 <= high:
            raise ClientError("Incompatible daemon protocol")
        preview = self.request(
            "POST", "/api/runs/plan", payload, idempotency_key=idempotency_key
        )
        try:
            plan_id = preview["plan_id"]
            if not isinstance(plan_id, str) or str(UUID(plan_id)) != plan_id:
                raise ValueError
        except (KeyError, ValueError):
            raise ClientError("Invalid daemon plan response") from None
        return (
            self.request("POST", "/api/runs", {"plan_id": plan_id})
            if submit
            else preview
        )


def build_parser():
    parser = argparse.ArgumentParser(prog="wavedl")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Start WaveDL and open it in your browser")
    run = commands.add_parser("run", help="Plan and submit an explicit daemon request")
    run.add_argument("--daemon", required=True)
    run.add_argument(
        "--request", type=Path, required=True, help="JSON lifecycle plan request"
    )
    run.add_argument(
        "--idempotency-key",
        required=True,
        help="Reuse this key when retrying the same request",
    )
    run.add_argument("--credential-file", type=Path)
    run.add_argument("--timeout", type=float, default=30)
    run.add_argument("--plan-only", action="store_true")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        from wavedl_daemon.cli import main as serve_main

        return serve_main(argv[1:])
    options = build_parser().parse_args(argv)
    try:
        token = (
            read_credential(options.credential_file)
            if options.credential_file
            else os.environ.get("WAVEDL_DAEMON_TOKEN")
        )
        credential_bytes(token)
        raw = read_regular_file(options.request, limit=65536)
        payload = strict_loads(raw)
        # Refuse duplicate request keys before re-encoding changes their meaning.
        json.loads(raw, object_pairs_hook=_unique_pairs)
        if not isinstance(payload, dict):
            raise ClientError("Request must be a JSON object")
        result = DaemonClient(options.daemon, token, timeout=options.timeout).run(
            payload, options.idempotency_key, submit=not options.plan_only
        )
        print(json.dumps(result, allow_nan=False))
    except (ClientError, OSError, ValueError, RecursionError):
        print(
            "Daemon request failed; reuse the same idempotency key when retrying.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
