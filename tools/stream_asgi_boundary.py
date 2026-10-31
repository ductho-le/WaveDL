"""Test-owned observation of real event handlers, never browser/local-state proof.

Mount around a real ASGI application (including the CPU browser daemon fixture).
Only numeric identities are retained: no credentials, log payloads or raw queries.
Snapshots are taken after the handler exits, not from a fabricated HTTP response.
The resulting evidence cannot establish that a client parsed or committed bytes.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl


_ROUTE = re.compile(r"^/api/runs/([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})/events$")
_ID = re.compile(rb"^([0-9]{1,19}):([0-9]{1,19})$")


def query_cursor(raw: bytes):
    """Record a complete unambiguous query cursor, or no cursor at all."""
    if len(raw) > 4096:
        return None
    try:
        pairs = parse_qsl(raw.decode("ascii"), max_num_fields=16)
        fields = {key: value for key, value in pairs if key in {"attempt", "seq"}}
        if len(fields) != 2 or sum(key in fields for key, _ in pairs) != 2:
            return None
        if not all(re.fullmatch(r"-?[0-9]{1,19}", value) for value in fields.values()):
            return None
        cursor = {key: int(value) for key, value in fields.items()}
        if (
            cursor["attempt"] < 0
            or cursor["seq"] < -1
            or cursor["attempt"] > 2**63 - 1
            or cursor["seq"] > 2**63 - 1
            or (cursor["attempt"] == 0 and cursor["seq"] != -1)
        ):
            return None
        return cursor
    except (UnicodeDecodeError, ValueError):
        return None


class StreamBoundary:
    """Observe forwarded bytes and independently query the real registry.

    The wrapper never modifies a request, response or broker. A disconnected
    request is recorded too, with handlerReturned=False if cancellation escapes.
    Evidence budgets fail closed instead of silently truncating a success claim.
    """

    def __init__(self, app, registry, *, max_requests=64, max_frames=4096):
        self.app, self.registry = app, registry
        self.max_requests, self.max_frames = max_requests, max_frames
        self.records = []

    async def __call__(self, scope, receive, send):
        match = _ROUTE.fullmatch(scope.get("path", ""))
        if scope.get("type") != "http" or scope.get("method") != "GET" or not match:
            await self.app(scope, receive, send)
            return
        if len(self.records) >= self.max_requests:
            raise RuntimeError("Stream boundary request evidence exceeds budget")
        run_id = match.group(1)
        record = {
            "kind": "asgi-event-boundary-v1",
            "runId": run_id,
            "queryCursor": query_cursor(scope.get("query_string", b"")),
            "status": None,
            "forwardedEventIds": [],
            "forwardedControls": [],
            "handlerReturned": False,
            "browserObserved": False,
            "localCursor": None,
            "registry": None,
        }
        self.records.append(record)
        pending = b""

        async def observed_send(message):
            nonlocal pending
            # Successful send is boundary delivery, not client acknowledgment.
            await send(message)
            if message["type"] == "http.response.start":
                record["status"] = message["status"]
            if message["type"] != "http.response.body" or record["status"] != 200:
                return
            pending += message.get("body", b"")
            if len(pending) > 262144:
                raise RuntimeError("Stream boundary frame evidence exceeds budget")
            pending = pending.replace(b"\r\n", b"\n")
            while b"\n\n" in pending:
                frame, pending = pending.split(b"\n\n", 1)
                fields = {}
                ambiguous = False
                for line in frame.split(b"\n"):
                    name, _, value = line.partition(b":")
                    if name in {b"id", b"event"}:
                        if name in fields:
                            ambiguous = True
                        fields[name] = value.removeprefix(b" ")
                if ambiguous:
                    raise RuntimeError("Ambiguous stream boundary identity fields")
                identity = _ID.fullmatch(fields.get(b"id", b""))
                if identity:
                    if (
                        not 1 <= int(identity[1]) <= 2**63 - 1
                        or int(identity[2]) > 2**63 - 1
                    ):
                        raise RuntimeError("Invalid event attempt identity")
                    record["forwardedEventIds"].append(
                        [int(identity[1]), int(identity[2])]
                    )
                kind = fields.get(b"event")
                if kind in {b"overflow", b"error"}:
                    record["forwardedControls"].append(kind.decode("ascii"))
                if (
                    len(record["forwardedEventIds"]) + len(record["forwardedControls"])
                    > self.max_frames
                ):
                    raise RuntimeError("Stream boundary evidence exceeds frame budget")

        try:
            await self.app(scope, receive, observed_send)
            record["handlerReturned"] = True
        finally:
            attempts = {identity[0] for identity in record["forwardedEventIds"]}
            if record["queryCursor"] is not None:
                attempts.add(record["queryCursor"]["attempt"])
            view = self.registry.get(run_id)
            record["registry"] = {
                "eventCount": view.event_count if view is not None else 0,
                "cursors": {
                    str(attempt): self.registry.cursor(run_id, attempt)
                    for attempt in sorted(attempts)
                },
            }


def verify_boundary_record(record):
    """Fail closed on mislabeled evidence and IDs beyond registry authority.

    This verifies ONLY boundary evidence, not browser rendering, deduplication,
    reconnect behavior, or the client's unobservable local cursor.
    """
    if (
        record.get("kind") != "asgi-event-boundary-v1"
        or record.get("browserObserved") is not False
        or record.get("localCursor") is not None
        or record.get("handlerReturned") is not True
        or record.get("status") != 200
    ):
        raise ValueError("Not completed ASGI-only stream evidence")
    registry = record.get("registry")
    if not isinstance(registry, dict) or type(registry.get("eventCount")) is not int:
        raise ValueError("Registry evidence missing")
    identities = record.get("forwardedEventIds")
    if not isinstance(identities, list):
        raise ValueError("Event identity evidence missing")
    for identity in identities:
        if (
            not isinstance(identity, list)
            or len(identity) != 2
            or any(type(value) is not int or value < 0 for value in identity)
            or identity[0] < 1
            or any(value > 2**63 - 1 for value in identity)
        ):
            raise ValueError("Invalid boundary event identity")
        committed = registry.get("cursors", {}).get(str(identity[0]))
        if type(committed) is not int or committed < identity[1]:
            raise ValueError("Forwarded identity exceeds registry authority")
    return {"scope": "asgi-only", "forwardedEvents": len(identities)}
