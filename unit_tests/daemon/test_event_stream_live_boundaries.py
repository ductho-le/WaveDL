"""Real authenticated ASGI routes over fragmented loopback HTTP, not API mocks.

The adapter below is test-owned HTTP framing, not a production ASGI server. SQLite,
ingestion, authentication and replay/live delivery are real; runtime negotiation is
deliberately the small accepted-controller test seam.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from wavedl.core.protocol import Envelope, decode
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.events.ingestion import Ingestor
from wavedl_daemon.http.app import SecurityBoundary
from wavedl_daemon.http.events import event_routes
from wavedl_daemon.runs.negotiation import NegotiationStatus
from wavedl_daemon.storage.event_log import EventLogStore
from wavedl_daemon.storage.registry import Registry


RUN = "00000000-0000-4000-8000-000000000001"
TOKEN = "event-boundary-test-private-token"


def event(seq, attempt=1):
    return Envelope(
        RUN,
        attempt,
        "log",
        seq,
        "2026-10-11T00:00:00Z",
        {"message": "Tiếng Việt 🌊", "level": "info", "stream": "stdout"},
    )


class AcceptedController:
    def accept_runtime_events(self, plan_id, events):
        return SimpleNamespace(status=NegotiationStatus.ACCEPTED, reason="")


@dataclass(frozen=True)
class StreamControl:
    kind: str
    cursor: tuple[int, int]


async def read_events(url, token, run_id, *, cursor=(0, -1), last_event_id=None):
    """Yield daemon SSE frames as envelopes or overflow/error controls."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
    params = None
    if last_event_id is None:
        params = {"attempt": cursor[0], "seq": cursor[1]}
    else:
        headers["Last-Event-ID"] = last_event_id
    # Deadlines belong to wire_phase; an httpx timeout would race it.
    async with (
        httpx.AsyncClient(timeout=None) as client,
        client.stream(
            "GET", f"{url}/api/runs/{run_id}/events", params=params, headers=headers
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frame = {}
        async for line in response.aiter_lines():
            if line:
                name, _, value = line.partition(":")
                if name:
                    frame.setdefault(name, []).append(value.removeprefix(" "))
                continue
            if "data" in frame:
                data = "\n".join(frame["data"])
                kind = frame.get("event", ["message"])[0]
                if kind == "message":
                    yield decode(data)
                else:
                    resume = json.loads(data)["resumePosition"]
                    yield StreamControl(kind, (resume["attempt"], resume["seq"]))
            frame = {}


@asynccontextmanager
async def loopback_app(app, *, on_body=None, cleanup_timeout=5):
    """Bridge real ASGI into byte-sized HTTP chunks, including split CRLF/UTF-8."""
    tasks, requests, failures = set(), [], []

    async def accepted(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        disconnected = asyncio.Event()
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            lines = raw.split(b"\r\n")
            method, target, _ = lines[0].split(b" ")
            path, _, query = target.partition(b"?")
            headers = [
                (name.lower(), value.strip())
                for name, value in (line.split(b":", 1) for line in lines[1:] if line)
            ]
            requests.append((target, dict(headers)))
            size = int(dict(headers).get(b"content-length", b"0"))
            body = await reader.readexactly(size)
            initial = True

            async def receive():
                nonlocal initial
                if initial:
                    initial = False
                    return {"type": "http.request", "body": body, "more_body": False}
                await disconnected.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    writer.write(
                        f"HTTP/1.1 {message['status']} Test\r\n".encode()
                        + b"".join(
                            name + b": " + value + b"\r\n"
                            for name, value in message["headers"]
                            if name != b"content-length"
                        )
                        + b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                    )
                elif message["type"] == "http.response.body":
                    payload = message.get("body", b"")
                    # Each byte is an independent HTTP chunk, so even coalesced
                    # TCP packets cannot conceal parser boundary regressions.
                    for value in payload.replace(b"\n", b"\r\n"):
                        writer.write(b"1\r\n" + bytes((value,)) + b"\r\n")
                    await writer.drain()
                    if on_body is not None:
                        on_body(payload, disconnected)
                await writer.drain()

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.0"},
                "http_version": "1.1",
                "method": method.decode(),
                "scheme": "http",
                "path": path.decode(),
                "raw_path": path,
                "query_string": query,
                "root_path": "",
                "headers": headers,
                "server": writer.get_extra_info("sockname"),
                "client": writer.get_extra_info("peername"),
            }
            await app(scope, receive, send)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as error:
            failures.append(error)
        finally:
            disconnected.set()
            writer.close()
            try:
                with suppress(ConnectionResetError, BrokenPipeError):
                    await writer.wait_closed()
            finally:
                tasks.discard(task)

    listener = await asyncio.start_server(accepted, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:

        async def cleanup():
            listener.close()
            await listener.wait_closed()
            pending = tuple(tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        # Even an already-expired wire deadline must not leave teardown with an
        # unbounded await. Normal teardown also remains inside that wire budget.
        await asyncio.wait_for(cleanup(), timeout=cleanup_timeout)
        assert not tasks, "Loopback request tasks leaked"
        assert not failures, failures


@asynccontextmanager
async def wire_phase(app, *, timeout=5, **options):
    """Bound real listener startup, wire exchange and cleanup, not DB setup."""
    async with asyncio.timeout(timeout):
        async with loopback_app(app, cleanup_timeout=timeout, **options) as connection:
            yield connection


def authenticated_app(store, broker):
    app = FastAPI(
        routes=event_routes(
            event_log=store,
            broker=broker,
            resolve_plan=lambda run: "test-owned-plan",
        )
    )
    return SecurityBoundary(app, token=TOKEN.encode(), allowed_hosts=("127.0.0.1",))


def test_restart_and_sparse_attempt_replay(tmp_path):
    path = tmp_path / "registry.db"

    async def scenario():
        with Registry(path) as registry:
            store, broker = EventLogStore(path), EventBroker()
            try:
                ingestor = Ingestor(registry, AcceptedController(), broker, None)
                # Journal-origin ingestion legitimately contains sparse seqs.
                ingestor.push(
                    "test-owned-plan", RUN, 1, [event(0), event(80)], journal=True
                )
                ingestor.push(
                    "test-owned-plan", RUN, 2, [event(0, 2), event(90, 2)], journal=True
                )
                ingestor.push("test-owned-plan", RUN, 10, [event(0, 10)], journal=True)
                async with (
                    wire_phase(authenticated_app(store, broker)) as (url, _),
                    httpx.AsyncClient() as client,
                ):
                    refused = await client.get(
                        f"{url}/api/runs/{RUN}/events",
                        headers={"Authorization": "Bearer wrong-token"},
                    )
                    assert refused.status_code == 401
            finally:
                store.close()

        # Reopen both connections: neither replay nor cursors rely on memory.
        with Registry(path) as registry:
            store, broker = EventLogStore(path), EventBroker()
            try:
                assert registry.cursor(RUN, 1) == 80
                assert registry.cursor(RUN, 2) == 90
                assert registry.cursor(RUN, 10) == 0

                def finish(payload, disconnected):
                    if b"id: 10:0" in payload:
                        disconnected.set()

                app = authenticated_app(store, broker)
                async with wire_phase(app, on_body=finish) as (url, requests):
                    received = [
                        item
                        async for item in read_events(
                            url, TOKEN, RUN, last_event_id="1:0"
                        )
                    ]
                    assert received == [
                        event(80),
                        event(0, 2),
                        event(90, 2),
                        event(0, 10),
                    ]
                    assert requests[0][1][b"last-event-id"] == b"1:0"
                    assert b"?" not in requests[0][0]
                assert not broker._subscribers
            finally:
                store.close()

    asyncio.run(scenario())


def test_real_overflow_does_not_advance(tmp_path):
    async def scenario():
        with Registry(tmp_path / "registry.db") as registry:
            registry.apply_events(RUN, 1, [event(0)])
            store, broker = EventLogStore(registry.path), EventBroker(queue_size=1)
            try:

                def overflow_after_delivery(payload, disconnected):
                    if b"id: 1:0" in payload:
                        # Commit while the real stream is delivering its first
                        # page, then overflow its bounded wakeup subscription.
                        registry.apply_events(RUN, 1, [event(1)])
                        broker.notify(RUN)
                        broker.notify(RUN)

                app = authenticated_app(store, broker)
                delivered, committed_cursor = [], (0, -1)
                async with wire_phase(app, on_body=overflow_after_delivery) as (url, _):

                    async def consume():
                        nonlocal committed_cursor
                        async for item in read_events(url, TOKEN, RUN):
                            delivered.append(item)
                            if isinstance(item, Envelope):
                                committed_cursor = item.attempt, item.seq

                    await consume()
                    assert delivered == [event(0), StreamControl("overflow", (1, 0))]
                assert committed_cursor == (1, 0)
                assert registry.cursor(RUN, 1) == 1
                assert store.read_after(RUN, committed_cursor) == [event(1)]
                assert not broker._subscribers

                # Reconnect over the real authenticated route using only the
                # last delivered data event, not a forged control's cursor.
                # The committed event hidden by overflow must arrive once;
                # the already-delivered event must not be replayed.
                def finish_reconnect(payload, disconnected):
                    if b"id: 1:1" in payload:
                        disconnected.set()

                async with wire_phase(app, on_body=finish_reconnect) as (
                    url,
                    requests,
                ):
                    replayed = [
                        item
                        async for item in read_events(
                            url,
                            TOKEN,
                            RUN,
                            last_event_id=(
                                f"{committed_cursor[0]}:{committed_cursor[1]}"
                            ),
                        )
                    ]
                    assert replayed == [event(1)]
                    assert requests[0][1][b"last-event-id"] == b"1:0"
                assert registry.cursor(RUN, 1) == 1
                assert registry.get(RUN).event_count == 2
                assert store.read_after(RUN) == [event(0), event(1)]
                assert not broker._subscribers
            finally:
                store.close()

    asyncio.run(scenario())


def test_durable_setup_does_not_consume_wire_phase_deadline(tmp_path):
    async def scenario():
        with Registry(tmp_path / "registry.db") as registry:
            store, broker = EventLogStore(registry.path), EventBroker()
            try:
                # Stand in for slow synchronous Windows migrations/fsync. The
                # original whole-scenario deadline expired at the next await.
                time.sleep(5.1)
                app = authenticated_app(store, broker)
                async with wire_phase(app):
                    pass
                assert not broker._subscribers
            finally:
                store.close()

    asyncio.run(scenario())


def test_stalled_wire_phase_times_out_and_reaps_real_stream(tmp_path):
    async def scenario():
        with Registry(tmp_path / "registry.db") as registry:
            registry.apply_events(RUN, 1, [event(0)])
            store, broker = EventLogStore(registry.path), EventBroker()
            delivered = []
            try:
                app = authenticated_app(store, broker)
                with pytest.raises(TimeoutError):
                    async with wire_phase(app) as (url, _):
                        async for item in read_events(url, TOKEN, RUN):
                            delivered.append(item)
                            # No finish callback: a real subscriber waits for
                            # another event until the wire deadline cancels it.
                assert delivered == [event(0)]
                assert not broker._subscribers
                assert not [
                    task
                    for task in asyncio.all_tasks()
                    if task is not asyncio.current_task() and not task.done()
                ]
            finally:
                store.close()

    asyncio.run(scenario())
