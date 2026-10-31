"""Real authenticated ASGI routes over fragmented loopback HTTP, not API mocks.

The adapter below is test-owned HTTP framing, not a production ASGI server. SQLite,
ingestion, authentication, replay/live delivery and the native client are real;
runtime negotiation is deliberately the small accepted-controller test seam.
"""

import asyncio
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from wavedl.core.client_transport import (
    ClientTransportError,
    HttpTransport,
    StreamControl,
)
from wavedl.core.protocol import Envelope
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


@asynccontextmanager
async def loopback_app(app, *, on_body=None, corrupt_control=None, cleanup_timeout=5):
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
                    if corrupt_control is not None and b"event: overflow" in payload:
                        payload = corrupt_control
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


def authenticated_app(ingestor, store, broker):
    app = FastAPI(
        routes=event_routes(
            ingestor=ingestor,
            event_log=store,
            broker=broker,
            resolve_plan=lambda run: "test-owned-plan",
        )
    )
    return SecurityBoundary(app, token=TOKEN.encode(), allowed_hosts=("127.0.0.1",))


def test_authenticated_commit_reject_restart_and_sparse_attempt_replay(tmp_path):
    path = tmp_path / "registry.db"

    async def scenario():
        with Registry(path) as registry:
            store, broker = EventLogStore(path), EventBroker()
            try:
                ingestor = Ingestor(registry, AcceptedController(), broker, None)
                app = authenticated_app(ingestor, store, broker)
                async with wire_phase(app) as (url, _):
                    client = HttpTransport(url, TOKEN)
                    route = f"/api/runs/{RUN}/events"
                    body = {"attempt": 1, "events": [asdict(event(0))]}
                    with pytest.raises(ClientTransportError) as refused:
                        await HttpTransport(url, "wrong-token").request(
                            "POST", route, body=body
                        )
                    assert refused.value.status_code == 401
                    assert registry.cursor(RUN, 1) == -1
                    assert store.read_after(RUN) == []
                    assert await client.request("POST", route, body=body) == {
                        "attempt": 1,
                        "seq": 0,
                    }
                    # A valid but gapped live push must not commit a cursor/row.
                    with pytest.raises(ClientTransportError) as rejected:
                        await client.request(
                            "POST",
                            route,
                            body={"attempt": 1, "events": [asdict(event(80))]},
                        )
                    assert rejected.value.status_code == 409
                    assert registry.cursor(RUN, 1) == 0
                    assert store.read_after(RUN) == [event(0)]
                    malformed = asdict(event(1))
                    malformed["seq"] = True
                    with pytest.raises(ClientTransportError) as malformed_rejection:
                        await client.request(
                            "POST",
                            route,
                            body={"attempt": 1, "events": [malformed]},
                        )
                    assert malformed_rejection.value.status_code == 409
                    assert registry.cursor(RUN, 1) == 0
                    assert registry.get(RUN).event_count == 1
                    assert store.read_after(RUN) == [event(0)]
                # Journal-origin ingestion legitimately contains sparse seqs.
                ingestor.push("test-owned-plan", RUN, 1, [event(80)], journal=True)
                ingestor.push(
                    "test-owned-plan", RUN, 2, [event(0, 2), event(90, 2)], journal=True
                )
                ingestor.push("test-owned-plan", RUN, 10, [event(0, 10)], journal=True)
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

                app = authenticated_app(None, store, broker)
                async with wire_phase(app, on_body=finish) as (url, requests):
                    received = [
                        item
                        async for item in HttpTransport(url, TOKEN).events(
                            RUN, last_event_id="1:0"
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


@pytest.mark.parametrize(
    "corrupt_control",
    [
        None,
        b'event: overflow\ndata: {"resumePosition":{"attempt":2,"seq":0}}\n\n',
        b'event: overflow\ndata: {"resumePosition":{"attempt":true,"seq":0}}\n\n',
        b"event: overflow\ndata: not-json\n\n",
    ],
)
def test_real_overflow_and_corrupt_controls_do_not_advance(tmp_path, corrupt_control):
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

                app = authenticated_app(None, store, broker)
                delivered, committed_cursor = [], (0, -1)
                async with wire_phase(
                    app,
                    on_body=overflow_after_delivery,
                    corrupt_control=corrupt_control,
                ) as (url, _):

                    async def consume():
                        nonlocal committed_cursor
                        async for item in HttpTransport(url, TOKEN).events(RUN):
                            delivered.append(item)
                            if isinstance(item, Envelope):
                                committed_cursor = item.attempt, item.seq

                    if corrupt_control is None:
                        await consume()
                        assert delivered == [
                            event(0),
                            StreamControl("overflow", (1, 0)),
                        ]
                    else:
                        with pytest.raises(ClientTransportError):
                            await consume()
                        assert delivered == [event(0)]
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
                        async for item in HttpTransport(url, TOKEN).events(
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
                app = authenticated_app(None, store, broker)
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
                app = authenticated_app(None, store, broker)
                with pytest.raises(TimeoutError):
                    async with wire_phase(app) as (url, _):
                        async for item in HttpTransport(url, TOKEN).events(RUN):
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
