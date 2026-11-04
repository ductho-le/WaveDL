"""Executable durability, handover and bounded-delivery failure contracts."""

import asyncio
import json
import sqlite3
import threading

import httpx
import pytest
from starlette.requests import ClientDisconnect, Request

from unit_tests.daemon.conftest import route_app
from unit_tests.daemon.test_event_stream_live_boundaries import (
    RUN,
    metrics,
    open_log,
    requeue,
    start,
)
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.events.streaming import stream_events
from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.http.events import event_routes
from wavedl_daemon.runs.model import Metric, RunId, canonical_json
from wavedl_daemon.storage.run_log import MAX_PAGE_BYTES, Expect, encode


RUN_ID = RUN


@pytest.fixture
def log(tmp_path):
    with open_log(tmp_path) as log:
        yield log


def positions(events):
    return [(item.attempt, item.seq) for item in events]


def test_frame_is_id_and_versioned_envelope(log):
    start(log)
    metrics(log, 1)

    async def scenario():
        stream = stream_events(log, EventBroker(), RUN_ID, (1, 2))
        frame = await anext(stream)
        await stream.aclose()
        return frame

    frame = asyncio.run(scenario())
    head, data = frame.split(b"\ndata: ")
    assert head == b"id: 1:3"
    assert data.endswith(b"}\n\n") and not data.endswith(b"\n\n\n")
    envelope = json.loads(data)
    assert envelope["protocol"] == "wavedl-run" and envelope["version"] == 2
    assert (envelope["run_id"], envelope["attempt"], envelope["seq"]) == (RUN_ID, 1, 3)
    assert envelope["type"] == "metric"
    assert envelope["payload"]["name"] == "epoch" and envelope["payload"]["step"] == 1
    assert "source_end" not in data.decode("utf-8")
    assert "Tiếng Việt 🌊" in data.decode("utf-8")


def test_cursor_orders_by_attempt_then_seq_across_requeues(log):
    start(log)
    requeue(log)
    metrics(log, 1)
    requeue(log)
    assert positions(log.read_after(RunId(RUN_ID), (1, 2))) == [
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
    ]
    assert positions(log.read_after(RunId(RUN_ID), (2, -1), limit=1)) == [(2, 0)]
    assert log.read_after(RunId(RUN_ID), (3, 0)) == []


def test_commit_without_notification_is_delivered(log):
    async def scenario():
        broker = EventBroker()
        stream = stream_events(log, broker, RUN_ID, poll_interval=0.01)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        start(log)
        assert await pending == b": heartbeat\n\n"
        assert (await anext(stream)).startswith(b"id: 1:0\n")
        await stream.aclose()
        assert len(broker._subscribers) == 0

    asyncio.run(scenario())


def test_register_before_replay_closes_handover_race(log):
    broker = EventBroker()

    class RacingStore:
        raced = False

        def read_after(self, *args, **kwargs):
            page = log.read_after(*args, **kwargs)
            if not self.raced:
                self.raced = True
                assert len(broker._subscribers) == 1
                start(log)
                broker.notify(RUN_ID)
            return page

    async def scenario():
        stream = stream_events(RacingStore(), broker, RUN_ID, poll_interval=0.01)
        assert await anext(stream) == b": heartbeat\n\n"
        frames = [await anext(stream) for _ in range(3)]
        assert [frame.split(b"\n")[0] for frame in frames] == [
            b"id: 1:0",
            b"id: 1:1",
            b"id: 1:2",
        ]
        await stream.aclose()

    asyncio.run(scenario())


def test_replay_then_live_handoff_has_no_gap_or_duplicate(log):
    start(log)

    async def scenario():
        broker = EventBroker()
        stream = stream_events(log, broker, RUN_ID, page_size=2, poll_interval=60)
        received = [await anext(stream) for _ in range(3)]
        for epoch in (1, 2, 3):
            metrics(log, epoch)
            broker.notify(RUN_ID)
            received.append(await anext(stream))
        await stream.aclose()
        return [frame.split(b"\n")[0] for frame in received]

    assert asyncio.run(scenario()) == [f"id: 1:{seq}".encode() for seq in range(6)]


def test_overflow_disconnects_with_last_delivered_position(log):
    start(log)

    async def scenario():
        broker = EventBroker(queue_size=1)
        stream = stream_events(log, broker, RUN_ID, (1, 1), page_size=1)
        first = await anext(stream)
        assert first.startswith(b"id: 1:2\n")
        metrics(log, 1, 2)
        broker.notify(RUN_ID)
        broker.notify(RUN_ID)
        overflow = await anext(stream)
        assert overflow.startswith(b"event: overflow\n")
        assert b'"resumePosition":{"attempt":1,"seq":2}' in overflow
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert len(broker._subscribers) == 0
        assert positions(log.read_after(RunId(RUN_ID), (1, 2))) == [(1, 3), (1, 4)]

    asyncio.run(scenario())


def test_replay_pages_are_byte_bounded_but_never_empty(log):
    start(log)
    wide = {f"metric_{index:05d}": 1.0 for index in range(5000)}
    head = log.head(RunId(RUN_ID))
    rows = [
        Metric(epoch, {**wide, "epoch": epoch}, head.metrics_offset + epoch)
        for epoch in range(1, 31)
    ]
    log.append(RunId(RUN_ID), rows, expect=Expect())
    page = log.read_after(RunId(RUN_ID), (1, 2), limit=1024)
    assert 0 < len(page) < len(rows)
    assert (
        sum(len(canonical_json(item.event.payload())) for item in page)
        <= MAX_PAGE_BYTES
    )
    single = log.read_after(RunId(RUN_ID), (1, 2), limit=1)
    assert positions(single) == [(1, 3)]
    assert len(encode(single[0])) > 0


def test_async_owner_adapter_replay(log):
    start(log)

    class AsyncStore:
        async def read_after(self, run_id, cursor, *, limit):
            await asyncio.sleep(0)
            return log.read_after(run_id, cursor, limit=limit)

    async def scenario():
        stream = stream_events(AsyncStore(), EventBroker(), RUN_ID)
        assert (await anext(stream)).startswith(b"id: 1:0\n")
        await stream.aclose()

    asyncio.run(scenario())


def test_disconnected_stream_cancels_async_wait(log):
    store = log

    async def scenario():
        broker = EventBroker()
        stream = stream_events(store, broker, RUN_ID, poll_interval=60)
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert broker._subscribers
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not broker._subscribers

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "query",
    [
        "attempt=1",
        "seq=0",
        "attempt=nan&seq=0",
        "attempt=1&seq=inf",
        "attempt=1&seq=-2",
        "attempt=1&attempt=2&seq=0",
        "attempt=0&seq=0",
        "attempt=9223372036854775808&seq=0",
    ],
)
def test_http_rejects_invalid_or_partial_cursors(query):
    routes = event_routes(event_log=None, broker=None, resolve_plan=lambda run: "plan")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/api/runs/{RUN_ID}/events?{query}")
            assert response.status_code == 400

    asyncio.run(scenario())


def test_http_rejects_noncanonical_run_identity_without_resolving():
    called = False

    def resolve(_run):
        nonlocal called
        called = True
        raise AssertionError("noncanonical identity must be rejected at the route")

    routes = event_routes(event_log=None, broker=None, resolve_plan=resolve)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/api/runs/DEADBEEF-0000-4000-8000-000000000001/events"
            )
            assert response.status_code == 400

    asyncio.run(scenario())
    assert not called


def test_cross_thread_wakeups_coalesce_bounded_callbacks_and_preserve_overflow():
    async def scenario():
        broker = EventBroker(queue_size=2)
        subscription = broker.subscribe(RUN_ID)
        scheduled = []
        original_schedule = subscription.loop.call_soon_threadsafe

        def schedule(*args):
            scheduled.append(args)
            return original_schedule(*args)

        subscription.loop.call_soon_threadsafe = schedule
        try:

            def notify():
                for _ in range(1000):
                    broker.notify(RUN_ID)

            thread = threading.Thread(target=notify)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert len(scheduled) == 1
            assert subscription.pending == 3
            await asyncio.sleep(0)
            assert subscription.closed and subscription.overflowed
            assert subscription.queue.qsize() <= 2
        finally:
            subscription.loop.call_soon_threadsafe = original_schedule

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", [BudgetExhausted, StorageWorkError, sqlite3.OperationalError]
)
def test_event_http_resolver_storage_failures_are_redacted_503(failure):
    async def resolver(run_id):
        raise failure("secret-token private-path")

    routes = event_routes(event_log=None, broker=EventBroker(), resolve_plan=resolver)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/api/runs/{RUN_ID}/events")
            assert response.status_code == 503
            assert "secret-token" not in response.text
            assert "private-path" not in response.text

    asyncio.run(scenario())


def test_event_http_full_broker_rejects_before_stream_headers():
    async def scenario():
        broker = EventBroker(max_subscribers=1)
        occupied = broker.subscribe(RUN_ID)
        routes = event_routes(
            event_log=None,
            broker=broker,
            resolve_plan=lambda run: "plan",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/api/runs/{RUN_ID}/events")
            assert response.status_code == 503
            assert (
                response.headers.get("content-type")
                != "text/event-stream; charset=utf-8"
            )
        assert broker._subscribers == {occupied}
        broker.unsubscribe(occupied)

    asyncio.run(scenario())


def test_event_http_first_read_failure_releases_admitted_subscriber():
    class FailedStore:
        async def read_after(self, *args, **kwargs):
            raise BudgetExhausted("private-path")

    async def scenario():
        broker = EventBroker()
        routes = event_routes(
            event_log=FailedStore(),
            broker=broker,
            resolve_plan=lambda run: "plan",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/api/runs/{RUN_ID}/events")
            assert response.status_code == 503
            assert "private-path" not in response.text
        assert not broker._subscribers

    asyncio.run(scenario())


def test_event_http_cancelled_initial_query_releases_admitted_subscriber():
    async def scenario():
        started = asyncio.Event()

        class SlowStore:
            async def read_after(self, *args, **kwargs):
                started.set()
                await asyncio.Event().wait()

        broker = EventBroker()
        routes = event_routes(
            event_log=SlowStore(),
            broker=broker,
            resolve_plan=lambda run: "plan",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            task = asyncio.create_task(client.get(f"/api/runs/{RUN_ID}/events"))
            await started.wait()
            assert broker._subscribers
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not broker._subscribers

    asyncio.run(scenario())


def test_live_read_failure_emits_redacted_resume_position():
    class FailingStore:
        def read_after(self, *args, **kwargs):
            raise BudgetExhausted("secret-token")

    async def scenario():
        broker = EventBroker()
        stream = stream_events(FailingStore(), broker, RUN_ID, (2, 100))
        frame = await anext(stream)
        assert b'"resumePosition":{"attempt":2,"seq":100}' in frame
        assert b"secret-token" not in frame
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not broker._subscribers

    asyncio.run(scenario())


def test_stream_asgi_failure_before_iteration_releases_preadmitted_subscriber():
    class Store:
        def read_after(self, *args, **kwargs):
            return []

    async def scenario():
        broker = EventBroker()
        routes = event_routes(
            event_log=Store(),
            broker=broker,
            resolve_plan=lambda run: "plan",
        )
        scope = {
            "type": "http",
            "method": "GET",
            "path": f"/api/runs/{RUN_ID}/events",
            "query_string": b"",
            "headers": [],
            "path_params": {"run_id": RUN_ID},
            "asgi": {"version": "3.0", "spec_version": "2.4"},
        }
        response = await routes[0].endpoint(RUN_ID, Request(scope))
        assert broker._subscribers

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            raise OSError("Disconnected before headers")

        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
        assert not broker._subscribers

    asyncio.run(scenario())
