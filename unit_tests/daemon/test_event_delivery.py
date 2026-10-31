"""Executable durability, handover and bounded-delivery failure contracts."""

import asyncio
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import ClientDisconnect, Request

from unit_tests.daemon import test_adoption_service as adoption_tests
from unit_tests.daemon.conftest import route_app
from wavedl.core.events import HelloPayload
from wavedl.core.journal_reader import JournalReader
from wavedl.core.protocol import Envelope, ProtocolError, decode, encode
from wavedl.journal import RunJournal, bootstrap_run
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.events.ingestion import Ingestor
from wavedl_daemon.events.streaming import stream_events
from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.http.events import event_routes
from wavedl_daemon.runs.negotiation import NegotiationStatus
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.event_log import (
    EVENT_LOG_MIGRATION,
    EventLogStore,
    append_in_transaction,
    record_bytes,
)
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.settings import Settings

from .test_hosted_app import Backend


RUN_ID = "00000000-0000-4000-8000-000000000001"
latest_adoption = adoption_tests.adoption_service


def event(seq, attempt=1):
    return Envelope(
        RUN_ID,
        attempt,
        "log",
        seq,
        "2026-10-03T00:00:00Z",
        {"message": "Tiếng Việt 🌊", "level": "info", "stream": "stdout"},
    )


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "events.db"
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA user_version=5")
    connection.execute("BEGIN")
    EVENT_LOG_MIGRATION.apply(connection)
    connection.execute("PRAGMA user_version=6")
    connection.execute("COMMIT")
    store = EventLogStore(path)
    yield connection, store
    store.close()
    connection.close()


def append(connection, *events):
    connection.execute("BEGIN")
    for item in events:
        append_in_transaction(connection, item)
    connection.execute("COMMIT")


def test_event_append_requires_transaction_and_rolls_back(log):
    connection, store = log
    with pytest.raises(RuntimeError):
        append_in_transaction(connection, event(0))
    connection.execute("BEGIN")
    append_in_transaction(connection, event(0))
    connection.execute("ROLLBACK")
    assert store.read_after(RUN_ID) == []


def test_numeric_attempt_cursor_and_sparse_sequence(log):
    connection, store = log
    append(connection, event(100), event(0, 2), event(80, 2), event(0, 10))
    assert [(e.attempt, e.seq) for e in store.read_after(RUN_ID, (1, 100))] == [
        (2, 0),
        (2, 80),
        (10, 0),
    ]
    assert store.read_after(RUN_ID, (2, -1), limit=1) == [event(0, 2)]


def test_duplicate_rows_are_not_retained_in_memory_or_replayed(log):
    connection, store = log
    append(connection, event(0), event(0))
    assert store.read_after(RUN_ID) == [event(0)]


def test_commit_without_notification_is_delivered(log):
    connection, store = log

    async def scenario():
        broker = EventBroker()
        stream = stream_events(store, broker, RUN_ID, poll_interval=0.01)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        append(connection, event(0))
        assert await pending == b": heartbeat\n\n"
        assert b"id: 1:0" in await anext(stream)
        await stream.aclose()
        assert len(broker._subscribers) == 0

    asyncio.run(scenario())


def test_register_before_replay_closes_handover_race(log):
    connection, store = log
    broker = EventBroker()

    class RacingStore:
        raced = False

        def read_after(self, *args, **kwargs):
            page = store.read_after(*args, **kwargs)
            if not self.raced:
                self.raced = True
                assert len(broker._subscribers) == 1
                append(connection, event(0))
                broker.notify(RUN_ID)
            return page

    async def scenario():
        stream = stream_events(RacingStore(), broker, RUN_ID, poll_interval=0.01)
        assert await anext(stream) == b": heartbeat\n\n"
        assert b"id: 1:0" in await anext(stream)
        await stream.aclose()

    asyncio.run(scenario())


def test_overflow_disconnects_with_last_delivered_position(log):
    connection, store = log
    append(connection, event(0))

    async def scenario():
        broker = EventBroker(queue_size=1)
        stream = stream_events(store, broker, RUN_ID, page_size=1)
        first = await anext(stream)
        assert "Tiếng Việt 🌊" in first.decode("utf-8")
        append(connection, event(1), event(2))
        broker.notify(RUN_ID)
        broker.notify(RUN_ID)
        overflow = await anext(stream)
        assert b'"resumePosition":{"attempt":1,"seq":0}' in overflow
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert len(broker._subscribers) == 0
        assert [e.seq for e in store.read_after(RUN_ID, (1, 0))] == [1, 2]

    asyncio.run(scenario())


def test_disconnected_stream_cancels_async_wait(log):
    _, store = log

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


class AcceptedController:
    calls = 0

    def accept_runtime_events(self, plan_id, events):
        self.calls += 1
        return SimpleNamespace(status=NegotiationStatus.ACCEPTED, reason="")


def test_registry_event_projection_and_cursor_rollback(tmp_path, monkeypatch):
    path = tmp_path / "registry.db"
    with Registry(path) as registry:
        store = EventLogStore(path)

        def fail(*args):
            raise RuntimeError("cursor failure after durable event insertion")

        monkeypatch.setattr(registry, "_write_cursor", fail)
        with pytest.raises(RuntimeError):
            registry.apply_events(RUN_ID, 1, [event(0)])
        assert registry.get(RUN_ID) is None
        assert registry.cursor(RUN_ID, 1) == -1
        assert store.read_after(RUN_ID) == []
        store.close()


def test_live_push_and_journal_replay_deduplicate(tmp_path):
    path = tmp_path / "registry.db"
    journal = tmp_path / "events.jsonl"
    journal.write_bytes(encode(event(0)) + b"\n" + encode(event(1)) + b"\n")
    with Registry(path) as registry, BoundedThreadPool(WorkBudget()) as pool:
        controller, broker = AcceptedController(), EventBroker()
        ingestor = Ingestor(registry, controller, broker, pool)
        ingestor.push("plan", RUN_ID, 1, [event(0)])
        assert asyncio.run(ingestor.poll_once("plan", RUN_ID, 1, journal)) == 1
        assert ingestor.push("plan", RUN_ID, 1, [event(0), event(1)]) == 1
        store = EventLogStore(path)
        assert [e.seq for e in store.read_after(RUN_ID)] == [0, 1]
        assert registry.get(RUN_ID).event_count == 2
        store.close()


def test_bad_first_hello_never_fabricates_training_projection(tmp_path):
    class RejectedController:
        def accept_runtime_events(self, plan_id, events):
            return SimpleNamespace(status=NegotiationStatus.FAILED, reason="bad hello")

    with Registry(tmp_path / "registry.db") as registry:
        ingestor = Ingestor(registry, RejectedController(), EventBroker(), None)
        with pytest.raises(ProtocolError, match="bad hello"):
            ingestor.push("plan", RUN_ID, 1, [event(0)])
        assert registry.get(RUN_ID) is None
        assert registry.cursor(RUN_ID, 1) == -1


def test_push_gap_rejection_does_not_skip_journal_records(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        ingestor = Ingestor(registry, AcceptedController(), EventBroker(), None)
        ingestor.push("plan", RUN_ID, 1, [event(0)])
        with pytest.raises(ProtocolError, match="gap"):
            ingestor.push("plan", RUN_ID, 1, [event(2)])
        assert registry.cursor(RUN_ID, 1) == 0
        ingestor.push("plan", RUN_ID, 1, [event(1), event(2)], journal=True)
        assert registry.cursor(RUN_ID, 1) == 2


def test_push_gap_rejects_before_runtime_provenance_side_effect(tmp_path):
    class RecordingController:
        def __init__(self):
            self.calls = 0

        def accept_runtime_events(self, plan_id, events):
            self.calls += 1
            return SimpleNamespace(status=NegotiationStatus.ACCEPTED, reason="")

    controller = RecordingController()
    with Registry(tmp_path / "registry.db") as registry:
        ingestor = Ingestor(registry, controller, EventBroker(), None)
        ingestor.push("plan", RUN_ID, 1, [event(0)])
        assert controller.calls == 1
        with pytest.raises(ProtocolError, match="gap"):
            ingestor.push("plan", RUN_ID, 1, [event(2)])
        assert controller.calls == 1
        assert registry.cursor(RUN_ID, 1) == 0


@pytest.mark.parametrize("include_hello", [False, True])
def test_first_push_gap_does_not_commit_hello_provenance(tmp_path, include_hello):
    backend = Backend()
    services = HostedServices(
        Settings(tmp_path / "state", tmp_path / "runs"),
        backend_factory=lambda lock: {"local": backend},
    )

    async def scenario():
        await services.start()
        try:
            plan = await services.run_service.plan(
                {
                    "backend": "local",
                    "operation": "train",
                    "config": {"epochs": 1},
                    "topology": {
                        "num_machines": 1,
                        "processes_per_machine": 1,
                        "machine_rank": 0,
                    },
                    "runtime": {
                        "python": sys.executable,
                        "mode": "existing",
                        "device": "cpu",
                        "environment": {},
                    },
                },
                "gap-preflight",
            )
            await services.run_service.submit(plan.plan_id)
            deadline = time.monotonic() + 3
            while backend.handle is None and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert backend.handle is not None
            binding = backend.runtime_startup()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                view = await services.run_service.get(plan.run_id)
                if view.attempt == binding.attempt:
                    break
                await asyncio.sleep(0.01)
            assert view.attempt == binding.attempt
            hello = Envelope(
                plan.run_id,
                binding.attempt,
                "hello",
                0,
                "2026-10-03T00:00:00Z",
                asdict(
                    HelloPayload(
                        1, 1, "actual", "3.11", "cpu", ["train"], sys.executable
                    )
                ),
            )
            later = replace(event(2), run_id=plan.run_id, attempt=binding.attempt)
            with pytest.raises(ProtocolError, match="gap"):
                await services.ingestor.push(
                    plan.plan_id,
                    plan.run_id,
                    binding.attempt,
                    [hello, later] if include_hello else [later],
                )
            snapshot = await services.host.call(
                lambda session: (
                    session.controller.provenance_store.get(plan.plan_id),
                    session.registry.cursor(plan.run_id, binding.attempt),
                )
            )
            assert snapshot == (None, -1)
            assert await services.event_log.read_after(plan.run_id) == []
            await services.ingestor.push(
                plan.plan_id, plan.run_id, binding.attempt, [hello]
            )
            assert await services.event_log.read_after(plan.run_id) == [hello]
        finally:
            await services.close()

    asyncio.run(scenario())


def test_conflicting_duplicate_push_is_rejected(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        ingestor = Ingestor(registry, AcceptedController(), EventBroker(), None)
        ingestor.push("plan", RUN_ID, 1, [event(0)])
        conflicting = Envelope(
            RUN_ID,
            1,
            "log",
            0,
            "2026-10-03T00:00:00Z",
            {
                "message": "different",
                "stream": "stdout",
                "level": "info",
            },
        )
        with pytest.raises(ProtocolError, match="Conflicting"):
            ingestor.push("plan", RUN_ID, 1, [conflicting])
        assert registry.get(RUN_ID).event_count == 1


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
    routes = event_routes(
        ingestor=None, event_log=None, broker=None, resolve_plan=lambda run: "plan"
    )

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

    routes = event_routes(
        ingestor=None, event_log=None, broker=None, resolve_plan=resolve
    )

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


def test_http_push_is_bounded_and_requires_explicit_attempt():
    routes = event_routes(
        ingestor=None, event_log=None, broker=None, resolve_plan=lambda run: "plan"
    )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            path = f"/api/runs/{RUN_ID}/events"
            response = await client.post(path, json={"events": []})
            assert response.status_code == 400
            response = await client.post(path, json={"attempt": True, "events": [{}]})
            assert response.status_code == 400
            response = await client.post(
                path, json={"attempt": 1, "events": [{}] * 1025}
            )
            assert response.status_code == 400
            response = await client.post(
                path,
                content=b"x" * (1024 * 1024 + 1),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413

    asyncio.run(scenario())


def test_only_filesystem_read_runs_on_shared_worker(tmp_path, monkeypatch):
    journal = tmp_path / "events.jsonl"
    journal.write_bytes(encode(event(0)) + b"\n")
    owner = threading.get_ident()
    readers, writers = [], []
    original_read = JournalReader.read

    def track_read(self, **options):
        readers.append(threading.get_ident())
        return original_read(self, **options)

    monkeypatch.setattr(JournalReader, "read", track_read)
    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        original_apply = registry.apply_events

        def track_apply(*args, **options):
            writers.append(threading.get_ident())
            return original_apply(*args, **options)

        monkeypatch.setattr(registry, "apply_events", track_apply)
        ingestor = Ingestor(registry, AcceptedController(), EventBroker(), pool)
        asyncio.run(ingestor.poll_once("plan", RUN_ID, 1, journal))
    assert readers and all(value != owner for value in readers)
    assert writers == [owner]


def test_schema5_journal_backfill_preserves_projection_and_cursor(tmp_path):
    path = tmp_path / "registry.db"
    with Registry(path) as registry:
        registry.apply_events(RUN_ID, 1, [event(0), event(1)])
        before = registry.get(RUN_ID)
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER writer_initial_activation")
    connection.execute("DROP TABLE initial_bindings")
    connection.execute("DROP TABLE submission_control")
    connection.execute(
        "DELETE FROM schema_migrations WHERE migration_id='0009_writer_submissions'"
    )
    connection.execute("DROP TABLE execution_collections")
    connection.execute(
        "DELETE FROM schema_migrations WHERE migration_id='0008_execution_collections'"
    )
    connection.execute("DROP TRIGGER execution_attempt_activation")
    connection.execute("DROP TRIGGER execution_identity_immutable")
    connection.execute("DROP TABLE requeue_bindings")
    connection.execute("DROP TABLE execution_attempt_history")
    connection.execute(
        "CREATE TRIGGER execution_identity_immutable BEFORE UPDATE OF plan_id,attempt,owner ON execution_control BEGIN SELECT RAISE(ABORT,'immutable'); END"
    )
    connection.execute(
        "DELETE FROM schema_migrations WHERE migration_id='0007_scheduler_requeues'"
    )
    connection.execute("DROP TABLE event_log")
    connection.execute(
        "DELETE FROM schema_migrations WHERE migration_id='0006_event_log'"
    )
    connection.execute("PRAGMA user_version=5")
    connection.commit()
    connection.close()
    with Registry(path) as registry:
        ingestor = Ingestor(registry, AcceptedController(), EventBroker(), None)
        with pytest.raises(ProtocolError, match="Conflicting"):
            ingestor.push("plan", RUN_ID, 1, [event(0)])
        assert ingestor.push("plan", RUN_ID, 1, [event(0), event(1)], journal=True) == 1
        assert registry.get(RUN_ID) == before
        assert registry.cursor(RUN_ID, 1) == 1
        store = EventLogStore(path)
        assert store.read_after(RUN_ID) == [event(0), event(1)]
        store.close()


def test_record_and_replay_pages_are_byte_bounded(log):
    connection, store = log
    batch = [
        Envelope(
            RUN_ID,
            1,
            "log",
            seq,
            "2026-10-03T00:00:00Z",
            {
                "message": "波" * 20000,
                "stream": "stdout",
                "level": "info",
            },
        )
        for seq in range(30)
    ]
    append(connection, *batch)
    page = store.read_after(RUN_ID, limit=1024)
    assert 0 < len(page) < len(batch)
    assert sum(len(record_bytes(item)) for item in page) <= 1024 * 1024
    large = Envelope(
        RUN_ID,
        1,
        "log",
        30,
        "2026-10-03T00:00:00Z",
        {
            "message": "波" * 22000,
            "stream": "stdout",
            "level": "info",
        },
    )
    connection.execute("BEGIN")
    with pytest.raises(ProtocolError, match="byte bound"):
        append_in_transaction(connection, large)
    connection.execute("ROLLBACK")


def test_unsupported_first_journal_protocol_records_rejection_on_owner(tmp_path):
    class Controller:
        def __init__(self):
            self.rejected = []

        def reject_runtime_events(self, plan_id, reason):
            self.rejected.append((threading.get_ident(), plan_id, reason))

    journal = tmp_path / "events.jsonl"
    journal.write_bytes(encode(event(0)).replace(b'"version":1', b'"version":99'))
    controller = Controller()
    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(registry, controller, EventBroker(), pool)
        with pytest.raises(ProtocolError, match="Unsupported version"):
            asyncio.run(ingestor.poll_once("plan", RUN_ID, 1, journal))
        assert registry.get(RUN_ID) is None
        assert registry.cursor(RUN_ID, 1) == -1
    assert controller.rejected == [
        (threading.get_ident(), "plan", "Unsupported version")
    ]


def test_async_owner_adapter_replay_and_http_push():
    class AsyncStore:
        async def read_after(self, run_id, cursor, *, limit):
            await asyncio.sleep(0)
            return [event(0)] if cursor == (0, -1) else []

    class AsyncIngestor:
        async def push(self, plan_id, run_id, attempt, events):
            await asyncio.sleep(0)
            assert (plan_id, run_id, attempt) == ("plan", RUN_ID, 1)
            return events[-1].seq

    async def resolve(run_id):
        await asyncio.sleep(0)
        return "plan"

    async def scenario():
        broker = EventBroker()
        stream = stream_events(AsyncStore(), broker, RUN_ID)
        assert b"id: 1:0" in await anext(stream)
        await stream.aclose()
        routes = event_routes(
            ingestor=AsyncIngestor(),
            event_log=AsyncStore(),
            broker=broker,
            resolve_plan=resolve,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            import json

            response = await client.post(
                f"/api/runs/{RUN_ID}/events",
                json={
                    "attempt": 1,
                    "events": [json.loads(encode(event(0)))],
                },
            )
            assert response.json() == {"attempt": 1, "seq": 0}

    asyncio.run(scenario())


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


def test_adopted_latest_attempt_ingests_historical_metrics_without_old_provenance(
    latest_adoption,
):
    service, original = latest_adoption
    run_id, layout = original.layout.run_id, original.layout
    first_hello = decode(layout.journal(1).read_bytes().splitlines()[0])
    metrics = [
        Envelope(
            run_id,
            1,
            "metric",
            step,
            first_hello.ts,
            {"name": "loss", "step": step, "value": float(step)},
        )
        for step in (1, 2)
    ]
    layout.journal(1).write_bytes(
        b"".join(encode(item) for item in [first_hello, *metrics])
    )
    second = bootstrap_run(
        layout.run_dir,
        resolved_config={"epochs": 1},
        runtime=original.request_manifest.runtime,
        environment={},
    )
    assert second.attempt == 2
    payload = HelloPayload.from_dict(first_hello.payload)
    with RunJournal(second, hello=payload) as journal:
        journal.emit(
            "supersede",
            {
                "source_attempt": 1,
                "step": 1,
                "generation": "gen-restored",
            },
        )
        journal.emit("metric", {"name": "loss", "step": 2, "value": 20.0})
    response = asyncio.run(
        service.adopt(str(layout.run_dir), idempotency_key="latest-two")
    )
    assert response.attempt == 2
    broker = EventBroker()
    ingestor = Ingestor(service.registry, service.controller, broker, service.pool)

    async def scenario():
        # Parent metric visibility can lag: retry must not reject the already
        # verified active hello or advance its uncommitted projection cursor.
        with pytest.raises(ProtocolError):
            await ingestor.poll_once(response.plan_id, run_id, 2, layout.journal(2))
        assert service.registry.cursor(run_id, 2) == -1
        assert (
            service.controller.accept_runtime_events(response.plan_id, []).status
            == NegotiationStatus.ACCEPTED
        )
        await ingestor.poll_once(
            response.plan_id, run_id, 1, layout.journal(1), historical=True
        )
        await ingestor.poll_once(response.plan_id, run_id, 2, layout.journal(2))

    asyncio.run(scenario())
    assert [(row.step, row.value) for row in service.registry.metrics(run_id)] == [
        (1, 1.0),
        (2, 20.0),
    ]
    provenance = service.provenance.get(response.plan_id)
    assert provenance.attempt == 2
    with sqlite3.connect(service.registry.path) as connection:
        assert connection.execute(
            "SELECT attempt FROM runtime_provenance"
        ).fetchall() == [(2,)]
    store = EventLogStore(service.registry.path)
    assert [(item.attempt, item.seq) for item in store.read_after(run_id)][:3] == [
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    store.close()
    with pytest.raises(ProtocolError, match="active attempt"):
        ingestor.push(response.plan_id, run_id, 1, [first_hello])


def test_unregistered_historical_record_never_rejects_active_runtime(tmp_path):
    run_dir = tmp_path / "run"
    old = run_dir / "attempts" / "1"
    old.mkdir(parents=True)
    path = old / "events.jsonl"
    path.write_bytes(encode(event(1)))

    class Controller:
        execution_store = SimpleNamespace(
            get=lambda plan: SimpleNamespace(
                run_id=RUN_ID,
                attempt=2,
                run_dir=str(run_dir),
            )
        )

        def reject_runtime_events(self, *args):
            pytest.fail("Historical data must not poison the current runtime")

        def accept_runtime_events(self, *args):
            pytest.fail("Historical data must not negotiate current runtime")

    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(registry, Controller(), EventBroker(), pool)
        with pytest.raises(ProtocolError, match="registered execution lineage"):
            asyncio.run(ingestor.poll_once("plan", RUN_ID, 1, path, historical=True))
        assert registry.cursor(RUN_ID, 1) == -1


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    "failure", [BudgetExhausted, StorageWorkError, sqlite3.OperationalError]
)
def test_event_http_resolver_storage_failures_are_redacted_503(method, failure):
    async def resolver(run_id):
        raise failure("secret-token private-path")

    routes = event_routes(
        ingestor=None, event_log=None, broker=EventBroker(), resolve_plan=resolver
    )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=route_app(routes=routes)),
            base_url="http://testserver",
        ) as client:
            response = await client.request(
                method,
                f"/api/runs/{RUN_ID}/events",
                json={"attempt": 1, "events": [{}]},
            )
            assert response.status_code == 503
            assert "secret-token" not in response.text
            assert "private-path" not in response.text

    asyncio.run(scenario())


def test_event_http_full_broker_rejects_before_stream_headers():
    async def scenario():
        broker = EventBroker(max_subscribers=1)
        occupied = broker.subscribe(RUN_ID)
        routes = event_routes(
            ingestor=None,
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
            ingestor=None,
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
            ingestor=None,
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
            ingestor=None,
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
        response = await routes[1].endpoint(RUN_ID, Request(scope))
        assert broker._subscribers

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            raise OSError("Disconnected before headers")

        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
        assert not broker._subscribers

    asyncio.run(scenario())
