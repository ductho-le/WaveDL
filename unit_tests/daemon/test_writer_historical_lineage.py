"""Managed replay authority is archived activation, never a numeric directory."""

import asyncio
import json
from contextlib import closing, contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from unit_tests.daemon.test_writer_submissions import binding, prepare, writer
from wavedl.core.events import HelloPayload
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import Envelope, ProtocolError, encode
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.events.ingestion import Ingestor
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.runs.runtime_service import ServiceSession
from wavedl_daemon.storage.event_log import EventLogStore
from wavedl_daemon.storage.registry import Registry


@pytest.fixture
def lineage(tmp_path):
    with contextmanager(writer.__wrapped__)(tmp_path) as (store, plans, saved):
        # Permit the real store's fenced requeue transition for this fixture.
        # Create a separate immutable plan rather than modifying saved bytes.
        saved = plans.put(
            replace(saved.plan, resources=(("requeue", True),)),
            idempotency_key="historical-lineage",
            request_payload={},
        )
        plans.request_submission(saved.plan_id)
        accepted = prepare((store, plans, saved))
        layout = RunLayout(tmp_path / "run", accepted.run_id)
        layout.attempts.mkdir(mode=0o700)
        for attempt in (1, 3):
            layout.attempt_dir(attempt).mkdir(mode=0o700)
        proof = binding(accepted, attempt=4)
        active = store.activate_initial(
            saved.plan_id, proof, expected_revision=accepted.revision
        )
        yield store, active, layout, proof


def journal(layout, attempt):
    directory = layout.attempt_dir(attempt)
    directory.mkdir(mode=0o700, exist_ok=True)
    hello = HelloPayload(1, 1, "test", "3.11", "cpu", ["train"], "/usr/bin/python3")
    events = [
        Envelope(
            layout.run_id,
            attempt,
            "hello",
            0,
            "2026-10-05T00:00:00Z",
            asdict(hello),
        ),
        Envelope(
            layout.run_id,
            attempt,
            "metric",
            1,
            "2026-10-05T00:00:00Z",
            {"name": "loss", "value": 999.0, "step": 1},
        ),
    ]
    layout.journal(attempt).write_bytes(b"".join(encode(event) for event in events))


def test_initial_gapped_attempt_rejects_preexisting_forged_journals(lineage):
    store, active, layout, _ = lineage
    for attempt in (1, 3, 4):
        journal(layout, attempt)
    session = ServiceSession(None)
    session.runs = SimpleNamespace(execution_store=store)
    selected = session._lineage_attempts(active, 0)
    assert selected == [4]
    assert session._attempts(layout, 4, selected) == [4]
    broker = EventBroker()
    notifications = []
    broker.notify = notifications.append
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as pool,
        closing(EventLogStore(store.path)) as log,
    ):
        ingestor = Ingestor(
            registry, SimpleNamespace(execution_store=store), broker, pool
        )
        for attempt in (1, 3):
            with pytest.raises(ProtocolError, match="activated archived lineage"):
                asyncio.run(
                    ingestor.poll_once(
                        active.plan_id,
                        active.run_id,
                        attempt,
                        layout.journal(attempt),
                        historical=True,
                    )
                )
            assert registry.cursor(active.run_id, attempt) == -1
        assert not registry.metrics(active.run_id)
        assert not log.read_after(active.run_id)
        assert not notifications
        assert not ingestor._readers and pool.outstanding == 0


def test_real_archived_activation_replays_but_skipped_directory_does_not(lineage):
    store, active, layout, proof = lineage
    journal(layout, 3)
    journal(layout, 4)
    journal(layout, 9)
    candidate = replace(
        active.handle,
        attempt=9,
        raw_json=json.dumps(json.loads(active.handle.raw_json) | {"restart_count": 1}),
    )
    current = store.activate_requeue(
        active.plan_id,
        replace(proof, attempt=9, restart_count=1),
        candidate,
        expected_revision=active.revision,
    )
    session = ServiceSession(None)
    session.runs = SimpleNamespace(execution_store=store)
    assert session._lineage_attempts(current, 0) == [4, 9]
    assert session._attempts(layout, 9, [4, 9]) == [4, 9]
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as pool,
        closing(EventLogStore(store.path)) as log,
    ):
        ingestor = Ingestor(
            registry, SimpleNamespace(execution_store=store), EventBroker(), pool
        )
        assert (
            asyncio.run(
                ingestor.poll_once(
                    active.plan_id, active.run_id, 4, layout.journal(4), historical=True
                )
            )
            == 1
        )
        assert registry.cursor(active.run_id, 4) == 1
        assert [(row.step, row.value) for row in registry.metrics(active.run_id)] == [
            (1, 999.0)
        ]
        assert [
            (event.attempt, event.seq) for event in log.read_after(active.run_id)
        ] == [
            (4, 0),
            (4, 1),
        ]
        with pytest.raises(ProtocolError, match="activated archived lineage"):
            asyncio.run(
                ingestor.poll_once(
                    active.plan_id, active.run_id, 3, layout.journal(3), historical=True
                )
            )
        assert registry.cursor(active.run_id, 3) == -1


def test_archived_first_record_failure_never_negotiates_active_runtime(lineage):
    store, active, layout, proof = lineage
    journal(layout, 4)
    metric = Envelope(
        active.run_id,
        4,
        "metric",
        0,
        "2026-10-05T00:00:00Z",
        {"name": "loss", "value": 999.0, "step": 1},
    )
    layout.journal(4).write_bytes(encode(metric))
    candidate = replace(
        active.handle,
        attempt=9,
        raw_json=json.dumps(json.loads(active.handle.raw_json) | {"restart_count": 1}),
    )
    store.activate_requeue(
        active.plan_id,
        replace(proof, attempt=9, restart_count=1),
        candidate,
        expected_revision=active.revision,
    )

    def forbidden(*args):
        pytest.fail("Historical failure must not negotiate or reject active runtime")

    controller = SimpleNamespace(
        execution_store=store,
        reject_runtime_events=forbidden,
        accept_runtime_events=forbidden,
    )
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(registry, controller, EventBroker(), pool)
        with pytest.raises(ProtocolError, match="First runtime event"):
            asyncio.run(
                ingestor.poll_once(
                    active.plan_id, active.run_id, 4, layout.journal(4), historical=True
                )
            )
        assert registry.cursor(active.run_id, 4) == -1


def test_owner_thread_history_window_is_bounded_and_fair():
    current = SimpleNamespace(
        plan_id="plan",
        run_id="run",
        request_id="request",
        owner="owner",
        backend="local",
        run_dir="/run",
        attempt=1001,
        handle=None,
    )
    # Test window selection independently of identity validation.
    from unittest.mock import patch

    pages = []
    history = [SimpleNamespace(attempt=number) for number in range(1, 1001)]

    def page(plan_id, *, limit, offset):
        pages.append((limit, offset))
        return history[offset : offset + limit]

    session = ServiceSession(None)
    session.runs = SimpleNamespace(
        execution_store=SimpleNamespace(history_count=lambda plan: 1000, history=page)
    )
    seen = set()
    with (
        patch(
            "wavedl_daemon.runs.runtime_service.archived_execution_matches",
            return_value=True,
        ),
        patch(
            "wavedl_daemon.runs.runtime_service.registered_handle_matches",
            return_value=True,
        ),
    ):
        for epoch in range(500):
            selected = session._lineage_attempts(current, epoch)
            assert len(selected) <= 3 and current.attempt in selected
            seen.update(selected)
    assert seen == set(range(1, 1002))
    assert all(limit <= 2 for limit, _ in pages)
