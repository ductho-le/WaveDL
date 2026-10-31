"""A retained runtime is diagnostic history while its native replacement waits."""

import asyncio
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from unit_tests.daemon import test_writer_submissions as writers
from unit_tests.daemon.test_bound_native_collections import proposal
from unit_tests.daemon.test_native_heads import prepare
from unit_tests.daemon.test_writer_historical_lineage import journal
from unit_tests.daemon.test_writer_submissions import binding
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import ProtocolError, decode, encode
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.events.ingestion import Ingestor
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.runs.negotiation import NegotiationStatus
from wavedl_daemon.runs.runtime_service import ServiceSession
from wavedl_daemon.storage.event_log import EventLogStore
from wavedl_daemon.storage.registry import Registry


writer = writers.writer


@pytest.fixture
def pending(writer):
    store, saved, current = prepare(writer)
    current = store.activate_initial(
        saved.plan_id, binding(current, attempt=1), expected_revision=current.revision
    )
    evidence = proposal(current, saved, 1)
    current = store.record_native_generation(
        saved.plan_id, evidence.identity, expected_revision=current.revision
    )
    layout = RunLayout(Path(current.run_dir), current.run_id)
    layout.attempts.mkdir(mode=0o700)
    journal(layout, current.attempt)
    return store, saved, current, layout


def test_pending_last_runtime_journal_is_history_without_negotiation(pending):
    store, saved, current, layout = pending
    controller = SimpleNamespace(
        execution_store=store,
        accept_runtime_events=Mock(),
        reject_runtime_events=Mock(),
    )
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as pool,
        closing(EventLogStore(store.path)) as log,
    ):
        ingestor = Ingestor(registry, controller, EventBroker(), pool)
        assert (
            asyncio.run(
                ingestor.poll_once(
                    saved.plan_id, current.run_id, 1, layout.journal(1), historical=True
                )
            )
            == 1
        )
        assert registry.cursor(current.run_id, 1) == 1
        assert [
            (row.attempt, row.value)
            for row in registry.metrics(current.run_id, active_only=False)
        ] == [(1, 999.0)]
        assert [
            (event.attempt, event.seq) for event in log.read_after(current.run_id)
        ] == [(1, 0), (1, 1)]
        assert store.get(saved.plan_id) == current
        assert list(layout.attempts.iterdir()) == [layout.attempt_dir(1)]
        controller.accept_runtime_events.assert_not_called()
        controller.reject_runtime_events.assert_not_called()


def test_pending_old_live_push_and_rejection_do_not_touch_provenance(pending):
    store, saved, current, layout = pending
    controller = SimpleNamespace(execution_store=store, reject_runtime_events=Mock())
    with Registry(store.path, dispatch_lock=store.dispatch_lock) as registry:
        ingestor = Ingestor(registry, controller, EventBroker(), None)
        hello = decode(layout.journal(1).read_bytes().splitlines()[0])
        with pytest.raises(ProtocolError, match="active attempt"):
            ingestor.push(saved.plan_id, current.run_id, 1, [hello])
        with pytest.raises(ProtocolError, match="active attempt"):
            ingestor.push(saved.plan_id, current.run_id, 1, [hello], journal=True)
        with pytest.raises(ProtocolError, match="active attempt"):
            ingestor.reject_runtime_events(
                saved.plan_id, current.run_id, 1, "bad old wire"
            )
        assert registry.cursor(current.run_id, 1) == -1
        controller.reject_runtime_events.assert_not_called()


def test_completed_old_runtime_can_append_diagnostic_metric_without_new_attempt(
    pending,
):
    store, saved, current, layout = pending
    events = [decode(line) for line in layout.journal(1).read_bytes().splitlines()]
    completed = replace(events[1], seq=2, type="state", payload={"state": "completed"})
    late_metric = replace(
        events[1], seq=3, payload={"name": "loss", "value": 12.0, "step": 2}
    )
    # Represent the earlier committed completed journal, then a late tail. No
    # runtime-2 directory or cursor is invented to label the native wait.
    layout.journal(1).write_bytes(
        b"".join(encode(event) for event in [*events, completed, late_metric])
    )
    controller = SimpleNamespace(execution_store=store)
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        registry.apply_events(current.run_id, 1, [*events, completed])
        assert registry.cursor(current.run_id, 1) == 2
        ingestor = Ingestor(registry, controller, EventBroker(), pool)
        assert (
            asyncio.run(
                ingestor.poll_once(
                    saved.plan_id, current.run_id, 1, layout.journal(1), historical=True
                )
            )
            == 3
        )
        assert [
            (row.step, row.value)
            for row in registry.metrics(current.run_id, active_only=False)
        ] == [(1, 999.0), (2, 12.0)]
        assert registry.cursor(current.run_id, 2) == -1
        assert store.get(saved.plan_id) == current
        assert list(layout.attempts.iterdir()) == [layout.attempt_dir(1)]


def test_pending_source_requires_matching_registered_handle(pending):
    store, saved, current, layout = pending
    forged = replace(current, handle=replace(current.handle, request_id="forged"))
    controller = SimpleNamespace(execution_store=SimpleNamespace(get=lambda _: forged))
    with Registry(store.path, dispatch_lock=store.dispatch_lock) as registry:
        ingestor = Ingestor(registry, controller, EventBroker(), None)
        with pytest.raises(ProtocolError, match="registered execution lineage"):
            asyncio.run(
                ingestor.poll_once(
                    saved.plan_id, current.run_id, 1, layout.journal(1), historical=True
                )
            )
        assert not ingestor._readers
        assert registry.cursor(current.run_id, 1) == -1


def test_active_read_of_pending_runtime_never_latches_protocol_rejection(pending):
    store, saved, current, layout = pending
    controller = SimpleNamespace(
        execution_store=store,
        reject_runtime_events=Mock(),
        accept_runtime_events=Mock(),
    )
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(registry, controller, EventBroker(), pool)
        with pytest.raises(ProtocolError, match="active attempt"):
            asyncio.run(
                ingestor.poll_once(saved.plan_id, current.run_id, 1, layout.journal(1))
            )
        controller.reject_runtime_events.assert_not_called()
        controller.accept_runtime_events.assert_not_called()
        assert registry.cursor(current.run_id, 1) == -1


def test_new_binding_restores_active_push_and_keeps_old_archived_source(pending):
    store, saved, current, layout = pending
    candidate = replace(
        current.handle,
        attempt=2,
        raw_json=json.dumps(json.loads(current.handle.raw_json) | {"restart_count": 1}),
    )
    current = store.activate_requeue(
        saved.plan_id,
        replace(binding(current), attempt=2, restart_count=1),
        candidate,
        expected_revision=current.revision,
    )
    journal(layout, 2)
    controller = SimpleNamespace(
        execution_store=store,
        accept_runtime_events=Mock(
            return_value=SimpleNamespace(status=NegotiationStatus.ACCEPTED)
        ),
    )
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(registry, controller, EventBroker(), pool)
        assert (
            asyncio.run(
                ingestor.poll_once(
                    saved.plan_id, current.run_id, 1, layout.journal(1), historical=True
                )
            )
            == 1
        )
        events = [decode(line) for line in layout.journal(2).read_bytes().splitlines()]
        assert ingestor.push(saved.plan_id, current.run_id, 2, events) == 1
        controller.accept_runtime_events.assert_called_once()


def test_pending_journal_remains_history_after_binding_advances_during_read(pending):
    store, saved, current, layout = pending
    candidate = replace(
        current.handle,
        attempt=2,
        raw_json=json.dumps(json.loads(current.handle.raw_json) | {"restart_count": 1}),
    )
    proof = replace(binding(current), attempt=2, restart_count=1)
    with (
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
        BoundedThreadPool(WorkBudget()) as real_pool,
    ):

        class AdvancingPool:
            def submit(self, callback):
                handle = real_pool.submit(callback)
                store.activate_requeue(
                    saved.plan_id, proof, candidate, expected_revision=current.revision
                )
                return handle

        ingestor = Ingestor(
            registry,
            SimpleNamespace(execution_store=store),
            EventBroker(),
            AdvancingPool(),
        )
        assert (
            asyncio.run(
                ingestor.poll_once(
                    saved.plan_id, current.run_id, 1, layout.journal(1), historical=True
                )
            )
            == 1
        )
        assert store.get(saved.plan_id).attempt == 2
        assert registry.cursor(current.run_id, 1) == 1
        assert registry.cursor(current.run_id, 2) == -1


def test_pending_pump_tails_retained_runtime_as_history(pending, monkeypatch):
    store, saved, current, _ = pending
    session = ServiceSession(None)
    session.controller = Mock()
    session.runs = SimpleNamespace(
        execution_store=store,
        plan_store=SimpleNamespace(list_pending=lambda **_: [saved]),
    )
    session.ingestor = SimpleNamespace(poll_once=AsyncMock())

    async def stop_after_lap(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "wavedl_daemon.runs.runtime_service.asyncio.sleep", stop_after_lap
    )
    with BoundedThreadPool(WorkBudget()) as pool:
        session.pool = pool
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(session._pump())
    assert session.errors == {}
    assert session.ingestor.poll_once.await_count == 1
    args, kwargs = session.ingestor.poll_once.await_args
    assert args[2] == current.attempt == 1
    assert kwargs == {"historical": True}
