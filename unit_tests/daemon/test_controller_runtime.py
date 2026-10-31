"""Actual journal startup is negotiated without importing the ML runtime."""

import json
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, replace

import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.protocol import Envelope
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    Plan,
)
from wavedl_daemon.execution.budget import StorageWorkError, WorkBudget
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.negotiation import NegotiationStatus, StaleNegotiation
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.executions import EXECUTION_MIGRATION, ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PLAN_MIGRATION, PlanStore
from wavedl_daemon.storage.runtime_provenance import (
    RUNTIME_PROVENANCE_MIGRATION,
    RuntimeProvenanceStore,
)
from wavedl_daemon.storage.schema import migrate


RUN = "00000000-0000-4000-8000-000000000031"


class Backend:
    owner = "test-user"

    def __init__(self):
        self.status = ObservationStatus.QUEUED
        self.observations = 0

    def submit(self, plan, *, request_id, attempt):
        return Handle(
            "slurm",
            plan.run_id,
            request_id,
            attempt,
            self.owner,
            plan.run_dir,
            job_id="31",
        )

    def observe(self, handle):
        self.observations += 1
        return BackendObservation(
            handle.attempt,
            self.status,
            exit_code=0 if self.status == ObservationStatus.TERMINAL else None,
        )


def hello(**changes):
    payload = asdict(
        HelloPayload(1, 1, "actual-runtime", "3.11", "cpu", ["train"], sys.executable)
    )
    payload.update(changes)
    return Envelope(RUN, 1, "hello", 0, "2026-09-05T12:00:00Z", payload)


@pytest.fixture
def runtime(tmp_path):
    database = tmp_path / "registry.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=1")
    migrate(
        database, [PLAN_MIGRATION, EXECUTION_MIGRATION, RUNTIME_PROVENANCE_MIGRATION]
    )
    layout = RunLayout(tmp_path / "runs" / RUN, RUN)
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    backend = Backend()
    now = [0.0]
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as provenance,
    ):
        plan = plans.put(
            Plan("slurm", json.dumps(asdict(manifest)), script="#!/bin/sh\n"),
            idempotency_key="runtime",
            request_payload={},
        )
        controller = SubmissionController(
            plans,
            executions,
            {"slurm": backend},
            lock,
            WorkBudget(max_workers=1, max_pending=0),
            provenance_store=provenance,
            clock=lambda: now[0],
            handshake_timeout_seconds=10,
        )
        controller.request_submit(plan.plan_id)
        yield (
            controller,
            plans,
            executions,
            provenance,
            backend,
            plan,
            layout,
            now,
            lock,
        )
        controller.close()


def until(controller, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        controller.reconcile_once()
        if predicate():
            return
        time.sleep(0.002)
    pytest.fail("bounded controller did not make progress")


def observe(runtime, status, now):
    controller, _, executions, _, backend, plan, _, clock, _ = runtime
    backend.status, clock[0] = status, now
    before = backend.observations
    revision = executions.get(plan.plan_id).revision
    until(
        controller,
        lambda: (
            backend.observations > before
            and executions.get(plan.plan_id).revision > revision
            and executions.get(plan.plan_id).latest_observation is not None
            and executions.get(plan.plan_id).latest_observation.status == status
        ),
    )


def test_waiting_hours_do_not_count_and_running_without_hello_times_out(runtime):
    controller, _, _, provenance, _, plan, _, _, _ = runtime
    observe(runtime, ObservationStatus.QUEUED, 0)
    observe(runtime, ObservationStatus.HELD, 7200)
    assert controller.derive(plan.plan_id).state == RunState.HELD
    observe(runtime, ObservationStatus.RUNNING, 10000)
    assert controller.derive(plan.plan_id).state == RunState.RUNNING
    observe(runtime, ObservationStatus.RUNNING, 10011)
    derived = controller.derive(plan.plan_id)
    assert derived.state == RunState.FAILED
    assert "handshake deadline" in derived.reason
    assert derived.backend.status == ObservationStatus.RUNNING
    assert derived.runtime.outcome is None
    assert provenance.get(plan.plan_id) is None


def test_observation_error_does_not_certify_handshake_timeout(runtime):
    controller, _, _, _, _, plan, _, _, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    observe(runtime, ObservationStatus.ERROR, 100)
    derived = controller.derive(plan.plan_id)
    assert derived.state == RunState.OBSERVATION_UNAVAILABLE
    assert derived.last_observation.status == ObservationStatus.RUNNING
    assert "handshake deadline" not in derived.reason


def test_backend_terminal_without_hello_remains_unknown(runtime):
    controller, _, _, _, _, plan, _, _, _ = runtime
    observe(runtime, ObservationStatus.TERMINAL, 100)
    assert controller.derive(plan.plan_id).state == RunState.TERMINAL_UNKNOWN


def test_actual_writer_hello_is_persisted_and_restored_without_changing_plan(runtime):
    controller, plans, executions, provenance, backend, plan, layout, _, lock = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    original = plans.get(plan.plan_id).plan.to_json()
    with JournalWriter(layout, attempt=1) as writer:
        event = writer.append("hello", hello().payload)
    until(controller, lambda: provenance.get(plan.plan_id) is not None)
    assert provenance.get(plan.plan_id).event == event
    assert controller.derive(plan.plan_id).runtime.hello_seen
    assert plans.get(plan.plan_id).plan.to_json() == original
    controller.close()
    with RuntimeProvenanceStore(provenance.path, dispatch_lock=lock) as reopened:
        restarted = SubmissionController(
            plans,
            executions,
            {"slurm": backend},
            lock,
            provenance_store=reopened,
        )
        try:
            assert restarted.derive(plan.plan_id).runtime.hello_seen
            assert (
                restarted.accept_runtime_events(plan.plan_id, [event]).status
                == NegotiationStatus.ACCEPTED
            )
        finally:
            restarted.close()


@pytest.mark.parametrize(
    "event",
    [
        replace(hello(), type="state", payload={"state": "running"}),
        replace(hello(), seq=1),
        hello(extension={"nested": [float("inf")]}),
    ],
)
def test_explicit_first_event_rejects_invalid_runtime_truth(runtime, event):
    controller, _, _, provenance, _, plan, _, _, _ = runtime
    result = controller.accept_runtime_events(plan.plan_id, [event])
    assert result.status == NegotiationStatus.FAILED
    assert controller.derive(plan.plan_id).state == RunState.FAILED
    assert provenance.get(plan.plan_id) is None


def test_stale_explicit_event_does_not_poison_current_attempt(runtime):
    controller, _, _, provenance, _, plan, _, _, _ = runtime
    with pytest.raises(StaleNegotiation):
        controller.accept_runtime_events(plan.plan_id, [replace(hello(), attempt=2)])
    assert provenance.get(plan.plan_id) is None
    assert (
        controller.accept_runtime_events(plan.plan_id, [hello()]).status
        == NegotiationStatus.ACCEPTED
    )


def test_identical_hello_is_idempotent_conflicting_hello_is_failure(runtime):
    controller, _, _, provenance, _, plan, _, _, _ = runtime
    controller.accept_runtime_events(plan.plan_id, [hello(), hello()])
    assert (
        controller.accept_runtime_events(
            plan.plan_id, [hello(runtime_version="other")]
        ).status
        == NegotiationStatus.FAILED
    )
    assert provenance.get(plan.plan_id).hello.runtime_version == "actual-runtime"
    assert controller.derive(plan.plan_id).state == RunState.FAILED


def test_store_failure_does_not_expose_accepted_hello(runtime, monkeypatch):
    controller, _, _, provenance, _, plan, _, _, _ = runtime

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(provenance, "record", fail)
    with pytest.raises(OSError, match="disk failure"):
        controller.accept_runtime_events(plan.plan_id, [hello()])
    assert not controller.derive(plan.plan_id).runtime.hello_seen
    assert (
        controller.handshake._runtime[plan.plan_id].result.status
        == NegotiationStatus.WAITING
    )


def test_malformed_first_journal_record_is_not_silently_skipped(runtime):
    controller, _, _, provenance, _, plan, layout, _, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    layout.journal(1).write_bytes(b"not JSON\n" + b"{}\n")
    until(controller, lambda: controller.derive(plan.plan_id).state == RunState.FAILED)
    assert provenance.get(plan.plan_id) is None


def test_huge_journal_only_reads_first_bounded_record(runtime):
    controller, _, _, provenance, _, plan, layout, _, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    with JournalWriter(layout, attempt=1) as writer:
        writer.append("hello", hello().payload)
    with layout.journal(1).open("ab") as stream:
        stream.write(b"x" * (2 * 1024 * 1024))
    until(controller, lambda: provenance.get(plan.plan_id) is not None)
    assert controller.derive(plan.plan_id).runtime.hello_seen


def test_fifo_probe_is_bounded_and_storage_failure_is_explicit(runtime):
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unavailable")
    controller, _, _, _, _, _, layout, _, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    os.mkfifo(layout.journal(1))
    with pytest.raises(StorageWorkError):
        until(controller, lambda: False)


def test_read_only_provenance_prevents_dispatch(runtime):
    controller, _, _, provenance, _, plan, _, _, lock = runtime
    with RuntimeProvenanceStore(
        provenance.path, dispatch_lock=lock, read_only=True
    ) as recovery:
        controller.context.provenance_store = recovery
        with pytest.raises(PermissionError, match="Read-only recovery"):
            controller.accept_runtime_events(plan.plan_id, [hello()])
        controller.context.provenance_store = provenance


def test_mismatched_provenance_ownership_prevents_dispatch(runtime):
    controller, _, _, provenance, _, plan, _, _, _ = runtime
    provenance.dispatch_lock = None
    from wavedl_daemon.storage.ownership import DispatchRequired

    with pytest.raises(DispatchRequired):
        controller.accept_runtime_events(plan.plan_id, [hello()])


def test_journal_worker_never_commits_sqlite(runtime, monkeypatch):
    controller, _, _, provenance, _, plan, layout, _, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    owner = threading.get_ident()
    commits = []
    original = provenance.record

    def record(*args):
        commits.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(provenance, "record", record)
    with JournalWriter(layout, attempt=1) as writer:
        writer.append("hello", hello().payload)
    until(controller, lambda: provenance.get(plan.plan_id) is not None)
    assert commits == [owner]
    assert controller.context.budget.max_workers == 1


def test_partial_first_record_waits_and_oversize_first_record_fails(runtime):
    controller, _, _, provenance, _, plan, layout, _, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    layout.journal(1).write_bytes(b'{"partial":')
    before = runtime[4].observations
    until(controller, lambda: runtime[4].observations > before + 2)
    assert controller.derive(plan.plan_id).state == RunState.RUNNING
    layout.journal(1).write_bytes(b"x" * 65537)
    until(controller, lambda: controller.derive(plan.plan_id).state == RunState.FAILED)
    assert "byte bound" in controller.derive(plan.plan_id).reason
    assert provenance.get(plan.plan_id) is None


def test_read_only_derivation_cannot_bypass_hello_acceptance(runtime):
    controller, _, _, provenance, _, plan, _, _, _ = runtime
    derived = controller.derive(plan.plan_id, [hello()])
    assert derived.state == RunState.FAILED
    assert not derived.runtime.hello_seen
    assert provenance.get(plan.plan_id) is None
    controller.accept_runtime_events(plan.plan_id, [hello()])
    invalid = replace(
        hello(),
        type="state",
        seq=1,
        payload={"state": "running", "extra": float("nan")},
    )
    derived = controller.derive(plan.plan_id, [invalid])
    assert derived.state == RunState.FAILED
    assert derived.runtime.hello_seen
    assert controller.derive(plan.plan_id).state != RunState.FAILED


def test_late_explicit_hello_cannot_evade_running_deadline(runtime):
    controller, _, _, provenance, _, plan, _, clock, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    clock[0] = 11
    assert (
        controller.accept_runtime_events(plan.plan_id, [hello()]).status
        == NegotiationStatus.FAILED
    )
    assert provenance.get(plan.plan_id) is None
    assert controller.derive(plan.plan_id).state == RunState.FAILED


def test_wrong_actual_interpreter_is_sticky_automatic_protocol_failure(runtime):
    controller, _, _, provenance, _, plan, layout, _, _ = runtime
    observe(runtime, ObservationStatus.RUNNING, 0)
    with JournalWriter(layout, attempt=1) as writer:
        writer.append(
            "hello",
            hello(
                interpreter=str((layout.run_dir / "unapproved-python").absolute())
            ).payload,
        )
    until(controller, lambda: controller.derive(plan.plan_id).state == RunState.FAILED)
    derived = controller.derive(plan.plan_id)
    assert "interpreter differs" in derived.reason
    assert derived.backend.status == ObservationStatus.RUNNING
    assert derived.runtime.outcome is None
    assert not derived.runtime.hello_seen
    assert provenance.get(plan.plan_id) is None
    for _ in range(5):
        controller.reconcile_once()
    assert plan.plan_id not in controller.handshake._runtime_work
    assert controller.derive(plan.plan_id).state == RunState.FAILED


def test_explicit_runtime_policy_failure_cannot_be_replaced_by_later_hello(runtime):
    controller, _, _, provenance, _, plan, _, _, _ = runtime
    rejected = controller.accept_runtime_events(plan.plan_id, [hello(device="cuda")])
    assert rejected.status == NegotiationStatus.FAILED
    assert (
        controller.accept_runtime_events(plan.plan_id, [hello()]).status
        == NegotiationStatus.FAILED
    )
    assert not controller.derive(plan.plan_id).runtime.hello_seen
    assert provenance.get(plan.plan_id) is None
