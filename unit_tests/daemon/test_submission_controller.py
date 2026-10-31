"""Crash boundaries use real SQLite; backend effects run outside its thread."""

import importlib.util
import json
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    Plan,
)
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.backends.slurm_submission import SubmissionRejected
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.executions import (
    EXECUTION_MIGRATION,
    ExecutionState,
    ExecutionStore,
)
from wavedl_daemon.storage.ownership import DispatchRequired, acquire_dispatch_lock
from wavedl_daemon.storage.plans import PLAN_MIGRATION, PlanStore
from wavedl_daemon.storage.schema import migrate


class FakeSlurm:
    owner = "test-user"

    def __init__(self):
        self.jobs = []
        self.cancel_count = 0
        self.lookup_error = None
        self.gate = None
        self.cancel_lost = False

    def submit(self, plan, *, request_id, attempt):
        handle = Handle(
            "slurm",
            plan.run_id,
            request_id,
            attempt,
            self.owner,
            plan.run_dir,
            job_id="1",
        )
        self.jobs.append(handle)
        if self.gate:
            self.gate.wait(2)
        return handle

    def find_by_request_id(self, plan, *, request_id, attempt):
        if self.lookup_error:
            raise self.lookup_error
        return self.jobs[0] if self.jobs else None

    def request_cancel(self, handle):
        self.cancel_count += 1
        if self.cancel_lost:
            self.cancel_lost = False
            raise TimeoutError("accepted cancellation response lost")

    def observe(self, handle):
        return BackendObservation(handle.attempt, ObservationStatus.RUNNING)


@pytest.fixture
def environment(tmp_path):
    database = tmp_path / "registry.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=1")
    migrate(database, [PLAN_MIGRATION, EXECUTION_MIGRATION])
    layout = RunLayout(
        tmp_path / "runs" / "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000001",
    )
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    plan = Plan("slurm", json.dumps(asdict(manifest)), script="#!/bin/sh\nexit 0\n")
    backend = FakeSlurm()
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
    ):
        record = plans.put(plan, idempotency_key="one", request_payload={})
        controller = SubmissionController(
            plans,
            executions,
            {"slurm": backend},
            lock,
            WorkBudget(max_workers=1, max_pending=0),
        )
        yield controller, plans, executions, backend, record, lock
        controller.close()


def tick_until(controller, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        controller.reconcile_once()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("controller failed to make bounded progress")


def reopen(controller, plans, executions, backend, lock):
    controller.close()
    executions.close()
    replacement = ExecutionStore(executions.path, dispatch_lock=lock)
    return replacement, SubmissionController(
        plans, replacement, {"slurm": backend}, lock
    )


def test_supplied_backend_owner_does_not_lookup_login(environment, monkeypatch):
    controller, _, executions, backend, record, _ = environment

    def missing_login():
        raise KeyError("UID has no passwd entry")

    monkeypatch.setattr(
        "wavedl_daemon.runs.control.dispatch.getpass.getuser", missing_login
    )
    submitted = controller.request_submit(record.plan_id)
    assert submitted.owner == backend.owner
    assert executions.get(record.plan_id).owner == backend.owner


def test_missing_backend_owner_uses_login(environment, monkeypatch):
    controller, _, _, backend, record, _ = environment
    monkeypatch.delattr(type(backend), "owner")
    monkeypatch.setattr(
        "wavedl_daemon.runs.control.dispatch.getpass.getuser", lambda: "current-login"
    )
    assert controller.dispatch.ensure(record).owner == "current-login"


def test_missing_backend_owner_preserves_login_failure(environment, monkeypatch):
    controller, _, _, backend, record, _ = environment
    monkeypatch.delattr(type(backend), "owner")

    def missing_login():
        raise KeyError("UID has no passwd entry")

    monkeypatch.setattr(
        "wavedl_daemon.runs.control.dispatch.getpass.getuser", missing_login
    )
    with pytest.raises(KeyError, match="UID has no passwd entry"):
        controller.dispatch.ensure(record)


def test_accept_before_receipt_crash_recovers_one_job(environment, monkeypatch):
    controller, plans, executions, backend, record, lock = environment
    controller.request_submit(record.plan_id)
    original = executions.record_handle
    monkeypatch.setattr(
        executions,
        "record_handle",
        lambda *args: (_ for _ in ()).throw(RuntimeError("power loss")),
    )
    with pytest.raises(RuntimeError, match="power loss"):
        tick_until(controller, lambda: False)
    monkeypatch.setattr(executions, "record_handle", original)
    replacement, restarted = reopen(controller, plans, executions, backend, lock)
    try:
        tick_until(
            restarted, lambda: replacement.get(record.plan_id).handle is not None
        )
        assert len(backend.jobs) == 1
        assert replacement.get(record.plan_id).attempt == 1
    finally:
        restarted.close()
        replacement.close()


@pytest.mark.parametrize(
    "error", [None, TimeoutError("query unavailable"), ValueError("ambiguous identity")]
)
def test_claim_crash_never_resubmits_on_missing_or_failed_search(environment, error):
    controller, plans, executions, backend, record, lock = environment
    requested = plans.request_submission(record.plan_id)
    controller.dispatch.ensure(requested)
    executions.claim_submission(record.plan_id)
    backend.lookup_error = error
    replacement, restarted = reopen(controller, plans, executions, backend, lock)
    try:
        tick_until(restarted, lambda: replacement.get(record.plan_id).revision > 1)
        assert replacement.get(record.plan_id).state == ExecutionState.UNKNOWN
        assert backend.jobs == []
        assert "unknown" in restarted.derive(record.plan_id).state
    finally:
        restarted.close()
        replacement.close()


def test_cancel_before_handle_survives_lost_cancel_response(environment):
    controller, _plans, executions, backend, record, _lock = environment
    backend.gate = threading.Event()
    backend.cancel_lost = True
    controller.request_submit(record.plan_id)
    controller.request_cancel(record.plan_id)
    assert executions.get(record.plan_id).cancellation_requested
    backend.gate.set()
    tick_until(controller, lambda: executions.get(record.plan_id).cancel_delivered)
    assert backend.cancel_count == 2
    assert controller.derive(record.plan_id).state != RunState.CANCELLED
    assert executions.get(record.plan_id).cancellation_requested


def test_durable_pending_request_is_submitted_on_startup_once(environment):
    controller, plans, executions, backend, record, _lock = environment
    plans.request_submission(record.plan_id)
    controller.startup()
    tick_until(controller, lambda: executions.get(record.plan_id).handle is not None)
    controller.request_submit(record.plan_id)
    assert len(backend.jobs) == 1


def test_owner_lock_release_blocks_all_external_actions(environment):
    controller, plans, _executions, backend, record, lock = environment
    lock.release()
    with pytest.raises(DispatchRequired):
        controller.request_submit(record.plan_id)
    assert not plans.get(record.plan_id).submission_requested
    assert backend.jobs == []


def test_wrapper_zero_does_not_complete_training(environment):
    controller, _plans, executions, backend, record, _lock = environment
    backend.observe = lambda handle: BackendObservation(
        handle.attempt, ObservationStatus.TERMINAL, exit_code=0
    )
    controller.request_submit(record.plan_id)
    tick_until(
        controller,
        lambda: executions.get(record.plan_id).latest_observation is not None,
    )
    assert controller.derive(record.plan_id).state == RunState.TERMINAL_UNKNOWN


def test_controller_workers_do_not_receive_sqlite_connections(environment):
    controller, _plans, executions, backend, record, _lock = environment
    owner_thread = threading.get_ident()
    original = backend.submit

    def submit(*args, **kwargs):
        assert threading.get_ident() != owner_thread
        return original(*args, **kwargs)

    backend.submit = submit
    controller.request_submit(record.plan_id)
    tick_until(controller, lambda: executions.get(record.plan_id).handle is not None)


def test_local_lost_supervision_never_adopts_or_restarts(environment):
    controller, plans, executions, backend, record, _lock = environment
    local_plan = replace(
        record.plan, target="local", script=None, argv=(sys.executable, "-c", "pass")
    )
    local = plans.put(
        local_plan, idempotency_key="local", request_payload={"local": True}
    )
    controller.backends["local"] = backend
    requested = plans.request_submission(local.plan_id)
    controller.dispatch.ensure(requested)
    executions.claim_submission(local.plan_id)
    controller.reconcile_once()
    assert executions.get(local.plan_id).state == ExecutionState.LOCAL_LOST
    assert controller.derive(local.plan_id).state == RunState.SUPERVISION_LOST
    assert backend.jobs == []


@pytest.mark.integration
def test_actual_cpu_local_subprocess_and_restart_loss(environment):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Actual CPU probe requires installed torch")
    controller, plans, executions, _backend, record, lock = environment

    class SmallPythonBackend(LocalBackend):
        # A small real runtime isolates controller/supervisor integration from
        # training datasets. The real CPU probe and real process remain intact.
        def plan(self, manifest, **_kwargs):
            return Plan(
                "local",
                json.dumps(asdict(manifest)),
                argv=(
                    sys.executable,
                    "-c",
                    "import torch; print(torch.ones(1).sum().item())",
                ),
            )

    backend = SmallPythonBackend(dispatch_lock=lock, probe_timeout=10)
    plan = backend.plan(record.plan.manifest)
    local = plans.put(plan, idempotency_key="cpu", request_payload={"cpu": True})
    controller.backends["local"] = backend
    try:
        controller.request_submit(local.plan_id)
        tick_until(
            controller,
            lambda: (
                executions.get(local.plan_id).latest_observation is not None
                and executions.get(local.plan_id).latest_observation.status
                == ObservationStatus.TERMINAL
            ),
        )
        receipt = executions.get(local.plan_id).handle
        assert receipt.pid > 0
        assert receipt.owner == backend.owner
        assert executions.get(local.plan_id).latest_observation.exit_code == 0
        assert controller.derive(local.plan_id).state == RunState.TERMINAL_UNKNOWN
        # Durable evidence remains authoritative after bounded live diagnostics
        # are released; losing an in-memory handle does not erase completion.
        assert backend.observe(receipt).status == ObservationStatus.SUPERVISION_LOST
        replacement = SmallPythonBackend(dispatch_lock=lock)
        try:
            assert (
                replacement.observe(receipt).status
                == ObservationStatus.SUPERVISION_LOST
            )
        finally:
            replacement.close()
    finally:
        backend.close()


def test_bounded_pages_are_fair_and_do_not_submit_cancelled_pending(environment):
    controller, plans, executions, backend, record, _lock = environment
    plans.request_cancellation(record.plan_id)
    ids = [record.plan_id]
    for number in (2, 3):
        manifest = record.plan.manifest
        run_id = f"00000000-0000-4000-8000-{number:012d}"
        run_dir = str(Path(manifest.run_directory).parent / run_id)
        updated = replace(
            manifest,
            run_id=run_id,
            run_directory=run_dir,
            working_directory=run_dir,
            resolved_config={"output_dir": run_dir},
        )
        plan = replace(record.plan, manifest_json=json.dumps(asdict(updated)))
        other = plans.put(
            plan, idempotency_key=str(number), request_payload={"number": number}
        )
        plans.request_submission(other.plan_id)
        ids.append(other.plan_id)
    assert len(controller.startup()) == 1
    tick_until(
        controller,
        lambda: all(
            executions.get(plan_id) is not None
            and executions.get(plan_id).handle is not None
            for plan_id in ids[1:]
        ),
    )
    assert len(backend.jobs) == 2
    assert executions.get(ids[0]).state == ExecutionState.PENDING


def test_late_successful_receipt_survives_pool_deadline(environment):
    previous, plans, executions, backend, record, lock = environment
    previous.close()
    controller = SubmissionController(
        plans,
        executions,
        {"slurm": backend},
        lock,
        WorkBudget(max_workers=1, max_pending=0, timeout_seconds=0.02),
    )
    original = backend.submit

    def delayed(*args, **kwargs):
        handle = original(*args, **kwargs)
        time.sleep(0.05)
        return handle

    backend.submit = delayed
    try:
        controller.request_submit(record.plan_id)
        controller.request_cancel(record.plan_id)
        tick_until(controller, lambda: executions.get(record.plan_id).cancel_delivered)
        assert executions.get(record.plan_id).state == ExecutionState.SUBMITTED
        assert len(backend.jobs) == 1
        assert backend.cancel_count == 1
    finally:
        controller.close()


def test_definite_rejection_is_failed_not_unknown_acceptance(environment):
    controller, _plans, executions, backend, record, _lock = environment

    def rejected(*_args, **_kwargs):
        raise SubmissionRejected("scheduler refused before acceptance")

    backend.submit = rejected
    controller.request_submit(record.plan_id)
    tick_until(
        controller,
        lambda: executions.get(record.plan_id).state == ExecutionState.REJECTED,
    )
    assert controller.derive(record.plan_id).state == RunState.FAILED
    assert executions.get(record.plan_id).handle is None
    assert backend.jobs == []


def test_slurm_fleet_tick_never_uses_individual_observe(environment):
    controller, _plans, executions, backend, record, _lock = environment
    from wavedl_daemon.runs.reconciliation import SchedulerReconciler

    queried = []

    def observe_many(handles):
        queried.append(handles)
        return {
            handle.job_id: BackendObservation(handle.attempt, ObservationStatus.RUNNING)
            for handle in handles
        }

    def refused_individual(_handle):
        pytest.fail("controller issued per-run Slurm query")

    backend.observe_many = observe_many
    backend.observe = refused_individual
    controller.fleet._fleets["slurm"] = SchedulerReconciler(backend)
    controller.request_submit(record.plan_id)
    tick_until(
        controller,
        lambda: executions.get(record.plan_id).latest_observation is not None,
    )
    assert len(queried) == 1
    assert controller.derive(record.plan_id).state == RunState.RUNNING


def test_lost_cancel_ack_is_observed_before_retry_and_terminal_stops_retry(environment):
    controller, _plans, executions, backend, record, _lock = environment
    backend.cancel_lost = True
    backend.observe = lambda handle: BackendObservation(
        handle.attempt, ObservationStatus.TERMINAL, native_state="CANCELLED"
    )
    controller.request_submit(record.plan_id)
    controller.request_cancel(record.plan_id)
    tick_until(
        controller,
        lambda: (
            executions.get(record.plan_id).latest_observation is not None
            and executions.get(record.plan_id).latest_observation.status
            == ObservationStatus.TERMINAL
        ),
    )
    for _ in range(5):
        controller.reconcile_once()
    assert backend.cancel_count == 1
    assert not executions.get(record.plan_id).cancel_delivered
    assert controller.derive(record.plan_id).state == RunState.CANCELLED


def test_receipt_persistence_failure_replays_same_response_without_resubmit(
    environment, monkeypatch
):
    controller, _plans, executions, backend, record, _lock = environment
    original = executions.record_handle

    def failed(*_args, **_kwargs):
        raise OSError("SQLite disk full")

    monkeypatch.setattr(executions, "record_handle", failed)
    controller.request_submit(record.plan_id)
    with pytest.raises(OSError, match="disk full"):
        tick_until(controller, lambda: False)
    monkeypatch.setattr(executions, "record_handle", original)
    tick_until(controller, lambda: executions.get(record.plan_id).handle is not None)
    assert len(backend.jobs) == 1


def test_arbitrary_run_directory_is_published_at_exact_manifest_path(environment):
    controller, plans, executions, backend, record, _lock = environment
    original = record.plan.manifest
    path = Path(original.run_directory).parent / "custom-run-directory"
    manifest = replace(
        original,
        run_directory=str(path),
        working_directory=str(path),
        resolved_config={"output_dir": str(path)},
    )
    plan = replace(record.plan, manifest_json=json.dumps(asdict(manifest)))
    custom = plans.put(plan, idempotency_key="custom", request_payload={"custom": True})
    controller.request_submit(custom.plan_id)
    tick_until(controller, lambda: executions.get(custom.plan_id).handle is not None)
    assert (path / "manifest.json").is_file()
    assert (path / "attempts" / "1").is_dir()
    assert len(backend.jobs) == 1


def test_multiple_pages_preserve_fleet_interval_and_backoff(environment):
    controller, plans, executions, backend, record, _lock = environment
    from wavedl_daemon.runs.reconciliation import SchedulerReconciler

    calls = []
    now = [0.0]

    def observe_many(handles):
        calls.append(handles)
        return {
            handle.job_id: BackendObservation(
                handle.attempt, ObservationStatus.ERROR, detail="scheduler outage"
            )
            for handle in handles
        }

    backend.observe_many = observe_many
    controller.fleet._fleets["slurm"] = SchedulerReconciler(
        backend, clock=lambda: now[0]
    )
    ids = [record.plan_id]
    manifest = record.plan.manifest
    run_id = "00000000-0000-4000-8000-000000000002"
    run_dir = str(Path(manifest.run_directory).parent / run_id)
    other_manifest = replace(
        manifest,
        run_id=run_id,
        run_directory=run_dir,
        working_directory=run_dir,
        resolved_config={"output_dir": run_dir},
    )
    other = plans.put(
        replace(record.plan, manifest_json=json.dumps(asdict(other_manifest))),
        idempotency_key="other-page",
        request_payload={"other": True},
    )
    ids.append(other.plan_id)
    for plan_id in ids:
        plans.request_submission(plan_id)
    tick_until(
        controller,
        lambda: all(
            executions.get(plan_id) is not None
            and executions.get(plan_id).latest_observation is not None
            for plan_id in ids
        ),
    )
    first_count = len(calls)
    for _ in range(20):
        controller.reconcile_once()
        time.sleep(0.003)
    assert len(calls) == first_count
    assert len(controller.fleet._fleet_active["slurm"]) == 2
    now[0] = 6
    tick_until(controller, lambda: len(calls) > first_count)
    assert len(controller.fleet._fleet_active["slurm"]) == 2


def test_controller_capability_probe_applies_short_min_job_age(environment):
    controller, _plans, _executions, backend, record, _lock = environment
    from wavedl_daemon.backends.slurm_observation import SlurmCapabilities
    from wavedl_daemon.runs.reconciliation import SchedulerReconciler

    called = []
    owner_thread = threading.get_ident()

    def capabilities():
        assert threading.get_ident() != owner_thread
        called.append(True)
        return SlurmCapabilities(accounting=False, min_job_age=5)

    def observe_many(handles):
        return {
            handle.job_id: BackendObservation(handle.attempt, ObservationStatus.RUNNING)
            for handle in handles
        }

    backend.capabilities = capabilities
    backend.observe_many = observe_many
    controller.fleet._fleets["slurm"] = SchedulerReconciler(backend)
    controller.request_submit(record.plan_id)
    tick_until(
        controller,
        lambda: (
            bool(called)
            and controller.fleet._fleets["slurm"].capabilities.min_job_age == 5
        ),
    )
    assert controller.fleet._fleets["slurm"]._interval() < 5
    assert controller.capability_errors == {}
