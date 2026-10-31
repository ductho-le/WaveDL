"""Controller activates writer generations only after durable native proof."""

from __future__ import annotations

import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.protocol import decode
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.executions import ExecutionState, ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


class WaitingLocal:
    """Native acceptance without any runtime writer allocation or proof."""

    owner = "controller-test"

    def __init__(self):
        self.status = ObservationStatus.QUEUED
        self.submissions = []
        self.cancellations = []

    def submit(self, plan, *, request_id, attempt):
        assert attempt is None
        self.submissions.append((plan, request_id))
        return SubmissionHandle(
            "local",
            plan.run_id,
            request_id,
            self.owner,
            plan.run_dir,
            pid=12345,
            supervisor_id="test-supervisor",
        )

    def observe(self, handle):
        return SubmissionObservation(self.status)

    def discover_initial(self, handle):
        return None

    def request_cancel(self, handle):
        self.cancellations.append(handle)
        self.status = ObservationStatus.TERMINAL


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import wavedl_daemon.runs.control.dispatch as controller_module

    def forbidden_allocation(*args, **kwargs):
        pytest.fail("daemon must not allocate a writer-mode runtime attempt")

    monkeypatch.setattr(controller_module, "allocate_attempt", forbidden_allocation)
    database = tmp_path / "registry.db"
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(database, dispatch_lock=lock),
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as provenance,
    ):
        controllers = []
        backends = []

        def create(backend=None):
            data = tmp_path / "data.npz"
            generator = np.random.default_rng(8)
            np.savez(
                data,
                X=generator.normal(size=(20, 32)).astype("float32"),
                y=generator.normal(size=(20, 1)).astype("float32"),
            )
            layout = RunLayout.create(tmp_path / "runs", str(uuid4()))
            manifest = build_manifest(
                layout,
                operation=OperationKind.TRAIN,
                resolved_config={
                    "data_path": str(data),
                    "model": "cnn",
                    "epochs": 1,
                    "workers": 0,
                    "batch_size": 8,
                    "precision": "no",
                },
                runtime=RuntimeSelection(
                    sys.executable,
                    "existing",
                    "cpu",
                    {
                        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                ),
                submission_cwd=tmp_path,
            )
            write_manifest(layout, manifest)
            planner = LocalBackend()
            plan = plans.put(
                replace(planner.plan(manifest), allocation_mode="writer"),
                idempotency_key=str(uuid4()),
                request_payload={},
            )
            if backend is None:
                backend = LocalBackend(dispatch_lock=lock, owner="controller-test")
                backends.append(backend)
            controller = SubmissionController(
                plans,
                executions,
                {"local": backend},
                lock,
                WorkBudget(max_workers=2, max_pending=2),
                provenance_store=provenance,
            )
            controllers.append(controller)
            return controller, plan, layout, backend

        yield create, plans, executions, provenance, lock
        for controller in controllers:
            if not controller._closed:
                controller.close()
        for backend in backends:
            backend.close()


def until(controller, predicate, *, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        controller.reconcile_once()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("controller made no bounded progress")


def binding_count(executions, plan_id):
    with sqlite3.connect(executions.path) as connection:
        return connection.execute(
            "SELECT count(*) FROM initial_bindings WHERE plan_id=?", (plan_id,)
        ).fetchone()[0]


def test_actual_cpu_writer_activation_and_hello_use_real_journal(runtime):
    pytest.importorskip("torch", reason="Actual Local CPU runtime requires ML extras")
    pytest.importorskip("accelerate", reason="Actual Local training requires ML extras")
    create, plans, executions, provenance, _ = runtime
    controller, plan, layout, _ = create()
    original = plans.get(plan.plan_id).plan.to_json()
    requested = controller.request_submit(plan.plan_id)
    assert requested.attempt is None and requested.handle is None
    assert not list(layout.attempts.iterdir())
    until(
        controller, lambda: executions.get(plan.plan_id).submission_handle is not None
    )
    native = executions.get(plan.plan_id)
    assert native.attempt is None and native.handle is None
    assert isinstance(native.submission_handle, SubmissionHandle)
    until(controller, lambda: executions.get(plan.plan_id).attempt is not None)
    active = executions.get(plan.plan_id)
    assert active.attempt == 1
    assert active.handle == native.submission_handle.bind_attempt(1)
    assert binding_count(executions, plan.plan_id) == 1
    until(controller, lambda: provenance.get(plan.plan_id, 1) is not None)
    until(
        controller,
        lambda: (
            executions.get(plan.plan_id).latest_observation is not None
            and executions.get(plan.plan_id).latest_observation.status
            == ObservationStatus.TERMINAL
        ),
    )
    events = [decode(line) for line in layout.journal(1).read_bytes().splitlines()]
    assert events[0].type == "hello" and events[0].payload["device"] == "cpu"
    assert provenance.get(plan.plan_id, 1).event == events[0]
    assert events[-1].payload["status"] == "completed"
    assert controller.derive(plan.plan_id, events).state == RunState.COMPLETED
    assert plans.get(plan.plan_id).plan.to_json() == original
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]


def test_queued_native_cancel_before_startup_never_creates_execution(runtime):
    create, _, executions, provenance, _ = runtime
    backend = WaitingLocal()
    controller, plan, layout, _ = create(backend)
    assert controller.request_submit(plan.plan_id).attempt is None
    until(
        controller,
        lambda: executions.get(plan.plan_id).submission_observation is not None,
    )
    assert controller.derive(plan.plan_id).state == RunState.QUEUED
    queued = executions.get(plan.plan_id)
    assert queued.attempt is None and binding_count(executions, plan.plan_id) == 0
    assert queued.handle is None
    controller.request_cancel(plan.plan_id)
    until(controller, lambda: executions.get(plan.plan_id).cancel_delivered)
    until(
        controller,
        lambda: (
            executions.get(plan.plan_id).latest_submission_observation.status
            == ObservationStatus.TERMINAL
        ),
    )
    cancelled = executions.get(plan.plan_id)
    assert cancelled.attempt is None and cancelled.handle is None
    assert backend.cancellations == [queued.submission_handle]
    assert len(backend.submissions) == 1
    assert not list(layout.attempts.iterdir())
    assert provenance.get(plan.plan_id) is None


def test_persisted_unbound_local_pid_is_not_adopted_or_resubmitted(runtime):
    create, plans, executions, provenance, lock = runtime
    controller, plan, layout, backend = create(WaitingLocal())
    controller.request_submit(plan.plan_id)
    until(
        controller, lambda: executions.get(plan.plan_id).submission_handle is not None
    )
    persisted = executions.get(plan.plan_id).submission_handle
    controller.close()
    replacement = LocalBackend(dispatch_lock=lock, owner=persisted.owner)
    restarted = SubmissionController(
        plans,
        executions,
        {"local": replacement},
        lock,
        provenance_store=provenance,
    )
    try:
        until(
            restarted,
            lambda: executions.get(plan.plan_id).state == ExecutionState.LOCAL_LOST,
        )
        lost = executions.get(plan.plan_id)
        assert lost.attempt is None and lost.handle is None
        assert lost.submission_handle == persisted
        assert restarted.derive(plan.plan_id).state == RunState.SUPERVISION_LOST
        assert len(backend.submissions) == 1
        assert not list(layout.attempts.iterdir())
    finally:
        restarted.close()
        replacement.close()


def test_invalid_startup_proof_preserves_native_queue_evidence(runtime):
    class InvalidProofLocal(WaitingLocal):
        def discover_initial(self, handle):
            raise ValueError("foreign supervisor startup proof")

    create, _, executions, provenance, _ = runtime
    backend = InvalidProofLocal()
    controller, plan, layout, _ = create(backend)
    controller.request_submit(plan.plan_id)
    until(
        controller,
        lambda: (
            executions.get(plan.plan_id).latest_submission_observation is not None
            and executions.get(plan.plan_id).latest_submission_observation.status
            == ObservationStatus.ERROR
        ),
    )
    record = executions.get(plan.plan_id)
    assert record.submission_observation.status == ObservationStatus.QUEUED
    assert "foreign supervisor" in record.latest_submission_observation.detail
    assert controller.derive(plan.plan_id).state == RunState.OBSERVATION_UNAVAILABLE
    assert record.attempt is None and record.handle is None
    assert binding_count(executions, plan.plan_id) == 0
    assert provenance.get(plan.plan_id) is None
    assert len(backend.submissions) == 1
    assert not list(layout.attempts.iterdir())


def test_terminal_unbound_local_releases_native_receipt_once_and_never_retries(runtime):
    class CompletedLocal(WaitingLocal):
        def __init__(self):
            super().__init__()
            self.discoveries = 0
            self.releases = []

        def observe(self, handle):
            return SubmissionObservation(ObservationStatus.TERMINAL, exit_code=2)

        def discover_initial(self, handle):
            self.discoveries += 1
            return None

        def release(self, handle):
            self.releases.append(handle)

    create, _, executions, provenance, _ = runtime
    backend = CompletedLocal()
    controller, plan, layout, _ = create(backend)
    controller.request_submit(plan.plan_id)
    until(controller, lambda: bool(backend.releases))
    record = executions.get(plan.plan_id)
    assert record.submission_observation.status == ObservationStatus.TERMINAL
    assert record.submission_observation.exit_code == 2
    assert record.attempt is None and record.handle is None
    assert backend.releases == [record.submission_handle]
    derived = controller.derive(plan.plan_id)
    assert derived.state == RunState.FAILED
    assert derived.runtime.outcome is None
    revision = record.revision
    for _ in range(10):
        controller.reconcile_once()
    assert backend.discoveries == 1
    assert len(backend.submissions) == 1
    assert backend.releases == [record.submission_handle]
    assert executions.get(plan.plan_id).revision == revision
    assert binding_count(executions, plan.plan_id) == 0
    assert provenance.get(plan.plan_id) is None
    assert not list(layout.attempts.iterdir())
