"""Writer controller fleet polling binds only native-verified core claims."""

import threading
import time
from uuid import uuid4

import pytest

from unit_tests.daemon.test_slurm_writer_submission import Runner, control, identity
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.scheduler_attempt import coordinate_attempt
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    OperationOutcome,
    OperationResult,
    OutcomeStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.slurm_observation import SlurmObservation
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite
from wavedl_daemon.backends.slurm_submission import SlurmSubmission
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.runs.control.records import initial_hint
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.registry import Registry


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeSlurm:
    owner = "alice"

    def __init__(self, lock):
        self.submitter = SlurmSubmission(
            Runner(sbatch="42;alpha\n"), owner=self.owner, dispatch_lock=lock
        )
        self.status = ObservationStatus.QUEUED
        self.batches = []
        self.discoveries = []
        self.cancellations = []
        self.collections = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.receipt = None

    def submit(self, plan, *, request_id, attempt):
        assert attempt is None
        self.receipt = self.submitter.submit(
            plan, request_id=request_id, attempt=attempt
        )
        return self.receipt

    def observe(self, handle):
        pytest.fail("unchanged unbound scheduler jobs must use fleet observation")

    def observe_many(self, handles):
        self.batches.append(handles)
        status = self.status
        if self.block:
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("test fleet barrier timed out")
        return {
            handle.job_id: (
                SubmissionObservation(status)
                if isinstance(handle, SubmissionHandle)
                else BackendObservation(handle.attempt, status)
            )
            for handle in handles
        }

    def discover_initial(self, handle):
        self.discoveries.append(handle)
        observer = SlurmObservation(Runner(scontrol=control(handle)), owner=self.owner)
        return observer.discover_initial(handle)

    def request_cancel(self, handle):
        self.cancellations.append(handle)

    def collect(self, handle):
        self.collections.append(handle)
        return OperationResult(
            OperationKind.TRAIN,
            OperationOutcome(OutcomeStatus.UNKNOWN),
            OperationOutcome(OutcomeStatus.FAILED, exit_code=7),
            OperationOutcome(OutcomeStatus.FAILED, exit_code=9),
        )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import wavedl_daemon.runs.control.dispatch as module

    def forbidden(*args, **kwargs):
        pytest.fail("daemon allocated a writer attempt")

    monkeypatch.setattr(module, "allocate_attempt", forbidden)
    database = tmp_path / "registry.db"
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(database, dispatch_lock=lock),
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
        CollectionStore(database, dispatch_lock=lock) as collections,
    ):
        layout = RunLayout.create(tmp_path / "runs", str(uuid4()))
        manifest = build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={"epochs": 1, "model": "mlp", "data_path": "train.npz"},
            runtime=RuntimeSelection("/usr/bin/python3", "selected", "cpu", {}),
            submission_cwd=tmp_path,
        )
        plan = plans.put(
            SlurmPlanner(SlurmSite(partition="cpu", cluster="alpha")).plan(
                manifest, allocation_mode="writer"
            ),
            idempotency_key=str(uuid4()),
            request_payload={},
        )
        clock, backend = Clock(), FakeSlurm(lock)
        controller = SubmissionController(
            plans,
            executions,
            {"slurm": backend},
            lock,
            WorkBudget(max_workers=4, max_pending=4),
            collection_store=collections,
            clock=clock,
        )
        controller.fleet._fleets["slurm"].clock = clock
        try:
            yield controller, plan, layout, backend, executions, collections, clock
        finally:
            backend.release.set()
            controller.close()


def until(controller, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        controller.reconcile_once()
        if predicate():
            return
        time.sleep(0.002)
    pytest.fail("controller did not make bounded progress")


def queued(runtime):
    controller, plan, _, _, executions, _, _ = runtime
    assert controller.request_submit(plan.plan_id).attempt is None
    until(
        controller,
        lambda: executions.get(plan.plan_id).submission_observation is not None,
    )
    record = executions.get(plan.plan_id)
    assert record.submission_observation.status == ObservationStatus.QUEUED
    assert record.attempt is None and record.handle is None
    return record


def test_queued_fleet_then_real_core_claim_positive_activation(runtime):
    controller, plan, layout, backend, executions, _, clock = runtime
    native = queued(runtime)
    assert controller.derive(plan.plan_id).state == RunState.QUEUED
    assert backend.batches == [(native.submission_handle,)]
    for _ in range(10):
        controller.reconcile_once()
    assert backend.discoveries == []
    assert list(layout.attempts.iterdir()) == []
    proof = coordinate_attempt(
        layout.run_dir, identity(native.submission_handle), initial_attempt=None
    )
    assert proof.attempt == 1
    assert initial_hint(native.submission_handle)
    clock.now = 5
    until(controller, lambda: executions.get(plan.plan_id).attempt == proof.attempt)
    active = executions.get(plan.plan_id)
    assert active.handle == native.submission_handle.bind_attempt(1)
    assert active.submission_handle == native.submission_handle
    assert (
        executions._connection.execute(
            "SELECT count(*) FROM initial_bindings WHERE plan_id=?", (plan.plan_id,)
        ).fetchone()[0]
        == 1
    )
    assert backend.discoveries == [native.submission_handle]


def test_stale_unbound_fleet_result_is_fenced_after_native_activation(runtime):
    controller, plan, layout, backend, executions, _, clock = runtime
    native = queued(runtime)
    backend.block = True
    backend.status = ObservationStatus.HELD
    clock.now = 30
    until(controller, backend.started.is_set)
    proof = coordinate_attempt(
        layout.run_dir, identity(native.submission_handle), initial_attempt=None
    )
    # Initial verification runs in another bounded worker while fleet IO waits.
    clock.now = 35
    until(controller, lambda: executions.get(plan.plan_id).attempt == proof.attempt)
    active = executions.get(plan.plan_id)
    backend.block = False
    backend.release.set()
    until(
        controller,
        lambda: (
            "slurm" not in controller.fleet._fleet_work
            or all(
                record.attempt is not None
                for record in controller.fleet._fleet_work["slurm"][1].values()
            )
        ),
    )
    current = executions.get(plan.plan_id)
    assert current.attempt == 1 and current.handle == active.handle
    assert isinstance(current.latest_observation, BackendObservation)
    assert current.latest_submission_observation == active.latest_submission_observation
    assert current.latest_submission_observation.status == ObservationStatus.QUEUED


def test_native_cancel_delivered_before_binding_and_preserved_by_activation(runtime):
    controller, plan, layout, backend, executions, _, clock = runtime
    native = queued(runtime)
    controller.request_cancel(plan.plan_id)
    until(controller, lambda: executions.get(plan.plan_id).cancel_delivered)
    assert backend.cancellations == [native.submission_handle]
    assert executions.get(plan.plan_id).attempt is None
    proof = coordinate_attempt(
        layout.run_dir, identity(native.submission_handle), initial_attempt=None
    )
    clock.now = 5
    until(controller, lambda: executions.get(plan.plan_id).attempt == proof.attempt)
    active = executions.get(plan.plan_id)
    assert active.cancellation_requested and active.cancel_delivered
    assert backend.cancellations == [native.submission_handle]


def test_native_setup_failure_collects_independent_outcomes_without_attempt(runtime):
    controller, plan, layout, backend, executions, collections, clock = runtime
    native = queued(runtime)
    backend.status = ObservationStatus.TERMINAL
    clock.now = 30
    until(controller, lambda: collections.get_submission(plan.plan_id) is not None)
    collected = collections.get_submission(plan.plan_id)
    assert collected.handle == native.submission_handle
    assert collected.result.runtime.status == OutcomeStatus.UNKNOWN
    assert collected.result.batch.exit_code == 7
    assert collected.result.stage_out.exit_code == 9
    assert collections.get(plan.plan_id) is None
    assert executions.get(plan.plan_id).attempt is None
    assert list(layout.attempts.iterdir()) == []
    assert collections._connection.execute("PRAGMA user_version").fetchone()[0] == 13
