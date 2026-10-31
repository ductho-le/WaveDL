"""Native pre-writer requeues remain durable without invented runtime attempts."""

import hashlib
import json
import threading
import time
from dataclasses import replace
from uuid import uuid4

import pytest

from unit_tests.daemon.test_slurm_writer_submission import Runner, control, identity
from unit_tests.daemon.test_writer_fleet_controller import Clock, until
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.scheduler_attempt import coordinate_attempt
from wavedl.core.submission_outcomes import write_submission_outcome_receipt
from wavedl_daemon.backends.contracts import ObservationStatus, OutcomeStatus
from wavedl_daemon.backends.slurm_observation import SlurmObservation
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite
from wavedl_daemon.backends.slurm_submission import SlurmSubmission
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.settings import Settings


class NativeBackend:
    owner = "alice"

    def __init__(self, lock):
        self.runner = Runner(sbatch="42;alpha\n")
        self.submitter = SlurmSubmission(
            self.runner, owner=self.owner, dispatch_lock=lock
        )
        self.observer = SlurmObservation(self.runner, owner=self.owner)
        self.handle = None
        self.batches = []
        self.collects = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def submit(self, plan, *, request_id, attempt):
        self.handle = self.submitter.submit(
            plan, request_id=request_id, attempt=attempt
        )
        self.native(0, "PENDING")
        return self.handle

    def native(self, generation, state, exit_code="0:0"):
        self.runner.responses["scontrol"] = control(
            self.handle, Restarts=str(generation), JobState=state, ExitCode=exit_code
        )

    def observe(self, handle):
        return self.observer.observe(handle)

    def observe_many(self, handles):
        self.batches.append(handles)
        result = self.observer.observe_many(handles)
        if self.block:
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("fleet barrier timeout")
        return result

    def discover_initial(self, handle):
        return self.observer.discover_initial(handle)

    def collect_submission(self, handle):
        self.collects.append(handle)
        return self.observer.collect_submission(handle)


@pytest.fixture
def native_runtime(tmp_path, monkeypatch):
    import wavedl_daemon.runs.control.dispatch as module

    def forbidden(*args, **kwargs):
        pytest.fail("daemon allocated an unproved writer attempt")

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
            SlurmPlanner(
                SlurmSite(partition="cpu", cluster="alpha", requeue=True)
            ).plan(manifest, allocation_mode="writer"),
            idempotency_key=str(uuid4()),
            request_payload={},
        )
        clock, backend = Clock(), NativeBackend(lock)
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
            controller.request_submit(plan.plan_id)
            until(
                controller,
                lambda: executions.get(plan.plan_id).submission_observation is not None,
            )
            yield controller, plan, layout, backend, executions, collections, clock
        finally:
            backend.release.set()
            controller.close()


def outcome(runtime, generation, *, batch=9, stage=0):
    _, plan, layout, backend, _, _, _ = runtime
    return write_submission_outcome_receipt(
        layout.run_dir,
        request_id=backend.handle.request_id,
        owner="alice",
        plan_sha256=hashlib.sha256(plan.plan.to_json().encode()).hexdigest(),
        job_id="42",
        cluster="alpha",
        restart_count=generation,
        operation="train",
        runtime_exit=125,
        batch_exit=batch,
        stage_out_exit=stage,
    )


def test_ack0_requeue2_setup_failure_persists_exact_native_collection(native_runtime):
    controller, plan, layout, backend, executions, collections, clock = native_runtime
    initial = executions.get(plan.plan_id)
    assert initial.submission_observation.native_identity == identity(backend.handle)
    outcome(native_runtime, 2, batch=9, stage=7)
    backend.native(2, "FAILED", "9:0")
    clock.now = 30
    expected = identity(backend.handle, 2)
    until(
        controller,
        lambda: (
            collections.get_submission_generation(plan.plan_id, expected) is not None
        ),
    )
    current = executions.get(plan.plan_id)
    saved = collections.get_submission_generation(plan.plan_id, expected)
    assert current.attempt is None and current.handle is None
    assert current.submission_handle == initial.submission_handle == backend.handle
    assert json.loads(current.submission_handle.raw_json)["restart_count"] == 0
    assert current.submission_observation.native_identity == expected
    assert saved.identity == expected
    assert saved.handle == backend.handle
    assert (
        saved.proposal.plan_sha256
        == hashlib.sha256(plan.plan.to_json().encode()).hexdigest()
    )
    assert len(saved.proposal.manifest_sha256) == 64
    assert saved.result.runtime.status == OutcomeStatus.UNKNOWN
    assert saved.result.batch.exit_code == 9
    assert saved.result.stage_out.exit_code == 7
    assert (
        collections.get_submission_generation(plan.plan_id, identity(backend.handle))
        is None
    )
    assert collections.get(plan.plan_id) is None
    assert list(layout.attempts.iterdir()) == []
    assert collections._connection.execute("PRAGMA user_version").fetchone()[0] == 13


def test_terminal_generation0_cache_does_not_suppress_generation2(native_runtime):
    controller, plan, layout, backend, executions, collections, clock = native_runtime
    outcome(native_runtime, 0, batch=3, stage=4)
    backend.native(0, "FAILED", "3:0")
    clock.now = 30
    until(
        controller,
        lambda: (
            collections.get_submission_generation(
                plan.plan_id, identity(backend.handle)
            )
            is not None
        ),
    )
    zero = collections.get_submission_generation(plan.plan_id, identity(backend.handle))
    outcome(native_runtime, 2, batch=9, stage=7)
    backend.native(2, "FAILED", "9:0")
    clock.now = 60
    until(
        controller,
        lambda: (
            collections.get_submission_generation(
                plan.plan_id, identity(backend.handle, 2)
            )
            is not None
        ),
    )
    two = collections.get_submission_generation(
        plan.plan_id, identity(backend.handle, 2)
    )
    assert collections.get_submission_generation(plan.plan_id, zero.identity) == zero
    assert two.result.batch.exit_code == 9 and two.result.stage_out.exit_code == 7
    assert executions.get(plan.plan_id).submission_handle == backend.handle
    assert list(layout.attempts.iterdir()) == []


def test_late_generation0_fleet_result_cannot_overwrite_generation2(native_runtime):
    controller, plan, _, backend, executions, _, clock = native_runtime
    backend.native(0, "PENDING")
    backend.block = True
    clock.now = 30
    until(controller, backend.started.is_set)
    old_work = controller.fleet._fleet_work["slurm"][0]
    backend.native(2, "FAILED", "9:0")
    current = executions.record_observation(
        plan.plan_id, backend.observe(backend.handle)
    )
    backend.block = False
    backend.release.set()
    until(
        controller,
        lambda: (
            "slurm" not in controller.fleet._fleet_work
            or controller.fleet._fleet_work["slurm"][0] is not old_work
        ),
    )
    saved = executions.get(plan.plan_id)
    assert saved.submission_observation.native_identity == identity(backend.handle, 2)
    assert saved.submission_observation == current.submission_observation
    assert saved.attempt is None


def test_generation0_outcome_receipt_is_not_merged_into_generation2(native_runtime):
    controller, plan, layout, backend, executions, collections, clock = native_runtime
    outcome(native_runtime, 0, batch=3, stage=4)
    backend.native(2, "FAILED", "9:0")
    clock.now = 30
    until(
        controller,
        lambda: (
            executions.get(plan.plan_id).submission_observation.native_identity
            == identity(backend.handle, 2)
        ),
    )
    until(controller, lambda: bool(backend.collects))
    for _ in range(10):
        controller.reconcile_once()
    assert (
        collections.get_submission_generation(plan.plan_id, identity(backend.handle, 2))
        is None
    )
    assert collections.get_submission(plan.plan_id) is None
    assert executions.get(plan.plan_id).attempt is None
    assert list(layout.attempts.iterdir()) == []


def test_public_attempt_page_exposes_native_collection_separately(native_runtime):
    controller, plan, layout, backend, executions, collections, clock = native_runtime
    outcome(native_runtime, 2, batch=9, stage=7)
    backend.native(2, "FAILED", "9:0")
    clock.now = 30
    native = identity(backend.handle, 2)
    until(
        controller,
        lambda: collections.get_submission_generation(plan.plan_id, native) is not None,
    )
    service = RunService(
        Settings(layout.run_dir.parent.parent / "state", layout.run_dir.parent),
        controller.plan_store,
        executions,
        controller,
        {"slurm": backend},
        collections=collections,
    )
    page = service.attempts(plan.plan.run_id)
    assert page["total"] == 1
    row = page["attempts"][0]
    assert row["attempt"] is None and row["handle"] is None
    assert row["collection"] is None and row["runtime"] is None
    assert row["scheduler_binding"] is None
    assert row["submission_handle"] == backend.handle.to_dict()
    proposal = row["submission_collection"]
    assert (
        proposal
        == collections.get_submission_generation(
            plan.plan_id, native
        ).proposal.to_dict()
    )
    assert proposal["identity"]["restart_count"] == 2
    assert (
        proposal["plan_sha256"]
        == hashlib.sha256(plan.plan.to_json().encode()).hexdigest()
    )
    assert len(proposal["manifest_sha256"]) == 64
    assert proposal["result"]["runtime"]["status"] == "unknown"
    assert proposal["result"]["batch"]["exit_code"] == 9
    assert proposal["result"]["stage_out"]["exit_code"] == 7
    with pytest.raises(KeyError):
        service.attempts(str(uuid4()))


@pytest.mark.parametrize("hash_field", ["plan_sha256", "manifest_sha256"])
def test_foreign_hash_collection_callback_is_retained_as_uncertainty(
    native_runtime, hash_field
):
    controller, plan, layout, backend, executions, collections, clock = native_runtime
    outcome(native_runtime, 2, batch=9, stage=7)
    backend.native(2, "FAILED", "9:0")
    proposal = backend.observer.collect_submission(backend.handle)
    assert proposal is not None
    malicious = replace(proposal, **{hash_field: "0" * 64})
    backend.collect_submission = lambda handle: malicious
    clock.now = 30
    until(
        controller,
        lambda: (
            plan.plan_id in controller.dispatch._inflight
            and controller.dispatch._inflight[plan.plan_id][0] == "collect_generation"
        ),
    )
    work = controller.dispatch._inflight[plan.plan_id][1]
    deadline = time.monotonic() + 5
    while not work.done() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert work.done()
    controller.dispatch.harvest()
    current = executions.get(plan.plan_id)
    assert current.submission_observation.native_identity == identity(backend.handle, 2)
    assert current.submission_observation.status == ObservationStatus.TERMINAL
    assert current.submission_handle == backend.handle
    assert current.attempt is None and current.handle is None
    assert (
        "Native collection proof differs"
        in current.latest_submission_observation.detail
    )
    assert (
        collections.get_submission_generation(plan.plan_id, proposal.identity) is None
    )
    assert collections.get_submission(plan.plan_id) is None
    assert list(layout.attempts.iterdir()) == []


def test_stale_initial_binding_callback_cannot_activate_older_native_generation(
    native_runtime,
):
    controller, plan, layout, backend, executions, _, clock = native_runtime
    # The real writer reserved generation 0, but native execution has since
    # requeued twice. The delayed discover callback deliberately returns that
    # valid old proof while its admitted source already observes generation 2.
    old_binding = coordinate_attempt(
        layout.run_dir, identity(backend.handle), initial_attempt=None
    )
    assert old_binding.attempt == 1
    backend.native(2, "RUNNING")
    executions.record_observation(plan.plan_id, backend.observe(backend.handle))
    backend.discover_initial = lambda handle: old_binding
    clock.now = 5
    until(
        controller,
        lambda: (
            plan.plan_id in controller.dispatch._inflight
            and controller.dispatch._inflight[plan.plan_id][0] == "startup"
            and controller.dispatch._inflight_sources[
                plan.plan_id
            ].submission_observation.native_identity
            == identity(backend.handle, 2)
        ),
    )
    work = controller.dispatch._inflight[plan.plan_id][1]
    deadline = time.monotonic() + 5
    while not work.done() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert work.done()
    controller.dispatch.harvest()
    current = executions.get(plan.plan_id)
    assert current.attempt is None and current.handle is None
    assert current.submission_handle == backend.handle
    assert current.submission_observation.native_identity == identity(backend.handle, 2)
    assert current.submission_observation.status == ObservationStatus.RUNNING
    assert current.latest_submission_observation.status == ObservationStatus.ERROR
    assert "Startup binding differs" in current.latest_submission_observation.detail
    for table in ("execution_control", "initial_bindings"):
        assert (
            executions._connection.execute(
                f"SELECT count(*) FROM {table} WHERE plan_id=?", (plan.plan_id,)
            ).fetchone()[0]
            == 0
        )
    assert plan.plan_id not in controller.dispatch._inflight
    assert plan.plan_id not in controller.dispatch._inflight_sources
    assert plan.plan_id not in controller.dispatch._receipts
    # The writer's authentic reservation remains untouched; the daemon neither
    # promotes it nor allocates a replacement to patch the generation mismatch.
    assert list(layout.attempts.iterdir()) == [layout.attempt_dir(1)]
    controller.reconcile_once()
    assert executions.get(plan.plan_id).attempt is None
