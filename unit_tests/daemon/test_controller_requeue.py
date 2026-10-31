"""Native proposals activate new generations without accepting stale work."""

import json
import os
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.protocol import Envelope, ProtocolError
from wavedl.core.scheduler_attempt import SchedulerBinding
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    Plan,
    UnsupportedPlan,
)
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.negotiation import NegotiationStatus
from wavedl_daemon.storage.executions import ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


RUN = "00000000-0000-4000-8000-000000001919"


class Backend:
    owner = "test-user"

    def __init__(self):
        self.proposal = None
        self.submissions = 0
        self.cancels = []
        self.cancel_entered = threading.Event()
        self.cancel_release = threading.Event()
        self.block_cancel = False
        self.cancel_error = False

    def submit(self, plan, *, request_id, attempt):
        self.submissions += 1
        return Handle(
            "slurm",
            plan.run_id,
            request_id,
            attempt,
            self.owner,
            plan.run_dir,
            job_id="19;fixture",
            raw_json=json.dumps(
                {
                    "cluster": "fixture",
                    "job_name": f"wavedl-{request_id}",
                    "comment": request_id,
                    "restart_count": 0,
                    "operation": "train",
                }
            ),
        )

    def discover_restart(self, handle):
        return (
            self.proposal
            if self.proposal is not None and self.proposal[1].attempt > handle.attempt
            else None
        )

    def observe(self, handle):
        return BackendObservation(handle.attempt, ObservationStatus.RUNNING)

    def request_cancel(self, handle):
        self.cancels.append(handle)
        if self.block_cancel and handle.attempt == 1:
            self.cancel_entered.set()
            self.cancel_release.wait(5)
            if self.cancel_error:
                raise OSError("old cancellation failure")


@pytest.fixture
def execution(tmp_path):
    database = tmp_path / "registry.db"
    backend = Backend()
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(database, dispatch_lock=lock),
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as provenance,
    ):
        manifest = build_manifest(
            RunLayout(tmp_path / "runs" / RUN, RUN),
            operation=OperationKind.TRAIN,
            resolved_config={"model": "cnn", "epochs": 1, "data_path": "unused.npz"},
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            submission_cwd=tmp_path,
        )
        planner = SlurmPlanner(
            SlurmSite(partition="cpu", cluster="fixture", requeue=True)
        )
        if os.name == "nt":
            # These controller tests use a fake scheduler, not native Slurm.
            # Preserve the real planner's refusal of Windows directive paths
            # while exercising every generation/provenance contract natively.
            with pytest.raises(UnsupportedPlan, match="Slurm directive paths"):
                planner.plan(manifest)
            proposal = Plan(
                "slurm",
                json.dumps(asdict(manifest)),
                argv=(sys.executable, "train.py"),
                resources=(("requeue", True),),
            )
        else:
            proposal = planner.plan(manifest)
        plan = plans.put(
            proposal,
            idempotency_key="requeue",
            request_payload={},
        )
        now = [0.0]
        controller = SubmissionController(
            plans,
            executions,
            {"slurm": backend},
            lock,
            WorkBudget(max_workers=4, max_pending=0),
            provenance_store=provenance,
            clock=lambda: now[0],
        )
        try:
            controller.request_submit(plan.plan_id)
            tick_until(
                controller, lambda: executions.get(plan.plan_id).handle is not None
            )
            # Drain startup's existing work without scheduling another lap before
            # tests inject an independently persisted terminal receipt.
            deadline = time.monotonic() + 3
            while (
                controller.dispatch._inflight
                or controller.handshake._runtime_work
                or controller.restarts._restart_work
            ):
                controller.dispatch.harvest()
                controller.handshake.harvest()
                controller.restarts.harvest()
                assert time.monotonic() < deadline
                time.sleep(0.005)
            yield SimpleNamespace(
                controller=controller,
                backend=backend,
                plans=plans,
                executions=executions,
                provenance=provenance,
                plan=plan,
                now=now,
            )
        finally:
            backend.cancel_release.set()
            controller.close()
            controller.drain()


def tick_until(controller, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        controller.reconcile_once()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Controller did not make bounded progress")


def proposed(record):
    binding = SchedulerBinding(
        record.run_id, record.request_id, "fixture", "19", record.owner, 1, 2
    )
    candidate = replace(
        record.handle,
        attempt=2,
        raw_json=json.dumps(
            json.loads(record.handle.raw_json)
            | {"restart_count": 1, "initial_attempt": 1}
        ),
    )
    return binding, candidate


def event(attempt):
    return Envelope(
        RUN,
        attempt,
        "hello",
        0,
        "2026-10-04T00:00:00Z",
        asdict(HelloPayload(1, 1, "actual", "3.11", "cpu", ["train"], sys.executable)),
    )


def activate(execution):
    current = execution.executions.get(execution.plan.plan_id)
    execution.backend.proposal = proposed(current)
    (Path(current.run_dir) / "attempts" / "2").mkdir(exist_ok=True)
    execution.now[0] = max(execution.now[0], 10)
    tick_until(
        execution.controller,
        lambda: execution.executions.get(current.plan_id).attempt == 2,
    )
    return execution.executions.get(current.plan_id)


def test_native_activation_keeps_old_provenance_and_requires_new_hello(execution):
    env = execution
    controller, plan_id = env.controller, env.plan.plan_id
    assert (
        controller.accept_runtime_events(plan_id, [event(1)]).status
        == NegotiationStatus.ACCEPTED
    )
    old_tracker = controller.handshake._runtime[plan_id]
    old_tracker.reject("prior failure")
    controller.handshake._runtime_errors[plan_id] = "prior failure"
    current = activate(env)
    assert current.attempt == 2 and current.observation is None
    assert env.backend.submissions == 1
    assert (
        controller.handshake._runtime[plan_id].result.status
        == NegotiationStatus.WAITING
    )
    assert plan_id not in controller.handshake._runtime_errors
    assert env.provenance.get(plan_id) is None
    assert env.provenance.get(plan_id, 1).event == event(1)
    assert (
        controller.accept_runtime_events(plan_id, [event(2)]).status
        == NegotiationStatus.ACCEPTED
    )
    tracker = controller.handshake._runtime[plan_id]
    env.now[0] = 20
    tick_until(controller, lambda: plan_id not in controller.restarts._restart_work)
    assert controller.handshake._runtime[plan_id] is tracker
    assert env.executions.history(plan_id)[0].attempt == 1


@pytest.mark.parametrize("error", [False, True])
def test_late_old_cancel_never_marks_new_attempt_delivered_or_failed(execution, error):
    env = execution
    env.backend.block_cancel = True
    env.backend.cancel_error = error
    env.controller.request_cancel(env.plan.plan_id)
    tick_until(env.controller, env.backend.cancel_entered.is_set)
    record = activate(env)
    assert record.cancellation_requested and not record.cancel_delivered
    env.backend.cancel_release.set()
    deadline = time.monotonic() + 2
    while (
        not env.controller.dispatch._inflight[env.plan.plan_id][1].done()
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
    env.controller.dispatch.harvest()
    record = env.executions.get(env.plan.plan_id)
    assert not record.cancel_delivered
    assert (
        record.latest_observation is None
        or record.latest_observation.status != ObservationStatus.ERROR
    )
    assert env.plan.plan_id not in env.controller.fleet._cancel_observe
    tick_until(
        env.controller, lambda: env.executions.get(env.plan.plan_id).cancel_delivered
    )
    assert [handle.attempt for handle in env.backend.cancels] == [1, 2]


def test_late_old_probe_error_cannot_poison_new_attempt(execution):
    env = execution
    entered, release = threading.Event(), threading.Event()
    # Finish any real bounded first-record probe before installing the deliberate
    # old-generation failure. The work stays admitted through actual cleanup.
    tick_until(
        env.controller,
        lambda: env.plan.plan_id not in env.controller.handshake._runtime_work,
    )

    def old_probe(token):
        entered.set()
        release.wait(3)
        raise ProtocolError("old malformed hello")

    work = env.controller.context.lane.submit(old_probe)
    env.controller.handshake._runtime_work[env.plan.plan_id] = (1, work)
    assert entered.wait(2)
    activate(env)
    release.set()
    deadline = time.monotonic() + 2
    while not work.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    env.controller.handshake.harvest()
    assert env.plan.plan_id not in env.controller.handshake._runtime_errors
    assert env.plan.plan_id not in env.controller.handshake._runtime_skip
    assert (
        env.controller.accept_runtime_events(env.plan.plan_id, [event(2)]).status
        == NegotiationStatus.ACCEPTED
    )


def test_terminal_then_later_restart_still_activates(execution):
    env = execution
    current = env.executions.get(env.plan.plan_id)
    env.executions.record_observation(
        current.plan_id,
        BackendObservation(
            1, ObservationStatus.TERMINAL, native_state="PREEMPTED", exit_code=143
        ),
    )
    env.now[0] = 70
    activated = activate(env)
    assert activated.attempt == 2
    assert (
        env.executions.history(current.plan_id)[0].observation.native_state
        == "PREEMPTED"
    )


def test_missing_native_restart_does_not_erase_terminal_evidence(execution):
    env = execution
    current = env.executions.get(env.plan.plan_id)
    env.executions.record_observation(
        current.plan_id,
        BackendObservation(
            1, ObservationStatus.TERMINAL, native_state="COMPLETED", exit_code=0
        ),
    )
    (Path(current.run_dir) / "attempts" / "2").mkdir()

    def unavailable(handle):
        raise OSError("private unavailable scheduler diagnostic")

    env.backend.discover_restart = unavailable
    env.now[0] = 70
    tick_until(env.controller, lambda: current.plan_id in env.controller.restart_errors)
    retained = env.executions.get(current.plan_id)
    assert retained.latest_observation.status == ObservationStatus.TERMINAL
    assert retained.attempt == 1
    assert env.controller.restart_errors[current.plan_id] == "StorageWorkError"


def test_fleet_capacity_refuses_before_durable_activation(execution):
    env = execution
    current = env.executions.get(env.plan.plan_id)
    env.backend.proposal = proposed(current)
    work = env.controller.context.lane.submit(lambda token: env.backend.proposal)
    work.result(timeout=2)
    env.controller.restarts._restart_work[current.plan_id] = (current, work)
    env.controller.fleet._fleets["slurm"] = SimpleNamespace(
        policy=SimpleNamespace(max_active=1)
    )
    env.controller.fleet._fleet_active["slurm"] = {"other": current}
    env.controller.restarts.harvest()
    assert env.executions.get(current.plan_id) == current
    assert env.executions.history(current.plan_id) == []
    assert env.controller.fleet._fleet_active["slurm"] == {"other": current}
    assert env.controller.restart_errors[current.plan_id] == "BudgetExhausted"
    assert current.plan_id not in env.controller.restarts._restart_work


def test_retired_restart_watch_cannot_exhaust_new_active_admission(execution):
    env = execution
    current = env.executions.get(env.plan.plan_id)
    env.controller.context.runtime_capacity = 1
    env.controller.restarts._restart_due = {"retired-missing-plan": 100}
    env.now[0] = 70
    env.controller.restarts.schedule(env.plan, current)
    assert current.plan_id in env.controller.restarts._restart_work
    assert "retired-missing-plan" not in env.controller.restarts._restart_due
    assert env.controller.restart_watch_refusals == 1
