"""Slurm runs on the run log: submit, recover, observe, requeue and cancel.

The run log, plans, controller, pool and backend are real. The scheduler is a
fake command runner answering with sacct, squeue and scontrol text, and the
test writes the attempt files the batch script would.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from wavedl_daemon.backends.slurm import (
    SchedulerUnavailable,
    SlurmBackend,
    SubmissionUnknown,
)
from wavedl_daemon.backends.slurm_plan import SlurmSite, job_name
from wavedl_daemon.events.metrics_tail import read_tail
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.runs.controller import RunController
from wavedl_daemon.runs.lifecycle import (
    AttemptFiles,
    Ended,
    Live,
    RunState,
    Work,
    WorkKind,
    react,
    settle,
    status,
)
from wavedl_daemon.runs.model import (
    AttemptOpened,
    ExitRecord,
    Exited,
    Metric,
    Observed,
    Outcome,
    Phase,
    RunId,
    SlurmHandle,
    Submitted,
    Submitting,
)
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.database import open_database
from wavedl_daemon.storage.ownership import DispatchRequired, acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.run_log import Expect, RunLog
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessResult

from .test_hosted_app import train_payload
from .test_slurm_observation import SINCE, account, controller, listed, shown


def ok(stdout=b""):
    return ProcessResult(0, stdout, b"")


class Scheduler:
    """Answers sbatch, scancel and every query from what the test set."""

    def __init__(self):
        self.calls = []
        self.sbatch = ok(b"4242;alpha\n")
        self.controller = ""
        self.accounting = ""
        self.found = ""  # recovery rows of squeue --Format and sacct --allocations

    def run(self, argv):
        self.calls.append(argv)
        match argv[0]:
            case "sbatch":
                if isinstance(self.sbatch, Exception):
                    raise self.sbatch
                return self.sbatch
            case "scancel":
                return ok()
            case "squeue" if any(arg.startswith("--Format") for arg in argv):
                return ok(self.found.encode())
            case "sacct" if "--allocations" in argv:
                return ok(self.found.encode())
            case "squeue":
                return ok(listed(self.controller).encode())
            case "scontrol":
                return ok(shown(self.controller, argv[-1]).encode())
            case "sacct":
                return ok(self.accounting.encode())
        return ok()

    def commands(self, name):
        return [call for call in self.calls if call[0] == name]


@dataclass
class Environment:
    service: RunService
    controller: RunController
    log: RunLog
    plans: PlanStore
    backend: SlurmBackend
    scheduler: Scheduler
    pool: BoundedThreadPool
    now: list[float]

    def restart(self) -> RunController:
        """A new controller over the same store, as after a daemon restart."""
        self.controller = RunController(
            self.log,
            self.plans,
            self.pool,
            local=None,
            slurm=self.backend,
            clock=lambda: self.now[0],
        )
        return self.controller


@pytest.fixture
def environment(tmp_path):
    settings = Settings(tmp_path / "state", tmp_path / "runs", shared_execution=True)
    settings.validate()
    scheduler = Scheduler()
    now = [1000.0]
    with acquire_dispatch_lock(settings.state_root / "dispatch.lock") as lock:
        connection = open_database(settings.state_root / "registry.db", dispatch_lock=lock)
        log, plans = RunLog(connection), PlanStore(connection)
        backend = SlurmBackend(
            SlurmSite("compute", cluster="alpha", requeue=True),
            scheduler,
            owner="alice",
            dispatch_lock=lock,
        )
        pool = BoundedThreadPool(WorkBudget())
        environment = Environment(
            RunService(settings, plans, log, {"slurm": backend}),
            None,
            log,
            plans,
            backend,
            scheduler,
            pool,
            now,
        )
        environment.restart()
        try:
            yield environment
        finally:
            pool.shutdown()
            connection.close()


def submitted(environment):
    """A submitted run whose job the scheduler lists as pending, as after sbatch."""
    body = train_payload(epochs=4, model="cnn", data_path="data.npz") | {"backend": "slurm"}
    plan = environment.service.plan(body, str(uuid4()))
    run_id = RunId(plan.run_id)
    environment.scheduler.controller = controller(target(run_id), "PENDING")
    environment.service.submit(plan.plan_id)
    return run_id, plan.preview.spec.paths


def tick_until(environment, run_id, predicate, *, advance=31.0):
    """Tick on a fake clock that makes each poll due, until ``predicate(head)``."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        environment.now[0] += advance
        environment.controller.tick()
        head = environment.log.head(run_id)
        if head is not None and predicate(head):
            return head
        time.sleep(0.005)
    pytest.fail(f"No bounded progress: {environment.log.head(run_id)}")


def events(environment, run_id, attempt):
    return [
        stored.event
        for stored in environment.log.read_after(run_id, (0, -1), limit=1000)
        if stored.attempt == attempt
    ]


def epochs(environment, run_id, attempt):
    return [
        event.epoch for event in events(environment, run_id, attempt) if isinstance(event, Metric)
    ]


def write_metrics(paths, attempt, *numbers):
    paths.attempt(attempt).mkdir(exist_ok=True)
    with paths.metrics(attempt).open("a") as stream:
        for epoch in numbers:
            stream.write(f'{{"epoch": {epoch}, "val_loss": {1 / epoch}}}\n')


def target(run_id):
    return run_id, SlurmHandle("4242", "alpha", "alice")


def live(phase):
    return lambda head: isinstance(head.stage, Live) and head.stage.phase is phase


def test_a_requeued_job_resumes_in_a_new_attempt_and_completes(environment):
    scheduler = environment.scheduler
    run_id, paths = submitted(environment)
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    assert head.stage.handle == SlurmHandle("4242", "alpha", "alice")
    (sbatch,) = scheduler.commands("sbatch")
    assert sbatch[-1] == str(paths.submission_script)
    assert paths.submission_script.read_text() == environment.plans.by_run(run_id).plan.script

    scheduler.controller = controller(target(run_id), "RUNNING")
    scheduler.accounting = account(target(run_id), "RUNNING", "")
    write_metrics(paths, 1, 1, 2, 3)
    tick_until(
        environment,
        run_id,
        lambda head: live(Phase.RUNNING)(head) and head.metrics_offset > 0,
    )
    assert epochs(environment, run_id, 1) == [1, 2, 3]

    # The node fails after epoch 3; Slurm requeues the job as restart 1.
    scheduler.controller = controller(target(run_id), "PENDING", restart=1)
    scheduler.accounting = account(target(run_id), "NODE_FAIL", "0:0")
    head = tick_until(environment, run_id, lambda head: head.attempt == 2)
    assert head.restart_count == 1 and live(Phase.QUEUED)(head)
    ending = events(environment, run_id, 1)[-1]
    assert ending == Exited(
        Outcome.REQUEUED, False, native_state="NODE_FAIL", detail="Requeued by Slurm"
    )
    assert events(environment, run_id, 2)[:2] == [
        AttemptOpened("requeue", 1),
        Observed(Phase.QUEUED, "PENDING", None),
    ]

    # Attempt 2 resumes from the epoch-2 checkpoint in the shared output dir.
    scheduler.controller = controller(target(run_id), "RUNNING", restart=1)
    write_metrics(paths, 2, 3, 4)
    tick_until(environment, run_id, lambda head: head.metrics_offset > 0)
    paths.exit_record(2).write_text("0 0 0\n")
    scheduler.controller = controller(target(run_id), "COMPLETED", restart=1)
    scheduler.accounting += account(target(run_id), "COMPLETED", restart=1)
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Ended))

    assert epochs(environment, run_id, 2) == [3, 4]
    assert head.stage.exit == Exited(
        Outcome.COMPLETED,
        True,
        exit_code=0,
        native_state="COMPLETED",
        attempt_exit=ExitRecord(0, 0, 0),
    )
    assert status(head).state is RunState.COMPLETED
    assert len(scheduler.commands("sbatch")) == 1
    assert environment.controller.observation_errors == {}


def test_react_and_settle_read_a_requeue_from_scheduler_text(environment, tmp_path):
    run_id, _ = submitted(environment)
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    runner = Scheduler()
    runner.controller = controller(target(run_id), "PENDING", restart=1)
    runner.accounting = account(target(run_id), "PREEMPTED", "0:15")
    backend = SlurmBackend(SlurmSite("compute", cluster="alpha"), runner, owner="alice")
    observation = backend.observe_many([target(run_id)], since=SINCE)[run_id]
    assert react(head, observation) == Work(
        WorkKind.SETTLE, run_id, head.revision, 1, head.metrics_offset
    )
    files = AttemptFiles(read_tail(tmp_path / "missing", start=0), None)
    assert settle(head, observation, files) == (
        Exited(Outcome.REQUEUED, False, native_state="PREEMPTED", detail="Requeued by Slurm"),
        AttemptOpened("requeue", 1),
        Observed(Phase.QUEUED, "PENDING", None),
    )


def recovery_row(run_id, job="4242", owner="alice", cluster="alpha", comment=None):
    return f"{job}|{owner}|{job_name(run_id)}|{comment or run_id}|{cluster}\n"


def test_an_unknown_sbatch_outcome_is_recovered_by_find_never_resubmitted(environment):
    scheduler = environment.scheduler
    scheduler.sbatch = ProcessResult(0, b"", b"", timed_out=True)
    run_id, _ = submitted(environment)
    scheduler.found = recovery_row(run_id)
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    assert head.stage.handle == SlurmHandle("4242", "alpha", "alice")
    assert len(scheduler.commands("sbatch")) == 1
    sacct = next(call for call in scheduler.commands("sacct") if "--allocations" in call)
    created = datetime.fromisoformat(environment.plans.by_run(run_id).created_at)
    since = (created - timedelta(minutes=5)).astimezone(UTC)
    assert f"--starttime={since:%Y-%m-%dT%H:%M:%S}" in sacct
    squeue = next(call for call in scheduler.commands("squeue") if "--states=all" in call and any(arg.startswith("--Format") for arg in call))
    for query in (sacct, squeue):
        assert f"--name={job_name(run_id)}" in query and "--user=alice" in query


@pytest.mark.parametrize("found", [True, False])
def test_a_crash_between_sbatch_and_its_commit_is_recovered(environment, found):
    run_id, _ = submitted(environment)
    head = environment.log.head(run_id)
    # The daemon committed the claim and died before recording sbatch's reply.
    environment.log.append(run_id, [Submitting()], expect=Expect(revision=head.revision))
    environment.scheduler.found = recovery_row(run_id) if found else ""
    environment.restart()
    head = tick_until(
        environment, run_id, lambda head: isinstance(head.stage, Live | Ended)
    )
    assert environment.scheduler.commands("sbatch") == []
    if found:
        assert events(environment, run_id, 1)[-1] == Submitted(
            SlurmHandle("4242", "alpha", "alice")
        )
    else:
        assert head.stage.exit.outcome is Outcome.UNKNOWN
        assert status(head).state is RunState.TERMINAL_UNKNOWN


def test_a_failed_sbatch_may_have_queued_the_job_so_find_settles_it(environment):
    scheduler = environment.scheduler
    scheduler.sbatch = ProcessResult(1, b"", b"Socket timed out on send/recv operation")
    run_id, _ = submitted(environment)
    scheduler.found = recovery_row(run_id)
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    assert head.stage.handle == SlurmHandle("4242", "alpha", "alice")

    scheduler.sbatch = ProcessResult(1, b"", b"Invalid partition specified")
    run_id, _ = submitted(environment)
    scheduler.found = ""
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Ended))
    assert head.stage.exit == Exited(
        Outcome.UNKNOWN,
        False,
        detail="sbatch exited 1: Invalid partition specified, and no job was found",
    )
    assert len(scheduler.commands("sbatch")) == 2


def test_cancel_interrupts_the_steps_then_ends_the_job_after_the_grace(environment):
    scheduler = environment.scheduler
    run_id, paths = submitted(environment)
    tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    scheduler.controller = controller(target(run_id), "RUNNING")
    tick_until(environment, run_id, live(Phase.RUNNING))

    environment.service.cancel(run_id)
    head = tick_until(
        environment, run_id, lambda head: head.stage.cancel_delivered, advance=1
    )
    filters = ("--clusters=alpha", "--user=alice", f"--name={job_name(run_id)}")
    assert scheduler.commands("scancel") == [("scancel", *filters, "--signal=INT", "4242")]
    assert status(head).state is RunState.STOPPING
    for _ in range(5):
        environment.now[0] += 1
        environment.controller.tick()
    assert len(scheduler.commands("scancel")) == 1

    environment.now[0] += 45
    tick_until(environment, run_id, lambda head: len(scheduler.commands("scancel")) == 2, advance=0)
    assert scheduler.commands("scancel")[1] == ("scancel", *filters, "4242")

    # Training saved its checkpoint and exited 130, which Slurm calls FAILED.
    paths.attempt(1).mkdir(exist_ok=True)
    paths.exit_record(1).write_text("0 130 0\n")
    scheduler.controller = controller(target(run_id), "FAILED", code="130:0")
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Ended))
    assert head.stage.exit.outcome is Outcome.CANCELLED and head.stage.exit.confirmed
    assert head.stage.exit.attempt_exit == ExitRecord(0, 130, 0)
    assert (head.stage.exit.exit_code, head.stage.exit.signal) == (130, None)
    interrupts = [call for call in scheduler.commands("scancel") if "--signal=INT" in call]
    assert len(interrupts) == 1


def test_a_completed_job_whose_exit_record_is_not_visible_completes(environment):
    scheduler = environment.scheduler
    run_id, _ = submitted(environment)
    tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    scheduler.controller = controller(target(run_id), "COMPLETED", code="0:0")
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Ended))
    assert head.stage.exit == Exited(
        Outcome.COMPLETED, True, exit_code=0, native_state="COMPLETED"
    )


def test_cancelling_a_queued_job_ends_it_without_a_signal(environment):
    scheduler = environment.scheduler
    run_id, _ = submitted(environment)
    tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    scheduler.controller = controller(target(run_id), "PENDING")
    tick_until(environment, run_id, live(Phase.QUEUED))
    environment.service.cancel(run_id)
    tick_until(environment, run_id, lambda head: head.stage.cancel_delivered, advance=1)
    (scancel,) = scheduler.commands("scancel")
    assert "--signal=INT" not in scancel
    scheduler.controller = controller(target(run_id), "CANCELLED", code="0:15")
    head = tick_until(environment, run_id, lambda head: isinstance(head.stage, Ended))
    assert head.stage.exit.outcome is Outcome.CANCELLED


def test_observation_failures_back_off_and_reach_health(environment):
    scheduler = environment.scheduler
    run_id, _ = submitted(environment)
    tick_until(environment, run_id, lambda head: isinstance(head.stage, Live))
    scheduler.controller = controller(target(run_id), "RUNNING", Comment="foreign")
    head = tick_until(environment, run_id, lambda head: head.stage.lost is not None)
    plan_id = head.ref.plan_id
    assert "identity mismatch" in environment.controller.observation_errors[plan_id]
    queries = len(scheduler.commands("squeue"))
    for _ in range(3):
        environment.now[0] += 1
        environment.controller.tick()
        time.sleep(0.01)
    assert len(scheduler.commands("squeue")) == queries
    scheduler.controller = controller(target(run_id), "RUNNING")
    tick_until(environment, run_id, live(Phase.RUNNING))
    assert environment.controller.observation_errors == {}


def test_submit_writes_the_script_once_and_needs_dispatch(tmp_path, environment):
    run_id, paths = submitted(environment)
    plan = environment.plans.by_run(run_id).plan
    handle = environment.backend.submit(plan)
    assert handle == SlurmHandle("4242", "alpha", "alice")
    assert paths.logs.is_dir() and paths.output.is_dir()
    assert paths.attempt(1).parent.is_dir() and not paths.attempt(1).exists()
    with pytest.raises(FileExistsError):
        environment.backend.submit(plan)
    assert len(environment.scheduler.commands("sbatch")) == 1
    unowned = SlurmBackend(SlurmSite("compute", cluster="alpha"), Scheduler(), owner="alice")
    with pytest.raises(DispatchRequired):
        unowned.submit(plan)


@pytest.mark.parametrize(
    "reply",
    [
        ok(b"accepted maybe"),
        ProcessResult(0, b"12;alpha", b"", timed_out=True),
        ProcessResult(0, b"12;alpha", b"", stdout_truncated=True),
        ok(b"12;beta"),
        ok(b"\xff"),
        OSError("lost acknowledgement"),
    ],
)
def test_an_uncertain_acknowledgement_is_unknown(environment, reply):
    environment.scheduler.sbatch = reply
    run_id, _ = submitted(environment)
    with pytest.raises(SubmissionUnknown):
        environment.backend.submit(environment.plans.by_run(run_id).plan)
    environment.scheduler.sbatch = ProcessResult(1, b"", b"denied")
    run_id, _ = submitted(environment)
    with pytest.raises(SubmissionUnknown, match="denied"):
        environment.backend.submit(environment.plans.by_run(run_id).plan)


def test_find_filters_identity_and_refuses_two_jobs(environment):
    run_id, _ = submitted(environment)
    record = environment.plans.by_run(run_id)
    created = datetime.fromisoformat(record.created_at)

    def find():
        return environment.backend.find(record.plan, created_at=created)

    scheduler = environment.scheduler
    scheduler.found = (
        recovery_row(run_id, owner="bob")
        + recovery_row(run_id, cluster="beta")
        + recovery_row(str(uuid4()))
    )
    assert find() is None
    scheduler.found = "CLUSTER: alpha\n" + recovery_row(run_id, comment="(null)")
    assert find() == SlurmHandle("4242", "alpha", "alice")
    scheduler.found = recovery_row(run_id, comment="foreign")
    assert find() is None
    scheduler.found = recovery_row(run_id) + recovery_row(run_id, job="4243")
    with pytest.raises(SchedulerUnavailable, match="Several"):
        find()
    scheduler.found = "garbage\n"
    with pytest.raises(SchedulerUnavailable):
        find()


def test_cancel_refuses_another_owners_job(environment):
    with pytest.raises(ValueError, match="another owner"):
        environment.backend.cancel(
            RunId(str(uuid4())), SlurmHandle("1", "alpha", "bob"), interrupt=True
        )
    assert environment.scheduler.commands("scancel") == []

