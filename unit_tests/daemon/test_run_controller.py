"""A committed claim always has its spawn in flight, and its result is kept."""

import threading
import time
from contextlib import ExitStack

import pytest

from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.runs.controller import RETRY_SECONDS, RunController
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.database import open_database
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.run_log import RunLog
from wavedl_daemon.storage.settings import Settings

from .test_hosted_app import StubSupervisor, train_payload


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def started(tmp_path):
    """Build a controller over a real run log with the given pool budget."""
    with ExitStack() as stack:

        def start(budget=WorkBudget()):
            settings = Settings(tmp_path / "state", tmp_path / "runs")
            settings.validate()
            lock = stack.enter_context(
                acquire_dispatch_lock(settings.state_root / "dispatch.lock")
            )
            connection = open_database(
                settings.state_root / "registry.db", dispatch_lock=lock
            )
            stack.callback(connection.close)
            pool = BoundedThreadPool(budget)
            stack.callback(pool.shutdown)
            supervisor = StubSupervisor()
            local = LocalBackend(dispatch_lock=lock, supervisor=supervisor)
            stack.callback(local.close)
            log, plans = RunLog(connection), PlanStore(connection)
            service = RunService(settings, plans, log, {"local": local})
            clock = Clock()
            controller = RunController(log, plans, pool, local=local, clock=clock)
            return service, controller, pool, supervisor, clock

        yield start


def submitted(service, key):
    plan = service.plan(train_payload(), key)
    service.submit(plan.plan_id)
    return plan.run_id


def kinds(service, run_id):
    return [stored.event.kind for stored in service.log.read_after(run_id, (0, -1))]


def tick_until(controller, service, run_id, states):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        controller.tick()
        view = service.get(run_id)
        if view.state in states:
            return view
        time.sleep(0.01)
    pytest.fail(f"{run_id} never reached {states}: {service.get(run_id)}")


def test_a_spawn_that_outlasts_the_work_deadline_is_recorded(started, monkeypatch):
    service, controller, _, supervisor, _ = started(WorkBudget(timeout_seconds=0.1))
    probe = LocalBackend._probe

    def slow_probe(backend, plan, cwd, environment):
        time.sleep(0.3)
        return probe(backend, plan, cwd, environment)

    monkeypatch.setattr(LocalBackend, "_probe", slow_probe)
    run_id = submitted(service, "slow")
    view = tick_until(controller, service, run_id, {"running", "supervision_lost"})
    assert view.state == "running" and not view.uncertainty
    assert len(supervisor.launches) == 1
    supervisor.launches[0].finish(0)
    assert (
        tick_until(controller, service, run_id, {"completed"}).evidence["exit_code"]
        == 0
    )


def test_recovery_adopts_the_process_this_daemon_started(started, monkeypatch):
    service, controller, _, supervisor, clock = started()
    spawn = LocalBackend.spawn

    def spawn_then_fail(backend, plan, attempt):
        spawn(backend, plan, attempt)
        raise RuntimeError("the spawn result was lost after the start")

    monkeypatch.setattr(LocalBackend, "spawn", spawn_then_fail)
    run_id = submitted(service, "lost-result")
    deadline = time.monotonic() + 10
    while not controller.errors:
        assert time.monotonic() < deadline, "the spawn never failed"
        controller.tick()
        time.sleep(0.01)
    assert service.get(run_id).state == "submission_unknown"
    clock.now += RETRY_SECONDS
    view = tick_until(controller, service, run_id, {"running", "supervision_lost"})
    assert view.state == "running" and not view.uncertainty
    assert kinds(service, run_id) == ["attempt", "submitting", "submitted", "observed"]
    assert len(supervisor.launches) == 1
    supervisor.launches[0].finish(0)
    assert tick_until(controller, service, run_id, {"completed"}).state == "completed"


def test_a_claim_waits_for_worker_capacity_instead_of_being_lost(started):
    service, controller, pool, supervisor, _ = started(
        WorkBudget(max_workers=1, max_pending=0)
    )
    release = threading.Event()
    blocker = pool.control_lane.submit(lambda token: release.wait(10))
    try:
        run_id = submitted(service, "queued")
        for _ in range(5):
            controller.tick()
        view = service.get(run_id)
        assert view.state == "pending" and not view.uncertainty
        assert kinds(service, run_id) == ["attempt"]
    finally:
        release.set()
    blocker.result(5)
    view = tick_until(controller, service, run_id, {"running", "supervision_lost"})
    assert view.state == "running"
    assert len(supervisor.launches) == 1
