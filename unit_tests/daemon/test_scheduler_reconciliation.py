"""Fleet reconciliation uses fake evidence only, never real scheduler commands."""

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
)
from wavedl_daemon.backends.slurm_observation import SlurmCapabilities
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.runs.reconciliation import PollPolicy, SchedulerReconciler


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Backend:
    def __init__(self):
        self.calls = []
        self.status = ObservationStatus.RUNNING
        self.missing = False
        self.error = False

    def observe_many(self, handles):
        self.calls.append(handles)
        if self.error:
            raise OSError("controller unavailable")
        return {
            handle.job_id: BackendObservation(
                handle.attempt, self.status, "native-state"
            )
            for handle in handles
            if not self.missing
        }


def handle(job=1, cluster="alpha", owner="alice", attempt=1):
    return Handle(
        "slurm",
        str(uuid4()),
        str(uuid4()),
        attempt,
        owner,
        str(Path.cwd() / "mock-reconciliation-run"),
        job_id=f"{job};{cluster}",
    )


def fleet(**kwargs):
    backend, clock = Backend(), Clock()
    reconciler = SchedulerReconciler(backend, clock=clock, **kwargs)
    return reconciler, backend, clock


def test_many_runs_one_batch_and_no_early_poll():
    poller, backend, clock = fleet()
    poller.set_handles(tuple(handle(job) for job in range(1, 101)))
    assert len(poller.tick()) == 100
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 100
    clock.now = 29
    assert poller.tick() == ()


def test_groups_separate_cluster_and_owner():
    poller, backend, _ = fleet()
    poller.set_handles((handle(1), handle(2, "beta"), handle(3, owner="bob")))
    poller.tick()
    assert len(backend.calls) == 3
    assert all(len(batch) == 1 for batch in backend.calls)


@pytest.mark.parametrize("mode", ["missing", "error", "absent"])
def test_errors_and_absence_retain_native_evidence(mode):
    poller, backend, clock = fleet()
    poller.set_handles((handle(),))
    prior = poller.tick()[0].observation
    clock.now = 30
    backend.missing = mode == "missing"
    backend.error = mode == "error"
    if mode == "absent":
        backend.status = ObservationStatus.ABSENT
    current = poller.tick()[0]
    assert current.last_known == prior
    assert current.observation.status == (
        ObservationStatus.ABSENT if mode == "absent" else ObservationStatus.ERROR
    )
    assert current.observation.exit_code is None


def test_backoff_capped_and_jitter_injected():
    poller, backend, clock = fleet(
        policy=PollPolicy(backoff_initial=2, backoff_max=10), jitter=lambda: 0.5
    )
    poller.set_handles((handle(),))
    backend.error = True
    for expected in (3, 6, 10, 10):
        update = poller.tick()[0]
        assert update.next_due == clock.now + expected
        clock.now = update.next_due


def test_retention_caps_interval_and_outage_backoff():
    poller, backend, clock = fleet(
        capabilities=SlurmCapabilities(False, 8),
        policy=PollPolicy(interval=30, backoff_initial=10),
    )
    poller.set_handles((handle(),))
    assert poller.tick()[0].next_due == 4
    backend.error = True
    clock.now = 4
    update = poller.tick()[0]
    assert update.next_due == 8
    assert update.retention_risk  # Next retry cannot guarantee catching retention.
    clock.now = 8
    update = poller.tick()[0]
    assert update.retention_risk
    assert update.last_known.status == ObservationStatus.RUNNING


def test_expired_window_remains_uncertain_not_terminal():
    poller, backend, clock = fleet(capabilities=SlurmCapabilities(False, 10))
    poller.set_handles((handle(),))
    poller.tick()
    clock.now = 100
    backend.status = ObservationStatus.ABSENT
    update = poller.tick()[0]
    assert update.retention_risk
    assert "unknown" in update.detail
    assert update.observation.status == ObservationStatus.ABSENT


def test_active_budget_refusal_atomic_and_batch_pagination_fair():
    poller, backend, _ = fleet(
        policy=PollPolicy(max_active=5, max_batch=2, max_batches_per_tick=1)
    )
    handles = tuple(handle(job) for job in range(1, 6))
    poller.set_handles(handles)
    with pytest.raises(BudgetExhausted):
        poller.set_handles((*handles, handle(6)))
    assert len(poller.snapshots) == 5
    assert len(poller.tick()) == 2
    assert len(poller.tick()) == 2
    assert len(poller.tick()) == 1
    assert len(backend.calls) == 3
    assert {item for batch in backend.calls for item in batch} == set(handles)


def test_terminal_snapshot_retained_without_repeated_queries_or_history():
    poller, backend, clock = fleet()
    poller.set_handles((handle(),))
    backend.status = ObservationStatus.TERMINAL
    terminal = poller.tick()[0]
    clock.now = 10000
    assert poller.tick() == (terminal,)
    assert poller.snapshots == (terminal,)
    assert len(backend.calls) == 1
    poller.set_handles(())
    assert poller.snapshots == ()


def test_same_job_ids_attempts_and_requeue_identity_never_share_history():
    poller, backend, clock = fleet()
    first = handle()
    second = replace(first, attempt=2, request_id="second")
    poller.set_handles((first, second))
    updates = poller.tick()
    assert len(backend.calls) == 2
    assert {item.observation.attempt for item in updates} == {1, 2}
    replacement = replace(first, raw_json='{"restart_count":1}')
    poller.set_handles((replacement,))
    assert poller.snapshots[0].last_known is None
    backend.error = True
    clock.now = 30
    assert poller.tick()[0].last_known is None


def test_bounded_worker_hook_and_pool_refusal():
    poller, _, _ = fleet()
    poller.set_handles((handle(),))
    with BoundedThreadPool(WorkBudget(max_workers=1, max_pending=0)) as pool:
        assert len(poller.submit_tick(pool).result(timeout=2)) == 1
    with pytest.raises(RuntimeError, match="closed"):
        poller.submit_tick(pool)


def test_wrong_attempt_is_error_and_old_evidence_not_replaced():
    poller, backend, clock = fleet()
    poller.set_handles((handle(),))
    previous = poller.tick()[0].observation
    backend.observe_many = lambda handles: {
        handles[0].job_id: BackendObservation(99, ObservationStatus.TERMINAL)
    }
    clock.now = 30
    update = poller.tick()[0]
    assert update.observation.status == ObservationStatus.ERROR
    assert update.last_known == previous


def test_unknown_retention_and_zero_window_honestly_report_risk():
    for capabilities in (SlurmCapabilities(), SlurmCapabilities(False, 0)):
        poller, backend, _ = fleet(capabilities=capabilities)
        poller.set_handles((handle(),))
        backend.error = True
        assert poller.tick()[0].retention_risk
