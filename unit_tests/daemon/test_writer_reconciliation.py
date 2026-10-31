"""Fleet polling preserves receipt evidence without fabricating attempts."""

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.reconciliation import PollPolicy, SchedulerReconciler


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Backend:
    def __init__(self):
        self.calls = []
        self.status = ObservationStatus.RUNNING
        self.error = False

    def observe_many(self, handles):
        self.calls.append(handles)
        if self.error:
            raise OSError("scheduler unavailable")
        return {
            handle.job_id: (
                SubmissionObservation(self.status)
                if isinstance(handle, SubmissionHandle)
                else BackendObservation(handle.attempt, self.status)
            )
            for handle in handles
        }


def receipt(job=1, cluster="alpha", owner="alice", restart=0):
    request_id = str(uuid4())
    return SubmissionHandle(
        "slurm",
        str(uuid4()),
        request_id,
        owner,
        str(Path(__file__).resolve().parent / "synthetic-run"),
        job_id=f"{job};{cluster}",
        raw_json=json.dumps(
            {
                "cluster": cluster,
                "comment": request_id,
                "job_name": f"wavedl-{request_id}",
                "restart_count": restart,
            }
        ),
    )


def fleet(**kwargs):
    backend, clock = Backend(), Clock()
    return SchedulerReconciler(backend, clock=clock, **kwargs), backend, clock


def test_mixed_fleet_batches_preserve_observation_sources():
    poller, backend, _ = fleet()
    native, bound = receipt(), receipt(2).bind_attempt(7)
    poller.set_handles((native, bound))
    updates = poller.tick()
    assert backend.calls == [(native, bound)]
    assert isinstance(updates[0].observation, SubmissionObservation)
    assert not hasattr(updates[0].observation, "attempt")
    assert isinstance(updates[1].observation, BackendObservation)
    assert updates[1].observation.attempt == 7


@pytest.mark.parametrize("mode", ["exception", "missing", "wrong_source"])
@pytest.mark.parametrize("bound", [False, True])
def test_poll_failure_is_typed_and_retains_only_matching_history(mode, bound):
    poller, backend, clock = fleet()
    native = receipt()
    handle = native.bind_attempt(3) if bound else native
    poller.set_handles((handle,))
    previous = poller.tick()[0].observation
    clock.now = 30
    if mode == "exception":
        backend.error = True
    else:
        wrong = (
            SubmissionObservation(ObservationStatus.TERMINAL)
            if bound
            else BackendObservation(3, ObservationStatus.TERMINAL)
        )
        backend.observe_many = lambda handles: (
            {} if mode == "missing" else {handle.job_id: wrong}
        )
    update = poller.tick()[0]
    assert type(update.observation) is type(previous)
    assert update.observation.status == ObservationStatus.ERROR
    assert update.observation.exit_code is None
    assert update.last_known == previous
    if not bound:
        assert not hasattr(update.observation, "attempt")


def test_same_native_job_different_sources_get_separate_batches_and_history():
    poller, backend, _ = fleet()
    native = receipt()
    bound = native.bind_attempt(2)
    poller.set_handles((native, bound))
    updates = poller.tick()
    assert backend.calls == [(native,), (bound,)]
    assert isinstance(updates[0].last_known, SubmissionObservation)
    assert isinstance(updates[1].last_known, BackendObservation)
    poller.set_handles((bound,))
    assert poller.snapshots == (updates[1],)


def test_binding_requeue_resets_native_history():
    poller, _, _ = fleet()
    native = receipt()
    poller.set_handles((native,))
    poller.tick()
    poller.set_handles((native.bind_attempt(1),))
    assert poller.snapshots[0].last_known is None
    raw = json.loads(native.raw_json) | {"restart_count": 1}
    requeued = replace(native, raw_json=json.dumps(raw))
    poller.set_handles((requeued,))
    assert poller.snapshots[0].last_known is None


def test_submission_terminal_redelivered_without_polling_until_removed():
    poller, backend, clock = fleet()
    native = receipt()
    poller.set_handles((native,))
    backend.status = ObservationStatus.TERMINAL
    terminal = poller.tick()[0]
    clock.now = 1000
    assert poller.tick() == (terminal,)
    assert backend.calls == [(native,)]
    poller.set_handles(())
    assert poller.tick() == ()


def test_mixed_fleet_capacity_refusal_and_fair_bounded_pagination():
    poller, backend, _ = fleet(
        policy=PollPolicy(max_active=4, max_batch=1, max_batches_per_tick=1)
    )
    handles = tuple(
        receipt(job) if job % 2 else receipt(job).bind_attempt(job)
        for job in range(1, 5)
    )
    poller.set_handles(handles)
    with pytest.raises(BudgetExhausted):
        poller.set_handles((*handles, receipt(5)))
    assert tuple(update.handle for update in poller.snapshots) == handles
    for _ in handles:
        assert len(poller.tick()) == 1
    assert tuple(batch[0] for batch in backend.calls) == handles


def test_submission_grouping_preserves_cluster_and_owner_boundaries():
    poller, backend, _ = fleet()
    handles = (receipt(), receipt(2, "beta"), receipt(3, owner="bob"))
    poller.set_handles(handles)
    poller.tick()
    assert backend.calls == [(handle,) for handle in handles]


def test_submission_errors_preserve_backoff_and_retention_risk():
    poller, backend, clock = fleet(
        policy=PollPolicy(backoff_initial=2, backoff_max=8), jitter=lambda: 0.5
    )
    poller.set_handles((receipt(),))
    backend.error = True
    for delay in (3, 6, 8):
        update = poller.tick()[0]
        assert update.next_due == clock.now + delay
        assert update.retention_risk
        assert isinstance(update.observation, SubmissionObservation)
        clock.now = update.next_due
