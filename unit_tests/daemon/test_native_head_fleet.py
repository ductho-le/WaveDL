"""Native head scope survives failed polls without losing fleet backoff."""

from dataclasses import replace

import pytest

from unit_tests.daemon.test_slurm_writer_submission import identity
from unit_tests.daemon.test_writer_reconciliation import fleet, receipt
from wavedl_daemon.backends.contracts import ObservationStatus, SubmissionObservation


def install_observations(backend, handle, *observations):
    pending = iter(observations)

    def observe(handles):
        backend.calls.append(handles)
        return {handle.job_id: next(pending)}

    backend.observe_many = observe


def test_newer_error_head_and_untyped_failures_preserve_scope_and_backoff():
    poller, backend, clock = fleet()
    ack = receipt(job=42)
    good = SubmissionObservation(
        ObservationStatus.RUNNING, native_identity=identity(ack, 2)
    )
    newer_error = SubmissionObservation(
        ObservationStatus.ERROR, native_identity=identity(ack, 3)
    )
    install_observations(
        backend,
        ack,
        good,
        newer_error,
        SubmissionObservation(ObservationStatus.ERROR),
        SubmissionObservation(ObservationStatus.ABSENT),
    )
    poller.set_handles((ack,), native_generations={ack: identity(ack, 2)})
    assert poller.tick()[0].last_known == good
    clock.now = 30
    advanced = poller.tick()[0]
    assert advanced.observation == newer_error
    assert advanced.last_known == good
    assert advanced.next_due == 35
    scope = {ack: identity(ack, 3)}
    for now in (30, 31, 34):
        clock.now = now
        poller.set_handles((ack,), native_generations=scope)
        assert poller.snapshots == (advanced,)
        assert poller.tick() == ()
        assert len(backend.calls) == 2
    clock.now = 35
    untyped = poller.tick()[0]
    assert untyped.observation.native_identity is None
    assert untyped.last_known == good
    assert untyped.next_due == 45
    for now in (35, 36, 44):
        clock.now = now
        poller.set_handles((ack,), native_generations=scope)
        assert poller.snapshots == (untyped,)
        assert poller.tick() == ()
        assert len(backend.calls) == 3
    clock.now = 45
    absent = poller.tick()[0]
    assert absent.observation.status == ObservationStatus.ABSENT
    assert absent.last_known == good
    assert absent.next_due == 75
    poller.set_handles((ack,), native_generations=scope)
    assert poller.snapshots == (absent,)
    assert poller.tick() == ()
    assert len(backend.calls) == 4


@pytest.mark.parametrize(
    "field", ["run_id", "request_id", "owner", "job_id", "cluster", "restart_count"]
)
@pytest.mark.parametrize(
    "status",
    [
        ObservationStatus.RUNNING,
        ObservationStatus.TERMINAL,
        ObservationStatus.ERROR,
        ObservationStatus.ABSENT,
    ],
)
def test_foreign_or_lower_native_evidence_cannot_poison_head_or_good_cache(
    field, status
):
    poller, backend, clock = fleet()
    ack = receipt(job=42)
    head = identity(ack, 3)
    good = SubmissionObservation(ObservationStatus.RUNNING, native_identity=head)
    foreign = replace(
        head,
        **{
            field: {
                "owner": "bob",
                "job_id": "43",
                "cluster": "beta",
                "restart_count": 2,
            }.get(field, getattr(identity(receipt()), field))
        },
    )
    install_observations(
        backend, ack, good, SubmissionObservation(status, native_identity=foreign)
    )
    poller.set_handles((ack,), native_generations={ack: head})
    poller.tick()
    clock.now = 30
    refused = poller.tick()[0]
    assert refused.observation.status == ObservationStatus.ERROR
    assert refused.observation.native_identity is None
    assert "native generation" in refused.observation.detail
    assert refused.last_known == good
    assert refused.next_due == 35
    poller.set_handles((ack,), native_generations={ack: head})
    assert poller.snapshots == (refused,)
    assert poller.tick() == ()
    assert len(backend.calls) == 2


def test_query_discovered_error_head_is_retained_without_initial_scope():
    poller, backend, _ = fleet()
    ack = receipt(job=42)
    error = SubmissionObservation(
        ObservationStatus.ERROR, native_identity=identity(ack, 3)
    )
    install_observations(backend, ack, error)
    poller.set_handles((ack,))
    observed = poller.tick()[0]
    poller.set_handles((ack,), native_generations={ack: identity(ack, 3)})
    assert poller.snapshots == (observed,)
    assert poller.tick() == ()
    assert len(backend.calls) == 1
