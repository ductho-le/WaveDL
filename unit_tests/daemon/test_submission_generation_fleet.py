"""Immutable ACK fleet caches must follow native execution generations."""

from dataclasses import replace

import pytest

from unit_tests.daemon.test_slurm_writer_submission import identity
from unit_tests.daemon.test_writer_reconciliation import fleet, receipt
from wavedl_daemon.backends.contracts import ObservationStatus, SubmissionObservation
from wavedl_daemon.runs.reconciliation import PollPolicy


def native_backend(backend, handle, generation, status=ObservationStatus.TERMINAL):
    def observe(handles):
        backend.calls.append(handles)
        return {
            handle.job_id: SubmissionObservation(
                status, native_identity=identity(handle, generation)
            )
        }

    backend.observe_many = observe


def test_scope_resets_cached_terminal_without_mutating_ack():
    poller, backend, clock = fleet()
    ack = receipt(job=42)
    native_backend(backend, ack, 0)
    poller.set_handles((ack,), native_generations={ack: identity(ack)})
    initial = poller.tick()[0]
    assert initial.observation.native_identity.restart_count == 0
    clock.now = 1
    poller.set_handles((ack,), native_generations={ack: identity(ack, 2)})
    assert poller.snapshots[0].observation is None
    assert poller.snapshots[0].last_known is None
    native_backend(backend, ack, 2)
    update = poller.tick()[0]
    assert update.handle == ack
    assert update.observation.native_identity == identity(ack, 2)
    assert not hasattr(update.observation, "attempt")
    poller.set_handles((ack,), native_generations={ack: identity(ack, 2)})
    assert poller.snapshots == (update,)
    assert poller.tick() == (update,)
    assert len(backend.calls) == 2


def test_terminal_watch_queries_due_same_ack_for_requeue():
    poller, backend, clock = fleet()
    ack = receipt(job=42)
    native_backend(backend, ack, 0)
    poller.set_handles(
        (ack,), native_generations={ack: identity(ack)}, terminal_watch=(ack,)
    )
    initial = poller.tick()[0]
    clock.now = 29
    assert poller.tick() == (initial,)
    native_backend(backend, ack, 2)
    clock.now = 30
    updates = poller.tick()
    assert len(updates) == 1
    assert updates[0].observation.native_identity == identity(ack, 2)
    assert backend.calls == [(ack,), (ack,)]
    poller.set_handles(())
    assert poller.snapshots == ()
    assert poller.tick() == ()


def test_terminal_watch_batch_bound_does_not_drop_cached_unpolled_evidence():
    poller, backend, clock = fleet(
        policy=PollPolicy(max_batch=1, max_batches_per_tick=1)
    )
    first, second = receipt(job=42), receipt(job=42)
    backend.observe_many = lambda handles: {
        handle.job_id: SubmissionObservation(
            ObservationStatus.TERMINAL, native_identity=identity(handle)
        )
        for handle in handles
    }
    handles = (first, second)
    scope = {handle: identity(handle) for handle in handles}
    poller.set_handles(handles, native_generations=scope, terminal_watch=handles)
    poller.tick()
    poller.tick()
    assert all(update.observation is not None for update in poller.snapshots)
    clock.now = 30
    updates = poller.tick()
    assert len(updates) == 2
    assert {update.handle for update in updates} == set(handles)


@pytest.mark.parametrize(
    "field", ["run_id", "request_id", "owner", "job_id", "cluster"]
)
def test_foreign_scope_rejected_atomically(field):
    poller, backend, _ = fleet()
    ack = receipt(job=42)
    native_backend(backend, ack, 0)
    poller.set_handles((ack,))
    saved = poller.tick()[0]
    foreign = replace(
        identity(ack),
        **{
            field: {"owner": "bob", "job_id": "43", "cluster": "beta"}.get(
                field, getattr(identity(receipt()), field)
            )
        },
    )
    with pytest.raises(ValueError, match="differs"):
        poller.set_handles((ack,), native_generations={ack: foreign})
    assert poller.snapshots == (saved,)


def test_unknown_scope_preserves_unknown_backoff_and_bound_cache():
    poller, backend, _ = fleet()
    ack = receipt(job=42)
    bound = receipt(job=43).bind_attempt(7)
    backend.error = True
    poller.set_handles((ack, bound), native_generations={})
    updates = poller.tick()
    poller.set_handles((ack, bound), native_generations={})
    assert poller.snapshots == updates
    assert poller.tick() == ()


def test_unknown_cached_identity_resets_once_current_generation_is_known():
    poller, _, _ = fleet()
    ack = receipt(job=42)
    poller.set_handles((ack,))
    poller.tick()
    poller.set_handles((ack,), native_generations={ack: identity(ack, 2)})
    assert poller.snapshots[0].last_known is None


def test_known_scope_retains_error_backoff_for_matching_history():
    poller, backend, clock = fleet()
    ack = receipt(job=42)
    scope = {ack: identity(ack, 2)}
    native_backend(backend, ack, 2, ObservationStatus.RUNNING)
    poller.set_handles((ack,), native_generations=scope)
    poller.tick()
    backend.observe_many = lambda handles: {}
    clock.now = 30
    update = poller.tick()[0]
    assert update.observation.status == ObservationStatus.ERROR
    assert update.last_known.native_identity == scope[ack]
    poller.set_handles((ack,), native_generations=scope)
    assert poller.snapshots == (update,)
    assert poller.tick() == ()


def test_scope_and_terminal_watch_require_active_native_receipts():
    poller, _, _ = fleet()
    ack = receipt(job=42)
    with pytest.raises(ValueError, match="active receipts"):
        poller.set_handles((), native_generations={ack: identity(ack)})
    with pytest.raises(ValueError, match="receipt subset"):
        poller.set_handles((ack.bind_attempt(1),), terminal_watch=(ack,))
