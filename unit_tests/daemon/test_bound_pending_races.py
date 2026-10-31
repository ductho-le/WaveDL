"""Bound-runtime and native-generation callbacks are fenced at harvest time."""

import json
import threading
import time

import pytest

from unit_tests.daemon.test_collections import result
from unit_tests.daemon.test_slurm_writer_submission import identity
from unit_tests.daemon.test_submission_generation_controller import outcome
from unit_tests.daemon.test_writer_fleet_controller import until
from unit_tests.daemon.test_writer_historical_lineage import journal
from wavedl.core.protocol import decode
from wavedl.core.scheduler_attempt import coordinate_attempt
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
)
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


pytest_plugins = ["unit_tests.daemon.test_submission_generation_controller"]


@pytest.fixture
def bound_runtime(native_runtime):
    _, plan, layout, backend, executions, _, _ = native_runtime
    binding = coordinate_attempt(
        layout.run_dir, identity(backend.handle), initial_attempt=None
    )
    verified = backend.observer.discover_initial(backend.handle)
    assert verified == binding
    current = executions.get(plan.plan_id)
    executions.activate_initial(
        plan.plan_id, verified, expected_revision=current.revision
    )
    return native_runtime


def _advance(runtime, generation, state):
    controller, plan, _, backend, executions, _, _ = runtime
    backend.native(generation, state)
    controller.observations.record(plan.plan_id, backend.observe(backend.handle))
    return executions.get(plan.plan_id)


def _wait(work):
    deadline = time.monotonic() + 5
    while not work.done() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert work.done()


@pytest.mark.parametrize("activate_replacement", [False, True])
def test_positive_runtime_collection_is_scoped_across_native_head_advance(
    bound_runtime, activate_replacement
):
    """A completed runtime collection remains attempt history across a requeue."""
    controller, plan, layout, backend, executions, collections, clock = bound_runtime
    current = executions.get(plan.plan_id)
    executions.record_observation(
        plan.plan_id,
        BackendObservation(current.attempt, ObservationStatus.TERMINAL),
    )

    started = threading.Event()
    release = threading.Event()
    collected = result()
    runtime_handle = current.handle

    def collect(handle):
        assert handle == runtime_handle
        started.set()
        assert release.wait(5)
        return collected

    backend.collect = collect
    clock.now = 30
    until(
        controller,
        lambda: (
            plan.plan_id in controller.dispatch._inflight
            and controller.dispatch._inflight[plan.plan_id][0] == "collect"
        ),
    )
    assert started.wait(5)

    advanced = _advance(bound_runtime, 1, "PENDING")
    assert advanced.handle == runtime_handle
    if activate_replacement:
        binding = coordinate_attempt(
            layout.run_dir, advanced.native_identity, initial_attempt=None
        )
        verified, candidate = backend.observer.discover_restart(runtime_handle)
        assert verified == binding
        advanced = executions.activate_requeue(
            plan.plan_id,
            verified,
            candidate,
            expected_revision=advanced.revision,
        )
    release.set()
    _wait(controller.dispatch._inflight[plan.plan_id][1])
    controller.dispatch.harvest()

    saved = collections.get(plan.plan_id, runtime_handle.attempt)
    if activate_replacement:
        assert saved is None
        assert collections.history(plan.plan_id) == []
    else:
        assert saved.result == collected
        assert saved.handle == runtime_handle
        assert json.loads(saved.handle.raw_json)["restart_count"] == 0
        assert collections.history(plan.plan_id) == [saved]
    assert executions.get(plan.plan_id).native_identity == advanced.native_identity
    derived = controller.derive(plan.plan_id)
    assert derived.runtime.outcome is None
    assert derived.state != RunState.COMPLETED


def test_late_native_cancel_callback_is_fenced_after_new_head(bound_runtime):
    """A cancellation response for generation 1 cannot acknowledge generation 2."""
    controller, plan, _, backend, executions, _, _ = bound_runtime
    _advance(bound_runtime, 1, "RUNNING")

    started = threading.Event()
    release = threading.Event()

    def request_cancel(handle):
        started.set()
        assert release.wait(5)

    backend.request_cancel = request_cancel
    controller.request_cancel(plan.plan_id)
    until(
        controller,
        lambda: (
            plan.plan_id in controller.dispatch._inflight
            and controller.dispatch._inflight[plan.plan_id][0] == "cancel"
        ),
    )
    assert started.wait(5)
    current = _advance(bound_runtime, 2, "RUNNING")
    assert current.native_identity == identity(backend.handle, 2)
    release.set()
    _wait(controller.dispatch._inflight[plan.plan_id][1])
    controller.dispatch.harvest()

    current = executions.get(plan.plan_id)
    assert current.native_identity == identity(backend.handle, 2)
    assert not current.cancel_delivered
    assert plan.plan_id not in controller.dispatch._inflight


@pytest.mark.parametrize("stage", [0, 7])
def test_late_bound_native_collection_cannot_commit_after_new_head(
    bound_runtime, stage
):
    """An authentic receipt for native 1 remains fenced once native 2 is known."""
    controller, plan, _, backend, executions, collections, _ = bound_runtime
    outcome(bound_runtime, 1, batch=9, stage=stage)
    source = _advance(bound_runtime, 1, "FAILED")
    pending = backend.observer.collect_submission(backend.handle)
    assert pending.identity == identity(backend.handle, 1)
    started = threading.Event()
    release = threading.Event()

    def collect_submission(handle):
        assert handle == source.submission_handle
        started.set()
        assert release.wait(5)
        return pending

    backend.collect_submission = collect_submission
    controller.dispatch.schedule(plan, source)
    assert controller.dispatch._inflight[plan.plan_id][0] == "collect_generation"
    assert started.wait(5)
    advanced = _advance(bound_runtime, 2, "PENDING")
    release.set()
    _wait(controller.dispatch._inflight[plan.plan_id][1])
    controller.dispatch.harvest()

    assert executions.get(plan.plan_id) == advanced
    assert collections.get_submission_generation(plan.plan_id, pending.identity) is None
    assert collections.history(plan.plan_id) == []
    assert plan.plan_id not in controller.dispatch._inflight_sources


def test_pending_fleet_error_does_not_mutate_retained_runtime_evidence(bound_runtime):
    """A scope-less native outage cannot relabel or renegotiate runtime 0."""
    controller, plan, layout, backend, executions, _, clock = bound_runtime
    current = executions.get(plan.plan_id)
    journal(layout, current.attempt)
    events = [
        decode(line)
        for line in layout.journal(current.attempt).read_bytes().splitlines()
    ]
    with RuntimeProvenanceStore(
        executions.path, dispatch_lock=executions.dispatch_lock
    ) as provenance:
        controller.context.provenance_store = provenance
        controller.accept_runtime_events(plan.plan_id, events)
        tracker = controller.handshake._runtime[plan.plan_id]
        negotiation = tracker.result
        protected = {
            table: [
                tuple(row)
                for row in executions._connection.execute(f"SELECT * FROM {table}")
            ]
            for table in (
                "submission_control",
                "initial_bindings",
                "execution_attempt_history",
                "runtime_provenance",
            )
        }
        contents = layout.journal(current.attempt).read_bytes()
        advanced = _advance(bound_runtime, 1, "RUNNING")

        def outage(handles):
            raise RuntimeError("native fleet outage")

        backend.observe_many = outage
        clock.now = 30
        until(
            controller,
            lambda: (
                "native fleet outage"
                in (executions.get(plan.plan_id).latest_native_observation.detail or "")
            ),
        )
        retained = executions.get(plan.plan_id)
        assert retained.native_identity == advanced.native_identity
        assert retained.handle == current.handle
        assert retained.observation == current.observation
        assert retained.latest_observation == current.latest_observation
        assert retained.latest_native_observation.native_identity is None
        assert controller.handshake._runtime[plan.plan_id] is tracker
        assert tracker.result == negotiation
        assert layout.journal(current.attempt).read_bytes() == contents
        for table, before in protected.items():
            assert [
                tuple(row)
                for row in executions._connection.execute(f"SELECT * FROM {table}")
            ] == before
        controller.context.provenance_store = None
