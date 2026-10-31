"""Scheduler generation authority survives unknown states and query outages."""

import time
from dataclasses import asdict

from unit_tests.daemon.test_slurm_writer_submission import identity
from unit_tests.daemon.test_submission_generation_controller import outcome
from unit_tests.daemon.test_writer_fleet_controller import until
from wavedl.core.scheduler_attempt import coordinate_attempt
from wavedl_daemon.backends.contracts import ObservationStatus, OutcomeStatus
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.runs.control.records import native_identity, same_generation
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.executions import ExecutionStore
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessResult


pytest_plugins = ["unit_tests.daemon.test_submission_generation_controller"]


def observe_generation(runtime, generation, state, exit_code="0:0"):
    controller, plan, _, backend, executions, _, _ = runtime
    backend.native(generation, state, exit_code)
    observation = backend.observe(backend.handle)
    record = executions.record_observation(plan.plan_id, observation)
    return controller, record


def head_after_unknown_state(runtime):
    observe_generation(runtime, 2, "PENDING")
    _controller, record = observe_generation(runtime, 3, "FUTURE_UNKNOWN")
    assert record.submission_observation.status == ObservationStatus.QUEUED
    assert record.submission_observation.native_identity == identity(
        record.submission_handle, 2
    )
    assert record.latest_submission_observation.status == ObservationStatus.ERROR
    assert record.latest_submission_observation.native_identity == identity(
        record.submission_handle, 3
    )
    assert record.native_identity == identity(record.submission_handle, 3)
    assert native_identity(record) == record.native_identity
    return record


def wait_work(work):
    deadline = time.monotonic() + 5
    while not work.done() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert work.done()


def public_service(runtime):
    controller, _, layout, backend, executions, collections, _ = runtime
    return RunService(
        Settings(layout.run_dir.parent.parent / "state", layout.run_dir.parent),
        controller.plan_store,
        executions,
        controller,
        {"slurm": backend},
        collections=collections,
    )


def public_row(runtime):
    _, plan, _, _, _, _, _ = runtime
    page = public_service(runtime).attempts(plan.plan.run_id)
    assert page["total"] == 1
    return page["attempts"][0]


def test_native_head_survives_unknown_state_query_outage_and_reopen(native_runtime):
    controller, plan, layout, backend, executions, collections, clock = native_runtime
    unknown = head_after_unknown_state(native_runtime)
    backend.runner.responses.update(
        {
            name: ProcessResult(1, b"", b"scheduler outage")
            for name in ("scontrol", "sacct", "squeue")
        }
    )
    outage = backend.observe(backend.handle)
    assert outage.status == ObservationStatus.ERROR
    assert outage.native_identity is None
    current = executions.record_observation(plan.plan_id, outage)
    assert current.submission_observation == unknown.submission_observation
    assert current.latest_submission_observation == outage
    assert current.native_identity == identity(backend.handle, 3)
    assert current.attempt is None and current.handle is None
    assert list(layout.attempts.iterdir()) == []
    assert not list(layout.run_dir.rglob("events.jsonl"))
    database = executions._connection.execute("PRAGMA database_list").fetchone()[2]
    assert executions._connection.execute("PRAGMA user_version").fetchone()[0] == 13
    controller.close()
    with ExecutionStore(database, dispatch_lock=controller.dispatch_lock) as reopened:
        restored = reopened.get(plan.plan_id)
        assert restored.native_identity == current.native_identity
        assert restored.submission_observation == current.submission_observation
        assert restored.latest_submission_observation == outage
        restarted = SubmissionController(
            controller.plan_store,
            reopened,
            {"slurm": backend},
            controller.dispatch_lock,
            WorkBudget(max_workers=4, max_pending=4),
            collection_store=collections,
            clock=clock,
        )
        restarted.fleet._fleets["slurm"].clock = clock
        try:
            assert native_identity(restored) == identity(backend.handle, 3)
            for _ in range(10):
                restarted.reconcile_once()
            assert reopened.get(plan.plan_id).native_identity == current.native_identity
            assert reopened.get(plan.plan_id).attempt is None
        finally:
            restarted.close()


def test_binding2_is_rejected_when_admitted_source_already_has_error_head3(
    native_runtime,
):
    controller, plan, layout, backend, executions, _, clock = native_runtime
    stale = coordinate_attempt(
        layout.run_dir, identity(backend.handle, 2), initial_attempt=None
    )
    assert stale.attempt == 1
    head_after_unknown_state(native_runtime)
    backend.discover_initial = lambda handle: stale
    clock.now = 5
    until(
        controller,
        lambda: (
            plan.plan_id in controller.dispatch._inflight
            and controller.dispatch._inflight[plan.plan_id][0] == "startup"
        ),
    )
    source = controller.dispatch._inflight_sources[plan.plan_id]
    assert source.native_identity == identity(backend.handle, 3)
    assert source.submission_observation.native_identity == identity(backend.handle, 2)
    wait_work(controller.dispatch._inflight[plan.plan_id][1])
    controller.dispatch.harvest()
    current = executions.get(plan.plan_id)
    assert current.attempt is None and current.handle is None
    assert current.native_identity == identity(backend.handle, 3)
    assert current.submission_observation == source.submission_observation
    assert current.latest_submission_observation.status == ObservationStatus.ERROR
    assert "Startup binding differs" in current.latest_submission_observation.detail
    assert (
        executions._connection.execute(
            "SELECT count(*) FROM initial_bindings WHERE plan_id=?", (plan.plan_id,)
        ).fetchone()[0]
        == 0
    )
    assert plan.plan_id not in controller.dispatch._inflight_sources
    assert list(layout.attempts.iterdir()) == [layout.attempt_dir(stale.attempt)]


def test_startup_callback_from_source2_is_fenced_after_error_head3(native_runtime):
    controller, plan, layout, backend, executions, _, clock = native_runtime
    stale = coordinate_attempt(
        layout.run_dir, identity(backend.handle, 2), initial_attempt=None
    )
    observe_generation(native_runtime, 2, "PENDING")
    backend.discover_initial = lambda handle: stale
    clock.now = 5
    until(
        controller,
        lambda: (
            plan.plan_id in controller.dispatch._inflight
            and controller.dispatch._inflight[plan.plan_id][0] == "startup"
        ),
    )
    source = controller.dispatch._inflight_sources[plan.plan_id]
    assert source.native_identity == identity(backend.handle, 2)
    wait_work(controller.dispatch._inflight[plan.plan_id][1])
    _, advanced = observe_generation(native_runtime, 3, "FUTURE_UNKNOWN")
    assert advanced.native_identity == identity(backend.handle, 3)
    assert not same_generation(advanced, source)
    controller.dispatch.harvest()
    current = executions.get(plan.plan_id)
    assert current.attempt is None and current.handle is None
    assert current.native_identity == advanced.native_identity
    assert (
        current.latest_submission_observation == advanced.latest_submission_observation
    )
    assert plan.plan_id not in controller.dispatch._inflight
    assert plan.plan_id not in controller.dispatch._inflight_sources
    assert plan.plan_id not in controller.dispatch._receipts


def test_collection2_is_not_current_at_head3_and_exact_receipt3_is_public(
    native_runtime, monkeypatch
):
    controller, plan, layout, backend, executions, collections, clock = native_runtime
    outcome(native_runtime, 2, batch=9, stage=7)
    backend.native(2, "FAILED", "9:0")
    clock.now = 30
    two = identity(backend.handle, 2)
    until(
        controller,
        lambda: collections.get_submission_generation(plan.plan_id, two) is not None,
    )
    saved_two = collections.get_submission_generation(plan.plan_id, two)
    _, uncertain = observe_generation(native_runtime, 3, "FUTURE_UNKNOWN")
    assert uncertain.submission_observation.native_identity == two
    assert uncertain.native_identity == identity(backend.handle, 3)
    assert controller.dispatch.submission_collection(uncertain) is None
    row = public_row(native_runtime)
    assert row["native_identity"] == asdict(identity(backend.handle, 3))
    assert row["submission_collection"] is None
    assert row["collection"] is None and row["runtime"] is None
    outcome(native_runtime, 3, batch=11, stage=8)
    backend.native(3, "FAILED", "11:0")
    clock.now = 60
    three = identity(backend.handle, 3)
    until(
        controller,
        lambda: collections.get_submission_generation(plan.plan_id, three) is not None,
    )
    saved_three = collections.get_submission_generation(plan.plan_id, three)
    assert collections.get_submission_generation(plan.plan_id, two) == saved_two
    assert saved_three.result.runtime.status == OutcomeStatus.UNKNOWN
    assert saved_three.result.runtime.exit_code is None
    assert saved_three.result.batch.exit_code == 11
    assert saved_three.result.stage_out.exit_code == 8
    row = public_row(native_runtime)
    assert row["submission_collection"] == saved_three.proposal.to_dict()
    assert row["submission_collection"]["identity"]["restart_count"] == 3
    assert row["native_identity"] == asdict(three)
    assert row["attempt"] is None and row["handle"] is None
    assert row["collection"] is None and row["runtime"] is None
    assert row["scheduler_binding"] is None
    derived = controller.derive(plan.plan_id)
    assert derived.runtime.hello_seen is False
    assert derived.runtime.outcome is None
    assert not list(layout.attempts.iterdir())
    assert not list(layout.run_dir.rglob("events.jsonl"))
    assert executions.get(plan.plan_id).native_identity == three
    current = executions.get(plan.plan_id)
    monkeypatch.setattr(
        controller.dispatch, "submission_collection", lambda record: saved_two
    )
    guarded = controller.derive(plan.plan_id)
    assert guarded.backend.native_identity == three
    assert guarded.batch == current.latest_native_observation.batch
    assert guarded.stage_out == current.latest_native_observation.stage_out
    assert guarded.last_observation.batch == current.native_observation.batch
    assert guarded.last_observation.stage_out == current.native_observation.stage_out


def test_collection_discovers_terminal_head_before_fleet_snapshot(native_runtime):
    """Exact current lifecycle proof does not depend on fleet callback ordering."""
    controller, plan, _, backend, executions, collections, clock = native_runtime
    observe_generation(native_runtime, 2, "FAILED", "9:0")
    _, source = observe_generation(native_runtime, 3, "FUTURE_UNKNOWN")
    uncertain = controller.derive(plan.plan_id)
    assert uncertain.backend.native_identity == identity(backend.handle, 3)
    assert uncertain.backend.status == ObservationStatus.ERROR
    assert uncertain.last_observation is None
    assert uncertain.batch is None and uncertain.stage_out is None
    uncertain_view = public_service(native_runtime).get(plan.plan.run_id)
    assert uncertain_view.uncertainty is True
    assert uncertain_view.evidence["last_observation"] is None
    outcome(native_runtime, 3, batch=11, stage=8)
    backend.native(3, "FAILED", "11:0")
    clock.now = 60
    controller.dispatch.schedule(plan, source)
    assert controller.dispatch._inflight[plan.plan_id][0] == "collect_generation"
    wait_work(controller.dispatch._inflight[plan.plan_id][1])
    # No fleet tick runs between admission and harvest: the collection query
    # itself is the first terminal lifecycle proof for generation three.
    controller.dispatch.harvest()
    current = executions.get(plan.plan_id)
    assert current.submission_observation.native_identity == identity(backend.handle, 2)
    assert current.native_observation.status == ObservationStatus.TERMINAL
    assert current.native_observation.native_identity == identity(backend.handle, 3)
    saved = collections.get_submission_generation(
        plan.plan_id, identity(backend.handle, 3)
    )
    assert saved is not None
    assert saved.result.batch.exit_code == 11
    assert saved.result.stage_out.exit_code == 8
    assert plan.plan_id not in controller.dispatch._collection_due
    derived = controller.derive(plan.plan_id)
    assert derived.backend.native_identity == identity(backend.handle, 3)
    assert derived.last_observation.native_identity == identity(backend.handle, 3)
    assert derived.batch.exit_code == 11
    assert derived.stage_out.exit_code == 8
    view = public_service(native_runtime).get(plan.plan.run_id)
    assert view.evidence["backend"]["native_identity"] == asdict(
        identity(backend.handle, 3)
    )
    assert view.evidence["batch"]["exit_code"] == 11
    assert view.evidence["stage_out"]["exit_code"] == 8


def test_cached_terminal_observation_does_not_churn_source_revision(native_runtime):
    controller, plan, _, backend, executions, _, _ = native_runtime
    observe_generation(native_runtime, 2, "FAILED", "9:0")
    observation = backend.observe(backend.handle)
    before = executions.get(plan.plan_id)
    for _ in range(10):
        controller.observations.record(plan.plan_id, observation)
    assert executions.get(plan.plan_id) == before


def test_cached_native_observation_still_populates_bound_runtime(native_runtime):
    controller, plan, layout, backend, executions, _, _ = native_runtime
    current = executions.get(plan.plan_id)
    binding = coordinate_attempt(
        layout.run_dir, identity(backend.handle, 0), initial_attempt=None
    )
    current = executions.activate_initial(
        plan.plan_id, binding, expected_revision=current.revision
    )
    backend.native(0, "RUNNING")
    observation = backend.observe(backend.handle)
    current = executions.observe_native_generation(
        plan.plan_id, observation, expected_revision=current.revision
    )
    assert current.latest_observation.status == ObservationStatus.QUEUED
    controller.observations.record(plan.plan_id, observation)
    current = executions.get(plan.plan_id)
    assert current.latest_observation == observation.bind_attempt(binding.attempt)
    controller.observations.record(plan.plan_id, observation)
    assert executions.get(plan.plan_id) == current


def test_collection_callback_source2_cannot_commit_after_error_head3(native_runtime):
    controller, plan, layout, backend, executions, collections, _ = native_runtime
    outcome(native_runtime, 2, batch=9, stage=7)
    _, source = observe_generation(native_runtime, 2, "FAILED")
    controller.dispatch.schedule(plan, source)
    assert controller.dispatch._inflight[plan.plan_id][0] == "collect_generation"
    wait_work(controller.dispatch._inflight[plan.plan_id][1])
    _, advanced = observe_generation(native_runtime, 3, "FUTURE_UNKNOWN")
    assert advanced.native_identity == identity(backend.handle, 3)
    assert not same_generation(advanced, source)
    controller.dispatch.harvest()
    current = executions.get(plan.plan_id)
    assert current.native_identity == advanced.native_identity
    assert (
        current.latest_submission_observation == advanced.latest_submission_observation
    )
    assert current.attempt is None
    assert (
        collections.get_submission_generation(plan.plan_id, identity(backend.handle, 2))
        is None
    )
    assert (
        collections.get_submission_generation(plan.plan_id, identity(backend.handle, 3))
        is None
    )
    assert plan.plan_id not in controller.dispatch._inflight_sources
    assert not list(layout.attempts.iterdir())


def test_fleet_error_head_change_does_not_repeat_poll_without_clock_advance(
    native_runtime,
):
    controller, plan, _, backend, executions, _, clock = native_runtime
    backend.native(2, "PENDING")
    clock.now = 30
    until(
        controller,
        lambda: (
            executions.get(plan.plan_id).native_identity == identity(backend.handle, 2)
        ),
    )
    backend.native(3, "FUTURE_UNKNOWN")
    clock.now = 60
    until(
        controller,
        lambda: (
            executions.get(plan.plan_id).native_identity == identity(backend.handle, 3)
        ),
    )
    # Let a changed durable scope reset at most once, then retain ERROR backoff.
    for _ in range(30):
        controller.reconcile_once()
        time.sleep(0.002)
    calls = len(backend.batches)
    assert controller.fleet._fleets["slurm"].snapshots[0].next_due > clock.now
    for _ in range(30):
        controller.reconcile_once()
        time.sleep(0.002)
    assert len(backend.batches) == calls
    current = executions.get(plan.plan_id)
    assert current.native_identity == identity(backend.handle, 3)
    assert current.submission_observation.native_identity == identity(backend.handle, 2)
    assert current.latest_submission_observation.status == ObservationStatus.ERROR
