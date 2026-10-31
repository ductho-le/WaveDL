"""Native proofs cannot regress when lifecycle or diagnostic evidence disappears."""

import hashlib
import sqlite3
from dataclasses import replace

import pytest

from unit_tests.daemon import test_writer_submissions as writers
from unit_tests.daemon.test_submission_generations import native_result
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    SubmissionCollectionResult,
    SubmissionObservation,
)
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import (
    ExecutionStore,
    StaleExecution,
    _manifest_sha256,
)
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.schema import backup_path
from wavedl_daemon.storage.submission_generations import receipt_identity


writer = writers.writer


def prepare(writer):
    store, plans, original = writer
    saved = plans.put(
        replace(original.plan, resources=(("requeue", True),)),
        idempotency_key="native-head",
        request_payload={},
    )
    plans.request_submission(saved.plan_id)
    current = writers.prepare((store, plans, saved))
    return store, saved, current


def observation(current, saved, count, status=ObservationStatus.ERROR):
    return SubmissionObservation(
        status,
        native_identity=receipt_identity(current.submission_handle, saved.plan, count),
    )


def test_running2_error3_unknown_error_never_permits_binding2(writer):
    store, saved, current = prepare(writer)
    running = observation(current, saved, 2, ObservationStatus.RUNNING)
    current = store.observe_submission(saved.plan_id, running)
    error = observation(current, saved, 3)
    current = store.observe_submission(saved.plan_id, error)
    first = tuple(
        store._connection.execute(
            "SELECT * FROM submission_native_generations WHERE restart_count=3"
        ).fetchone()
    )
    current = store.observe_submission(
        saved.plan_id, SubmissionObservation(ObservationStatus.ERROR)
    )
    assert current.native_identity == error.native_identity
    assert current.submission_observation == running
    assert current.latest_submission_observation.native_identity is None
    with pytest.raises(StaleExecution, match="head"):
        store.activate_initial(
            saved.plan_id,
            replace(writers.binding(current), restart_count=2),
            expected_revision=current.revision,
        )
    assert store.get(saved.plan_id) == current
    current = store.observe_submission(
        saved.plan_id, replace(error, detail="new diagnostic")
    )
    assert (
        tuple(
            store._connection.execute(
                "SELECT * FROM submission_native_generations WHERE restart_count=3"
            ).fetchone()
        )
        == first
    )
    current = store.activate_initial(
        saved.plan_id,
        replace(writers.binding(current), restart_count=3),
        expected_revision=current.revision,
    )
    assert current.native_identity == error.native_identity
    assert current.observation is None
    with (
        ExecutionStore(store.path, read_only=True) as recovered,
        CollectionStore(store.path, read_only=True) as collections,
    ):
        assert recovered.get(saved.plan_id) == current
        assert recovered.list_pending()[-1].native_identity == error.native_identity
        assert collections.get_native_head(saved.plan_id) == error.native_identity


@pytest.mark.parametrize("status", [ObservationStatus.ERROR, ObservationStatus.ABSENT])
def test_error_absent_proof_atomic_revision_foreign_and_late_fences(writer, status):
    store, saved, current = prepare(writer)
    proof = observation(current, saved, 3, status)
    with pytest.raises(StaleExecution, match="revision"):
        store.observe_submission(
            saved.plan_id, proof, expected_revision=current.revision - 1
        )
    assert store.get(saved.plan_id).native_identity is None
    store._connection.execute(
        "CREATE TRIGGER fail_diagnostic BEFORE UPDATE ON submission_control BEGIN SELECT RAISE(ABORT,'test CAS abort'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="CAS abort"):
        store.observe_submission(saved.plan_id, proof)
    store._connection.execute("DROP TRIGGER fail_diagnostic")
    assert (
        store._connection.execute(
            "SELECT COUNT(*) FROM submission_native_generations"
        ).fetchone()[0]
        == 0
    )
    current = store.observe_submission(saved.plan_id, proof)
    for invalid in (
        replace(proof, native_identity=replace(proof.native_identity, owner="foreign")),
        observation(current, saved, 2, status),
    ):
        with pytest.raises(StaleExecution):
            store.observe_submission(saved.plan_id, invalid)
        assert store.get(saved.plan_id) == current
    for sql in (
        "DELETE FROM submission_native_generations",
        "UPDATE submission_native_generations SET source_revision=0",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute(sql)


def downgrade(store):
    store._connection.execute("DROP TABLE submission_native_lifecycle")
    store._connection.execute("DELETE FROM schema_migrations WHERE to_version=13")
    store._connection.execute("DROP TABLE submission_native_generations")
    store._connection.execute("DELETE FROM schema_migrations WHERE to_version=12")
    store._connection.execute("PRAGMA user_version=11")


def test_actual_upgrade_seeds_latest_error_not_older_good_and_keeps_backup(writer):
    store, saved, current = prepare(writer)
    current = store.observe_submission(
        saved.plan_id, observation(current, saved, 2, ObservationStatus.RUNNING)
    )
    current = store.observe_submission(saved.plan_id, observation(current, saved, 3))
    before = tuple(
        store._connection.execute("SELECT * FROM submission_control").fetchone()
    )
    downgrade(store)
    assert (
        store._connection.execute("SELECT COUNT(*) FROM initial_bindings").fetchone()[0]
        == 0
    )
    with Registry(store.path, dispatch_lock=store.dispatch_lock):
        assert store.get(saved.plan_id).native_identity.restart_count == 3
        assert (
            tuple(
                store._connection.execute("SELECT * FROM submission_control").fetchone()
            )
            == before
        )
    with sqlite3.connect(
        backup_path(store.path, "0012_submission_native_generations")
    ) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 11
        assert backup.execute("SELECT * FROM submission_control").fetchone() == before


def test_actual_upgrade_seeds_unbound_finalized_collection_without_observations(writer):
    store, saved, current = prepare(writer)
    terminal = observation(current, saved, 3, ObservationStatus.TERMINAL)
    current = store.observe_submission(saved.plan_id, terminal)
    proposal = SubmissionCollectionResult(
        terminal.native_identity,
        hashlib.sha256(saved.plan.to_json().encode()).hexdigest(),
        _manifest_sha256(saved.plan),
        native_result(),
        terminal,
    )
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_submission_generation(
            saved.plan_id,
            current.submission_handle,
            proposal,
            expected_revision=current.revision,
        )
    downgrade(store)
    store._connection.execute(
        "UPDATE submission_control SET submission_observation_json=NULL,latest_submission_observation_json=NULL WHERE plan_id=?",
        (saved.plan_id,),
    )
    before = tuple(
        store._connection.execute(
            "SELECT * FROM submission_generation_collections"
        ).fetchone()
    )
    with Registry(store.path, dispatch_lock=store.dispatch_lock):
        recovered = store.get(saved.plan_id)
        assert recovered.native_identity == terminal.native_identity
        assert recovered.attempt is None
        assert recovered.submission_observation is None
        assert (
            tuple(
                store._connection.execute(
                    "SELECT * FROM submission_generation_collections"
                ).fetchone()
            )
            == before
        )


def test_actual_upgrade_seeds_verified_binding_when_no_typed_lifecycle(writer):
    store, saved, current = prepare(writer)
    current = store.activate_initial(
        saved.plan_id,
        replace(writers.binding(current), restart_count=3),
        expected_revision=current.revision,
    )
    assert current.native_identity.restart_count == 3
    downgrade(store)
    with Registry(store.path, dispatch_lock=store.dispatch_lock):
        assert store.get(saved.plan_id).native_identity.restart_count == 3


def test_damaged_native_head_read_requires_recovery(writer):
    store, saved, current = prepare(writer)
    store.observe_submission(saved.plan_id, observation(current, saved, 3))
    store._connection.execute("DROP TRIGGER submission_native_generations_no_update")
    store._connection.execute(
        "UPDATE submission_native_generations SET first_observation_json='{}'"
    )
    with pytest.raises(PlanRecoveryRequired, match="head is damaged"):
        store.get(saved.plan_id)
    with (
        CollectionStore(store.path, read_only=True) as collections,
        pytest.raises(PlanRecoveryRequired, match="head is damaged"),
    ):
        collections.get_native_head(saved.plan_id)


def test_active_binding_can_lag_later_held_head_without_recovery_damage(writer):
    store, saved, current = prepare(writer)
    current = store.activate_initial(
        saved.plan_id,
        replace(writers.binding(current), restart_count=2),
        expected_revision=current.revision,
    )
    current = store.record_native_generation(
        saved.plan_id,
        observation(current, saved, 3).native_identity,
        expected_revision=current.revision,
    )
    with ExecutionStore(store.path, read_only=True) as recovered:
        record = recovered.get(saved.plan_id)
        assert record.native_identity.restart_count == 3
        assert record.handle == current.handle
        assert record.attempt == current.attempt
    controller = SubmissionController(
        writer[1],
        store,
        {},
        store.dispatch_lock,
    )
    try:
        derived = controller.derive(saved.plan_id)
        assert derived.state == RunState.OBSERVATION_UNAVAILABLE
        assert derived.confidence == "unconfirmed"
        assert "runtime binding unavailable" in derived.reason
        before = store.get(saved.plan_id)
        with pytest.raises(StaleExecution, match="behind"):
            store.record_native_generation(
                saved.plan_id,
                observation(current, saved, 2).native_identity,
                expected_revision=before.revision,
            )
        assert store.get(saved.plan_id) == before
    finally:
        controller.close()
