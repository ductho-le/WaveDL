"""Native lifecycle is durable, head-scoped, and independent of runtime state."""

import sqlite3
from dataclasses import replace

import pytest

from unit_tests.daemon import (
    test_native_heads as heads,
    test_writer_submissions as writers,
)
from unit_tests.daemon.test_bound_native_collections import bound, proposal
from wavedl_daemon.backends.contracts import ObservationStatus, SubmissionObservation
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import ExecutionStore, StaleExecution
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.schema import backup_path


writer = writers.writer


def protected(store):
    return {
        table: [
            tuple(row) for row in store._connection.execute(f"SELECT * FROM {table}")
        ]
        for table in (
            "submission_control",
            "initial_bindings",
            "execution_attempt_history",
            "execution_collections",
        )
    }


def test_bound_native_running2_error3_does_not_reuse_known2(writer):
    store, saved, current = bound(writer)
    runtime = current.handle, current.observation, current.latest_observation
    before = protected(store)
    running = replace(
        proposal(current, saved, 2).observation,
        status=ObservationStatus.RUNNING,
        native_state="RUNNING",
    )
    current = store.observe_native_generation(
        saved.plan_id, running, expected_revision=current.revision
    )
    assert current.native_observation == running
    error = replace(
        proposal(current, saved, 3).observation,
        status=ObservationStatus.ERROR,
        native_state=None,
    )
    current = store.observe_native_generation(
        saved.plan_id, error, expected_revision=current.revision
    )
    assert current.native_observation is None
    assert current.latest_native_observation == error
    assert (current.handle, current.observation, current.latest_observation) == runtime
    assert protected(store) == before
    current = store.observe_native_generation(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.ERROR, detail="outage"),
        expected_revision=current.revision,
    )
    assert (
        current.native_observation is None
        and current.native_identity.restart_count == 3
    )
    with ExecutionStore(store.path, read_only=True) as recovered:
        assert recovered.get(saved.plan_id) == current
    assert (
        store._connection.execute(
            "SELECT COUNT(*) FROM submission_native_lifecycle"
        ).fetchone()[0]
        == 3
    )


def test_native_lifecycle_terminal_errors_preserve_known_and_reject_contradiction(
    writer,
):
    store, saved, current = bound(writer)
    terminal = proposal(current, saved, 3).observation
    current = store.observe_native_generation(
        saved.plan_id, terminal, expected_revision=current.revision
    )
    current = store.observe_native_generation(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.ERROR),
        expected_revision=current.revision,
    )
    assert current.native_observation == terminal
    for invalid in (
        replace(terminal, native_state="COMPLETED"),
        replace(terminal, status=ObservationStatus.RUNNING),
        replace(
            terminal, native_identity=replace(terminal.native_identity, restart_count=2)
        ),
        replace(
            terminal, native_identity=replace(terminal.native_identity, owner="foreign")
        ),
    ):
        with pytest.raises(StaleExecution):
            store.observe_native_generation(
                saved.plan_id, invalid, expected_revision=current.revision
            )
        assert store.get(saved.plan_id) == current


def test_native_lifecycle_cas_rollback_includes_new_head(writer):
    store, saved, current = bound(writer)
    terminal = proposal(current, saved, 3).observation
    with pytest.raises(StaleExecution, match="revision"):
        store.observe_native_generation(
            saved.plan_id, terminal, expected_revision=current.revision - 1
        )
    store._connection.execute(
        "CREATE TRIGGER fail_native_update BEFORE UPDATE ON execution_control BEGIN SELECT RAISE(ABORT,'native CAS abort'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="CAS abort"):
        store.observe_native_generation(
            saved.plan_id, terminal, expected_revision=current.revision
        )
    assert store.get(saved.plan_id) == current
    assert (
        store._connection.execute(
            "SELECT COUNT(*) FROM submission_native_lifecycle WHERE restart_count=3"
        ).fetchone()[0]
        == 0
    )


def test_archived_native_lifecycle_round_trip_keeps_typed_observations(writer):
    store, saved, current = bound(writer)
    terminal = proposal(current, saved, 2).observation
    current = store.observe_native_generation(
        saved.plan_id, terminal, expected_revision=current.revision
    )
    before = current
    next_binding = replace(
        writers.binding(current), attempt=current.attempt + 1, restart_count=3
    )
    import json

    next_handle = replace(
        current.handle,
        attempt=next_binding.attempt,
        raw_json=json.dumps(json.loads(current.handle.raw_json) | {"restart_count": 3}),
    )
    store.activate_requeue(
        saved.plan_id, next_binding, next_handle, expected_revision=current.revision
    )
    archived = store.history(saved.plan_id)[0]
    assert archived == before
    assert isinstance(archived.native_observation, SubmissionObservation)
    assert isinstance(archived.latest_native_observation, SubmissionObservation)
    assert archived.native_observation.native_identity == terminal.native_identity
    with ExecutionStore(store.path, read_only=True) as reader:
        assert reader.history(saved.plan_id)[0] == before


def test_native_lifecycle_unbound_legacy_observation_same_transaction(writer):
    store, saved, current = heads.prepare(writer)
    terminal = proposal(current, saved, 3).observation
    current = store.observe_submission(
        saved.plan_id, terminal, expected_revision=current.revision
    )
    assert current.native_observation == current.submission_observation == terminal
    current = store.observe_submission(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.ERROR),
        expected_revision=current.revision,
    )
    assert current.native_observation == terminal
    assert current.latest_native_observation == current.latest_submission_observation


def test_native_lifecycle_requires_current_terminal_proof_for_bound_collection(writer):
    store, saved, current = bound(writer)
    pending = proposal(current, saved, 3)
    current = store.record_native_generation(
        saved.plan_id, pending.identity, expected_revision=current.revision
    )
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        with pytest.raises(StaleExecution, match="terminal lifecycle"):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                pending,
                expected_revision=current.revision,
            )
        current = store.observe_native_generation(
            saved.plan_id, pending.observation, expected_revision=current.revision
        )
        assert (
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                pending,
                expected_revision=current.revision,
            ).observation
            == pending.observation
        )


def test_native_lifecycle_actual_schema12_forward_migration_keeps_backup(writer):
    store, saved, current = heads.prepare(writer)
    running = replace(
        proposal(current, saved, 2).observation,
        status=ObservationStatus.RUNNING,
        native_state="RUNNING",
    )
    current = store.observe_submission(saved.plan_id, running)
    error = replace(
        proposal(current, saved, 3).observation,
        status=ObservationStatus.ERROR,
        native_state=None,
    )
    current = store.observe_submission(saved.plan_id, error)
    store._connection.execute("DROP TABLE submission_native_lifecycle")
    store._connection.execute("DELETE FROM schema_migrations WHERE to_version=13")
    store._connection.execute("PRAGMA user_version=12")
    before = protected(store)
    with Registry(store.path, dispatch_lock=store.dispatch_lock):
        migrated = store.get(saved.plan_id)
        assert migrated.native_observation is None
        assert migrated.latest_native_observation == error
        assert migrated.submission_observation == running
        assert protected(store) == before
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 13
    with sqlite3.connect(
        backup_path(store.path, "0013_submission_native_lifecycle")
    ) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 12
        assert (
            backup.execute(
                "SELECT 1 FROM sqlite_master WHERE name='submission_native_lifecycle'"
            ).fetchone()
            is None
        )


@pytest.mark.parametrize("damage", ["known_json='{}'", "source_revision=999999"])
def test_native_lifecycle_damage_requires_recovery(writer, damage):
    store, saved, current = bound(writer)
    current = store.observe_native_generation(
        saved.plan_id,
        proposal(current, saved, 3).observation,
        expected_revision=current.revision,
    )
    store._connection.execute(
        f"UPDATE submission_native_lifecycle SET {damage} WHERE restart_count=3"
    )
    with pytest.raises(PlanRecoveryRequired, match="lifecycle"):
        store.get(saved.plan_id)
