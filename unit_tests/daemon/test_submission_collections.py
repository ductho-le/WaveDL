"""Native setup outcomes remain durable and independent of runtime attempts."""

import json
import sqlite3
from dataclasses import replace

import pytest

from unit_tests.daemon import test_writer_submissions as writer_tests
from unit_tests.daemon.test_collections import result
from wavedl_daemon.backends.contracts import OutcomeStatus
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import StaleExecution
from wavedl_daemon.storage.ownership import DispatchRequired
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.schema import (
    Migration,
    MigrationFailed,
    backup_path,
    migrate,
)
from wavedl_daemon.storage.submission_collections import SUBMISSION_COLLECTION_MIGRATION


writer = writer_tests.writer
binding = writer_tests.binding
prepare = writer_tests.prepare


def test_unbound_native_collection_is_private_durable_and_immutable(writer):
    store, _, saved = writer
    native = prepare(writer).submission_handle
    outcome = result(runtime=OutcomeStatus.UNKNOWN, batch=OutcomeStatus.FAILED)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_submission_collection(
            saved.plan_id, native, outcome
        )
        assert record.handle == native and record.result == outcome
        assert collections.get(saved.plan_id) is None
        assert collections.history(saved.plan_id) == []
        assert (
            collections.record_submission_collection(saved.plan_id, native, outcome)
            == record
        )
        for query in (
            "DELETE FROM submission_collections",
            "UPDATE submission_collections SET observed_at='changed'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                collections._connection.execute(query)
        with pytest.raises(StaleExecution, match="Conflicting"):
            collections.record_submission_collection(
                saved.plan_id,
                native,
                replace(
                    outcome, stage_out=result(stage_out=OutcomeStatus.FAILED).stage_out
                ),
            )
    with CollectionStore(store.path, read_only=True) as reader:
        assert reader.get_submission(saved.plan_id) == record
        with pytest.raises(PermissionError):
            reader.record_submission_collection(saved.plan_id, native, outcome)
    with CollectionStore(store.path) as unowned, pytest.raises(DispatchRequired):
        unowned.record_submission_collection(saved.plan_id, native, outcome)


@pytest.mark.parametrize("field", ["allocation_mode", "plan_sha256", "operation"])
def test_collection_recovery_rechecks_native_writer_approval(writer, field):
    store, _, saved = writer
    native = prepare(writer).submission_handle
    outcome = result(runtime=OutcomeStatus.UNKNOWN, batch=OutcomeStatus.FAILED)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_submission_collection(saved.plan_id, native, outcome)
        raw = json.loads(native.raw_json)
        raw.pop(field)
        corrupt = replace(native, raw_json=json.dumps(raw))
        collections._connection.execute("DROP TRIGGER submission_collections_no_update")
        collections._connection.execute("DROP TRIGGER submission_receipt_immutable")
        # Mutate both copies: matching native receipt bytes alone cannot certify
        # the exact writer allocation mode, approved digest and operation.
        collections._connection.execute(
            "UPDATE submission_control SET submission_handle_json=? WHERE plan_id=?",
            (corrupt.to_json(), saved.plan_id),
        )
        collections._connection.execute(
            "UPDATE submission_collections SET handle_json=? WHERE plan_id=?",
            (corrupt.to_json(), saved.plan_id),
        )
        with pytest.raises(
            PlanRecoveryRequired, match="collection evidence is damaged"
        ):
            collections.get_submission(saved.plan_id)


def test_unknown_foreign_positive_and_stale_native_outcomes_are_not_certified(writer):
    store, _, saved = writer
    current = prepare(writer)
    native = current.submission_handle
    outcome = result(runtime=OutcomeStatus.UNKNOWN, batch=OutcomeStatus.FAILED)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        assert (
            collections.record_submission_collection(
                saved.plan_id,
                native,
                result(runtime=OutcomeStatus.UNKNOWN, batch=OutcomeStatus.UNKNOWN),
            )
            is None
        )
        for foreign, evidence in (
            (replace(native, job_id="456;alpha"), outcome),
            (native, result()),
        ):
            with pytest.raises(StaleExecution):
                collections.record_submission_collection(
                    saved.plan_id, foreign, evidence
                )
        record = collections.record_submission_collection(
            saved.plan_id, native, outcome
        )
        store.activate_initial(
            saved.plan_id, binding(current), expected_revision=current.revision
        )
        assert collections.get_submission(saved.plan_id) == record
        with pytest.raises(StaleExecution, match="current unbound"):
            collections.record_submission_collection(saved.plan_id, native, outcome)
        assert collections.get(saved.plan_id) is None


def test_real_schema9_upgrade_preserves_native_control_and_backs_it_up(writer):
    store, _, saved = writer
    native = prepare(writer).submission_handle
    connection = store._connection
    before = tuple(connection.execute("SELECT * FROM submission_control").fetchone())
    connection.execute("DROP TABLE submission_generation_collections")
    connection.execute("DELETE FROM schema_migrations WHERE to_version=11")
    connection.execute("DROP TABLE submission_collections")
    connection.execute("DELETE FROM schema_migrations WHERE to_version=10")
    connection.execute("PRAGMA user_version=9")
    with Registry(store.path, dispatch_lock=store.dispatch_lock):
        assert (
            tuple(connection.execute("SELECT * FROM submission_control").fetchone())
            == before
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 11
    backup = sqlite3.connect(backup_path(store.path, "0010_submission_collections"))
    try:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 9
        assert backup.execute("SELECT * FROM submission_control").fetchone() == before
    finally:
        backup.close()
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        assert (
            collections.record_submission_collection(
                saved.plan_id,
                native,
                result(runtime=OutcomeStatus.UNKNOWN, batch=OutcomeStatus.FAILED),
            ).handle
            == native
        )


@pytest.mark.parametrize("change", ["owner", "cluster", "generation", "operation"])
def test_foreign_native_identity_and_operation_never_persist(writer, change):
    store, _, saved = writer
    native = prepare(writer).submission_handle
    outcome = result(runtime=OutcomeStatus.UNKNOWN, batch=OutcomeStatus.FAILED)
    if change == "owner":
        native = replace(native, owner="mallory")
    elif change == "cluster":
        raw = json.loads(native.raw_json)
        raw["cluster"] = "beta"
        native = replace(native, job_id="123;beta", raw_json=json.dumps(raw))
    elif change == "generation":
        raw = json.loads(native.raw_json)
        raw["restart_count"] = 1
        native = replace(native, raw_json=json.dumps(raw))
    else:
        outcome = replace(outcome, operation=writer_tests.OperationKind.TEST)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        with pytest.raises(StaleExecution):
            collections.record_submission_collection(saved.plan_id, native, outcome)
        assert collections.get_submission(saved.plan_id) is None


def test_native_collection_damage_and_lost_proof_require_recovery(writer):
    store, _, saved = writer
    native = prepare(writer).submission_handle
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_submission_collection(
            saved.plan_id, native, result(runtime=OutcomeStatus.UNKNOWN)
        )
        collections._connection.execute("DROP TRIGGER submission_collections_no_update")
        collections._connection.execute(
            "UPDATE submission_collections SET result_json='{}'"
        )
        with pytest.raises(PlanRecoveryRequired, match="damaged"):
            collections.get_submission(saved.plan_id)
    store._connection.execute("PRAGMA foreign_keys=OFF")
    store._connection.execute("DELETE FROM submission_control")
    with pytest.raises(PlanRecoveryRequired, match="submission proof"):
        CollectionStore(store.path, read_only=True)


def test_schema10_failure_rolls_back_without_losing_schema9_receipt(writer):
    store, _, _ = writer
    prepare(writer)
    connection = store._connection
    before = tuple(connection.execute("SELECT * FROM submission_control").fetchone())
    connection.execute("DROP TABLE submission_generation_collections")
    connection.execute("DELETE FROM schema_migrations WHERE to_version=11")
    connection.execute("DROP TABLE submission_collections")
    connection.execute("DELETE FROM schema_migrations WHERE to_version=10")
    connection.execute("PRAGMA user_version=9")

    def fail(database):
        SUBMISSION_COLLECTION_MIGRATION.apply(database)
        raise RuntimeError("migration fault")

    with pytest.raises(MigrationFailed, match="rolled back"):
        migrate(
            store.path,
            [Migration("0010_fault", 10, fail)],
            dispatch_lock=store.dispatch_lock,
        )
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
    assert (
        tuple(connection.execute("SELECT * FROM submission_control").fetchone())
        == before
    )
    assert (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE name='submission_collections'"
        ).fetchone()
        is None
    )
