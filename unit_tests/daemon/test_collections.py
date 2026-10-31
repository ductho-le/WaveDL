"""Complete backend receipts survive restarts without claiming missing evidence."""

import sqlite3
from dataclasses import replace

import pytest

from unit_tests.daemon import test_requeue_store as requeue_tests
from wavedl.core.layout import OperationKind
from wavedl_daemon.backends.contracts import (
    OperationOutcome,
    OperationResult,
    OutcomeStatus,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.collections import COLLECTION_MIGRATION, CollectionStore
from wavedl_daemon.storage.executions import StaleExecution
from wavedl_daemon.storage.ownership import DispatchRequired, acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.schema import Migration, MigrationFailed, migrate


execution = requeue_tests.execution


def result(
    *,
    runtime=OutcomeStatus.SUCCESS,
    batch=OutcomeStatus.SUCCESS,
    stage_out=OutcomeStatus.SUCCESS,
):
    return OperationResult(
        OperationKind.TRAIN,
        OperationOutcome(
            runtime, exit_code=125 if runtime == OutcomeStatus.UNKNOWN else 0
        ),
        OperationOutcome(batch, exit_code=1 if batch == OutcomeStatus.FAILED else 0),
        OperationOutcome(
            stage_out, exit_code=1 if stage_out == OutcomeStatus.FAILED else 0
        ),
    )


def test_complete_collection_is_immutable_idempotent_and_durable(execution):
    store, _, plan_id = execution
    source = store.get(plan_id).handle
    outcome = result(stage_out=OutcomeStatus.FAILED)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_collection(plan_id, source, outcome)
        assert record.result == outcome and record.handle == source
        assert collections.record_collection(plan_id, source, outcome) == record
        assert collections.history(plan_id) == [record]
        with pytest.raises(StaleExecution, match="Conflicting"):
            collections.record_collection(plan_id, source, result())
        for query in (
            "DELETE FROM execution_collections",
            "UPDATE execution_collections SET observed_at='changed'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                collections._connection.execute(query)
    with CollectionStore(store.path, read_only=True) as reopened:
        assert reopened.get(plan_id) == record


@pytest.mark.parametrize("missing", ["batch", "stage_out"])
def test_unknown_reads_do_not_certify_or_replace_complete_outcomes(execution, missing):
    store, _, plan_id = execution
    source = store.get(plan_id).handle
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        assert (
            collections.record_collection(
                plan_id, source, result(**{missing: OutcomeStatus.UNKNOWN})
            )
            is None
        )
        assert collections.get(plan_id) is None
        complete = collections.record_collection(plan_id, source, result())
        assert (
            collections.record_collection(
                plan_id, source, result(**{missing: OutcomeStatus.UNKNOWN})
            )
            is None
        )
        assert collections.get(plan_id) == complete


def test_setup_failure_preserves_unknown_runtime_but_known_wrapper_outcomes(execution):
    store, _, plan_id = execution
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_collection(
            plan_id,
            store.get(plan_id).handle,
            result(runtime=OutcomeStatus.UNKNOWN, batch=OutcomeStatus.FAILED),
        )
        assert record.result.runtime.status == OutcomeStatus.UNKNOWN
        assert record.result.runtime.exit_code == 125
        assert record.result.batch.status == OutcomeStatus.FAILED
        assert record.result.stage_out.status == OutcomeStatus.SUCCESS


def test_requeue_retains_old_collection_but_never_accepts_late_old_source(execution):
    store, _, plan_id = execution
    old = store.get(plan_id)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_collection(plan_id, old.handle, result())
        binding, handle = requeue_tests.candidate(old)
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
        assert collections.get(plan_id) is None
        assert collections.get(plan_id, 1) == record
        with pytest.raises(StaleExecution, match="current"):
            collections.record_collection(plan_id, old.handle, result())
        current = collections.record_collection(
            plan_id, handle, result(stage_out=OutcomeStatus.FAILED)
        )
        assert collections.get(plan_id) == current
        assert collections.history(plan_id, limit=1, offset=1) == [current]


def test_foreign_handle_or_operation_is_rejected(execution):
    store, _, plan_id = execution
    source = store.get(plan_id).handle
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        with pytest.raises(StaleExecution):
            collections.record_collection(
                plan_id, replace(source, job_id="456;alpha"), result()
            )
        with pytest.raises(StaleExecution, match="operation"):
            collections.record_collection(
                plan_id, source, replace(result(), operation=OperationKind.TEST)
            )
        assert collections.history(plan_id) == []


def test_record_failure_rolls_back_persisted_receipt(execution, monkeypatch):
    store, _, plan_id = execution
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        original = collections._insert

        def fail(*args):
            original(*args)
            raise RuntimeError("Fault after receipt write")

        monkeypatch.setattr(collections, "_insert", fail)
        with pytest.raises(RuntimeError):
            collections.record_collection(plan_id, store.get(plan_id).handle, result())
        assert collections.get(plan_id) is None


def test_dispatch_readonly_and_json_bounds_are_enforced(execution):
    store, _, plan_id = execution
    source = store.get(plan_id).handle
    with CollectionStore(store.path) as unowned, pytest.raises(DispatchRequired):
        unowned.record_collection(plan_id, source, result())
    with (
        CollectionStore(store.path, read_only=True) as readonly,
        pytest.raises(PermissionError),
    ):
        readonly.record_collection(plan_id, source, result())
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        oversized = replace(
            result(),
            runtime=OperationOutcome(OutcomeStatus.SUCCESS, detail="波" * 100000),
        )
        with pytest.raises(BudgetExhausted):
            collections.record_collection(plan_id, source, oversized)
        assert collections.get(plan_id) is None
        for limit in (0, True, 1001):
            with pytest.raises(ValueError):
                collections.history(plan_id, limit=limit)


def schema7(path):
    with Registry(path) as registry:
        registry._connection.execute("DROP TRIGGER writer_initial_activation")
        registry._connection.execute("DROP TABLE initial_bindings")
        registry._connection.execute("DROP TABLE submission_control")
        registry._connection.execute(
            "DELETE FROM schema_migrations WHERE migration_id='0009_writer_submissions'"
        )
        registry._connection.execute("DROP TABLE execution_collections")
        registry._connection.execute(
            "DELETE FROM schema_migrations WHERE migration_id='0008_execution_collections'"
        )
        registry._connection.execute("PRAGMA user_version=7")


def test_schema7_upgrade_and_failed_migration_rolls_back(tmp_path):
    path = tmp_path / "registry.db"
    schema7(path)

    def fail(connection):
        COLLECTION_MIGRATION.apply(connection)
        raise RuntimeError("Migration interrupted")

    with pytest.raises(MigrationFailed):
        migrate(path, [Migration("0008_execution_collections", 8, fail)])
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='execution_collections'"
            ).fetchone()
            is None
        )
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(path, dispatch_lock=lock),
        CollectionStore(path, dispatch_lock=lock) as collections,
    ):
        assert collections.history("missing") == []
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with CollectionStore(path, read_only=True) as recovered:
        assert recovered.get("missing") is None
    with pytest.raises(PlanRecoveryRequired):
        CollectionStore(path)


def test_history_byte_bound_is_visible_and_per_attempt_pages_recover(execution):
    store, _, plan_id = execution
    large = replace(
        result(), runtime=OperationOutcome(OutcomeStatus.SUCCESS, detail="x" * 200000)
    )
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        for attempt in range(1, 7):
            current = store.get(plan_id)
            collections.record_collection(plan_id, current.handle, large)
            if attempt < 6:
                binding, handle = requeue_tests.candidate(
                    current, attempt=attempt + 1, restart=attempt
                )
                store.activate_requeue(
                    plan_id, binding, handle, expected_revision=current.revision
                )
        with pytest.raises(BudgetExhausted, match="history"):
            collections.history(plan_id)
        assert len(collections.history(plan_id, limit=1)) == 1
        assert collections.history(plan_id, limit=1, offset=5)[0].attempt == 6


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "00000000-0000-4000-8000-000000000003"},
        {"request_id": "foreign-request"},
        {"owner": "mallory"},
        {"job_id": "999;alpha"},
        {"raw_json": '{"restart_count":9}'},
    ],
)
def test_well_typed_foreign_collected_handle_is_rejected_on_every_read(
    execution, change
):
    store, _, plan_id = execution
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        source = store.get(plan_id).handle
        collections.record_collection(plan_id, source, result())
        connection = store._connection
        connection.execute("DROP TRIGGER execution_collections_no_update")
        connection.execute(
            "UPDATE execution_collections SET handle_json=?",
            (replace(source, **change).to_json(),),
        )
        with pytest.raises(PlanRecoveryRequired):
            collections.get(plan_id)
        with pytest.raises(PlanRecoveryRequired):
            collections.history(plan_id)


@pytest.mark.parametrize(
    "timestamp", ["", "2026-99-04T00:00:00Z", "2026-10-04T00:00:00+07:00", "x" * 65]
)
def test_collected_timestamp_corruption_never_reaches_ui(execution, timestamp):
    store, _, plan_id = execution
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_collection(plan_id, store.get(plan_id).handle, result())
        store._connection.execute("DROP TRIGGER execution_collections_no_update")
        store._connection.execute(
            "UPDATE execution_collections SET observed_at=?", (timestamp,)
        )
        with pytest.raises(PlanRecoveryRequired):
            collections.get(plan_id)


def test_collected_operation_corruption_is_rejected_against_immutable_plan(execution):
    store, _, plan_id = execution
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_collection(plan_id, store.get(plan_id).handle, result())
        store._connection.execute("DROP TRIGGER execution_collections_no_update")
        store._connection.execute(
            "UPDATE execution_collections SET result_json=?",
            (
                collections._connection.execute(
                    "SELECT result_json FROM execution_collections"
                )
                .fetchone()[0]
                .replace('"operation":"train"', '"operation":"test"'),
            ),
        )
        with pytest.raises(PlanRecoveryRequired):
            collections.get(plan_id)


def test_missing_archived_execution_proof_fails_read_and_readonly_open(execution):
    store, _, plan_id = execution
    old = store.get(plan_id)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_collection(plan_id, old.handle, result())
        binding, handle = requeue_tests.candidate(old)
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
        store._connection.execute("DROP TRIGGER execution_attempt_history_no_delete")
        store._connection.execute("DELETE FROM execution_attempt_history")
        with pytest.raises(PlanRecoveryRequired):
            collections.get(plan_id, 1)
        with pytest.raises(PlanRecoveryRequired):
            collections.history(plan_id)
        with pytest.raises(PlanRecoveryRequired):
            CollectionStore(store.path, read_only=True)


def test_dangling_collection_without_plan_execution_is_refused_readonly(execution):
    store, _, plan_id = execution
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_collection(
            plan_id, store.get(plan_id).handle, result()
        )
    import sqlite3

    with sqlite3.connect(store.path) as broken:
        broken.execute(
            "INSERT INTO execution_collections VALUES (?,?,?,?,?)",
            (
                "00000000-0000-4000-8000-000000000099",
                1,
                record.handle_json,
                record.result_json,
                record.observed_at,
            ),
        )
    with pytest.raises(PlanRecoveryRequired):
        CollectionStore(store.path, read_only=True)


def test_archived_collected_native_handle_must_match_exact_old_execution(execution):
    store, _, plan_id = execution
    old = store.get(plan_id)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_collection(plan_id, old.handle, result())
        binding, handle = requeue_tests.candidate(old)
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
        store._connection.execute("DROP TRIGGER execution_collections_no_update")
        store._connection.execute(
            "UPDATE execution_collections SET handle_json=?",
            (replace(old.handle, job_id="999;alpha").to_json(),),
        )
        with pytest.raises(PlanRecoveryRequired):
            collections.get(plan_id, 1)
        with pytest.raises(PlanRecoveryRequired):
            collections.history(plan_id)


def test_duplicate_collected_receipt_fields_are_not_last_value_wins(execution):
    store, _, plan_id = execution
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_collection(
            plan_id, store.get(plan_id).handle, result()
        )
        store._connection.execute("DROP TRIGGER execution_collections_no_update")
        raw = record.result_json[:-1] + ',"operation":"train"}'
        store._connection.execute(
            "UPDATE execution_collections SET result_json=?", (raw,)
        )
        with pytest.raises(PlanRecoveryRequired):
            collections.get(plan_id)
