"""Durable scheduler generation activation never trusts runtime telemetry."""

import json
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.protocol import Envelope
from wavedl.core.scheduler_attempt import SchedulerBinding
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    Plan,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.executions import (
    ExecutionState,
    ExecutionStore,
    StaleExecution,
)
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.requeues import REQUEUE_MIGRATION
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore
from wavedl_daemon.storage.schema import Migration, MigrationFailed, migrate


@pytest.fixture
def execution(tmp_path, request):
    database = tmp_path / "registry.db"
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(database, dispatch_lock=lock),
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        run_id = "00000000-0000-4000-8000-000000000001"
        manifest = build_manifest(
            RunLayout(tmp_path / "run", run_id),
            operation=OperationKind.TRAIN,
            resolved_config={"epochs": 1},
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            submission_cwd=tmp_path,
        )
        plan = plans.put(
            Plan(
                "slurm",
                json.dumps(asdict(manifest)),
                argv=(sys.executable, "train.py"),
                resources=()
                if getattr(request, "param", True) is None
                else (("requeue", getattr(request, "param", True)),),
            ),
            idempotency_key="scheduler",
            request_payload={},
        )
        plans.request_submission(plan.plan_id)
        store.ensure(plan.plan_id, 1, owner="alice")
        store.claim_submission(plan.plan_id)
        handle = Handle(
            "slurm",
            run_id,
            plan.request_id,
            1,
            "alice",
            plan.plan.run_dir,
            job_id="123;alpha",
            raw_json=json.dumps(
                {
                    "cluster": "alpha",
                    "comment": plan.request_id,
                    "job_name": f"wavedl-{plan.request_id}",
                    "restart_count": 0,
                    "operation": "train",
                }
            ),
        )
        store.record_handle(plan.plan_id, handle)
        store.record_observation(
            plan.plan_id, BackendObservation(1, ObservationStatus.RUNNING)
        )
        yield store, plans, plan.plan_id


def candidate(record, *, attempt=2, restart=1):
    binding = SchedulerBinding(
        record.run_id, record.request_id, "alpha", "123", "alice", restart, attempt
    )
    raw = json.loads(record.handle.raw_json) | {
        "restart_count": restart,
        "initial_attempt": 1,
        "native_evidence": "validated",
    }
    return binding, replace(record.handle, attempt=attempt, raw_json=json.dumps(raw))


@pytest.mark.parametrize("execution", [False, None, 1], indirect=True)
def test_unapproved_requeue_cannot_activate_even_with_matching_native_identity(
    execution,
):
    store, _, plan_id = execution
    old = store.get(plan_id)
    binding, handle = candidate(old)
    with pytest.raises(StaleExecution, match="did not approve"):
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
    assert store.get(plan_id) == old
    assert store.history(plan_id) == []
    assert store.requeue_binding(plan_id, 2) is None


def test_activation_archives_once_preserves_intent_and_resets_attempt_observations(
    execution,
):
    store, plans, plan_id = execution
    plans.request_cancellation(plan_id)
    store.mark_cancel_delivered(plan_id)
    old = store.get(plan_id)
    binding, handle = candidate(old)
    current = store.activate_requeue(
        plan_id, binding, handle, expected_revision=old.revision
    )
    assert current.attempt == 2 and current.handle == handle
    assert current.state == ExecutionState.SUBMITTED
    assert current.observation is current.latest_observation is None
    assert current.submission_requested and current.cancellation_requested
    assert not current.cancel_delivered
    assert current.revision == old.revision + 1
    assert store.history(plan_id) == [old]
    assert store.requeue_binding(plan_id, 2) == binding
    assert (
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
        == current
    )
    assert store.history(plan_id) == [old]
    with pytest.raises(StaleExecution):
        store.activate_requeue(
            plan_id, binding, handle, expected_revision=current.revision
        )


@pytest.mark.parametrize("point", ["archive", "update"])
def test_failure_mid_activation_rolls_back_history_binding_and_current(
    execution, monkeypatch, point
):
    store, _, plan_id = execution
    old = store.get(plan_id)
    binding, handle = candidate(old)
    method = "_archive_attempt" if point == "archive" else "_update"
    original = getattr(store, method)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Crash after partial transactional writes")

    monkeypatch.setattr(store, method, fail)
    with pytest.raises(RuntimeError):
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
    assert store.get(plan_id) == old
    assert store.history(plan_id) == []
    assert store.requeue_binding(plan_id, 2) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "00000000-0000-4000-8000-000000000003"},
        {"request_id": "00000000-0000-4000-8000-000000000003"},
        {"owner": "bob"},
        {"cluster": "beta"},
        {"job_id": "456"},
        {"restart_count": 0},
        {"attempt": 1},
    ],
)
def test_foreign_or_regressing_binding_never_activates(execution, changes):
    store, _, plan_id = execution
    old = store.get(plan_id)
    binding, handle = candidate(old)
    with pytest.raises(StaleExecution):
        store.activate_requeue(
            plan_id, replace(binding, **changes), handle, expected_revision=old.revision
        )
    assert store.get(plan_id) == old and store.history(plan_id) == []


def test_stale_revision_conflicting_handle_and_late_old_observation_rejected(execution):
    store, _, plan_id = execution
    old = store.get(plan_id)
    binding, handle = candidate(old)
    with pytest.raises(StaleExecution):
        store.activate_requeue(
            plan_id, binding, handle, expected_revision=old.revision - 1
        )
    with pytest.raises(StaleExecution):
        store.activate_requeue(
            plan_id,
            binding,
            replace(handle, run_dir=str(Path(handle.run_dir).parent / "foreign-run")),
            expected_revision=old.revision,
        )
    current = store.activate_requeue(
        plan_id, binding, handle, expected_revision=old.revision
    )
    with pytest.raises(StaleExecution):
        store.record_observation(
            plan_id, BackendObservation(1, ObservationStatus.TERMINAL)
        )
    assert store.get(plan_id) == current


def test_restart_count_and_attempt_gaps_preserve_original_reservation(execution):
    store, _, plan_id = execution
    old = store.get(plan_id)
    binding, handle = candidate(old, attempt=9, restart=3)
    second = store.activate_requeue(
        plan_id, binding, handle, expected_revision=old.revision
    )
    next_binding, next_handle = candidate(second, attempt=15, restart=5)
    third = store.activate_requeue(
        plan_id, next_binding, next_handle, expected_revision=second.revision
    )
    assert third.attempt == 15
    assert store.history(plan_id, limit=1) == [old]
    assert store.history(plan_id, limit=1, offset=1) == [second]


def test_sql_attempt_changes_and_restart_history_mutations_fail_closed(execution):
    store, _, plan_id = execution
    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute(
            "UPDATE execution_control SET attempt=2 WHERE plan_id=?", (plan_id,)
        )
    old = store.get(plan_id)
    binding, handle = candidate(old)
    store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
    for table in ("execution_attempt_history", "requeue_bindings"):
        for sql in (f"DELETE FROM {table}", f"UPDATE {table} SET attempt=attempt"):
            with pytest.raises(sqlite3.IntegrityError):
                store._connection.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute(
            "UPDATE execution_control SET owner='bob' WHERE plan_id=?", (plan_id,)
        )


def schema6(path):
    with Registry(path) as registry:
        connection = registry._connection
        connection.execute("DROP TRIGGER writer_initial_activation")
        connection.execute("DROP TABLE initial_bindings")
        connection.execute("DROP TABLE submission_control")
        connection.execute(
            "DELETE FROM schema_migrations WHERE migration_id='0009_writer_submissions'"
        )
        connection.execute("DROP TABLE execution_collections")
        connection.execute(
            "DELETE FROM schema_migrations WHERE migration_id='0008_execution_collections'"
        )
        connection.execute("DROP TRIGGER execution_attempt_activation")
        for table in ("requeue_bindings", "execution_attempt_history"):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("DROP TRIGGER execution_identity_immutable")
        connection.execute(
            "CREATE TRIGGER execution_identity_immutable BEFORE UPDATE OF plan_id,attempt,owner ON execution_control BEGIN SELECT RAISE(ABORT,'immutable'); END"
        )
        connection.execute(
            "DELETE FROM schema_migrations WHERE migration_id='0007_scheduler_requeues'"
        )
        connection.execute("PRAGMA user_version=6")


def test_schema6_upgrade_and_future_read_only_recovery(tmp_path):
    path = tmp_path / "registry.db"
    schema6(path)
    with Registry(path) as registry:
        assert registry._connection.execute("PRAGMA user_version").fetchone()[0] == 9
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(MigrationFailed), Registry(path):
        pass
    with Registry(path, read_only=True), ExecutionStore(path, read_only=True) as store:
        assert store.history("missing") == []
        with pytest.raises(PermissionError):
            store.activate_requeue(
                "missing",
                SchedulerBinding(
                    "00000000-0000-4000-8000-000000000001",
                    "00000000-0000-4000-8000-000000000002",
                    "alpha",
                    "123",
                    "alice",
                    1,
                    2,
                ),
                Handle(
                    "slurm",
                    "00000000-0000-4000-8000-000000000001",
                    "request",
                    2,
                    "alice",
                    str(tmp_path / "run"),
                ),
                expected_revision=0,
            )


def test_schema7_migration_failure_rolls_back_trigger_and_all_tables(tmp_path):
    path = tmp_path / "registry.db"
    schema6(path)

    def fail(connection):
        REQUEUE_MIGRATION.apply(connection)
        raise RuntimeError("Failed migration")

    with pytest.raises(MigrationFailed):
        migrate(path, [Migration("0007_scheduler_requeues", 7, fail)])
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='requeue_bindings'"
            ).fetchone()
            is None
        )
        assert (
            "attempt"
            in connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='execution_identity_immutable'"
            ).fetchone()[0]
        )


def test_active_provenance_never_returns_previous_attempt_hello(execution):
    store, _, plan_id = execution
    old = store.get(plan_id)
    with RuntimeProvenanceStore(
        store.path, dispatch_lock=store.dispatch_lock
    ) as provenance:
        hello = Envelope(
            old.run_id,
            1,
            "hello",
            0,
            "2026-10-04T00:00:00Z",
            asdict(
                HelloPayload(1, 1, "actual", "3.11", "cpu", ["train"], sys.executable)
            ),
        )
        original = provenance.record(plan_id, hello)
        binding, handle = candidate(old)
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
        assert provenance.get(plan_id) is None
        assert provenance.get(plan_id, 1) == original
        current = provenance.record(plan_id, replace(hello, attempt=2))
        assert provenance.get(plan_id) == current
        assert provenance.get(plan_id, 1) == original


def test_history_byte_bound_is_visible_and_smaller_pages_recover(execution):
    store, _, plan_id = execution
    for attempt in range(2, 8):
        old = store.get(plan_id)
        with store._transaction():
            store._update(old, detail="x" * 200000)
        old = store.get(plan_id)
        binding, handle = candidate(old, attempt=attempt, restart=attempt - 1)
        store.activate_requeue(plan_id, binding, handle, expected_revision=old.revision)
    with pytest.raises(BudgetExhausted):
        store.history(plan_id)
    assert len(store.history(plan_id, limit=1)) == 1
    assert store.history(plan_id, limit=1, offset=5)[0].attempt == 6
