"""Integration boundaries between telemetry, immutable plans, and schema recovery."""

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl_daemon.backends.contracts import Plan
from wavedl_daemon.storage.event_log import EVENT_LOG_MIGRATION
from wavedl_daemon.storage.executions import EXECUTION_MIGRATION, ExecutionStore
from wavedl_daemon.storage.metric_projection import METRIC_PROJECTION_MIGRATION
from wavedl_daemon.storage.ownership import (
    DispatchHeld,
    DispatchRequired,
    acquire_dispatch_lock,
)
from wavedl_daemon.storage.plans import (
    PLAN_MIGRATION,
    PlanRecoveryRequired,
    PlanStore,
)
from wavedl_daemon.storage.registry import _SCHEMA, SCHEMA_VERSION, Registry
from wavedl_daemon.storage.runtime_provenance import RUNTIME_PROVENANCE_MIGRATION
from wavedl_daemon.storage.schema import (
    applied_migrations,
    backup_path,
    migrate,
    schema_version,
)


RUN_ID = "00000000-0000-4000-8000-000000000001"


def setup_plan(tmp_path):
    import json

    layout = RunLayout.create(tmp_path / "runs", RUN_ID)
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 3},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    plan = Plan(
        "local", json.dumps(asdict(manifest)), argv=(sys.executable, "train.py")
    )
    return layout, plan


def test_existing_version_one_registry_upgrades_without_losing_telemetry(tmp_path):
    path = tmp_path / "registry.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "INSERT INTO runs (run_id, event_count) VALUES (?, 17)", (RUN_ID,)
        )
        connection.execute("INSERT INTO cursors VALUES (?, 1, 20)", (RUN_ID,))
    for _ in range(2):
        with Registry(path) as registry:
            assert registry.get(RUN_ID).event_count == 17
            assert registry.cursor(RUN_ID, 1) == 20
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == SCHEMA_VERSION == 13
        assert applied_migrations(connection) == {
            PLAN_MIGRATION.migration_id,
            EXECUTION_MIGRATION.migration_id,
            RUNTIME_PROVENANCE_MIGRATION.migration_id,
            METRIC_PROJECTION_MIGRATION.migration_id,
            EVENT_LOG_MIGRATION.migration_id,
            "0007_scheduler_requeues",
            "0008_execution_collections",
            "0009_writer_submissions",
            "0010_submission_collections",
            "0011_submission_generation_collections",
            "0012_submission_native_generations",
            "0013_submission_native_lifecycle",
        }
        assert connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0
    with sqlite3.connect(backup_path(path, PLAN_MIGRATION.migration_id)) as before:
        assert schema_version(before) == 1
        assert before.execute("SELECT event_count FROM runs").fetchone()[0] == 17


def test_version_two_upgrade_preserves_existing_idempotency_and_intent(tmp_path):
    path = tmp_path / "registry.db"
    _, plan = setup_plan(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute("PRAGMA user_version=1")
    migrate(path, [PLAN_MIGRATION])
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(path, dispatch_lock=lock) as plans,
    ):
        original = plans.put(plan, idempotency_key="old-key", request_payload={})
        original = plans.request_submission(original.plan_id)
        original = plans.request_cancellation(original.plan_id)
    with Registry(path), PlanStore(path, read_only=True) as plans:
        assert plans.get_by_key("old-key") == original
    with sqlite3.connect(backup_path(path, EXECUTION_MIGRATION.migration_id)) as backup:
        assert schema_version(backup) == 2
        assert (
            backup.execute("SELECT request_id FROM plans").fetchone()[0]
            == original.request_id
        )
    with ExecutionStore(path, read_only=True) as executions:
        assert executions.list_pending() == []


def test_plan_intent_pages_are_bounded_stable_and_include_prehandle_cancel(tmp_path):
    path = tmp_path / "registry.db"
    _, plan = setup_plan(tmp_path)
    with Registry(path):
        pass
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(path, dispatch_lock=lock) as plans,
    ):
        ids = []
        for key in ("submit", "cancel", "preview"):
            record = plans.put(plan, idempotency_key=key, request_payload={})
            if key == "submit":
                plans.request_submission(record.plan_id)
            elif key == "cancel":
                plans.request_cancellation(record.plan_id)
            else:
                continue
            ids.append(record.plan_id)
        assert [record.plan_id for record in plans.list_pending()] == ids
        assert plans.list_pending(limit=1, offset=1)[0].plan_id == ids[1]
        assert plans.list_pending(limit=1, offset=2) == []
        for limit, offset in ((True, 0), (0, 0), (1001, 0), (1, -1)):
            with pytest.raises(ValueError):
                plans.list_pending(limit=limit, offset=offset)


def test_telemetry_rebuild_preserves_pre_runtime_plan_and_cancellation(tmp_path):
    path = tmp_path / "registry.db"
    layout, plan = setup_plan(tmp_path)
    with Registry(path):
        pass
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(path, dispatch_lock=lock) as store,
    ):
        original = store.put(
            plan, idempotency_key="once", request_payload={"epochs": 3}
        )
        original = store.request_submission(original.plan_id)
        with ExecutionStore(path, dispatch_lock=lock) as executions:
            execution = executions.ensure(original.plan_id, 1)
            executions.claim_submission(original.plan_id)
            execution = executions.mark_unknown(
                original.plan_id, "lost acknowledgement"
            )
            original = store.request_cancellation(original.plan_id)
            execution = executions.get(original.plan_id)
        # Opening registry with borrowed ownership must not release that lock.
        with Registry(path, dispatch_lock=lock) as registry:
            assert registry.rebuild_from_journals(layout.run_dir.parent) == 0
            assert store.get(original.plan_id) == original
            with ExecutionStore(path, dispatch_lock=lock) as executions:
                assert executions.get(original.plan_id) == execution
            with JournalWriter(layout) as writer:
                writer.append("hello", {})
            assert registry.rebuild_from_journals(layout.run_dir.parent) == 1
            assert store.get(original.plan_id) == original
            assert registry.get(RUN_ID).operation == "train"
            assert registry.cursor(RUN_ID, 1) == 0
        assert lock.held


def test_startup_borrowed_lock_must_belong_to_same_registry(tmp_path):
    other = tmp_path / "other" / "dispatch.lock"
    with acquire_dispatch_lock(other) as lock:
        with pytest.raises(DispatchRequired, match="different registry"):
            Registry(tmp_path / "registry.db", dispatch_lock=lock)
        with pytest.raises(DispatchRequired, match="different registry"):
            migrate(tmp_path / "registry.db", [], dispatch_lock=lock)
        assert lock.held
    assert not (tmp_path / "registry.db").exists()


def test_cancellation_checked_inside_claim_transaction(tmp_path):
    path = tmp_path / "registry.db"
    _, plan = setup_plan(tmp_path)
    with Registry(path):
        pass
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(path, dispatch_lock=lock) as plans,
        ExecutionStore(path, dispatch_lock=lock) as executions,
    ):
        saved = plans.put(plan, idempotency_key="one", request_payload={})
        plans.request_submission(saved.plan_id)
        executions.ensure(saved.plan_id, 1)
        plans.request_cancellation(saved.plan_id)
        record, claimed = executions.claim_submission(saved.plan_id)
        assert not claimed
        assert record.cancellation_requested
        assert record.state == "pending"


def test_concurrent_initializers_cannot_reset_schema_or_repeat_migration(tmp_path):
    path = tmp_path / "registry.db"

    def initialize(_):
        try:
            with Registry(path):
                return "ready"
        except DispatchHeld:
            return "contended"

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(initialize, range(4)))
    assert "ready" in outcomes
    with Registry(path), PlanStore(path, read_only=True):
        pass
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 13
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE migration_id=?",
                (PLAN_MIGRATION.migration_id,),
            ).fetchone()[0]
            == 1
        )


def test_schema_backup_failure_keeps_previous_registry_readable(tmp_path, monkeypatch):
    from wavedl_daemon.storage import schema

    path = tmp_path / "registry.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute("PRAGMA user_version=1")
        connection.execute("INSERT INTO runs(run_id) VALUES (?)", (RUN_ID,))

    def fail(*args):
        raise OSError("backup volume full")

    monkeypatch.setattr(schema, "_backup", fail)
    with pytest.raises(OSError, match="backup volume full"):
        Registry(path)
    with Registry(path, read_only=True) as recovered:
        assert recovered.get(RUN_ID) is not None
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 1
        assert not applied_migrations(connection)


def test_deleted_control_database_cannot_reissue_idempotent_identity(tmp_path):
    path = tmp_path / "registry.db"
    _, plan = setup_plan(tmp_path)
    with Registry(path):
        pass
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(path, dispatch_lock=lock) as store,
    ):
        record = store.put(plan, idempotency_key="once", request_payload={"epochs": 3})
        store.request_submission(record.plan_id)
    path.unlink()
    with pytest.raises(PlanRecoveryRequired, match="missing"):
        Registry(path)
    with pytest.raises(PlanRecoveryRequired, match="missing"):
        PlanStore(path)
    assert not path.exists()


def test_interrupted_initialization_marker_requires_explicit_recovery(tmp_path):
    path = tmp_path / "registry.db"
    marker = tmp_path / ".registry.db.initialized"
    marker.write_text('{"registry_id":"interrupted"}')
    with pytest.raises(PlanRecoveryRequired, match="missing"):
        Registry(path)
    assert not path.exists()
    sqlite3.connect(path).close()
    with pytest.raises(PlanRecoveryRequired, match="interrupted"):
        Registry(path)


def test_marker_publication_failure_cannot_create_an_unmarked_database(
    tmp_path, monkeypatch
):
    from wavedl_daemon.storage import registry as module

    def fail(*args):
        raise OSError("marker storage full")

    monkeypatch.setattr(module, "write_once", fail)
    path = tmp_path / "registry.db"
    with pytest.raises(OSError, match="marker storage full"):
        Registry(path)
    assert not path.exists()
