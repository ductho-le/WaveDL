import getpass
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    Plan,
)
from wavedl_daemon.storage.executions import (
    EXECUTION_MIGRATION,
    ExecutionState,
    ExecutionStore,
    StaleExecution,
)
from wavedl_daemon.storage.ownership import DispatchRequired, acquire_dispatch_lock
from wavedl_daemon.storage.plans import PLAN_MIGRATION, PlanRecoveryRequired, PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.schema import migrate


@pytest.fixture
def execution(tmp_path):
    database = tmp_path / "registry.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=1")
    migrate(database, [PLAN_MIGRATION, EXECUTION_MIGRATION])
    run_id = "00000000-0000-4000-8000-000000000001"
    manifest = build_manifest(
        RunLayout(tmp_path / run_id, run_id),
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 3},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    plan = Plan(
        "local", json.dumps(asdict(manifest)), argv=(sys.executable, "train.py")
    )
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as plans,
    ):
        record = plans.put(plan, idempotency_key="one", request_payload={})
        plans.request_submission(record.plan_id)
    return database, record


def receipt(record):
    return Handle(
        record.backend,
        record.run_id,
        record.request_id,
        record.attempt,
        record.owner,
        record.run_dir,
        pid=123,
        supervisor_id="supervisor-one",
    )


def test_claim_is_durable_and_never_reissued(execution):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        record = store.ensure(plan.plan_id, 7)
        assert record.request_id == plan.request_id
        assert record.attempt == 7
        assert store.claim_submission(plan.plan_id)[1]
        assert not store.claim_submission(plan.plan_id)[1]
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        record = store.get(plan.plan_id)
        assert record.state == ExecutionState.UNKNOWN
        assert not store.claim_submission(plan.plan_id)[1]
        assert (
            store.record_handle(plan.plan_id, receipt(record)).state
            == ExecutionState.SUBMITTED
        )


def test_two_connections_claim_only_once(execution):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as first,
        ExecutionStore(database, dispatch_lock=lock) as second,
    ):
        first.ensure(plan.plan_id, 1)
        assert first.claim_submission(plan.plan_id)[1]
        assert not second.claim_submission(plan.plan_id)[1]
        assert not first.claim_submission(plan.plan_id)[1]


def test_sigkill_after_claim_keeps_unknown_acceptance(execution):
    database, plan = execution
    code = """
import os, signal, sys
from pathlib import Path
from wavedl_daemon.storage.executions import ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
database = Path(sys.argv[1])
with (
    acquire_dispatch_lock(database.parent / 'dispatch.lock') as lock,
    ExecutionStore(database, dispatch_lock=lock) as store,
):
    store.ensure(sys.argv[2], 4)
    assert store.claim_submission(sys.argv[2])[1]
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        assert kernel.TerminateProcess(kernel.GetCurrentProcess(), 73)
    else:
        os.kill(os.getpid(), signal.SIGKILL)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(database), plan.plan_id], check=False
    )
    assert result.returncode == 73 if os.name == "nt" else result.returncode < 0
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        assert store.get(plan.plan_id).state == ExecutionState.UNKNOWN
        assert not store.claim_submission(plan.plan_id)[1]


def test_stale_and_foreign_receipts_are_rejected(execution):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        original = store.ensure(plan.plan_id, 1)
        with pytest.raises(KeyError):
            store.ensure("foreign-plan", 1)
        with pytest.raises(StaleExecution):
            store.ensure(plan.plan_id, 2)
        with pytest.raises(StaleExecution):
            store.record_handle(plan.plan_id, receipt(original))
        record, _ = store.claim_submission(plan.plan_id)
        handle = receipt(record)
        for invalid in (
            replace(handle, request_id="foreign"),
            replace(handle, attempt=2),
            replace(handle, owner="different-user"),
            replace(handle, backend="slurm"),
        ):
            with pytest.raises(StaleExecution):
                store.record_handle(plan.plan_id, invalid)
        with pytest.raises(StaleExecution):
            store.record_handle(
                plan.plan_id, handle, expected_revision=original.revision
            )
        attached = store.record_handle(plan.plan_id, handle)
        assert store.record_handle(plan.plan_id, handle) == attached
        with pytest.raises(StaleExecution):
            store.record_handle(
                plan.plan_id, replace(handle, supervisor_id="replacement")
            )


def test_cancel_intent_is_joined_and_survives_receipt(execution):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
        PlanStore(database, dispatch_lock=lock) as plans,
    ):
        store.ensure(plan.plan_id, 1)
        record, claimed = store.claim_submission(plan.plan_id)
        assert claimed
        plans.request_cancellation(plan.plan_id)
        assert store.get(plan.plan_id).cancellation_requested is True
        with pytest.raises(StaleExecution):
            store.mark_cancel_delivered(plan.plan_id)
        store.record_handle(plan.plan_id, receipt(record))
        assert store.mark_cancel_delivered(plan.plan_id).cancel_delivered is True
        assert plans.get(plan.plan_id).cancellation_requested is True


def test_latest_error_preserves_last_known_evidence(execution):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        store.ensure(plan.plan_id, 1)
        record, _ = store.claim_submission(plan.plan_id)
        store.record_handle(plan.plan_id, receipt(record))
        running = BackendObservation(1, ObservationStatus.RUNNING)
        store.record_observation(plan.plan_id, running)
        error = BackendObservation(
            1, ObservationStatus.ERROR, detail="temporarily unavailable"
        )
        record = store.record_observation(plan.plan_id, error)
        assert record.observation == running
        assert record.latest_observation == error
        with pytest.raises(StaleExecution):
            store.record_observation(plan.plan_id, replace(running, attempt=2))
        terminal = BackendObservation(1, ObservationStatus.TERMINAL, exit_code=0)
        store.record_observation(plan.plan_id, terminal)
        refreshed = replace(terminal, native_state="COMPLETED", detail="collected")
        assert (
            store.record_observation(plan.plan_id, refreshed).observation == refreshed
        )
        failed_refresh = store.record_observation(plan.plan_id, error)
        assert failed_refresh.observation == refreshed
        assert failed_refresh.latest_observation == error
        with pytest.raises(StaleExecution):
            store.record_observation(plan.plan_id, running)
        with pytest.raises(StaleExecution):
            store.mark_unknown(plan.plan_id, "late submit error")


def test_recovery_and_database_identity_fail_closed(execution, tmp_path):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        store.ensure(plan.plan_id, 1)
        with ExecutionStore(database, read_only=True) as recovery:
            assert recovery.get(plan.plan_id).owner == getpass.getuser()
            with pytest.raises(PermissionError):
                recovery.claim_submission(plan.plan_id)
        with (
            ExecutionStore(database) as unowned,
            pytest.raises(DispatchRequired),
        ):
            unowned.claim_submission(plan.plan_id)
        saved = database.with_suffix(".saved")
        if os.name == "nt":
            original_identity = database.stat()
            original_bytes = database.read_bytes()
            before = store.get(plan.plan_id)
            with pytest.raises(PermissionError) as denied:
                database.rename(saved)
            assert denied.value.winerror == 32
            assert not saved.exists()
            assert database.stat().st_ino == original_identity.st_ino
            assert database.read_bytes() == original_bytes
            assert store.get(plan.plan_id) == before
        else:
            database.rename(saved)
            with pytest.raises(PlanRecoveryRequired):
                store.get(plan.plan_id)
            with sqlite3.connect(database):
                pass
            with pytest.raises(PlanRecoveryRequired):
                store.claim_submission(plan.plan_id)
    with pytest.raises(PlanRecoveryRequired):
        ExecutionStore(tmp_path / "missing.db")


def test_migration_is_once_and_pages_are_bounded(execution):
    database, plan = execution
    assert migrate(database, [PLAN_MIGRATION, EXECUTION_MIGRATION]) == []
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        record = store.ensure(plan.plan_id, 1)
        assert store.list_pending(1) == [record]
        assert store.list_pending(1, offset=1) == []
        for bound in (0, 1001, True):
            with pytest.raises(ValueError):
                store.list_pending(bound)
        oversized = BackendObservation(1, ObservationStatus.ERROR, detail="x" * 300000)
        with pytest.raises(ValueError):
            store.record_observation(plan.plan_id, oversized)


def test_future_schema_is_read_only(execution):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        store.ensure(plan.plan_id, 1)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "ALTER TABLE execution_control ADD COLUMN future_metadata TEXT"
        )
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(PlanRecoveryRequired):
        ExecutionStore(database)
    with ExecutionStore(database, read_only=True) as store:
        assert store.get(plan.plan_id).attempt == 1
        with pytest.raises(PermissionError):
            store.ensure(plan.plan_id, 1)


def test_missing_authoritative_intent_is_not_hidden(execution):
    database, plan = execution
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        ExecutionStore(database, dispatch_lock=lock) as store,
    ):
        store.ensure(plan.plan_id, 1)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM plan_control WHERE plan_id=?", (plan.plan_id,))
    with pytest.raises(PlanRecoveryRequired):
        ExecutionStore(database, read_only=True)


def test_telemetry_rebuild_preserves_execution_control(tmp_path):
    database = tmp_path / "registry.db"
    (tmp_path / "runs").mkdir()
    run_id = "00000000-0000-4000-8000-000000000001"
    manifest = build_manifest(
        RunLayout(tmp_path / "runs" / run_id, run_id),
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    plan = Plan(
        "local", json.dumps(asdict(manifest)), argv=(sys.executable, "train.py")
    )
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(database, dispatch_lock=lock) as registry,
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
    ):
        persisted = plans.put(plan, idempotency_key="one", request_payload={})
        plans.request_submission(persisted.plan_id)
        plans.request_cancellation(persisted.plan_id)
        executions.ensure(persisted.plan_id, 1)
        executions.claim_submission(persisted.plan_id)
        before = executions.get(persisted.plan_id)
        assert registry.rebuild_from_journals(tmp_path / "runs") == 0
        assert executions.get(persisted.plan_id) == before
