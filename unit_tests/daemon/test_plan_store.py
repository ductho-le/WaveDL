import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl_daemon.backends.contracts import Plan
from wavedl_daemon.storage.ownership import DispatchRequired, acquire_dispatch_lock
from wavedl_daemon.storage.plans import (
    PLAN_MIGRATION,
    IdempotencyConflict,
    PlanRecoveryRequired,
    PlanStore,
)
from wavedl_daemon.storage.schema import backup_path, migrate, schema_version


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "registry.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=1")
    migrate(path, [PLAN_MIGRATION])
    return path


@pytest.fixture
def plan(tmp_path):
    run_id = "00000000-0000-4000-8000-000000000001"
    layout = RunLayout(tmp_path / run_id, run_id)
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 3},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    return Plan(
        "local", json.dumps(asdict(manifest)), argv=(sys.executable, "train.py")
    )


def put(store, plan, **kwargs):
    return store.put(
        plan,
        idempotency_key="request-key",
        request_payload={"operation": "train", "config": {"epochs": 3}},
        **kwargs,
    )


def test_lost_response_replays_same_record_after_reopen(database, plan):
    with acquire_dispatch_lock(database.parent / "dispatch.lock") as lock:
        with PlanStore(database, dispatch_lock=lock) as store:
            original = put(store, plan)
        with PlanStore(database, dispatch_lock=lock) as store:
            # A replanner may have selected different defaults after restart.
            assert put(store, replace(plan, warnings=("changed defaults",))) == original
            assert store.get(original.plan_id) == original
            assert original.request_id != original.plan_id
            assert not original.submission_requested


def test_request_key_is_bound_to_canonical_payload(database, plan):
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        original = put(store, plan)
        assert (
            store.put(
                plan,
                idempotency_key="request-key",
                request_payload={"config": {"epochs": 3}, "operation": "train"},
            )
            == original
        )
        with pytest.raises(IdempotencyConflict):
            store.put(
                plan,
                idempotency_key="request-key",
                request_payload={"operation": "train", "config": {"epochs": 4}},
            )
        assert store.get_by_key("request-key") == original


def test_snapshot_is_exact_and_cannot_be_changed(database, plan):
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        original = put(store, plan)
        assert original.plan_json == plan.to_json()
        recovered = original.plan
        recovered.manifest.resolved_config["epochs"] = 100
        assert store.get(original.plan_id).plan == plan
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute(
                "UPDATE plans SET plan_json='{}' WHERE plan_id=?", (original.plan_id,)
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute("DELETE FROM plans")
        with pytest.raises(sqlite3.IntegrityError):
            store._connection.execute(
                "INSERT INTO plans SELECT * FROM plans WHERE plan_id=?",
                (original.plan_id,),
            )


def test_failed_control_boundary_rolls_back_plan(database, plan, monkeypatch):
    def fail(plan_id):
        raise RuntimeError("control boundary failed")

    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        monkeypatch.setattr(store, "_write_control", fail)
        with pytest.raises(RuntimeError, match="boundary"):
            put(store, plan)
        assert store.get_by_key("request-key") is None
        assert (
            store._connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0
        )


def test_crash_before_control_commit_rolls_back(database, plan, paused_process):
    script = f"""
import sys, time
sys.path.insert(0, {str(Path("src").resolve())!r})
from wavedl_daemon.backends.contracts import Plan
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
def pause(self, plan_id):
    print('plan inserted', flush=True)
    time.sleep(300)
PlanStore._write_control = pause
with acquire_dispatch_lock({str(database.parent / "dispatch.lock")!r}) as lock:
    with PlanStore({str(database)!r}, dispatch_lock=lock) as store:
        store.put(Plan.from_json({plan.to_json()!r}), idempotency_key='request-key',
                  request_payload={{'operation': 'train'}})
"""
    with paused_process(script, "plan inserted") as victim:
        victim.kill()
        victim.wait(timeout=10)
    with PlanStore(database, read_only=True) as store:
        assert store.get_by_key("request-key") is None


def test_separate_connections_collapse_concurrent_identical_requests(database, plan):
    with acquire_dispatch_lock(database.parent / "dispatch.lock") as lock:

        def write(_):
            with PlanStore(database, dispatch_lock=lock) as store:
                return put(store, plan)

        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(write, range(8)))
        assert len(set(records)) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM plan_control").fetchone()[0] == 1
        )


def test_separate_processes_collapse_requests_under_dispatch_ownership(database, plan):
    script = """
import sys, time
from wavedl_daemon.backends.contracts import Plan
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.ownership import DispatchLock, DispatchHeld
from pathlib import Path
path = Path(sys.argv[1])
lock = DispatchLock(path.parent / 'dispatch.lock')
deadline = time.monotonic() + 5
while True:
    try:
        lock.acquire()
        break
    except DispatchHeld:
        if time.monotonic() > deadline:
            raise
        time.sleep(0.01)
try:
    with PlanStore(path, dispatch_lock=lock) as store:
        record = store.put(Plan.from_json(sys.argv[2]), idempotency_key='shared',
                           request_payload={'operation': 'train'})
        print(record.plan_id, record.request_id)
finally:
    lock.release()
"""
    processes = []
    try:
        for _ in range(4):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", script, str(database), plan.to_json()],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
                )
            )
        outputs = []
        for process in processes:
            output, errors = process.communicate(timeout=10)
            assert process.returncode == 0, errors
            outputs.append(output.strip())
        assert len(set(outputs)) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


def test_intents_persist_before_handle_and_survive_reopen(database, plan):
    with acquire_dispatch_lock(database.parent / "dispatch.lock") as lock:
        with PlanStore(database, dispatch_lock=lock) as store:
            original = put(store, plan)
            cancelled = store.request_cancellation(original.plan_id)
            assert cancelled.cancellation_requested
            assert not cancelled.submission_requested
            submitted = store.request_submission(original.plan_id)
            assert submitted.submission_requested and submitted.cancellation_requested
            assert store.request_submission(original.plan_id) == submitted
        with PlanStore(database, read_only=True) as store:
            assert store.get(original.plan_id) == submitted
            assert submitted.request_id == original.request_id


def test_mutations_require_matching_live_dispatch_token(database, plan, tmp_path):
    with PlanStore(database) as store:
        with pytest.raises(DispatchRequired):
            put(store, plan)
        with (
            acquire_dispatch_lock(tmp_path / "other" / "dispatch.lock") as other,
            pytest.raises(DispatchRequired, match="different registry"),
        ):
            put(store, plan, dispatch_lock=other)
        with acquire_dispatch_lock(database.parent / "dispatch.lock") as lock:
            record = put(store, plan, dispatch_lock=lock)
        with pytest.raises(DispatchRequired):
            store.request_cancellation(record.plan_id, dispatch_lock=lock)


def test_read_only_recovery_can_inspect_but_not_write(database, plan):
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        original = put(store, plan)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=100")
    with pytest.raises(PlanRecoveryRequired, match="unsupported"):
        PlanStore(database)
    with PlanStore(database, read_only=True) as store:
        assert store.get(original.plan_id) == original
        with pytest.raises(PermissionError):
            store.request_cancellation(original.plan_id)
        with pytest.raises(PermissionError):
            put(store, plan)


def test_missing_database_is_not_recreated(tmp_path):
    path = tmp_path / "missing.db"
    for read_only in (False, True):
        with pytest.raises(PlanRecoveryRequired, match="missing"):
            PlanStore(path, read_only=read_only)
        assert not path.exists()


def test_deleted_open_database_blocks_mutations(database, plan):
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        if os.name == "nt":
            original_identity = database.stat()
            original_bytes = database.read_bytes()
            before = store.get_by_key("request-key")
            with pytest.raises(PermissionError) as denied:
                database.unlink()
            assert denied.value.winerror == 32
            assert database.stat().st_ino == original_identity.st_ino
            assert database.read_bytes() == original_bytes
            assert store.get_by_key("request-key") == before
            assert put(store, plan).plan == plan
            return
        database.unlink()
        with pytest.raises(PlanRecoveryRequired, match="removed"):
            put(store, plan)
        assert not database.exists()


def test_constructor_does_not_migrate_unknown_schema(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=1")
    with pytest.raises(PlanRecoveryRequired):
        PlanStore(path)
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 1
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_plan_migration_is_backed_up_and_applied_once(database):
    assert backup_path(database, PLAN_MIGRATION.migration_id).is_file()
    assert migrate(database, [PLAN_MIGRATION]) == []
    assert migrate(database, [PLAN_MIGRATION]) == []
    with sqlite3.connect(database) as connection:
        assert schema_version(connection) == 2
        assert connection.execute(
            "SELECT migration_id FROM schema_migrations"
        ).fetchall() == [(PLAN_MIGRATION.migration_id,)]


def test_unknown_plan_control_request_is_not_inserted(database):
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
        pytest.raises(KeyError),
    ):
        store.request_cancellation("missing")


@pytest.mark.parametrize(
    "payload",
    [
        {"n": float("nan")},
        {"n": float("inf")},
        {1: "ambiguous"},
        {True: "ambiguous"},
        {"nested": {1: "ambiguous"}},
        {"nested": {True: "ambiguous"}},
        {"tuple": (1, 2)},
        MappingProxyType({"custom": "mapping"}),
    ],
)
def test_invalid_payload_cannot_reserve_key(database, plan, payload):
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        with pytest.raises(ValueError):
            store.put(plan, idempotency_key="bad", request_payload=payload)
        assert store.get_by_key("bad") is None


@pytest.mark.parametrize("kind", ["cycle", "deep"])
def test_cyclic_and_deep_payloads_cannot_reserve_key(database, plan, kind):
    payload = {}
    if kind == "cycle":
        payload["self"] = payload
    else:
        child = payload
        for _ in range(70):
            child["nested"] = {}
            child = child["nested"]
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        with pytest.raises(ValueError, match=r"cycle|nesting"):
            store.put(plan, idempotency_key="bad", request_payload=payload)
        assert store.get_by_key("bad") is None


def test_control_loss_is_reported_not_silently_recreated(database, plan):
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        put(store, plan)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM plan_control")
    with pytest.raises(PlanRecoveryRequired, match="control record"):
        PlanStore(database)


def test_stamped_but_missing_tables_requires_recovery(tmp_path):
    path = tmp_path / "damaged.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=2")
    with pytest.raises(PlanRecoveryRequired, match="damaged"):
        PlanStore(path)
