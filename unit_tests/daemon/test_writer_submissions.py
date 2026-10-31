"""Writer mode has native submission identity before any positive runtime attempt."""

import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict, replace

import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.local_attempt import LocalStartupBinding
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.protocol import Envelope
from wavedl.core.scheduler_attempt import SchedulerBinding
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    OperationOutcome,
    OperationResult,
    OutcomeStatus,
    Plan,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import (
    ExecutionState,
    ExecutionStore,
    StaleExecution,
)
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanRecoveryRequired, PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore
from wavedl_daemon.storage.schema import (
    Migration,
    MigrationFailed,
    backup_path,
    migrate,
)
from wavedl_daemon.storage.submissions import SUBMISSION_MIGRATION


@pytest.fixture
def writer(tmp_path):
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(tmp_path / "registry.db", dispatch_lock=lock),
        PlanStore(tmp_path / "registry.db", dispatch_lock=lock) as plans,
        ExecutionStore(tmp_path / "registry.db", dispatch_lock=lock) as executions,
    ):
        layout = RunLayout(tmp_path / "run", "00000000-0000-4000-8000-000000000001")
        manifest = build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={"epochs": 1},
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            submission_cwd=tmp_path,
        )
        layout.run_dir.mkdir(mode=0o700)
        write_manifest(layout, manifest)
        saved = plans.put(
            Plan(
                "slurm",
                json.dumps(asdict(manifest)),
                argv=(sys.executable, "train.py"),
                allocation_mode="writer",
            ),
            idempotency_key="writer",
            request_payload={},
        )
        plans.request_submission(saved.plan_id)
        yield executions, plans, saved


def native(record, plan):
    return SubmissionHandle(
        record.backend,
        record.run_id,
        record.request_id,
        record.owner,
        record.run_dir,
        job_id="123;alpha",
        raw_json=json.dumps(
            {
                "cluster": "alpha",
                "comment": record.request_id,
                "job_name": f"wavedl-{record.request_id}",
                "restart_count": 0,
                "operation": "train",
                **(
                    {
                        "allocation_mode": "writer",
                        "plan_sha256": hashlib.sha256(
                            plan.to_json().encode()
                        ).hexdigest(),
                    }
                    if plan.allocation_mode == "writer"
                    else {}
                ),
            }
        ),
    )


def prepare(writer):
    store, _, plan = writer
    unbound = store.ensure(plan.plan_id, owner="alice")
    claimed, did_claim = store.claim_submission(plan.plan_id)
    assert did_claim
    return store.attach_submission(
        plan.plan_id, native(unbound, plan.plan), expected_revision=claimed.revision
    )


def binding(record, attempt=7):
    return SchedulerBinding(
        record.run_id, record.request_id, "alpha", "123", "alice", 0, attempt
    )


def test_writer_submission_is_unbound_claim_before_sideeffect_and_immutable_native_receipt(
    writer,
):
    store, _, plan = writer
    initial = store.ensure(plan.plan_id, owner="alice")
    assert initial.attempt is None and initial.handle is None
    assert store.list_pending() == [initial]
    assert (
        store._connection.execute("SELECT COUNT(*) FROM execution_control").fetchone()[
            0
        ]
        == 0
    )
    with pytest.raises(StaleExecution):
        store.attach_submission(
            plan.plan_id, native(initial, plan.plan), expected_revision=initial.revision
        )
    claimed, yes = store.claim_submission(plan.plan_id)
    assert yes and claimed.attempt is None
    assert not store.claim_submission(plan.plan_id)[1]
    accepted = store.attach_submission(
        plan.plan_id, native(initial, plan.plan), expected_revision=claimed.revision
    )
    assert accepted.submission_handle.job_id == "123;alpha"
    assert accepted.attempt is None and accepted.handle is None
    assert (
        store.attach_submission(
            plan.plan_id, native(initial, plan.plan), expected_revision=claimed.revision
        )
        == accepted
    )
    with pytest.raises(StaleExecution):
        store.attach_submission(
            plan.plan_id,
            replace(
                native(initial, plan.plan),
                raw_json=native(initial, plan.plan).raw_json.replace("train", "test"),
            ),
            expected_revision=accepted.revision,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"allocation_mode": None},
        {"allocation_mode": "reserved"},
        {"plan_sha256": None},
        {"plan_sha256": "0" * 64},
        {"operation": None},
        {"operation": "test"},
        {"initial_attempt": None},
    ],
)
def test_writer_native_receipt_requires_exact_allocation_metadata(writer, mutation):
    store, _, saved = writer
    initial = store.ensure(saved.plan_id, owner="alice")
    claimed, _ = store.claim_submission(saved.plan_id)
    receipt = native(initial, saved.plan)
    raw = json.loads(receipt.raw_json)
    for name, value in mutation.items():
        if value is None and name != "initial_attempt":
            raw.pop(name)
        else:
            raw[name] = value
    corrupt = replace(receipt, raw_json=json.dumps(raw))
    with pytest.raises(StaleExecution, match="writer allocation plan"):
        store.attach_submission(
            saved.plan_id, corrupt, expected_revision=claimed.revision
        )
    assert store.get(saved.plan_id) == claimed
    accepted = store.attach_submission(
        saved.plan_id, receipt, expected_revision=claimed.revision
    )
    store._connection.execute("DROP TRIGGER submission_receipt_immutable")
    store._connection.execute(
        "UPDATE submission_control SET submission_handle_json=? WHERE plan_id=?",
        (corrupt.to_json(), saved.plan_id),
    )
    with pytest.raises(PlanRecoveryRequired, match="native receipt is damaged"):
        store.get(saved.plan_id)
    assert accepted.attempt is None


def test_initial_activation_allocates_no_attempt_and_atomically_uses_verified_writer_number(
    writer,
):
    store, plans, saved = writer
    old = prepare(writer)
    plans.request_cancellation(saved.plan_id)
    store.mark_cancel_delivered(saved.plan_id)
    old = store.get(saved.plan_id)
    proof = binding(old)
    activated = store.activate_initial(
        saved.plan_id, proof, expected_revision=old.revision
    )
    assert (
        activated.attempt == 7
        and activated.handle == old.submission_handle.bind_attempt(7)
    )
    assert activated.submission_handle == old.submission_handle
    assert (
        activated.cancellation_requested
        and activated.submission_requested
        and activated.cancel_delivered
    )
    assert activated.revision == old.revision + 1
    assert store.list_pending() == [activated]
    assert (
        store.activate_initial(saved.plan_id, proof, expected_revision=old.revision)
        == activated
    )
    assert (
        store._connection.execute("SELECT COUNT(*) FROM initial_bindings").fetchone()[0]
        == 1
    )
    with pytest.raises(StaleExecution):
        store.activate_initial(
            saved.plan_id, replace(proof, attempt=8), expected_revision=old.revision
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"owner": "mallory"},
        {"job_id": "456"},
        {"cluster": "beta"},
        {"restart_count": 1},
        {"request_id": "00000000-0000-4000-8000-000000000002"},
    ],
)
def test_foreign_initial_binding_does_not_create_runtime_execution(writer, changes):
    store, _, saved = writer
    old = prepare(writer)
    with pytest.raises(StaleExecution):
        store.activate_initial(
            saved.plan_id,
            replace(binding(old), **changes),
            expected_revision=old.revision,
        )
    assert store.get(saved.plan_id) == old
    assert (
        store._connection.execute("SELECT COUNT(*) FROM execution_control").fetchone()[
            0
        ]
        == 0
    )


def test_sql_insert_attempt_without_verified_binding_is_refused_and_receipt_is_immutable(
    writer,
):
    store, _, saved = writer
    prepare(writer)
    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute(
            "INSERT INTO execution_control(plan_id,attempt,owner,state,updated_at) VALUES (?,1,'alice','submitted','now')",
            (saved.plan_id,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute(
            "UPDATE submission_control SET submission_handle_json='{}'"
        )


def test_initial_activation_insert_failure_rolls_back_binding_and_native_source(
    writer, monkeypatch
):
    store, _, saved = writer
    old = prepare(writer)
    store._connection.execute(
        "CREATE TRIGGER fault_initial BEFORE INSERT ON execution_control BEGIN SELECT RAISE(ABORT,'injected fault'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.activate_initial(
            saved.plan_id, binding(old), expected_revision=old.revision
        )
    assert store.get(saved.plan_id) == old
    assert (
        store._connection.execute("SELECT COUNT(*) FROM initial_bindings").fetchone()[0]
        == 0
    )


def test_unbound_observations_unknown_transitions_and_cancel_ack_need_native_receipt(
    writer,
):
    store, plans, saved = writer
    store.ensure(saved.plan_id, owner="alice")
    plans.request_cancellation(saved.plan_id)
    with pytest.raises(StaleExecution):
        store.mark_cancel_delivered(saved.plan_id)
    assert not store.claim_submission(saved.plan_id)[1]
    assert store.get(saved.plan_id).attempt is None


def test_submission_observation_does_not_manufacture_runtime_outcome(writer):
    store, _, saved = writer
    old = prepare(writer)
    current = store.observe_submission(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.RUNNING),
        expected_revision=old.revision,
    )
    assert current.attempt is None and current.observation is None
    assert current.submission_observation.status == ObservationStatus.RUNNING
    latest = store.observe_submission(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.ERROR, detail="Unavailable"),
    )
    assert latest.submission_observation == current.submission_observation
    assert latest.latest_submission_observation.status == ObservationStatus.ERROR


def test_allocation_modes_reject_positive_writer_and_unbound_reserved(writer):
    store, plans, saved = writer
    with pytest.raises(StaleExecution):
        store.ensure(saved.plan_id, 1, owner="alice")
    reserved = plans.put(
        replace(saved.plan, allocation_mode="reserved"),
        idempotency_key="legacy",
        request_payload={},
    )
    with pytest.raises(StaleExecution):
        store.ensure(reserved.plan_id, owner="alice")
    assert store.ensure(reserved.plan_id, 1, owner="alice").attempt == 1


def test_schema9_migration_fault_rolls_back_without_rebuilding_legacy_control(tmp_path):
    path = tmp_path / "registry.db"
    with Registry(path) as registry:
        connection = registry._connection
        connection.execute("DROP TRIGGER writer_initial_activation")
        connection.execute("DROP TABLE initial_bindings")
        connection.execute("DROP TABLE submission_control")
        connection.execute(
            "DELETE FROM schema_migrations WHERE migration_id='0009_writer_submissions'"
        )
        connection.execute("PRAGMA user_version=8")

    def fail(connection):
        SUBMISSION_MIGRATION.apply(connection)
        raise RuntimeError("Migration interrupted")

    with pytest.raises(MigrationFailed):
        migrate(path, [Migration("0009_writer_submissions", 9, fail)])
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 8
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='submission_control'"
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='execution_identity_immutable'"
        ).fetchone()


def test_local_initial_binding_matches_native_supervisor_and_approved_publisher_digests(
    writer,
):
    store, plans, original = writer
    saved = plans.put(
        replace(original.plan, target="local"),
        idempotency_key="local-writer",
        request_payload={},
    )
    plans.request_submission(saved.plan_id)
    initial = store.ensure(saved.plan_id, owner="alice")
    claimed, _ = store.claim_submission(saved.plan_id)
    receipt = SubmissionHandle(
        "local",
        saved.plan.run_id,
        saved.request_id,
        "alice",
        saved.plan.run_dir,
        pid=345,
        supervisor_id="owned-supervisor",
    )
    accepted = store.attach_submission(
        saved.plan_id, receipt, expected_revision=claimed.revision
    )
    proof = LocalStartupBinding(
        initial.run_id,
        initial.request_id,
        initial.owner,
        receipt.supervisor_id,
        receipt.run_dir,
        hashlib.sha256(
            (store.path.parent / "run" / "manifest.json").read_bytes()
        ).hexdigest(),
        hashlib.sha256(saved.plan.to_json().encode("utf-8")).hexdigest(),
        12,
    )
    with pytest.raises(StaleExecution, match="digests"):
        store.activate_initial(
            saved.plan_id,
            replace(proof, plan_sha256="f" * 64),
            expected_revision=accepted.revision,
        )
    with pytest.raises(StaleExecution, match="supervisor"):
        store.activate_initial(
            saved.plan_id,
            replace(proof, supervisor_id="other-supervisor"),
            expected_revision=accepted.revision,
        )
    current = store.activate_initial(
        saved.plan_id, proof, expected_revision=accepted.revision
    )
    assert current.handle.pid == 345 and current.handle.attempt == 12
    assert current.submission_handle == receipt

    # Recovery must recheck the immutable plan digests, not merely trust that a
    # binding has the right native supervisor and positive attempt.
    store._connection.execute("DROP TRIGGER initial_bindings_no_update")
    store._connection.execute(
        "UPDATE initial_bindings SET binding_json=? WHERE plan_id=?",
        (json.dumps(asdict(replace(proof, plan_sha256="f" * 64))), saved.plan_id),
    )
    with pytest.raises(PlanRecoveryRequired, match="damaged"):
        store.get(saved.plan_id)


def test_unbound_claim_reopen_is_unknown_and_never_reissued(writer):
    store, _, saved = writer
    store.ensure(saved.plan_id, owner="alice")
    claimed, _ = store.claim_submission(saved.plan_id)
    store.mark_unknown(
        saved.plan_id,
        "Submit acknowledgement unavailable",
        expected_revision=claimed.revision,
    )
    with ExecutionStore(store.path, dispatch_lock=store.dispatch_lock) as reopened:
        uncertain = reopened.get(saved.plan_id)
        assert uncertain.state == ExecutionState.UNKNOWN and uncertain.attempt is None
        assert not reopened.claim_submission(saved.plan_id)[1]
        recovered = reopened.attach_submission(
            saved.plan_id,
            native(uncertain, saved.plan),
            expected_revision=uncertain.revision,
        )
        assert recovered.attempt is None and recovered.submission_handle is not None


def test_unbound_initial_binding_rejects_raw_dict_and_stale_revision(writer):
    store, _, saved = writer
    old = prepare(writer)
    with pytest.raises(ValueError, match="Typed"):
        store.activate_initial(
            saved.plan_id, asdict(binding(old)), expected_revision=old.revision
        )
    with pytest.raises(StaleExecution):
        store.activate_initial(
            saved.plan_id, binding(old), expected_revision=old.revision - 1
        )
    assert store.get(saved.plan_id).attempt is None


def test_initial_binding_preserves_native_observations_and_source_on_active_record(
    writer,
):
    store, _, saved = writer
    before = prepare(writer)
    running = SubmissionObservation(ObservationStatus.RUNNING)
    before = store.observe_submission(
        saved.plan_id, running, expected_revision=before.revision
    )
    diagnostic = SubmissionObservation(
        ObservationStatus.ERROR, detail="Current native observation unavailable"
    )
    before = store.observe_submission(
        saved.plan_id, diagnostic, expected_revision=before.revision
    )
    active = store.activate_initial(
        saved.plan_id, binding(before), expected_revision=before.revision
    )
    assert active.submission_handle == before.submission_handle
    assert active.submission_observation == running
    assert active.latest_submission_observation == diagnostic
    assert active.observation == running.bind_attempt(active.attempt)
    assert active.latest_observation == diagnostic.bind_attempt(active.attempt)
    assert store.list_pending() == [active]


def test_corrupt_frozen_initial_binding_is_never_promoted_as_active_execution(writer):
    store, _, saved = writer
    old = prepare(writer)
    store.activate_initial(saved.plan_id, binding(old), expected_revision=old.revision)
    store._connection.execute("DROP TRIGGER initial_bindings_no_update")
    store._connection.execute(
        "UPDATE initial_bindings SET binding_json=?",
        (json.dumps(asdict(replace(binding(old), job_id="999"))),),
    )
    with pytest.raises(PlanRecoveryRequired):
        store.get(saved.plan_id)
    with pytest.raises(PlanRecoveryRequired):
        store.list_pending()


def test_initial_native_revision_corruption_refuses_recovery(writer):
    store, _, saved = writer
    old = prepare(writer)
    store.activate_initial(saved.plan_id, binding(old), expected_revision=old.revision)
    store._connection.execute(
        "UPDATE submission_control SET revision=revision+1 WHERE plan_id=?",
        (saved.plan_id,),
    )
    with pytest.raises(PlanRecoveryRequired, match="binding evidence"):
        store.get(saved.plan_id)


def test_first_seen_scheduler_restart_binds_current_generation_and_keeps_ack(writer):
    store, plans, original = writer
    saved = plans.put(
        replace(original.plan, resources=(("requeue", True),)),
        idempotency_key="missed-first-writer",
        request_payload={},
    )
    plans.request_submission(saved.plan_id)
    unbound = store.ensure(saved.plan_id, owner="alice")
    claimed, _ = store.claim_submission(saved.plan_id)
    receipt = native(unbound, saved.plan)
    accepted = store.attach_submission(
        saved.plan_id, receipt, expected_revision=claimed.revision
    )
    accepted = store.observe_submission(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.TERMINAL, exit_code=0),
        expected_revision=accepted.revision,
    )
    proof = replace(binding(accepted), restart_count=2)
    active = store.activate_initial(
        saved.plan_id, proof, expected_revision=accepted.revision
    )
    assert active.submission_handle == receipt
    assert json.loads(active.handle.raw_json)["restart_count"] == 2
    assert json.loads(active.submission_handle.raw_json)["restart_count"] == 0
    assert active.observation is None  # Earlier-generation evidence is not rebound.
    assert active.submission_observation == accepted.submission_observation
    next_proof = replace(proof, attempt=9, restart_count=3)
    next_handle = replace(
        active.handle,
        attempt=9,
        raw_json=json.dumps(json.loads(active.handle.raw_json) | {"restart_count": 3}),
    )
    for mutation in (
        {"allocation_mode": "reserved"},
        {"plan_sha256": "f" * 64},
        {"initial_attempt": None},
        {"initial_attempt": active.attempt},
    ):
        with pytest.raises(StaleExecution, match="immutable allocation plan"):
            store.activate_requeue(
                saved.plan_id,
                next_proof,
                replace(
                    next_handle,
                    raw_json=json.dumps(json.loads(next_handle.raw_json) | mutation),
                ),
                expected_revision=active.revision,
            )
        assert store.get(saved.plan_id) == active
    current = store.activate_requeue(
        saved.plan_id, next_proof, next_handle, expected_revision=active.revision
    )
    assert current.attempt == 9 and current.submission_handle == receipt
    assert store.history(saved.plan_id)[0].handle == active.handle


def test_actual_version8_upgrade_preserves_legacy_evidence_bytes_and_consistent_backup(
    writer,
):
    store, plans, original = writer
    manifest = replace(
        original.plan.manifest, run_id="00000000-0000-4000-8000-000000000008"
    )
    saved = plans.put(
        replace(
            original.plan,
            manifest_json=json.dumps(asdict(manifest)),
            allocation_mode="reserved",
            resources=(("requeue", True),),
        ),
        idempotency_key="v8-legacy",
        request_payload={},
    )
    plans.request_submission(saved.plan_id)
    first = store.ensure(saved.plan_id, 1, owner="alice")
    claimed, _ = store.claim_submission(saved.plan_id)
    store.record_handle(
        saved.plan_id,
        native(first, saved.plan).bind_attempt(1),
        expected_revision=claimed.revision,
    )
    with (
        RuntimeProvenanceStore(
            store.path, dispatch_lock=store.dispatch_lock
        ) as provenance,
        CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections,
        Registry(store.path, dispatch_lock=store.dispatch_lock) as registry,
    ):
        hello = Envelope(
            first.run_id,
            1,
            "hello",
            0,
            "2026-10-04T00:00:00Z",
            asdict(
                HelloPayload(1, 1, "actual", "3.11", "cpu", ["train"], sys.executable)
            ),
        )
        provenance.record(saved.plan_id, hello)
        registry.apply_events(
            first.run_id,
            1,
            [
                hello,
                Envelope(
                    first.run_id,
                    1,
                    "metric",
                    1,
                    hello.ts,
                    {"name": "loss", "value": 0.5, "step": 1},
                ),
            ],
        )
        success = OperationOutcome(OutcomeStatus.SUCCESS, exit_code=0)
        current = store.get(saved.plan_id)
        collections.record_collection(
            saved.plan_id,
            current.handle,
            OperationResult(OperationKind.TRAIN, success, success, success),
        )
        native_raw = json.loads(current.handle.raw_json) | {
            "restart_count": 1,
            "initial_attempt": 1,
        }
        restart = SchedulerBinding(
            first.run_id, first.request_id, "alpha", "123", "alice", 1, 2
        )
        store.activate_requeue(
            saved.plan_id,
            restart,
            replace(current.handle, attempt=2, raw_json=json.dumps(native_raw)),
            expected_revision=current.revision,
        )
    tables = (
        "plans",
        "plan_control",
        "execution_control",
        "execution_attempt_history",
        "requeue_bindings",
        "runtime_provenance",
        "execution_collections",
        "runs",
        "cursors",
        "metric_records",
        "event_log",
    )
    # Snapshot exact stored columns, not just decoded semantic comparisons.
    tables = tuple(
        name
        for name in tables
        if store._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    )
    before = {
        name: [
            tuple(row)
            for row in store._connection.execute(f"SELECT * FROM {name} ORDER BY rowid")
        ]
        for name in tables
    }
    connection = store._connection
    connection.execute("DROP TRIGGER writer_initial_activation")
    connection.execute("DROP TABLE initial_bindings")
    connection.execute("DROP TABLE submission_control")
    connection.execute(
        "DELETE FROM schema_migrations WHERE migration_id='0009_writer_submissions'"
    )
    connection.execute("PRAGMA user_version=8")
    with Registry(store.path, dispatch_lock=store.dispatch_lock):
        pass
    after = {
        name: [
            tuple(row)
            for row in connection.execute(f"SELECT * FROM {name} ORDER BY rowid")
        ]
        for name in tables
    }
    assert after == before
    with sqlite3.connect(backup_path(store.path, "0009_writer_submissions")) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 8
        assert {
            name: backup.execute(f"SELECT * FROM {name} ORDER BY rowid").fetchall()
            for name in tables
        } == before
    assert store.get(saved.plan_id).attempt == 2
    assert store.history(saved.plan_id)[0].attempt == 1
