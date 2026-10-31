"""Pre-runtime native receipts never rewrite an earlier positive runtime."""

import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from unit_tests.daemon import (
    test_native_heads as heads,
    test_writer_submissions as writers,
)
from unit_tests.daemon.test_collections import result
from unit_tests.daemon.test_submission_generations import native_result
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    OperationOutcome,
    OutcomeStatus,
    SubmissionCollectionResult,
    SubmissionObservation,
)
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import (
    ExecutionStore,
    StaleExecution,
    _manifest_sha256,
)
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.submission_generations import receipt_identity


writer = writers.writer


def bound(writer):
    store, saved, current = heads.prepare(writer)
    current = store.activate_initial(
        saved.plan_id, writers.binding(current), expected_revision=current.revision
    )
    return store, saved, current


def proposal(current, saved, count):
    identity = receipt_identity(current.submission_handle, saved.plan, count)
    return SubmissionCollectionResult(
        identity,
        hashlib.sha256(saved.plan.to_json().encode()).hexdigest(),
        _manifest_sha256(saved.plan),
        native_result(batch=OutcomeStatus.FAILED),
        SubmissionObservation(
            ObservationStatus.TERMINAL, "FAILED", native_identity=identity
        ),
    )


def snapshot(store):
    return {
        table: store._connection.execute(f"SELECT * FROM {table}").fetchall()
        for table in (
            "execution_control",
            "execution_attempt_history",
            "execution_collections",
            "initial_bindings",
            "submission_control",
            "submission_native_generations",
        )
    }


def test_bound_runtime0_native3_collection_preserves_runtime_and_recovery(writer):
    store, saved, current = bound(writer)
    source = current.submission_handle
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        runtime_collection = collections.record_collection(
            saved.plan_id, current.handle, result()
        )
        pending = proposal(current, saved, 3)
        current = store.record_native_generation(
            saved.plan_id, pending.identity, expected_revision=current.revision
        )
        current = store.observe_native_generation(
            saved.plan_id, pending.observation, expected_revision=current.revision
        )
        before = snapshot(store)
        native_collection = collections.record_submission_generation(
            saved.plan_id, source, pending, expected_revision=current.revision
        )
        assert snapshot(store) == before
        assert store.get(saved.plan_id) == current
        assert json.loads(current.handle.raw_json)["restart_count"] == 0
        assert collections.get(saved.plan_id) == runtime_collection
        assert native_collection.identity.restart_count == 3
        assert native_collection.result.runtime == OperationOutcome(
            OutcomeStatus.UNKNOWN
        )
        binding = replace(writers.binding(current), restart_count=5, attempt=8)
        handle = replace(
            current.handle,
            attempt=8,
            raw_json=json.dumps(
                json.loads(current.handle.raw_json) | {"restart_count": 5}
            ),
        )
        activated = store.activate_requeue(
            saved.plan_id, binding, handle, expected_revision=current.revision
        )
        for revision in (current.revision, activated.revision):
            with pytest.raises(StaleExecution):
                collections.record_submission_generation(
                    saved.plan_id, source, pending, expected_revision=revision
                )
        assert collections.history(saved.plan_id) == [runtime_collection]
        assert (
            collections.get_submission_generation(saved.plan_id, pending.identity)
            == native_collection
        )
    with (
        ExecutionStore(store.path, read_only=True) as recovered,
        CollectionStore(store.path, read_only=True) as collections,
    ):
        assert recovered.get(saved.plan_id) == activated
        assert collections.history(saved.plan_id) == [runtime_collection]
        assert (
            collections.get_submission_generation(saved.plan_id, pending.identity)
            == native_collection
        )


def test_bound_native_collection_revision_head_hash_and_atomic_rollback(writer):
    store, saved, current = bound(writer)
    pending = proposal(current, saved, 3)
    old_revision = current.revision
    current = store.record_native_generation(
        saved.plan_id, pending.identity, expected_revision=current.revision
    )
    current = store.observe_native_generation(
        saved.plan_id, pending.observation, expected_revision=current.revision
    )
    before = snapshot(store)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        invalid = (
            (pending, old_revision),
            (proposal(current, saved, 0), current.revision),
            (replace(pending, plan_sha256="a" * 64), current.revision),
            (replace(pending, manifest_sha256="a" * 64), current.revision),
        )
        for evidence, revision in invalid:
            with pytest.raises(StaleExecution):
                collections.record_submission_generation(
                    saved.plan_id,
                    current.submission_handle,
                    evidence,
                    expected_revision=revision,
                )
        collections._connection.execute(
            "CREATE TRIGGER fail_native_collection BEFORE INSERT ON "
            "submission_generation_collections BEGIN SELECT RAISE(ABORT,'test CAS abort'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="CAS abort"):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                pending,
                expected_revision=current.revision,
            )
        assert not collections._connection.in_transaction
        assert (
            collections.get_submission_generation(saved.plan_id, pending.identity)
            is None
        )
        assert snapshot(store) == before
        collections._connection.execute("DROP TRIGGER fail_native_collection")
        collections.record_submission_generation(
            saved.plan_id,
            current.submission_handle,
            pending,
            expected_revision=current.revision,
        )
        collections._connection.execute(
            "DROP TRIGGER submission_generation_collections_no_update"
        )
        collections._connection.execute(
            "UPDATE submission_generation_collections SET proposal_json='{}'"
        )
        with pytest.raises(PlanRecoveryRequired, match="damaged"):
            collections.get_submission_generation(saved.plan_id, pending.identity)


def test_equal_bound_native_count_keeps_unknown_runtime_without_fake_attempt(writer):
    store, saved, current = bound(writer)
    pending = proposal(current, saved, 0)
    current = store.observe_native_generation(
        saved.plan_id, pending.observation, expected_revision=current.revision
    )
    before = snapshot(store)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_submission_generation(
            saved.plan_id,
            current.submission_handle,
            pending,
            expected_revision=current.revision,
        )
        assert record.result.runtime.status == OutcomeStatus.UNKNOWN
        assert record.result.runtime.exit_code is None
        assert record.result.runtime.signal is None
        assert collections.get(saved.plan_id) is None
        assert snapshot(store) == before
        with pytest.raises(ValueError, match="runtime outcome"):
            replace(pending, result=result())


@pytest.mark.parametrize(
    "mutation",
    [
        {"plan_sha256": "a" * 64},
        {"operation": "test"},
        {"restart_count": 4},
        {"cluster": "foreign"},
    ],
)
def test_bound_source_cannot_use_foreign_or_newer_active_runtime(writer, mutation):
    store, saved, current = bound(writer)
    pending = proposal(current, saved, 3)
    current = store.record_native_generation(
        saved.plan_id, pending.identity, expected_revision=current.revision
    )
    damaged = replace(
        current.handle,
        raw_json=json.dumps(json.loads(current.handle.raw_json) | mutation),
    )
    # Storage must independently reject a damaged positive source, rather than
    # trusting only the immutable ACK0 and a matching independent proposal.
    store._connection.execute(
        "UPDATE execution_control SET handle_json=? WHERE plan_id=?",
        (damaged.to_json(), saved.plan_id),
    )
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        with pytest.raises(StaleExecution):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                pending,
                expected_revision=current.revision,
            )
        assert (
            collections.get_submission_generation(saved.plan_id, pending.identity)
            is None
        )


def test_bound_collection_native_head_advance_fences_new_revision(writer):
    store, saved, current = bound(writer)
    pending = proposal(current, saved, 3)
    current = store.record_native_generation(
        saved.plan_id, pending.identity, expected_revision=current.revision
    )
    current = store.record_native_generation(
        saved.plan_id,
        proposal(current, saved, 5).identity,
        expected_revision=current.revision,
    )
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        with pytest.raises(StaleExecution, match="head"):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                pending,
                expected_revision=current.revision,
            )
        assert (
            collections.get_submission_generation(saved.plan_id, pending.identity)
            is None
        )
