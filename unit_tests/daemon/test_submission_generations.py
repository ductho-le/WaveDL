"""Generation-scoped native evidence is CAS fenced and durable after activation."""

import hashlib
import sqlite3
from dataclasses import replace

import pytest

from unit_tests.daemon import test_writer_submissions as writers
from unit_tests.daemon.test_collections import result
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    OperationOutcome,
    OutcomeStatus,
    SubmissionCollectionResult,
    SubmissionObservation,
)
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.executions import StaleExecution, _manifest_sha256
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.schema import backup_path
from wavedl_daemon.storage.submission_generations import receipt_identity


writer = writers.writer


def native_result(**kwargs):
    return replace(result(**kwargs), runtime=OperationOutcome(OutcomeStatus.UNKNOWN))


def evidence(writer, restart=0):
    store, _, saved = writer
    current = writers.prepare(writer)
    identity = receipt_identity(current.submission_handle, saved.plan, restart)
    observation = SubmissionObservation(
        ObservationStatus.TERMINAL, "FAILED", native_identity=identity
    )
    current = store.observe_submission(
        saved.plan_id, observation, expected_revision=current.revision
    )
    proposal = SubmissionCollectionResult(
        identity,
        hashlib.sha256(saved.plan.to_json().encode()).hexdigest(),
        _manifest_sha256(saved.plan),
        native_result(),
        observation,
    )
    return current, proposal


def test_generation_collection_revision_identity_immutability_and_activation(writer):
    store, _, saved = writer
    current, proposal = evidence(writer)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        with pytest.raises(StaleExecution, match="revision"):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                proposal,
                expected_revision=current.revision - 1,
            )
        record = collections.record_submission_generation(
            saved.plan_id,
            current.submission_handle,
            proposal,
            expected_revision=current.revision,
        )
        assert record.identity == proposal.identity and record.result == proposal.result
        assert (
            collections.get_submission_generation(saved.plan_id, proposal.identity)
            == record
        )
        for sql in (
            "DELETE FROM submission_generation_collections",
            "UPDATE submission_generation_collections SET observed_at='bad'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                collections._connection.execute(sql)
        with pytest.raises(StaleExecution):
            collections.get_submission_generation(
                saved.plan_id, replace(proposal.identity, owner="mallory")
            )
        store.activate_initial(
            saved.plan_id, writers.binding(current), expected_revision=current.revision
        )
        assert (
            collections.get_submission_generation(saved.plan_id, proposal.identity)
            == record
        )
        with pytest.raises(StaleExecution, match="unbound"):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                proposal,
                expected_revision=current.revision,
            )
    with CollectionStore(store.path, read_only=True) as collections:
        assert (
            collections.get_submission_generation(saved.plan_id, proposal.identity)
            == record
        )
        with pytest.raises(PermissionError):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                proposal,
                expected_revision=current.revision,
            )


def test_unknown_and_foreign_proposals_never_finalize(writer):
    store, _, saved = writer
    current, proposal = evidence(writer)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        incomplete = replace(
            proposal,
            result=native_result(batch=OutcomeStatus.UNKNOWN),
        )
        assert (
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                incomplete,
                expected_revision=current.revision,
            )
            is None
        )
        with pytest.raises(StaleExecution):
            collections.record_submission_generation(
                saved.plan_id,
                current.submission_handle,
                replace(proposal, manifest_sha256="a" * 64),
                expected_revision=current.revision,
            )
        assert (
            collections.get_submission_generation(saved.plan_id, proposal.identity)
            is None
        )
        collections.record_submission_generation(
            saved.plan_id,
            current.submission_handle,
            proposal,
            expected_revision=current.revision,
        )
        collections._connection.execute(
            "DROP TRIGGER submission_generation_collections_no_update"
        )
        collections._connection.execute(
            "UPDATE submission_generation_collections SET proposal_json='{}'"
        )
        with pytest.raises(PlanRecoveryRequired, match="damaged"):
            collections.get_submission_generation(saved.plan_id, proposal.identity)


def test_actual_schema10_upgrade_retains_legacy_collection_and_backup(writer):
    store, _, saved = writer
    current = writers.prepare(writer)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        collections.record_submission_collection(
            saved.plan_id,
            current.submission_handle,
            result(runtime=OutcomeStatus.UNKNOWN),
        )
    before = tuple(
        store._connection.execute("SELECT * FROM submission_collections").fetchone()
    )
    store._connection.execute("DROP TABLE submission_native_lifecycle")
    store._connection.execute("DROP TABLE submission_native_generations")
    store._connection.execute("DELETE FROM schema_migrations WHERE to_version>=12")
    store._connection.execute("DROP TABLE submission_generation_collections")
    store._connection.execute("DELETE FROM schema_migrations WHERE to_version=11")
    store._connection.execute("PRAGMA user_version=10")
    with Registry(store.path, dispatch_lock=store.dispatch_lock):
        assert (
            tuple(
                store._connection.execute(
                    "SELECT * FROM submission_collections"
                ).fetchone()
            )
            == before
        )
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 13
    with sqlite3.connect(
        backup_path(store.path, "0011_submission_generation_collections")
    ) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 10
        assert (
            backup.execute("SELECT * FROM submission_collections").fetchone() == before
        )


def test_unapproved_forward_and_late_or_foreign_generation_observations_rejected(
    writer,
):
    store, _, saved = writer
    current, proposal = evidence(writer)
    for identity in (
        replace(proposal.identity, restart_count=2),
        replace(proposal.identity, cluster="foreign"),
    ):
        with pytest.raises(StaleExecution):
            store.observe_submission(
                saved.plan_id,
                SubmissionObservation(
                    ObservationStatus.QUEUED, native_identity=identity
                ),
                expected_revision=current.revision,
            )
    assert store.get(saved.plan_id).submission_observation == proposal.observation


def test_approved_forward_generation_collection_retains_ack_and_refuses_late(writer):
    store, plans, saved = writer
    saved = plans.put(
        replace(saved.plan, resources=(("requeue", True),)),
        idempotency_key="approved-requeue",
        request_payload={"requeue": True},
    )
    plans.request_submission(saved.plan_id)
    current, first = evidence((store, plans, saved))
    receipt = current.submission_handle
    identity = replace(first.identity, restart_count=2)
    current = store.observe_submission(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.QUEUED, native_identity=identity),
        expected_revision=current.revision,
    )
    with pytest.raises(StaleExecution, match="late"):
        store.observe_submission(
            saved.plan_id, first.observation, expected_revision=current.revision
        )
    current = store.observe_submission(
        saved.plan_id,
        SubmissionObservation(ObservationStatus.ERROR, detail="temporary outage"),
        expected_revision=current.revision,
    )
    assert current.submission_observation.native_identity == identity
    terminal = SubmissionObservation(
        ObservationStatus.TERMINAL, "FAILED", native_identity=identity
    )
    current = store.observe_submission(
        saved.plan_id, terminal, expected_revision=current.revision
    )
    proposal = replace(first, identity=identity, observation=terminal)
    with CollectionStore(store.path, dispatch_lock=store.dispatch_lock) as collections:
        record = collections.record_submission_generation(
            saved.plan_id, receipt, proposal, expected_revision=current.revision
        )
        assert record.handle == receipt and record.identity.restart_count == 2
        assert (
            collections.get_submission_generation(saved.plan_id, first.identity) is None
        )
        assert collections.get_submission_generation(saved.plan_id, identity) == record
    assert store.get(saved.plan_id).submission_handle == receipt


@pytest.mark.parametrize(
    "observed_restart,binding_restart,expected_exit",
    [(2, 0, "reject"), (2, 2, 42), (0, 2, None), (None, 2, None), (None, 0, 42)],
)
def test_initial_activation_fences_and_copies_only_exact_native_generation(
    writer, observed_restart, binding_restart, expected_exit
):
    store, plans, saved = writer
    saved = plans.put(
        replace(saved.plan, resources=(("requeue", True),)),
        idempotency_key="generation-activation",
        request_payload={"requeue": True},
    )
    plans.request_submission(saved.plan_id)
    current = writers.prepare((store, plans, saved))
    identity = (
        None
        if observed_restart is None
        else receipt_identity(current.submission_handle, saved.plan, observed_restart)
    )
    current = store.observe_submission(
        saved.plan_id,
        SubmissionObservation(
            ObservationStatus.TERMINAL, "FAILED", exit_code=42, native_identity=identity
        ),
        expected_revision=current.revision,
    )
    binding = replace(writers.binding(current), restart_count=binding_restart)
    if expected_exit == "reject":
        with pytest.raises(StaleExecution, match="behind"):
            store.activate_initial(
                saved.plan_id, binding, expected_revision=current.revision
            )
        assert store.get(saved.plan_id).attempt is None
        assert (
            store._connection.execute(
                "SELECT COUNT(*) FROM execution_control WHERE plan_id=?",
                (saved.plan_id,),
            ).fetchone()[0]
            == 0
        )
        return
    active = store.activate_initial(
        saved.plan_id, binding, expected_revision=current.revision
    )
    assert active.attempt == binding.attempt
    if expected_exit is None:
        assert active.observation is None and active.latest_observation is None
    else:
        assert active.observation.exit_code == expected_exit
        assert active.latest_observation.exit_code == expected_exit
    assert store.get(saved.plan_id) == active


def test_active_recovery_refuses_initial_binding_behind_frozen_native_evidence(writer):
    store, plans, saved = writer
    saved = plans.put(
        replace(saved.plan, resources=(("requeue", True),)),
        idempotency_key="generation-recovery",
        request_payload={"requeue": True},
    )
    plans.request_submission(saved.plan_id)
    current, proposal = evidence((store, plans, saved))
    store.activate_initial(
        saved.plan_id, writers.binding(current), expected_revision=current.revision
    )
    corrupt = replace(
        proposal.observation,
        native_identity=replace(proposal.identity, restart_count=2),
    )
    store._connection.execute(
        "UPDATE submission_control SET submission_observation_json=? WHERE plan_id=?",
        (corrupt.to_json(), saved.plan_id),
    )
    with pytest.raises(PlanRecoveryRequired, match="binding is damaged"):
        store.get(saved.plan_id)
