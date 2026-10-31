# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Immutable pre-runtime collections keyed by native scheduler generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    OutcomeStatus,
    Plan,
    SubmissionCollectionResult,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.storage.collections import _json, _strict_data
from wavedl_daemon.storage.executions import StaleExecution, _manifest_sha256
from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version
from wavedl_daemon.storage.submissions import validate_writer_receipt


def _create_tables(connection):
    if schema_version(connection) != 10:
        raise MigrationFailed("Submission generations require schema 10")
    connection.execute(
        "CREATE TABLE submission_generation_collections ("
        "plan_id TEXT NOT NULL REFERENCES submission_control(plan_id), "
        "restart_count INTEGER NOT NULL CHECK(restart_count>=0), "
        "handle_json TEXT NOT NULL CHECK(length(CAST(handle_json AS BLOB))<=262144), "
        "proposal_json TEXT NOT NULL CHECK(length(CAST(proposal_json AS BLOB))<=262144), "
        "observed_at TEXT NOT NULL, PRIMARY KEY(plan_id,restart_count))"
    )
    for operation in ("UPDATE", "DELETE"):
        connection.execute(
            f"CREATE TRIGGER submission_generation_collections_no_{operation.lower()} "
            f"BEFORE {operation} ON submission_generation_collections "
            "BEGIN SELECT RAISE(ABORT, 'native generation collection is immutable'); END"
        )


SUBMISSION_GENERATION_MIGRATION = Migration(
    "0011_submission_generation_collections", 11, _create_tables
)


def receipt_identity(receipt, plan, restart_count):
    """Validate the immutable ACK0 before deriving any generation identity."""
    raw = validate_writer_receipt(receipt, plan)
    if (
        receipt.backend != "slurm"
        or type(raw.get("restart_count")) is not int
        or raw["restart_count"] != 0
        or receipt.job_id is None
        or raw.get("comment") != receipt.request_id
        or raw.get("job_name") != f"wavedl-{receipt.request_id}"
    ):
        raise ValueError("Native generation requires original Slurm ACK0")
    job, cluster = receipt.job_id.split(";")
    if cluster != raw.get("cluster"):
        raise ValueError("ACK cluster differs from native job identity")
    if restart_count > 0 and dict(plan.resources).get("requeue") is not True:
        raise ValueError("Immutable plan did not approve scheduler requeue")
    return SchedulerIdentity(
        receipt.run_id, receipt.request_id, cluster, job, receipt.owner, restart_count
    )


@dataclass(frozen=True)
class SubmissionGenerationRecord:
    plan_id: str
    restart_count: int
    handle_json: str
    proposal_json: str
    observed_at: str

    @property
    def handle(self):
        return SubmissionHandle.from_json(self.handle_json)

    @property
    def proposal(self):
        return SubmissionCollectionResult.from_json(self.proposal_json)

    @property
    def identity(self):
        return self.proposal.identity

    @property
    def result(self):
        return self.proposal.result

    @property
    def observation(self):
        return self.proposal.observation

    @property
    def plan_sha256(self):
        return self.proposal.plan_sha256

    @property
    def manifest_sha256(self):
        return self.proposal.manifest_sha256


def _validate(record, row):
    plan = Plan.from_json(row["plan_json"])
    _strict_data(record.handle_json)
    _strict_data(row["submission_handle_json"])
    _strict_data(record.proposal_json)
    proposal, handle = record.proposal, record.handle
    if (
        str(UUID(record.plan_id)) != record.plan_id
        or type(record.restart_count) is not int
        or record.restart_count != proposal.identity.restart_count
        or handle != SubmissionHandle.from_json(row["submission_handle_json"])
        or (handle.run_id, handle.request_id, handle.owner, handle.run_dir)
        != (plan.run_id, row["request_id"], row["owner"], plan.run_dir)
        or proposal.identity != receipt_identity(handle, plan, record.restart_count)
        or proposal.plan_sha256 != hashlib.sha256(plan.to_json().encode()).hexdigest()
        or proposal.manifest_sha256 != _manifest_sha256(plan)
        or proposal.result.operation != plan.operation
    ):
        raise ValueError("Native generation differs from durable approved submission")
    stamp = datetime.fromisoformat(record.observed_at.replace("Z", "+00:00"))
    if (
        len(record.observed_at) > 64
        or stamp.utcoffset() is None
        or stamp.utcoffset().total_seconds() != 0
    ):
        raise ValueError("Invalid native collection UTC timestamp")
    return record


def get_submission_generation(store, plan_id, identity):
    if not isinstance(identity, SchedulerIdentity):
        raise ValueError("Exact scheduler identity required")
    store._guard()
    if schema_version(store._connection) < 11:
        return None
    row = store._connection.execute(
        "SELECT c.*,p.plan_json,p.request_id,s.owner,s.submission_handle_json "
        "FROM submission_generation_collections c LEFT JOIN plans p ON p.plan_id=c.plan_id "
        "LEFT JOIN submission_control s ON s.plan_id=c.plan_id "
        "WHERE c.plan_id=? AND c.restart_count=?",
        (plan_id, identity.restart_count),
    ).fetchone()
    if row is None:
        return None
    try:
        record = _validate(
            SubmissionGenerationRecord(
                row["plan_id"],
                row["restart_count"],
                row["handle_json"],
                row["proposal_json"],
                row["observed_at"],
            ),
            row,
        )
        if (
            record.result.batch.status == OutcomeStatus.UNKNOWN
            or record.result.stage_out.status == OutcomeStatus.UNKNOWN
        ):
            raise ValueError("Incomplete native generation collection")
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError) as error:
        raise PlanRecoveryRequired(
            "Stored native generation evidence is damaged"
        ) from error
    if record.identity != identity:
        raise StaleExecution("Requested native generation has foreign identity")
    return record


def _validate_bound_source(row, receipt, proposal):
    """Fence a backend-verified attemptless receipt against the active runtime.

    The trusted collector must have read the native schema-2 receipt with
    attempt=None. Its typed proposal carries that generation's terminal proof;
    the previous runtime observation is neither replaced nor used as proof.
    Equal native counts are valid for this receipt branch, but never confer
    authority to claim runtime completion.
    """
    from wavedl_daemon.backends.contracts import Handle

    _strict_data(row["runtime_handle_json"])
    handle = Handle.from_json(row["runtime_handle_json"])
    plan = Plan.from_json(row["plan_json"])
    # SubmissionHandle performs the full scheduler identity validation omitted
    # by the more general positive-runtime Handle contract.
    active = SubmissionHandle(
        **{key: value for key, value in handle.to_dict().items() if key != "attempt"}
    )
    raw = validate_writer_receipt(active, plan)
    if (
        handle.attempt != row["runtime_attempt"]
        or row["runtime_owner"] != receipt.owner
        or any(
            getattr(handle, name) != getattr(receipt, name)
            for name in (
                "backend",
                "run_id",
                "request_id",
                "owner",
                "run_dir",
                "job_id",
                "pid",
                "supervisor_id",
            )
        )
        or proposal.identity.restart_count < raw["restart_count"]
    ):
        raise ValueError("Native collection differs from current positive runtime")
    if (
        proposal.identity.restart_count > raw["restart_count"]
        and dict(plan.resources).get("requeue") is not True
    ):
        raise ValueError("Immutable plan did not approve scheduler requeue")


def record_submission_generation(
    store, plan_id, source_handle, proposal, *, expected_revision
):
    if not isinstance(source_handle, SubmissionHandle) or not isinstance(
        proposal, SubmissionCollectionResult
    ):
        raise ValueError("Typed native submission handle/proposal required")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Explicit source revision required")
    if store.read_only:
        raise PermissionError("Collection storage is read-only")
    require_dispatch(store.dispatch_lock)
    if store.dispatch_lock.path.resolve() != store.path.parent / "dispatch.lock":
        raise DispatchRequired("Collection ownership belongs to another registry")
    store._guard()
    if schema_version(store._connection) not in {11, 12, 13}:
        raise PlanRecoveryRequired("Native generations require schema 11 or 12")
    record = SubmissionGenerationRecord(
        plan_id,
        proposal.identity.restart_count,
        _json(source_handle.to_dict()),
        _json(proposal.to_dict()),
        datetime.now(UTC).isoformat(),
    )
    store._connection.execute("BEGIN IMMEDIATE")
    try:
        store._guard()
        row = store._connection.execute(
            "SELECT p.plan_json,p.request_id,s.*, e.attempt AS runtime_attempt, "
            "e.handle_json AS runtime_handle_json,e.owner AS runtime_owner, "
            "e.revision AS runtime_revision "
            "FROM submission_control s JOIN plans p ON p.plan_id=s.plan_id "
            "LEFT JOIN execution_control e ON e.plan_id=s.plan_id WHERE s.plan_id=?",
            (plan_id,),
        ).fetchone()
        if row is None:
            raise KeyError(plan_id)
        bound = row["runtime_attempt"] is not None
        revision = row["runtime_revision"] if bound else row["revision"]
        if revision != expected_revision or row["submission_handle_json"] is None:
            raise StaleExecution(
                "Native generation source is not current bound execution/unbound submission revision"
            )
        try:
            _validate(record, row)
            from wavedl_daemon.storage.native_heads import get_native_head

            head = get_native_head(store._connection, plan_id)
            if (bound or head is not None) and proposal.identity != head:
                raise ValueError(
                    "Native collection source differs from durable native head"
                )
            if bound:
                _validate_bound_source(row, source_handle, proposal)
            if schema_version(store._connection) >= 13:
                from wavedl_daemon.storage.native_lifecycle import (
                    read_native_lifecycle,
                )

                # Collection can itself discover a newer terminal generation
                # before the fleet refreshes the legacy submission snapshot.
                # Both bound and unbound receipts need the exact current native
                # lifecycle proof, never an older known-good generation.
                known, _ = read_native_lifecycle(
                    store._connection, plan_id, proposal.identity
                )
                if (
                    known is None
                    or known.status != ObservationStatus.TERMINAL
                    or known.native_identity != proposal.identity
                ):
                    raise ValueError(
                        "Native generation lacks current terminal lifecycle proof"
                    )
            elif not bound:
                latest = SubmissionObservation.from_json(
                    row["submission_observation_json"]
                )
                if (
                    latest.status != ObservationStatus.TERMINAL
                    or latest.native_identity != proposal.identity
                ):
                    raise ValueError(
                        "Native generation lacks current matching terminal evidence"
                    )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise StaleExecution(str(error)) from error
        if (
            proposal.result.batch.status == OutcomeStatus.UNKNOWN
            or proposal.result.stage_out.status == OutcomeStatus.UNKNOWN
        ):
            store._connection.execute("COMMIT")
            return None
        existing = get_submission_generation(store, plan_id, proposal.identity)
        if existing is not None:
            if (
                existing.handle_json != record.handle_json
                or existing.proposal_json != record.proposal_json
            ):
                raise StaleExecution(
                    "Conflicting finalized native generation collection"
                )
            store._connection.execute("COMMIT")
            return existing
        store._connection.execute(
            "INSERT INTO submission_generation_collections VALUES (?,?,?,?,?)",
            (
                record.plan_id,
                record.restart_count,
                record.handle_json,
                record.proposal_json,
                record.observed_at,
            ),
        )
        store._connection.execute("COMMIT")
        return record
    except BaseException:
        if store._connection.in_transaction:
            store._connection.execute("ROLLBACK")
        raise
