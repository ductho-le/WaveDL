# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Immutable native wrapper outcomes without a fabricated runtime attempt."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import UUID

from wavedl_daemon.backends.contracts import (
    OperationResult,
    OutcomeStatus,
    Plan,
    SubmissionHandle,
)
from wavedl_daemon.storage.collections import _json, _result, _strict_data
from wavedl_daemon.storage.executions import StaleExecution
from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version
from wavedl_daemon.storage.submissions import validate_writer_receipt


def _create_tables(connection):
    if schema_version(connection) != 9:
        raise MigrationFailed("Native submission collections require schema 9")
    connection.execute(
        "CREATE TABLE submission_collections ("
        "plan_id TEXT PRIMARY KEY REFERENCES submission_control(plan_id), "
        "handle_json TEXT NOT NULL CHECK(length(CAST(handle_json AS BLOB))<=262144), "
        "result_json TEXT NOT NULL CHECK(length(CAST(result_json AS BLOB))<=262144), "
        "observed_at TEXT NOT NULL)"
    )
    for operation in ("UPDATE", "DELETE"):
        connection.execute(
            f"CREATE TRIGGER submission_collections_no_{operation.lower()} BEFORE {operation} "
            "ON submission_collections BEGIN SELECT RAISE(ABORT, 'native collected outcome is immutable'); END"
        )


SUBMISSION_COLLECTION_MIGRATION = Migration(
    "0010_submission_collections", 10, _create_tables
)


@dataclass(frozen=True)
class SubmissionCollectionRecord:
    plan_id: str
    handle_json: str
    result_json: str
    observed_at: str

    @property
    def handle(self):
        return SubmissionHandle.from_json(self.handle_json)

    @property
    def result(self):
        return _result(self.result_json)


def _validate(plan_id, handle, result, plan, request_id, owner, expected):
    try:
        validate_writer_receipt(handle, plan)
    except ValueError as error:
        raise StaleExecution(str(error)) from error
    if (
        str(UUID(plan_id)) != plan_id
        or handle != SubmissionHandle.from_json(expected)
        or handle.backend != "slurm"
        or _strict_data(handle.raw_json).get("restart_count") != 0
        or (
            handle.run_id,
            handle.request_id,
            handle.owner,
            handle.backend,
            handle.run_dir,
        )
        != (plan.run_id, request_id, owner, plan.target, plan.run_dir)
        or plan.allocation_mode != "writer"
    ):
        raise StaleExecution("Native collection differs from approved submission")
    if result.operation != plan.operation:
        raise StaleExecution("Collected operation differs from immutable plan")
    if result.runtime.status != OutcomeStatus.UNKNOWN:
        raise StaleExecution("Unbound collection cannot claim runtime evidence")


def get_submission(store, plan_id):
    store._guard()
    if schema_version(store._connection) < 10:
        return None
    row = store._connection.execute(
        "SELECT c.*, p.plan_json,p.request_id,s.owner,s.submission_handle_json "
        "FROM submission_collections c LEFT JOIN plans p ON p.plan_id=c.plan_id "
        "LEFT JOIN submission_control s ON s.plan_id=c.plan_id WHERE c.plan_id=?",
        (plan_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        record = SubmissionCollectionRecord(
            row["plan_id"], row["handle_json"], row["result_json"], row["observed_at"]
        )
        _strict_data(record.handle_json)
        _validate(
            record.plan_id,
            record.handle,
            record.result,
            Plan.from_json(row["plan_json"]),
            row["request_id"],
            row["owner"],
            row["submission_handle_json"],
        )
        if (
            record.result.batch.status == OutcomeStatus.UNKNOWN
            or record.result.stage_out.status == OutcomeStatus.UNKNOWN
        ):
            raise ValueError("Incomplete native collection")
        if (
            not isinstance(record.observed_at, str)
            or len(record.observed_at) > 64
            or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
                record.observed_at,
            )
        ):
            raise ValueError("Invalid native collection timestamp")
        observed = datetime.fromisoformat(record.observed_at.replace("Z", "+00:00"))
        if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
            raise ValueError("Native collection timestamp must be UTC")
        return record
    except (
        AttributeError,
        KeyError,
        RecursionError,
        TypeError,
        ValueError,
        StaleExecution,
    ) as error:
        raise PlanRecoveryRequired(
            "Stored native collection evidence is damaged"
        ) from error


def record_submission_collection(store, plan_id, source_handle, result):
    if not isinstance(source_handle, SubmissionHandle) or not isinstance(
        result, OperationResult
    ):
        raise ValueError("Typed native submission handle/result required")
    if store.read_only:
        raise PermissionError("Collection storage is read-only")
    require_dispatch(store.dispatch_lock)
    if store.dispatch_lock.path.resolve() != store.path.parent / "dispatch.lock":
        raise DispatchRequired("Collection ownership belongs to another registry")
    store._guard()
    if schema_version(store._connection) not in {10, 11, 12, 13}:
        raise PlanRecoveryRequired("Native collections require schema 10")
    encoded_handle, encoded_result = (
        _json(source_handle.to_dict()),
        _json(asdict(result)),
    )
    store._connection.execute("BEGIN IMMEDIATE")
    try:
        store._guard()
        row = store._connection.execute(
            "SELECT p.plan_json,p.request_id,s.owner,s.submission_handle_json,"
            "EXISTS(SELECT 1 FROM execution_control e WHERE e.plan_id=s.plan_id) AS bound "
            "FROM submission_control s JOIN plans p ON p.plan_id=s.plan_id WHERE s.plan_id=?",
            (plan_id,),
        ).fetchone()
        if row is None:
            raise KeyError(plan_id)
        if row["bound"] or row["submission_handle_json"] is None:
            raise StaleExecution(
                "Native collection source is not the current unbound submission"
            )
        _validate(
            plan_id,
            source_handle,
            result,
            Plan.from_json(row["plan_json"]),
            row["request_id"],
            row["owner"],
            row["submission_handle_json"],
        )
        if (
            result.batch.status == OutcomeStatus.UNKNOWN
            or result.stage_out.status == OutcomeStatus.UNKNOWN
        ):
            store._connection.execute("COMMIT")
            return None
        existing = get_submission(store, plan_id)
        if existing is not None:
            if (
                existing.handle_json != encoded_handle
                or existing.result_json != encoded_result
            ):
                raise StaleExecution("Conflicting finalized native collection")
            store._connection.execute("COMMIT")
            return existing
        store._connection.execute(
            "INSERT INTO submission_collections VALUES (?,?,?,?)",
            (plan_id, encoded_handle, encoded_result, datetime.now(UTC).isoformat()),
        )
        record = get_submission(store, plan_id)
        store._connection.execute("COMMIT")
        return record
    except BaseException:
        if store._connection.in_transaction:
            store._connection.execute("ROLLBACK")
        raise
