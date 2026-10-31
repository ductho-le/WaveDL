# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Pre-runtime native submissions without a fabricated runtime attempt."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


if TYPE_CHECKING:
    import sqlite3

    from wavedl_daemon.backends.contracts import Plan, SubmissionHandle


def validate_writer_receipt(receipt: SubmissionHandle, plan: Plan) -> dict:
    """Validate scheduler allocation metadata against immutable writer approval."""
    raw = json.loads(receipt.raw_json)
    if receipt.backend == "slurm" and (
        plan.allocation_mode != "writer"
        or raw.get("allocation_mode") != "writer"
        or raw.get("plan_sha256")
        != hashlib.sha256(plan.to_json().encode("utf-8")).hexdigest()
        or raw.get("operation") != plan.operation.value
        or "initial_attempt" in raw
    ):
        raise ValueError("Native receipt differs from immutable writer allocation plan")
    return raw


def _create_tables(connection: sqlite3.Connection):
    if schema_version(connection) != 8:
        raise MigrationFailed("Writer submissions require schema 8")
    connection.execute(
        "CREATE TABLE submission_control ("
        "plan_id TEXT PRIMARY KEY REFERENCES plans(plan_id), owner TEXT NOT NULL, "
        "state TEXT NOT NULL CHECK(state IN ('pending','submitting','unknown','submitted','rejected','local_lost')), "
        "submission_handle_json TEXT CHECK(length(CAST(submission_handle_json AS BLOB))<=262144), "
        "submission_observation_json TEXT CHECK(length(CAST(submission_observation_json AS BLOB))<=262144), "
        "latest_submission_observation_json TEXT CHECK(length(CAST(latest_submission_observation_json AS BLOB))<=262144), "
        "cancel_delivered INTEGER NOT NULL DEFAULT 0 CHECK(cancel_delivered IN (0,1)), "
        "revision INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '')"
    )
    connection.execute(
        "CREATE TABLE initial_bindings ("
        "plan_id TEXT PRIMARY KEY REFERENCES plans(plan_id), attempt INTEGER NOT NULL CHECK(attempt>0), "
        "previous_revision INTEGER NOT NULL CHECK(previous_revision>=0), binding_json TEXT NOT NULL "
        "CHECK(length(CAST(binding_json AS BLOB))<=262144), "
        "native_receipt_json TEXT NOT NULL CHECK(length(CAST(native_receipt_json AS BLOB))<=262144), "
        "runtime_handle_json TEXT NOT NULL CHECK(length(CAST(runtime_handle_json AS BLOB))<=262144), "
        "observed_at TEXT NOT NULL)"
    )
    for operation in ("UPDATE", "DELETE"):
        connection.execute(
            f"CREATE TRIGGER initial_bindings_no_{operation.lower()} BEFORE {operation} ON initial_bindings "
            "BEGIN SELECT RAISE(ABORT, 'initial runtime binding is immutable'); END"
        )
    connection.execute(
        "CREATE TRIGGER submission_identity_immutable BEFORE UPDATE OF plan_id,owner ON submission_control "
        "BEGIN SELECT RAISE(ABORT, 'submission identity is immutable'); END"
    )
    connection.execute(
        "CREATE TRIGGER submission_receipt_immutable BEFORE UPDATE OF submission_handle_json ON submission_control "
        "WHEN OLD.submission_handle_json IS NOT NULL AND NEW.submission_handle_json IS NOT OLD.submission_handle_json "
        "BEGIN SELECT RAISE(ABORT, 'native submission receipt is immutable'); END"
    )
    connection.execute(
        "CREATE TRIGGER submission_writer_mode BEFORE INSERT ON submission_control "
        "WHEN coalesce(json_extract((SELECT plan_json FROM plans WHERE plan_id=NEW.plan_id),'$.allocation_mode'),'reserved')<>'writer' "
        "BEGIN SELECT RAISE(ABORT, 'unbound submission requires writer allocation mode'); END"
    )
    connection.execute(
        "CREATE TRIGGER writer_initial_activation BEFORE INSERT ON execution_control "
        "WHEN json_extract((SELECT plan_json FROM plans WHERE plan_id=NEW.plan_id),'$.allocation_mode')='writer' "
        "BEGIN SELECT RAISE(ABORT, 'writer execution lacks initial verified binding') WHERE NOT EXISTS "
        "(SELECT 1 FROM initial_bindings b JOIN submission_control s ON s.plan_id=b.plan_id "
        "WHERE b.plan_id=NEW.plan_id AND b.attempt=NEW.attempt AND b.runtime_handle_json=NEW.handle_json "
        "AND b.native_receipt_json=s.submission_handle_json AND b.previous_revision=s.revision "
        "AND NEW.owner=s.owner AND NEW.revision=s.revision+1 AND NEW.state='submitted'); END"
    )


SUBMISSION_MIGRATION = Migration("0009_writer_submissions", 9, _create_tables)
