# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Immutable independently collected outcomes, never reconstructed from telemetry."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from wavedl.core.layout import OperationKind
from wavedl.core.protocol import strict_loads
from wavedl_daemon.backends.contracts import (
    Handle,
    OperationOutcome,
    OperationResult,
    OutcomeStatus,
    Plan,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.executions import StaleExecution
from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


MAX_JSON_BYTES = 262144
_SELECT = (
    "SELECT c.*,p.plan_json AS approved_plan,p.request_id AS approved_request, "
    "e.owner AS approved_owner, CASE WHEN c.attempt=e.attempt THEN e.handle_json "
    "ELSE h.handle_json END AS expected_handle "
    "FROM execution_collections c LEFT JOIN plans p ON p.plan_id=c.plan_id "
    "LEFT JOIN execution_control e ON e.plan_id=c.plan_id "
    "LEFT JOIN execution_attempt_history h ON h.plan_id=c.plan_id AND h.attempt=c.attempt "
)


def _create_tables(connection: sqlite3.Connection):
    if schema_version(connection) != 7:
        raise MigrationFailed("Collected outcomes require schema 7")
    connection.execute(
        "CREATE TABLE execution_collections ("
        "plan_id TEXT NOT NULL REFERENCES execution_control(plan_id), "
        "attempt INTEGER NOT NULL CHECK(attempt>0), handle_json TEXT NOT NULL "
        "CHECK(length(CAST(handle_json AS BLOB))<=262144), "
        "result_json TEXT NOT NULL CHECK(length(CAST(result_json AS BLOB))<=262144), "
        "observed_at TEXT NOT NULL, PRIMARY KEY(plan_id,attempt))"
    )
    for operation in ("UPDATE", "DELETE"):
        connection.execute(
            f"CREATE TRIGGER execution_collections_no_{operation.lower()} BEFORE {operation} "
            "ON execution_collections BEGIN SELECT RAISE(ABORT, 'collected outcome is immutable'); END"
        )


COLLECTION_MIGRATION = Migration("0008_execution_collections", 8, _create_tables)


def _json(value):
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise BudgetExhausted("Collected outcome exceeds byte bound")
    return encoded


def _result(encoded):
    value = _strict_data(encoded)
    value["operation"] = OperationKind(value["operation"])
    for name in ("runtime", "batch", "stage_out"):
        value[name]["status"] = OutcomeStatus(value[name]["status"])
        value[name] = OperationOutcome(**value[name])
    return OperationResult(**value)


def _strict_data(encoded):
    if not isinstance(encoded, str) or len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Collected JSON exceeds byte bound")
    value = strict_loads(encoded)

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate collected evidence field")
            result[key] = item
        return result

    json.loads(encoded, object_pairs_hook=unique)
    return value


@dataclass(frozen=True)
class CollectionRecord:
    plan_id: str
    attempt: int
    handle_json: str
    result_json: str
    observed_at: str

    @property
    def handle(self):
        return Handle.from_json(self.handle_json)

    @property
    def result(self):
        return _result(self.result_json)


class CollectionStore:
    """Owner-thread SQLite records of verified complete backend receipts."""

    def __init__(self, path: Path, *, dispatch_lock=None, read_only=False):
        self.path = Path(path).resolve()
        self.dispatch_lock, self.read_only = dispatch_lock, read_only
        self._identity = self._stat()
        self._connection = sqlite3.connect(
            self.path.as_uri() + ("?mode=ro" if read_only else "?mode=rw"),
            uri=True,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            version = schema_version(self._connection)
            if version < 8 or (version not in {8, 9, 10, 11, 12, 13} and not read_only):
                raise PlanRecoveryRequired("Unsupported collection schema")
            self._connection.execute("SELECT * FROM execution_collections LIMIT 0")
            if version >= 10:
                self._connection.execute("SELECT * FROM submission_collections LIMIT 0")
                native_orphan = self._connection.execute(
                    "SELECT 1 FROM submission_collections c "
                    "LEFT JOIN plans p ON p.plan_id=c.plan_id "
                    "LEFT JOIN submission_control s ON s.plan_id=c.plan_id "
                    "WHERE p.plan_id IS NULL OR s.submission_handle_json IS NULL LIMIT 1"
                ).fetchone()
                if native_orphan:
                    raise PlanRecoveryRequired(
                        "Native collection has lost its immutable submission proof"
                    )
            if version >= 12:
                native_orphan = self._connection.execute(
                    "SELECT 1 FROM submission_native_generations n "
                    "LEFT JOIN submission_control s ON s.plan_id=n.plan_id "
                    "LEFT JOIN plans p ON p.plan_id=n.plan_id "
                    "WHERE s.submission_handle_json IS NULL OR p.plan_id IS NULL LIMIT 1"
                ).fetchone()
                if native_orphan:
                    raise PlanRecoveryRequired("Native head has lost submission proof")
            orphan = self._connection.execute(
                "SELECT 1 FROM execution_collections c LEFT JOIN plans p ON p.plan_id=c.plan_id "
                "LEFT JOIN execution_control e ON e.plan_id=c.plan_id "
                "LEFT JOIN execution_attempt_history h ON h.plan_id=c.plan_id AND h.attempt=c.attempt "
                "WHERE p.plan_id IS NULL OR e.plan_id IS NULL OR "
                "(CASE WHEN c.attempt=e.attempt THEN e.handle_json ELSE h.handle_json END) IS NULL LIMIT 1"
            ).fetchone()
            if orphan:
                raise PlanRecoveryRequired(
                    "Collection has lost its immutable execution proof"
                )
            if not read_only:
                self._connection.execute("PRAGMA foreign_keys=ON")
                self._connection.execute("PRAGMA synchronous=FULL")
            if version >= 11:
                self._connection.execute(
                    "SELECT * FROM submission_generation_collections LIMIT 0"
                )
                if self._connection.execute(
                    "SELECT 1 FROM submission_generation_collections c "
                    "LEFT JOIN plans p ON p.plan_id=c.plan_id "
                    "LEFT JOIN submission_control s ON s.plan_id=c.plan_id "
                    "WHERE p.plan_id IS NULL OR s.submission_handle_json IS NULL LIMIT 1"
                ).fetchone():
                    raise PlanRecoveryRequired(
                        "Native generation has lost submission proof"
                    )
        except sqlite3.DatabaseError as error:
            self.close()
            raise PlanRecoveryRequired("Collection schema is damaged") from error
        except BaseException:
            self.close()
            raise

    def _record(self, row):
        if row is None:
            return None
        try:
            values = dict(row)
            plan = Plan.from_json(values.pop("approved_plan"))
            request, owner = (
                values.pop("approved_request"),
                values.pop("approved_owner"),
            )
            expected = values.pop("expected_handle")
            record = CollectionRecord(**values)
            if (
                str(UUID(record.plan_id)) != record.plan_id
                or type(record.attempt) is not int
                or not 1 <= record.attempt <= 2**63 - 1
            ):
                raise ValueError("Invalid collected row identity")
            if (
                not isinstance(record.observed_at, str)
                or len(record.observed_at) > 64
                or not re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
                    record.observed_at,
                )
            ):
                raise ValueError("Invalid collected observation timestamp")
            observed = datetime.fromisoformat(record.observed_at.replace("Z", "+00:00"))
            if observed.utcoffset().total_seconds() != 0:
                raise ValueError("Collected timestamp must be UTC")
            _strict_data(record.handle_json)
            _strict_data(expected)
            handle, result = record.handle, record.result
            if (
                handle.attempt != record.attempt
                or expected is None
                or handle != Handle.from_json(expected)
                or (
                    handle.run_id,
                    handle.request_id,
                    handle.owner,
                    handle.backend,
                    handle.run_dir,
                )
                != (plan.run_id, request, owner, plan.target, plan.run_dir)
                or result.operation != plan.operation
                or result.batch.status == OutcomeStatus.UNKNOWN
                or result.stage_out.status == OutcomeStatus.UNKNOWN
            ):
                raise ValueError("Incomplete or foreign collected receipt")
        except (
            AttributeError,
            KeyError,
            RecursionError,
            TypeError,
            ValueError,
        ) as error:
            raise PlanRecoveryRequired(
                "Stored collection evidence is damaged"
            ) from error
        return record

    def _stat(self):
        try:
            info = self.path.stat()
        except FileNotFoundError:
            raise PlanRecoveryRequired("Collection database missing") from None
        return info.st_dev, info.st_ino

    def _guard(self):
        if self._stat() != self._identity:
            raise PlanRecoveryRequired("Collection database replaced")

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get(self, plan_id, attempt=None):
        self._guard()
        if attempt is not None and (
            type(attempt) is not int or not 1 <= attempt <= 2**63 - 1
        ):
            raise ValueError("Invalid collection attempt")
        row = self._connection.execute(
            _SELECT
            + "WHERE c.plan_id=? "
            + (
                "AND c.attempt=?"
                if attempt is not None
                else "AND c.attempt=(SELECT e.attempt FROM execution_control e WHERE e.plan_id=c.plan_id)"
            ),
            (plan_id, attempt) if attempt is not None else (plan_id,),
        ).fetchone()
        return self._record(row)

    def get_submission(self, plan_id):
        """Read native collection evidence without inventing runtime attempt zero."""
        from wavedl_daemon.storage.submission_collections import get_submission

        return get_submission(self, plan_id)

    def record_submission_collection(self, plan_id, source_handle, result):
        """Persist exact current unbound native outcomes independently of telemetry."""
        from wavedl_daemon.storage.submission_collections import (
            record_submission_collection,
        )

        return record_submission_collection(self, plan_id, source_handle, result)

    def get_submission_generation(self, plan_id, identity):
        from wavedl_daemon.storage.submission_generations import (
            get_submission_generation,
        )

        return get_submission_generation(self, plan_id, identity)

    def get_native_head(self, plan_id):
        """Read verified current native scope independently of stored collections."""
        from wavedl_daemon.storage.native_heads import get_native_head

        self._guard()
        return get_native_head(self._connection, plan_id)

    def record_submission_generation(
        self, plan_id, source_handle, proposal, *, expected_revision
    ):
        """Store caller-verified attemptless evidence at the current source CAS.

        Bound sources use the active execution revision; unbound sources use
        the submission revision. Neither path changes positive runtime evidence.
        """
        from wavedl_daemon.storage.submission_generations import (
            record_submission_generation,
        )

        return record_submission_generation(
            self, plan_id, source_handle, proposal, expected_revision=expected_revision
        )

    def history(self, plan_id, *, limit=100, offset=0):
        self._guard()
        if (
            type(limit) is not int
            or not 1 <= limit <= 1000
            or type(offset) is not int
            or not 0 <= offset <= 2**63 - 1
        ):
            raise ValueError("Invalid collection history page")
        rows = self._connection.execute(
            _SELECT + "WHERE c.plan_id=? ORDER BY c.attempt LIMIT ? OFFSET ?",
            (plan_id, limit, offset),
        )
        records, size = [], 0
        for row in rows:
            size += len(row["handle_json"].encode()) + len(row["result_json"].encode())
            if size > 1024 * 1024:
                raise BudgetExhausted("Collection history page exceeds byte bound")
            records.append(self._record(row))
        return records

    def _insert(self, plan_id, attempt, handle_json, result_json):
        self._connection.execute(
            "INSERT INTO execution_collections VALUES (?,?,?,?,?)",
            (
                plan_id,
                attempt,
                handle_json,
                result_json,
                datetime.now(UTC).isoformat(),
            ),
        )

    def record_collection(
        self, plan_id, source_handle: Handle, result: OperationResult
    ):
        """Fence delayed work by its exact current handle; unknown reads stay diagnostic."""
        if not isinstance(source_handle, Handle) or not isinstance(
            result, OperationResult
        ):
            raise ValueError("Typed backend handle/result required")
        if self.read_only:
            raise PermissionError("Collection storage is read-only")
        require_dispatch(self.dispatch_lock)
        if self.dispatch_lock.path.resolve() != self.path.parent / "dispatch.lock":
            raise DispatchRequired("Collection ownership belongs to another registry")
        self._guard()
        encoded_handle, encoded_result = (
            _json(source_handle.to_dict()),
            _json(asdict(result)),
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._guard()
            row = self._connection.execute(
                "SELECT e.attempt,e.handle_json,p.plan_json,p.request_id,e.owner FROM execution_control e "
                "JOIN plans p ON p.plan_id=e.plan_id WHERE e.plan_id=?",
                (plan_id,),
            ).fetchone()
            if row is None:
                raise KeyError(plan_id)
            if (
                row["attempt"] != source_handle.attempt
                or row["handle_json"] is None
                or Handle.from_json(row["handle_json"]) != source_handle
            ):
                raise StaleExecution(
                    "Collection source is not the current execution handle"
                )
            plan = Plan.from_json(row["plan_json"])
            if (
                source_handle.run_id,
                source_handle.request_id,
                source_handle.owner,
                source_handle.backend,
                source_handle.run_dir,
            ) != (
                plan.run_id,
                row["request_id"],
                row["owner"],
                plan.target,
                plan.run_dir,
            ):
                raise StaleExecution(
                    "Collection handle differs from approved execution"
                )
            if result.operation != plan.operation:
                raise StaleExecution("Collected operation differs from immutable plan")
            if (
                result.batch.status == OutcomeStatus.UNKNOWN
                or result.stage_out.status == OutcomeStatus.UNKNOWN
            ):
                self._connection.execute("COMMIT")
                return None
            existing = self.get(plan_id, source_handle.attempt)
            if existing:
                if (
                    existing.handle_json != encoded_handle
                    or existing.result_json != encoded_result
                ):
                    raise StaleExecution("Conflicting finalized collection")
                self._connection.execute("COMMIT")
                return existing
            self._insert(plan_id, source_handle.attempt, encoded_handle, encoded_result)
            record = self.get(plan_id, source_handle.attempt)
            self._connection.execute("COMMIT")
            return record
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
