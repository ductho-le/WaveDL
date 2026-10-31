# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Authoritative execution receipts; never reconstructed from runtime telemetry.

A committed claim precedes every external submission. Interrupted claims are
unknown acceptance, not permission to submit again. Intent remains in plan_control.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.local_attempt import LocalStartupBinding
from wavedl.core.scheduler_attempt import SchedulerBinding, SchedulerIdentity
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    OperationOutcome,
    OutcomeStatus,
    Plan,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version
from wavedl_daemon.storage.submissions import validate_writer_receipt


if TYPE_CHECKING:
    from collections.abc import Iterator

    from wavedl_daemon.storage.ownership import DispatchLock


class ExecutionState(StrEnum):
    PENDING = "pending"
    SUBMITTING = "submitting"
    UNKNOWN = "unknown"
    SUBMITTED = "submitted"
    REJECTED = "rejected"
    LOCAL_LOST = "local_lost"


class StaleExecution(ValueError):
    """Evidence or a revision does not identify this execution."""


def _create_tables(connection: sqlite3.Connection) -> None:
    if schema_version(connection) != 2:
        raise MigrationFailed("Execution control requires the version 2 registry")
    connection.execute(
        "CREATE TABLE execution_control ("
        "plan_id TEXT PRIMARY KEY REFERENCES plans(plan_id), "
        "attempt INTEGER NOT NULL CHECK(attempt>0), owner TEXT NOT NULL, "
        "state TEXT NOT NULL CHECK(state IN "
        "('pending','submitting','unknown','submitted','rejected','local_lost')), "
        "handle_json TEXT, observation_json TEXT, latest_observation_json TEXT, "
        "cancel_delivered INTEGER NOT NULL DEFAULT 0 CHECK(cancel_delivered IN (0,1)), "
        "revision INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, "
        "detail TEXT NOT NULL DEFAULT '')"
    )
    connection.execute(
        "CREATE TRIGGER execution_identity_immutable BEFORE UPDATE OF "
        "plan_id, attempt, owner ON execution_control "
        "BEGIN SELECT RAISE(ABORT, 'execution identity is immutable'); END"
    )


EXECUTION_MIGRATION = Migration("0003_execution_control", 3, _create_tables)
MAX_JSON_BYTES = 256 * 1024


def _json(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Execution evidence exceeds JSON size bound")
    return encoded


def _observation(value: str | None) -> BackendObservation | None:
    if value is None:
        return None
    data = json.loads(value)
    data["status"] = ObservationStatus(data["status"])
    for key in ("batch", "stage_out"):
        if data[key] is not None:
            data[key]["status"] = OutcomeStatus(data[key]["status"])
            data[key] = OperationOutcome(**data[key])
    return BackendObservation(**data)


def _manifest_sha256(plan: Plan) -> str:
    """Match the immutable runtime manifest's canonical bytes, including newline."""
    encoded = (
        json.dumps(
            plan.manifest.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def _initial_handle(
    receipt: SubmissionHandle,
    binding: SchedulerBinding | LocalStartupBinding,
    plan: Plan,
) -> Handle:
    """Bind current verified startup while retaining the immutable native ACK."""
    handle = receipt.bind_attempt(binding.attempt)
    if receipt.backend != "slurm":
        return handle
    if not isinstance(binding, SchedulerBinding):
        raise StaleExecution("Slurm activation requires scheduler binding")
    raw = _writer_receipt(receipt, plan)
    if (
        raw["restart_count"] != 0
        or receipt.job_id != binding.job_id + ";" + binding.cluster
        or raw["cluster"] != binding.cluster
    ):
        raise StaleExecution("Initial scheduler binding differs from native generation")
    if binding.restart_count > 0 and dict(plan.resources).get("requeue") is not True:
        raise StaleExecution("Immutable plan did not approve scheduler requeue")
    return replace(
        handle, raw_json=_json(raw | {"restart_count": binding.restart_count})
    )


def _binding_identity(binding: SchedulerBinding) -> SchedulerIdentity:
    return SchedulerIdentity(
        binding.run_id,
        binding.request_id,
        binding.cluster,
        binding.job_id,
        binding.owner,
        binding.restart_count,
    )


def _initial_native_observation(observation, binding):
    """Copy only this native generation; schema-9 legacy evidence is ACK0 only."""
    if observation is None:
        return None
    if isinstance(binding, SchedulerBinding):
        if observation.native_identity is None:
            # Older writer registries and local/mock backends omitted identity.
            # Their scheduler evidence can certify only the original generation.
            if binding.restart_count != 0:
                return None
        elif observation.native_identity != _binding_identity(binding):
            return None
    return observation.bind_attempt(binding.attempt)


def _validate_initial_native_generation(receipt, plan, binding, observation):
    if not isinstance(binding, SchedulerBinding) or observation is None:
        return
    identity = observation.native_identity
    if identity is None:
        return
    from wavedl_daemon.storage.submission_generations import receipt_identity

    if identity != receipt_identity(receipt, plan, identity.restart_count):
        raise ValueError("Native observation differs from immutable submission")
    if binding.restart_count < identity.restart_count:
        raise ValueError("Initial scheduler binding is behind known native generation")


def _writer_receipt(receipt: SubmissionHandle, plan: Plan) -> dict:
    """Reject legacy-looking scheduler ACKs for writer-mode plans."""
    try:
        return validate_writer_receipt(receipt, plan)
    except ValueError as error:
        raise StaleExecution(str(error)) from error


@dataclass(frozen=True)
class ExecutionRecord:
    plan_id: str
    request_id: str
    run_id: str
    backend: str
    attempt: int | None
    owner: str
    run_dir: str
    state: ExecutionState
    handle: Handle | None
    observation: BackendObservation | None
    latest_observation: BackendObservation | None
    submission_requested: bool
    cancellation_requested: bool
    cancel_delivered: bool
    revision: int
    updated_at: str
    detail: str
    submission_handle: SubmissionHandle | None = None
    submission_observation: SubmissionObservation | None = None
    latest_submission_observation: SubmissionObservation | None = None
    native_identity: SchedulerIdentity | None = None
    native_observation: SubmissionObservation | None = None
    latest_native_observation: SubmissionObservation | None = None


def native_runtime_pending(record) -> bool:
    """A verified scheduler generation has no matching positive runtime yet."""
    handle = getattr(record, "handle", None)
    identity = getattr(record, "native_identity", None)
    if (
        handle is None
        or not isinstance(identity, SchedulerIdentity)
        or getattr(record, "submission_handle", None) is None
    ):
        return False
    count = json.loads(handle.raw_json).get("restart_count")
    return type(count) is int and identity.restart_count > count


_SELECT = (
    "SELECT e.plan_id, e.attempt, e.owner, e.state, e.handle_json, "
    "e.observation_json, e.latest_observation_json, e.cancel_delivered, "
    "e.revision, e.updated_at, e.detail, p.request_id, p.plan_json, c.submission_requested, "
    "c.cancellation_requested FROM execution_control e JOIN plans p "
    "ON p.plan_id=e.plan_id JOIN plan_control c ON c.plan_id=e.plan_id "
)


class ExecutionStore:
    """Use an existing migrated database under its matching live dispatch lock."""

    def __init__(
        self,
        path: Path,
        *,
        dispatch_lock: DispatchLock | None = None,
        read_only: bool = False,
    ):
        self.path = Path(path).resolve()
        self.dispatch_lock = dispatch_lock
        self.read_only = read_only
        self._identity = self._stat()
        self._connection = sqlite3.connect(
            self.path.as_uri() + ("?mode=ro" if read_only else "?mode=rw"),
            uri=True,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            version = schema_version(self._connection)
            if version < 3 or (
                version not in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13} and not read_only
            ):
                raise PlanRecoveryRequired(f"Unsupported execution schema {version}")
            self._connection.execute(_SELECT + "LIMIT 0")
            if version in {7, 8, 9, 10, 11, 12, 13}:
                self._connection.execute("SELECT * FROM requeue_bindings LIMIT 0")
                self._connection.execute(
                    "SELECT * FROM execution_attempt_history LIMIT 0"
                )
            self._writer_schema = (
                self._connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='submission_control'"
                ).fetchone()
                is not None
            )
            if version in {9, 10, 11, 12, 13} and not self._writer_schema:
                raise PlanRecoveryRequired("Writer submission schema is missing")
            if self._writer_schema:
                self._connection.execute("SELECT * FROM submission_control LIMIT 0")
                if version >= 13:
                    self._connection.execute(
                        "SELECT * FROM submission_native_lifecycle LIMIT 0"
                    )
                if version >= 12:
                    native_orphan = self._connection.execute(
                        "SELECT 1 FROM submission_native_generations n "
                        "LEFT JOIN submission_control s ON s.plan_id=n.plan_id "
                        "LEFT JOIN plans p ON p.plan_id=n.plan_id "
                        "WHERE s.submission_handle_json IS NULL OR p.plan_id IS NULL LIMIT 1"
                    ).fetchone()
                    if native_orphan:
                        raise PlanRecoveryRequired(
                            "Native head has lost submission proof"
                        )
                incomplete_submission = self._connection.execute(
                    "SELECT 1 FROM submission_control s LEFT JOIN plans p ON p.plan_id=s.plan_id "
                    "LEFT JOIN plan_control c ON c.plan_id=s.plan_id WHERE p.plan_id IS NULL OR c.plan_id IS NULL LIMIT 1"
                ).fetchone()
                if incomplete_submission:
                    raise PlanRecoveryRequired(
                        "Submission has lost its immutable plan or intent"
                    )
            incomplete = self._connection.execute(
                "SELECT 1 FROM execution_control e LEFT JOIN plans p "
                "ON p.plan_id=e.plan_id LEFT JOIN plan_control c ON c.plan_id=e.plan_id "
                "WHERE p.plan_id IS NULL OR c.plan_id IS NULL LIMIT 1"
            ).fetchone()
            if incomplete:
                raise PlanRecoveryRequired(
                    "Execution has lost its immutable plan or intent"
                )
            self._interrupted = {
                row[0]
                for row in self._connection.execute(
                    "SELECT plan_id FROM execution_control WHERE state='submitting'"
                )
            }
            if self._writer_schema:
                self._interrupted.update(
                    row[0]
                    for row in self._connection.execute(
                        "SELECT plan_id FROM submission_control WHERE state='submitting'"
                    )
                )
            if not read_only:
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA foreign_keys=ON")
        except sqlite3.DatabaseError as exc:
            self.close()
            raise PlanRecoveryRequired(
                "Execution schema is damaged or incompatible"
            ) from exc
        except BaseException:
            self.close()
            raise

    def _stat(self) -> tuple[int, int]:
        try:
            stat = self.path.stat()
        except FileNotFoundError as exc:
            raise PlanRecoveryRequired(
                "Execution database missing; restore backup"
            ) from exc
        return stat.st_dev, stat.st_ino

    def _guard(self) -> None:
        if self._stat() != self._identity:
            raise PlanRecoveryRequired("Execution database replaced; reopen recovery")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ExecutionStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _record(self, row: sqlite3.Row | None) -> ExecutionRecord | None:
        if row is None:
            return None
        values = dict(row)
        plan = Plan.from_json(values.pop("plan_json"))
        state = ExecutionState(values.pop("state"))
        if (
            state == ExecutionState.SUBMITTING
            and values["plan_id"] in self._interrupted
        ):
            state = ExecutionState.UNKNOWN
        handle_json = values.pop("handle_json")
        observation_json = values.pop("observation_json")
        latest_json = values.pop("latest_observation_json")
        for name in (
            "submission_requested",
            "cancellation_requested",
            "cancel_delivered",
        ):
            values[name] = bool(values[name])
        return ExecutionRecord(
            **values,
            run_id=plan.run_id,
            backend=plan.target,
            run_dir=plan.run_dir,
            state=state,
            handle=Handle.from_json(handle_json) if handle_json else None,
            observation=_observation(observation_json),
            latest_observation=_observation(latest_json),
        )

    def get(self, plan_id: str) -> ExecutionRecord | None:
        self._guard()
        active = self._record(
            self._connection.execute(
                _SELECT + "WHERE e.plan_id=?", (plan_id,)
            ).fetchone()
        )
        source = None
        if self._writer_schema:
            source = self._connection.execute(
                "SELECT s.*,p.request_id,p.plan_json,c.submission_requested,c.cancellation_requested "
                "FROM submission_control s JOIN plans p ON p.plan_id=s.plan_id "
                "JOIN plan_control c ON c.plan_id=s.plan_id WHERE s.plan_id=?",
                (plan_id,),
            ).fetchone()
        if source is None:
            if active is not None and self._writer_schema:
                plan = Plan.from_json(
                    self._connection.execute(
                        "SELECT plan_json FROM plans WHERE plan_id=?", (plan_id,)
                    ).fetchone()[0]
                )
                if plan.allocation_mode == "writer":
                    raise PlanRecoveryRequired(
                        "Active writer execution lost native submission proof"
                    )
            return active
        data = dict(source)
        raw_receipt, raw_observation, raw_latest = (
            data.pop("submission_handle_json"),
            data.pop("submission_observation_json"),
            data.pop("latest_submission_observation_json"),
        )
        receipt = SubmissionHandle.from_json(raw_receipt) if raw_receipt else None
        observation = (
            SubmissionObservation.from_json(raw_observation)
            if raw_observation
            else None
        )
        latest = SubmissionObservation.from_json(raw_latest) if raw_latest else None
        source_plan = Plan.from_json(data["plan_json"])
        from wavedl_daemon.storage.native_heads import get_native_head

        native_identity = get_native_head(self._connection, plan_id)
        from wavedl_daemon.storage.native_lifecycle import read_native_lifecycle

        native_observation, latest_native_observation = read_native_lifecycle(
            self._connection, plan_id, native_identity
        )
        if receipt is not None:
            try:
                _writer_receipt(receipt, source_plan)
            except StaleExecution as error:
                raise PlanRecoveryRequired(
                    "Stored writer native receipt is damaged"
                ) from error
        if source_plan.allocation_mode != "writer" or (
            receipt is not None
            and (
                receipt.run_id,
                receipt.request_id,
                receipt.owner,
                receipt.backend,
                receipt.run_dir,
            )
            != (
                source_plan.run_id,
                data["request_id"],
                data["owner"],
                source_plan.target,
                source_plan.run_dir,
            )
        ):
            raise PlanRecoveryRequired(
                "Stored submission identity differs from immutable writer plan"
            )
        if active is not None:
            proof = self._connection.execute(
                "SELECT * FROM initial_bindings WHERE plan_id=?", (plan_id,)
            ).fetchone()
            if (
                proof is None
                or receipt is None
                or proof["native_receipt_json"] != _json(receipt.to_dict())
                or proof["previous_revision"] != data["revision"]
                or active.owner != receipt.owner
            ):
                raise PlanRecoveryRequired(
                    "Active writer execution has lost initial binding evidence"
                )
            if active.attempt == proof["attempt"] and (
                active.handle is None
                or active.handle.to_json() != proof["runtime_handle_json"]
            ):
                raise PlanRecoveryRequired(
                    "Active initial execution handle differs from immutable binding"
                )
            try:
                typed = (
                    SchedulerBinding
                    if receipt.backend == "slurm"
                    else LocalStartupBinding
                )(**json.loads(proof["binding_json"]))
                if (
                    (typed.run_id, typed.request_id, typed.owner, typed.attempt)
                    != (
                        receipt.run_id,
                        receipt.request_id,
                        receipt.owner,
                        proof["attempt"],
                    )
                    or _initial_handle(receipt, typed, source_plan).to_json()
                    != proof["runtime_handle_json"]
                    or active.attempt < proof["attempt"]
                ):
                    raise ValueError(
                        "Initial proof identity differs from native receipt"
                    )
                if isinstance(typed, SchedulerBinding):
                    _validate_initial_native_generation(
                        receipt, source_plan, typed, observation
                    )
                    raw = json.loads(receipt.raw_json)
                    if (
                        raw["restart_count"] != 0
                        or typed.job_id + ";" + typed.cluster != receipt.job_id
                    ):
                        raise ValueError(
                            "Initial scheduler proof differs from native receipt"
                        )
                elif (
                    typed.supervisor_id != receipt.supervisor_id
                    or typed.run_directory != receipt.run_dir
                    or typed.plan_sha256
                    != hashlib.sha256(source_plan.to_json().encode("utf-8")).hexdigest()
                    or typed.manifest_sha256 != _manifest_sha256(source_plan)
                ):
                    raise ValueError("Initial local proof differs from native receipt")
            except (KeyError, TypeError, ValueError) as error:
                raise PlanRecoveryRequired(
                    "Stored initial writer binding is damaged"
                ) from error
            return replace(
                active,
                submission_handle=receipt,
                submission_observation=observation,
                latest_submission_observation=latest,
                native_identity=native_identity,
                native_observation=native_observation,
                latest_native_observation=latest_native_observation,
            )
        plan = Plan.from_json(data.pop("plan_json"))
        state = ExecutionState(data.pop("state"))
        if state == ExecutionState.SUBMITTING and plan_id in self._interrupted:
            state = ExecutionState.UNKNOWN
        for field in (
            "submission_requested",
            "cancellation_requested",
            "cancel_delivered",
        ):
            data[field] = bool(data[field])
        return ExecutionRecord(
            **data,
            run_id=plan.run_id,
            backend=plan.target,
            run_dir=plan.run_dir,
            attempt=None,
            state=state,
            handle=None,
            observation=None,
            latest_observation=None,
            submission_handle=receipt,
            submission_observation=observation,
            latest_submission_observation=latest,
            native_identity=native_identity,
            native_observation=native_observation,
            latest_native_observation=latest_native_observation,
        )

    def list_pending(self, limit: int = 100, offset: int = 0) -> list[ExecutionRecord]:
        if (
            type(limit) is not int
            or not 1 <= limit <= 1000
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("Invalid execution page bounds")
        self._guard()
        if self._writer_schema:
            rows = self._connection.execute(
                "SELECT p.plan_id FROM plans p WHERE EXISTS (SELECT 1 FROM execution_control e WHERE e.plan_id=p.plan_id) "
                "OR EXISTS (SELECT 1 FROM submission_control s WHERE s.plan_id=p.plan_id) "
                "ORDER BY p.created_at,p.plan_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [self.get(row[0]) for row in rows]
        return [
            self._record(row)
            for row in self._connection.execute(
                _SELECT + "ORDER BY p.created_at, e.plan_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
        ]

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        if self.read_only:
            raise PermissionError("ExecutionStore is in read-only recovery mode")
        require_dispatch(self.dispatch_lock)
        if self.dispatch_lock.path.resolve() != self.path.parent / "dispatch.lock":
            raise DispatchRequired("dispatch ownership is for a different registry")
        self._guard()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def ensure(
        self, plan_id: str, attempt: int | None = None, *, owner: str | None = None
    ) -> ExecutionRecord:
        if attempt is not None and (type(attempt) is not int or attempt < 1):
            raise ValueError("attempt must be a positive reserved integer")
        owner = getpass.getuser() if owner is None else owner
        if not isinstance(owner, str) or not owner or "\0" in owner:
            raise ValueError("Invalid execution owner")
        with self._transaction():
            row = self._connection.execute(
                "SELECT plan_json FROM plans WHERE plan_id=?", (plan_id,)
            ).fetchone()
            if row is None:
                raise KeyError(plan_id)
            plan = Plan.from_json(row[0])
            if (plan.allocation_mode == "writer") != (attempt is None):
                raise StaleExecution(
                    "Allocation mode differs from reserved/writer attempt"
                )
            if attempt is None and not self._writer_schema:
                raise PlanRecoveryRequired("Writer submissions require schema 9")
            existing = self.get(plan_id)
            if existing is not None:
                if (
                    attempt is not None and existing.attempt != attempt
                ) or existing.owner != owner:
                    raise StaleExecution("Attempt/owner differs from durable execution")
                return existing
            if attempt is None:
                self._connection.execute(
                    "INSERT INTO submission_control(plan_id,owner,state,updated_at) VALUES (?,?,'pending',?)",
                    (plan_id, owner, datetime.now(UTC).isoformat()),
                )
                return self.get(plan_id)
            self._connection.execute(
                "INSERT INTO execution_control(plan_id,attempt,owner,state,updated_at) VALUES (?,?,?,'pending',?)",
                (plan_id, attempt, owner, datetime.now(UTC).isoformat()),
            )
            return self.get(plan_id)

    def claim_submission(self, plan_id: str) -> tuple[ExecutionRecord, bool]:
        with self._transaction():
            record = self._required(plan_id)
            if (
                record.state != ExecutionState.PENDING
                or not record.submission_requested
                or record.cancellation_requested
            ):
                return record, False
            self._update(record, state=ExecutionState.SUBMITTING)
            return self._required(plan_id), True

    def _required(
        self, plan_id: str, expected_revision: int | None = None
    ) -> ExecutionRecord:
        record = self.get(plan_id)
        if record is None:
            raise KeyError(plan_id)
        if expected_revision is not None and record.revision != expected_revision:
            raise StaleExecution("Execution revision changed")
        return record

    def _update(self, record: ExecutionRecord, **values) -> None:
        values["updated_at"] = datetime.now(UTC).isoformat()
        assignments = ",".join(f"{key}=?" for key in values)
        table = "submission_control" if record.attempt is None else "execution_control"
        result = self._connection.execute(
            f"UPDATE {table} SET {assignments},revision=revision+1 WHERE plan_id=? AND revision=?",
            (*values.values(), record.plan_id, record.revision),
        )
        if result.rowcount != 1:
            raise StaleExecution("Execution revision changed")

    def _transition(
        self,
        plan_id: str,
        state: ExecutionState,
        reason: str,
        expected_revision: int | None,
    ) -> ExecutionRecord:
        if not isinstance(reason, str) or len(reason.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("Invalid execution detail")
        with self._transaction():
            record = self._required(plan_id, expected_revision)
            allowed = {ExecutionState.SUBMITTING, ExecutionState.UNKNOWN}
            if state == ExecutionState.LOCAL_LOST:
                allowed |= {ExecutionState.SUBMITTED}
                if record.backend != "local":
                    raise StaleExecution("Only local executions lose supervision")
            if record.state not in allowed and record.state != state:
                raise StaleExecution("Invalid execution lifecycle transition")
            self._update(record, state=state, detail=reason)
            return self._required(plan_id)

    def mark_unknown(
        self, plan_id: str, reason: str, *, expected_revision: int | None = None
    ) -> ExecutionRecord:
        return self._transition(
            plan_id, ExecutionState.UNKNOWN, reason, expected_revision
        )

    def mark_rejected(
        self, plan_id: str, reason: str, *, expected_revision: int | None = None
    ) -> ExecutionRecord:
        return self._transition(
            plan_id, ExecutionState.REJECTED, reason, expected_revision
        )

    def mark_local_lost(
        self, plan_id: str, reason: str, *, expected_revision: int | None = None
    ) -> ExecutionRecord:
        return self._transition(
            plan_id, ExecutionState.LOCAL_LOST, reason, expected_revision
        )

    def record_handle(
        self, plan_id: str, handle: Handle, *, expected_revision: int | None = None
    ) -> ExecutionRecord:
        if isinstance(handle, SubmissionHandle):
            return self.attach_submission(
                plan_id, handle, expected_revision=expected_revision
            )
        encoded = _json(handle.to_dict())
        with self._transaction():
            record = self._required(plan_id, expected_revision)
            if (
                handle.backend,
                handle.run_id,
                handle.request_id,
                handle.attempt,
                handle.owner,
                handle.run_dir,
            ) != (
                record.backend,
                record.run_id,
                record.request_id,
                record.attempt,
                record.owner,
                record.run_dir,
            ):
                raise StaleExecution("Handle identity differs from durable execution")
            if record.handle is not None:
                if record.handle != handle:
                    raise StaleExecution("Cannot replace a durable receipt")
                return record
            if record.state not in {ExecutionState.SUBMITTING, ExecutionState.UNKNOWN}:
                raise StaleExecution(
                    "Cannot attach a receipt before claim or after rejection"
                )
            self._update(record, state=ExecutionState.SUBMITTED, handle_json=encoded)
            return self._required(plan_id)

    def attach_submission(
        self, plan_id, receipt: SubmissionHandle, *, expected_revision: int
    ):
        """Attach native acceptance to an unbound writer submission, never attempt 0."""
        if not isinstance(receipt, SubmissionHandle):
            raise ValueError("Typed native submission receipt required")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Explicit submission revision required")
        encoded = _json(receipt.to_dict())
        with self._transaction():
            record = self._required(plan_id)
            if record.attempt is not None:
                raise StaleExecution("Runtime attempt is already activated")
            if record.submission_handle is not None:
                if record.submission_handle == receipt and expected_revision in {
                    record.revision,
                    record.revision - 1,
                }:
                    return record
                raise StaleExecution("Cannot replace native submission receipt")
            if record.revision != expected_revision:
                raise StaleExecution("Submission revision changed")
            if (
                receipt.backend,
                receipt.run_id,
                receipt.request_id,
                receipt.owner,
                receipt.run_dir,
            ) != (
                record.backend,
                record.run_id,
                record.request_id,
                record.owner,
                record.run_dir,
            ):
                raise StaleExecution("Native submission receipt identity mismatch")
            if record.state not in {ExecutionState.SUBMITTING, ExecutionState.UNKNOWN}:
                raise StaleExecution("Native receipt requires prior committed claim")
            approved_plan = Plan.from_json(
                self._connection.execute(
                    "SELECT plan_json FROM plans WHERE plan_id=?", (plan_id,)
                ).fetchone()[0]
            )
            _writer_receipt(receipt, approved_plan)
            self._update(
                record, state=ExecutionState.SUBMITTED, submission_handle_json=encoded
            )
            return self._required(plan_id)

    def record_native_generation(
        self, plan_id, identity: SchedulerIdentity, *, expected_revision: int
    ):
        """Retain caller-verified native identity without fabricating a runtime.

        A requeued job can prove a newer generation before publishing any
        startup binding. Its durable head must survive a daemon restart, while
        the preceding positive runtime remains independently identifiable.
        """
        from wavedl_daemon.storage.native_heads import record_native_observation

        if not isinstance(identity, SchedulerIdentity):
            raise ValueError("Typed verified native identity required")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Explicit native source revision required")
        with self._transaction():
            record = self._required(plan_id, expected_revision)
            if schema_version(self._connection) < 12:
                raise PlanRecoveryRequired(
                    "Durable native generation heads require schema 12"
                )
            if record.submission_handle is None or record.backend != "slurm":
                raise StaleExecution("Native head requires a writer scheduler receipt")
            source_revision = self._connection.execute(
                "SELECT revision FROM submission_control WHERE plan_id=?", (plan_id,)
            ).fetchone()[0]
            record_native_observation(
                self._connection,
                plan_id,
                SubmissionObservation(
                    ObservationStatus.ERROR,
                    detail="Verified native generation; runtime binding unavailable",
                    native_identity=identity,
                ),
                source_revision,
            )
            if record.native_identity != identity:
                self._update(
                    record,
                    cancel_delivered=0,
                    detail="New native generation awaits verified runtime binding",
                )
            return self._required(plan_id)

    def observe_native_generation(
        self, plan_id, observation: SubmissionObservation, *, expected_revision: int
    ):
        """CAS native lifecycle independently of any current positive runtime."""
        from wavedl_daemon.storage.native_heads import record_native_observation
        from wavedl_daemon.storage.native_lifecycle import write_native_lifecycle

        if not isinstance(observation, SubmissionObservation):
            raise ValueError("Typed native lifecycle observation required")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Explicit native source revision required")
        with self._transaction():
            record = self._required(plan_id, expected_revision)
            if schema_version(self._connection) < 13:
                raise PlanRecoveryRequired("Native lifecycle requires schema 13")
            if record.backend != "slurm" or record.submission_handle is None:
                raise StaleExecution("Native lifecycle requires scheduler ACK0")
            identity = observation.native_identity
            if identity is None:
                if record.native_identity is None or observation.status not in {
                    ObservationStatus.ERROR,
                    ObservationStatus.ABSENT,
                }:
                    raise StaleExecution(
                        "Unknown diagnostic requires current native head"
                    )
                identity = record.native_identity
            else:
                source_revision = self._connection.execute(
                    "SELECT revision FROM submission_control WHERE plan_id=?",
                    (plan_id,),
                ).fetchone()[0]
                record_native_observation(
                    self._connection, plan_id, observation, source_revision
                )
            write_native_lifecycle(
                self._connection, plan_id, identity, observation, record.revision
            )
            values = {"detail": "Native lifecycle observation recorded"}
            if identity != record.native_identity:
                values["cancel_delivered"] = 0
            self._update(record, **values)
            return self._required(plan_id)

    def observe_submission(
        self, plan_id, observation: SubmissionObservation, *, expected_revision=None
    ):
        if not isinstance(observation, SubmissionObservation):
            raise ValueError("Typed unbound submission observation required")
        with self._transaction():
            record = self._required(plan_id, expected_revision)
            if record.attempt is not None or record.submission_handle is None:
                raise StaleExecution("Observation needs current unbound native receipt")
            values = {
                "latest_submission_observation_json": _json(observation.to_dict())
            }
            previous = record.submission_observation
            identity = observation.native_identity
            from wavedl_daemon.storage.native_heads import record_native_observation

            record_native_observation(
                self._connection, plan_id, observation, record.revision
            )
            old_identity = previous.native_identity if previous is not None else None
            if identity is not None:
                from wavedl_daemon.storage.submission_generations import (
                    receipt_identity,
                )

                plan = Plan.from_json(
                    self._connection.execute(
                        "SELECT plan_json FROM plans WHERE plan_id=?", (plan_id,)
                    ).fetchone()[0]
                )
                try:
                    if identity != receipt_identity(
                        record.submission_handle, plan, identity.restart_count
                    ):
                        raise ValueError("Foreign native submission identity")
                except ValueError as error:
                    raise StaleExecution(str(error)) from error
                if (
                    old_identity is not None
                    and identity.restart_count < old_identity.restart_count
                ):
                    raise StaleExecution(
                        "Cannot replace current native generation with late evidence"
                    )
            if observation.status not in {
                ObservationStatus.ERROR,
                ObservationStatus.ABSENT,
            }:
                if old_identity is not None and (
                    identity is None
                    or identity.restart_count < old_identity.restart_count
                ):
                    raise StaleExecution(
                        "Cannot replace current native generation with late evidence"
                    )
                advanced = identity is not None and identity.restart_count > (
                    old_identity.restart_count if old_identity is not None else 0
                )
                if (
                    previous is not None
                    and previous.status == ObservationStatus.TERMINAL
                    and observation != previous
                    and not advanced
                ):
                    raise StaleExecution(
                        "Cannot contradict terminal submission evidence"
                    )
                values["submission_observation_json"] = _json(observation.to_dict())
            if schema_version(self._connection) >= 13:
                from wavedl_daemon.storage.native_lifecycle import (
                    write_native_lifecycle,
                )

                head = observation.native_identity or record.native_identity
                if head is not None:
                    write_native_lifecycle(
                        self._connection, plan_id, head, observation, record.revision
                    )
                if head is not None and head != record.native_identity:
                    values["cancel_delivered"] = 0
            self._update(record, **values)
            return self._required(plan_id)

    def activate_initial(self, plan_id, binding, *, expected_revision: int):
        """CAS a caller-verified typed startup binding into the first positive attempt."""
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Explicit activation revision required")
        if not isinstance(binding, (SchedulerBinding, LocalStartupBinding)):
            raise ValueError("Typed verified startup binding required")
        encoded_binding = _json(asdict(binding))
        with self._transaction():
            record = self._required(plan_id)
            existing = self._connection.execute(
                "SELECT * FROM initial_bindings WHERE plan_id=?", (plan_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["binding_json"] == encoded_binding
                    and existing["previous_revision"] == expected_revision
                    and record.attempt == existing["attempt"]
                    and record.handle is not None
                    and record.handle.to_json() == existing["runtime_handle_json"]
                ):
                    return record
                raise StaleExecution("Conflicting initial runtime activation")
            if (
                record.revision != expected_revision
                or record.attempt is not None
                or record.submission_handle is None
            ):
                raise StaleExecution(
                    "Initial binding is not for current unbound submission"
                )
            receipt = record.submission_handle
            plan = Plan.from_json(
                self._connection.execute(
                    "SELECT plan_json FROM plans WHERE plan_id=?", (plan_id,)
                ).fetchone()[0]
            )
            if (binding.run_id, binding.request_id, binding.owner) != (
                record.run_id,
                record.request_id,
                record.owner,
            ):
                raise StaleExecution("Initial binding has foreign submission identity")
            if receipt.backend == "slurm":
                if not isinstance(binding, SchedulerBinding):
                    raise StaleExecution("Slurm activation requires scheduler binding")
            elif not isinstance(binding, LocalStartupBinding):
                raise StaleExecution("Local activation requires local startup binding")
            else:
                if (
                    binding.run_directory != receipt.run_dir
                    or binding.supervisor_id != receipt.supervisor_id
                ):
                    raise StaleExecution(
                        "Local startup binding differs from native supervisor"
                    )
                if binding.plan_sha256 != hashlib.sha256(
                    plan.to_json().encode("utf-8")
                ).hexdigest() or binding.manifest_sha256 != _manifest_sha256(plan):
                    raise StaleExecution(
                        "Local startup differs from immutable approved request digests"
                    )
            handle = _initial_handle(receipt, binding, plan)
            if (
                isinstance(binding, SchedulerBinding)
                and record.native_identity is not None
                and binding.restart_count < record.native_identity.restart_count
            ):
                raise StaleExecution(
                    "Initial scheduler binding is behind durable native head"
                )
            try:
                _validate_initial_native_generation(
                    receipt, plan, binding, record.submission_observation
                )
            except ValueError as error:
                raise StaleExecution(str(error)) from error
            native_observation = _initial_native_observation(
                record.submission_observation, binding
            )
            latest_native = _initial_native_observation(
                record.latest_submission_observation, binding
            )
            encoded_handle = _json(handle.to_dict())
            if isinstance(binding, SchedulerBinding):
                from wavedl_daemon.storage.native_heads import record_native_observation

                record_native_observation(
                    self._connection,
                    plan_id,
                    SubmissionObservation(
                        ObservationStatus.ERROR,
                        detail="Startup binding proves native identity; lifecycle unconfirmed",
                        native_identity=_binding_identity(binding),
                    ),
                    record.revision,
                )
            self._connection.execute(
                "INSERT INTO initial_bindings VALUES (?,?,?,?,?,?,?)",
                (
                    plan_id,
                    binding.attempt,
                    record.revision,
                    encoded_binding,
                    _json(receipt.to_dict()),
                    encoded_handle,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._connection.execute(
                "INSERT INTO execution_control(plan_id,attempt,owner,state,handle_json,observation_json,latest_observation_json,cancel_delivered,revision,updated_at,detail) VALUES (?,?,?,'submitted',?,?,?,?,?,?,?)",
                (
                    plan_id,
                    binding.attempt,
                    record.owner,
                    encoded_handle,
                    _json(asdict(native_observation)) if native_observation else None,
                    _json(asdict(latest_native)) if latest_native else None,
                    int(record.cancel_delivered),
                    record.revision + 1,
                    datetime.now(UTC).isoformat(),
                    "Verified initial writer startup activated",
                ),
            )
            return self._required(plan_id)

    def requeue_binding(self, plan_id: str, attempt: int) -> SchedulerBinding | None:
        """Read immutable activated evidence; journals cannot activate it."""
        self._guard()
        if type(attempt) is not int or not 1 <= attempt <= 2**63 - 1:
            raise ValueError("Invalid scheduler attempt")
        row = self._connection.execute(
            "SELECT binding_json FROM requeue_bindings WHERE plan_id=? AND attempt=?",
            (plan_id, attempt),
        ).fetchone()
        return SchedulerBinding(**json.loads(row[0])) if row else None

    def history(self, plan_id: str, *, limit=100, offset=0) -> list[ExecutionRecord]:
        """Bounded snapshots retain the prior observations and intent at activation."""
        self._guard()
        if (
            type(limit) is not int
            or not 1 <= limit <= 1000
            or type(offset) is not int
            or not 0 <= offset <= 2**63 - 1
        ):
            raise ValueError("Invalid execution history page")
        rows = self._connection.execute(
            "SELECT record_json FROM execution_attempt_history WHERE plan_id=? "
            "ORDER BY attempt LIMIT ? OFFSET ?",
            (plan_id, limit, offset),
        )
        history, size = [], 0
        for row in rows:
            size += len(row[0].encode("utf-8"))
            if size > 1024 * 1024:
                raise BudgetExhausted("Execution history page exceeds byte bound")
            data = json.loads(row[0])
            if data.get("native_identity") is not None:
                data["native_identity"] = SchedulerIdentity(**data["native_identity"])
            data["state"] = ExecutionState(data["state"])
            data["handle"] = (
                Handle.from_json(_json(data["handle"])) if data["handle"] else None
            )
            for name in ("observation", "latest_observation"):
                data[name] = _observation(_json(data[name])) if data[name] else None
            for name in (
                "submission_observation",
                "latest_submission_observation",
                "native_observation",
                "latest_native_observation",
            ):
                if data.get(name) is not None:
                    data[name] = SubmissionObservation.from_dict(data[name])
            if data.get("submission_handle") is not None:
                data["submission_handle"] = SubmissionHandle.from_dict(
                    data["submission_handle"]
                )
            history.append(ExecutionRecord(**data))
        return history

    def history_count(self, plan_id: str) -> int:
        """Count immutable snapshots on the owner thread for bounded pagination."""
        self._guard()
        row = self._connection.execute(
            "SELECT COUNT(*) FROM execution_attempt_history WHERE plan_id=?", (plan_id,)
        ).fetchone()
        return row[0]

    def _archive_attempt(self, record: ExecutionRecord) -> None:
        self._connection.execute(
            "INSERT INTO execution_attempt_history VALUES (?,?,?,?,?)",
            (
                record.plan_id,
                record.attempt,
                record.revision,
                record.handle.to_json(),
                _json(asdict(record)),
            ),
        )

    def activate_requeue(
        self,
        plan_id: str,
        binding: SchedulerBinding,
        new_handle: Handle,
        *,
        expected_revision: int,
    ) -> ExecutionRecord:
        """Atomically activate a caller-verified native scheduler generation.

        Native proof is the controller's responsibility. This API checks durable
        identity and CAS fences; it is never called from runtime push ingestion.
        """
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("An explicit nonnegative revision is required")
        if not isinstance(binding, SchedulerBinding) or not isinstance(
            new_handle, Handle
        ):
            raise ValueError("Typed scheduler binding and handle are required")
        binding_json, handle_json = _json(asdict(binding)), _json(new_handle.to_dict())
        with self._transaction():
            record = self._required(plan_id)
            approved_plan = Plan.from_json(
                self._connection.execute(
                    "SELECT plan_json FROM plans WHERE plan_id=?", (plan_id,)
                ).fetchone()[0]
            )
            if dict(approved_plan.resources).get("requeue") is not True:
                raise StaleExecution("Immutable plan did not approve scheduler requeue")
            if (
                record.native_identity is not None
                and binding.restart_count < record.native_identity.restart_count
            ):
                raise StaleExecution("Scheduler binding is behind durable native head")
            existing = self._connection.execute(
                "SELECT binding_json,handle_json,previous_revision FROM requeue_bindings "
                "WHERE plan_id=? AND attempt=?",
                (plan_id, binding.attempt),
            ).fetchone()
            if existing is not None:
                if (
                    existing["binding_json"] == binding_json
                    and existing["handle_json"] == handle_json
                    and existing["previous_revision"] == expected_revision
                    and record.attempt == binding.attempt
                    and record.handle == new_handle
                ):
                    return record
                raise StaleExecution("Conflicting or obsolete scheduler activation")
            if record.revision != expected_revision:
                raise StaleExecution("Execution revision changed")
            old = record.handle
            if record.backend != "slurm" or old is None or old.backend != "slurm":
                raise StaleExecution("Only attached Slurm executions can requeue")
            if binding.attempt <= record.attempt:
                raise StaleExecution("Scheduler attempt must increase")
            if (binding.run_id, binding.request_id, binding.owner) != (
                record.run_id,
                record.request_id,
                record.owner,
            ):
                raise StaleExecution("Scheduler binding differs from durable execution")
            raw, new_raw = json.loads(old.raw_json), json.loads(new_handle.raw_json)
            restart = raw.get("restart_count")
            if (
                type(restart) is not int
                or restart < 0
                or binding.restart_count <= restart
            ):
                raise StaleExecution("Scheduler restart count must increase")
            writer = approved_plan.allocation_mode == "writer"
            initial = None if writer else raw.get("initial_attempt", record.attempt)
            if not writer and (
                type(initial) is not int or not 1 <= initial <= record.attempt
            ):
                raise StaleExecution("Invalid original scheduler reservation")
            if writer:
                digest = hashlib.sha256(
                    approved_plan.to_json().encode("utf-8")
                ).hexdigest()
                if any(
                    native.get("allocation_mode") != "writer"
                    or native.get("plan_sha256") != digest
                    or "initial_attempt" in native
                    for native in (raw, new_raw)
                ):
                    raise StaleExecution(
                        "Writer receipt differs from immutable allocation plan"
                    )
            expected_job = binding.job_id + ";" + binding.cluster
            if old.job_id != expected_job or raw.get("cluster") != binding.cluster:
                raise StaleExecution("Scheduler binding differs from native allocation")
            immutable = (
                "backend",
                "run_id",
                "request_id",
                "owner",
                "run_dir",
                "job_id",
            )
            if any(
                getattr(old, name) != getattr(new_handle, name) for name in immutable
            ):
                raise StaleExecution("New handle differs from native allocation")
            if (
                new_handle.attempt != binding.attempt
                or new_handle.pid is not None
                or new_handle.supervisor_id is not None
            ):
                raise StaleExecution("Invalid Slurm generation handle")
            if old.attempt != record.attempt or old.run_dir != record.run_dir:
                raise StaleExecution("Prior receipt differs from durable execution")
            if (old.run_id, old.request_id, old.owner) != (
                record.run_id,
                record.request_id,
                record.owner,
            ):
                raise StaleExecution("Prior receipt has foreign execution identity")
            for native in (raw, new_raw):
                if (
                    native.get("comment") != record.request_id
                    or native.get("job_name") != f"wavedl-{record.request_id}"
                    or native.get("cluster") != binding.cluster
                ):
                    raise StaleExecution("Scheduler receipt native identity mismatch")
            if (
                type(new_raw.get("restart_count")) is not int
                or new_raw["restart_count"] != binding.restart_count
                or (
                    not writer
                    and (
                        new_raw.get("initial_attempt") != initial
                        or type(new_raw.get("initial_attempt")) is not int
                    )
                )
            ):
                raise StaleExecution("New receipt lacks matching restart reservation")
            if new_raw.get("operation") != raw.get("operation"):
                raise StaleExecution("Scheduler operation identity changed")
            if raw.get("operation") != approved_plan.operation.value:
                raise StaleExecution("Scheduler operation differs from approved plan")
            collision = self._connection.execute(
                "SELECT 1 FROM requeue_bindings WHERE cluster=? AND job_id=? "
                "AND owner=? AND request_id=? AND restart_count=?",
                (
                    binding.cluster,
                    binding.job_id,
                    binding.owner,
                    binding.request_id,
                    binding.restart_count,
                ),
            ).fetchone()
            if collision:
                raise StaleExecution("Native scheduler generation is already activated")
            if writer:
                from wavedl_daemon.storage.native_heads import record_native_observation

                record_native_observation(
                    self._connection,
                    plan_id,
                    SubmissionObservation(
                        ObservationStatus.ERROR,
                        detail="Startup binding proves native identity; lifecycle unconfirmed",
                        native_identity=_binding_identity(binding),
                    ),
                    self._connection.execute(
                        "SELECT revision FROM submission_control WHERE plan_id=?",
                        (plan_id,),
                    ).fetchone()[0],
                )
            self._archive_attempt(record)
            self._connection.execute(
                "INSERT INTO requeue_bindings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    plan_id,
                    binding.attempt,
                    record.attempt,
                    record.revision,
                    binding.run_id,
                    binding.request_id,
                    binding.owner,
                    binding.cluster,
                    binding.job_id,
                    binding.restart_count,
                    binding_json,
                    handle_json,
                ),
            )
            self._update(
                record,
                attempt=binding.attempt,
                handle_json=handle_json,
                state=ExecutionState.SUBMITTED,
                observation_json=None,
                latest_observation_json=None,
                cancel_delivered=0,
                detail="Validated scheduler restart activated",
            )
            return self._required(plan_id)

    def record_observation(
        self,
        plan_id: str,
        observation: BackendObservation,
        *,
        expected_revision: int | None = None,
    ) -> ExecutionRecord:
        if isinstance(observation, SubmissionObservation):
            return self.observe_submission(
                plan_id, observation, expected_revision=expected_revision
            )
        encoded = _json(asdict(observation))
        with self._transaction():
            record = self._required(plan_id, expected_revision)
            if observation.attempt != record.attempt or record.handle is None:
                raise StaleExecution("Observation is not for an attached execution")
            values = {"latest_observation_json": encoded}
            if observation.status not in {
                ObservationStatus.ERROR,
                ObservationStatus.ABSENT,
            }:
                previous = record.observation
                if (
                    previous is not None
                    and previous.status == ObservationStatus.TERMINAL
                ):
                    if observation.status != ObservationStatus.TERMINAL or any(
                        getattr(previous, key) is not None
                        and getattr(previous, key) != getattr(observation, key)
                        for key in (
                            "native_state",
                            "exit_code",
                            "signal",
                            "batch",
                            "stage_out",
                        )
                    ):
                        raise StaleExecution(
                            "Cannot contradict terminal backend evidence"
                        )
                values["observation_json"] = encoded
            self._update(record, **values)
            return self._required(plan_id)

    def mark_cancel_delivered(
        self, plan_id: str, *, expected_revision: int | None = None
    ) -> ExecutionRecord:
        with self._transaction():
            record = self._required(plan_id, expected_revision)
            if not record.cancellation_requested or (
                record.handle is None and record.submission_handle is None
            ):
                raise StaleExecution("Cancellation needs durable intent and receipt")
            if not record.cancel_delivered:
                self._update(record, cancel_delivered=1)
            return self._required(plan_id)
