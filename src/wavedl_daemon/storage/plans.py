# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Authoritative pre-runtime plans and control intent, not journal projections.

Request identity survives lost responses and runtime attempts. SQLite uniqueness
does not promise exactly-once external submission: a future dispatcher must
reconcile the durable request_id before retrying an ambiguous backend call.
Losing this database requires backup/operator recovery, never journal replay.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from wavedl_daemon.backends.contracts import Plan
    from wavedl_daemon.storage.ownership import DispatchLock


class IdempotencyConflict(ValueError):
    """An existing idempotency key was reused for a different request."""


class PlanRecoveryRequired(MigrationFailed):
    """Intent cannot safely be recovered from runtime telemetry."""


def _create_tables(connection: sqlite3.Connection) -> None:
    if schema_version(connection) != 1:
        raise MigrationFailed("Persisted plans require the existing version 1 registry")
    connection.execute(
        "CREATE TABLE plans ("
        "plan_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, "
        "request_payload_json TEXT NOT NULL, plan_json TEXT NOT NULL, "
        "request_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE plan_control ("
        "plan_id TEXT PRIMARY KEY REFERENCES plans(plan_id), "
        "submission_requested INTEGER NOT NULL DEFAULT 0 "
        "CHECK(submission_requested IN (0, 1)), "
        "cancellation_requested INTEGER NOT NULL DEFAULT 0 "
        "CHECK(cancellation_requested IN (0, 1)))"
    )
    connection.execute(
        "CREATE TRIGGER plans_immutable_update BEFORE UPDATE ON plans "
        "BEGIN SELECT RAISE(ABORT, 'persisted plan is immutable'); END"
    )
    connection.execute(
        "CREATE TRIGGER plans_immutable_delete BEFORE DELETE ON plans "
        "BEGIN SELECT RAISE(ABORT, 'persisted plan is immutable'); END"
    )


PLAN_MIGRATION = Migration("0002_persisted_plans", 2, _create_tables)


@dataclass(frozen=True)
class PlanRecord:
    """Immutable preview with durable request identity and current intent."""

    plan_id: str
    idempotency_key: str
    request_payload_json: str
    plan_json: str
    request_id: str
    created_at: str
    submission_requested: bool
    cancellation_requested: bool

    @property
    def plan(self) -> Plan:
        from wavedl_daemon.backends.contracts import Plan

        return Plan.from_json(self.plan_json)


def _canonical_request(payload: Mapping[str, object]) -> str:
    # Use only the submitted JSON payload; never re-read files or environment.
    if type(payload) is not dict:
        raise ValueError("Request payload must be a JSON object, not a custom mapping")
    active: set[int] = set()

    def validate(value: object, depth: int = 0) -> None:
        if depth > 64:
            raise ValueError("Request payload exceeds maximum JSON nesting")
        if type(value) in {dict, list}:
            identity = id(value)
            if identity in active:
                raise ValueError("Request payload contains a JSON cycle")
            active.add(identity)
            try:
                if type(value) is dict:
                    if not all(type(key) is str for key in value):
                        raise ValueError("Request payload keys must be strings")
                    children = value.values()
                else:
                    children = value
                for child in children:
                    validate(child, depth + 1)
            finally:
                active.remove(identity)
        elif type(value) not in {str, int, float, bool, type(None)}:
            raise ValueError("Request payload must contain only JSON values")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("Request payload numbers must be finite")

    validate(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


_SELECT = (
    "SELECT p.plan_id, p.idempotency_key, p.request_payload_json, "
    "p.plan_json, p.request_id, p.created_at, "
    "c.submission_requested, c.cancellation_requested "
    "FROM plans p JOIN plan_control c ON c.plan_id = p.plan_id "
)


class PlanStore:
    """Open an already migrated registry; mutation requires dispatch ownership."""

    def __init__(
        self,
        path: Path,
        *,
        dispatch_lock: DispatchLock | None = None,
        read_only: bool = False,
    ):
        self.path = Path(path).resolve()
        self.read_only = read_only
        self.dispatch_lock = dispatch_lock
        try:
            stat = self.path.stat()
        except FileNotFoundError as exc:
            raise PlanRecoveryRequired(
                "Intent database is missing; restore a backup, not runtime journals"
            ) from exc
        self._identity = (stat.st_dev, stat.st_ino)
        mode = "ro" if read_only else "rw"
        self._connection = sqlite3.connect(
            self.path.as_uri() + f"?mode={mode}", uri=True, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        try:
            version = schema_version(self._connection)
            if version < 2 or (
                version not in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
                and not read_only
            ):
                raise PlanRecoveryRequired(
                    f"Intent schema {version} is unsupported; migrate or open "
                    "read-only recovery, never recreate control records"
                )
            self._connection.execute(_SELECT + "LIMIT 0")
            incomplete = self._connection.execute(
                "SELECT 1 FROM plans p LEFT JOIN plan_control c "
                "ON c.plan_id=p.plan_id WHERE c.plan_id IS NULL LIMIT 1"
            ).fetchone()
            if incomplete:
                raise PlanRecoveryRequired("Persisted plan has lost its control record")
            if not read_only:
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA foreign_keys=ON")
        except sqlite3.DatabaseError as exc:
            self._connection.close()
            raise PlanRecoveryRequired(
                "Intent schema is damaged or incompatible"
            ) from exc
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PlanStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _record(self, row: sqlite3.Row | None) -> PlanRecord | None:
        if row is None:
            return None
        values = dict(row)
        for name in ("submission_requested", "cancellation_requested"):
            values[name] = bool(values[name])
        return PlanRecord(**values)

    def get(self, plan_id: str) -> PlanRecord | None:
        self._guard_read()
        return self._record(
            self._connection.execute(
                _SELECT + "WHERE p.plan_id = ?", (plan_id,)
            ).fetchone()
        )

    def get_by_key(self, idempotency_key: str) -> PlanRecord | None:
        self._guard_read()
        return self._record(
            self._connection.execute(
                _SELECT + "WHERE p.idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        )

    def get_by_run_id(self, run_id: str) -> PlanRecord | None:
        """Resolve immutable execution identity without scanning mutable telemetry."""
        from uuid import UUID

        self._guard_read()

        if str(UUID(run_id)) != run_id:
            raise ValueError("Run ID must be a canonical UUID")
        # Immutable snapshots have no scalar run_id column in older schemas.
        # JSON extraction is implemented by SQLite, not by an unbounded Python
        # scan. Identity belongs to the root of the persisted manifest JSON.
        row = self._connection.execute(
            _SELECT + "WHERE json_extract(p.plan_json, '$.manifest_json') IS NOT NULL "
            "AND json_extract(json_extract(p.plan_json, '$.manifest_json'), '$.run_id')=? "
            "LIMIT 1",
            (run_id,),
        ).fetchone()
        return self._record(row)

    def list_plans(self, limit: int = 100, offset: int = 0) -> list[PlanRecord]:
        self._guard_read()
        if (
            type(limit) is not int
            or not 1 <= limit <= 1000
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("Invalid plan page bounds")
        return [
            self._record(row)
            for row in self._connection.execute(
                _SELECT + "ORDER BY p.created_at,p.plan_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
        ]

    def _guard_read(self):
        try:
            info = self.path.stat()
        except FileNotFoundError:
            raise PlanRecoveryRequired(
                "Intent database is missing; restore recovery state"
            ) from None
        if (info.st_dev, info.st_ino) != self._identity:
            raise PlanRecoveryRequired("Intent database was replaced; reopen recovery")

    def list_pending(self, limit: int = 100, offset: int = 0) -> list[PlanRecord]:
        """Page durable intents in stable order, never an unbounded fleet scan."""
        if (
            type(limit) is not int
            or not 1 <= limit <= 1000
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("Invalid plan page bounds")
        try:
            stat = self.path.stat()
        except FileNotFoundError as exc:
            raise PlanRecoveryRequired("Intent database was removed") from exc
        if (stat.st_dev, stat.st_ino) != self._identity:
            raise PlanRecoveryRequired("Intent database was replaced; reopen recovery")
        return [
            self._record(row)
            for row in self._connection.execute(
                _SELECT
                + "WHERE c.submission_requested=1 OR c.cancellation_requested=1 "
                "ORDER BY p.created_at, p.plan_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
        ]

    @contextmanager
    def _transaction(self, dispatch_lock: DispatchLock | None) -> Iterator[None]:
        if self.read_only:
            raise PermissionError("PlanStore is in read-only recovery mode")
        lock = dispatch_lock if dispatch_lock is not None else self.dispatch_lock
        require_dispatch(lock)
        if lock.path.resolve() != self.path.parent / "dispatch.lock":
            raise DispatchRequired("dispatch ownership is for a different registry")
        try:
            stat = self.path.stat()
        except FileNotFoundError as exc:
            raise PlanRecoveryRequired("Intent database was removed") from exc
        if (stat.st_dev, stat.st_ino) != self._identity:
            raise PlanRecoveryRequired("Intent database was replaced; reopen recovery")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def put(
        self,
        plan: Plan,
        *,
        idempotency_key: str,
        request_payload: Mapping[str, object],
        dispatch_lock: DispatchLock | None = None,
    ) -> PlanRecord:
        """Commit snapshot and control row together, or replay the original."""
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a nonempty string")
        payload_json = _canonical_request(request_payload)
        with self._transaction(dispatch_lock):
            existing = self.get_by_key(idempotency_key)
            if existing is not None:
                if existing.request_payload_json != payload_json:
                    raise IdempotencyConflict("Key is bound to a different request")
                return existing
            plan_id = str(uuid4())
            self._connection.execute(
                "INSERT INTO plans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    plan_id,
                    idempotency_key,
                    payload_json,
                    plan.to_json(),
                    str(uuid4()),
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._write_control(plan_id)
            record = self.get(plan_id)
            if record is None:
                raise PlanRecoveryRequired("Plan/control transaction is incomplete")
            return record

    def _write_control(self, plan_id: str) -> None:
        self._connection.execute(
            "INSERT INTO plan_control(plan_id) VALUES (?)", (plan_id,)
        )

    def request_submission(
        self, plan_id: str, *, dispatch_lock: DispatchLock | None = None
    ) -> PlanRecord:
        """Persist intent before any backend call; this does not submit work."""
        return self._request_intent(plan_id, "submission_requested", dispatch_lock)

    def request_cancellation(
        self, plan_id: str, *, dispatch_lock: DispatchLock | None = None
    ) -> PlanRecord:
        """Record cancellation even before an external handle is known."""
        return self._request_intent(plan_id, "cancellation_requested", dispatch_lock)

    def _request_intent(
        self, plan_id: str, column: str, dispatch_lock: DispatchLock | None
    ) -> PlanRecord:
        with self._transaction(dispatch_lock):
            if self.get(plan_id) is None:
                raise KeyError(plan_id)
            self._connection.execute(
                f"UPDATE plan_control SET {column} = 1 WHERE plan_id = ?", (plan_id,)
            )
            record = self.get(plan_id)
            if record is None:
                raise PlanRecoveryRequired("Plan/control record disappeared")
            return record
