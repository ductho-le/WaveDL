# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Immutable execution-environment evidence, separate from approved requests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.events import HelloPayload
from wavedl.core.protocol import Envelope, ProtocolError, decode
from wavedl.core.runtime_identity import checked_python_identity
from wavedl_daemon.backends.contracts import Handle, Plan
from wavedl_daemon.runs.negotiation import checked_hello
from wavedl_daemon.storage.executions import StaleExecution
from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


if TYPE_CHECKING:
    from collections.abc import Iterator

    from wavedl_daemon.storage.ownership import DispatchLock


def _create_tables(connection: sqlite3.Connection) -> None:
    if schema_version(connection) != 3:
        raise MigrationFailed("Runtime provenance requires schema 3")
    connection.execute(
        "CREATE TABLE runtime_provenance ("
        "plan_id TEXT NOT NULL REFERENCES execution_control(plan_id), "
        "run_id TEXT NOT NULL, attempt INTEGER NOT NULL CHECK(attempt>0), "
        "event_json TEXT NOT NULL, protocol_version INTEGER NOT NULL "
        "CHECK(protocol_version>0), observed_at TEXT NOT NULL, "
        "PRIMARY KEY(plan_id,attempt), UNIQUE(run_id,attempt))"
    )
    connection.execute(
        "CREATE TRIGGER runtime_provenance_immutable BEFORE UPDATE "
        "ON runtime_provenance "
        "BEGIN SELECT RAISE(ABORT, 'runtime provenance is immutable'); END"
    )
    connection.execute(
        "CREATE TRIGGER runtime_provenance_no_delete BEFORE DELETE "
        "ON runtime_provenance "
        "BEGIN SELECT RAISE(ABORT, 'runtime provenance is immutable'); END"
    )


RUNTIME_PROVENANCE_MIGRATION = Migration("0004_runtime_provenance", 4, _create_tables)
MAX_JSON_BYTES = 256 * 1024


def _approved_runtime(
    plan: Plan, hello: HelloPayload, handle: Handle | None = None
) -> None:
    """Check the actual environment without dereferencing selected venv paths."""
    for name in ("runtime_version", "interpreter", "python"):
        value = getattr(hello, name)
        if not value.strip() or "\0" in value:
            raise ProtocolError(f"Runtime hello requires nonempty actual {name}")
    if not os.path.isabs(hello.interpreter):
        raise ProtocolError("Actual runtime interpreter must be absolute")
    version = re.fullmatch(
        r"([1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})(?:\.(0|[1-9][0-9]{0,3}))?",
        hello.python,
    )
    if version is None or tuple(map(int, version.group(1, 2))) < (3, 11):
        raise ProtocolError("Actual runtime Python must be a valid version >=3.11")
    selected = plan.manifest.runtime
    raw = {} if handle is None else json.loads(handle.raw_json)
    if "runtime_attestation" in raw:
        attestation = raw["runtime_attestation"]
        expected = {
            "schema_version": 1,
            "selected_python": selected.python,
            "plan_sha256": hashlib.sha256(plan.to_json().encode()).hexdigest(),
            "run_id": plan.run_id,
            "request_id": handle.request_id,
            "owner": handle.owner,
            "supervisor_id": handle.supervisor_id,
        }
        if (
            handle.backend != "local"
            or plan.target != "local"
            or handle.run_id != plan.run_id
            or handle.run_dir != plan.run_dir
            or not handle.supervisor_id
            or type(attestation) is not dict
            or type(attestation.get("schema_version")) is not int
            or set(attestation) != {*expected, "identity"}
            or any(attestation[key] != value for key, value in expected.items())
        ):
            raise ProtocolError("Runtime attestation differs from approved execution")
        try:
            approved = checked_python_identity(attestation["identity"])
            actual = checked_python_identity(
                {
                    "interpreter": hello.interpreter,
                    "prefix": hello.prefix,
                    "base_prefix": hello.base_prefix,
                }
            )
        except ValueError as error:
            raise ProtocolError("Invalid runtime Python attestation") from error
        if actual != approved:
            raise ProtocolError(
                "Actual Python environment differs from probe-attested runtime"
            )
    elif os.path.normcase(os.path.normpath(selected.python)) != os.path.normcase(
        os.path.normpath(hello.interpreter)
    ):
        raise ProtocolError("Actual interpreter differs from approved runtime")
    if selected.device != "auto" and hello.device != selected.device:
        raise ProtocolError("Actual device differs from approved runtime")
    devices = hello.allocated_devices
    if any(not device.strip() or "\0" in device for device in devices) or len(
        set(devices)
    ) != len(devices):
        raise ProtocolError("Allocated device identities must be distinct and nonempty")
    if hello.device == "cuda" and len(devices) < plan.topology.processes_per_machine:
        raise ProtocolError("Actual CUDA allocation cannot satisfy approved topology")


@dataclass(frozen=True)
class RuntimeProvenanceRecord:
    plan_id: str
    run_id: str
    attempt: int
    event_json: str
    protocol_version: int
    observed_at: str

    @property
    def event(self) -> Envelope:
        return decode(self.event_json)

    @property
    def hello(self) -> HelloPayload:
        """Fresh typed snapshot so list mutations cannot alter stored evidence."""
        return HelloPayload.from_dict(self.event.payload)


class RuntimeProvenanceStore:
    """Append actual provenance once under matching live dispatch ownership."""

    def __init__(
        self,
        path: Path,
        *,
        dispatch_lock: DispatchLock | None = None,
        read_only: bool = False,
    ):
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
            if version < 4 or (
                version not in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13} and not read_only
            ):
                raise PlanRecoveryRequired(f"Unsupported runtime schema {version}")
            self._connection.execute("SELECT * FROM runtime_provenance LIMIT 0")
            incomplete = self._connection.execute(
                "SELECT 1 FROM runtime_provenance r "
                "LEFT JOIN execution_control e ON e.plan_id=r.plan_id "
                "LEFT JOIN plans p ON p.plan_id=r.plan_id "
                "WHERE e.plan_id IS NULL OR p.plan_id IS NULL OR "
                "e.attempt < r.attempt LIMIT 1"
            ).fetchone()
            if incomplete:
                raise PlanRecoveryRequired("Runtime provenance lost execution identity")
            if not read_only:
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA foreign_keys=ON")
        except sqlite3.DatabaseError as exc:
            self.close()
            raise PlanRecoveryRequired("Runtime provenance schema is damaged") from exc
        except BaseException:
            self.close()
            raise

    def _stat(self) -> tuple[int, int]:
        try:
            stat = self.path.stat()
        except FileNotFoundError as exc:
            raise PlanRecoveryRequired("Runtime database is missing") from exc
        return stat.st_dev, stat.st_ino

    def _guard(self) -> None:
        if self._stat() != self._identity:
            raise PlanRecoveryRequired("Runtime database replaced; reopen recovery")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> RuntimeProvenanceStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        if self.read_only:
            raise PermissionError("Runtime provenance is in read-only recovery mode")
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

    def get(
        self, plan_id: str, attempt: int | None = None
    ) -> RuntimeProvenanceRecord | None:
        if attempt is not None and (type(attempt) is not int or attempt < 1):
            raise ValueError("attempt must be a positive integer")
        self._guard()
        row = self._connection.execute(
            "SELECT plan_id,run_id,attempt,event_json,protocol_version,observed_at "
            "FROM runtime_provenance WHERE plan_id=? "
            + (
                "AND attempt=? "
                if attempt is not None
                else "AND attempt=(SELECT e.attempt FROM execution_control e WHERE e.plan_id=runtime_provenance.plan_id) "
            )
            + "ORDER BY attempt DESC LIMIT 1",
            (plan_id, attempt) if attempt is not None else (plan_id,),
        ).fetchone()
        if row is None:
            return None
        record = RuntimeProvenanceRecord(**dict(row))
        event, _, version = checked_hello(record.event)
        if (
            event != record.event_json
            or version != record.protocol_version
            or record.event.run_id != record.run_id
            or record.event.attempt != record.attempt
        ):
            raise PlanRecoveryRequired("Stored runtime provenance identity is damaged")
        return record

    def record(self, plan_id: str, event: Envelope) -> RuntimeProvenanceRecord:
        encoded, hello, version = checked_hello(event)
        if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("Runtime provenance exceeds JSON size bound")
        with self._transaction():
            row = self._connection.execute(
                "SELECT e.attempt,e.handle_json,p.plan_json FROM execution_control e "
                "JOIN plans p ON p.plan_id=e.plan_id WHERE e.plan_id=?",
                (plan_id,),
            ).fetchone()
            if row is None:
                raise KeyError(plan_id)
            plan = Plan.from_json(row["plan_json"])
            if event.run_id != plan.run_id or event.attempt != row["attempt"]:
                raise StaleExecution("Runtime hello is not for the active execution")
            handle = (
                None
                if row["handle_json"] is None
                else Handle.from_json(row["handle_json"])
            )
            _approved_runtime(plan, hello, handle)
            existing = self.get(plan_id, event.attempt)
            if existing is not None:
                if existing.event_json != encoded:
                    raise StaleExecution("Cannot replace immutable actual provenance")
                return existing
            self._connection.execute(
                "INSERT INTO runtime_provenance "
                "(plan_id,run_id,attempt,event_json,protocol_version,observed_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    plan_id,
                    event.run_id,
                    event.attempt,
                    encoded,
                    version,
                    datetime.now(UTC).isoformat(),
                ),
            )
            return self.get(plan_id, event.attempt)
