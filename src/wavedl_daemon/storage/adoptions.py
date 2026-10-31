# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Atomic registration of verified external runs, never dispatchable intents."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from wavedl_daemon.backends.contracts import Handle
from wavedl_daemon.runs.negotiation import checked_hello
from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.plans import (
    IdempotencyConflict,
    PlanRecord,
    PlanRecoveryRequired,
    _canonical_request,
)
from wavedl_daemon.storage.runtime_provenance import _approved_runtime
from wavedl_daemon.storage.schema import schema_version


class AdoptionRegistrationStore:
    """Open an existing authoritative database on its owner thread only."""

    def __init__(
        self, path: Path, *, dispatch_lock, read_only=False, expected_identity=None
    ):
        self.path = Path(path).resolve()
        self.dispatch_lock, self.read_only = dispatch_lock, read_only
        self._identity = self._stat()
        if expected_identity is not None and self._identity != expected_identity:
            raise PlanRecoveryRequired("Adoption database differs from owner session")
        self.connection = sqlite3.connect(
            self.path.as_uri() + ("?mode=ro" if read_only else "?mode=rw"),
            uri=True,
            isolation_level=None,
        )
        self.connection.row_factory = sqlite3.Row
        try:
            if schema_version(self.connection) not in {6, 7, 8, 9, 10, 11, 12, 13}:
                raise PlanRecoveryRequired("Verified adoption requires schema 6")
            self.connection.execute("PRAGMA foreign_keys=ON")
            if not read_only:
                self.connection.execute("PRAGMA synchronous=FULL")
        except BaseException:
            self.close()
            raise

    def _stat(self):
        try:
            info = self.path.stat()
        except FileNotFoundError:
            raise PlanRecoveryRequired("Adoption database is missing") from None
        return info.st_dev, info.st_ino

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _guard(self):
        if self.read_only:
            raise PermissionError("Adoption is unavailable in read-only recovery")
        require_dispatch(self.dispatch_lock)
        if self.dispatch_lock.path.resolve() != self.path.parent / "dispatch.lock":
            raise DispatchRequired("Dispatch ownership belongs to another registry")
        if self._stat() != self._identity:
            raise PlanRecoveryRequired("Adoption database changed; reopen recovery")

    def register_adoption(
        self,
        plan,
        *,
        request_id,
        attempt,
        owner,
        idempotency_key,
        payload,
        hello,
        verified_guard,
    ) -> PlanRecord:
        """Commit all identities and LOCAL_LOST state together, or nothing."""
        if str(UUID(request_id)) != request_id or plan.target != "local":
            raise ValueError("Invalid standalone adoption identity")
        if type(attempt) is not int or attempt < 1 or not owner or "\0" in owner:
            raise ValueError("Invalid adoption attempt/owner")
        if type(idempotency_key) is not str or not 1 <= len(idempotency_key) <= 256:
            raise ValueError("A bounded idempotency key is required")
        encoded, actual, version = checked_hello(hello)
        if hello.run_id != plan.run_id or hello.attempt != attempt:
            raise ValueError("Adopted hello identity mismatch")
        _approved_runtime(plan, actual)
        payload_json = _canonical_request(payload)
        self._guard()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._guard()
            verified_guard()
            existing = self.connection.execute(
                "SELECT p.*,c.submission_requested,c.cancellation_requested, "
                "e.attempt,e.state,e.handle_json,r.event_json FROM plans p "
                "LEFT JOIN plan_control c ON c.plan_id=p.plan_id "
                "LEFT JOIN execution_control e ON e.plan_id=p.plan_id "
                "LEFT JOIN runtime_provenance r ON r.plan_id=p.plan_id AND r.attempt=e.attempt "
                "WHERE p.idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["request_payload_json"] != payload_json
                    or existing["request_id"] != request_id
                    or existing["plan_json"] != plan.to_json()
                    or existing["attempt"] != attempt
                    or existing["state"] != "local_lost"
                    or existing["event_json"] != encoded
                    or existing["handle_json"] is None
                    or not Handle.from_json(existing["handle_json"]).raw_json
                    == '{"adopted":true}'
                ):
                    raise IdempotencyConflict("Key belongs to another registration")
                record = self._record(existing)
            else:
                duplicate = self.connection.execute(
                    "SELECT 1 FROM plans WHERE request_id=? OR "
                    "json_extract(json_extract(plan_json,'$.manifest_json'),'$.run_id')=?",
                    (request_id, plan.run_id),
                ).fetchone()
                if duplicate:
                    raise IdempotencyConflict("Run/request is already registered")
                plan_id, now = str(uuid4()), datetime.now(UTC).isoformat()
                self.connection.execute(
                    "INSERT INTO plans VALUES (?,?,?,?,?,?)",
                    (
                        plan_id,
                        idempotency_key,
                        payload_json,
                        plan.to_json(),
                        request_id,
                        now,
                    ),
                )
                self.connection.execute(
                    "INSERT INTO plan_control(plan_id,submission_requested) VALUES (?,1)",
                    (plan_id,),
                )
                handle = Handle(
                    "local",
                    plan.run_id,
                    request_id,
                    attempt,
                    owner,
                    plan.run_dir,
                    raw_json='{"adopted":true}',
                )
                self._insert_execution(plan_id, attempt, owner, now, handle.to_json())
                self.connection.execute(
                    "INSERT INTO runtime_provenance VALUES (?,?,?,?,?,?)",
                    (plan_id, plan.run_id, attempt, encoded, version, now),
                )
                record = PlanRecord(
                    plan_id,
                    idempotency_key,
                    payload_json,
                    plan.to_json(),
                    request_id,
                    now,
                    True,
                    False,
                )
            self.connection.execute("COMMIT")
            return record
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def _insert_execution(self, plan_id, attempt, owner, now, handle_json):
        self.connection.execute(
            "INSERT INTO execution_control(plan_id,attempt,owner,state,handle_json,updated_at,detail) "
            "VALUES (?,?,?,'local_lost',?,?,?)",
            (
                plan_id,
                attempt,
                owner,
                handle_json,
                now,
                "Explicitly adopted standalone run has no daemon-owned supervisor",
            ),
        )

    def _record(self, row):
        return PlanRecord(
            **{
                name: (
                    bool(row[name])
                    if name in {"submission_requested", "cancellation_requested"}
                    else row[name]
                )
                for name in PlanRecord.__dataclass_fields__
            }
        )
