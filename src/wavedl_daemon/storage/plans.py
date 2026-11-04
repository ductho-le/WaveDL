"""
Immutable plans and the request identity that replays them.

Everything after the plan, including submit and cancel intent, lives in the run
log, so this table never changes after an insert.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from wavedl_daemon.backends.contracts import Plan


if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping


class IdempotencyConflict(ValueError):
    """An existing idempotency key was reused for a different request."""


@dataclass(frozen=True)
class PlanRecord:
    plan_id: str
    run_id: str
    request_id: str
    idempotency_key: str
    request_json: str
    plan: Plan
    created_at: str


def canonical_request(payload: Mapping[str, object]) -> str:
    """The request as submitted JSON; files and environment are never re-read."""
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
    "SELECT plan_id, run_id, request_id, idempotency_key, request_json, "
    "plan_json, created_at FROM plans "
)


def _record(row: sqlite3.Row | None) -> PlanRecord | None:
    if row is None:
        return None
    return PlanRecord(
        row["plan_id"],
        row["run_id"],
        row["request_id"],
        row["idempotency_key"],
        row["request_json"],
        Plan.from_json(row["plan_json"]),
        row["created_at"],
    )


class PlanStore:
    """Reads and inserts on a connection from ``open_database``."""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def get(self, plan_id: str) -> PlanRecord | None:
        return _record(
            self._connection.execute(
                _SELECT + "WHERE plan_id = ?", (plan_id,)
            ).fetchone()
        )

    def get_by_key(self, idempotency_key: str) -> PlanRecord | None:
        return _record(
            self._connection.execute(
                _SELECT + "WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        )

    def by_run(self, run_id: str) -> PlanRecord | None:
        return _record(
            self._connection.execute(_SELECT + "WHERE run_id = ?", (run_id,)).fetchone()
        )

    def page(self, *, limit: int, offset: int) -> list[PlanRecord]:
        rows = self._connection.execute(
            _SELECT + "ORDER BY created_at, plan_id LIMIT ? OFFSET ?", (limit, offset)
        )
        return [_record(row) for row in rows]

    def put(
        self, plan: Plan, *, idempotency_key: str, request_payload: Mapping[str, object]
    ) -> PlanRecord:
        """Insert the plan, or replay the one this key already names."""
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a nonempty string")
        request_json = canonical_request(request_payload)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.get_by_key(idempotency_key)
            if existing is None:
                self._connection.execute(
                    "INSERT INTO plans (plan_id, run_id, request_id, idempotency_key, "
                    "request_json, plan_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid4()),
                        plan.run_id,
                        str(uuid4()),
                        idempotency_key,
                        request_json,
                        plan.to_json(),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        if existing is not None and existing.request_json != request_json:
            raise IdempotencyConflict("Key is bound to a different request")
        record = existing or self.get_by_key(idempotency_key)
        if record is None:
            raise RuntimeError("The plan insert did not commit")
        return record
