"""
The run log: the only writer of ``run_events`` and ``run_heads``.

``run_events`` is the full history of every run. ``run_heads`` caches the fold
of those events and is written in the same transaction by the same ``evolve``
that validated them, so the two cannot disagree. SQL triggers repeat the
ordering rule, so no writer can commit a row out of order behind a replay
cursor.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon.runs.lifecycle import (
    Claimed,
    Ended,
    IllegalTransition,
    Live,
    Requested,
    RunHead,
    RunRef,
    evolve,
    status,
)
from wavedl_daemon.runs.model import (
    BackendKind,
    Exited,
    Metric,
    Operation,
    Outcome,
    Phase,
    PlanId,
    RunId,
    SourceWarning,
    StoredEvent,
    canonical_json,
    handle_payload,
    parse_event,
    parse_handle,
)


if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Iterator, Sequence

    from wavedl_daemon.runs.lifecycle import RunState, Stage
    from wavedl_daemon.runs.model import RunEvent


RUN_LOG_SCHEMA = (
    "CREATE TABLE run_events ("
    "run_id TEXT NOT NULL, attempt INTEGER NOT NULL CHECK(attempt >= 1), "
    "seq INTEGER NOT NULL CHECK(seq >= 0), type TEXT NOT NULL, ts TEXT NOT NULL, "
    "payload_json TEXT NOT NULL, "
    "source_end INTEGER CHECK(source_end > 0), "
    "CHECK(source_end IS NULL OR type IN ('metric', 'warning')), "
    "PRIMARY KEY(run_id, attempt, seq)) WITHOUT ROWID",
    "CREATE TRIGGER run_events_in_order BEFORE INSERT ON run_events WHEN "
    "(NEW.attempt NOT BETWEEN "
    "coalesce((SELECT max(attempt) FROM run_events WHERE run_id = NEW.run_id), 1) "
    "AND coalesce((SELECT max(attempt) FROM run_events WHERE run_id = NEW.run_id), 0) + 1) "
    "OR NEW.seq != coalesce((SELECT max(seq) FROM run_events "
    "WHERE run_id = NEW.run_id AND attempt = NEW.attempt), -1) + 1 "
    "OR (NEW.seq = 0) != (NEW.type = 'attempt') "
    "BEGIN SELECT RAISE(ABORT, 'run event out of commit order'); END",
    "CREATE TRIGGER run_events_no_update BEFORE UPDATE ON run_events "
    "BEGIN SELECT RAISE(ABORT, 'run events are append-only'); END",
    "CREATE TRIGGER run_events_no_delete BEFORE DELETE ON run_events "
    "BEGIN SELECT RAISE(ABORT, 'run events are append-only'); END",
    "CREATE TABLE run_heads ("
    "run_id TEXT PRIMARY KEY, attempt INTEGER NOT NULL, seq INTEGER NOT NULL, "
    "revision INTEGER NOT NULL, state TEXT NOT NULL, "
    "active INTEGER NOT NULL CHECK(active IN (0, 1)), "
    "head_json TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE INDEX run_heads_active ON run_heads(updated_at) WHERE active = 1",
    "CREATE INDEX run_heads_by_state ON run_heads(state, updated_at)",
)

MAX_PAGE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class New:
    """The run must have no events yet."""

    ref: RunRef


@dataclass(frozen=True)
class Expect:
    """The head tokens the caller's work was read under; None skips a token."""

    revision: int | None = None
    offset: tuple[int, int] | None = None  # (attempt, metrics_offset)


class Stale(Exception):
    """The head moved since the caller read it. Re-read and retry."""


def validate_cursor(cursor: tuple[int, int]) -> tuple[int, int]:
    """A replay position: ``(0, -1)`` before everything, else ``(attempt, seq)``."""
    attempt, seq = cursor
    if (
        type(attempt) is not int
        or type(seq) is not int
        or not 0 <= attempt <= 2**63 - 1
        or not -1 <= seq <= 2**63 - 1
        or (attempt == 0 and seq != -1)
    ):
        raise ValueError("Invalid attempt/sequence cursor")
    return cursor


def _now() -> str:
    return datetime.now(UTC).isoformat()


_HEAD_COLUMNS = "attempt, seq, revision, head_json"
_EVENT_COLUMNS = "attempt, seq, type, ts, payload_json, source_end"


class RunLog:
    """Owns its connection. ``notify(run_id)`` runs after each commit."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        notify: Callable[[str], None] = lambda run_id: None,
        clock: Callable[[], str] = _now,
    ):
        self._connection = connection
        self._notify = notify
        self._clock = clock

    def head(self, run_id: RunId) -> RunHead | None:
        row = self._connection.execute(
            f"SELECT {_HEAD_COLUMNS} FROM run_heads WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else _head(run_id, row)

    def heads(
        self,
        *,
        active: bool | None = None,
        state: RunState | None = None,
        limit: int = 256,
        offset: int = 0,
    ) -> list[RunHead]:
        # A literal ``active = 1`` lets SQLite use the partial index.
        where = {None: "1", True: "active = 1", False: "active = 0"}[active]
        if state is not None:
            where += " AND state = ?"
        rows = self._connection.execute(
            f"SELECT run_id, {_HEAD_COLUMNS} FROM run_heads WHERE {where} "
            "ORDER BY updated_at DESC, run_id LIMIT ? OFFSET ?",
            (*(() if state is None else (state.value,)), limit, offset),
        )
        return [_head(RunId(row["run_id"]), row) for row in rows]

    def append(
        self, run_id: RunId, events: Sequence[RunEvent], *, expect: New | Expect
    ) -> RunHead:
        """Validate, number and commit ``events`` together, or none of them."""
        if not events:
            raise ValueError("Nothing to append")
        ts = self._clock()
        with self._transaction():
            current = self.head(run_id)
            head = _start(run_id, current, expect)
            rows = []
            for event in events:
                head = evolve(head, event)
                rows.append(
                    (
                        run_id,
                        head.attempt,
                        head.seq,
                        event.kind,
                        ts,
                        canonical_json(event.payload()),
                        event.source_end
                        if isinstance(event, Metric | SourceWarning)
                        else None,
                    )
                )
            if (
                isinstance(head.stage, Ended)
                and head.stage.exit.outcome is Outcome.REQUEUED
            ):
                raise IllegalTransition(
                    "A requeue must open the next attempt in the same append"
                )
            self._connection.executemany(
                "INSERT INTO run_events (run_id, attempt, seq, type, ts, payload_json, "
                "source_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._connection.execute(
                "INSERT INTO run_heads (run_id, attempt, seq, revision, state, active, "
                "head_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET attempt = excluded.attempt, "
                "seq = excluded.seq, revision = excluded.revision, "
                "state = excluded.state, active = excluded.active, "
                "head_json = excluded.head_json, updated_at = excluded.updated_at",
                (
                    run_id,
                    head.attempt,
                    head.seq,
                    head.revision,
                    status(head).state.value,
                    int(not isinstance(head.stage, Ended)),
                    _head_json(head),
                    ts,
                ),
            )
        self._notify(run_id)
        return head

    def read_after(
        self, run_id: RunId, cursor: tuple[int, int], *, limit: int = 256
    ) -> list[StoredEvent]:
        """Events after ``cursor`` in ``(attempt, seq)`` order, the SSE source.

        A page holds at most MAX_PAGE_BYTES of payload but always one event,
        so a single large row cannot stall a stream.
        """
        rows = self._connection.execute(
            f"SELECT {_EVENT_COLUMNS} FROM run_events WHERE run_id = ? "
            "AND (attempt, seq) > (?, ?) ORDER BY attempt, seq LIMIT ?",
            (run_id, *cursor, limit),
        )
        events, size = [], 0
        for row in rows:
            size += len(row["payload_json"].encode("utf-8"))
            if events and size > MAX_PAGE_BYTES:
                break
            events.append(
                StoredEvent(
                    run_id,
                    row["attempt"],
                    row["seq"],
                    row["ts"],
                    parse_event(
                        row["type"], json.loads(row["payload_json"]), row["source_end"]
                    ),
                )
            )
        return events

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise


def _start(
    run_id: RunId, current: RunHead | None, expect: New | Expect
) -> RunHead | RunRef:
    match expect:
        case New(ref=ref) if current is None and ref.run_id == run_id:
            return ref
        case Expect(revision=revision, offset=offset) if current is not None and (
            revision in (None, current.revision)
            and offset in (None, (current.attempt, current.metrics_offset))
        ):
            return current
    raise Stale(f"Run {run_id} is not at the expected head")


def encode(event: StoredEvent) -> bytes:
    """The SSE ``data`` of one event."""
    return canonical_json(
        {
            "run_id": event.run_id,
            "attempt": event.attempt,
            "seq": event.seq,
            "ts": event.ts,
            "type": event.event.kind,
            "payload": event.event.payload(),
            "protocol": "wavedl-run",
            "version": 2,
        }
    ).encode("utf-8")


def _head_json(head: RunHead) -> str:
    ref = head.ref
    return canonical_json(
        {
            "plan_id": ref.plan_id,
            "backend": ref.backend.value,
            "operation": ref.operation.value,
            "run_dir": str(ref.run_dir),
            "stage": _stage_json(head.stage),
            "restart_count": head.restart_count,
            "cancel_requested": head.cancel_requested,
            "metrics_offset": head.metrics_offset,
            "metrics_sealed": head.metrics_sealed,
        }
    )


def _stage_json(stage: Stage) -> dict[str, object]:
    match stage:
        case Requested():
            return {"name": "requested"}
        case Claimed():
            return {"name": "claimed"}
        case Live():
            return {
                "name": "live",
                "handle": handle_payload(stage.handle),
                "phase": stage.phase,
                "native_state": stage.native_state,
                "reason": stage.reason,
                "lost": stage.lost,
                "cancel_delivered": stage.cancel_delivered,
            }
        case Ended(exit=ending, handle=handle):
            return {
                "name": "ended",
                "exit": ending.payload(),
                "handle": None if handle is None else handle_payload(handle),
            }
    raise AssertionError(stage)


def _parse_stage(value: dict) -> Stage:
    match value:
        case {"name": "requested"}:
            return Requested()
        case {"name": "claimed"}:
            return Claimed()
        case {"name": "live", "handle": handle, "phase": phase}:
            return Live(
                parse_handle(handle),
                None if phase is None else Phase(phase),
                value["native_state"],
                value["reason"],
                value["lost"],
                value["cancel_delivered"],
            )
        case {"name": "ended", "exit": ending, "handle": handle}:
            return Ended(
                Exited.from_payload(ending, None),
                None if handle is None else parse_handle(handle),
            )
    raise ValueError(f"Unknown run stage {value!r}")


def _head(run_id: RunId, row: sqlite3.Row) -> RunHead:
    value = json.loads(row["head_json"])
    ref = RunRef(
        run_id,
        PlanId(value["plan_id"]),
        BackendKind(value["backend"]),
        Operation(value["operation"]),
        Path(value["run_dir"]),
    )
    return RunHead(
        ref,
        row["attempt"],
        row["seq"],
        row["revision"],
        _parse_stage(value["stage"]),
        value["restart_count"],
        value["cancel_requested"],
        value["metrics_offset"],
        value["metrics_sealed"],
    )
