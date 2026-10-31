# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Telemetry projections alongside authoritative, durable plan/control records.

Runtime journals can rebuild only telemetry projections and ingestion cursors.
Plans, idempotency identities, and pre-runtime control intent are authoritative
SQLite records, restored from backups/operator recovery rather than telemetry.
``rebuild_from_journals`` never deletes or rewrites those control records.

The projection and the cursor that records how far it has consumed are written
in ONE transaction. That is the whole durability argument: a crash cannot leave
the projection ahead of the cursor (double-applying on restart) or behind it
(silently dropping events), because neither is committed without the other.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from wavedl.core.generations import PublishedGeneration, published_generations
from wavedl.core.journal_reader import JournalReader
from wavedl.core.layout import RunLayout
from wavedl.core.manifest import describes_layout, read_manifest
from wavedl.core.persistence import write_once
from wavedl.core.protocol import ProtocolError
from wavedl_daemon.storage.collections import COLLECTION_MIGRATION
from wavedl_daemon.storage.event_log import EVENT_LOG_MIGRATION, append_in_transaction
from wavedl_daemon.storage.executions import EXECUTION_MIGRATION
from wavedl_daemon.storage.metric_projection import (
    METRIC_PROJECTION_MIGRATION,
    acknowledge_generation,
    apply_metric_event,
    read_metrics,
)
from wavedl_daemon.storage.native_heads import NATIVE_HEAD_MIGRATION
from wavedl_daemon.storage.native_lifecycle import NATIVE_LIFECYCLE_MIGRATION
from wavedl_daemon.storage.ownership import (
    DispatchRequired,
    acquire_dispatch_lock,
    require_dispatch,
)
from wavedl_daemon.storage.plans import PLAN_MIGRATION, PlanRecoveryRequired
from wavedl_daemon.storage.requeues import REQUEUE_MIGRATION
from wavedl_daemon.storage.runtime_provenance import RUNTIME_PROVENANCE_MIGRATION
from wavedl_daemon.storage.schema import (
    MigrationFailed,
    migrate,
    open_read_only,
    schema_version,
)
from wavedl_daemon.storage.submission_collections import SUBMISSION_COLLECTION_MIGRATION
from wavedl_daemon.storage.submission_generations import SUBMISSION_GENERATION_MIGRATION
from wavedl_daemon.storage.submissions import SUBMISSION_MIGRATION


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from wavedl.core.protocol import Envelope
    from wavedl_daemon.storage.ownership import DispatchLock


SCHEMA_VERSION = 13

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    operation     TEXT NOT NULL DEFAULT '',
    state         TEXT NOT NULL DEFAULT 'unknown',
    attempt       INTEGER NOT NULL DEFAULT 0,
    event_count   INTEGER NOT NULL DEFAULT 0,
    last_type     TEXT NOT NULL DEFAULT '',
    updated_at    TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS cursors (
    run_id  TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    seq     INTEGER NOT NULL,
    PRIMARY KEY (run_id, attempt)
);
CREATE INDEX IF NOT EXISTS runs_by_state ON runs (state, updated_at DESC);
"""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    operation: str = ""
    state: str = "unknown"
    attempt: int = 0
    event_count: int = 0
    last_type: str = ""
    updated_at: str = ""


_RUN_COLUMNS = ", ".join(RunRecord.__dataclass_fields__)


def default_projection(record: RunRecord | None, event: Envelope) -> RunRecord:
    """Build the registry's lightweight run record from an event."""
    current = record or RunRecord(run_id=event.run_id)
    state = current.state
    if event.type == "hello":
        state = "running"
    elif event.type == "state":
        state = str(event.payload.get("state", state))
    elif event.type == "exit":
        state = str(event.payload.get("status", event.payload.get("state", "exited")))
    if event.attempt < current.attempt:
        return replace(current, event_count=current.event_count + 1)
    return replace(
        current,
        operation=str(event.payload.get("operation", current.operation)),
        state=state,
        attempt=max(current.attempt, event.attempt),
        event_count=current.event_count + 1,
        last_type=event.type,
        updated_at=event.ts,
    )


class Registry:
    """Owns one SQLite file. Not shared between processes; T8b gates writers."""

    def __init__(
        self,
        path: Path,
        *,
        read_only: bool = False,
        dispatch_lock: DispatchLock | None = None,
    ):
        self.path = Path(path).resolve()
        self.read_only = read_only
        self.rebuild_diagnostics: deque[tuple[Path, str]] = deque(maxlen=16)
        if read_only:
            self._connection = open_read_only(self.path)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.parent / "dispatch.lock"
        if dispatch_lock is not None:
            require_dispatch(dispatch_lock)
            if dispatch_lock.path.resolve() != lock_path:
                raise DispatchRequired("dispatch ownership is for a different registry")
        ownership = (
            contextlib.nullcontext(dispatch_lock)
            if dispatch_lock is not None
            else acquire_dispatch_lock(lock_path)
        )
        with ownership as lock:
            self._initialize(lock)
        self._connection = sqlite3.connect(self.path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")

    def _initialize(self, lock: DispatchLock) -> None:
        marker = self.path.with_name(f".{self.path.name}.initialized")
        marked = marker.exists()
        if marked and not self.path.is_file():
            raise PlanRecoveryRequired(
                "Initialized intent database is missing; restore backup before dispatch"
            )
        if not self.path.exists():
            # Publish before SQLite creates anything. Death between marker and
            # creation intentionally requires recovery instead of silently
            # assigning new request identities after lost authoritative state.
            self._mark_initialized(marker)
        connection = sqlite3.connect(self.path, isolation_level=None)
        try:
            version = schema_version(connection)
            if version > SCHEMA_VERSION:
                raise MigrationFailed(
                    f"Schema {version} is newer than supported {SCHEMA_VERSION}; "
                    "open the registry in read-only recovery mode"
                )
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if version == 0 and tables:
                raise MigrationFailed(
                    "Unversioned existing database refused; no legacy import is supported"
                )
            if marked and version == 0:
                raise PlanRecoveryRequired(
                    "Initialized intent database was replaced or initialization "
                    "was interrupted; recovery is required"
                )
            if not marker.exists():
                self._mark_initialized(marker)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute("PRAGMA user_version=1")
                connection.execute("COMMIT")
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()
        migrate(
            self.path,
            [
                PLAN_MIGRATION,
                EXECUTION_MIGRATION,
                RUNTIME_PROVENANCE_MIGRATION,
                METRIC_PROJECTION_MIGRATION,
                EVENT_LOG_MIGRATION,
                REQUEUE_MIGRATION,
                COLLECTION_MIGRATION,
                SUBMISSION_MIGRATION,
                SUBMISSION_COLLECTION_MIGRATION,
                SUBMISSION_GENERATION_MIGRATION,
                NATIVE_HEAD_MIGRATION,
                NATIVE_LIFECYCLE_MIGRATION,
            ],
            dispatch_lock=lock,
        )

    def _mark_initialized(self, marker: Path) -> None:
        write_once(
            marker,
            json.dumps(
                {"database": self.path.name, "registry_id": str(uuid4())},
                sort_keys=True,
            ).encode("utf-8"),
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- reads ---------------------------------------------------------------

    def cursor(self, run_id: str, attempt: int) -> int:
        """Highest applied seq for THIS attempt; -1 when nothing has been applied.

        Cursors are per attempt because a requeued Slurm job reuses its job id
        and restarts seq at zero. A per-run cursor would discard the new
        attempt's early events as already seen.
        """
        row = self._connection.execute(
            "SELECT seq FROM cursors WHERE run_id = ? AND attempt = ?",
            (run_id, attempt),
        ).fetchone()
        return -1 if row is None else int(row["seq"])

    def get(self, run_id: str) -> RunRecord | None:
        row = self._connection.execute(
            f"SELECT {_RUN_COLUMNS} FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else RunRecord(**dict(row))

    def list_runs(
        self, *, state: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[RunRecord]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if state is None:
            rows = self._connection.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs "
                "ORDER BY updated_at DESC, run_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            rows = self._connection.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE state = ? "
                "ORDER BY updated_at DESC, run_id LIMIT ? OFFSET ?",
                (state, limit, offset),
            )
        return [RunRecord(**dict(row)) for row in rows]

    # -- writes --------------------------------------------------------------

    def metrics(self, run_id, *, active_only=True, limit=1000, offset=0):
        """Bounded active history, or audit history including abandoned points."""
        return read_metrics(
            self._connection,
            run_id,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    def lifecycle_events(self, run_id: str, attempt: int):
        """Fold only state-bearing events, not large metric/history payloads."""
        from wavedl.core.protocol import decode

        if type(attempt) is not int or attempt < 1:
            raise ValueError("Attempt must be positive")
        events = []
        for kind in ("hello", "state"):
            row = self._connection.execute(
                "SELECT event_json FROM event_log WHERE run_id=? AND attempt=? "
                "AND event_type=? ORDER BY seq DESC LIMIT 1",
                (run_id, attempt, kind),
            ).fetchone()
            if row is not None:
                events.append(decode(row[0]))
        exits = self._connection.execute(
            "SELECT event_json FROM event_log WHERE run_id=? AND attempt=? "
            "AND event_type='exit' GROUP BY json_extract(event_json,'$.payload.code'), "
            "json_extract(event_json,'$.payload.status') ORDER BY MAX(seq) DESC LIMIT 2",
            (run_id, attempt),
        )
        events.extend(decode(row[0]) for row in exits)
        return iter(sorted(events, key=lambda event: event.seq))

    def acknowledge_metric_generation(
        self, layout: RunLayout, verified_generation: PublishedGeneration
    ):
        """Accept only a generation reverified against this run's publication."""
        if verified_generation not in published_generations(layout):
            raise ProtocolError(
                "Metric boundary is not a verified published generation"
            )
        with self._transaction():
            acknowledge_generation(
                self._connection,
                layout.run_id,
                verified_generation.attempt,
                verified_generation.step,
                verified_generation.generation,
            )

    def apply_events(
        self,
        run_id: str,
        attempt: int,
        events: Iterable[Envelope],
    ) -> int:
        """Fold events into the projection and advance the cursor, atomically.

        Events at or below the stored cursor are skipped, so re-delivering a
        batch after a lost response changes nothing. Returns the new cursor.
        """
        batch: Sequence[Envelope] = list(events)
        previous = -1
        for event in batch:
            if event.run_id != run_id or event.attempt != attempt:
                raise ProtocolError(
                    f"event {event.run_id}/{event.attempt} does not belong to "
                    f"{run_id}/{attempt}"
                )
            if event.seq <= previous:
                raise ProtocolError("Sequence must increase within a batch")
            previous = event.seq

        with self._transaction():
            return self._fold_events(run_id, attempt, batch)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        if self.read_only:
            raise PermissionError("Registry is in read-only recovery mode")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _fold_events(self, run_id, attempt, batch, operation: str | None = None) -> int:
        position = self.cursor(run_id, attempt)
        record = self.get(run_id)
        applied = position
        for event in batch:
            append_in_transaction(self._connection, event)
            if event.seq <= position:
                continue
            record = default_projection(record, event)
            if operation is not None:
                record = replace(record, operation=operation)
            apply_metric_event(self._connection, event)
            applied = event.seq
        if applied != position:
            self._write_record(record)
            self._write_cursor(run_id, attempt, applied)
        return applied

    def _write_record(self, record: RunRecord) -> None:
        self._connection.execute(
            "INSERT INTO runs (run_id, operation, state, attempt, event_count, "
            "last_type, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET operation = excluded.operation, "
            "state = excluded.state, attempt = excluded.attempt, "
            "event_count = excluded.event_count, last_type = excluded.last_type, "
            "updated_at = excluded.updated_at",
            (
                record.run_id,
                record.operation,
                record.state,
                record.attempt,
                record.event_count,
                record.last_type,
                record.updated_at,
            ),
        )

    def _write_cursor(self, run_id: str, attempt: int, seq: int) -> None:
        self._connection.execute(
            "INSERT INTO cursors (run_id, attempt, seq) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, attempt) DO UPDATE SET seq = excluded.seq",
            (run_id, attempt, seq),
        )

    # -- recovery ------------------------------------------------------------

    def rebuild_from_journals(self, run_root: Path) -> int:
        """Discard the projection and refold it from the journals.

        Rebuilds telemetry only, never submission intent/idempotency data. A
        storage error restores the previous projection. Damaged attempts keep
        their good prefix and a bounded diagnostic. Returns attempts folded.
        """
        run_root = Path(run_root)
        if not run_root.is_dir():
            raise FileNotFoundError(run_root)
        with self._transaction():
            self._connection.execute("DELETE FROM runs")
            self._connection.execute("DELETE FROM cursors")
            self._connection.execute("DELETE FROM metric_history")
            self._connection.execute("DELETE FROM event_log")
            self._connection.execute("DELETE FROM metric_attempts")
            self._connection.execute("DELETE FROM metric_generations")
            return self._rebuild(run_root)

    def _rebuild(self, run_root: Path) -> int:
        self.rebuild_diagnostics.clear()
        folded = 0
        journals = []
        for journal in Path(run_root).glob("*/attempts/*/events.jsonl"):
            run_id = journal.parents[2].name
            try:
                attempt = int(journal.parents[0].name)
            except ValueError:
                continue
            if attempt > 0:
                journals.append((run_id, attempt, journal))
        for run_id, attempt, journal in sorted(journals):
            operation = None
            manifest_path = journal.parents[2] / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = read_manifest(manifest_path)
                    layout = RunLayout(journal.parents[2], run_id)
                    if not describes_layout(manifest, layout):
                        raise ValueError("Manifest/layout identity mismatch")
                    operation = manifest.operation.value
                except (TypeError, ValueError, KeyError, RecursionError) as exc:
                    self.rebuild_diagnostics.append((manifest_path, str(exc)[:160]))
                    continue

            reader = JournalReader(journal, run_id=run_id, attempt=attempt)
            if operation is not None:
                for generation in published_generations(layout):
                    acknowledge_generation(
                        self._connection,
                        run_id,
                        generation.attempt,
                        generation.step,
                        generation.generation,
                    )
            position = -1
            while True:
                try:
                    # Reading one record preserves every committed good prefix
                    # even when the next line violates the envelope schema.
                    events = reader.read(max_records=1)
                except ProtocolError as exc:
                    self.rebuild_diagnostics.append((journal, str(exc)[:160]))
                    break  # a damaged tail stops this attempt, not the rebuild
                if not events:
                    # A bounded poll may not yet contain a full record. Only
                    # stop at an unchanged offset, not at the first empty batch.
                    if reader.offset == position:
                        break
                    position = reader.offset
                    continue
                self._connection.execute("SAVEPOINT rebuild_event")
                try:
                    self._fold_events(run_id, attempt, events, operation)
                except ProtocolError as exc:
                    self._connection.execute("ROLLBACK TO rebuild_event")
                    self.rebuild_diagnostics.append((journal, str(exc)[:160]))
                    break
                finally:
                    self._connection.execute("RELEASE rebuild_event")
            if reader.malformed_count:
                self.rebuild_diagnostics.append(
                    (journal, f"Skipped {reader.malformed_count} malformed records")
                )
            if self.cursor(run_id, attempt) >= 0:
                folded += 1
        return folded
