# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""SQLite metric telemetry fold; its caller owns the cursor transaction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from wavedl.core.events import ArtifactPayload, SupersedePayload
from wavedl.core.metric_history import MetricRecord, checked_metric, validate_boundary
from wavedl.core.protocol import ProtocolError
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


if TYPE_CHECKING:
    import sqlite3

    from wavedl.core.protocol import Envelope


def _create_tables(connection: sqlite3.Connection) -> None:
    if schema_version(connection) != 4:
        raise MigrationFailed("Metric projection requires schema 4")
    for statement in (
        "CREATE TABLE metric_history (run_id TEXT NOT NULL, attempt INTEGER NOT NULL, "
        "seq INTEGER NOT NULL, step INTEGER NOT NULL, name TEXT NOT NULL, value REAL, "
        "unit TEXT NOT NULL, metrics_json TEXT NOT NULL, source_generation TEXT, "
        "active INTEGER NOT NULL, PRIMARY KEY(run_id,attempt,seq))",
        "CREATE INDEX metric_history_active ON metric_history(run_id,active,step)",
        "CREATE INDEX metric_history_identity ON metric_history(run_id,step,name,attempt,seq)",
        "CREATE INDEX metric_history_boundary ON metric_history(run_id,attempt,step)",
        "CREATE TABLE metric_attempts (run_id TEXT NOT NULL, attempt INTEGER NOT NULL, "
        "source_generation TEXT, source_attempt INTEGER, restore_step INTEGER, restore_seq INTEGER, "
        "PRIMARY KEY(run_id,attempt))",
        "CREATE TABLE metric_generations (run_id TEXT NOT NULL, attempt INTEGER NOT NULL, "
        "step INTEGER NOT NULL, generation TEXT NOT NULL, "
        "PRIMARY KEY(run_id,attempt,step,generation))",
    ):
        connection.execute(statement)


METRIC_PROJECTION_MIGRATION = Migration("0005_metric_projection", 5, _create_tables)


def acknowledge_generation(connection, run_id, source_attempt, step, generation):
    """Record independently verified publication, never journal artifact claims."""
    payload = SupersedePayload.from_dict(
        {"source_attempt": source_attempt, "step": step, "generation": generation}
    )
    connection.execute(
        "INSERT OR IGNORE INTO metric_generations VALUES (?,?,?,?)",
        (run_id, payload.source_attempt, payload.step, payload.generation),
    )


def apply_metric_event(connection: sqlite3.Connection, event: Envelope) -> None:
    run_id, attempt = event.run_id, event.attempt
    if event.type == "metric":
        payload = checked_metric(event.payload)
        row = connection.execute(
            "SELECT source_generation FROM metric_attempts WHERE run_id=? AND attempt=?",
            (run_id, attempt),
        ).fetchone()
        head = connection.execute(
            "SELECT max(attempt) FROM metric_attempts WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        active = head is None or attempt >= head
        if active:
            connection.execute(
                "UPDATE metric_history SET active=0 WHERE run_id=? AND step=? AND name=?",
                (run_id, payload.step, payload.name),
            )
        connection.execute(
            "INSERT INTO metric_history VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                attempt,
                event.seq,
                payload.step,
                payload.name,
                payload.value,
                payload.unit,
                json.dumps(payload.metrics, allow_nan=False),
                None if row is None else row[0],
                int(active),
            ),
        )
    elif event.type == "supersede":
        payload = SupersedePayload.from_dict(event.payload)
        boundary = connection.execute(
            "SELECT 1 FROM metric_history WHERE run_id=? AND attempt=? AND step=? "
            "AND (? IS NULL OR seq<=?) LIMIT 1",
            (
                run_id,
                payload.source_attempt,
                payload.step,
                payload.source_seq,
                payload.source_seq,
            ),
        ).fetchone()
        known_steps = (
            set() if boundary is None else {(payload.source_attempt, payload.step)}
        )
        known_attempts = {
            row[0]
            for row in connection.execute(
                "SELECT attempt FROM metric_attempts WHERE run_id=?", (run_id,)
            )
        }
        published = {
            tuple(row)
            for row in connection.execute(
                "SELECT attempt,step,generation FROM metric_generations WHERE run_id=?",
                (run_id,),
            )
        }
        validate_boundary(event, payload, known_steps, known_attempts, published)
        lineage = {
            row[0]: (row[1], row[2], row[3])
            for row in connection.execute(
                "SELECT attempt,source_attempt,restore_step,restore_seq FROM metric_attempts WHERE run_id=?",
                (run_id,),
            )
        }
        if lineage.get(attempt, (None, None, None))[0] is not None:
            raise ProtocolError("Attempt already has a supersession boundary")
        connection.execute(
            "INSERT INTO metric_attempts VALUES (?,?,?,?,?,?) ON CONFLICT(run_id,attempt) "
            "DO UPDATE SET source_generation=excluded.source_generation, "
            "source_attempt=excluded.source_attempt, restore_step=excluded.restore_step, "
            "restore_seq=excluded.restore_seq",
            (
                run_id,
                attempt,
                payload.generation,
                payload.source_attempt,
                payload.step,
                payload.source_seq,
            ),
        )
        if attempt >= max(lineage, default=0):
            connection.execute(
                "UPDATE metric_history SET active=0 WHERE run_id=?", (run_id,)
            )
            current, bound, source_seq = (
                payload.source_attempt,
                payload.step,
                payload.source_seq,
            )
            while True:
                connection.execute(
                    "UPDATE metric_history SET active=1 WHERE run_id=? AND attempt=? AND step<=? "
                    "AND (? IS NULL OR seq<=?)",
                    (run_id, current, bound, source_seq, source_seq),
                )
                parent, restored_step, source_seq = lineage.get(
                    current, (None, None, None)
                )
                if parent is None:
                    break
                current, bound = parent, min(bound, restored_step)
            connection.execute(
                "UPDATE metric_history AS m SET active=0 WHERE m.run_id=? AND m.active=1 "
                "AND EXISTS (SELECT 1 FROM metric_history AS newer WHERE newer.run_id=m.run_id "
                "AND newer.active=1 AND newer.step=m.step AND newer.name=m.name "
                "AND (newer.attempt>m.attempt OR (newer.attempt=m.attempt AND newer.seq>m.seq)))",
                (run_id,),
            )
        return
    elif event.type == "artifact":
        if ArtifactPayload.from_dict(event.payload).kind != "checkpoint":
            return
    elif event.type != "hello":
        return
    connection.execute(
        "INSERT OR IGNORE INTO metric_attempts (run_id,attempt) VALUES (?,?)",
        (run_id, attempt),
    )


def read_metrics(connection, run_id, *, active_only=True, limit=1000, offset=0):
    if type(active_only) is not bool:
        raise ValueError("active_only must be a boolean")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    rows = connection.execute(
        "SELECT * FROM metric_history WHERE run_id=? "
        + ("AND active=1 " if active_only else "")
        + "ORDER BY step,attempt,seq LIMIT ? OFFSET ?",
        (run_id, limit, offset),
    )
    return [
        MetricRecord(
            row["run_id"],
            row["attempt"],
            row["seq"],
            row["step"],
            row["name"],
            row["value"],
            row["unit"],
            json.loads(row["metrics_json"]),
            row["source_generation"],
            bool(row["active"]),
        )
        for row in rows
    ]
