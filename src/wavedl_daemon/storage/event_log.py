# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Durable replay rows committed with projections and ingestion cursors."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from wavedl.core.protocol import Envelope, ProtocolError, decode
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


def _create_tables(connection: sqlite3.Connection) -> None:
    if schema_version(connection) != 5:
        raise MigrationFailed("Event log requires schema 5")
    connection.execute(
        "CREATE TABLE event_log (run_id TEXT NOT NULL, "
        "attempt INTEGER NOT NULL CHECK(attempt>0), "
        "seq INTEGER NOT NULL CHECK(seq>=0), event_json TEXT NOT NULL, "
        "event_type TEXT NOT NULL, "
        "PRIMARY KEY(run_id,attempt,seq))"
    )
    connection.execute(
        "CREATE INDEX event_log_lifecycle ON event_log(run_id,attempt,event_type,seq)"
    )


EVENT_LOG_MIGRATION = Migration("0006_event_log", 6, _create_tables)
MAX_RECORD_BYTES = 65536
MAX_PAGE_BYTES = 1024 * 1024


def record_bytes(event: Envelope) -> bytes:
    """Finite, canonical UTF-8 with the journal newline included in its bound."""
    try:
        raw = (
            json.dumps(
                asdict(event),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        decode(raw)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("Invalid durable event JSON") from exc
    if len(raw) > MAX_RECORD_BYTES:
        raise ProtocolError("Runtime event exceeds record byte bound")
    return raw


def append_in_transaction(connection: sqlite3.Connection, event: Envelope) -> None:
    """Never commit independently of the projection and cursor."""
    if not connection.in_transaction:
        raise RuntimeError("Event append requires the projection transaction")
    validate_cursor((event.attempt, event.seq))
    raw = record_bytes(event).decode("utf-8")
    existing = connection.execute(
        "SELECT event_json FROM event_log WHERE run_id=? AND attempt=? AND seq=?",
        (event.run_id, event.attempt, event.seq),
    ).fetchone()
    if existing:
        if record_bytes(decode(existing[0])).decode("utf-8") != raw:
            raise ProtocolError("Conflicting duplicate durable event")
        return
    connection.execute(
        "INSERT INTO event_log(run_id,attempt,seq,event_json,event_type) VALUES (?,?,?,?,?)",
        (event.run_id, event.attempt, event.seq, raw, event.type),
    )


def validate_cursor(cursor: tuple[int, int]) -> tuple[int, int]:
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


class EventLogStore:
    """Read bounded pages in numeric producer order, never timestamp order."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        info = self.path.stat()
        self._identity = info.st_dev, info.st_ino
        self._connection = sqlite3.connect(
            self.path.as_uri() + "?mode=ro", uri=True, isolation_level=None
        )
        try:
            if schema_version(self._connection) < 6:
                raise MigrationFailed("Event replay requires schema 6")
            self._connection.execute("SELECT * FROM event_log LIMIT 0")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def _guard(self):
        try:
            info = self.path.stat()
        except FileNotFoundError:
            raise MigrationFailed(
                "Event database is missing; restore recovery state"
            ) from None
        if (info.st_dev, info.st_ino) != self._identity:
            raise MigrationFailed("Event database was replaced; reopen recovery")

    def get(self, run_id: str, attempt: int, seq: int) -> Envelope | None:
        self._guard()
        validate_cursor((attempt, seq))
        row = self._connection.execute(
            "SELECT event_json FROM event_log WHERE run_id=? AND attempt=? AND seq=?",
            (run_id, attempt, seq),
        ).fetchone()
        return decode(row[0]) if row else None

    def read_after(
        self, run_id: str, cursor: tuple[int, int] = (0, -1), *, limit: int = 256
    ) -> list[Envelope]:
        self._guard()
        attempt, seq = validate_cursor(cursor)
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ValueError("Replay page must contain 1 to 1024 records")
        rows = self._connection.execute(
            "SELECT event_json FROM event_log WHERE run_id=? "
            "AND (attempt,seq)>(?,?) "
            "ORDER BY attempt,seq LIMIT ?",
            (run_id, attempt, seq, limit),
        )
        events, size = [], 0
        for row in rows:
            count = len(row[0].encode("utf-8"))
            if count > MAX_RECORD_BYTES:
                raise ProtocolError("Stored event exceeds record byte bound")
            if size + count > MAX_PAGE_BYTES:
                break
            events.append(decode(row[0]))
            size += count
        return events
