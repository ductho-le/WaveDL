# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Open the daemon database at schema 1, created fresh.

Databases from earlier daemons are refused, never imported: their runs were
written by a trainer protocol that no longer exists.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch
from wavedl_daemon.storage.run_log import RUN_LOG_SCHEMA


if TYPE_CHECKING:
    from wavedl_daemon.storage.ownership import DispatchLock


SCHEMA_VERSION = 1

_PLANS_SCHEMA = (
    "CREATE TABLE plans ("
    "plan_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, "
    "request_id TEXT NOT NULL UNIQUE, idempotency_key TEXT NOT NULL UNIQUE, "
    "request_json TEXT NOT NULL, plan_json TEXT NOT NULL, created_at TEXT NOT NULL)",
    "CREATE TRIGGER plans_immutable_update BEFORE UPDATE ON plans "
    "BEGIN SELECT RAISE(ABORT, 'persisted plan is immutable'); END",
    "CREATE TRIGGER plans_immutable_delete BEFORE DELETE ON plans "
    "BEGIN SELECT RAISE(ABORT, 'persisted plan is immutable'); END",
)
_TABLES = frozenset({"plans", "run_events", "run_heads"})


class UnsupportedDatabase(RuntimeError):
    """The file is not a schema 1 daemon database."""


def _check(connection: sqlite3.Connection, path: Path) -> bool:
    """True for an empty file, False for schema 1; anything else is refused."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if version == 0 and not tables:
        return True
    if version != SCHEMA_VERSION or not _TABLES <= tables:
        raise UnsupportedDatabase(
            f"{path} is not a schema {SCHEMA_VERSION} daemon database; "
            "move it aside to start fresh"
        )
    return False


def open_database(path: Path, *, dispatch_lock: DispatchLock) -> sqlite3.Connection:
    path = Path(path).resolve()
    require_dispatch(dispatch_lock)
    if dispatch_lock.path.resolve() != path.parent / "dispatch.lock":
        raise DispatchRequired("dispatch ownership is for a different database")
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.row_factory = sqlite3.Row
        # Refuse before any PRAGMA can change a file this daemon does not own.
        fresh = _check(connection, path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        if fresh:
            connection.execute("BEGIN IMMEDIATE")
            if not _check(connection, path):
                raise UnsupportedDatabase(f"{path} changed while it was created")
            for statement in (*_PLANS_SCHEMA, *RUN_LOG_SCHEMA):
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        connection.close()
        raise
    return connection
