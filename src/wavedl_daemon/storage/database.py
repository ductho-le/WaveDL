"""
Opens the daemon database at schema 1, created fresh.

Databases from earlier daemons are refused, not imported, because their runs
came from a trainer protocol that no longer exists.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from secrets import token_hex
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
    version_row = connection.execute("PRAGMA user_version").fetchone()
    if version_row is None:
        raise UnsupportedDatabase(f"{path} has no schema version")
    version = version_row[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if version == 0 and not tables:
        return True
    if version != SCHEMA_VERSION or not tables >= _TABLES:
        raise UnsupportedDatabase(
            f"{path} is not a schema {SCHEMA_VERSION} daemon database; "
            "move it aside to start fresh"
        )
    return False


def database_generation(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT generation FROM database_identity WHERE singleton=1"
    ).fetchone()
    if row is None:
        raise UnsupportedDatabase("database generation is missing")
    generation = row[0]
    if not isinstance(generation, str) or not generation:
        raise UnsupportedDatabase("database generation is invalid")
    return generation


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
        connection.execute("BEGIN IMMEDIATE")
        if fresh:
            if not _check(connection, path):
                raise UnsupportedDatabase(f"{path} changed while it was created")
            for statement in (*_PLANS_SCHEMA, *RUN_LOG_SCHEMA):
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        # Supplemental metadata lets existing schema 1 databases acquire an
        # identity without changing or invalidating their run/plan schema.
        connection.execute(
            "CREATE TABLE IF NOT EXISTS database_identity ("
            "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
            "generation TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO database_identity (singleton, generation) "
            "VALUES (1, ?)",
            (token_hex(32),),
        )
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        connection.close()
        raise
    return connection
