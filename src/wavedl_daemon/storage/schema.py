# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Forward-only migrations and consistent SQLite recovery snapshots.

Only the daemon schema is handled here; legacy Qt download tables are not
imported. Dispatch ownership excludes external work while a schema changes.
DDL, migration identity, and user_version commit together. SQLite's backup API
includes committed WAL frames; copying the main database file does not.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.persistence import sync_directory
from wavedl_daemon.storage.ownership import (
    DispatchRequired,
    acquire_dispatch_lock,
    require_dispatch,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from wavedl_daemon.storage.ownership import DispatchLock


class MigrationFailed(RuntimeError):
    """A schema operation failed; recovery may require operator intervention."""


_BOOKKEEPING = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    to_version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


@dataclass(frozen=True)
class Migration:
    """One forward step. Callbacks must not commit or use executescript."""

    migration_id: str
    to_version: int
    apply: Callable[[sqlite3.Connection], None]

    def __post_init__(self):
        if not isinstance(self.migration_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", self.migration_id
        ):
            raise ValueError("Invalid migration identity")
        if type(self.to_version) is not int or self.to_version < 1:
            raise ValueError("Migration version must be a positive integer")


MIGRATIONS: tuple[Migration, ...] = ()
"""Future forward changes; Registry creates and stamps the initial version."""


def schema_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def applied_migrations(connection: sqlite3.Connection) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if not exists:
        return set()
    return {
        row[0]
        for row in connection.execute("SELECT migration_id FROM schema_migrations")
    }


def open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def backup_path(path: Path, migration_id: str) -> Path:
    if not isinstance(migration_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", migration_id
    ):
        raise ValueError("Invalid migration identity")
    return Path(f"{path}.backup-{migration_id}")


def _copy_database(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    deadline = time.monotonic() + 5

    def progress(status, remaining, total):
        if time.monotonic() > deadline:
            raise MigrationFailed("Database backup deadline exceeded")

    source.backup(target, pages=256, progress=progress, sleep=0.02)


def _backup(connection: sqlite3.Connection, path: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        target = sqlite3.connect(temporary)
        try:
            _copy_database(connection, target)
        finally:
            target.close()
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _deny_transaction_control(action, arg1, arg2, database, trigger):
    # executescript() implicitly commits even in autocommit mode. Prohibit that
    # and explicit COMMIT/ROLLBACK so a callback cannot escape our transaction.
    if action in {sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT}:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def migrate(
    path: Path,
    migrations: Sequence[Migration] = MIGRATIONS,
    *,
    dispatch_lock: DispatchLock | None = None,
) -> list[str]:
    """Apply changes once under dispatch ownership, backing up before each step."""
    path = Path(path).resolve()
    if len({item.migration_id for item in migrations}) != len(migrations):
        raise ValueError("Duplicate migration identity")
    if len({item.to_version for item in migrations}) != len(migrations):
        raise ValueError("Duplicate migration version")
    lock_path = path.parent / "dispatch.lock"
    if dispatch_lock is not None:
        require_dispatch(dispatch_lock)
        if dispatch_lock.path.resolve() != lock_path:
            raise DispatchRequired("dispatch ownership is for a different registry")
    ownership = (
        nullcontext(dispatch_lock)
        if dispatch_lock is not None
        else acquire_dispatch_lock(lock_path)
    )
    with ownership:
        connection = sqlite3.connect(path, isolation_level=None)
        applied = []
        try:
            connection.execute("PRAGMA synchronous=FULL")
            for migration in sorted(migrations, key=lambda item: item.to_version):
                if migration.migration_id in applied_migrations(connection):
                    stored = connection.execute(
                        "SELECT to_version FROM schema_migrations WHERE migration_id=?",
                        (migration.migration_id,),
                    ).fetchone()[0]
                    if stored != migration.to_version:
                        raise MigrationFailed("Migration identity changed its version")
                    continue
                if migration.to_version <= schema_version(connection):
                    raise MigrationFailed(
                        f"{migration.migration_id} targets version "
                        f"{migration.to_version}, not ahead of "
                        f"{schema_version(connection)}"
                    )
                _backup(connection, backup_path(path, migration.migration_id))
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(_BOOKKEEPING)
                    connection.set_authorizer(_deny_transaction_control)
                    try:
                        migration.apply(connection)
                    finally:
                        connection.set_authorizer(None)
                    connection.execute(
                        "INSERT INTO schema_migrations (migration_id, to_version) "
                        "VALUES (?, ?)",
                        (migration.migration_id, migration.to_version),
                    )
                    connection.execute(f"PRAGMA user_version = {migration.to_version}")
                    connection.execute("COMMIT")
                except BaseException as exc:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise MigrationFailed(
                        f"{migration.migration_id} failed and was rolled back: {exc}"
                    ) from exc
                applied.append(migration.migration_id)
        finally:
            connection.close()
        return applied


def restore_backup(path: Path, migration_id: str) -> None:
    """Restore through SQLite, preserving WAL consistency. All writers must stop."""
    path = Path(path).resolve()
    backup = backup_path(path, migration_id)
    if not backup.is_file():
        raise MigrationFailed(f"no backup for {migration_id}: {backup}")
    with acquire_dispatch_lock(path.parent / "dispatch.lock"):
        source = open_read_only(backup)
        target = sqlite3.connect(path, isolation_level=None)
        try:
            target.execute("PRAGMA synchronous=FULL")
            _copy_database(source, target)
        finally:
            target.close()
            source.close()
