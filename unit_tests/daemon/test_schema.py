import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl_daemon.storage.schema import (
    Migration,
    MigrationFailed,
    applied_migrations,
    migrate,
    open_read_only,
    restore_backup,
    schema_version,
)


def add_table(name):
    return lambda connection: connection.execute(f"CREATE TABLE {name}(run_id TEXT)")


def fresh(tmp_path):
    path = tmp_path / "registry.db"
    sqlite3.connect(path).close()
    return path


def tables(path):
    with sqlite3.connect(path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }


def test_a_migration_is_applied_exactly_once(tmp_path):
    path = fresh(tmp_path)
    migration = Migration("0001_add_tags", 1, add_table("tags"))

    assert migrate(path, [migration]) == ["0001_add_tags"]
    assert migrate(path, [migration]) == []
    assert migrate(path, [migration]) == []

    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 1
        assert applied_migrations(connection) == {"0001_add_tags"}
    assert "tags" in tables(path)


def test_migrations_apply_in_version_order(tmp_path):
    path = fresh(tmp_path)
    order = []
    migrations = [
        Migration("0002_second", 2, lambda c: order.append(2)),
        Migration("0001_first", 1, lambda c: order.append(1)),
    ]

    assert migrate(path, migrations) == ["0001_first", "0002_second"]
    assert order == [1, 2]


def test_a_backward_migration_is_refused(tmp_path):
    path = fresh(tmp_path)
    migrate(path, [Migration("0001_first", 2, add_table("a"))])

    with pytest.raises(MigrationFailed, match="not ahead of"):
        migrate(path, [Migration("0002_backward", 1, add_table("b"))])

    assert "b" not in tables(path)


def test_a_failing_migration_rolls_back_completely(tmp_path):
    path = fresh(tmp_path)

    def half_then_fail(connection):
        connection.execute("CREATE TABLE partial(run_id TEXT)")
        raise RuntimeError("boom")

    with pytest.raises(MigrationFailed, match="rolled back"):
        migrate(path, [Migration("0001_broken", 1, half_then_fail)])

    assert "partial" not in tables(path)
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 0
        assert applied_migrations(connection) == set()


VICTIM = """
import sys, time
sys.path.insert(0, {src!r})
from wavedl_daemon.storage.schema import Migration, migrate

def slow(connection):
    connection.execute("CREATE TABLE partial(run_id TEXT)")
    print("mid-migration", flush=True)
    time.sleep(300)

migrate({path!r}, [Migration("0001_slow", 1, slow)])
"""


def test_an_interrupted_migration_leaves_the_database_unchanged(
    tmp_path, paused_process
):
    """SIGKILL runs no rollback code; SQLite's own recovery must do it."""
    path = fresh(tmp_path)
    script = VICTIM.format(src=str(Path("src").resolve()), path=str(path))
    with paused_process(script, "mid-migration") as victim:
        victim.kill()
        victim.wait(timeout=10)

    assert "partial" not in tables(path)
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 0
        assert applied_migrations(connection) == set()


def test_backup_restores_a_completed_migration(tmp_path):
    """The transaction cannot undo a migration that succeeded and was wrong."""
    path = fresh(tmp_path)
    migrate(path, [Migration("0001_regrettable", 1, add_table("regret"))])
    assert "regret" in tables(path)

    restore_backup(path, "0001_regrettable")

    assert "regret" not in tables(path)
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 0


def test_restoring_without_a_backup_is_refused(tmp_path):
    with pytest.raises(MigrationFailed, match="no backup"):
        restore_backup(fresh(tmp_path), "0001_never_ran")


def test_read_only_recovery_start_cannot_write(tmp_path):
    """A schema newer than this build is reportable, not writable."""
    path = fresh(tmp_path)
    migrate(path, [Migration("0001_add_tags", 1, add_table("tags"))])

    connection = open_read_only(path)
    try:
        assert schema_version(connection) == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE nope(x INT)")
    finally:
        connection.close()


def test_the_daemon_never_imports_the_gui_database(tmp_path):
    """The legacy table is deleted by T41d, not imported. Enforce it."""
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"wavedl_gui", "PySide6", "torch", "numpy"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from wavedl_daemon.storage import ownership, registry, schema, settings
assert schema.MIGRATIONS == ()
"""
    import os

    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_backup_and_restore_include_committed_wal_frames(tmp_path):
    from wavedl_daemon.storage.schema import backup_path

    path = fresh(tmp_path)
    # Keep the original connection open so SQLite cannot checkpoint/delete WAL
    # on its final close. A file-copy backup would omit the committed row.
    writer = sqlite3.connect(path, isolation_level=None)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE original(value TEXT)")
        writer.execute("INSERT INTO original VALUES ('before')")
        assert Path(f"{path}-wal").stat().st_size > 0
        migrate(path, [Migration("0001_added", 1, add_table("added"))])
        with sqlite3.connect(backup_path(path, "0001_added")) as backup:
            assert backup.execute("SELECT * FROM original").fetchall() == [("before",)]
        writer.execute("INSERT INTO original VALUES ('after')")
        restore_backup(path, "0001_added")
        assert writer.execute("SELECT * FROM original").fetchall() == [("before",)]
        assert "added" not in tables(path)
        assert schema_version(writer) == 0
    finally:
        writer.close()


@pytest.mark.parametrize("escape", ["COMMIT", "script"])
def test_migration_callback_cannot_escape_its_transaction(tmp_path, escape):
    path = fresh(tmp_path)

    def unsafe(connection):
        connection.execute("CREATE TABLE partial(x INT)")
        if escape == "script":
            connection.executescript("CREATE TABLE escaped(x INT);")
        else:
            connection.execute(escape)

    with pytest.raises(MigrationFailed, match="rolled back"):
        migrate(path, [Migration("0001_unsafe", 1, unsafe)])
    assert "partial" not in tables(path)
    assert "escaped" not in tables(path)
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 0
        assert applied_migrations(connection) == set()


def test_migration_and_restore_refuse_active_dispatch_owner(tmp_path):
    from wavedl_daemon.storage.ownership import DispatchHeld, acquire_dispatch_lock

    path = fresh(tmp_path)
    migrate(path, [Migration("0001_added", 1, add_table("added"))])
    with acquire_dispatch_lock(path.parent / "dispatch.lock"):
        with pytest.raises(DispatchHeld):
            migrate(path, [Migration("0002_more", 2, add_table("more"))])
        with pytest.raises(DispatchHeld):
            restore_backup(path, "0001_added")
    assert "more" not in tables(path)
    assert "added" in tables(path)


def test_read_only_migration_queries_do_not_create_bookkeeping(tmp_path):
    path = fresh(tmp_path)
    connection = open_read_only(path)
    try:
        assert applied_migrations(connection) == set()
    finally:
        connection.close()
    assert tables(path) == set()


@pytest.mark.parametrize("identity", ["", "../escape", "sub/path"])
def test_migration_identity_cannot_escape_backup_directory(identity):
    with pytest.raises(ValueError, match="identity"):
        Migration(identity, 1, lambda c: None)


def test_reusing_migration_identity_for_different_version_is_refused(tmp_path):
    path = fresh(tmp_path)
    migrate(path, [Migration("0001_added", 1, add_table("added"))])
    with pytest.raises(MigrationFailed, match="identity changed"):
        migrate(path, [Migration("0001_added", 2, add_table("changed"))])
    assert "changed" not in tables(path)
