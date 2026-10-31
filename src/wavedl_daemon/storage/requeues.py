# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Immutable scheduler restart bindings and archived execution generations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


if TYPE_CHECKING:
    import sqlite3


def _create_tables(connection: sqlite3.Connection) -> None:
    if schema_version(connection) != 6:
        raise MigrationFailed("Scheduler requeues require schema 6")
    connection.execute(
        "CREATE TABLE execution_attempt_history ("
        "plan_id TEXT NOT NULL REFERENCES execution_control(plan_id), "
        "attempt INTEGER NOT NULL CHECK(attempt>0), revision INTEGER NOT NULL, "
        "handle_json TEXT NOT NULL, record_json TEXT NOT NULL "
        "CHECK(length(CAST(record_json AS BLOB))<=262144), "
        "PRIMARY KEY(plan_id,attempt))"
    )
    connection.execute(
        "CREATE TABLE requeue_bindings ("
        "plan_id TEXT NOT NULL REFERENCES execution_control(plan_id), "
        "attempt INTEGER NOT NULL CHECK(attempt>0), "
        "previous_attempt INTEGER NOT NULL CHECK(previous_attempt>0), "
        "previous_revision INTEGER NOT NULL CHECK(previous_revision>=0), "
        "run_id TEXT NOT NULL, request_id TEXT NOT NULL, owner TEXT NOT NULL, "
        "cluster TEXT NOT NULL, job_id TEXT NOT NULL, "
        "restart_count INTEGER NOT NULL CHECK(restart_count>0), "
        "binding_json TEXT NOT NULL, handle_json TEXT NOT NULL, "
        "PRIMARY KEY(plan_id,attempt), "
        "UNIQUE(cluster,job_id,owner,request_id,restart_count), "
        "CHECK(attempt>previous_attempt))"
    )
    for table in ("execution_attempt_history", "requeue_bindings"):
        for operation in ("UPDATE", "DELETE"):
            connection.execute(
                f"CREATE TRIGGER {table}_no_{operation.lower()} BEFORE {operation} "
                f"ON {table} BEGIN SELECT RAISE(ABORT, 'restart evidence is immutable'); END"
            )
    connection.execute("DROP TRIGGER execution_identity_immutable")
    connection.execute(
        "CREATE TRIGGER execution_identity_immutable BEFORE UPDATE OF plan_id,owner "
        "ON execution_control BEGIN SELECT RAISE(ABORT, 'execution identity is immutable'); END"
    )
    connection.execute(
        "CREATE TRIGGER execution_attempt_activation BEFORE UPDATE OF attempt "
        "ON execution_control WHEN NEW.attempt<>OLD.attempt BEGIN "
        "SELECT RAISE(ABORT, 'restart activation lacks matching immutable evidence') "
        "WHERE NOT EXISTS (SELECT 1 FROM requeue_bindings b "
        "JOIN execution_attempt_history h ON h.plan_id=b.plan_id "
        "AND h.attempt=b.previous_attempt "
        "WHERE b.plan_id=OLD.plan_id AND b.attempt=NEW.attempt "
        "AND b.previous_attempt=OLD.attempt AND b.previous_revision=OLD.revision "
        "AND h.revision=OLD.revision AND h.handle_json=OLD.handle_json "
        "AND b.owner=OLD.owner AND b.handle_json=NEW.handle_json "
        "AND NEW.revision=OLD.revision+1 AND NEW.state='submitted' "
        "AND NEW.observation_json IS NULL AND NEW.latest_observation_json IS NULL "
        "AND NEW.cancel_delivered=0); END"
    )


REQUEUE_MIGRATION = Migration("0007_scheduler_requeues", 7, _create_tables)
