# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Mutable native lifecycle diagnostics scoped to immutable scheduler proofs."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from wavedl_daemon.backends.contracts import ObservationStatus, SubmissionObservation
from wavedl_daemon.storage.collections import _json, _strict_data
from wavedl_daemon.storage.executions import StaleExecution
from wavedl_daemon.storage.native_heads import get_native_head
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


def _revision(connection, plan_id):
    row = connection.execute(
        "SELECT COALESCE(e.revision,s.revision) FROM submission_control s "
        "LEFT JOIN execution_control e ON e.plan_id=s.plan_id WHERE s.plan_id=?",
        (plan_id,),
    ).fetchone()
    if row is None:
        raise ValueError("Native lifecycle lost its source")
    return row[0]


def _observation(encoded, identity, *, known=False):
    if encoded is None:
        return None
    _strict_data(encoded)
    observation = SubmissionObservation.from_json(encoded)
    if observation.native_identity != identity and not (
        not known
        and observation.native_identity is None
        and observation.status in {ObservationStatus.ERROR, ObservationStatus.ABSENT}
    ):
        raise ValueError("Native lifecycle observation differs from held identity")
    if known and observation.status in {
        ObservationStatus.ERROR,
        ObservationStatus.ABSENT,
    }:
        raise ValueError("Native diagnostic cannot become known lifecycle")
    return observation


def read_native_lifecycle(connection, plan_id, identity):
    """Validate and read only the held head, never an older known generation."""
    if schema_version(connection) < 13 or identity is None:
        return None, None
    try:
        if get_native_head(connection, plan_id) != identity:
            raise ValueError("Native lifecycle is not current head")
        row = connection.execute(
            "SELECT known_json,latest_json,source_revision,updated_at "
            "FROM submission_native_lifecycle WHERE plan_id=? AND restart_count=?",
            (plan_id, identity.restart_count),
        ).fetchone()
        if row is None:
            raise ValueError("Native lifecycle head row is missing")
        stamp = datetime.fromisoformat(row[3])
        if (
            type(row[2]) is not int
            or not 0 <= row[2] <= _revision(connection, plan_id)
            or len(row[3]) > 64
            or stamp.utcoffset() is None
            or stamp.utcoffset().total_seconds() != 0
        ):
            raise ValueError("Native lifecycle source revision or timestamp is invalid")
        known = _observation(row[0], identity, known=True)
        latest = _observation(row[1], identity)
        if latest is None:
            raise ValueError("Native lifecycle diagnostic is missing")
        if (
            latest.status not in {ObservationStatus.ERROR, ObservationStatus.ABSENT}
            and latest != known
        ):
            raise ValueError("Native known lifecycle differs from latest good evidence")
        return known, latest
    except (
        sqlite3.Error,
        AttributeError,
        KeyError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise PlanRecoveryRequired("Stored native lifecycle is damaged") from error


def seed_native_lifecycle(connection, plan_id, identity, observation):
    """Seed a new proof in its caller's transaction without inventing state."""
    if schema_version(connection) < 13:
        return
    known = (
        observation
        if observation.status not in {ObservationStatus.ERROR, ObservationStatus.ABSENT}
        else None
    )
    connection.execute(
        "INSERT INTO submission_native_lifecycle VALUES (?,?,?,?,?,?)",
        (
            plan_id,
            identity.restart_count,
            _json(known.to_dict()) if known else None,
            _json(observation.to_dict()),
            _revision(connection, plan_id),
            datetime.now(UTC).isoformat(),
        ),
    )


def write_native_lifecycle(connection, plan_id, identity, observation, source_revision):
    """Update the exact current native scope within the source CAS transaction."""
    if schema_version(connection) < 13:
        return
    if get_native_head(connection, plan_id) != identity:
        raise StaleExecution("Native lifecycle source differs from current head")
    if source_revision != _revision(connection, plan_id):
        raise StaleExecution("Native lifecycle source revision changed")
    previous, _ = read_native_lifecycle(connection, plan_id, identity)
    try:
        observation = _observation(_json(observation.to_dict()), identity)
    except ValueError as error:
        raise StaleExecution(str(error)) from error
    known = previous
    if observation.status not in {ObservationStatus.ERROR, ObservationStatus.ABSENT}:
        if previous is not None and previous.status == ObservationStatus.TERMINAL:
            if observation.status != ObservationStatus.TERMINAL or any(
                getattr(previous, field) is not None
                and getattr(previous, field) != getattr(observation, field)
                for field in (
                    "native_state",
                    "exit_code",
                    "signal",
                    "batch",
                    "stage_out",
                )
            ):
                raise StaleExecution("Cannot contradict terminal native lifecycle")
        known = observation
    connection.execute(
        "UPDATE submission_native_lifecycle SET known_json=?,latest_json=?,"
        "source_revision=?,updated_at=? WHERE plan_id=? AND restart_count=?",
        (
            _json(known.to_dict()) if known else None,
            _json(observation.to_dict()),
            source_revision,
            datetime.now(UTC).isoformat(),
            plan_id,
            identity.restart_count,
        ),
    )


def _create_tables(connection):
    if schema_version(connection) != 12:
        raise MigrationFailed("Native lifecycle requires schema 12")
    connection.execute(
        "CREATE TABLE submission_native_lifecycle ("
        "plan_id TEXT NOT NULL,restart_count INTEGER NOT NULL CHECK(restart_count>=0),"
        "known_json TEXT CHECK(length(CAST(known_json AS BLOB))<=262144),"
        "latest_json TEXT NOT NULL CHECK(length(CAST(latest_json AS BLOB))<=262144),"
        "source_revision INTEGER NOT NULL CHECK(source_revision>=0),updated_at TEXT NOT NULL,"
        "PRIMARY KEY(plan_id,restart_count),FOREIGN KEY(plan_id,restart_count) "
        "REFERENCES submission_native_generations(plan_id,restart_count))"
    )
    # Every retained generation gets its actual first proof. Only observations
    # matching that identity can improve its seed; older good state never leaks
    # into the latest generation's lifecycle.
    for row in connection.execute(
        "SELECT n.plan_id,n.restart_count,n.identity_json,n.first_observation_json,"
        "s.submission_observation_json,s.latest_submission_observation_json "
        "FROM submission_native_generations n JOIN submission_control s ON s.plan_id=n.plan_id"
    ).fetchall():
        from wavedl.core.scheduler_attempt import SchedulerIdentity

        identity = SchedulerIdentity(**_strict_data(row[2]))
        first = _observation(row[3], identity)
        known = (
            first
            if first.status not in {ObservationStatus.ERROR, ObservationStatus.ABSENT}
            else None
        )
        latest = first
        for encoded in row[4:]:
            if encoded is not None:
                _strict_data(encoded)
                candidate = SubmissionObservation.from_json(encoded)
                if candidate.native_identity == identity:
                    latest = candidate
                    if candidate.status not in {
                        ObservationStatus.ERROR,
                        ObservationStatus.ABSENT,
                    }:
                        known = candidate
        connection.execute(
            "INSERT INTO submission_native_lifecycle VALUES (?,?,?,?,?,?)",
            (
                row[0],
                row[1],
                _json(known.to_dict()) if known else None,
                _json(latest.to_dict()),
                _revision(connection, row[0]),
                datetime.now(UTC).isoformat(),
            ),
        )


NATIVE_LIFECYCLE_MIGRATION = Migration(
    "0013_submission_native_lifecycle", 13, _create_tables
)
