# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Immutable verified scheduler proofs, independent of lifecycle diagnostics."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

from wavedl.core.scheduler_attempt import SchedulerBinding, SchedulerIdentity
from wavedl_daemon.backends.contracts import (
    Handle,
    ObservationStatus,
    Plan,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.storage.schema import Migration, MigrationFailed, schema_version


def _source(connection, plan_id):
    row = connection.execute(
        "SELECT p.plan_json,p.request_id,s.owner,s.submission_handle_json,s.revision "
        "FROM submission_control s JOIN plans p ON p.plan_id=s.plan_id WHERE s.plan_id=?",
        (plan_id,),
    ).fetchone()
    if row is None or row[3] is None:
        raise ValueError("Native proof has lost its approved submission")
    plan, receipt = Plan.from_json(row[0]), SubmissionHandle.from_json(row[3])
    if (receipt.run_id, receipt.request_id, receipt.owner, receipt.run_dir) != (
        plan.run_id,
        row[1],
        row[2],
        plan.run_dir,
    ) or plan.allocation_mode != "writer":
        raise ValueError("Native proof differs from approved writer submission")
    return plan, receipt, row[4]


def _validate(connection, plan_id, identity, observation, revision):
    from wavedl_daemon.storage.submission_generations import receipt_identity

    plan, receipt, current_revision = _source(connection, plan_id)
    if (
        not isinstance(identity, SchedulerIdentity)
        or observation.native_identity != identity
        or identity != receipt_identity(receipt, plan, identity.restart_count)
        or type(revision) is not int
        or not 0 <= revision <= current_revision <= 2**63 - 1
    ):
        raise ValueError("Native proof differs from immutable ACK or source revision")


def get_native_head(connection, plan_id):
    """Return the maximal held proof, validating immutable ACK and plan on read."""
    from wavedl_daemon.storage.collections import _strict_data
    from wavedl_daemon.storage.plans import PlanRecoveryRequired

    if schema_version(connection) < 12:
        return None
    try:
        row = connection.execute(
            "SELECT restart_count,identity_json,first_observation_json,source_revision,observed_at "
            "FROM submission_native_generations WHERE plan_id=? ORDER BY restart_count DESC LIMIT 1",
            (plan_id,),
        ).fetchone()
        if row is None:
            return None
        identity = SchedulerIdentity(**_strict_data(row[1]))
        _strict_data(row[2])
        observation = SubmissionObservation.from_json(row[2])
        _validate(connection, plan_id, identity, observation, row[3])
        stamp = datetime.fromisoformat(row[4])
        if (
            row[0] != identity.restart_count
            or len(row[4]) > 64
            or stamp.utcoffset() is None
            or stamp.utcoffset().total_seconds() != 0
        ):
            raise ValueError("Invalid native proof key or timestamp")
        return identity
    except (
        sqlite3.Error,
        AttributeError,
        KeyError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise PlanRecoveryRequired(
            "Stored native generation head is damaged"
        ) from error


def record_native_observation(connection, plan_id, observation, source_revision):
    """Append a first proof in the caller's observation/CAS transaction."""
    from wavedl_daemon.storage.executions import StaleExecution, _json

    identity = observation.native_identity
    if identity is None or schema_version(connection) < 12:
        return
    try:
        _validate(connection, plan_id, identity, observation, source_revision)
    except ValueError as error:
        raise StaleExecution(str(error)) from error
    head = get_native_head(connection, plan_id)
    if head is not None:
        if identity.restart_count < head.restart_count:
            raise StaleExecution("late native generation is behind durable native head")
        if identity.restart_count == head.restart_count:
            if identity != head:
                raise StaleExecution("Native generation counter collision")
            return
    connection.execute(
        "INSERT INTO submission_native_generations VALUES (?,?,?,?,?,?)",
        (
            plan_id,
            identity.restart_count,
            _json(asdict(identity)),
            _json(observation.to_dict()),
            source_revision,
            datetime.now(UTC).isoformat(),
        ),
    )
    from wavedl_daemon.storage.native_lifecycle import seed_native_lifecycle

    seed_native_lifecycle(connection, plan_id, identity, observation)


def _create_tables(connection):
    if schema_version(connection) != 11:
        raise MigrationFailed("Native generation heads require schema 11")
    connection.execute(
        "CREATE TABLE submission_native_generations ("
        "plan_id TEXT NOT NULL REFERENCES submission_control(plan_id), "
        "restart_count INTEGER NOT NULL CHECK(restart_count>=0), "
        "identity_json TEXT NOT NULL CHECK(length(CAST(identity_json AS BLOB))<=262144), "
        "first_observation_json TEXT NOT NULL CHECK(length(CAST(first_observation_json AS BLOB))<=262144), "
        "source_revision INTEGER NOT NULL CHECK(source_revision>=0), observed_at TEXT NOT NULL, "
        "PRIMARY KEY(plan_id,restart_count))"
    )
    for operation in ("UPDATE", "DELETE"):
        connection.execute(
            f"CREATE TRIGGER submission_native_generations_no_{operation.lower()} BEFORE {operation} "
            "ON submission_native_generations BEGIN SELECT RAISE(ABORT, 'native generation proof is immutable'); END"
        )
    # Only surviving real schema-11 observations can establish an upgrade head.
    from wavedl_daemon.storage.collections import _strict_data
    from wavedl_daemon.storage.executions import _initial_handle, _json
    from wavedl_daemon.storage.submission_generations import (
        SubmissionGenerationRecord,
        _validate as validate_collection,
    )

    for row in connection.execute(
        "SELECT plan_id,submission_observation_json,latest_submission_observation_json,revision "
        "FROM submission_control"
    ).fetchall():
        candidates = []
        for encoded in row[1:3]:
            if encoded is not None:
                _strict_data(encoded)
                observation = SubmissionObservation.from_json(encoded)
                if observation.native_identity is not None:
                    _validate(
                        connection,
                        row[0],
                        observation.native_identity,
                        observation,
                        row[3],
                    )
                    candidates.append(observation)
        for collection in connection.execute(
            "SELECT c.plan_id,c.restart_count,c.handle_json,c.proposal_json,c.observed_at,"
            "p.plan_json,p.request_id,s.owner,s.submission_handle_json "
            "FROM submission_generation_collections c JOIN plans p ON p.plan_id=c.plan_id "
            "JOIN submission_control s ON s.plan_id=c.plan_id WHERE c.plan_id=?",
            (row[0],),
        ).fetchall():
            proof = validate_collection(
                SubmissionGenerationRecord(*collection[:5]),
                dict(
                    zip(
                        ("plan_json", "request_id", "owner", "submission_handle_json"),
                        collection[5:],
                        strict=True,
                    )
                ),
            )
            _validate(connection, row[0], proof.identity, proof.observation, row[3])
            candidates.append(proof.observation)
        plan, receipt, _ = (
            _source(connection, row[0])
            if candidates
            or connection.execute(
                "SELECT 1 FROM initial_bindings WHERE plan_id=?", (row[0],)
            ).fetchone()
            or connection.execute(
                "SELECT 1 FROM requeue_bindings WHERE plan_id=?", (row[0],)
            ).fetchone()
            else (None, None, None)
        )
        if receipt is not None and receipt.backend == "slurm":
            initial = connection.execute(
                "SELECT native_receipt_json,previous_revision FROM initial_bindings WHERE plan_id=?",
                (row[0],),
            ).fetchone()
            if (
                initial is None
                and connection.execute(
                    "SELECT 1 FROM requeue_bindings WHERE plan_id=?", (row[0],)
                ).fetchone()
            ):
                raise ValueError("Stored requeue has lost initial startup proof")
            if initial is not None and (
                initial[0] != _json(receipt.to_dict()) or initial[1] != row[3]
            ):
                raise ValueError(
                    "Stored startup has lost immutable submission receipt or revision"
                )
            bindings = connection.execute(
                "SELECT binding_json,runtime_handle_json FROM initial_bindings WHERE plan_id=? "
                "UNION ALL SELECT binding_json,handle_json FROM requeue_bindings WHERE plan_id=?",
                (row[0], row[0]),
            ).fetchall()
            for encoded_binding, encoded_handle in bindings:
                typed = SchedulerBinding(**_strict_data(encoded_binding))
                _strict_data(encoded_handle)
                handle = Handle.from_json(encoded_handle)
                if handle != _initial_handle(receipt, typed, plan):
                    raise ValueError(
                        "Stored startup proof differs from immutable native ACK"
                    )
                identity = SchedulerIdentity(
                    typed.run_id,
                    typed.request_id,
                    typed.cluster,
                    typed.job_id,
                    typed.owner,
                    typed.restart_count,
                )
                observation = SubmissionObservation(
                    ObservationStatus.ERROR,
                    detail="Startup binding proves native identity; lifecycle unconfirmed",
                    native_identity=identity,
                )
                _validate(connection, row[0], identity, observation, row[3])
                candidates.append(observation)
        if candidates:
            observation = max(
                candidates, key=lambda value: value.native_identity.restart_count
            )
            identity = observation.native_identity
            connection.execute(
                "INSERT INTO submission_native_generations VALUES (?,?,?,?,?,?)",
                (
                    row[0],
                    identity.restart_count,
                    _json(asdict(identity)),
                    _json(observation.to_dict()),
                    row[3],
                    datetime.now(UTC).isoformat(),
                ),
            )


NATIVE_HEAD_MIGRATION = Migration(
    "0012_submission_native_generations", 12, _create_tables
)
