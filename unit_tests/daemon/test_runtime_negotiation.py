import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.protocol import Envelope, ProtocolError
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    Plan,
    Topology,
)
from wavedl_daemon.runs.negotiation import (
    NegotiationStatus,
    RuntimeNegotiation,
    StaleNegotiation,
)
from wavedl_daemon.storage.executions import (
    EXECUTION_MIGRATION,
    ExecutionStore,
    StaleExecution,
)
from wavedl_daemon.storage.ownership import DispatchRequired, acquire_dispatch_lock
from wavedl_daemon.storage.plans import PLAN_MIGRATION, PlanRecoveryRequired, PlanStore
from wavedl_daemon.storage.runtime_provenance import (
    RUNTIME_PROVENANCE_MIGRATION,
    RuntimeProvenanceStore,
)
from wavedl_daemon.storage.schema import backup_path, migrate


RUN = "00000000-0000-4000-8000-000000000001"


def hello(**changes):
    payload = asdict(
        HelloPayload(
            1,
            1,
            "actual-runtime",
            "3.11",
            "cpu",
            ["train"],
            sys.executable,
            "container@sha256:abc",
            ["GPU-1"],
        )
    )
    payload.update(changes)
    return Envelope(RUN, 1, "hello", 0, "2026-09-05T12:00:00Z", payload)


def observation(status, attempt=1):
    return BackendObservation(attempt, status)


@pytest.mark.parametrize(
    "event",
    [
        replace(hello(), type="state", payload={"state": "running"}),
        replace(hello(), seq=1),
    ],
)
def test_first_event_must_be_zero_hello(event):
    result = RuntimeNegotiation(RUN, 1).accept_event(event)
    assert result.status == NegotiationStatus.FAILED
    assert "First runtime event" in result.reason


def test_protocol_range_and_actual_environment():
    assert (
        RuntimeNegotiation(RUN, 1).accept_event(hello(protocol_max=3)).status
        == NegotiationStatus.ACCEPTED
    )
    result = RuntimeNegotiation(RUN, 1).accept_event(
        hello(protocol_min=2, protocol_max=3)
    )
    assert result.status == NegotiationStatus.FAILED
    assert "incompatible" in result.reason


def test_stale_attempt_and_full_run_do_not_poison_active_tracker():
    tracker = RuntimeNegotiation(RUN, 1)
    for event in [
        replace(hello(), attempt=2),
        replace(hello(), run_id="00000000-0000-4000-8000-000000000002"),
    ]:
        with pytest.raises(StaleNegotiation):
            tracker.accept_event(event)
    with pytest.raises(StaleNegotiation):
        tracker.observe_backend(observation(ObservationStatus.RUNNING, 2), now=0)
    assert tracker.accept_event(hello()).status == NegotiationStatus.ACCEPTED


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_hello_extensions_and_clocks_rejected(number):
    assert (
        RuntimeNegotiation(RUN, 1)
        .accept_event(hello(extension={"nested": [number]}))
        .status
        == NegotiationStatus.FAILED
    )
    with pytest.raises(ValueError):
        RuntimeNegotiation(RUN, 1, timeout_seconds=number)
    with pytest.raises(ValueError):
        RuntimeNegotiation(RUN, 1).observe_backend(None, now=number)


def test_scheduler_hours_do_not_start_handshake_deadline():
    tracker = RuntimeNegotiation(RUN, 1, timeout_seconds=10)
    for status, now in [
        (ObservationStatus.QUEUED, 0),
        (ObservationStatus.HELD, 7200),
        (ObservationStatus.RUNNING, 10000),
    ]:
        assert (
            tracker.observe_backend(observation(status), now=now).status
            == NegotiationStatus.WAITING
        )
    assert tracker.result.running_since == 10000
    assert tracker.accept_event(hello(), now=10009).status == NegotiationStatus.ACCEPTED


def test_running_deadline_failure_is_sticky_and_visible():
    tracker = RuntimeNegotiation(RUN, 1, timeout_seconds=10)
    tracker.observe_backend(observation(ObservationStatus.RUNNING), now=20)
    assert (
        tracker.observe_backend(observation(ObservationStatus.RUNNING), now=29).status
        == NegotiationStatus.WAITING
    )
    result = tracker.observe_backend(observation(ObservationStatus.RUNNING), now=30)
    assert result.status == NegotiationStatus.FAILED
    assert result.handshake_timed_out
    assert "deadline" in result.reason
    assert tracker.accept_event(hello(), now=31) == result


@pytest.mark.parametrize(
    "status",
    [
        ObservationStatus.ERROR,
        ObservationStatus.ABSENT,
        ObservationStatus.SUPERVISION_LOST,
        ObservationStatus.QUEUED,
        ObservationStatus.HELD,
        ObservationStatus.TERMINAL,
    ],
)
def test_backend_uncertainty_and_scheduler_wait_cannot_certify_timeout(status):
    tracker = RuntimeNegotiation(RUN, 1, timeout_seconds=10)
    tracker.observe_backend(observation(ObservationStatus.RUNNING), now=0)
    assert not tracker.observe_backend(observation(status), now=100).handshake_timed_out
    assert tracker.accept_event(hello(), now=101).status == NegotiationStatus.ACCEPTED


def test_hello_replay_exact_is_idempotent_conflict_fails():
    tracker = RuntimeNegotiation(RUN, 1)
    accepted = tracker.accept_event(hello())
    assert tracker.accept_event(hello()) == accepted
    result = tracker.accept_event(hello(interpreter="/different/python"))
    assert result.status == NegotiationStatus.FAILED
    assert "Conflicting" in result.reason
    assert result.hello_json == accepted.hello_json


@pytest.fixture
def provenance_database(tmp_path, request):
    settings = getattr(request, "param", {})
    database = tmp_path / "registry.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=1")
    migrate(database, [PLAN_MIGRATION, EXECUTION_MIGRATION])
    manifest = build_manifest(
        RunLayout(tmp_path / RUN, RUN),
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 3},
        runtime=RuntimeSelection(
            settings.get("python", sys.executable),
            "existing",
            settings.get("device", "cpu"),
            {},
        ),
        submission_cwd=tmp_path,
    )
    plan = Plan(
        "local",
        json.dumps(asdict(manifest)),
        argv=(sys.executable, "train.py"),
        topology=Topology(processes_per_machine=settings.get("processes", 1)),
    )
    with acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock:
        with PlanStore(database, dispatch_lock=lock) as plans:
            record = plans.put(plan, idempotency_key="one", request_payload={})
        with ExecutionStore(database, dispatch_lock=lock) as executions:
            executions.ensure(record.plan_id, 1)
    migrate(database, [RUNTIME_PROVENANCE_MIGRATION])
    return database, record


def test_provenance_immutable_independent_request_and_reopen(provenance_database):
    database, plan = provenance_database
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as store,
    ):
        actual = store.record(plan.plan_id, hello())
        assert actual.hello.interpreter == sys.executable
        assert actual.hello.device == "cpu"
        assert actual.hello.runtime_version == "actual-runtime"
        assert store.record(plan.plan_id, hello()) == actual
        with pytest.raises(StaleExecution):
            store.record(plan.plan_id, hello(runtime_version="different-runtime"))
        for event in [
            replace(hello(), attempt=2),
            replace(hello(), run_id="00000000-0000-4000-8000-000000000002"),
        ]:
            with pytest.raises(StaleExecution):
                store.record(plan.plan_id, event)
        with pytest.raises(ProtocolError):
            store.record(plan.plan_id, hello(extension=float("nan")))
    with RuntimeProvenanceStore(database, read_only=True) as store:
        recovered = store.get(plan.plan_id, 1)
        assert recovered == actual
        recovered.hello.capabilities.append("mutation")
        assert store.get(plan.plan_id).hello.capabilities == ["train"]
        with pytest.raises(PermissionError):
            store.record(plan.plan_id, hello())
    with PlanStore(database, read_only=True) as plans:
        assert plans.get(plan.plan_id).plan.manifest.runtime.device == "cpu"


def test_store_requires_dispatch_and_matching_db_inode(provenance_database, tmp_path):
    database, plan = provenance_database
    with RuntimeProvenanceStore(database) as store:
        with pytest.raises(DispatchRequired):
            store.record(plan.plan_id, hello())
        if os.name == "nt":
            before = store.get(plan.plan_id)
            original_identity = database.stat()
            original_bytes = database.read_bytes()
            with pytest.raises(PermissionError) as denied:
                database.rename(tmp_path / "old.db")
            assert denied.value.winerror == 32
            assert not (tmp_path / "old.db").exists()
            assert database.stat().st_ino == original_identity.st_ino
            assert database.read_bytes() == original_bytes
            assert store.get(plan.plan_id) == before
            return
        database.rename(tmp_path / "old.db")
        with sqlite3.connect(database):
            pass
        with pytest.raises(PlanRecoveryRequired):
            store.get(plan.plan_id)


def test_migration_backup_and_upgrade_survive(provenance_database):
    database, _ = provenance_database
    assert migrate(database, [RUNTIME_PROVENANCE_MIGRATION]) == []
    with sqlite3.connect(
        backup_path(database, RUNTIME_PROVENANCE_MIGRATION.migration_id)
    ) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            backup.execute("SELECT COUNT(*) FROM execution_control").fetchone()[0] == 1
        )
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


def test_read_only_future_schema_and_sql_immutability(provenance_database):
    database, plan = provenance_database
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as store,
    ):
        actual = store.record(plan.plan_id, hello())
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE runtime_provenance SET attempt=2")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM runtime_provenance")
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(PlanRecoveryRequired):
        RuntimeProvenanceStore(database)
    with RuntimeProvenanceStore(database, read_only=True) as store:
        assert store.get(plan.plan_id) == actual


def test_clock_backwards_and_out_of_range_rejected():
    tracker = RuntimeNegotiation(RUN, 1)
    tracker.observe_backend(None, now=10)
    with pytest.raises(ValueError):
        tracker.observe_backend(None, now=9)
    with pytest.raises(ValueError):
        RuntimeNegotiation(RUN, 1, timeout_seconds=10**10000)


def test_external_policy_rejection_is_sticky_without_accepted_hello():
    tracker = RuntimeNegotiation(RUN, 1)
    result = tracker.reject("Actual runtime differs from approved request")
    assert result.status == NegotiationStatus.FAILED
    assert result.hello_json is None
    assert not result.handshake_timed_out
    assert tracker.reject("Different reason") == result
    assert tracker.accept_event(hello()) == result


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "cuda"},
        {"interpreter": "/unapproved/python"},
        {"interpreter": "python"},
        {"runtime_version": " "},
        {"python": "3.10.12"},
        {"python": "3.11garbage"},
        {"python": ""},
        {"allocated_devices": [""]},
        {"allocated_devices": ["GPU-1", "GPU-1"]},
    ],
)
def test_approved_runtime_policy_rejects_before_provenance_commit(
    provenance_database, changes
):
    database, plan = provenance_database
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as store,
    ):
        with pytest.raises(ProtocolError):
            store.record(plan.plan_id, hello(**changes))
        assert store.get(plan.plan_id) is None


@pytest.mark.parametrize(
    "provenance_database", [{"device": "cuda", "processes": 2}], indirect=True
)
def test_cuda_allocation_must_satisfy_approved_topology(provenance_database):
    database, plan = provenance_database
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as store,
    ):
        with pytest.raises(ProtocolError, match="allocation"):
            store.record(
                plan.plan_id, hello(device="cuda", allocated_devices=["GPU-1"])
            )
        result = store.record(
            plan.plan_id,
            hello(device="cuda", allocated_devices=["GPU-1", "MIG-device/2/3"]),
        )
        assert result.hello.allocated_devices == ["GPU-1", "MIG-device/2/3"]


@pytest.mark.parametrize("provenance_database", [{"device": "auto"}], indirect=True)
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_auto_device_accepts_actual_supported_device(provenance_database, device):
    database, plan = provenance_database
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as store,
    ):
        result = store.record(plan.plan_id, hello(device=device))
        assert result.hello.device == device


def test_interpreter_lexical_normalization_never_resolves_symlink(
    provenance_database, tmp_path
):
    database, plan = provenance_database
    selected = Path(sys.executable)
    alias = tmp_path / "venv-python"
    alias.symlink_to(selected)
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as store,
    ):
        with pytest.raises(ProtocolError, match="interpreter differs"):
            store.record(plan.plan_id, hello(interpreter=str(alias)))
        lexical = str(selected.parent) + "/./" + selected.name
        assert (
            store.record(plan.plan_id, hello(interpreter=lexical)).hello.interpreter
            == lexical
        )


def attested_handle(plan, *, identity=None):
    from wavedl_daemon.backends.contracts import Handle

    handle = Handle(
        "local",
        plan.run_id,
        "request",
        1,
        "owner",
        plan.run_dir,
        supervisor_id="supervisor",
    )
    proof = {
        "schema_version": 1,
        "selected_python": plan.manifest.runtime.python,
        "plan_sha256": hashlib.sha256(plan.to_json().encode()).hexdigest(),
        "run_id": plan.run_id,
        "request_id": handle.request_id,
        "owner": handle.owner,
        "supervisor_id": handle.supervisor_id,
        "identity": identity
        or {
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
    }
    return replace(handle, raw_json=json.dumps({"runtime_attestation": proof}))


def test_attested_alias_exact_environment_and_legacy_exact_only(provenance_database):
    from wavedl_daemon.storage.runtime_provenance import _approved_runtime

    _, record = provenance_database
    plan = record.plan
    manifest = replace(
        plan.manifest,
        runtime=replace(
            plan.manifest.runtime,
            python=str(Path(sys.executable).anchor + "selected/system/alias"),
        ),
    )
    plan = replace(plan, manifest_json=json.dumps(asdict(manifest)))
    actual = HelloPayload.from_dict(
        hello(prefix=sys.prefix, base_prefix=sys.base_prefix).payload
    )
    _approved_runtime(plan, actual, attested_handle(plan))
    with pytest.raises(ProtocolError, match="interpreter differs"):
        _approved_runtime(plan, actual)
    with pytest.raises(ProtocolError, match="attestation"):
        _approved_runtime(
            plan, HelloPayload.from_dict(hello().payload), attested_handle(plan)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefix", "/different/env"),
        ("base_prefix", "/different/base"),
        ("interpreter", "/different/python"),
    ],
)
def test_same_binary_or_prefix_cannot_replace_attested_environment(
    provenance_database, field, value
):
    from wavedl_daemon.storage.runtime_provenance import _approved_runtime

    _, record = provenance_database
    value = str(Path(sys.executable).anchor + value.lstrip("/"))
    actual = (
        hello(prefix=sys.prefix, base_prefix=sys.base_prefix, **{field: value})
        if field == "interpreter"
        else hello(
            **({"prefix": sys.prefix, "base_prefix": sys.base_prefix} | {field: value})
        )
    )
    with pytest.raises(ProtocolError, match="environment differs"):
        _approved_runtime(
            record.plan,
            HelloPayload.from_dict(actual.payload),
            attested_handle(record.plan),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "foreign"),
        ("request_id", "foreign"),
        ("plan_sha256", "0" * 64),
        ("supervisor_id", "foreign"),
        ("run_id", "foreign"),
        ("selected_python", "/different/python"),
        ("identity", {"interpreter": sys.executable}),
    ],
)
def test_foreign_or_corrupt_probe_proof_refused(provenance_database, field, value):
    from wavedl_daemon.storage.runtime_provenance import _approved_runtime

    _, record = provenance_database
    handle = attested_handle(record.plan)
    raw = json.loads(handle.raw_json)
    raw["runtime_attestation"][field] = value
    handle = replace(handle, raw_json=json.dumps(raw))
    with pytest.raises(ProtocolError):
        _approved_runtime(
            record.plan,
            HelloPayload.from_dict(
                hello(prefix=sys.prefix, base_prefix=sys.base_prefix).payload
            ),
            handle,
        )


def test_scheduler_alias_never_uses_local_probe_attestation(provenance_database):
    from wavedl_daemon.storage.runtime_provenance import _approved_runtime

    _, record = provenance_database
    handle = replace(attested_handle(record.plan), backend="slurm")
    with pytest.raises(ProtocolError, match="attestation differs"):
        _approved_runtime(
            record.plan,
            HelloPayload.from_dict(
                hello(prefix=sys.prefix, base_prefix=sys.base_prefix).payload
            ),
            handle,
        )


def test_null_attestation_is_corruption_not_a_legacy_receipt(provenance_database):
    from wavedl_daemon.storage.runtime_provenance import _approved_runtime

    _, record = provenance_database
    handle = replace(
        attested_handle(record.plan), raw_json='{"runtime_attestation":null}'
    )
    with pytest.raises(ProtocolError, match="attestation differs"):
        _approved_runtime(record.plan, HelloPayload.from_dict(hello().payload), handle)


def test_duplicate_attestation_identity_cannot_be_canonicalized(provenance_database):
    _, record = provenance_database
    handle = attested_handle(record.plan)
    raw = handle.raw_json.replace(
        '"owner":"owner"', '"owner":"foreign","owner":"owner"'
    )
    assert raw != handle.raw_json
    with pytest.raises(ValueError, match="Duplicate"):
        replace(handle, raw_json=raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"interpreter":"/foreign","interpreter":"/actual","prefix":"/env","base_prefix":"/base"}',
        '{"identity":{"prefix":"/foreign","prefix":"/env"}}',
    ],
)
def test_duplicate_probe_json_refused(raw):
    from wavedl.core.runtime_identity import load_runtime_identity_json

    with pytest.raises(ValueError, match="Duplicate"):
        load_runtime_identity_json(raw)


def test_persisted_attestation_survives_restart_and_cannot_be_replaced(
    provenance_database,
):
    database, record = provenance_database
    with acquire_dispatch_lock(database.parent / "dispatch.lock") as lock:
        with PlanStore(database, dispatch_lock=lock) as plans:
            plans.request_submission(record.plan_id)
        with ExecutionStore(database, dispatch_lock=lock) as executions:
            execution, claimed = executions.claim_submission(record.plan_id)
            assert claimed
            handle = replace(
                attested_handle(record.plan),
                request_id=record.request_id,
                owner=execution.owner,
            )
            proof = json.loads(handle.raw_json)
            proof["runtime_attestation"].update(
                request_id=handle.request_id, owner=handle.owner
            )
            handle = replace(handle, raw_json=json.dumps(proof))
            executions.record_handle(record.plan_id, handle)
        # Reopen both durable stores, without live backend ownership.
        with ExecutionStore(database, dispatch_lock=lock) as executions:
            assert executions.get(record.plan_id).handle == handle
        event = hello(prefix=sys.prefix, base_prefix=sys.base_prefix)
        with RuntimeProvenanceStore(database, dispatch_lock=lock) as store:
            stored = store.record(record.plan_id, event)
        with RuntimeProvenanceStore(database, dispatch_lock=lock) as store:
            assert store.record(record.plan_id, event) == stored
            with pytest.raises((StaleExecution, ProtocolError)):
                store.record(
                    record.plan_id,
                    hello(prefix="/foreign", base_prefix=sys.base_prefix),
                )
