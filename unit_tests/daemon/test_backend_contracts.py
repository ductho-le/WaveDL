"""Planning snapshots and native evidence retain identity without ML dependencies."""

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import Manifest, RuntimeSelection, build_manifest
from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    OperationOutcome,
    OperationResult,
    OutcomeStatus,
    Plan,
    SubmissionCollectionResult,
    SubmissionHandle,
    SubmissionObservation,
    Topology,
    _json,
)


CONTRACT_FIXTURE = Path(__file__).with_name("fixtures") / "contracts_codec.json"


def contract_fixture():
    return json.loads(CONTRACT_FIXTURE.read_text())


def make_plan(tmp_path):
    manifest = build_manifest(
        RunLayout(tmp_path / "run", str(uuid4())),
        operation=OperationKind.TRAIN,
        resolved_config={"model": "cnn", "epochs": 2},
        runtime=RuntimeSelection("venv/bin/python", "selected", "cpu", {"TEST": "1"}),
        submission_cwd=tmp_path,
    )
    return Plan(
        "local",
        _json(asdict(manifest)),
        argv=(manifest.runtime.python, "-m", "wavedl.train"),
        environment=(("TEST", "1"),),
    )


def frozen_contract_plan(*, allocation_mode="reserved"):
    run_id = "00000000-0000-4000-8000-000000000001"
    root = Path("/tmp/wavedl-contract-fixture")
    run_dir = root / "run" / run_id
    manifest = Manifest(
        run_id=run_id,
        operation=OperationKind.TRAIN,
        resolved_config={
            "model": "cnn",
            "epochs": 2,
            "output_dir": str(run_dir),
        },
        runtime=RuntimeSelection(
            "/opt/wavedl/bin/python", "selected", "cpu", {"TEST": "1"}
        ),
        submission_directory=str(root),
        working_directory=str(run_dir),
        run_directory=str(run_dir),
        storage_map=(),
        path_fields=(),
    )
    return Plan(
        "local",
        _json(manifest.to_dict()),
        argv=("/opt/wavedl/bin/python", "-m", "wavedl.train"),
        environment=(("TEST", "1"),),
        allocation_mode=allocation_mode,
    )


@pytest.mark.skipif(os.name == "nt", reason="fixture pins POSIX absolute paths")
def test_contract_codec_fixtures_pin_plan_bytes_hash_and_reserved_omission(tmp_path):
    import sqlite3

    from wavedl_daemon.storage.ownership import acquire_dispatch_lock
    from wavedl_daemon.storage.plans import PLAN_MIGRATION, PlanStore
    from wavedl_daemon.storage.schema import migrate
    from wavedl_daemon.storage.submissions import validate_writer_receipt

    fixture = contract_fixture()
    reserved = frozen_contract_plan()
    assert reserved.to_json() == fixture["plan_json"]
    assert Plan.from_json(fixture["plan_json"]).to_json() == fixture["plan_json"]
    assert "allocation_mode" not in json.loads(fixture["plan_json"])
    assert (
        hashlib.sha256(reserved.to_json().encode("utf-8")).hexdigest()
        == fixture["plan_sha256"]
    )
    database = tmp_path / "registry.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=1")
    migrate(database, [PLAN_MIGRATION])
    with (
        acquire_dispatch_lock(database.parent / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as store,
    ):
        record = store.put(
            reserved, idempotency_key="codec-fixture", request_payload={}
        )
    assert record.plan_json == fixture["plan_json"]
    assert record.plan.to_json() == record.plan_json

    writer = frozen_contract_plan(allocation_mode="writer")
    assert writer.to_json() == fixture["writer_plan_json"]
    assert (
        Plan.from_json(fixture["writer_plan_json"]).to_json()
        == fixture["writer_plan_json"]
    )
    assert (
        hashlib.sha256(writer.to_json().encode("utf-8")).hexdigest()
        == fixture["writer_plan_sha256"]
    )
    assert json.loads(fixture["writer_plan_json"])["allocation_mode"] == "writer"

    request_id = "00000000-0000-4000-8000-000000000002"
    receipt = SubmissionHandle(
        "slurm",
        writer.run_id,
        request_id,
        "fixture-owner",
        writer.run_dir,
        job_id="42;alpha",
        raw_json=_json(
            {
                "cluster": "alpha",
                "comment": request_id,
                "job_name": f"wavedl-{request_id}",
                "restart_count": 0,
                "operation": "train",
                "allocation_mode": "writer",
                "plan_sha256": fixture["writer_plan_sha256"],
            }
        ),
    )
    assert (
        validate_writer_receipt(receipt, writer)["plan_sha256"]
        == fixture["writer_plan_sha256"]
    )


def test_plan_exact_serialization_and_nested_snapshot(tmp_path):
    plan = make_plan(tmp_path)
    wire = plan.to_json()
    assert Plan.from_json(wire) == plan
    assert Plan.from_json(wire).to_json() == wire
    plan.manifest.resolved_config["epochs"] = 99
    plan.manifest.runtime.environment["TEST"] = "changed"
    assert plan.to_json() == wire
    with pytest.raises(FrozenInstanceError):
        plan.target = "slurm"


@pytest.mark.parametrize("topology", [(0, 1, 0), (True, 1, 0), (1, 0, 0), (1, 1, 1)])
def test_bad_topology_is_not_renderable(topology):
    with pytest.raises(ValueError):
        Topology(*topology)


@pytest.mark.parametrize(
    "mutation",
    [
        {"argv": [], "script": None},
        {"argv": "python"},
        {"script": "echo success"},
        {"resources": [["gpus", float("nan")]]},
        {"resources": [["gpus", {"mutable": True}]]},
        {"environment": [["TEST", "different"]]},
        {"unexpected_field": 1},
    ],
)
def test_malformed_persisted_plan_rejected(tmp_path, mutation):
    data = make_plan(tmp_path).to_dict() | mutation
    with pytest.raises((ValueError, TypeError)):
        Plan.from_dict(data)


def test_handle_requires_supervision_identity_and_retains_owner(tmp_path):
    data = {
        "backend": "local",
        "run_id": str(uuid4()),
        "request_id": "request",
        "attempt": 1,
        "owner": "user",
        "run_dir": str(tmp_path),
        "pid": 42,
    }
    with pytest.raises(ValueError, match="PID alone"):
        Handle(**data)
    handle = Handle(**data, supervisor_id="session-unique", raw_json='{"b":2,"a":1}')
    assert Handle.from_json(handle.to_json()) == handle
    assert handle.owner == "user"
    assert handle.raw_json == '{"a":1,"b":2}'


def test_wrapper_success_does_not_overwrite_runtime_failure():
    runtime = OperationOutcome(OutcomeStatus.FAILED, exit_code=3)
    batch = OperationOutcome(OutcomeStatus.SUCCESS, exit_code=0)
    result = OperationResult(
        OperationKind.TEST, runtime, batch, OperationOutcome(OutcomeStatus.UNKNOWN)
    )
    assert result.runtime.exit_code == 3
    assert result.batch.exit_code == 0
    with pytest.raises(ValueError):
        OperationOutcome(OutcomeStatus.SUCCESS, exit_code=3)


@pytest.mark.parametrize(
    "status",
    [
        ObservationStatus.RUNNING,
        ObservationStatus.ERROR,
        ObservationStatus.ABSENT,
        ObservationStatus.SUPERVISION_LOST,
    ],
)
def test_nonterminal_observation_does_not_claim_exit(status):
    with pytest.raises(ValueError):
        BackendObservation(1, status, exit_code=0)


def test_native_state_and_signal_are_preserved():
    observation = BackendObservation(
        2,
        ObservationStatus.TERMINAL,
        native_state="OUT_OF_MEMORY",
        exit_code=137,
        signal=9,
    )
    assert observation.native_state == "OUT_OF_MEMORY"
    assert observation.signal == 9


def test_reserved_allocation_mode_preserves_exact_legacy_preview_bytes_and_hash(
    tmp_path,
):
    plan = make_plan(tmp_path)
    legacy = {
        "target": plan.target,
        "manifest_json": plan.manifest_json,
        "argv": plan.argv,
        "script": plan.script,
        "topology": asdict(plan.topology),
        "resources": plan.resources,
        "environment": plan.environment,
        "warnings": plan.warnings,
    }
    wire = _json(legacy)
    assert plan.allocation_mode == "reserved"
    assert "allocation_mode" not in plan.to_dict()
    assert plan.to_json() == wire
    assert (
        hashlib.sha256(plan.to_json().encode()).hexdigest()
        == hashlib.sha256(wire.encode()).hexdigest()
    )
    restored = Plan.from_json(wire)
    assert restored.allocation_mode == "reserved" and restored.to_json() == wire
    data = json.loads(wire)
    before = deepcopy(data)
    assert Plan.from_dict(data).to_json() == wire
    assert data == before


def test_writer_allocation_mode_is_explicit_and_cannot_mutate_approved_plan(tmp_path):
    plan = make_plan(tmp_path)
    writer = replace(plan, allocation_mode="writer")
    assert writer.to_dict()["allocation_mode"] == "writer"
    assert Plan.from_json(writer.to_json()) == writer
    assert Plan.from_json(writer.to_json()).to_json() == writer.to_json()
    assert (
        plan.allocation_mode == "reserved" and "allocation_mode" not in plan.to_dict()
    )
    for mode in (None, True, "unknown", "", [], 0):
        with pytest.raises(ValueError):
            replace(plan, allocation_mode=mode)


def submission_data(tmp_path, backend="slurm"):
    run_id, request_id = str(uuid4()), str(uuid4())
    if backend == "slurm":
        return {
            "backend": backend,
            "run_id": run_id,
            "request_id": request_id,
            "owner": "alice",
            "run_dir": str(tmp_path),
            "job_id": "42;alpha",
            "raw_json": _json(
                {
                    "cluster": "alpha",
                    "job_name": f"wavedl-{request_id}",
                    "comment": request_id,
                    "restart_count": 0,
                }
            ),
        }
    return {
        "backend": backend,
        "run_id": run_id,
        "request_id": request_id,
        "owner": "opaque local owner",
        "run_dir": str(tmp_path),
        "pid": 42,
        "supervisor_id": "process-supervisor-unique",
        "raw_json": '{"b":2,"a":1}',
    }


@pytest.mark.parametrize("backend", ["local", "slurm"])
def test_submission_receipt_has_no_attempt_and_binds_without_allocating_files(
    tmp_path, backend
):
    data = submission_data(tmp_path, backend)
    before = deepcopy(data)
    native = SubmissionHandle.from_dict(data)
    assert data == before
    assert "attempt" not in native.to_dict() and not hasattr(native, "attempt")
    assert SubmissionHandle.from_json(native.to_json()) == native
    bound = native.bind_attempt(3)
    assert type(bound) is Handle and bound.attempt == 3
    assert {
        name: value for name, value in bound.to_dict().items() if name != "attempt"
    } == native.to_dict()
    assert list(tmp_path.iterdir()) == []
    for attempt in (0, -1, True, None, 1.5, float("nan")):
        with pytest.raises(ValueError):
            native.bind_attempt(attempt)
    with pytest.raises(FrozenInstanceError):
        native.owner = "forged"


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "not-uuid"},
        {"request_id": "request"},
        {"run_id": True},
        {"backend": "other"},
        {"run_dir": "relative"},
        {"owner": "alice\nroot"},
        {"job_id": "0;alpha"},
        {"job_id": "01;alpha"},
        {"job_id": "42"},
        {"job_id": "42;../alpha"},
        {"job_id": "42;alpha\0"},
        {"pid": 42},
        {"raw_json": "{}"},
        {"raw_json": '{"number":NaN}'},
    ],
)
def test_forged_or_unverified_submission_identity_is_rejected(tmp_path, changes):
    with pytest.raises((ValueError, TypeError)):
        SubmissionHandle(**(submission_data(tmp_path) | changes))


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster", "foreign"),
        ("comment", "wrong"),
        ("job_name", "wavedl-foreign"),
        ("restart_count", True),
        ("restart_count", -1),
        ("restart_count", float("inf")),
    ],
)
def test_submission_native_metadata_cannot_forge_request_or_generation(
    tmp_path, field, value
):
    data = submission_data(tmp_path)
    raw = json.loads(data["raw_json"])
    raw[field] = value
    data["raw_json"] = json.dumps(raw)
    with pytest.raises(ValueError):
        SubmissionHandle(**data)


def test_local_submission_pid_needs_durable_supervisor_without_scheduler_identity(
    tmp_path,
):
    data = submission_data(tmp_path, "local")
    with pytest.raises(ValueError, match="PID alone"):
        SubmissionHandle(**(data | {"supervisor_id": None}))
    with pytest.raises(ValueError):
        SubmissionHandle(**(data | {"job_id": "42;alpha"}))
    with pytest.raises(ValueError):
        SubmissionHandle(**(data | {"pid": None}))
    with pytest.raises(ValueError):
        SubmissionHandle(**(data | {"supervisor_id": "unsafe\0"}))
    # Legacy execution handles retain their existing opaque request contract.
    assert (
        Handle(**(data | {"request_id": "legacy-request", "attempt": 1})).request_id
        == "legacy-request"
    )


def test_submission_observation_roundtrip_is_attemptless_typed_and_immutable():
    observation = SubmissionObservation(
        ObservationStatus.TERMINAL,
        "OUT_OF_MEMORY",
        137,
        9,
        "native terminal",
        OperationOutcome(OutcomeStatus.FAILED, 137),
        OperationOutcome(OutcomeStatus.UNKNOWN),
    )
    wire = observation.to_json()
    data = observation.to_dict()
    before = deepcopy(data)
    assert "attempt" not in data and not hasattr(observation, "attempt")
    assert SubmissionObservation.from_dict(data) == observation
    assert data == before
    assert SubmissionObservation.from_json(wire).to_json() == wire
    bound = observation.bind_attempt(2)
    assert bound == BackendObservation(
        2,
        ObservationStatus.TERMINAL,
        "OUT_OF_MEMORY",
        137,
        9,
        "native terminal",
        observation.batch,
        observation.stage_out,
    )
    with pytest.raises(FrozenInstanceError):
        observation.detail = "changed"
    for attempt in (None, False, 0, -1):
        with pytest.raises(ValueError):
            observation.bind_attempt(attempt)


def test_duplicate_native_metadata_and_oversized_receipts_refused(tmp_path):
    data = submission_data(tmp_path, "local")
    for raw in (
        '{"owner":"forged","owner":"valid"}',
        '{"blob":"' + "x" * 262144 + '"}',
    ):
        with pytest.raises(ValueError):
            SubmissionHandle(**(data | {"raw_json": raw}))
    # Local owner identifiers remain opaque and need not be Slurm username tokens.
    assert SubmissionHandle(**data).owner == "opaque local owner"


@pytest.mark.parametrize(
    "status",
    [
        ObservationStatus.RUNNING,
        ObservationStatus.ERROR,
        ObservationStatus.QUEUED,
        ObservationStatus.ABSENT,
        ObservationStatus.SUPERVISION_LOST,
    ],
)
def test_submission_nonterminal_evidence_never_claims_an_exit(status):
    with pytest.raises(ValueError):
        SubmissionObservation(status, exit_code=0)
    with pytest.raises(ValueError):
        SubmissionObservation(status, signal=9)
    assert (
        SubmissionObservation.from_json(SubmissionObservation(status).to_json()).status
        == status
    )


@pytest.mark.parametrize(
    "data",
    [
        {"status": "invalid"},
        {"status": "terminal", "exit_code": True},
        {"status": "terminal", "batch": {"status": "success", "exit_code": 7}},
        {"status": "terminal", "signal": float("nan")},
        {"status": "running", "attempt": 0},
    ],
)
def test_malformed_submission_observations_refused(data):
    with pytest.raises((ValueError, TypeError)):
        SubmissionObservation.from_dict(data)


def scheduler_identity():
    return SchedulerIdentity(str(uuid4()), str(uuid4()), "alpha", "42", "alice", 0)


def collection_result(identity=None):
    identity = identity or scheduler_identity()
    return SubmissionCollectionResult(
        identity,
        "a" * 64,
        "b" * 64,
        OperationResult(
            OperationKind.TRAIN,
            OperationOutcome(OutcomeStatus.UNKNOWN),
            OperationOutcome(OutcomeStatus.FAILED, 137, 9),
            OperationOutcome(OutcomeStatus.UNKNOWN),
        ),
        SubmissionObservation(
            ObservationStatus.TERMINAL,
            "OUT_OF_MEMORY",
            137,
            9,
            native_identity=identity,
        ),
    )


def test_native_observation_preserves_legacy_bytes_and_typed_identity():
    legacy = SubmissionObservation(ObservationStatus.QUEUED)
    assert legacy.to_json() == _json(
        {
            "status": "queued",
            "native_state": None,
            "exit_code": None,
            "signal": None,
            "detail": "",
            "batch": None,
            "stage_out": None,
        }
    )
    assert "native_identity" not in legacy.to_dict()
    native = replace(legacy, native_identity=scheduler_identity())
    data = native.to_dict()
    before = deepcopy(data)
    assert SubmissionObservation.from_dict(data) == native
    assert data == before
    assert (
        SubmissionObservation.from_json(native.to_json()).to_json() == native.to_json()
    )
    assert native.bind_attempt(3) == legacy.bind_attempt(3)
    assert not hasattr(native.bind_attempt(3), "native_identity")
    with pytest.raises(ValueError):
        replace(native, native_identity=asdict(native.native_identity))


def test_collection_is_exact_attemptless_immutable_and_generation_scoped():
    collection = collection_result()
    data = collection.to_dict()
    before = deepcopy(data)
    assert SubmissionCollectionResult.from_dict(data) == collection
    assert data == before
    assert (
        SubmissionCollectionResult.from_json(collection.to_json()).to_json()
        == collection.to_json()
    )
    assert "attempt" not in data and not hasattr(collection, "attempt")
    assert collection.result.runtime.status == OutcomeStatus.UNKNOWN
    assert collection.result.batch.exit_code == 137
    with pytest.raises(FrozenInstanceError):
        collection.plan_sha256 = "c" * 64
    for field, value in (
        ("run_id", str(uuid4())),
        ("request_id", str(uuid4())),
        ("cluster", "foreign"),
        ("job_id", "43"),
        ("owner", "bob"),
        ("restart_count", 1),
    ):
        foreign = replace(collection.identity, **{field: value})
        with pytest.raises(ValueError, match="matching terminal"):
            replace(
                collection,
                observation=replace(collection.observation, native_identity=foreign),
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": True},
        {"request_id": "wrong"},
        {"cluster": "../alpha"},
        {"job_id": 42},
        {"job_id": "01"},
        {"job_id": str(2**63)},
        {"owner": "alice\0"},
        {"restart_count": True},
        {"restart_count": -1},
        {"restart_count": 2**63},
        {"restart_count": 1.5},
        {"extra": 1},
    ],
)
def test_native_identity_strict_fields_and_numeric_bounds(changes):
    collection = collection_result()
    identity = asdict(collection.identity) | changes
    observation = collection.observation.to_dict() | {"native_identity": identity}
    with pytest.raises((ValueError, TypeError)):
        SubmissionObservation.from_dict(observation)
    with pytest.raises((ValueError, TypeError)):
        SubmissionCollectionResult.from_dict(
            collection.to_dict() | {"identity": identity}
        )


@pytest.mark.parametrize("value", [None, True, 1, "a" * 63, "A" * 64, "g" * 64])
def test_collection_hashes_fail_closed(value):
    collection = collection_result()
    for field in ("plan_sha256", "manifest_sha256"):
        with pytest.raises(ValueError):
            replace(collection, **{field: value})


@pytest.mark.parametrize(
    "runtime",
    [
        OperationOutcome(OutcomeStatus.SUCCESS),
        OperationOutcome(OutcomeStatus.FAILED, 3),
        OperationOutcome(OutcomeStatus.UNKNOWN, 0),
        OperationOutcome(OutcomeStatus.UNKNOWN, signal=9),
    ],
)
def test_unbound_collection_cannot_claim_runtime(runtime):
    collection = collection_result()
    with pytest.raises(ValueError, match="runtime outcome"):
        replace(collection, result=replace(collection.result, runtime=runtime))


def test_collection_refuses_missing_nonterminal_and_untyped_fields():
    collection = collection_result()
    for observation in (
        None,
        {},
        SubmissionObservation(ObservationStatus.TERMINAL),
        replace(
            collection.observation,
            status=ObservationStatus.RUNNING,
            exit_code=None,
            signal=None,
        ),
    ):
        with pytest.raises(ValueError):
            replace(collection, observation=observation)
    for data in (
        collection.to_dict() | {"extra": 1},
        collection.to_dict() | {"result": {}},
        collection.to_dict() | {"identity": []},
    ):
        with pytest.raises((ValueError, TypeError)):
            SubmissionCollectionResult.from_dict(data)


@pytest.mark.parametrize(
    "contract", [SubmissionObservation, SubmissionCollectionResult]
)
def test_submission_evidence_json_rejects_duplicates_nonfinite_and_byte_overflow(
    contract,
):
    for raw in (
        '{"status":"queued","status":"queued"}',
        '{"nested":{"x":1,"x":2}}',
        '{"x":NaN}',
        '{"blob":"' + "é" * 131072 + '"}',
        "[]",
    ):
        with pytest.raises((ValueError, TypeError)):
            contract.from_json(raw)
    collection = collection_result()
    with pytest.raises(ValueError, match="byte bound"):
        replace(collection.observation, detail="é" * 131072)
    with pytest.raises(ValueError, match="byte bound"):
        replace(
            collection,
            result=replace(
                collection.result,
                batch=OperationOutcome(OutcomeStatus.UNKNOWN, detail="é" * 131072),
            ),
        )
