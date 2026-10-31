"""Real CPU startup proves Local writer identity before attempt observation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.local_attempt import LocalStartupBinding
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.protocol import decode
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    OutcomeStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.backends.local_execution import SupervisionLost
from wavedl_daemon.storage.ownership import DispatchLock


pytest.importorskip("torch", reason="Actual Local CPU runtime requires ML extras")
pytest.importorskip("accelerate", reason="Actual Local training requires ML extras")


@pytest.fixture
def writer(tmp_path):
    with DispatchLock(tmp_path / "dispatch.lock") as dispatch:
        backend = LocalBackend(dispatch_lock=dispatch, owner="writer-controller")
        try:
            yield backend
        finally:
            backend.close()


def prepared(tmp_path, *, epochs=1):
    data = tmp_path / "data.npz"
    generator = np.random.default_rng(8)
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    layout = RunLayout.create(tmp_path / "runs", str(uuid4()))
    source = str(Path(__file__).resolve().parents[2] / "src")
    environment = {
        "PYTHONPATH": os.pathsep.join(filter(None, (source, os.getenv("PYTHONPATH")))),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={
            "data_path": str(data),
            "model": "cnn",
            "epochs": epochs,
            "workers": 0,
            "batch_size": 8,
            "precision": "no",
        },
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", environment),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    planner = LocalBackend()
    reserved = planner.plan(manifest)
    assert reserved.allocation_mode == "reserved"
    return layout, planner.plan(manifest, allocation_mode="writer")


def wait_binding(backend, handle):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        binding = backend.discover_initial(handle)
        if binding is not None:
            return binding
        if backend.process_result(handle) is not None:
            raise AssertionError(f"startup failed: {backend.process_result(handle)}")
        time.sleep(0.01)
    raise AssertionError("Local startup never published its binding")


def wait_terminal(backend, handle):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        observed = backend.observe(handle)
        if observed.status == ObservationStatus.TERMINAL:
            return observed
        time.sleep(0.01)
    raise AssertionError("Local runtime never became terminal")


def test_real_cpu_writer_native_receipt_binding_and_positive_observation(
    tmp_path, writer
):
    layout, plan = prepared(tmp_path)
    request = str(uuid4())
    assert not list(layout.attempts.iterdir())
    handle = writer.submit(plan, request_id=request, attempt=None)
    assert isinstance(handle, SubmissionHandle)
    assert "attempt" not in handle.to_dict()
    assert handle.pid > 0 and handle.supervisor_id
    assert isinstance(writer.observe(handle), SubmissionObservation)
    assert writer.submit(plan, request_id=request, attempt=None) == handle
    binding = wait_binding(writer, handle)
    assert isinstance(binding, LocalStartupBinding)
    assert binding.run_id == handle.run_id
    assert binding.request_id == request
    assert binding.owner == handle.owner
    assert binding.supervisor_id == handle.supervisor_id
    assert binding.plan_sha256 == hashlib.sha256(plan.to_json().encode()).hexdigest()
    assert binding.attempt == 1
    runtime = handle.bind_attempt(binding.attempt)
    observed = wait_terminal(writer, runtime)
    assert isinstance(observed, BackendObservation)
    assert observed.attempt == 1
    assert observed.batch.status == OutcomeStatus.SUCCESS, writer.process_result(
        runtime
    )
    journal = [
        decode(line)
        for line in layout.attempt_dir(1)
        .joinpath("events.jsonl")
        .read_bytes()
        .splitlines()
    ]
    assert journal[0].type == "hello"
    assert journal[0].payload["device"] == "cpu"
    assert journal[-1].payload["status"] == "completed"
    assert writer.discover_initial(handle) == binding
    assert writer.submit(plan, request_id=request, attempt=None) == handle
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]
    writer.release(runtime)
    with pytest.raises(SupervisionLost, match="never resubmit"):
        writer.submit(plan, request_id=request, attempt=None)
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]


def test_owned_unbound_submission_can_cancel_without_inventing_an_attempt(
    tmp_path, writer
):
    layout, plan = prepared(tmp_path, epochs=10000)
    handle = writer.submit(plan, request_id=str(uuid4()), attempt=None)
    binding = wait_binding(writer, handle)
    writer.request_cancel(handle)
    observed = wait_terminal(writer, handle)
    assert isinstance(observed, SubmissionObservation)
    assert observed.batch.status == OutcomeStatus.CANCELLED
    assert not hasattr(observed, "attempt")
    assert writer.observe(handle.bind_attempt(binding.attempt)).batch == observed.batch
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]


def test_foreign_receipt_and_foreign_startup_proof_cannot_authorize_binding(
    tmp_path, writer
):
    layout, plan = prepared(tmp_path)
    handle = writer.submit(plan, request_id=str(uuid4()), attempt=None)
    binding = wait_binding(writer, handle)
    wait_terminal(writer, handle)
    for changes in (
        {"owner": "foreign-owner"},
        {"supervisor_id": "foreign-supervisor"},
        {"pid": handle.pid + 1},
        {"run_id": str(uuid4())},
    ):
        foreign = replace(handle, **changes)
        with pytest.raises(ValueError, match="foreign or altered"):
            writer.discover_initial(foreign)
        with pytest.raises(ValueError, match="foreign or altered"):
            writer.request_cancel(foreign)
    proof = layout.run_dir / "submissions" / handle.request_id / "startup.json"
    original = proof.read_bytes()
    for changes in (
        {"owner": "foreign-owner"},
        {"supervisor_id": "foreign-supervisor"},
        {"request_id": str(uuid4())},
        {"plan_sha256": "0" * 64},
    ):
        proof.write_text(json.dumps(json.loads(original) | changes))
        try:
            with pytest.raises((ValueError, FileNotFoundError)):
                writer.discover_initial(handle)
        finally:
            proof.write_bytes(original)
    proof.write_text(
        json.dumps(json.loads(original) | {"attempt": binding.attempt + 1})
    )
    try:
        # Missing positive-attempt proof is pending, never usable authority.
        assert writer.discover_initial(handle) is None
    finally:
        proof.write_bytes(original)
    assert writer.discover_initial(handle) == binding
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]


def test_reopened_backend_retains_no_resubmit_fence_not_pid_authority(tmp_path, writer):
    layout, plan = prepared(tmp_path)
    handle = writer.submit(plan, request_id=str(uuid4()), attempt=None)
    binding = wait_binding(writer, handle)
    wait_terminal(writer, handle)
    reopened = LocalBackend(dispatch_lock=writer.dispatch_lock, owner=handle.owner)
    try:
        for receipt in (handle, handle.bind_attempt(binding.attempt)):
            assert (
                reopened.observe(receipt).status == ObservationStatus.SUPERVISION_LOST
            )
            with pytest.raises(SupervisionLost):
                reopened.request_cancel(receipt)
        with pytest.raises(SupervisionLost):
            reopened.discover_initial(SubmissionHandle.from_json(handle.to_json()))
        with pytest.raises(SupervisionLost, match="never resubmit"):
            reopened.submit(plan, request_id=handle.request_id, attempt=None)
        assert [path.name for path in layout.attempts.iterdir()] == ["1"]
    finally:
        reopened.close()


def test_spawn_failure_retains_unbound_fence_and_never_retries(
    tmp_path, writer, monkeypatch
):
    layout, plan = prepared(tmp_path)
    request = str(uuid4())
    start = writer._supervisor.start
    launches = []

    def failing_start(spec, **kwargs):
        launches.append(spec.argv)
        if "wavedl.local_startup" in spec.argv:
            raise OSError("runtime spawn failed")
        return start(spec, **kwargs)

    monkeypatch.setattr(writer._supervisor, "start", failing_start)
    with pytest.raises(OSError, match="runtime spawn failed"):
        writer.submit(plan, request_id=request, attempt=None)
    assert len(launches) == 2
    assert (layout.run_dir / "submissions" / request).is_dir()
    assert not list(layout.attempts.iterdir())
    with pytest.raises(SupervisionLost, match="never resubmit"):
        writer.submit(plan, request_id=request, attempt=None)
    assert len(launches) == 2
    assert not list(layout.attempts.iterdir())
