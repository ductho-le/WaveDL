"""The Local bootstrap consumes approved submission authority before runtime."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.local_attempt import LocalStartupIdentity, read_local_binding
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.protocol import decode
from wavedl.core.submission import (
    create_submission,
    identity_for_submission,
    publish_submission_file,
)
from wavedl.local_startup import prepare_local_startup
from wavedl_daemon.backends.contracts import Plan


RUN = "00000000-0000-4000-8000-000000000001"
REQUEST = "00000000-0000-4000-8000-000000000002"


def submission(tmp_path, *, argv=None, config=None):
    layout = RunLayout.create(tmp_path, RUN)
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config=config or {"epochs": 1},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    plan = Plan(
        "local",
        json.dumps(asdict(manifest)),
        argv=argv or (sys.executable, "-u", "-m", "wavedl.launcher"),
        allocation_mode="writer",
    )
    content = plan.to_json().encode()
    identity = identity_for_submission(
        layout.run_dir,
        request_id=REQUEST,
        backend="local",
        owner="-owner-token",
        plan_sha256=hashlib.sha256(content).hexdigest(),
    )
    namespace = create_submission(layout.run_dir, identity)
    publish_submission_file(layout.run_dir, identity, "plan.json", content)
    publish_submission_file(
        layout.run_dir,
        identity,
        "local-submission.json",
        json.dumps(
            {
                "schema_version": 1,
                "run_id": RUN,
                "request_id": REQUEST,
                "owner": identity.owner,
                "supervisor_id": "-supervisor-token",
                "plan_sha256": identity.plan_sha256,
            }
        ).encode(),
    )
    environment = {
        "WAVEDL_ATTEMPT_ALLOCATION": "writer",
        "WAVEDL_RUN_ID": RUN,
        "WAVEDL_REQUEST_ID": REQUEST,
        "WAVEDL_OWNER": identity.owner,
        "WAVEDL_SUPERVISOR_ID": "-supervisor-token",
        "WAVEDL_PLAN_SHA256": identity.plan_sha256,
    }
    return layout, namespace, plan, environment


def test_bootstrap_allocates_and_propagates_one_writer_binding(tmp_path):
    layout, _, plan, environment = submission(tmp_path)
    assert not list(layout.attempts.iterdir())
    original = dict(environment)
    command, selected, binding = prepare_local_startup(layout.run_dir, environment)
    assert command == plan.argv
    assert selected["WAVEDL_ATTEMPT"] == "1"
    assert selected["WAVEDL_RUN_DIR"] == str(layout.run_dir)
    assert environment == original and "WAVEDL_ATTEMPT" not in environment
    expected = LocalStartupIdentity(
        **{key: value for key, value in asdict(binding).items() if key != "attempt"}
    )
    assert binding == read_local_binding(layout.run_dir, expected)
    assert prepare_local_startup(layout.run_dir, environment)[2] == binding
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]


@pytest.mark.parametrize("changed", [None, "owner", "plan_sha256", "prefix"])
def test_probe_proof_checked_before_attempt_allocation(tmp_path, changed):
    layout, namespace, plan, environment = submission(tmp_path)
    identity = {
        "interpreter": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }
    proof = {
        "schema_version": 1,
        "selected_python": plan.manifest.runtime.python,
        "run_id": RUN,
        "request_id": REQUEST,
        "owner": environment["WAVEDL_OWNER"],
        "supervisor_id": environment["WAVEDL_SUPERVISOR_ID"],
        "plan_sha256": environment["WAVEDL_PLAN_SHA256"],
        "identity": identity,
    }
    if changed == "prefix":
        identity["prefix"] = "/different/environment"
    elif changed:
        proof[changed] = "foreign"
    (namespace / "runtime-probe.json").write_text(json.dumps(proof))
    (namespace / "runtime-probe.json").chmod(0o600)
    fence_path = namespace / "local-submission.json"
    fence = json.loads(fence_path.read_text())
    fence["runtime_probe_sha256"] = hashlib.sha256(
        (namespace / "runtime-probe.json").read_bytes()
    ).hexdigest()
    fence_path.write_text(json.dumps(fence))
    if changed:
        with pytest.raises(ValueError, match="differs"):
            prepare_local_startup(layout.run_dir, environment)
        assert not list(layout.attempts.iterdir())
    else:
        assert prepare_local_startup(layout.run_dir, environment)[2].attempt == 1


@pytest.mark.parametrize("damage", ["missing", "changed", "null"])
def test_new_receipt_missing_or_changed_probe_cannot_downgrade_to_legacy(
    tmp_path, damage
):
    layout, namespace, plan, environment = submission(tmp_path)
    proof = {
        "schema_version": 1,
        "selected_python": plan.manifest.runtime.python,
        "run_id": RUN,
        "request_id": REQUEST,
        "owner": environment["WAVEDL_OWNER"],
        "supervisor_id": environment["WAVEDL_SUPERVISOR_ID"],
        "plan_sha256": environment["WAVEDL_PLAN_SHA256"],
        "identity": {
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
    }
    raw = json.dumps(proof).encode()
    fence_path = namespace / "local-submission.json"
    fence = json.loads(fence_path.read_text())
    fence["runtime_probe_sha256"] = hashlib.sha256(raw).hexdigest()
    fence_path.write_text(json.dumps(fence))
    if damage != "missing":
        (namespace / "runtime-probe.json").write_bytes(
            b"null" if damage == "null" else raw + b" "
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_local_startup(layout.run_dir, environment)
    assert not list(layout.attempts.iterdir())


@pytest.mark.parametrize(
    "changes",
    [
        {"WAVEDL_ATTEMPT_ALLOCATION": "reserved"},
        {"WAVEDL_ATTEMPT": "1"},
        {"WAVEDL_RUN_ID": REQUEST},
        {"WAVEDL_REQUEST_ID": RUN},
        {"WAVEDL_OWNER": "foreign-owner"},
        {"WAVEDL_SUPERVISOR_ID": "foreign-supervisor"},
        {"WAVEDL_PLAN_SHA256": "0" * 64},
    ],
)
def test_foreign_or_conflicting_startup_identity_never_allocates(tmp_path, changes):
    layout, _, _, environment = submission(tmp_path)
    with pytest.raises((ValueError, OSError)):
        prepare_local_startup(layout.run_dir, environment | changes)
    assert not list(layout.attempts.iterdir())


def test_altered_approved_command_is_refused_before_writer_allocation(tmp_path):
    layout, _, _, environment = submission(
        tmp_path, argv=(sys.executable, "-u", "-m", "unapproved_module")
    )
    with pytest.raises(ValueError, match="command"):
        prepare_local_startup(layout.run_dir, environment)
    assert not list(layout.attempts.iterdir())


def test_changed_plan_file_is_refused_before_writer_allocation(tmp_path):
    layout, namespace, _, environment = submission(tmp_path)
    (namespace / "plan.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="digest"):
        prepare_local_startup(layout.run_dir, environment)
    assert not list(layout.attempts.iterdir())


def test_cli_startup_refusal_redacts_context_and_creates_no_attempt(tmp_path):
    layout, _, _, environment = submission(tmp_path)
    environment["WAVEDL_SUPERVISOR_ID"] = "raw-private-supervisor"
    completed = subprocess.run(
        [sys.executable, "-m", "wavedl.local_startup", str(layout.run_dir)],
        env=os.environ | environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 2
    assert "Local startup refused: ValueError" in completed.stderr
    assert "raw-private" not in completed.stderr
    assert not list(layout.attempts.iterdir())


@pytest.mark.integration
def test_real_selected_interpreter_cpu_runtime_starts_with_writer_attempt(tmp_path):
    import numpy as np

    generator = np.random.default_rng(42)
    data = tmp_path / "data.npz"
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    config = {
        "model": "cnn",
        "data_path": str(data),
        "epochs": 1,
        "workers": 0,
        "batch_size": 8,
        "precision": "no",
        "pretrained": False,
    }
    # The writer has no numeric reservation when the selected process starts.
    output = tmp_path / RUN
    argv = (
        sys.executable,
        "-u",
        "-m",
        "wavedl.launcher",
        "--cpu",
        "--num_gpus",
        "1",
        "--mixed_precision",
        "no",
        "--model",
        "cnn",
        "--data_path",
        str(data),
        "--epochs",
        "1",
        "--workers",
        "0",
        "--batch_size",
        "8",
        "--precision",
        "no",
        "--output_dir",
        str(output),
    )
    layout, _, _, environment = submission(tmp_path, argv=argv, config=config)
    assert layout.run_dir == output and not list(layout.attempts.iterdir())
    source = Path(__file__).resolve().parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "-m", "wavedl.local_startup", str(output)],
        cwd=tmp_path,
        env=os.environ
        | environment
        | {"PYTHONPATH": str(source), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = [decode(line) for line in layout.journal(1).read_bytes().splitlines()]
    assert records[0].type == "hello" and records[0].seq == 0
    runtime_identity = json.loads(
        (layout.attempt_dir(1) / "runtime_identity.json").read_text()
    )
    assert runtime_identity["request_id"] == REQUEST
    assert runtime_identity["attempt"] == 1
    assert records[-1].type == "exit" and records[-1].payload["status"] == "completed"
    assert (layout.run_dir / "submissions" / REQUEST / "startup.json").is_file()
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]
