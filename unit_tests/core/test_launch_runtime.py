"""Launcher snapshots precede spawn and are shared across all worker ranks."""

import hashlib
import json
import os
import sys
from argparse import Namespace
from dataclasses import asdict
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import (
    RuntimeSelection,
    build_manifest,
    read_manifest,
    write_manifest,
)
from wavedl.core.persistence import write_once
from wavedl.core.training_cli import parse_training_args
from wavedl.journal import bootstrap_run
from wavedl.launch_runtime import bind_scheduler_startup, prepare_runtime


def options(**changes):
    return Namespace(
        **(
            {
                "mixed_precision": "bf16",
                "original_argv": (),
                "cpu": True,
                "machine_rank": 0,
                "num_machines": 1,
                "launch_id": None,
            }
            | changes
        )
    )


@pytest.fixture
def writer_scheduler(tmp_path):
    if os.name != "posix":
        pytest.skip("Slurm writer allocation requires POSIX scheduler coordination")
    from wavedl.core.submission import (
        create_submission,
        identity_for_submission,
        publish_submission_file,
    )

    layout = RunLayout.create(tmp_path, str(uuid4()))
    args = [
        "--output_dir",
        str(layout.run_dir),
        "--data_path",
        str(tmp_path / "data.npz"),
        "--epochs",
        "1",
    ]
    config, _ = parse_training_args(args)
    runtime = RuntimeSelection(sys.executable, "selected", "cpu", {})
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config=vars(config),
        runtime=runtime,
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    plan = json.dumps(
        {
            "target": "slurm",
            "allocation_mode": "writer",
            "manifest_json": json.dumps(manifest.to_dict()),
            "resources": [["cluster", "cluster-a"]],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    request = str(uuid4())
    digest = hashlib.sha256(plan).hexdigest()
    submission = identity_for_submission(
        layout.run_dir,
        request_id=request,
        backend="slurm",
        owner="alice",
        plan_sha256=digest,
    )
    create_submission(layout.run_dir, submission)
    publish_submission_file(layout.run_dir, submission, "plan.json", plan)
    env = {
        "WAVEDL_RUN_ID": layout.run_id,
        "WAVEDL_REQUEST_ID": request,
        "WAVEDL_PLAN_SHA256": digest,
        "WAVEDL_ATTEMPT_ALLOCATION": "writer",
        "WAVEDL_SCHEDULER_BINDING": "1",
        "WAVEDL_SCHEDULER_OWNER": "alice",
        "WAVEDL_SCHEDULER_CLUSTER": "cluster-a",
        "SLURM_JOB_ID": "12345",
        "SLURM_RESTART_COUNT": "0",
    }
    return layout, args, config, runtime, env


def test_writer_scheduler_allocates_at_startup_and_propagates_rank_proof(
    writer_scheduler,
):
    layout, args, config, runtime, env = writer_scheduler
    approved = layout.manifest.read_bytes()
    assert list(layout.attempts.iterdir()) == []
    initial = bind_scheduler_startup(layout.run_dir, env)
    assert initial.attempt == 1
    assert bind_scheduler_startup(layout.run_dir, env) == initial
    assert (layout.attempt_dir(1) / "runtime_request.json").read_bytes() == approved
    selected = env | {"WAVEDL_ATTEMPT": "1"}
    _, main = prepare_runtime(
        args, options(), python=sys.executable, environment=selected
    )
    worker = bootstrap_run(
        layout.run_dir,
        resolved_config=vars(config),
        runtime=runtime,
        environment=main,
        is_main_process=False,
        rank=1,
    )
    assert worker.attempt == initial.attempt
    restarted = bind_scheduler_startup(
        layout.run_dir, env | {"SLURM_RESTART_COUNT": "1"}
    )
    assert restarted.attempt == 2
    assert (layout.attempt_dir(2) / "runtime_request.json").read_bytes() == approved
    assert layout.manifest.read_bytes() == approved
    assert not (layout.run_dir / "scheduler_attempts").exists()
    assert json.loads((layout.attempt_dir(1) / "runtime_identity.json").read_bytes())[
        "scheduler_binding"
    ] == asdict(initial)


@pytest.mark.parametrize(
    "changes",
    [
        {"WAVEDL_ATTEMPT": "1"},
        {"WAVEDL_PLAN_SHA256": "a" * 64},
        {"WAVEDL_PLAN_SHA256": None},
        {"WAVEDL_SCHEDULER_CLUSTER": "foreign"},
        {"WAVEDL_SCHEDULER_OWNER": "bob"},
    ],
)
def test_writer_scheduler_refuses_foreign_authority_before_allocation(
    writer_scheduler, changes
):
    layout, _, _, _, env = writer_scheduler
    with pytest.raises((ValueError, FileNotFoundError)):
        bind_scheduler_startup(layout.run_dir, env | changes)
    assert list(layout.attempts.iterdir()) == []


def test_writer_rank_requires_published_request(writer_scheduler):
    from wavedl.core.scheduler_attempt import (
        coordinate_attempt,
        scheduler_environment_identity,
    )

    layout, _, config, runtime, env = writer_scheduler
    identity, initial_attempt = scheduler_environment_identity(env)
    binding = coordinate_attempt(
        layout.run_dir, identity, initial_attempt=initial_attempt
    )
    with pytest.raises(ValueError, match="runtime request has not been published"):
        bootstrap_run(
            layout.run_dir,
            resolved_config=vars(config),
            runtime=runtime,
            environment=env | {"WAVEDL_ATTEMPT": str(binding.attempt)},
        )
    assert not (layout.attempt_dir(1) / "runtime_identity.json").exists()


def test_writer_scheduler_plan_corruption_never_allocates(writer_scheduler):
    layout, _, _, _, env = writer_scheduler
    plan = layout.run_dir / "submissions" / env["WAVEDL_REQUEST_ID"] / "plan.json"
    plan.write_bytes(plan.read_bytes() + b" ")
    with pytest.raises(ValueError, match="plan digest differs"):
        bind_scheduler_startup(layout.run_dir, env)
    assert list(layout.attempts.iterdir()) == []


@pytest.mark.parametrize("pin_present", [False, True])
def test_writer_scheduler_rejects_manifest_source_field_differing_from_approved_bytes(
    writer_scheduler, pin_present
):
    """Absent fields cannot be silently normalized when verifying byte authority."""
    from wavedl.core.submission import (
        create_submission,
        identity_for_submission,
        publish_submission_file,
    )

    layout, _, _, _, env = writer_scheduler
    data = json.loads(layout.manifest.read_bytes())
    data["checkpoint_source"] = (
        {
            "run_id": str(uuid4()),
            "generation": "gen-000001",
            "manifest_sha256": "a" * 64,
            "complete_sha256": "b" * 64,
            "files": {
                name: {"size": 1, "sha256": "c" * 64}
                for name in ("model_weights.pth", "training_meta.pkl", "scaler.pkl")
            },
            "selection": {"schema_version": 1, "kind": "best"},
        }
        if pin_present
        else None
    )
    plan = json.dumps(
        {
            "target": "slurm",
            "allocation_mode": "writer",
            "manifest_json": json.dumps(data),
            "resources": [["cluster", "cluster-a"]],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    request_id = str(uuid4())
    digest = hashlib.sha256(plan).hexdigest()
    submission = identity_for_submission(
        layout.run_dir,
        request_id=request_id,
        backend="slurm",
        owner="alice",
        plan_sha256=digest,
    )
    create_submission(layout.run_dir, submission)
    publish_submission_file(layout.run_dir, submission, "plan.json", plan)
    expected_error = (
        "Invalid scheduler writer manifest"
        if pin_present
        else "manifest differs from approved file"
    )
    with pytest.raises(ValueError, match=expected_error):
        bind_scheduler_startup(
            layout.run_dir,
            env | {"WAVEDL_REQUEST_ID": request_id, "WAVEDL_PLAN_SHA256": digest},
        )
    assert list(layout.attempts.iterdir()) == []


def test_writer_scheduler_never_overwrites_conflicting_runtime_request(
    writer_scheduler,
):
    layout, _, _, _, env = writer_scheduler
    binding = bind_scheduler_startup(layout.run_dir, env)
    request = layout.attempt_dir(binding.attempt) / "runtime_request.json"
    conflicting = b"not-approved"
    request.write_bytes(conflicting)
    with pytest.raises(ValueError, match="differs from initial request"):
        bind_scheduler_startup(layout.run_dir, env)
    assert request.read_bytes() == conflicting
    assert len(list(layout.attempts.iterdir())) == 1


def test_writer_scheduler_accepts_matching_concurrent_request_publication(
    writer_scheduler, monkeypatch
):
    from wavedl import launch_runtime

    layout, _, _, _, env = writer_scheduler
    publish = launch_runtime._publish

    def competing_publish(directory, name, raw):
        publish(directory, name, raw)
        if name == "runtime_request.json":
            raise FileExistsError("A matching participant published first")

    monkeypatch.setattr(launch_runtime, "_publish", competing_publish)
    binding = bind_scheduler_startup(layout.run_dir, env)
    assert binding.attempt == 1
    assert (layout.attempt_dir(1) / "runtime_request.json").read_bytes() == (
        layout.manifest.read_bytes()
    )


def test_scheduler_allocation_coordinator_binds_all_launchers_and_workers(tmp_path):
    layout = RunLayout.create(tmp_path, str(uuid4()))
    args = [
        "--output_dir",
        str(layout.run_dir),
        "--data_path",
        str(tmp_path / "data.npz"),
        "--epochs",
        "1",
    ]
    config, _ = parse_training_args(args)
    runtime = RuntimeSelection(sys.executable, "selected", "cpu", {})
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config=vars(config),
        runtime=runtime,
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    assert allocate_attempt(layout) == 1
    request = str(uuid4())
    write_once(
        layout.attempt_dir(1) / "submission.json",
        json.dumps(
            {
                "request_id": request,
                "attempt": 1,
                "owner": "alice",
                "created_at": "2026-10-04T00:00:00Z",
            }
        ).encode(),
    )
    original_request = json.dumps(asdict(manifest)).encode()
    write_once(layout.attempt_dir(1) / "runtime_request.json", original_request)
    env = {
        "WAVEDL_RUN_ID": layout.run_id,
        "WAVEDL_REQUEST_ID": request,
        "WAVEDL_ATTEMPT": "1",
        "WAVEDL_SCHEDULER_BINDING": "1",
        "WAVEDL_SCHEDULER_INITIAL_ATTEMPT": "1",
        "WAVEDL_SCHEDULER_OWNER": "alice",
        "WAVEDL_SCHEDULER_CLUSTER": "cluster-a",
        "SLURM_JOB_ID": "12345",
        "SLURM_RESTART_COUNT": "0",
    }
    if os.name != "posix":
        with pytest.raises(
            OSError, match="Scheduler startup coordination requires POSIX"
        ):
            bind_scheduler_startup(layout.run_dir, env)
        assert list(layout.attempts.iterdir()) == [layout.attempt_dir(1)]
        assert (layout.attempt_dir(1) / "runtime_request.json").read_bytes() == (
            original_request
        )
        assert not (layout.attempt_dir(1) / "runtime_identity.json").exists()
        return
    initial = bind_scheduler_startup(layout.run_dir, env)
    assert initial.attempt == 1
    _, first = prepare_runtime(
        args, options(num_machines=2), python=sys.executable, environment=env
    )
    env["SLURM_RESTART_COUNT"] = "1"
    restarted = bind_scheduler_startup(layout.run_dir, env)
    assert restarted.attempt == 2
    assert (
        layout.attempt_dir(2) / "runtime_request.json"
    ).read_bytes() == original_request
    with pytest.raises(ValueError, match="differs from scheduler"):
        prepare_runtime(args, options(), python=sys.executable, environment=env)
    env["WAVEDL_ATTEMPT"] = "2"
    _, main = prepare_runtime(
        args, options(num_machines=2), python=sys.executable, environment=env
    )
    _, follower = prepare_runtime(
        args,
        options(num_machines=2, machine_rank=1),
        python=sys.executable,
        environment=env,
    )
    assert main["WAVEDL_ATTEMPT"] == follower["WAVEDL_ATTEMPT"] == "2"
    worker = bootstrap_run(
        layout.run_dir,
        resolved_config=vars(config),
        runtime=runtime,
        environment=main,
        is_main_process=False,
        rank=3,
    )
    assert worker.attempt == 2
    receipt = json.loads((layout.attempt_dir(2) / "runtime_identity.json").read_bytes())
    assert receipt["scheduler_binding"] == asdict(restarted)
    assert receipt["restart_count"] == 1
    assert json.loads((layout.attempt_dir(1) / "runtime_identity.json").read_bytes())[
        "scheduler_binding"
    ] == asdict(initial)
    assert first["WAVEDL_ATTEMPT"] == "1"
    assert len(list(layout.attempts.iterdir())) == 2
    forged = main | {"SLURM_JOB_ID": "12346"}
    with pytest.raises((ValueError, FileNotFoundError)):
        bootstrap_run(
            layout.run_dir,
            resolved_config=vars(config),
            runtime=runtime,
            environment=forged,
        )


def test_startup_cli_is_option_gated_and_exposes_no_environment_secrets(
    tmp_path, monkeypatch, capsys
):
    from wavedl import launch_runtime

    monkeypatch.delenv("WAVEDL_SCHEDULER_BINDING", raising=False)
    monkeypatch.setenv("WAVEDL_DAEMON_TOKEN", "ephemeral-test-secret")
    with pytest.raises(SystemExit) as result:
        launch_runtime.main(["scheduler-bind", str(tmp_path)])
    assert result.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "ephemeral-test-secret" not in captured.err
    assert not list(tmp_path.iterdir())


def test_launcher_publishes_once_and_does_not_record_ambient_secrets(tmp_path):
    args = ["--output_dir", str(tmp_path), "--epochs", "3"]
    _, env = prepare_runtime(
        args,
        options(),
        python=sys.executable,
        environment={"TOKEN": "unique-sensitive-test-value"},
    )
    manifest = read_manifest(tmp_path / "manifest.json")
    assert manifest.runtime.environment == {}
    assert "unique-sensitive-test-value" not in (tmp_path / "manifest.json").read_text()
    assert env["TOKEN"] == "unique-sensitive-test-value"
    assert env["WAVEDL_RUN_ID"] == manifest.run_id
    assert env["WAVEDL_ATTEMPT"] == "1"
    assert (
        json.loads((tmp_path / "attempts/1/runtime_request.json").read_bytes())[
            "resolved_config"
        ]["epochs"]
        == 3
    )
    _, same = prepare_runtime(args, options(), python=sys.executable, environment=env)
    assert same["WAVEDL_ATTEMPT"] == "1"
    assert list((tmp_path / "attempts").iterdir()) == [tmp_path / "attempts/1"]


def test_yaml_cli_defaults_are_resolved_before_bootstrap(tmp_path):
    pytest.importorskip("yaml")
    yaml = tmp_path / "config.yaml"
    yaml.write_text("epochs: 99\nworkers: 7\nprecision: 'no'\n")
    args = ["--output_dir", str(tmp_path), "--config", str(yaml), "--workers", "-1"]
    opts = options()
    prepare_runtime(args, opts, python=sys.executable, environment={})
    config = read_manifest(tmp_path / "manifest.json").resolved_config
    assert config["epochs"] == 99
    assert config["workers"] == -1
    assert opts.mixed_precision == "no"


def test_launcher_precision_is_forwarded_and_conflict_refused(tmp_path):
    opts = options(original_argv=("--mixed_precision", "no"), mixed_precision="no")
    args, _ = prepare_runtime(
        ["--output_dir", str(tmp_path)], opts, python=sys.executable, environment={}
    )
    assert args[-2:] == ["--precision", "no"]
    with pytest.raises(ValueError, match="precision"):
        prepare_runtime(
            ["--output_dir", str(tmp_path), "--precision", "fp16"],
            opts,
            python=sys.executable,
            environment={},
        )


def test_standalone_two_node_rendezvous_propagates_identical_attempt(tmp_path):
    launch_id = str(uuid4())
    args = ["--output_dir", str(tmp_path)]
    _, main = prepare_runtime(
        args,
        options(num_machines=2, launch_id=launch_id),
        python=sys.executable,
        environment={},
    )
    _, follower = prepare_runtime(
        args,
        options(num_machines=2, machine_rank=1, launch_id=launch_id),
        python=sys.executable,
        environment={},
    )
    for name in ("WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT"):
        assert follower[name] == main[name]
    with pytest.raises(ValueError, match="already"):
        prepare_runtime(
            args,
            options(num_machines=2, launch_id=launch_id),
            python=sys.executable,
            environment={},
        )


def test_missing_main_node_never_generates_follower_identity(tmp_path):
    with pytest.raises(TimeoutError):
        prepare_runtime(
            ["--output_dir", str(tmp_path)],
            options(num_machines=2, machine_rank=1, launch_id=str(uuid4())),
            python=sys.executable,
            environment={},
            timeout_seconds=0.02,
        )
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("attempt", ["../foreign", "0", "01", "true"])
def test_invalid_injected_attempt_refused_before_path_read(tmp_path, attempt):
    with pytest.raises(ValueError, match="attempt"):
        prepare_runtime(
            ["--output_dir", str(tmp_path)],
            options(),
            python=sys.executable,
            environment={
                "WAVEDL_RUN_ID": str(uuid4()),
                "WAVEDL_REQUEST_ID": str(uuid4()),
                "WAVEDL_ATTEMPT": attempt,
            },
        )
