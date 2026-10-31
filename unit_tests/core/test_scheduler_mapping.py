"""Declared view validation; mirrored fixtures are not mount-coherence proof."""

import json
import os
import shutil
import sys
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import PureWindowsPath
from types import SimpleNamespace
from uuid import uuid4

import pytest

from wavedl.core import scheduler_attempt
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import (
    RuntimeSelection,
    StorageMapping,
    build_manifest,
    write_manifest,
)
from wavedl.core.persistence import write_once
from wavedl.core.scheduler_attempt import (
    SchedulerIdentity,
    read_scheduler_binding,
    validate_scheduler_view,
)
from wavedl.core.training_cli import parse_training_args
from wavedl.journal import bootstrap_run
from wavedl.launch_runtime import bind_scheduler_startup, prepare_runtime


posix_mapping = pytest.mark.skipif(os.name != "posix", reason="POSIX scheduler mapping")


def options():
    return Namespace(
        mixed_precision="bf16",
        original_argv=(),
        cpu=True,
        machine_rank=0,
        num_machines=1,
        launch_id=None,
    )


@posix_mapping
def test_mapped_execution_binding_is_readable_from_both_declared_views(tmp_path):
    run_id, request = str(uuid4()), str(uuid4())
    host = RunLayout.create(tmp_path / "daemon", run_id)
    execution = RunLayout.create(tmp_path / "compute", run_id)
    args = [
        "--output_dir",
        str(execution.run_dir),
        "--data_path",
        str(tmp_path / "data.npz"),
        "--epochs",
        "1",
    ]
    config, _ = parse_training_args(args)
    runtime = RuntimeSelection(sys.executable, "selected", "cpu", {})
    manifest = build_manifest(
        host,
        operation=OperationKind.TRAIN,
        resolved_config=vars(config),
        runtime=runtime,
        submission_cwd=tmp_path,
        storage_map=(
            StorageMapping(str(host.run_dir.parent), str(execution.run_dir.parent)),
        ),
    )
    assert validate_scheduler_view(manifest, host.run_dir) == "host"
    assert validate_scheduler_view(manifest, execution.run_dir) == "execution"
    for layout in (host, execution):
        write_manifest(layout, manifest)
        assert allocate_attempt(layout) == 1
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
        write_once(
            layout.attempt_dir(1) / "runtime_request.json",
            json.dumps(asdict(manifest)).encode(),
        )
    env = {
        "WAVEDL_RUN_ID": run_id,
        "WAVEDL_REQUEST_ID": request,
        "WAVEDL_ATTEMPT": "1",
        "WAVEDL_SCHEDULER_BINDING": "1",
        "WAVEDL_SCHEDULER_INITIAL_ATTEMPT": "1",
        "WAVEDL_SCHEDULER_OWNER": "alice",
        "WAVEDL_SCHEDULER_CLUSTER": "mapped",
        "SLURM_JOB_ID": "19",
        "SLURM_RESTART_COUNT": "1",
    }
    binding = bind_scheduler_startup(execution.run_dir, env)
    assert binding.attempt == 2
    # Explicitly simulate completed shared-storage visibility. These are separate
    # directories, so this checks declarations/bytes, not NFS/Lustre coherence.
    shutil.copytree(
        execution.run_dir / "scheduler_attempts", host.run_dir / "scheduler_attempts"
    )
    shutil.copytree(execution.attempt_dir(2), host.attempt_dir(2))
    identity = SchedulerIdentity(run_id, request, "mapped", "19", "alice", 1)
    assert read_scheduler_binding(host.run_dir, identity, initial_attempt=1) == binding
    assert (
        read_scheduler_binding(execution.run_dir, identity, initial_attempt=1)
        == binding
    )
    env["WAVEDL_ATTEMPT"] = "2"
    _, forwarded = prepare_runtime(
        args, options(), python=sys.executable, environment=env
    )
    worker = bootstrap_run(
        execution.run_dir,
        resolved_config=vars(config),
        runtime=runtime,
        environment=forwarded,
        is_main_process=False,
        rank=3,
    )
    assert worker.attempt == 2
    assert worker.manifest == manifest
    assert worker.request_manifest == manifest


@posix_mapping
def test_unmapped_working_directory_and_conflicting_normalized_roots_refused(tmp_path):
    run_id = str(uuid4())
    layout = RunLayout(tmp_path / "host" / run_id, run_id)
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "selected", "cpu", {}),
        submission_cwd=tmp_path,
    )
    wrong = str(tmp_path / "unrelated")
    forged = replace(
        manifest,
        working_directory=wrong,
        resolved_config=manifest.resolved_config | {"output_dir": wrong},
    )
    with pytest.raises(ValueError, match="exact declared mapping"):
        validate_scheduler_view(forged, layout.run_dir)
    duplicate = replace(
        manifest,
        storage_map=(
            StorageMapping(str(tmp_path), str(tmp_path)),
            StorageMapping(str(tmp_path / "child" / ".."), str(tmp_path / "compute")),
        ),
    )
    with pytest.raises(ValueError, match="Ambiguous normalized"):
        validate_scheduler_view(duplicate, layout.run_dir)
    with pytest.raises(ValueError, match="outside the approved"):
        validate_scheduler_view(manifest, tmp_path / "other")


@pytest.mark.parametrize(
    ("ordinary", "extended"),
    [
        (r"C:\runs\job", r"\\?\C:\runs\job"),
        (r"c:\RUNS\job", r"\\?\C:\runs\JOB"),
        (r"\\server\share\runs\job", r"\\?\UNC\server\share\runs\job"),
    ],
)
def test_windows_storage_namespace_comparison_accepts_only_equivalent_spellings(
    ordinary, extended
):
    assert scheduler_attempt._windows_scheduler_path(extended) == PureWindowsPath(
        ordinary
    )


@pytest.mark.parametrize(
    "path",
    [
        r"C:relative",
        r"\rooted",
        "relative",
        r"\\.\C:\runs\job",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1\runs\job",
        r"\\?\Volume{123}\runs\job",
        r"\\?\UNC\server",
        r"\\server",
    ],
)
def test_windows_storage_namespace_comparison_rejects_device_and_relative_paths(path):
    with pytest.raises(ValueError):
        scheduler_attempt._windows_scheduler_path(path)


def test_windows_exact_mapping_keeps_component_authority(monkeypatch):
    # Supply already-resolved Windows paths on every host; native integration
    # still exercises real final-path handles through the submission tests.
    monkeypatch.setattr(scheduler_attempt, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        scheduler_attempt,
        "Path",
        lambda value: SimpleNamespace(resolve=lambda: PureWindowsPath(value)),
    )
    manifest = SimpleNamespace(
        run_directory=r"C:\host\job",
        working_directory=r"D:\compute\job",
        storage_map=[
            SimpleNamespace(host_root=r"C:\host", execution_root=r"D:\compute")
        ],
    )
    assert validate_scheduler_view(manifest, r"\\?\c:\HOST\job") == "host"
    assert validate_scheduler_view(manifest, r"\\?\D:\compute\job") == "execution"
    manifest.run_directory = r"\\?\C:\host\job"
    manifest.working_directory = r"\\?\D:\compute\job"
    manifest.storage_map = [
        SimpleNamespace(host_root=r"\\?\C:\host", execution_root=r"\\?\D:\compute")
    ]
    assert validate_scheduler_view(manifest, r"C:\host\job") == "host"
    assert validate_scheduler_view(manifest, r"D:\compute\job") == "execution"
    for foreign in (r"C:\hostile\job", r"E:\compute\job", r"D:\compute\job2"):
        with pytest.raises(ValueError, match="outside the approved"):
            validate_scheduler_view(manifest, foreign)
    manifest.working_directory = r"D:\compute\job2"
    with pytest.raises(ValueError, match="exact declared mapping"):
        validate_scheduler_view(manifest, r"C:\host\job")
    manifest.working_directory = r"D:\compute\job"
    manifest.storage_map.append(
        SimpleNamespace(host_root=r"\\?\c:\HOST", execution_root=r"E:\other")
    )
    with pytest.raises(ValueError, match="Ambiguous normalized"):
        validate_scheduler_view(manifest, r"C:\host\job")


def test_windows_unc_mapping_does_not_alias_other_server_or_share(monkeypatch):
    monkeypatch.setattr(scheduler_attempt, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        scheduler_attempt,
        "Path",
        lambda value: SimpleNamespace(resolve=lambda: PureWindowsPath(value)),
    )
    manifest = SimpleNamespace(
        run_directory=r"\\server\share\job",
        working_directory=r"\\server\share\job",
        storage_map=[],
    )
    assert validate_scheduler_view(manifest, r"\\?\UNC\SERVER\SHARE\job") == "host"
    for foreign in (r"\\other\share\job", r"\\server\other\job"):
        with pytest.raises(ValueError, match="outside the approved"):
            validate_scheduler_view(manifest, foreign)


@posix_mapping
def test_posix_scheduler_paths_do_not_rewrite_windows_literal_names(tmp_path):
    path = tmp_path / r"\\?\C:\runs\job"
    assert scheduler_attempt._scheduler_path(path) == path.resolve()
