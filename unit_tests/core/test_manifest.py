import errno
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Barrier

import pytest

from wavedl.core import persistence
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import (
    RuntimeSelection,
    StorageMapping,
    build_manifest,
    read_manifest,
    write_manifest,
)


RUN_ID = "00000000-0000-4000-8000-000000000001"


def plan(tmp_path, **changes):
    layout = RunLayout.create(tmp_path / "runs", RUN_ID)
    options = {
        "operation": OperationKind.TRAIN,
        "resolved_config": {
            "data_path": "dataset.npz",
            "config": "config.yaml",
            "lr": 0.01,
        },
        "runtime": RuntimeSelection("venv/python", "existing", "cpu", {}),
        "submission_cwd": tmp_path,
    }
    return layout, build_manifest(layout, **(options | changes))


def test_relative_paths_resolve_before_execution_changes_cwd(tmp_path, monkeypatch):
    config = {
        "data_path": "data/train.npz",
        "config": "cfg/train.yaml",
        "resume": "old/best",
        "lr": 0.02,
    }
    layout, manifest = plan(tmp_path, resolved_config=config)
    monkeypatch.chdir(layout.run_dir)
    assert manifest.resolved_config["data_path"] == str(tmp_path / "data/train.npz")
    assert manifest.resolved_config["config"] == str(tmp_path / "cfg/train.yaml")
    assert manifest.resolved_config["resume"] == str(tmp_path / "old/best")
    assert manifest.resolved_config["lr"] == 0.02
    assert manifest.runtime.python == str(tmp_path / "venv/python")
    assert manifest.working_directory == str(layout.run_dir)
    assert manifest.resolved_config["output_dir"] == str(layout.run_dir)
    assert config["data_path"] == "data/train.npz"


def test_interpreter_symlink_is_not_dereferenced(tmp_path):
    """A venv's bin/python is a symlink; following it selects another runtime."""
    base = tmp_path / "base" / "bin"
    base.mkdir(parents=True)
    (base / "python3").write_text("#!/bin/sh\n")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(base / "python3")

    _, manifest = plan(
        tmp_path, runtime=RuntimeSelection("venv/bin/python", "existing", "cpu", {})
    )

    assert manifest.runtime.python == str(venv_bin / "python")


def test_interpreter_is_still_absolute_and_mapped(tmp_path):
    """Not dereferencing must not cost absoluteness or storage mapping."""
    execution_root = str(tmp_path / "execution")
    _, manifest = plan(
        tmp_path,
        runtime=RuntimeSelection("venv/bin/python", "existing", "cpu", {}),
        storage_map=(StorageMapping(str(tmp_path), execution_root),),
    )

    assert manifest.runtime.python == str(
        Path(execution_root) / "venv" / "bin" / "python"
    )


def test_storage_mapping_and_runtime_selection_are_persisted(tmp_path):
    runtime = RuntimeSelection(
        "venv/python", "container", "cuda", {"CUDA_VISIBLE_DEVICES": "0,1"}, "image.sif"
    )
    execution_root = str(tmp_path / "execution")
    layout, manifest = plan(
        tmp_path,
        runtime=runtime,
        storage_map=(StorageMapping(str(tmp_path), execution_root),),
    )
    write_manifest(layout, manifest)
    restored = read_manifest(layout.manifest)
    assert restored == manifest
    assert restored.runtime.environment == {"CUDA_VISIBLE_DEVICES": "0,1"}
    assert restored.runtime.container == "image.sif"
    assert restored.resolved_config["data_path"] == str(
        Path(execution_root) / "dataset.npz"
    )
    assert restored.working_directory == str(Path(execution_root) / "runs" / RUN_ID)


def test_host_mapping_alias_persists_the_canonical_root_actually_used(tmp_path):
    host = tmp_path / "actual-storage"
    host.mkdir()
    alias = tmp_path / "host-link"
    alias.symlink_to(host, target_is_directory=True)
    layout = RunLayout.create(host / "runs", RUN_ID)
    execution = str(tmp_path / "execution-storage")
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={"data_path": str(alias / "data.npz")},
        runtime=RuntimeSelection(sys.executable, "selected", "cpu", {}),
        submission_cwd=tmp_path,
        storage_map=(StorageMapping(str(alias), execution),),
    )
    assert manifest.storage_map == (StorageMapping(str(host.resolve()), execution),)
    assert manifest.working_directory == str(Path(execution) / "runs" / RUN_ID)
    write_manifest(layout, manifest)
    assert read_manifest(layout.manifest) == manifest


@pytest.mark.skipif(sys.platform != "darwin", reason="Actual macOS /var alias")
def test_macos_var_alias_in_map_is_canonical_approved_provenance(tmp_path):
    canonical = tmp_path.resolve()
    if not str(canonical).startswith("/private/var/"):
        pytest.skip("Temporary directory does not live under /private/var")
    alias = Path(str(canonical).removeprefix("/private"))
    assert alias.resolve() == canonical
    layout, manifest = plan(
        tmp_path, storage_map=(StorageMapping(str(alias), "/node/shared"),)
    )
    assert manifest.storage_map[0].host_root == str(canonical)
    assert manifest.working_directory == str(Path("/node/shared") / "runs" / RUN_ID)
    write_manifest(layout, manifest)
    assert read_manifest(layout.manifest).storage_map == manifest.storage_map


def test_equivalent_host_alias_mappings_are_rejected_before_rewrite(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(host, target_is_directory=True)
    with pytest.raises(ValueError, match="Duplicate"):
        plan(
            tmp_path,
            storage_map=(
                StorageMapping(str(host), str(tmp_path / "execution/a")),
                StorageMapping(str(alias), str(tmp_path / "execution/b")),
            ),
        )


def test_execution_root_remains_lexical_and_is_not_host_dereferenced(tmp_path):
    actual = tmp_path / "unrelated-host-target"
    actual.mkdir()
    execution = tmp_path / "execution-link"
    execution.symlink_to(actual, target_is_directory=True)
    _, manifest = plan(
        tmp_path, storage_map=(StorageMapping(str(tmp_path), str(execution)),)
    )
    assert manifest.storage_map[0].execution_root == str(execution)
    assert manifest.working_directory == str(execution / "runs" / RUN_ID)


def test_overwrite_refused_and_attempt_index_does_not_mutate_plan(tmp_path):
    layout, manifest = plan(tmp_path)
    write_manifest(layout, manifest)
    original = layout.manifest.read_bytes()
    assert allocate_attempt(layout) == 1
    assert allocate_attempt(layout) == 2
    assert manifest.attempt_index == "attempts"
    with pytest.raises(FileExistsError):
        write_manifest(layout, manifest)
    assert layout.manifest.read_bytes() == original


def test_interruption_before_link_exposes_no_partial_manifest(tmp_path, monkeypatch):
    """An fsync error after bytes are written cannot expose manifest.json."""
    layout, manifest = plan(tmp_path)

    def fail(fd):
        raise OSError(errno.ENOSPC, "full at file fsync")

    monkeypatch.setattr(persistence.os, "fsync", fail)
    with pytest.raises(OSError, match="full at file fsync"):
        write_manifest(layout, manifest)
    assert not layout.manifest.exists()
    assert not list(layout.run_dir.glob(".manifest.json-*"))


def test_simultaneous_publication_has_one_complete_winner(tmp_path, monkeypatch):
    """Two publishers race at the exclusive link boundary, not at exists()."""
    layout, manifest = plan(tmp_path)
    barrier = Barrier(2)
    real_link = os.link
    observed = []

    def link(source, destination):
        assert not layout.manifest.exists()
        observed.append(json.loads(Path(source).read_bytes())["run_id"])
        barrier.wait(timeout=5)
        return real_link(source, destination)

    monkeypatch.setattr(persistence.os, "link", link)

    def publish():
        try:
            write_manifest(layout, manifest)
        except FileExistsError:
            return "exists"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish) for _ in range(2)]
        results = sorted(future.result(timeout=10) for future in futures)
    assert results == ["created", "exists"]
    assert observed == [RUN_ID, RUN_ID]
    assert read_manifest(layout.manifest) == manifest


def test_process_death_before_publication_leaves_only_hidden_temporary(tmp_path):
    """Abrupt death bypasses finally, yet readers cannot see a partial manifest."""
    layout, manifest = plan(tmp_path)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    script = """
import os
import sys
from pathlib import Path
from wavedl.core import persistence
from wavedl.core.layout import RunLayout
from wavedl.core.manifest import read_manifest, write_manifest
manifest = read_manifest(Path(sys.argv[1]))
layout = RunLayout(Path(manifest.run_directory), manifest.run_id)
def die(source, target):
    os._exit(73)
persistence.os.link = die
write_manifest(layout, manifest)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(input_path)],
        env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
        timeout=15,
        capture_output=True,
    )
    assert result.returncode == 73, result.stderr
    assert not layout.manifest.exists()
    temporary = list(layout.run_dir.glob(".manifest.json-*"))
    assert len(temporary) == 1
    assert json.loads(temporary[0].read_bytes())["run_id"] == RUN_ID
    write_manifest(layout, manifest)
    assert read_manifest(layout.manifest) == manifest


def test_invalid_configuration_does_not_create_manifest(tmp_path):
    with pytest.raises(ValueError):
        plan(tmp_path, resolved_config={"lr": float("inf")})
    assert not list(tmp_path.rglob("manifest.json"))


def test_partial_temporary_write_never_exposes_manifest(tmp_path, monkeypatch):
    """Fail inside writing after half the JSON bytes reached the temporary file."""
    layout, manifest = plan(tmp_path)
    real_fdopen = persistence.os.fdopen
    observed = []

    class InterruptedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[: len(data) // 2])
            self.stream.flush()
            temporary = list(layout.run_dir.glob(".manifest.json-*"))
            observed.append(temporary[0].read_bytes())
            assert not layout.manifest.exists()
            raise OSError(errno.ENOSPC, "interrupted JSON write")

    def fdopen(fd, mode):
        return InterruptedFile(real_fdopen(fd, mode))

    monkeypatch.setattr(persistence.os, "fdopen", fdopen)
    with pytest.raises(OSError, match="interrupted JSON write"):
        write_manifest(layout, manifest)
    assert len(observed) == 1 and observed[0]
    with pytest.raises(json.JSONDecodeError):
        json.loads(observed[0])
    assert not layout.manifest.exists()
    assert not list(layout.run_dir.glob(".manifest.json-*"))
