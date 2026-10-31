import errno
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl.core import generations, persistence, safe_files
from wavedl.core.generations import publish_generation, published_generations
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import (
    RuntimeSelection,
    StorageMapping,
    build_manifest,
    write_manifest,
)


RUN_ID = "00000000-0000-4000-8000-000000000001"


def setup_run(tmp_path):
    layout = RunLayout.create(tmp_path / "runs", RUN_ID)
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    attempt = allocate_attempt(layout)
    source = tmp_path / "staging"
    source.mkdir()
    (source / "model.bin").write_bytes(b"weights")
    (source / "rank").mkdir()
    (source / "rank" / "state.bin").write_bytes(b"optimizer")
    return layout, attempt, source


def test_publication_works_on_the_execution_side_of_a_storage_map(tmp_path):
    """The compute node's layout is rooted at working_directory, not run_directory."""
    host_root = tmp_path / "home" / "runs"
    execution_root = tmp_path / "scratch" / "runs"
    host_root.mkdir(parents=True)
    execution_root.mkdir(parents=True)

    host = RunLayout.create(host_root, RUN_ID)
    manifest = build_manifest(
        host,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
        storage_map=(StorageMapping(str(host_root), str(execution_root)),),
    )
    write_manifest(host, manifest)
    assert manifest.run_directory != manifest.working_directory

    # The run travels to the compute node, where the same manifest is read back
    # from a layout rooted at the execution path.
    execution = RunLayout.create(execution_root, RUN_ID)
    execution.manifest.write_bytes(host.manifest.read_bytes())
    attempt = allocate_attempt(execution)
    source = tmp_path / "staging"
    source.mkdir()
    (source / "model.bin").write_bytes(b"weights")

    published = publish_generation(execution, source, attempt=attempt, step=10)

    assert published.path.is_relative_to(execution.checkpoints)
    assert [item.step for item in published_generations(execution)] == [10]


def test_publication_still_rejects_a_foreign_manifest(tmp_path):
    """Accepting both sides of a mapping must not accept an unrelated run."""
    layout, attempt, source = setup_run(tmp_path)
    other = RunLayout.create(tmp_path / "other", "00000000-0000-4000-8000-000000000002")
    foreign = build_manifest(
        other,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    write_manifest(other, foreign)
    other.manifest.replace(layout.manifest)  # a manifest for a different run

    with pytest.raises(ValueError, match="identity mismatch"):
        publish_generation(layout, source, attempt=attempt, step=10)


def test_publication_is_immutable_and_records_step(tmp_path):
    layout, attempt, source = setup_run(tmp_path)
    first = publish_generation(layout, source, attempt=attempt, step=10)
    original = (first.path / "model.bin").read_bytes()
    (source / "model.bin").write_bytes(b"new weights")
    second = publish_generation(layout, source, attempt=attempt, step=20)
    assert [item.step for item in published_generations(layout)] == [10, 20]
    assert (first.path / "model.bin").read_bytes() == original
    assert first.path != second.path
    marker = json.loads((first.path / "COMPLETE.json").read_bytes())
    assert marker["attempt"] == attempt
    assert marker["run_id"] == RUN_ID
    assert marker["files"]["model.bin"]["size"] == len(original)


def test_failure_during_real_copy_keeps_previous_generation(tmp_path, monkeypatch):
    """Interrupt the second content copy after writing a fragment into the new tree."""
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=10)
    original_copy = generations._copy_file
    copied = []

    def interrupted(incoming, outgoing):
        copied.append(outgoing)
        if len(copied) == 2:
            with outgoing.open("xb") as stream:
                stream.write(b"partial")
            raise OSError(errno.ENOSPC, "mid-copy")
        return original_copy(incoming, outgoing)

    monkeypatch.setattr(generations, "_copy_file", interrupted)
    with pytest.raises(OSError, match="mid-copy"):
        publish_generation(layout, source, attempt=attempt, step=20)
    assert len(copied) == 2
    incomplete = layout.checkpoints / "gen-000002"
    assert (incomplete / "rank/state.bin").read_bytes() == b"partial"
    assert not (incomplete / "COMPLETE.json").exists()
    assert published_generations(layout) == [prior]


def test_contents_sync_before_marker_and_directory_sync_after_link(
    tmp_path, monkeypatch
):
    """Trace actual fsync/link calls while publication runs, including nested files."""
    layout, attempt, source = setup_run(tmp_path)
    events = []
    phase = ["other"]
    copying = [None]
    marker_bytes = [None]
    real_copy = generations._copy_file
    real_fsync = os.fsync
    real_link = os.link
    real_write_once = generations.write_once
    real_sync_directory = persistence.sync_directory

    def fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            if phase[0].startswith("content:"):
                incoming, outgoing = copying[0]
                assert outgoing.read_bytes() == incoming.read_bytes()
            elif phase[0] == "marker":
                marker = layout.checkpoints / "gen-000001"
                temporary = list(marker.glob(".COMPLETE.json-*"))
                assert len(temporary) == 1
                assert temporary[0].read_bytes() == marker_bytes[0]
        events.append(("fsync", phase[0]))
        real_fsync(fd)

    def copy(incoming, outgoing):
        phase[0] = "content:" + incoming.relative_to(source).as_posix()
        copying[0] = (incoming, outgoing)
        try:
            assert not (
                outgoing.parents[len(incoming.relative_to(source).parts) - 1]
                / "COMPLETE.json"
            ).exists()
            return real_copy(incoming, outgoing)
        finally:
            phase[0] = "other"

    def sync_directory(path):
        events.append(("directory", str(path)))
        real_sync_directory(path)

    def link(incoming, outgoing):
        if outgoing.name == "COMPLETE.json":
            assert ("fsync", "marker") in events
            events.append(("link", str(outgoing)))
        return real_link(incoming, outgoing)

    def write_once(path, content):
        assert ("fsync", "content:model.bin") in events
        assert ("fsync", "content:rank/state.bin") in events
        assert ("directory", str(path.parent / "rank")) in events
        assert ("directory", str(path.parent)) in events
        assert events.index(("directory", str(path.parent / "rank"))) < events.index(
            ("directory", str(path.parent))
        )
        assert not path.exists()
        phase[0] = "marker"
        marker_bytes[0] = content
        try:
            return real_write_once(path, content)
        finally:
            phase[0] = "other"

    monkeypatch.setattr(persistence.os, "fsync", fsync)
    monkeypatch.setattr(persistence.os, "link", link)
    monkeypatch.setattr(persistence, "sync_directory", sync_directory)
    monkeypatch.setattr(generations, "sync_directory", sync_directory)
    monkeypatch.setattr(generations, "_copy_file", copy)
    monkeypatch.setattr(generations, "write_once", write_once)
    published = publish_generation(layout, source, attempt=attempt, step=1)
    marker = published.path / "COMPLETE.json"
    link_index = events.index(("link", str(marker)))
    assert ("directory", str(published.path)) in events[link_index + 1 :]
    assert published_generations(layout) == [published]


def test_sync_failure_before_marker_is_invisible(tmp_path, monkeypatch):
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    original = generations.sync_directory

    def fail(path):
        if path.name == "gen-000002":
            raise OSError(errno.EIO, "content directory sync")
        original(path)

    monkeypatch.setattr(generations, "sync_directory", fail)
    with pytest.raises(OSError, match="content directory sync"):
        publish_generation(layout, source, attempt=attempt, step=2)
    assert not (layout.checkpoints / "gen-000002/COMPLETE.json").exists()
    assert published_generations(layout) == [prior]


def test_sync_failure_after_link_reports_uncertain_durability(tmp_path, monkeypatch):
    layout, attempt, source = setup_run(tmp_path)
    original = persistence.sync_directory

    def fail(path):
        if (path / "COMPLETE.json").exists():
            raise OSError(errno.EIO, "publication directory sync")
        original(path)

    monkeypatch.setattr(persistence, "sync_directory", fail)
    with pytest.raises(OSError, match="publication directory sync"):
        publish_generation(layout, source, attempt=attempt, step=2)
    assert len(published_generations(layout)) == 1


def test_incomplete_and_tampered_generations_are_not_discovered(tmp_path):
    layout, attempt, source = setup_run(tmp_path)
    result = publish_generation(layout, source, attempt=attempt, step=10)
    (layout.checkpoints / "gen-000002").mkdir()
    assert published_generations(layout) == [result]
    (result.path / "model.bin").write_bytes(b"corrupt")
    assert published_generations(layout) == []


def test_standalone_process_publishes_without_daemon_or_ml(tmp_path):
    """A fresh -S process builds a real manifest and generation using only core."""
    script = """
import importlib.abc
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "wavedl_daemon", "wavedl_gui", "torch", "numpy", "PySide6"
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from wavedl.core.layout import RunLayout, OperationKind, allocate_attempt
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.generations import publish_generation, published_generations
base = Path(sys.argv[1])
layout = RunLayout.create(base / "runs", "00000000-0000-4000-8000-000000000001")
manifest = build_manifest(
    layout, operation=OperationKind.TRAIN, resolved_config={"lr": 0.01},
    runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
    submission_cwd=base,
)
write_manifest(layout, manifest)
attempt = allocate_attempt(layout)
source = base / "stage"
source.mkdir()
(source / "model.bin").write_bytes(b"standalone weights")
published = publish_generation(layout, source, attempt=attempt, step=7)
assert published_generations(layout) == [published]
assert "wavedl_daemon" not in sys.modules
print(published.path)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path)],
        env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    generation = Path(result.stdout.strip())
    assert (generation / "COMPLETE.json").is_file()
    assert (generation / "model.bin").read_bytes() == b"standalone weights"


@pytest.mark.parametrize("step", [-1, True])
def test_invalid_step_refused_before_publication(tmp_path, step):
    layout, attempt, source = setup_run(tmp_path)
    with pytest.raises(ValueError):
        publish_generation(layout, source, attempt=attempt, step=step)
    assert list(layout.checkpoints.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
@pytest.mark.parametrize("member", ["COMPLETE.json", "model.bin"])
def test_discovery_replacement_with_fifo_is_bounded_and_preserves_prior(
    tmp_path, member
):
    """Replace after directory enumeration, not before an is_file guard."""
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    later = publish_generation(layout, source, attempt=attempt, step=2)
    script = """
import os
import sys
from pathlib import Path
from wavedl.core import safe_files
from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
layout = RunLayout(Path(sys.argv[1]), sys.argv[2])
target = layout.checkpoints / "gen-000002" / sys.argv[3]
original = safe_files.SafeDirectory.entries
def entries(self):
    values = list(original(self))
    if self.path.name == "checkpoints":
        target.unlink()
        os.mkfifo(target)
    yield from values
safe_files.SafeDirectory.entries = entries
assert [g.step for g in published_generations(layout)] == [1]
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(layout.run_dir), RUN_ID, member],
        env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert (prior.path / "model.bin").read_bytes() == b"weights"
    assert stat.S_ISFIFO((later.path / member).stat().st_mode)


@pytest.mark.parametrize("member", ["COMPLETE.json", "model.bin", "rank"])
def test_discovery_rejects_replaced_symlink_without_losing_prior(
    tmp_path, monkeypatch, member
):
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    later = publish_generation(layout, source, attempt=attempt, step=2)
    target = later.path / member
    unchanged = {
        path.relative_to(later.path): path.read_bytes()
        for path in later.path.rglob("*")
        if path.is_file()
    }
    blocked = []
    original = safe_files.SafeDirectory.entries

    def entries(anchor):
        values = list(original(anchor))
        if anchor.path.name == "checkpoints":
            moved = tmp_path / (member + ".outside")
            if os.name == "nt":
                with pytest.raises(PermissionError) as denied:
                    target.rename(moved)
                assert denied.value.winerror == 32
                assert not moved.exists()
                blocked.append(member)
                yield from values
                return
            target.rename(moved)
            target.symlink_to(moved, target_is_directory=moved.is_dir())
        yield from values

    monkeypatch.setattr(safe_files.SafeDirectory, "entries", entries)
    assert published_generations(layout) == (
        [prior, later] if os.name == "nt" else [prior]
    )
    if os.name == "nt":
        assert blocked == [member]
        assert {
            path.relative_to(later.path): path.read_bytes()
            for path in later.path.rglob("*")
            if path.is_file()
        } == unchanged


def test_discovery_rejects_symlink_generation_but_keeps_prior(tmp_path):
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    (layout.checkpoints / "gen-000002").symlink_to(prior.path, target_is_directory=True)
    assert published_generations(layout) == [prior]


def test_discovery_hashes_after_enumeration_and_not_cached_size(tmp_path, monkeypatch):
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    later = publish_generation(layout, source, attempt=attempt, step=2)
    original = safe_files.SafeDirectory.entries

    def entries(anchor):
        values = list(original(anchor))
        if anchor.path.name == "checkpoints":
            with (later.path / "model.bin").open("ab") as stream:
                stream.write(b"late append")
        yield from values

    monkeypatch.setattr(safe_files.SafeDirectory, "entries", entries)
    assert published_generations(layout) == [prior]


def test_discovery_rejects_oversized_marker_without_losing_prior(tmp_path):
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    later = publish_generation(layout, source, attempt=attempt, step=2)
    (later.path / "COMPLETE.json").write_bytes(b" " * (1024 * 1024 + 1))
    assert published_generations(layout) == [prior]


@pytest.mark.parametrize("member", ["COMPLETE.json", "model.bin", "rank/state.bin"])
def test_discovery_missing_member_after_enumeration_keeps_prior(
    tmp_path, monkeypatch, member
):
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    later = publish_generation(layout, source, attempt=attempt, step=2)
    original = safe_files.SafeDirectory.entries

    def entries(anchor):
        values = list(original(anchor))
        if anchor.path.name == "checkpoints":
            (later.path / member).unlink()
        yield from values

    monkeypatch.setattr(safe_files.SafeDirectory, "entries", entries)
    assert published_generations(layout) == [prior]


@pytest.mark.parametrize("marker", [b"[]", b"{broken", b'{"schema_version": NaN}'])
def test_discovery_malformed_new_marker_keeps_prior(tmp_path, marker):
    layout, attempt, source = setup_run(tmp_path)
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    later = publish_generation(layout, source, attempt=attempt, step=2)
    (later.path / "COMPLETE.json").write_bytes(marker)
    assert published_generations(layout) == [prior]


@pytest.mark.parametrize(
    "failure",
    [
        OSError(errno.EIO, "read failed"),
        safe_files.UnsupportedSafeRead(errno.ENOTSUP, "no native primitive"),
    ],
)
def test_discovery_surfaces_operational_errors(tmp_path, monkeypatch, failure):
    layout, attempt, source = setup_run(tmp_path)
    publish_generation(layout, source, attempt=attempt, step=1)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(safe_files.SafeDirectory, "digest_file", fail)
    with pytest.raises(type(failure), match=str(failure.strerror)):
        published_generations(layout)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_discovery_rejects_fifo_manifest_without_blocking(tmp_path):
    layout, attempt, source = setup_run(tmp_path)
    publish_generation(layout, source, attempt=attempt, step=1)
    layout.manifest.unlink()
    os.mkfifo(layout.manifest)
    script = """
import sys
from pathlib import Path
from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
try:
    published_generations(RunLayout(Path(sys.argv[1]), sys.argv[2]))
except ValueError:
    pass
else:
    raise AssertionError("FIFO manifest was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(layout.run_dir), RUN_ID],
        env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
