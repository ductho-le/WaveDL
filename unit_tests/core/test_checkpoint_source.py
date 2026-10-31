"""Approved explicit inference pins survive only byte-identical publications."""

import hashlib
import json
import sys
from dataclasses import asdict

import pytest

from wavedl.core.checkpoint_source import (
    capture_checkpoint_source,
    snapshot_checkpoint_source,
    verify_checkpoint_source,
)
from wavedl.core.generations import publish_generation
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import (
    RuntimeSelection,
    build_manifest,
    read_manifest,
    write_manifest,
)


@pytest.fixture
def published(tmp_path):
    layout = RunLayout.create(tmp_path / "runs", "00000000-0000-4000-8000-000000000001")
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in ("model_weights.pth", "training_meta.pkl", "scaler.pkl"):
        (staging / name).write_bytes(b"covered-state")
    (staging / "checkpoint_selection.json").write_text(
        json.dumps({"schema_version": 1, "kind": "best"})
    )
    generation = publish_generation(
        layout, staging, attempt=allocate_attempt(layout), step=1
    )
    digest = hashlib.sha256(layout.manifest.read_bytes()).hexdigest()
    return layout, generation, staging, digest


def test_source_pin_covers_full_publication_and_preserves_legacy_manifest(published):
    layout, generation, _, digest = published
    assert "checkpoint_source" not in json.loads(layout.manifest.read_bytes())
    legacy = asdict(read_manifest(layout.manifest))
    legacy.pop("checkpoint_source")
    legacy_bytes = (
        json.dumps(
            legacy,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    assert layout.manifest.read_bytes() == legacy_bytes
    assert hashlib.sha256(legacy_bytes).hexdigest() == digest
    source = capture_checkpoint_source(
        layout, generation.generation, manifest_sha256=digest
    )
    verify_checkpoint_source(generation.path, source)
    assert set(source["files"]) == set(generation.files)
    assert (
        source["complete_sha256"]
        == hashlib.sha256((generation.path / "COMPLETE.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation", ["weights", "extra", "marker", "manifest", "symlink"]
)
def test_source_pin_rejects_mutations(published, mutation):
    layout, generation, _, digest = published
    source = capture_checkpoint_source(
        layout, generation.generation, manifest_sha256=digest
    )
    if mutation == "weights":
        (generation.path / "model_weights.pth").write_bytes(b"changed")
    elif mutation == "extra":
        (generation.path / "model.safetensors").write_bytes(b"preferred-unlisted")
    elif mutation == "marker":
        marker = generation.path / "COMPLETE.json"
        marker.write_bytes(marker.read_bytes() + b" ")
    elif mutation == "manifest":
        layout.manifest.write_bytes(layout.manifest.read_bytes() + b" ")
    else:
        weights = generation.path / "model_weights.pth"
        weights.unlink()
        weights.symlink_to(layout.manifest)
    with pytest.raises((ValueError, OSError)):
        verify_checkpoint_source(generation.path, source)


@pytest.mark.parametrize(
    "kind",
    ["periodic", "interrupted", "cross_validation_fold", "cross_validation_summary"],
)
def test_explicit_kind_policy(published, kind):
    layout, _, staging, digest = published
    (staging / "checkpoint_selection.json").write_text(
        json.dumps({"schema_version": 1, "kind": kind})
    )
    generation = publish_generation(layout, staging, attempt=1, step=2)
    if kind == "cross_validation_summary":
        with pytest.raises(ValueError, match="inference checkpoint"):
            capture_checkpoint_source(
                layout, generation.generation, manifest_sha256=digest
            )
    else:
        assert (
            capture_checkpoint_source(
                layout, generation.generation, manifest_sha256=digest
            )["selection"]["kind"]
            == kind
        )


def test_hashing_cooperatively_checks_worker_deadline(published):
    layout, generation, _, digest = published
    calls = []

    def check():
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        capture_checkpoint_source(
            layout, generation.generation, manifest_sha256=digest, check=check
        )
    assert len(calls) == 3


def test_inventory_traversal_checks_deadline_before_scanning_next_member(
    published, monkeypatch
):
    """Cancellation during enumeration must not wait for the entire tree."""
    from wavedl.core.safe_files import SafeDirectory

    layout, generation, _, digest = published
    entries = SafeDirectory.entries
    scanned = []

    def enumerate_members(directory):
        for entry in entries(directory):
            scanned.append(entry.name)
            yield entry

    def check():
        if scanned:
            raise RuntimeError("cancelled inventory scan")

    monkeypatch.setattr(SafeDirectory, "entries", enumerate_members)
    with pytest.raises(RuntimeError, match="cancelled inventory scan"):
        capture_checkpoint_source(
            layout, generation.generation, manifest_sha256=digest, check=check
        )
    assert len(scanned) == 1


def test_inference_snapshot_consumes_approved_bytes_not_mutable_source(published):
    layout, generation, _, digest = published
    pin = capture_checkpoint_source(
        layout, generation.generation, manifest_sha256=digest
    )
    with snapshot_checkpoint_source(generation.path, pin) as captured:
        (generation.path / "model_weights.pth").write_bytes(b"changed-after-capture")
        assert (captured / "model_weights.pth").read_bytes() == b"covered-state"
        assert (captured / "scaler.pkl").read_bytes() == b"covered-state"
    assert not captured.exists()


def test_nested_member_traversal_checks_deadline_between_ancestors(
    published, monkeypatch
):
    """Nested paths must not defer cancellation until the file hash begins."""
    from wavedl.core.safe_files import SafeDirectory

    layout, _, staging, digest = published
    nested = staging / "nested" / "level" / "deep"
    nested.mkdir(parents=True)
    (nested / "state.bin").write_bytes(b"covered ancillary state")
    generation = publish_generation(layout, staging, attempt=1, step=2)
    original = SafeDirectory.open_directory
    opened = []

    def open_member(directory, name):
        result = original(directory, name)
        opened.append(name)
        return result

    def check():
        if "nested" in opened:
            raise RuntimeError("cancelled ancestor traversal")

    monkeypatch.setattr(SafeDirectory, "open_directory", open_member)
    with pytest.raises(RuntimeError, match="cancelled ancestor traversal"):
        capture_checkpoint_source(
            layout, generation.generation, manifest_sha256=digest, check=check
        )
    assert "nested" in opened
    assert "level" not in opened


def test_snapshot_detects_change_between_verification_and_copy(published, monkeypatch):
    import wavedl.core.checkpoint_source as module

    layout, generation, _, digest = published
    pin = capture_checkpoint_source(
        layout, generation.generation, manifest_sha256=digest
    )
    verify = module.verify_checkpoint_source

    def change_after_verify(path, source):
        verify(path, source)
        (path / "model_weights.pth").write_bytes(b"replacement")

    monkeypatch.setattr(module, "verify_checkpoint_source", change_after_verify)
    with (
        pytest.raises(ValueError, match="changed during inference capture"),
        snapshot_checkpoint_source(generation.path, pin),
    ):
        pytest.fail("A replacement checkpoint must never reach the loader")
