"""Standalone checkpoint selection and inference consumer compatibility."""

import json
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wavedl.core import checkpoint_selection
from wavedl.core.checkpoint_selection import (
    SELECTION_METADATA,
    UnsupportedSafeRead,
    read_selection_metadata,
    resolve_checkpoint_path,
    select_checkpoint,
)
from wavedl.core.generations import publish_generation
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.safe_files import SafeDirectory


@pytest.fixture
def run(tmp_path):
    layout = RunLayout.create(tmp_path / "runs", "00000000-0000-4000-8000-000000000001")
    write_manifest(
        layout,
        build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={},
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            submission_cwd=tmp_path,
        ),
    )
    attempt = allocate_attempt(layout)
    source = tmp_path / "staging"
    source.mkdir()
    (source / "model.bin").write_bytes(b"weights")
    return layout, attempt, source


def publish(run, *, step=1, kind=None, attempt=None):
    layout, reserved, source = run
    metadata = source / SELECTION_METADATA
    if kind is not None:
        metadata.write_text(json.dumps({"schema_version": 1, "kind": kind}))
    elif metadata.exists():
        metadata.unlink()
    return publish_generation(layout, source, attempt=attempt or reserved, step=step)


def test_resume_and_best_have_distinct_publication_order(run):
    layout, _, _ = run
    best = publish(run, step=20, kind="best")
    publish(run, step=50, kind="periodic")
    rewind = publish(run, step=3, kind="interrupted", attempt=allocate_attempt(layout))
    assert select_checkpoint(layout).path == rewind.path
    assert select_checkpoint(layout, purpose="best").path == best.path


def test_legacy_t20a_generations_are_resumable_not_assumed_best(run):
    generation = publish(run)
    assert select_checkpoint(run[0]).path == generation.path
    assert select_checkpoint(run[0], purpose="best") is None
    assert read_selection_metadata(generation) is None


@pytest.mark.parametrize("kind", ["cross_validation_fold", "cross_validation_summary"])
def test_cross_validation_requires_explicit_generation_choice(run, kind):
    layout, _, _ = run
    best = publish(run, kind="best")
    cv = publish(run, step=2, kind=kind)
    assert read_selection_metadata(cv)["kind"] == kind
    assert select_checkpoint(layout).path == best.path
    assert select_checkpoint(layout, purpose="best").path == best.path
    assert resolve_checkpoint_path(cv.path) == cv.path
    with pytest.raises(ValueError, match="inference-only"):
        resolve_checkpoint_path(cv.path, purpose="resume")


def test_cross_validation_only_run_has_no_automatic_checkpoint(run):
    layout, _, _ = run
    publish(run, kind="cross_validation_fold")
    publish(run, kind="cross_validation_summary")
    assert select_checkpoint(layout) is None
    assert select_checkpoint(layout, purpose="best") is None


@pytest.mark.parametrize("source_seq", [0, 3, -1, True, False, "4", None, 1.5])
def test_selection_metadata_validates_optional_journal_cutoff(run, source_seq):
    layout, attempt, source = run
    metadata = {"schema_version": 1, "kind": "best", "source_seq": source_seq}
    (source / SELECTION_METADATA).write_text(json.dumps(metadata))
    generation = publish_generation(layout, source, attempt=attempt, step=3)
    if type(source_seq) is int and source_seq >= 0:
        assert read_selection_metadata(generation) == metadata
        assert checkpoint_selection._kind(generation) == "best"
    else:
        with pytest.raises(ValueError, match="source_seq"):
            read_selection_metadata(generation)
        assert select_checkpoint(layout) is None


def test_missing_journal_cutoff_is_supported_legacy_metadata(run):
    generation = publish(run, kind="best")
    assert read_selection_metadata(generation) == {"schema_version": 1, "kind": "best"}


def test_latest_best_publication_wins_even_after_rewind(run):
    publish(run, step=50, kind="best")
    latest = publish(run, step=3, kind="best")
    assert select_checkpoint(run[0], purpose="best").path == latest.path


@pytest.mark.parametrize("invalid", ["incomplete", "tampered", "extra", "symlink"])
def test_invalid_newer_generation_never_selected(run, invalid):
    older = publish(run, kind="best")
    newer = publish(run, step=50, kind="best")
    if invalid == "incomplete":
        (newer.path / "COMPLETE.json").unlink()
    elif invalid == "tampered":
        (newer.path / "model.bin").write_bytes(b"bad")
    elif invalid == "extra":
        (newer.path / "model.safetensors").write_bytes(b"unpublished")
    else:
        (newer.path / "extra").symlink_to(older.path / "model.bin")
    assert select_checkpoint(run[0]).path == older.path
    assert select_checkpoint(run[0], purpose="best").path == older.path
    with pytest.raises(ValueError, match="incomplete or invalid"):
        resolve_checkpoint_path(newer.path)


@pytest.mark.parametrize("kind", ["unknown", True, 42])
def test_invalid_selection_metadata_never_selected(run, kind):
    older = publish(run)
    newer = publish(run, kind=kind)
    assert select_checkpoint(run[0]).path == older.path
    with pytest.raises(ValueError, match="selection metadata"):
        resolve_checkpoint_path(newer.path)


def test_metadata_must_be_hash_covered_and_bounded(run):
    older = publish(run)
    newer = publish(run, kind="best")
    (newer.path / SELECTION_METADATA).write_text(" " * 5000)
    assert select_checkpoint(run[0]).path == older.path


def test_published_oversized_selection_metadata_is_ineligible(run):
    layout, attempt, source = run
    older = publish(run)
    (source / SELECTION_METADATA).write_text(
        json.dumps({"schema_version": 1, "kind": "best", "extra": "x" * 5000})
    )
    oversized = publish_generation(layout, source, attempt=attempt, step=3)
    assert select_checkpoint(layout).path == older.path
    with pytest.raises(ValueError, match="byte bound"):
        resolve_checkpoint_path(oversized.path)


def test_roots_and_explicit_generation_resolve(run):
    layout, _, _ = run
    best = publish(run, kind="best")
    latest = publish(run, kind="periodic")
    assert resolve_checkpoint_path(layout.run_dir) == best.path
    assert resolve_checkpoint_path(layout.checkpoints) == best.path
    assert resolve_checkpoint_path(layout.run_dir, purpose="resume") == latest.path
    assert resolve_checkpoint_path(latest.path) == latest.path


def test_inventory_walk_preserves_empty_directories_but_is_bounded(run):
    layout, _, source = run
    (source / "empty").mkdir()
    generation = publish(run, kind="best")
    assert select_checkpoint(layout).path == generation.path
    for index in range(1025):
        (generation.path / f"extra-{index}").mkdir()
    assert select_checkpoint(layout) is None


def test_detached_completion_marker_is_not_a_legacy_checkpoint(tmp_path):
    detached = tmp_path / "detached"
    detached.mkdir()
    (detached / "COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="outside a run layout"):
        resolve_checkpoint_path(detached)


@pytest.mark.parametrize("replacement", ["fifo", "symlink", "hash"])
def test_cached_metadata_validation_refuses_replacement_without_blocking(
    run, replacement
):
    generation = publish(run, kind="best")
    metadata = generation.path / SELECTION_METADATA
    original = metadata.read_text()
    metadata.unlink()
    if replacement == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO native gate requires POSIX")
        os.mkfifo(metadata)
        record = generation.files[SELECTION_METADATA]
        # A regression must fail within a hard deadline, not hang pytest itself.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from pathlib import Path\n"
                "from wavedl.core.checkpoint_selection import _kind, SELECTION_METADATA\n"
                "from wavedl.core.generations import PublishedGeneration, FileRecord\n"
                "g = PublishedGeneration(Path(sys.argv[1]), 'gen', 1, 1, '', "
                "{SELECTION_METADATA: FileRecord(int(sys.argv[2]), sys.argv[3])})\n"
                "try:\n"
                "    _kind(g)\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('FIFO metadata was accepted')\n",
                str(generation.path),
                str(record.size),
                record.sha256,
            ],
            check=True,
            timeout=5,
        )
        return
    elif replacement == "symlink":
        metadata.symlink_to(run[2] / SELECTION_METADATA)
    else:
        metadata.write_text(original.replace('"best"', '"xxxx"'))
    with pytest.raises(ValueError):
        checkpoint_selection._kind(generation)


@pytest.mark.parametrize("swap_before_open", [True, False])
def test_inventory_directory_swap_cannot_traverse_outside_generation(
    run, monkeypatch, tmp_path, swap_before_open
):
    _, _, source = run
    nested = source / "nested"
    nested.mkdir()
    (nested / "state.bin").write_bytes(b"optimizer")
    generation = publish(run, kind="best")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "unverified.bin").write_bytes(b"outside")
    real_open = SafeDirectory.open_directory
    swapped = []
    denied = []

    def swapping_open(directory, name):
        if name != "nested":
            return real_open(directory, name)
        if not swap_before_open:
            anchor = real_open(directory, name)
        try:
            (generation.path / "nested").rename(tmp_path / "original-nested")
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) != 32:
                raise
            denied.append(True)
            if not swap_before_open:
                anchor.close()
            raise
        (generation.path / "nested").symlink_to(outside, target_is_directory=True)
        swapped.append(True)
        if swap_before_open:
            return real_open(directory, name)
        return anchor

    monkeypatch.setattr(SafeDirectory, "open_directory", swapping_open)
    if os.name == "nt":
        with pytest.raises(PermissionError) as refusal:
            checkpoint_selection._covered_inventory(generation)
        assert refusal.value.winerror == 32
        assert denied == [True] and swapped == []
        assert not (generation.path / "nested").is_symlink()
        assert (generation.path / "nested/state.bin").read_bytes() == b"optimizer"
        assert (outside / "unverified.bin").read_bytes() == b"outside"
        assert not (tmp_path / "original-nested").exists()
        monkeypatch.setattr(SafeDirectory, "open_directory", real_open)
        assert checkpoint_selection._covered_inventory(generation)
    else:
        assert (
            checkpoint_selection._covered_inventory(generation) is not swap_before_open
        )
        assert swapped == [True] and denied == []


def test_checkpoint_io_errors_propagate_not_hidden_as_ineligible(run, monkeypatch):
    generation = publish(run, kind="best")

    def fail_open(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(checkpoint_selection, "open_directory", fail_open)
    with pytest.raises(OSError, match="storage unavailable"):
        checkpoint_selection._kind(generation)
    with pytest.raises(OSError, match="storage unavailable"):
        checkpoint_selection._covered_inventory(generation)


def test_selection_metadata_uses_portable_safe_directory_contract(run, monkeypatch):
    generation = publish(run, kind="best")
    raw = (generation.path / SELECTION_METADATA).read_bytes()
    calls = []

    class PortableAnchor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("close")

        def read_file(self, name, *, limit):
            calls.append((name, limit))
            return raw

    monkeypatch.setattr(
        checkpoint_selection, "open_directory", lambda path: PortableAnchor()
    )
    assert checkpoint_selection._kind(generation) == "best"
    assert calls == [(SELECTION_METADATA, 4096), "close"]


@pytest.mark.parametrize("missing", ["dir_fd", "O_NOFOLLOW"])
def test_descriptor_reads_fail_closed_when_platform_support_is_unavailable(
    run, monkeypatch, missing
):
    generation = publish(run, kind="best")

    def unsupported(*args, **kwargs):
        raise UnsupportedSafeRead(f"Safe native capability unavailable: {missing}")

    monkeypatch.setattr(checkpoint_selection, "open_directory", unsupported)
    (run[0].run_dir / "best_checkpoint").mkdir()
    with pytest.raises(UnsupportedSafeRead, match="unavailable"):
        checkpoint_selection._covered_inventory(generation)
    with pytest.raises(UnsupportedSafeRead, match="unavailable"):
        checkpoint_selection._kind(generation)
    with pytest.raises(UnsupportedSafeRead, match="unavailable"):
        select_checkpoint(run[0])
    with pytest.raises(UnsupportedSafeRead, match="unavailable"):
        resolve_checkpoint_path(run[0].run_dir)


def test_legacy_paths_and_best_fallback_are_preserved(run):
    layout, _, _ = run
    publish(run)
    legacy = layout.run_dir / "best_checkpoint"
    legacy.mkdir()
    assert resolve_checkpoint_path(legacy) == legacy
    assert resolve_checkpoint_path(layout.run_dir) == legacy


def test_missing_best_does_not_infer_periodic_or_legacy_resume(run):
    layout, _, _ = run
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(layout.run_dir, purpose="resume")
    publish(run, kind="periodic")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(layout.run_dir)


def test_consumer_import_is_independent_of_ml_and_daemon():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wavedl.core.checkpoint_selection; "
            "assert 'torch' not in sys.modules; "
            "assert not any(k.startswith('wavedl_daemon') for k in sys.modules)",
        ],
        check=True,
    )


def test_real_inference_loader_uses_published_best_and_local_scaler(run, monkeypatch):
    torch = pytest.importorskip("torch")
    inference = pytest.importorskip("wavedl.test")

    class TinyModel(torch.nn.Linear):
        def parameter_summary(self):
            return {"trainable_parameters": 2, "total_mb": 0.001}

    layout, _, source = run
    model = TinyModel(1, 1)
    torch.save(model.state_dict(), source / "model.bin")
    (source / "training_meta.pkl").write_bytes(
        pickle.dumps({"model_name": "tiny", "epoch": 1})
    )
    (source / "scaler.pkl").write_bytes(pickle.dumps("generation scaler"))
    best = publish(run, kind="best")
    # A later periodic checkpoint must not become the inferred plotting model.
    with torch.no_grad():
        model.weight.fill_(999)
    torch.save(model.state_dict(), source / "model.bin")
    publish(run, step=2, kind="periodic")
    (layout.checkpoints / "scaler.pkl").write_bytes(pickle.dumps("stale scaler"))
    monkeypatch.setattr(inference, "get_model", lambda name: TinyModel)
    monkeypatch.setattr(inference, "build_model", lambda *a, **k: TinyModel(1, 1))
    loaded, scaler = inference.load_checkpoint(str(layout.run_dir), (1,), 1)
    expected = torch.load(best.path / "model.bin", weights_only=True)
    assert torch.equal(loaded.weight, expected["weight"])
    assert scaler == "generation scaler"
    assert not loaded.training


def test_real_inference_loader_refuses_missing_generation_scaler(run, monkeypatch):
    torch = pytest.importorskip("torch")
    inference = pytest.importorskip("wavedl.test")
    _, _, source = run
    torch.save(torch.nn.Linear(1, 1).state_dict(), source / "model.bin")
    generation = publish(run, kind="best")
    (generation.path.parent / "scaler.pkl").write_bytes(pickle.dumps("stale"))
    monkeypatch.setattr(inference, "get_model", lambda name: torch.nn.Linear)
    monkeypatch.setattr(inference, "build_model", lambda *a, **k: torch.nn.Linear(1, 1))
    with pytest.raises(FileNotFoundError, match="Scaler not found"):
        inference.load_checkpoint(str(generation.path), (1,), 1, model_name="tiny")


@pytest.mark.parametrize("include_scaler", [True, False])
def test_main_root_infers_shape_and_output_size_from_same_generation(
    run, monkeypatch, include_scaler
):
    torch = pytest.importorskip("torch")
    inference = pytest.importorskip("wavedl.test")
    layout, _, source = run
    (source / "training_meta.pkl").write_bytes(
        pickle.dumps({"model_name": "tiny", "in_shape": (4,)})
    )
    if include_scaler:
        (source / "scaler.pkl").write_bytes(
            pickle.dumps(SimpleNamespace(scale_=[1, 1]))
        )
    best = publish(run, kind="best")
    (layout.run_dir / "scaler.pkl").write_bytes(
        pickle.dumps(SimpleNamespace(scale_=[1] * 7))
    )
    (layout.checkpoints / "scaler.pkl").write_bytes(
        pickle.dumps(SimpleNamespace(scale_=[1] * 9))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-test",
            "--checkpoint",
            str(layout.run_dir),
            "--data_path",
            "unused",
            "--output_dir",
            str(layout.run_dir / "inference-test-output"),
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr(
        inference,
        "load_data_for_inference",
        lambda *a, **k: (torch.zeros(2, 1, 4), None),
    )
    calls = []

    def stop_at_load(path, shape, size, name, *, out_shape=None):
        calls.append((path, shape, size, name, out_shape))
        raise RuntimeError("stop before inference writes")

    monkeypatch.setattr(inference, "load_checkpoint", stop_at_load)
    if include_scaler:
        with pytest.raises(RuntimeError, match="stop before inference writes"):
            inference.main()
        assert calls == [(str(best.path), (4,), 2, None, (2,))]
    else:
        with pytest.raises(ValueError, match="Cannot determine output size"):
            inference.main()
        assert calls == []
