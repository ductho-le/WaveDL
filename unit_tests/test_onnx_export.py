"""
Tests for single-file ONNX export and crash-safe replacement.

Exports must be complete, self-contained files at exactly the requested opset,
and a failed export must never damage an existing file at the destination.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from wavedl import test  # noqa: E402
from wavedl.models.cnn import CNN  # noqa: E402
from wavedl.utils import onnx_export  # noqa: E402
from wavedl.utils.onnx_export import (  # noqa: E402
    ExportCommitUnknown,
    replace_export,
    validate_self_contained_onnx,
)


def _write_external_model(path: Path, location: str = "missing.data") -> None:
    """Write an ONNX model whose weight points at an external-data file."""
    weight = onnx.numpy_helper.from_array(
        np.ones((3, 2), dtype=np.float32), name="weight"
    )
    onnx.external_data_helper.set_external_data(weight, location=location)
    weight.data_location = onnx.TensorProto.EXTERNAL
    weight.ClearField("raw_data")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["input", "weight"], ["predictions"])],
        "external",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])],
        [
            onnx.helper.make_tensor_value_info(
                "predictions", onnx.TensorProto.FLOAT, [1, 2]
            )
        ],
        [weight],
    )
    path.write_bytes(onnx.helper.make_model(graph).SerializeToString())


def _no_staging_left(directory: Path) -> bool:
    return not list(directory.glob(".wavedl-export-*")) and not list(
        directory.glob(".onnx-recovery-*")
    )


# ==============================================================================
# REAL EXPORTS
# ==============================================================================
@pytest.mark.parametrize("opset", [11, 17])
def test_cnn_export_has_exact_opset_and_dynamic_batch(tmp_path, opset):
    """A registered CNN exports exactly the requested opset, not an upgraded one."""
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(14)
    model = CNN(in_shape=(32, 32), out_size=2, base_channels=8).eval()
    output = tmp_path / "cnn.onnx"
    assert test.export_to_onnx(
        model, torch.randn(2, 1, 32, 32), str(output), opset_version=opset
    )
    validate_self_contained_onnx(output, expected_opset=opset)
    graph = onnx.load(str(output))
    assert [(item.domain, item.version) for item in graph.opset_import] == [("", opset)]
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for batch_size in (1, 3):
        inputs = torch.randn(batch_size, 1, 32, 32)
        with torch.inference_mode():
            expected = model(inputs).numpy()
        np.testing.assert_allclose(
            session.run(None, {"input": inputs.numpy()})[0],
            expected,
            rtol=1e-4,
            atol=1e-5,
        )
    assert _no_staging_left(tmp_path)


def test_export_is_single_file_that_runs_after_being_moved(tmp_path):
    """The export runs from another directory, keeps metadata, and has no sidecars."""
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(14)
    model = torch.nn.Linear(3, 2).eval()
    sample = torch.randn(4, 3)
    exported = tmp_path / "model.onnx"
    scaler = SimpleNamespace(mean_=np.array([2.0, -3.0]), scale_=np.array([4.0, 5.0]))
    assert test.export_to_onnx(
        model,
        sample,
        str(exported),
        scaler=scaler,
        include_denorm=True,
        model_name="tiny",
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.onnx"]

    props = {p.key: p.value for p in onnx.load(str(exported)).metadata_props}
    assert props["model_name"] == "tiny"
    assert props["output_units"] == "physical"
    assert props["input_shape"] == "(3,)"

    moved = tmp_path / "elsewhere" / "different-name.onnx"
    moved.parent.mkdir()
    shutil.copyfile(exported, moved)
    exported.unlink()
    validate_self_contained_onnx(moved)
    session = ort.InferenceSession(str(moved), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(sample).numpy()
    expected = expected * scaler.scale_ + scaler.mean_
    np.testing.assert_allclose(
        session.run(None, {"input": sample.numpy()})[0], expected, rtol=1e-4, atol=1e-5
    )


def test_export_replaces_prior_file(tmp_path):
    """A successful export replaces an existing file and cleans up after itself."""
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")
    assert test.export_to_onnx(
        torch.nn.Linear(3, 2).eval(), torch.randn(1, 3), str(destination)
    )
    validate_self_contained_onnx(destination, expected_opset=17)
    assert _no_staging_left(tmp_path)


# ==============================================================================
# REFUSALS (prior file untouched, nothing left behind)
# ==============================================================================
def _graph_model(opset: int):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["predictions"])],
        "identity",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])],
        [
            onnx.helper.make_tensor_value_info(
                "predictions", onnx.TensorProto.FLOAT, [1, 3]
            )
        ],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)]
    )


@pytest.mark.parametrize("prior_exists", [False, True])
def test_exporter_upgrading_opset_is_refused_before_commit(
    tmp_path, monkeypatch, caplog, prior_exists
):
    """An exporter that ignores the requested opset cannot replace the output."""
    output = tmp_path / "model.onnx"
    if prior_exists:
        output.write_bytes(b"previous model")

    def upgrading_export(model, sample, destination, *, dynamo=True, **kwargs):
        assert dynamo is False
        assert kwargs["opset_version"] == 17
        Path(destination).write_bytes(_graph_model(18).SerializeToString())

    monkeypatch.setattr(torch.onnx, "export", upgrading_export)
    monkeypatch.setattr(
        test,
        "_validate_onnx_export",
        lambda *args: pytest.fail("numerical check preceded the opset check"),
    )
    monkeypatch.setattr(
        onnx_export,
        "replace_export",
        lambda *args: pytest.fail("invalid opset reached commit"),
    )
    assert not test.export_to_onnx(
        torch.nn.Linear(3, 2), torch.randn(1, 3), str(output), validate=True
    )
    assert "requested opset 17" in caplog.text
    assert "produced standard-domain opsets [18]" in caplog.text
    if prior_exists:
        assert output.read_bytes() == b"previous model"
    else:
        assert not output.exists()
    assert _no_staging_left(tmp_path)


def test_external_tensor_references_are_rejected_without_loading(tmp_path):
    path = tmp_path / "model.onnx"
    _write_external_model(path, "../secret.data")
    with pytest.raises(ValueError, match="external tensor"):
        validate_self_contained_onnx(path)


def test_external_tensor_in_nested_subgraph_is_rejected(tmp_path):
    path = tmp_path / "model.onnx"
    _write_external_model(path)
    inner = onnx.load(str(path), load_external_data=False).graph
    outer = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "If", ["cond"], ["out"], then_branch=inner, else_branch=inner
            )
        ],
        "nested",
        [],
        [],
    )
    path.write_bytes(onnx.helper.make_model(outer).SerializeToString())
    with pytest.raises(ValueError, match="external tensor"):
        validate_self_contained_onnx(path)


def test_non_regular_or_empty_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.onnx"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="bounded regular single file"):
        validate_self_contained_onnx(empty)
    with pytest.raises(ValueError, match="bounded regular single file"):
        validate_self_contained_onnx(tmp_path)


def test_exporter_writing_external_data_cannot_replace_prior_output(
    tmp_path, monkeypatch, caplog
):
    """external_data=False is requested; an exporter ignoring it is still refused."""
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")
    (tmp_path / "secret.data").write_bytes(b"\0" * 24)

    def external_export(model, sample, output, *, external_data=True, **kwargs):
        assert external_data is False
        _write_external_model(Path(output), "../secret.data")

    monkeypatch.setattr(torch.onnx, "export", external_export)
    assert not test.export_to_onnx(
        torch.nn.Linear(3, 2), torch.randn(1, 3), str(destination), validate=False
    )
    assert "external tensor" in caplog.text
    assert destination.read_bytes() == b"previous model"
    assert _no_staging_left(tmp_path)


def test_model_of_two_gib_or_more_is_refused_before_exporting(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(onnx_export, "MAX_ONNX_BYTES", 8)
    monkeypatch.setattr(
        torch.onnx,
        "export",
        lambda *args, **kwargs: pytest.fail("oversized model reached exporter"),
    )
    destination = tmp_path / "model.onnx"
    assert not test.export_to_onnx(
        torch.nn.Linear(3, 2), torch.randn(1, 3), str(destination)
    )
    assert "external-data" in caplog.text
    assert not destination.exists()
    assert _no_staging_left(tmp_path)


# ==============================================================================
# CRASH-SAFE REPLACEMENT
# ==============================================================================
@pytest.fixture
def staged_pair(tmp_path):
    staged = tmp_path / "staged.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    destination.write_bytes(b"previous model")
    return staged, destination


@pytest.mark.parametrize("failed_sync", [1, 3])
def test_sync_failure_keeps_prior_file(tmp_path, monkeypatch, staged_pair, failed_sync):
    """Directory syncs: recovery dir, pre-commit parent, post-commit parent."""
    staged, destination = staged_pair
    calls = 0

    def fail_once(path):
        nonlocal calls
        calls += 1
        if calls == failed_sync:
            raise OSError("injected directory sync failure")

    monkeypatch.setattr(onnx_export, "sync_directory", fail_once)
    with pytest.raises(OSError, match="sync failure"):
        replace_export(staged, destination)
    assert destination.read_bytes() == b"previous model"
    assert not list(tmp_path.glob(".onnx-recovery-*"))


@pytest.mark.parametrize("failed_rollback", ["replace", "sync"])
def test_failed_rollback_raises_and_retains_prior_bytes(
    tmp_path, monkeypatch, staged_pair, failed_rollback
):
    staged, destination = staged_pair
    sync_calls = 0

    def fail_sync(path):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 3 or (failed_rollback == "sync" and sync_calls == 4):
            raise OSError("injected directory sync failure")

    original_replace = os.replace
    replace_calls = 0

    def fail_replace(source, target):
        nonlocal replace_calls
        replace_calls += 1
        if failed_rollback == "replace" and replace_calls == 2:
            raise OSError("injected rollback failure")
        return original_replace(source, target)

    monkeypatch.setattr(onnx_export, "sync_directory", fail_sync)
    monkeypatch.setattr(onnx_export.os, "replace", fail_replace)
    with pytest.raises(ExportCommitUnknown, match="prior export retained"):
        replace_export(staged, destination)
    recovery = list(tmp_path.glob(".onnx-recovery-*"))
    assert len(recovery) == 1
    assert (recovery[0] / "prior.onnx").read_bytes() == b"previous model"


def test_first_export_postcommit_failure_removes_only_new_file(tmp_path, monkeypatch):
    staged = tmp_path / "staged.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    calls = 0

    def fail_second(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("post-commit sync failure")

    monkeypatch.setattr(onnx_export, "sync_directory", fail_second)
    with pytest.raises(OSError, match="post-commit"):
        replace_export(staged, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".onnx-recovery-*"))


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlink_destination_is_not_followed(tmp_path):
    staged = tmp_path / "staged.onnx"
    target = tmp_path / "unrelated.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    target.write_bytes(b"unrelated bytes")
    destination.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        replace_export(staged, destination)
    assert destination.is_symlink()
    assert target.read_bytes() == b"unrelated bytes"


def test_prior_backup_fsync_failure_precedes_commit(monkeypatch, staged_pair):
    staged, destination = staged_pair

    def fail_fsync(descriptor):
        raise OSError("backup fsync failure")

    monkeypatch.setattr(onnx_export.os, "fsync", fail_fsync)
    monkeypatch.setattr(
        onnx_export.os,
        "replace",
        lambda *args: pytest.fail("commit preceded a durable backup"),
    )
    with pytest.raises(OSError, match="backup fsync"):
        replace_export(staged, destination)
    assert destination.read_bytes() == b"previous model"
    assert staged.read_bytes() == b"new model"
    assert not list(staged.parent.glob(".onnx-recovery-*"))


def test_backup_falls_back_to_copy_without_hardlinks(monkeypatch, staged_pair):
    """Filesystems without hardlinks (e.g. exFAT) still get a rollback copy."""
    staged, destination = staged_pair

    def no_link(*args, **kwargs):
        raise OSError("hardlinks unsupported")

    calls = 0

    def fail_post_commit(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("post-commit sync failure")

    monkeypatch.setattr(onnx_export.os, "link", no_link)
    monkeypatch.setattr(onnx_export, "sync_directory", fail_post_commit)
    with pytest.raises(OSError, match="post-commit"):
        replace_export(staged, destination)
    assert destination.read_bytes() == b"previous model"
    assert not list(staged.parent.glob(".onnx-recovery-*"))


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_export_api_surfaces_unknown_commit(tmp_path, monkeypatch, rollback_fails):
    """False means "unchanged"; an uncertain on-disk state raises instead."""
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")
    calls = 0

    def fail_sync(path):
        nonlocal calls
        calls += 1
        if calls == 3 or (rollback_fails and calls == 4):
            raise OSError("injected directory sync failure")

    monkeypatch.setattr(onnx_export, "sync_directory", fail_sync)

    def export():
        return test.export_to_onnx(
            torch.nn.Linear(3, 2).eval(),
            torch.randn(1, 3),
            str(destination),
            validate=False,
        )

    if rollback_fails:
        with pytest.raises(ExportCommitUnknown):
            export()
        recovery = next(tmp_path.glob(".onnx-recovery-*"))
        assert (recovery / "prior.onnx").read_bytes() == b"previous model"
    else:
        assert export() is False
        assert not list(tmp_path.glob(".onnx-recovery-*"))
    assert destination.read_bytes() == b"previous model"
    assert not list(tmp_path.glob(".wavedl-export-*"))
