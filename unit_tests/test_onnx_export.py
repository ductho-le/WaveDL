"""Real single-file exports remain runnable after independent download/move."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from wavedl import test  # noqa: E402
from wavedl.core.generations import published_generations  # noqa: E402
from wavedl.core.manifest import RuntimeSelection  # noqa: E402
from wavedl.journal import bootstrap_run  # noqa: E402
from wavedl.models.cnn import CNN  # noqa: E402
from wavedl.operations import (  # noqa: E402
    emit_operation_artifact,
    publish_operation_outputs,
)
from wavedl.utils.onnx_export import (  # noqa: E402
    ExportCommitUnknown,
    replace_export,
    validate_self_contained_onnx,
)


def external_model(path, location="missing.data"):
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


@pytest.mark.parametrize("opset", range(11, 18))
def test_actual_cnn_exact_opset_and_dynamic_batch_golden_outputs(tmp_path, opset):
    """Registered CNN exports exactly the requested opset, not an upgraded one."""
    torch.manual_seed(14)
    model = CNN(in_shape=(32, 32), out_size=2, base_channels=8).eval()
    sample = torch.randn(2, 1, 32, 32)
    output = tmp_path / "cnn.onnx"
    assert test.export_to_onnx(model, sample, str(output), opset_version=opset)
    validate_self_contained_onnx(output, expected_opset=opset)
    graph = onnx.load(output)
    assert [(item.domain, item.version) for item in graph.opset_import] == [("", opset)]
    runtime = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for batch_size in (1, 3):
        inputs = torch.randn(batch_size, 1, 32, 32)
        with torch.inference_mode():
            expected = model(inputs).numpy()
        np.testing.assert_allclose(
            runtime.run(None, {"input": inputs.numpy()})[0],
            expected,
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("prior_exists", [False, True])
def test_exporter_upgrading_opset_refused_before_commit(
    tmp_path, monkeypatch, validate, prior_exists, caplog
):
    """An exporter ignoring the requested version cannot publish changed bytes."""
    output = tmp_path / "model.onnx"
    if prior_exists:
        output.write_bytes(b"previous model")

    def upgrading_export(model, sample, destination, *, dynamo=True, **kwargs):
        assert dynamo is False
        assert kwargs["opset_version"] == 17
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["input"], ["predictions"])],
            "upgraded",
            [
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [1, 3]
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "predictions", onnx.TensorProto.FLOAT, [1, 3]
                )
            ],
        )
        exported = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 18)]
        )
        Path(destination).write_bytes(exported.SerializeToString())

    monkeypatch.setattr(torch.onnx, "export", upgrading_export)
    monkeypatch.setattr(
        test,
        "_validate_onnx_export",
        lambda *args: pytest.fail("Numerical check preceded mandatory opset check"),
    )
    monkeypatch.setattr(
        "wavedl.utils.onnx_export.replace_export",
        lambda *args: pytest.fail("Invalid opset reached commit"),
    )
    assert not test.export_to_onnx(
        torch.nn.Linear(3, 2), torch.randn(1, 3), str(output), validate=validate
    )
    assert "requested opset 17" in caplog.text
    assert "produced standard-domain opsets [18]" in caplog.text
    if prior_exists:
        assert output.read_bytes() == b"previous model"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".onnx-export-*"))
    assert not list(tmp_path.glob(".onnx-recovery-*"))


@pytest.mark.parametrize("denormalize", [False, True])
def test_actual_export_download_is_single_file_and_golden_outputs(
    tmp_path, denormalize
):
    torch.manual_seed(14)
    model = torch.nn.Linear(3, 2).eval()
    sample = torch.randn(4, 3)
    exported = tmp_path / "model.onnx"
    scaler = SimpleNamespace(mean_=np.array([2.0, -3.0]), scale_=np.array([4.0, 5.0]))
    assert test.export_to_onnx(
        model, sample, str(exported), scaler=scaler, include_denorm=denormalize
    )
    assert not list(tmp_path.glob("*.data"))
    download = tmp_path / "download" / "different-name.onnx"
    download.parent.mkdir()
    shutil.copyfile(exported, download)
    exported.unlink()
    validate_self_contained_onnx(download)
    runtime = ort.InferenceSession(str(download), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(sample).numpy()
    if denormalize:
        expected = expected * scaler.scale_ + scaler.mean_
    np.testing.assert_allclose(
        runtime.run(None, {"input": sample.numpy()})[0], expected, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    "location", ["missing.data", "../secret.data", "/tmp/secret.data"]
)
def test_external_tensor_refs_never_loaded_or_followed(tmp_path, location):
    path = tmp_path / "model.onnx"
    external_model(path, location)
    with pytest.raises(ValueError, match="external tensor"):
        validate_self_contained_onnx(path)


def test_exporter_ignoring_single_file_option_cannot_replace_prior_output(
    tmp_path, monkeypatch
):
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")

    def bad_export(model, sample, output, *, external_data=True, **kwargs):
        assert external_data is False
        external_model(Path(output))

    monkeypatch.setattr(torch.onnx, "export", bad_export)
    assert not test.export_to_onnx(
        torch.nn.Linear(3, 2), torch.randn(1, 3), str(destination), validate=False
    )
    assert destination.read_bytes() == b"previous model"
    assert not list(tmp_path.glob(".onnx-export-*"))


def test_large_export_refused_before_exporter(tmp_path, monkeypatch):
    monkeypatch.setattr("wavedl.utils.onnx_export.MAX_ONNX_BYTES", 8)

    def unexpected(*args, **kwargs):
        pytest.fail("Oversized model reached exporter")

    monkeypatch.setattr(torch.onnx, "export", unexpected)
    assert not test.export_to_onnx(
        torch.nn.Linear(3, 2), torch.randn(1, 3), str(tmp_path / "model.onnx")
    )


def test_tampered_external_model_not_published_in_generation(tmp_path):
    bootstrap = bootstrap_run(
        tmp_path / "run",
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        environment={},
    )
    journal = SimpleNamespace(
        bootstrap=bootstrap,
        _operation_artifacts={},
        emit=lambda *args: pytest.fail("Invalid ONNX announced"),
    )
    output = bootstrap.layout.run_dir / "model.onnx"
    external_model(output)
    emit_operation_artifact(journal, output, kind="other", label="ONNX model")
    with pytest.raises(ValueError, match="external tensor"):
        publish_operation_outputs(journal)
    assert not published_generations(bootstrap.layout)


def test_actual_operation_generation_contains_runnable_model_only(tmp_path):
    bootstrap = bootstrap_run(
        tmp_path / "run",
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        environment={},
    )
    announcements = []
    journal = SimpleNamespace(
        bootstrap=bootstrap,
        _operation_artifacts={},
        emit=lambda *args: announcements.append(args),
    )
    output = bootstrap.layout.run_dir / "model.onnx"
    model = torch.nn.Linear(3, 2).eval()
    sample = torch.randn(4, 3)
    assert test.export_to_onnx(model, sample, str(output))
    emit_operation_artifact(journal, output, kind="other", label="ONNX model")
    publish_operation_outputs(journal)
    generation = published_generations(bootstrap.layout)[0]
    assert set(generation.files) == {"model.onnx"}
    download = tmp_path / "download.onnx"
    shutil.copyfile(generation.path / "model.onnx", download)
    validate_self_contained_onnx(download)
    actual = ort.InferenceSession(
        str(download), providers=["CPUExecutionProvider"]
    ).run(None, {"input": sample[:2].numpy()})[0]
    with torch.inference_mode():
        np.testing.assert_allclose(
            actual, model(sample[:2]).numpy(), rtol=1e-4, atol=1e-5
        )
    assert announcements[0][0] == "artifact"


def test_nested_graph_external_tensor_is_rejected(tmp_path):
    path = tmp_path / "model.onnx"
    external_model(path)
    inner = onnx.load(path, load_external_data=False).graph
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


@pytest.mark.parametrize("failed_sync", [1, 2, 3])
def test_commit_sync_failure_preserves_prior_target(tmp_path, monkeypatch, failed_sync):
    staged = tmp_path / "staged.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    destination.write_bytes(b"previous model")
    calls = 0

    def fail_once(path):
        nonlocal calls
        calls += 1
        if calls == failed_sync:
            raise OSError("injected directory sync failure")

    monkeypatch.setattr("wavedl.utils.onnx_export.sync_directory", fail_once)
    with pytest.raises(OSError, match="sync failure"):
        replace_export(staged, destination)
    assert destination.read_bytes() == b"previous model"
    assert not list(tmp_path.glob(".onnx-recovery-*"))


@pytest.mark.parametrize("failed_rollback", ["replace", "fsync"])
def test_failed_rollback_retains_recoverable_prior_backup(
    tmp_path, monkeypatch, failed_rollback
):
    import os

    staged = tmp_path / "staged.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    destination.write_bytes(b"previous model")
    sync_calls = 0

    def fail_sync(path):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 3 or (failed_rollback == "fsync" and sync_calls == 4):
            raise OSError("injected directory sync failure")

    original = os.replace
    replace_calls = 0

    def fail_replace(source, target):
        nonlocal replace_calls
        replace_calls += 1
        if failed_rollback == "replace" and replace_calls == 2:
            raise OSError("injected rollback replacement failure")
        return original(source, target)

    monkeypatch.setattr("wavedl.utils.onnx_export.sync_directory", fail_sync)
    monkeypatch.setattr("wavedl.utils.onnx_export.os.replace", fail_replace)
    with pytest.raises(ExportCommitUnknown, match="recovery retained"):
        replace_export(staged, destination)
    recovery = list(tmp_path.glob(".onnx-recovery-*"))
    assert len(recovery) == 1
    assert (recovery[0] / "prior.onnx").read_bytes() == b"previous model"


def test_first_export_postcommit_failure_removes_only_its_new_target(
    tmp_path, monkeypatch
):
    staged = tmp_path / "staged.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    calls = 0

    def fail_once(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("postcommit fsync failure")

    monkeypatch.setattr("wavedl.utils.onnx_export.sync_directory", fail_once)
    with pytest.raises(OSError):
        replace_export(staged, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".onnx-recovery-*"))


def test_symlink_destination_not_followed_or_replaced(tmp_path):
    staged = tmp_path / "staged.onnx"
    actual = tmp_path / "unrelated.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    actual.write_bytes(b"private prior")
    destination.symlink_to(actual)
    with pytest.raises(ValueError, match="symlink"):
        replace_export(staged, destination)
    assert destination.is_symlink() and actual.read_bytes() == b"private prior"


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_export_api_does_not_hide_unknown_commit_or_overwrite_prior_on_false(
    tmp_path, monkeypatch, rollback_fails
):
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")

    def fake_export(model, sample, output, *, external_data=True, **kwargs):
        Path(output).write_bytes(b"new model")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(
        "wavedl.utils.onnx_export.validate_self_contained_onnx",
        lambda path, **kwargs: None,
    )
    calls = 0

    def fail_sync(path):
        nonlocal calls
        calls += 1
        if calls == 3 or (rollback_fails and calls == 4):
            raise OSError("injected directory sync failure")

    monkeypatch.setattr("wavedl.utils.onnx_export.sync_directory", fail_sync)

    def export():
        return test.export_to_onnx(
            torch.nn.Linear(3, 2), torch.randn(1, 3), str(destination), validate=False
        )

    if rollback_fails:
        with pytest.raises(ExportCommitUnknown):
            export()
        assert (
            next(tmp_path.glob(".onnx-recovery-*")) / "prior.onnx"
        ).read_bytes() == b"previous model"
    else:
        assert export() is False
        assert not list(tmp_path.glob(".onnx-recovery-*"))
    assert destination.read_bytes() == b"previous model"
    assert not list(tmp_path.glob(".onnx-export-*"))


def test_prior_inode_fsync_failure_precedes_any_commit(tmp_path, monkeypatch):
    staged = tmp_path / "staged.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    destination.write_bytes(b"previous model")

    def fail_fsync(descriptor):
        raise OSError("prior inode fsync failure")

    monkeypatch.setattr("wavedl.utils.onnx_export.os.fsync", fail_fsync)
    monkeypatch.setattr(
        "wavedl.utils.onnx_export.os.replace",
        lambda *args: pytest.fail("Commit preceded durable prior bytes"),
    )
    with pytest.raises(OSError, match="prior inode"):
        replace_export(staged, destination)
    assert destination.read_bytes() == b"previous model"
    assert staged.read_bytes() == b"new model"
    assert not list(tmp_path.glob(".onnx-recovery-*"))


def test_prior_bytes_sync_before_directories_and_commit(tmp_path, monkeypatch):
    import os

    staged = tmp_path / "staged.onnx"
    destination = tmp_path / "model.onnx"
    staged.write_bytes(b"new model")
    destination.write_bytes(b"previous model")
    ordering = []
    original_fsync = os.fsync
    original_replace = os.replace

    def file_sync(descriptor):
        ordering.append("file")
        original_fsync(descriptor)

    def directory_sync(path):
        ordering.append("directory")

    def commit(source, target):
        ordering.append("commit")
        return original_replace(source, target)

    monkeypatch.setattr("wavedl.utils.onnx_export.os.fsync", file_sync)
    monkeypatch.setattr("wavedl.utils.onnx_export.sync_directory", directory_sync)
    monkeypatch.setattr("wavedl.utils.onnx_export.os.replace", commit)
    replace_export(staged, destination)
    assert ordering == ["file", "directory", "directory", "commit", "directory"]
    assert destination.read_bytes() == b"new model"
