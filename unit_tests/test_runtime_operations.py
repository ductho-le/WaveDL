"""Actual CPU inference and plots through the typed operation lifecycle."""

import hashlib
import json
import pickle
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from wavedl.core.generations import publish_generation, published_generations
from wavedl.core.inference_cli import inference_defaults, parse_inference_args
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, read_manifest
from wavedl.core.protocol import decode
from wavedl.journal import bootstrap_run
from wavedl.models import build_model


def test_pinned_operation_does_not_enumerate_unrelated_generations(
    tmp_path, monkeypatch
):
    """Pinned sources use exact capture instead of broad legacy discovery."""
    from wavedl import test
    from wavedl.core import checkpoint_source

    path = tmp_path / "checkpoints" / "gen-000001"
    captured = tmp_path / "private-capture"
    source = {"approved": "source pin"}
    journal = SimpleNamespace(
        bootstrap=SimpleNamespace(
            request_manifest=SimpleNamespace(checkpoint_source=source)
        )
    )
    args = SimpleNamespace(checkpoint=str(path))

    @contextmanager
    def snapshot(actual_path, actual_source):
        assert actual_path == path
        assert actual_source is source
        yield captured

    def legacy_discovery(*args, **kwargs):
        pytest.fail("Pinned inference must not enumerate all source generations")

    def load_checkpoint(actual_args, device, actual_journal, operation, directory):
        assert actual_args is args
        assert actual_journal is journal
        assert directory == captured
        return "consumed exact capture"

    monkeypatch.setattr(checkpoint_source, "snapshot_checkpoint_source", snapshot)
    monkeypatch.setattr(test, "resolve_checkpoint_path", legacy_discovery)
    monkeypatch.setattr(test, "_run_operation_checkpoint", load_checkpoint)
    assert (
        test._run_operation(args, "cpu", journal, OperationKind.TEST)
        == "consumed exact capture"
    )


def events(output, attempt=1):
    return [
        decode(line)
        for line in (output / f"attempts/{attempt}/events.jsonl")
        .read_bytes()
        .splitlines()
    ]


@pytest.fixture
def inference(tmp_path, monkeypatch):
    from wavedl import test

    torch.set_num_threads(1)
    inputs = np.random.default_rng(2).normal(size=(8, 32)).astype("float32")
    targets = np.random.default_rng(3).normal(size=(8, 1)).astype("float32")
    data = tmp_path / "data.npz"
    np.savez(data, X=inputs, y=targets)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = build_model("cnn", in_shape=(32,), out_size=1, pretrained=False)
    torch.save(model.state_dict(), checkpoint / "model_weights.pth")
    with (checkpoint / "scaler.pkl").open("wb") as stream:
        pickle.dump(StandardScaler().fit(targets), stream)
    with (checkpoint / "training_meta.pkl").open("wb") as stream:
        pickle.dump({"model_name": "cnn", "in_shape": (32,), "out_dim": 1}, stream)
    output = tmp_path / "output"
    args = parse_inference_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--data_path",
            str(data),
            "--output_dir",
            str(output),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--save_predictions",
        ]
    )
    monkeypatch.setattr(test, "parse_args", lambda: args)
    return test, args, output, checkpoint


def test_cpu_test_operation_metrics_and_prediction_artifact(inference):
    test, _args, output, _ = inference
    test.main()
    journal = events(output)
    assert journal[0].payload["device"] == "cpu"
    assert journal[0].payload["interpreter"] == sys.executable
    assert read_manifest(output / "manifest.json").operation == OperationKind.TEST
    assert journal[-1].payload["status"] == "completed"
    metric = next(item for item in journal if item.type == "metric")
    assert isinstance(metric.payload["value"], float)
    artifact = next(item for item in journal if item.type == "artifact")
    prediction = output / artifact.payload["path"]
    assert (
        artifact.payload["sha256"]
        == hashlib.sha256(prediction.read_bytes()).hexdigest()
    )
    assert artifact.payload["size"] == prediction.stat().st_size
    assert artifact.payload["path"].startswith("checkpoints/gen-")
    assert (output / "predictions.csv").exists()
    layout = RunLayout(output, read_manifest(output / "manifest.json").run_id)
    assert set(published_generations(layout)[0].files) == {"predictions.csv"}


@pytest.mark.parametrize(
    "exception,status",
    [
        (RuntimeError("setup failed"), "failed"),
        (KeyboardInterrupt("cancelled"), "cancelled"),
    ],
)
def test_setup_failure_and_cancellation_have_truthful_terminal_events(
    inference, monkeypatch, exception, status
):
    test, _args, output, _ = inference

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(test, "load_data_for_inference", fail)
    with pytest.raises(type(exception)):
        test.main()
    assert events(output)[-1].payload["status"] == status


def test_export_failure_is_nonzero_failed_operation(inference, monkeypatch):
    test, args, output, _ = inference
    args.export = "onnx"
    monkeypatch.setattr(test, "export_to_onnx", lambda **kwargs: False)
    with pytest.raises(RuntimeError, match="ONNX export failed"):
        test.main()
    assert events(output)[-1].payload["code"] == 1


def test_plot_consumes_published_generation_and_preserves_training_run(
    inference, tmp_path, monkeypatch
):
    test, args, _, checkpoint = inference
    training = bootstrap_run(
        tmp_path / "training",
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        environment={},
    )
    training.layout.checkpoints.mkdir()
    (checkpoint / "checkpoint_selection.json").write_text(
        json.dumps({"schema_version": 1, "kind": "best"})
    )
    generation = publish_generation(training.layout, checkpoint, attempt=1, step=1)
    original_manifest = training.layout.manifest.read_bytes()
    args.operation = "plot"
    args.plot = True
    args.save_predictions = False
    args.checkpoint = str(training.layout.run_dir)
    args.output_dir = str(training.layout.run_dir)
    # Real plotting, without unavailable external LaTeX dependencies.
    import matplotlib

    import wavedl.utils.metrics as plotting_metrics

    monkeypatch.setitem(matplotlib.rcParams, "text.usetex", False)
    monkeypatch.setattr(plotting_metrics, "_is_latex_available", lambda: False)
    test.main()
    output = Path(args.output_dir)
    assert output.is_relative_to(training.layout.run_dir / "operations")
    assert read_manifest(output / "manifest.json").operation == OperationKind.PLOT
    assert training.layout.manifest.read_bytes() == original_manifest
    journal = events(output)
    assert journal[-1].payload["status"] == "completed"
    plots = [
        item
        for item in journal
        if item.type == "artifact" and item.payload["kind"] == "plot"
    ]
    assert len(plots) == 9
    assert all(item.payload["path"].startswith("checkpoints/gen-") for item in plots)
    assert any(
        str(generation.path) in item.payload.get("message", "") for item in journal
    )


def test_injected_test_request_uses_declared_device_and_same_identity(
    inference, monkeypatch
):
    test, args, output, _ = inference
    approved = bootstrap_run(
        output,
        resolved_config=dict(vars(args)),
        runtime=RuntimeSelection(sys.executable, "selected", "cpu", {}),
        operation=OperationKind.TEST,
        environment={},
    )
    for name, value in approved.environment.items():
        monkeypatch.setenv(name, value)
    test.main()
    assert [path.name for path in (output / "attempts").iterdir()] == ["1"]
    assert events(output)[0].run_id == approved.layout.run_id
    assert events(output)[0].payload["device"] == "cpu"


def test_cpu_inference_consumes_pinned_publication_without_legacy_discovery(
    inference, monkeypatch, tmp_path
):
    from wavedl.core.checkpoint_source import capture_checkpoint_source

    test, args, output, checkpoint = inference
    training = bootstrap_run(
        tmp_path / "training-source",
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        environment={},
    )
    training.layout.checkpoints.mkdir()
    (checkpoint / "checkpoint_selection.json").write_text(
        json.dumps({"schema_version": 1, "kind": "best"})
    )
    generation = publish_generation(training.layout, checkpoint, attempt=1, step=1)
    source = capture_checkpoint_source(
        training.layout,
        generation.generation,
        manifest_sha256=hashlib.sha256(
            training.layout.manifest.read_bytes()
        ).hexdigest(),
    )
    args.checkpoint = str(generation.path)
    approved = bootstrap_run(
        output,
        resolved_config=dict(vars(args)),
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        operation=OperationKind.TEST,
        environment={},
    )
    request = replace(approved.request_manifest, checkpoint_source=source)
    (output / "attempts/1/runtime_request.json").write_text(
        json.dumps(request.to_dict()) + "\n"
    )
    for name, value in approved.environment.items():
        monkeypatch.setenv(name, value)

    original_resolve = test.resolve_checkpoint_path

    def legacy_discovery(path, **kwargs):
        assert Path(path) != generation.path, "Approved source must bypass discovery"
        assert Path(path).name.startswith("wavedl-inference-")
        return original_resolve(path, **kwargs)

    monkeypatch.setattr(test, "resolve_checkpoint_path", legacy_discovery)
    test.main()
    assert events(output)[-1].payload["status"] == "completed"
    assert (output / "predictions.csv").exists()


def test_inference_defaults_are_dependency_light_and_isolated():
    first = inference_defaults()
    first["plot_format"].append("changed")
    assert inference_defaults()["plot_format"] == ["png"]


def test_unavailable_device_failure_does_not_claim_cuda_execution(
    inference, monkeypatch
):
    test, args, output, _ = inference
    args.device = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="Requested operation device differs"):
        test.main()
    assert events(output)[0].payload["device"] == "cpu"
    assert events(output)[-1].payload["status"] == "failed"


def test_help_has_no_operation_side_effects(tmp_path, monkeypatch):
    from wavedl import test

    monkeypatch.setattr(sys, "argv", ["wavedl-test", "--help"])
    with pytest.raises(SystemExit) as result:
        test.main()
    assert result.value.code == 0
    assert not list(tmp_path.iterdir())


def test_plot_without_targets_reports_failed_not_completed(inference):
    test, args, output, _ = inference
    data = np.load(args.data_path)
    inputs = data["X"]
    data.close()
    np.savez(args.data_path, X=inputs)
    args.operation = "plot"
    with pytest.raises(ValueError, match="ground truth"):
        test.main()
    assert events(output)[-1].payload["status"] == "failed"


def test_single_sample_metrics_are_finite_json(inference):
    test, args, output, _ = inference
    with np.load(args.data_path) as data:
        inputs, targets = data["X"][:1], data["y"][:1]
    np.savez(args.data_path, X=inputs, y=targets)
    test.main()
    metric = next(event for event in events(output) if event.type == "metric")
    assert metric.payload["value"] is None
    assert metric.payload["metrics"]["pearson_corr"] is None


def _checkpoint_bytes(checkpoint):
    return {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }


def _assert_invalid_predictions(test, output, checkpoint, original, message):
    # Loader/inference validation raises ValueError for malformed inputs and
    # FloatingPointError for non-finite model output; both precede publication.
    with pytest.raises((ValueError, FloatingPointError), match=message):
        test.main()
    journal = events(output)
    assert journal[-1].payload["status"] == "failed"
    assert journal[-1].payload["code"] != 0
    assert not any(event.type == "artifact" for event in journal)
    assert not (output / "predictions.csv").exists()
    assert not list(output.rglob("COMPLETE.json"))
    assert _checkpoint_bytes(checkpoint) == original


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("with_targets", [False, True])
def test_nonfinite_model_predictions_fail_before_publication(
    inference, value, with_targets
):
    test, args, output, checkpoint = inference
    with np.load(args.data_path) as data:
        targets = data["y"]
    arrays = {"X": np.full((8, 32), value, dtype="float32")}
    if with_targets:
        arrays["y"] = targets
    np.savez(args.data_path, **arrays)
    original = _checkpoint_bytes(checkpoint)
    # Non-finite inputs are refused by the loader before the model runs.
    _assert_invalid_predictions(
        test, output, checkpoint, original, "Inputs must contain only finite"
    )


def test_finite_inputs_nonfinite_checkpoint_fail_before_publication(inference):
    test, args, output, checkpoint = inference
    with np.load(args.data_path) as data:
        inputs = data["X"]
    np.savez(args.data_path, X=inputs)
    path = checkpoint / "model_weights.pth"
    weights = torch.load(path, weights_only=True)
    for value in weights.values():
        if value.is_floating_point():
            value.fill_(float("nan"))
    torch.save(weights, path)
    original = _checkpoint_bytes(checkpoint)
    _assert_invalid_predictions(
        test, output, checkpoint, original, "Model produced non-finite predictions"
    )


def test_inverse_scaler_overflow_fails_before_publication(inference):
    test, args, output, checkpoint = inference
    with np.load(args.data_path) as data:
        inputs = data["X"]
    np.savez(args.data_path, X=inputs)
    path = checkpoint / "model_weights.pth"
    weights = torch.load(path, weights_only=True)
    for name, value in weights.items():
        if value.is_floating_point():
            value.fill_(10 if name.endswith("bias") else 0)
    torch.save(weights, path)
    path = checkpoint / "scaler.pkl"
    with path.open("rb") as stream:
        scaler = pickle.load(stream)
    scaler.scale_ = np.array([1e38])
    with path.open("wb") as stream:
        pickle.dump(scaler, stream)
    original = _checkpoint_bytes(checkpoint)
    with np.errstate(over="ignore"):
        _assert_invalid_predictions(
            test,
            output,
            checkpoint,
            original,
            "Inverse-scaled predictions contain nonfinite",
        )


@pytest.mark.parametrize("shape", [(8,), (7, 1), (8, 2)])
def test_invalid_model_prediction_shape_refuses_publication(
    inference, monkeypatch, shape
):
    """Unit shape fault injection complements the actual CPU numerical cases."""
    test, _args, output, checkpoint = inference
    original = _checkpoint_bytes(checkpoint)
    monkeypatch.setattr(test, "run_inference", lambda *args, **kwargs: np.zeros(shape))
    _assert_invalid_predictions(
        test, output, checkpoint, original, "Model prediction shape"
    )


@pytest.mark.parametrize(
    "shape,value,message",
    [
        ((8, 1), np.nan, "Inverse-scaled predictions contain nonfinite"),
        ((8, 1), np.inf, "Inverse-scaled predictions contain nonfinite"),
        ((8, 1), -np.inf, "Inverse-scaled predictions contain nonfinite"),
        ((8,), 0, "Inverse-scaled prediction shape"),
        ((7, 1), 0, "Inverse-scaled prediction shape"),
        ((8, 2), 0, "Inverse-scaled prediction shape"),
    ],
)
def test_invalid_inverse_scaler_result_refuses_publication(
    inference, monkeypatch, shape, value, message
):
    """Real CPU inference precedes a controlled malformed inverse result."""
    test, _args, output, checkpoint = inference
    original = _checkpoint_bytes(checkpoint)
    monkeypatch.setattr(
        StandardScaler,
        "inverse_transform",
        lambda self, values: np.full(shape, value),
    )
    _assert_invalid_predictions(test, output, checkpoint, original, message)


@pytest.mark.parametrize("with_targets", [False, True])
@pytest.mark.parametrize("save_predictions", [False, True])
def test_empty_evaluation_fails_without_requested_outputs(
    inference, with_targets, save_predictions
):
    test, args, output, checkpoint = inference
    arrays = {"X": np.empty((0, 32), dtype="float32")}
    if with_targets:
        arrays["y"] = np.empty((0, 1), dtype="float32")
    np.savez(args.data_path, **arrays)
    args.save_predictions = save_predictions
    original = _checkpoint_bytes(checkpoint)
    _assert_invalid_predictions(test, output, checkpoint, original, "non-empty array")


def test_finite_prediction_only_publishes_finite_values(inference):
    test, args, output, checkpoint = inference
    with np.load(args.data_path) as data:
        inputs = data["X"]
    np.savez(args.data_path, X=inputs)
    original = _checkpoint_bytes(checkpoint)
    test.main()
    assert events(output)[-1].payload["status"] == "completed"
    artifact = next(event for event in events(output) if event.type == "artifact")
    values = np.loadtxt(output / artifact.payload["path"], skiprows=1, delimiter=",")
    assert values.shape == (8,)
    assert np.isfinite(values).all()
    assert _checkpoint_bytes(checkpoint) == original


def test_export_only_still_bypasses_prediction_validation(inference, monkeypatch):
    """The new prediction policy does not change export-only dispatch."""
    test, args, output, _ = inference
    # The loader requires a nonempty input; export traces a single sample.
    np.savez(args.data_path, X=np.zeros((1, 32), dtype="float32"))
    args.export = "onnx"
    args.save_predictions = False
    monkeypatch.setattr(test, "export_to_onnx", lambda **kwargs: True)
    monkeypatch.setattr(test, "get_onnx_model_info", lambda path: {"error": "stub"})
    monkeypatch.setattr(test, "emit_operation_artifact", lambda *args, **kwargs: None)

    def unexpected_inference(*args, **kwargs):
        pytest.fail("Export-only operations must not run prediction inference")

    monkeypatch.setattr(test, "run_inference", unexpected_inference)
    test.main()
    assert events(output)[-1].payload["status"] == "completed"


def test_output_publication_failure_fails_without_artifact_announcement(
    inference, monkeypatch
):
    from wavedl import operations

    test, _args, output, _ = inference

    def fail(*args, **kwargs):
        raise OSError("publication failed")

    monkeypatch.setattr(operations, "publish_generation", fail)
    with pytest.raises(OSError, match="publication failed"):
        test.main()
    journal = events(output)
    assert journal[-1].payload["status"] == "failed"
    assert not any(event.type == "artifact" for event in journal)
    assert (output / "predictions.csv").exists()
    assert not list(output.glob(".operation-staging-*"))
