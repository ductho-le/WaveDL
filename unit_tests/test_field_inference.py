"""Regression coverage for dense inference's public reporting/export helpers."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from wavedl.test import (
    ModelWithDenormalization,
    compute_metrics,
    export_to_onnx,
    plot_field_results,
    print_results,
    save_predictions,
)


def test_scalar_map_reporting_and_storage_use_one_channel(tmp_path, capsys):
    target = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    prediction = target + 1
    metrics = compute_metrics(target, prediction)

    print_results(target, prediction, metrics, verbose=True)
    output = capsys.readouterr().out
    assert "PER-CHANNEL MAE" in output
    assert "channel=P0" in output
    assert "P1" not in output

    # Invalid name counts fall back consistently across vector and field APIs.
    save_predictions(target, prediction, str(tmp_path / "maps.csv"), ["a", "b"])
    with np.load(tmp_path / "maps.npz") as saved:
        assert saved["channel_names"].tolist() == ["C0"]
        np.testing.assert_array_equal(saved["target"], target[:, None])
        np.testing.assert_array_equal(saved["prediction"], prediction[:, None])
        np.testing.assert_array_equal(saved["error"], np.ones((2, 1, 3, 4)))


def test_scalar_vector_reporting_and_storage(tmp_path, capsys):
    target = np.arange(4, dtype=np.float32)
    metrics = compute_metrics(target, target + 1)
    assert metrics["mae_p0"] == 1.0
    print_results(target, target + 1, metrics)
    assert "PER-PARAMETER MAE" in capsys.readouterr().out
    save_predictions(target, target + 1, str(tmp_path / "vector.csv"))
    saved = pd.read_csv(tmp_path / "vector.csv")
    assert len(saved) == 4
    assert "True_P0" in saved.columns


@pytest.mark.parametrize("shape", [(4, 1), (1, 1, 2, 2)])
def test_evaluation_metrics_preserve_large_offset_variation(shape):
    target = np.array([1e9, 1e9 + 64, 1e9, 1e9 + 64], dtype=np.float32)
    prediction = target[::-1].copy()
    metrics = compute_metrics(target.reshape(shape), prediction.reshape(shape))
    assert metrics["r2_score"] == pytest.approx(-3.0)
    assert metrics["pearson_corr"] == pytest.approx(-1.0)


def test_field_plot_handles_wrong_channel_name_count(tmp_path):
    target = np.arange(48, dtype=np.float32).reshape(1, 2, 4, 6)
    plot_field_results(target, target + 1, str(tmp_path), param_names=["only_one"])
    assert (tmp_path / "field_sample_000.png").stat().st_size > 0


@pytest.mark.parametrize("helper", ["metrics", "print", "save"])
def test_reporting_rejects_broadcastable_mismatched_shapes(helper, tmp_path):
    target = np.zeros((2, 2, 3, 4), dtype=np.float32)
    prediction = np.zeros((2, 1, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        if helper == "metrics":
            compute_metrics(target, prediction)
        elif helper == "print":
            print_results(target, prediction, {})
        else:
            save_predictions(target, prediction, str(tmp_path / "wrong.npz"))


def test_denormalization_preserves_large_mean_residual():
    mean = np.array([[[1e9 + 32]]], dtype=np.float64)
    scale = np.array([[[32.0]]], dtype=np.float64)
    wrapper = ModelWithDenormalization(nn.Identity(), mean, scale)
    scores = torch.tensor([[[[-1.0, 1.0]]]])
    expected = (scores.numpy().astype(np.float64) * scale + mean).astype(np.float32)
    np.testing.assert_array_equal(wrapper(scores).numpy(), expected)
    assert all(buffer.dtype == torch.float32 for buffer in wrapper.buffers())


def test_onnx_denormalization_preserves_large_mean_residual(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from wavedl.utils import SpatialTargetScaler

    target = np.array([[[[1e9, 1e9 + 64]]]], dtype=np.float32)
    scaler = SpatialTargetScaler().fit(target)
    scores = torch.tensor([[[[-1.0, 1.0]]]])
    path = tmp_path / "large_offset.onnx"
    assert export_to_onnx(
        nn.Identity(), scores, str(path), scaler=scaler, include_denorm=True
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    predictions = session.run(None, {"input": scores.numpy()})[0]
    np.testing.assert_array_equal(predictions, target)


def test_loading_unversioned_dense_checkpoint_requires_retraining(
    tmp_path, monkeypatch
):
    from wavedl import test as inference
    from wavedl.utils import SpatialTargetScaler

    model = nn.Linear(1, 1)
    model.supports_spatial_output = True
    model.requires_spatial_output = True
    torch.save(model.state_dict(), tmp_path / "model.bin")
    scaler = SpatialTargetScaler().fit(
        np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    )
    del scaler.normalization_version_
    with open(tmp_path / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    monkeypatch.setattr(inference, "get_model", lambda _name: model)
    monkeypatch.setattr(inference, "build_model", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="older target normalization version"):
        inference.load_checkpoint(
            str(tmp_path), (2, 2), 1, model_name="unet_field", out_shape=(1, 2, 2)
        )


@pytest.mark.parametrize("shape", [(3, 1), (3, 1, 2, 2)])
def test_physical_metrics_do_not_overflow_float32_squares(shape):
    target = np.full(shape, 1e20, dtype=np.float32)
    prediction = target * 0.5
    metrics = compute_metrics(target, prediction)
    assert np.isfinite(metrics["rmse"])
    assert metrics["rmse"] == pytest.approx(float(target.flat[0]) * 0.5)
    if len(shape) == 4:
        assert metrics["relative_l2"] == pytest.approx(0.5)


def test_dense_metrics_are_independent_of_streaming_chunk_size(monkeypatch):
    import wavedl.test as test_module

    rng = np.random.default_rng(5)
    target = (1e6 + rng.normal(size=(11, 2, 5, 6))).astype(np.float32)
    prediction = (target + rng.normal(scale=0.3, size=target.shape)).astype(np.float32)
    single_pass = compute_metrics(target, prediction)
    # One sample per chunk exercises every cross-chunk moment merge.
    monkeypatch.setattr(test_module, "_FIELD_METRIC_CHUNK_BYTES", 1)
    chunked = compute_metrics(target, prediction)
    assert chunked.keys() == single_pass.keys()
    for key, value in single_pass.items():
        assert chunked[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
