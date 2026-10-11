"""End-to-end contracts for dense 2D regression targets."""

import logging
import pickle
from argparse import Namespace

import h5py
import numpy as np
import pytest
import torch

from wavedl.models import build_model
from wavedl.test import (
    compute_metrics,
    load_checkpoint,
    run_inference,
    save_predictions,
)
from wavedl.train import _build_model_for_targets, _validate_resume_metadata
from wavedl.utils.data import _compute_file_hash, load_test_data, prepare_data
from wavedl.utils.losses import WeightedMSELoss
from wavedl.utils.targets import (
    SpatialTargetScaler,
    canonicalize_targets,
    flatten_targets_by_channel,
)


class _CPUAccelerator:
    is_main_process = True
    device = torch.device("cpu")
    num_processes = 1

    @staticmethod
    def wait_for_everyone():
        return None


def test_canonicalize_scalar_and_multichannel_fields():
    scalar = np.zeros((4, 8, 9), dtype=np.float32)
    multi = np.zeros((4, 3, 8, 9), dtype=np.float32)

    assert canonicalize_targets(scalar).shape == (4, 1, 8, 9)
    assert canonicalize_targets(multi).shape == (4, 3, 8, 9)


def test_spatial_scaler_round_trip_and_channel_statistics():
    values = np.zeros((3, 2, 4, 5), dtype=np.float32)
    values[:, 0] = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    values[:, 1] = 100.0 + 2.0 * values[:, 0]

    scaler = SpatialTargetScaler().fit(values)
    scaled = scaler.transform(values)
    recovered = scaler.inverse_transform(scaled)

    assert scaler.target_shape_ == (2, 4, 5)
    assert scaler.mean_.shape == (2, 1, 1)
    np.testing.assert_allclose(scaled.mean(axis=(0, 2, 3)), 0.0, atol=1e-6)
    np.testing.assert_allclose(scaled.std(axis=(0, 2, 3)), 1.0, atol=1e-6)
    np.testing.assert_allclose(recovered, values, atol=1e-5)


def test_spatial_scaler_batched_fit_is_stable_for_large_physical_offsets():
    rng = np.random.default_rng(17)
    values = 1.0e9 + rng.normal(size=(7, 2, 5, 6))

    direct = SpatialTargetScaler().fit(values)
    streamed = SpatialTargetScaler().fit_batches([values[:2], values[2:5], values[5:]])

    np.testing.assert_allclose(streamed.mean_, direct.mean_, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(streamed.var_, direct.var_, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(streamed.scale_, direct.scale_, rtol=1e-7, atol=1e-10)
    assert np.all((streamed.var_ > 0.5) & (streamed.var_ < 2.0))


def test_spatial_scaler_preserves_float32_transform_dtype():
    rng = np.random.default_rng(21)
    values = rng.normal(size=(3, 2, 4, 5)).astype(np.float32)
    scaler = SpatialTargetScaler().fit(values)

    scaled = scaler.transform(values)
    recovered = scaler.inverse_transform(scaled)

    assert scaled.dtype == np.float32
    assert recovered.dtype == np.float32


def test_spatial_scaler_preserves_large_offset_centering_and_inverse():
    values = np.array([1.0e9, 1.0e9 + 64], dtype=np.float32).reshape(2, 1, 1, 1)
    scaler = SpatialTargetScaler().fit(values)
    scaled = scaler.transform(values)

    np.testing.assert_array_equal(scaled.ravel(), [-1.0, 1.0])
    np.testing.assert_array_equal(scaler.inverse_transform(scaled), values)
    assert scaled.dtype == np.float32


def test_spatial_scaler_skips_empty_chunks_but_rejects_an_empty_fit():
    values = np.arange(24, dtype=np.float32).reshape(3, 1, 2, 4)
    direct = SpatialTargetScaler().fit(values)
    streamed = SpatialTargetScaler().fit_batches(
        [values[:0], values[:2], values[:0], values[2:]]
    )
    np.testing.assert_allclose(streamed.mean_, direct.mean_)
    np.testing.assert_allclose(streamed.scale_, direct.scale_)
    with pytest.raises(ValueError, match="empty"):
        SpatialTargetScaler().fit(values[:0])


@pytest.mark.parametrize("shape", [(2, 0, 3, 4), (2, 1, 0, 4), (2, 1, 3, 0)])
def test_spatial_scaler_rejects_empty_field_dimensions(shape):
    with pytest.raises(ValueError, match="positive"):
        SpatialTargetScaler().fit(np.empty(shape))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1.0j])
def test_spatial_scaler_rejects_nonfinite_and_complex_targets(invalid):
    values = np.full((2, 1, 3, 4), invalid)
    with pytest.raises(ValueError, match="finite real"):
        SpatialTargetScaler().fit(values)
    fitted = SpatialTargetScaler().fit(np.ones((2, 1, 3, 4)))
    with pytest.raises(ValueError, match="finite real"):
        fitted.transform(values)


def test_flatten_targets_preserves_channels():
    values = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    flattened = flatten_targets_by_channel(values)

    assert flattened.shape == (40, 3)
    np.testing.assert_array_equal(flattened[0], values[0, :, 0, 0])


def test_weighted_mse_weights_field_channels_not_width():
    prediction = torch.ones(1, 2, 3, 4)
    target = torch.zeros_like(prediction)
    loss = WeightedMSELoss(weights=[1.0, 3.0], reduction="none")

    weighted = loss(prediction, target)

    assert torch.all(weighted[:, 0] == 1.0)
    assert torch.all(weighted[:, 1] == 3.0)


def test_weighted_mse_rejects_accidental_field_channel_broadcast():
    with pytest.raises(ValueError, match="does not match"):
        WeightedMSELoss(weights=[1.0, 3.0])(
            torch.zeros(2, 2, 3, 4), torch.zeros(2, 1, 3, 4)
        )


@pytest.mark.parametrize(
    "weights", [[], [[1.0], [2.0]], [-1.0, 1.0], [float("nan")], [float("inf")]]
)
def test_weighted_mse_rejects_invalid_channel_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        WeightedMSELoss(weights=weights)


def test_fast_cache_hash_detects_changes_after_first_megabyte(tmp_path):
    path = tmp_path / "medium_data.bin"
    path.write_bytes(bytes(1536 * 1024))
    original = _compute_file_hash(str(path), mode="fast")
    with path.open("r+b") as file:
        file.seek(-1, 2)
        file.write(b"1")
    assert _compute_file_hash(str(path), mode="fast") != original


def test_prepare_data_keeps_dense_target_shape(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 16, 16)).astype(np.float32)
    y = rng.normal(size=(10, 2, 11, 13)).astype(np.float32)
    data_path = tmp_path / "field_train.npz"
    np.savez(data_path, input_train=x, output_train=y)
    args = Namespace(
        data_path=str(data_path),
        batch_size=2,
        workers=0,
        cache_validate="fast",
        single_channel=False,
        seed=7,
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    train_loader, _val_loader, scaler, in_shape, out_dim = prepare_data(
        args,
        logging.getLogger("field-test"),
        _CPUAccelerator(),
        cache_dir=str(cache_dir),
    )
    batch_x, batch_y = next(iter(train_loader))

    assert in_shape == (16, 16)
    assert out_dim == 2
    assert scaler.target_shape_ == (2, 11, 13)
    assert batch_x.shape == (2, 1, 16, 16)
    assert batch_y.shape == (2, 2, 11, 13)
    with open(cache_dir / "data_metadata.pkl", "rb") as file:
        metadata = pickle.load(file)
    assert metadata["output_kind"] == "field"
    assert metadata["target_shape"] == (2, 11, 13)
    assert (cache_dir / "target_data_cache.dat").exists()


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".mat"])
@pytest.mark.parametrize("channels", [None, 2])
def test_field_cache_preserves_values_and_fits_only_training_split(
    tmp_path, suffix, channels
):
    from sklearn.model_selection import train_test_split

    x = np.arange(10 * 8 * 9, dtype=np.float32).reshape(10, 8, 9)
    shape = (10, 5, 6) if channels is None else (10, channels, 5, 6)
    y = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path = tmp_path / f"training{suffix}"
    if suffix == ".npz":
        np.savez(path, input_train=x, output_train=y)
    else:
        with h5py.File(path, "w") as file:
            file.create_dataset("input_train", data=x.T if suffix == ".mat" else x)
            file.create_dataset("output_train", data=y.T if suffix == ".mat" else y)
    args = Namespace(
        data_path=str(path), batch_size=2, workers=0, cache_validate="fast", seed=7
    )
    logger = logging.getLogger("field-cache-values")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    train, val, scaler, _, _ = prepare_data(
        args, logger, _CPUAccelerator(), cache_dir=str(cache_dir)
    )
    indices, _ = train_test_split(np.arange(10), test_size=0.2, random_state=7)
    expected = SpatialTargetScaler().fit(y[indices])
    np.testing.assert_allclose(scaler.mean_, expected.mean_)
    np.testing.assert_allclose(scaler.scale_, expected.scale_)
    scaled = scaler.transform(y)
    for loader in (train, val):
        assert loader.dataset.targets is None
        for offset, index in enumerate(loader.dataset.indices):
            batch_x, batch_y = loader.dataset[offset]
            np.testing.assert_array_equal(batch_x.numpy(), x[index][None])
            np.testing.assert_allclose(batch_y.numpy(), scaled[index])
    # Reusing a valid field cache must retain the same normalization and maps.
    reused, _, cached_scaler, _, _ = prepare_data(
        args,
        logger,
        _CPUAccelerator(),
        cache_dir=str(cache_dir),
        expected_scaler=scaler,
    )
    np.testing.assert_array_equal(cached_scaler.mean_, scaler.mean_)
    np.testing.assert_array_equal(reused.dataset[0][1], train.dataset[0][1])


@pytest.mark.parametrize("invalidate_cache", [False, True])
def test_resume_scaler_mismatch_preserves_existing_cache(tmp_path, invalidate_cache):
    path = tmp_path / "data.npz"
    x = np.zeros((10, 8, 8), dtype=np.float32)
    y = np.arange(10 * 5 * 6, dtype=np.float32).reshape(10, 1, 5, 6)
    np.savez(path, input_train=x, output_train=y)
    args = Namespace(
        data_path=str(path), batch_size=2, workers=0, cache_validate="fast", seed=7
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    logger = logging.getLogger("resume-cache")
    _, _, scaler, _, _ = prepare_data(
        args, logger, _CPUAccelerator(), cache_dir=str(cache_dir)
    )
    before = {file.name: file.read_bytes() for file in cache_dir.iterdir()}
    if invalidate_cache:
        np.savez(path, input_train=x, output_train=y + 10)
    else:
        scaler.mean_ += 10  # Simulate a checkpoint from a different run.
    with pytest.raises(ValueError, match="Target normalization changed"):
        prepare_data(
            args,
            logger,
            _CPUAccelerator(),
            cache_dir=str(cache_dir),
            expected_scaler=scaler,
        )
    assert {file.name: file.read_bytes() for file in cache_dir.iterdir()} == before


def test_resume_scaler_rejection_reaches_non_main_rank(tmp_path, monkeypatch):
    path = tmp_path / "data.npz"
    np.savez(path, input_train=np.zeros((10, 8, 8)), output_train=np.ones((10, 5, 6)))
    args = Namespace(
        data_path=str(path), batch_size=2, workers=0, cache_validate="fast", seed=7
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    logger = logging.getLogger("resume-cache-distributed")
    _, _, scaler, _, _ = prepare_data(
        args, logger, _CPUAccelerator(), cache_dir=str(cache_dir)
    )
    worker = _CPUAccelerator()
    worker.is_main_process = False

    def broadcast_rejection(messages, src):
        assert src == 0
        messages[0] = "Target normalization changed since the checkpoint was saved."

    def unexpected_barrier():
        pytest.fail("Rejected ranks must raise before waiting on cache creation")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_rejection)
    worker.wait_for_everyone = unexpected_barrier
    with pytest.raises(ValueError, match="Target normalization changed"):
        prepare_data(
            args, logger, worker, cache_dir=str(cache_dir), expected_scaler=scaler
        )


def test_legacy_dense_cache_is_rebuilt_and_legacy_resume_is_rejected(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, input_train=np.zeros((10, 8, 8)), output_train=np.ones((10, 5, 6)))
    args = Namespace(
        data_path=str(path), batch_size=2, workers=0, cache_validate="fast", seed=7
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    logger = logging.getLogger("legacy-dense-normalization")
    _, _, scaler, _, _ = prepare_data(
        args, logger, _CPUAccelerator(), cache_dir=str(cache_dir)
    )
    meta_path = cache_dir / "data_metadata.pkl"
    with meta_path.open("rb") as file:
        metadata = pickle.load(file)
    metadata.pop("target_normalization_version")
    with meta_path.open("wb") as file:
        pickle.dump(metadata, file)
    target_cache = np.memmap(
        cache_dir / "target_data_cache.dat", mode="r+", dtype=np.float32
    )
    target_cache[:] = 123  # Same-size stale contents must not be reused.
    target_cache.flush()
    del target_cache
    del scaler.normalization_version_
    with pytest.raises(ValueError, match="older target normalization version"):
        prepare_data(
            args,
            logger,
            _CPUAccelerator(),
            cache_dir=str(cache_dir),
            expected_scaler=scaler,
        )
    rebuilt, _, current, _, _ = prepare_data(
        args, logger, _CPUAccelerator(), cache_dir=str(cache_dir)
    )
    assert current.normalization_version_ == SpatialTargetScaler.NORMALIZATION_VERSION
    assert torch.all(rebuilt.dataset[0][1] == 0)


def test_model_target_compatibility_is_checked():
    field_scaler = SpatialTargetScaler().fit(np.zeros((2, 1, 8, 8)))

    with pytest.raises(ValueError, match="vector outputs"):
        _build_model_for_targets("cnn", (32, 32), 1, field_scaler)

    field_model, target_shape = _build_model_for_targets(
        "unet_field", (8, 8), 1, field_scaler, base_channels=2, depth=2
    )
    assert field_model.supports_spatial_output
    assert target_shape == (1, 8, 8)


def test_resume_rejects_changed_dense_target_grid():
    metadata = {
        "model_name": "unet_field",
        "in_shape": (16, 16),
        "out_dim": 1,
        "target_shape": (1, 8, 8),
    }

    with pytest.raises(ValueError, match="target_shape"):
        _validate_resume_metadata(
            metadata,
            "unet_field",
            (16, 16),
            1,
            (1, 12, 12),
        )


def test_field_inference_metrics_and_npz_export(tmp_path):
    model = build_model(
        "unet_field",
        in_shape=(8, 8),
        out_size=1,
        out_shape=(1, 6, 7),
        base_channels=2,
        depth=2,
    )
    prediction = run_inference(model, torch.randn(3, 1, 8, 8), batch_size=2)
    target = prediction + 0.25
    metrics = compute_metrics(target, prediction)
    output_path = tmp_path / "predictions.npz"
    save_predictions(target, prediction, str(output_path), ["shear_modulus"])

    assert prediction.shape == (3, 1, 6, 7)
    assert metrics["mae_avg"] == pytest.approx(0.25)
    assert "relative_l2" in metrics
    with np.load(output_path) as saved:
        assert saved["target"].shape == (3, 1, 6, 7)
        assert saved["prediction"].shape == (3, 1, 6, 7)
        assert saved["channel_names"].tolist() == ["shear_modulus"]


def test_load_checkpoint_infers_field_shape_and_prefers_local_scaler(
    tmp_path, monkeypatch
):
    import wavedl.test as test_module

    checkpoint_dir = tmp_path / "portable" / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True)
    local_scaler = SpatialTargetScaler().fit(
        np.arange(2 * 6 * 7, dtype=np.float32).reshape(2, 1, 6, 7)
    )
    parent_scaler = SpatialTargetScaler().fit(
        np.full((2, 1, 6, 7), 999.0, dtype=np.float32)
    )
    with open(checkpoint_dir / "scaler.pkl", "wb") as file:
        pickle.dump(local_scaler, file)
    with open(checkpoint_dir.parent / "scaler.pkl", "wb") as file:
        pickle.dump(parent_scaler, file)
    with open(checkpoint_dir / "training_meta.pkl", "wb") as file:
        pickle.dump(
            {
                "model_name": "unet_field",
                "target_shape": (1, 6, 7),
                "epoch": 1,
            },
            file,
        )
    torch.save({}, checkpoint_dir / "model.bin")

    captured = {}

    class _EmptyFieldModel(torch.nn.Module):
        def parameter_summary(self):
            return {"trainable_parameters": 0, "total_mb": 0.0}

    def _build_model(model_name, in_shape, out_size, **kwargs):
        captured.update(kwargs)
        return _EmptyFieldModel()

    monkeypatch.setattr(test_module, "build_model", _build_model)
    _model, loaded_scaler = load_checkpoint(
        str(checkpoint_dir), in_shape=(8, 8), out_size=1
    )

    assert captured["out_shape"] == (1, 6, 7)
    np.testing.assert_array_equal(loaded_scaler.mean_, local_scaler.mean_)


def test_load_test_data_canonicalizes_scalar_field(tmp_path):
    path = tmp_path / "field_test.npz"
    np.savez(
        path,
        input_test=np.zeros((3, 8, 8), dtype=np.float32),
        output_test=np.zeros((3, 5, 6), dtype=np.float32),
    )

    x, y = load_test_data(str(path))

    assert x.shape == (3, 1, 8, 8)
    assert y is not None
    assert y.shape == (3, 1, 5, 6)


@pytest.mark.parametrize("suffix", [".h5", ".mat"])
def test_load_test_data_preserves_multichannel_fields_across_hdf5_formats(
    tmp_path, suffix
):
    logical_x = np.arange(3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)
    logical_y = np.arange(3 * 2 * 5 * 6, dtype=np.float32).reshape(3, 2, 5, 6)
    path = tmp_path / f"field{suffix}"

    with h5py.File(path, "w") as file:
        if suffix == ".mat":
            # MAT v7.3 arrays are stored in MATLAB order; WaveDL reverses axes.
            file.create_dataset("input_test", data=logical_x.T)
            file.create_dataset("output_test", data=logical_y.T)
        else:
            file.create_dataset("input_test", data=logical_x)
            file.create_dataset("output_test", data=logical_y)

    x, y = load_test_data(str(path))

    assert x.shape == (3, 1, 8, 9)
    assert y is not None
    assert y.shape == (3, 2, 5, 6)
    np.testing.assert_array_equal(y.numpy(), logical_y)


@pytest.mark.parametrize("suffix", [".h5", ".mat"])
def test_hpo_detects_dense_shape_without_eager_format_conversion(tmp_path, suffix):
    from wavedl.hpo import _detect_target_shape

    logical_x = np.zeros((4, 8, 9), dtype=np.float32)
    logical_y = np.zeros((4, 2, 5, 6), dtype=np.float32)
    path = tmp_path / f"field_train{suffix}"
    with h5py.File(path, "w") as file:
        if suffix == ".mat":
            file.create_dataset("input_train", data=logical_x.T)
            file.create_dataset("output_train", data=logical_y.T)
        else:
            file.create_dataset("input_train", data=logical_x)
            file.create_dataset("output_train", data=logical_y)

    assert _detect_target_shape(str(path)) == (2, 5, 6)
