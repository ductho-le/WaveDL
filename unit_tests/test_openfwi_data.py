"""Synthetic CPU coverage for OpenFWI's data/preprocessing/metric contract."""

import importlib.util
import logging
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from wavedl import test as inference
from wavedl.models.base import BaseModel
from wavedl.train import (
    _build_model_for_targets,
    _save_best_checkpoint,
    _validate_resume_metadata,
)
from wavedl.utils.data import load_test_data, prepare_data
from wavedl.utils.openfwi import (
    ManifestInputs,
    MinMaxTargetScaler,
    OpenFWIManifestDataset,
    StreamingNormalizedMetrics,
    configure_tf32,
    dataset_config,
    preprocessing_from_args,
    ssim,
    transform_inputs,
)


class CPUAccelerator:
    is_main_process = True
    device = torch.device("cpu")
    num_processes = 1

    @staticmethod
    def wait_for_everyone():
        pass


def manifest(tmp_path, name="samples", lengths=(2, 3), value=0):
    root = tmp_path / name
    root.mkdir()
    lines, xs, ys = [], [], []
    for index, length in enumerate(lengths):
        x = np.arange(length * 5 * 8 * 6, dtype=np.float32).reshape(length, 5, 8, 6)
        x = x / 100 + value + index
        y = np.arange(length * 4 * 6, dtype=np.float32).reshape(length, 1, 4, 6)
        y = y + 1500 + value + index
        np.save(root / f"data{index}.npy", x)
        np.save(root / f"label{index}.npy", y)
        lines.append(f"data{index}.npy\tlabel{index}.npy")
        xs.append(x)
        ys.append(y)
    path = root / "manifest.txt"
    path.write_text("\n\n" + "\n\n".join(lines) + "\n")
    return path, np.concatenate(xs), np.concatenate(ys)


def args_for(path, **overrides):
    options = {
        "data_path": str(path),
        "batch_size": 2,
        "workers": 0,
        "seed": 13,
        "single_channel": False,
    }
    options.update(overrides)
    return Namespace(**options)


def test_manifest_relative_paths_unequal_shards_and_readonly_copies(tmp_path):
    path, xs, ys = manifest(tmp_path)
    source = OpenFWIManifestDataset(path)
    assert len(source) == 5
    assert source.in_channels == 5
    assert source.input_shape == (5, 8, 6)
    for index in range(len(xs)):
        x, y = source[index]
        np.testing.assert_array_equal(x, xs[index])
        np.testing.assert_array_equal(y, ys[index])
        x.fill_(999)
        y.fill_(999)
    assert all(
        isinstance(array, np.memmap) and not array.flags.writeable
        for array in source._maps.values()
    )
    np.testing.assert_array_equal(np.load(path.parent / "data0.npy"), xs[:2])
    np.testing.assert_array_equal(np.load(path.parent / "label1.npy"), ys[2:])
    restored = pickle.loads(pickle.dumps(source))
    assert not restored._maps
    np.testing.assert_array_equal(restored[3][0], xs[3])


@pytest.mark.parametrize(
    "line", ["data.npy", "data.npy label.npy", "data.npy\t", "a\tb\tc"]
)
def test_manifest_rejects_invalid_pairs(tmp_path, line):
    path = tmp_path / "bad.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match="expected"):
        OpenFWIManifestDataset(path)


def test_manifest_inference_inputs_remain_lazy(tmp_path):
    path, xs, ys = manifest(tmp_path)
    inputs, targets = load_test_data(str(path))
    assert isinstance(inputs, ManifestInputs)
    assert all(
        target for (_, target) in inputs.source._maps
    )  # seismic inputs still unmapped
    np.testing.assert_array_equal(inputs[1:3], xs[1:3])
    np.testing.assert_array_equal(targets, ys)


def test_preset_and_numpy_input_math():
    args = Namespace(openfwi_dataset="FlatVel_A")
    config = preprocessing_from_args(args)
    assert config["input_transform"] == "openfwi"
    assert config["target_scaling"] == "minmax"
    assert config["input_range"] == [-26.95, 52.77]
    assert len(dataset_config()) >= 10
    x = np.array([-100, -26.95, -1, 0, 1, 52.77, 100], dtype=np.float32)
    original = x.copy()
    low, high = [np.sign(v) * np.log1p(abs(v)) for v in config["input_range"]]
    expected = np.sign(x) * np.log1p(np.abs(x))
    expected -= low
    expected /= high - low
    expected = (expected - 0.5) * 2
    np.testing.assert_array_equal(transform_inputs(x, config), expected)
    np.testing.assert_allclose(
        transform_inputs(torch.from_numpy(x), config), expected, atol=2e-7
    )
    np.testing.assert_array_equal(x, original)
    assert transform_inputs(x, config)[0] < -1  # deliberately not clipped


@pytest.mark.parametrize("bounds", [None, [1, 1], [2, 1], [0, np.inf]])
def test_invalid_preprocessing_bounds(bounds):
    with pytest.raises(ValueError, match="range"):
        preprocessing_from_args(
            Namespace(input_transform="openfwi", input_range=bounds)
        )


def test_minmax_scaler_roundtrip_upstream_math():
    labels = np.linspace(1500, 4500, 120, dtype=np.float32).reshape(5, 1, 4, 6)
    scaler = MinMaxTargetScaler([1500, 4500], (1, 4, 6))
    expected = ((labels - 1500) / 3000 - 0.5) * 2
    np.testing.assert_array_equal(scaler.transform(labels), expected)
    np.testing.assert_allclose(scaler.inverse_transform(expected), labels, atol=3e-4)
    np.testing.assert_array_equal(scaler.mean_, [[[3000]]])
    np.testing.assert_array_equal(scaler.scale_, [[[1500]]])


def test_prepare_manifest_fixed_validation_no_cache_and_channel_inference(tmp_path):
    train, _, _ = manifest(tmp_path, "train", (2, 3), value=0)
    validation, _, labels = manifest(tmp_path, "validation", (4,), value=100)
    args = args_for(train, val_data_path=str(validation), openfwi_dataset="flatvel-a")
    train_dl, val_dl, scaler, in_shape, out_dim = prepare_data(
        args,
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(train_dl.dataset) == 5  # no random holdout
    assert len(val_dl.dataset) == 4
    assert in_shape == (8, 6)
    assert out_dim == 1
    assert args.in_channels == 5
    np.testing.assert_array_equal(val_dl.dataset[0][1], scaler.transform(labels[:1])[0])
    assert not list((tmp_path / "cache").glob("*.dat"))
    with (tmp_path / "cache" / "data_metadata.pkl").open("rb") as file:
        metadata = pickle.load(file)
    assert metadata["in_channels"] == 5
    assert metadata["preprocessing"] == args.preprocessing
    assert metadata["val_data_path"] == str(validation)


def test_fixed_validation_npz_fits_only_training_targets(tmp_path):
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    np.savez(train, input_train=np.ones((5, 8, 6)), output_train=np.arange(5)[:, None])
    np.savez(
        validation, input_train=np.zeros((3, 8, 6)), output_train=np.full((3, 1), 100)
    )
    train_dl, val_dl, scaler, shape, _ = prepare_data(
        args_for(train, val_data_path=str(validation)),
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(train_dl.dataset) == 5
    assert len(val_dl.dataset) == 3
    assert shape == (8, 6)
    np.testing.assert_array_equal(scaler.mean_, [2.0])


def test_manifest_random_split_and_resume_scaler(tmp_path):
    path, _, _ = manifest(tmp_path)
    args = args_for(path, openfwi_dataset="flatvel-a")
    result = prepare_data(
        args,
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(result[0].dataset) == 4
    assert len(result[1].dataset) == 1
    resumed = prepare_data(
        args,
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
        expected_scaler=result[2],
    )
    np.testing.assert_array_equal(resumed[2].scale_, result[2].scale_)


def test_reject_model_channels_before_construction():
    scaler = MinMaxTargetScaler([0, 1], (1, 70, 70))
    with pytest.raises(ValueError, match="in_channels=3"):
        _build_model_for_targets("inversionnet", (1000, 70), 1, scaler, in_channels=3)


@pytest.mark.parametrize("channels", [1, 2])
def test_ssim_matches_upstream_and_streams(channels):
    path = Path(os.environ.get("OPENFWI_SRC", "/tmp/openfwi_src")) / "pytorch_ssim.py"
    if not path.exists():
        pytest.skip("Optional upstream reference not installed")
    spec = importlib.util.spec_from_file_location("openfwi_reference_ssim", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    generator = torch.Generator().manual_seed(32)
    a = torch.rand(7, channels, 13, 17, generator=generator)
    b = torch.rand(7, channels, 13, 17, generator=generator)
    assert ssim(a, b).item() == pytest.approx(reference.ssim(a, b).item(), abs=1e-6)
    full, streamed = StreamingNormalizedMetrics(), StreamingNormalizedMetrics()
    full.update(a * 2 - 1, b * 2 - 1)
    for start, end in [(0, 2), (2, 3), (3, 7)]:
        streamed.update(a[start:end] * 2 - 1, b[start:end] * 2 - 1)
    for key, value in full.compute().items():
        assert streamed.compute()[key] == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("flag", ["no_tf32", "deterministic"])
def test_no_tf32_disables_both_backends(flag):
    matmul, cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        configure_tf32(Namespace(**{flag: True}))
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            matmul,
            cudnn,
        )


class TinyDense(BaseModel):
    supports_spatial_output = requires_spatial_output = True
    supported_input_channels = None

    def __init__(
        self, in_shape=(8, 6), out_size=1, in_channels=5, out_shape=(1, 4, 6), **kwargs
    ):
        super().__init__(in_shape, out_size)
        self.head = nn.Conv2d(in_channels, out_size, 1)
        self.output_grid = out_shape[-2:]

    def forward(self, x):
        return nn.functional.adaptive_avg_pool2d(self.head(x), self.output_grid)


def test_checkpoint_metadata_restored_and_inference_applies_transform(
    tmp_path, monkeypatch
):
    config = preprocessing_from_args(Namespace(openfwi_dataset="flatvel-a"))
    model = TinyDense()
    scaler = MinMaxTargetScaler([1500, 4500], (1, 4, 6))
    args = Namespace(
        output_dir=str(tmp_path),
        model="tiny_dense",
        in_channels=5,
        preprocessing=config,
    )
    with (tmp_path / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)

    class SavingAccelerator(CPUAccelerator):
        @staticmethod
        def save_state(path, **kwargs):
            Path(path).mkdir()
            torch.save(model.state_dict(), Path(path) / "model.bin")

        @staticmethod
        def unwrap_model(model, **kwargs):
            return model

    _save_best_checkpoint(
        SavingAccelerator(),
        model,
        args,
        0,
        0.5,
        (8, 6),
        1,
        scaler,
        logging.getLogger(__name__),
    )
    checkpoint = tmp_path / "best_checkpoint"
    with (checkpoint / "training_meta.pkl").open("rb") as file:
        metadata = pickle.load(file)
    assert metadata["in_channels"] == 5
    assert metadata["preprocessing"] == config
    monkeypatch.setattr(inference, "get_model", lambda name: TinyDense)
    monkeypatch.setattr(
        inference, "build_model", lambda name, **kwargs: TinyDense(**kwargs)
    )
    loaded, loaded_scaler = inference.load_checkpoint(
        checkpoint, (8, 6), 1, in_channels=5
    )
    assert loaded._wavedl_preprocessing == config
    assert loaded._wavedl_in_channels == 5
    raw = torch.randn(3, 5, 8, 6)
    expected = model(transform_inputs(raw, config)).detach().numpy()
    actual = inference.run_inference(
        loaded, raw, batch_size=2, device=torch.device("cpu")
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(loaded_scaler.bounds, scaler.bounds)
    with pytest.raises(ValueError, match="in_channels"):
        _validate_resume_metadata(
            metadata, "tiny_dense", (8, 6), 1, (1, 4, 6), in_channels=1
        )
    with pytest.raises(ValueError, match="preprocessing"):
        _validate_resume_metadata(
            metadata, "tiny_dense", (8, 6), 1, (1, 4, 6), preprocessing={}
        )


def test_onnx_embeds_input_transform_and_minmax_denorm(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = TinyDense().eval()
    model._wavedl_preprocessing = preprocessing_from_args(
        Namespace(openfwi_dataset="flatvel-a")
    )
    scaler = MinMaxTargetScaler([1500, 4500], (1, 4, 6))
    raw = torch.randn(2, 5, 8, 6)
    expected = scaler.inverse_transform(
        inference.run_inference(model, raw, device=torch.device("cpu"))
    )
    path = tmp_path / "transformed.onnx"
    assert inference.export_to_onnx(
        model, raw[:1], str(path), scaler=scaler, include_denorm=True
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": raw.numpy()})[0]
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=5e-4)
    metadata = {entry.key: entry.value for entry in onnx.load(path).metadata_props}
    assert metadata["input_transform_embedded"] == "true"
    assert '"input_transform": "openfwi"' in metadata["preprocessing"]


def test_train_to_test_manifest_smoke(tmp_path, monkeypatch, capsys):
    """Use the real trainer, loaders, metadata and test CLI without OpenFWI models."""
    from wavedl import train
    from wavedl.models.registry import MODEL_REGISTRY

    monkeypatch.setitem(MODEL_REGISTRY, "tiny_openfwi_test", TinyDense)
    path, _, _ = manifest(tmp_path, lengths=(8, 8))
    validation, _, _ = manifest(tmp_path, "val", (3,), value=20)
    output = tmp_path / "run"
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-train",
            "--model",
            "tiny_openfwi_test",
            "--data_path",
            str(path),
            "--val_data_path",
            str(validation),
            "--openfwi_dataset",
            "flatvel-a",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--workers",
            "0",
            "--precision",
            "no",
            "--no_tf32",
            "--output_dir",
            str(output),
        ],
    )
    # Suppress only training diagnostic image generation, not any computation.
    monkeypatch.setattr(
        train, "plot_training_progress", lambda *args, **kwargs: None, raising=False
    )
    train.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-test",
            "--checkpoint",
            str(output / "best_checkpoint"),
            "--data_path",
            str(validation),
            "--batch_size",
            "2",
            "--no_tf32",
            "--output_dir",
            str(tmp_path / "results"),
        ],
    )
    inference.main()
    report = capsys.readouterr().out
    assert "OPENFWI NORMALIZED METRICS" in report
    history = (output / "training_history.csv").read_text()
    assert "val_normalized_mae" in history
    assert "val_normalized_ssim" in history


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_channel_option(count):
    with pytest.raises(ValueError, match="positive integer"):
        preprocessing_from_args(Namespace(input_channels=count))


def test_manifest_explicit_channel_mismatch(tmp_path):
    path, _, _ = manifest(tmp_path)
    with pytest.raises(ValueError, match="channels"):
        load_test_data(str(path), input_channels=3)
    with pytest.raises(ValueError, match="channels"):
        prepare_data(
            args_for(path, input_channels=3),
            logging.getLogger(__name__),
            CPUAccelerator(),
            cache_dir=str(tmp_path / "cache"),
        )


def test_fixed_validation_can_mix_manifest_and_npz(tmp_path):
    path, x, y = manifest(tmp_path)
    validation = tmp_path / "val.npz"
    np.savez(validation, input_train=x[:2], output_train=y[:2])
    train_dl, val_dl, _, _, _ = prepare_data(
        args_for(path, val_data_path=str(validation), openfwi_dataset="flatvel-a"),
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(train_dl.dataset) == 5
    assert len(val_dl.dataset) == 2
    assert val_dl.dataset.in_channels == 5


@pytest.mark.parametrize(
    ("input_shape", "options"),
    [
        ((4, 1, 8, 10, 12), {}),
        ((4, 8, 10, 12), {"input_channels": 1}),
        ((4, 8, 10, 12), {"single_channel": True}),
    ],
)
def test_preprocessed_single_channel_volume_layout(tmp_path, input_shape, options):
    path = tmp_path / "volumes.npz"
    np.savez(
        path,
        input_train=np.ones(input_shape, dtype=np.float32),
        output_train=np.arange(4, dtype=np.float32)[:, None],
    )
    args = args_for(path, val_data_path=str(path), **options)
    train_dl, val_dl, _, in_shape, _ = prepare_data(
        args,
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert args.in_channels == 1
    assert in_shape == (8, 10, 12)
    for loader in (train_dl, val_dl):
        inputs, _ = next(iter(loader))
        assert inputs.shape == (2, 1, 8, 10, 12)
