"""Unit tests for domain-agnostic shard manifests, preprocessing, and TF32 controls."""

import logging
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
from wavedl.utils.config import validate_config
from wavedl.utils.data import load_test_data, memmap_worker_init_fn, prepare_data
from wavedl.utils.preprocessing import (
    ManifestInputs,
    MinMaxTargetScaler,
    ShardManifestDataset,
    configure_tf32,
    inverse_transform_inputs,
    prepare_preprocessed_data,
    preprocessing_from_args,
    transform_inputs,
)


class CPUAccelerator:
    is_main_process = True
    device = torch.device("cpu")
    num_processes = 1

    @staticmethod
    def wait_for_everyone():
        pass


def make_field_manifest(tmp_path, name="samples", lengths=(2, 3), channels=5, value=0):
    root = tmp_path / name
    root.mkdir()
    lines, xs, ys = [], [], []
    for index, length in enumerate(lengths):
        x = np.arange(length * channels * 8 * 6, dtype=np.float32).reshape(
            length, channels, 8, 6
        )
        x = x / 100.0 + value + index
        y = np.arange(length * 4 * 6, dtype=np.float32).reshape(length, 1, 4, 6)
        y = y + 1500.0 + value + index
        np.save(root / f"data{index}.npy", x)
        np.save(root / f"label{index}.npy", y)
        lines.append(f"data{index}.npy\tlabel{index}.npy")
        xs.append(x)
        ys.append(y)
    path = root / "manifest.txt"
    path.write_text("\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")
    return path, np.concatenate(xs), np.concatenate(ys)


def make_1d_vector_manifest(tmp_path, name="wave1d", lengths=(3, 4), channels=2):
    root = tmp_path / name
    root.mkdir()
    lines, xs, ys = [], [], []
    for index, length in enumerate(lengths):
        x = np.arange(length * channels * 16, dtype=np.float32).reshape(
            length, channels, 16
        )
        y = np.arange(length * 3, dtype=np.float32).reshape(length, 3) + index * 10
        np.save(root / f"sig{index}.npy", x)
        np.save(root / f"param{index}.npy", y)
        lines.append(f"sig{index}.npy\tparam{index}.npy")
        xs.append(x)
        ys.append(y)
    path = root / "manifest.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


class TinyMultiChannelDense(BaseModel):
    supports_spatial_output = True
    requires_spatial_output = True
    supported_input_channels = None

    def __init__(
        self, in_shape=(8, 6), out_size=1, in_channels=5, out_shape=(1, 4, 6), **kwargs
    ):
        super().__init__(in_shape, out_size)
        self.head = nn.Conv2d(in_channels, out_size, 1)
        self.output_grid = out_shape[-2:]

    def forward(self, x):
        return nn.functional.adaptive_avg_pool2d(self.head(x), self.output_grid)


def test_manifest_relative_paths_unequal_shards_and_readonly_copies(tmp_path):
    path, xs, ys = make_field_manifest(tmp_path)
    source = ShardManifestDataset(path)
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


def test_manifest_supports_1d_waveforms_with_vector_targets(tmp_path):
    path, xs, _ = make_1d_vector_manifest(tmp_path)
    train_dl, val_dl, scaler, in_shape, out_dim = prepare_data(
        args_for(path, input_channels=2),
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert in_shape == (16,)
    assert out_dim == 3
    assert len(train_dl.dataset) + len(val_dl.dataset) == len(xs)
    bx, by = next(iter(train_dl))
    assert bx.shape[1:] == (2, 16)
    assert by.shape[1:] == (3,)
    assert scaler.mean_.shape == (3,)


@pytest.mark.parametrize(
    "line", ["data.npy", "data.npy label.npy", "data.npy\t", "a\tb\tc"]
)
def test_manifest_rejects_invalid_pairs(tmp_path, line):
    path = tmp_path / "bad.txt"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        ShardManifestDataset(path)


def test_manifest_inference_inputs_remain_lazy(tmp_path):
    path, xs, ys = make_field_manifest(tmp_path)
    inputs, targets = load_test_data(str(path))
    assert isinstance(inputs, ManifestInputs)
    assert all(target for (_, target) in inputs.source._maps)
    np.testing.assert_array_equal(inputs[1:3], xs[1:3])
    np.testing.assert_array_equal(targets, ys)


def test_signed_log_minmax_input_transform_math_and_dtype():
    config = preprocessing_from_args(
        Namespace(
            input_transform="signed_log_minmax",
            input_range=[-25.0, 50.0],
            target_scaling="minmax",
            target_range=[1500.0, 4500.0],
        )
    )
    assert config["input_transform"] == "signed_log_minmax"
    assert config["target_scaling"] == "minmax"
    x = np.array([-100, -25.0, -1, 0, 1, 50.0, 100], dtype=np.float32)
    original = x.copy()
    low, high = [np.sign(v) * np.log1p(abs(v)) for v in config["input_range"]]
    expected = np.sign(x) * np.log1p(np.abs(x))
    expected -= low
    expected /= high - low
    expected = (expected - 0.5) * 2
    out_np = transform_inputs(x, config)
    assert out_np.dtype == np.float32
    np.testing.assert_allclose(out_np, expected, atol=5e-7)
    assert transform_inputs(x.astype(np.float64), config).dtype == np.float32
    np.testing.assert_allclose(
        transform_inputs(torch.from_numpy(x), config), expected, atol=2e-7
    )
    np.testing.assert_array_equal(x, original)
    assert out_np[0] < -1  # deliberately unclipped


@pytest.mark.parametrize("bounds", [None, [1, 1], [2, 1], [0, np.inf]])
def test_invalid_preprocessing_bounds(bounds):
    with pytest.raises(ValueError, match="range"):
        preprocessing_from_args(
            Namespace(input_transform="signed_log_minmax", input_range=bounds)
        )
    with pytest.raises(ValueError, match="range"):
        preprocessing_from_args(Namespace(target_scaling="minmax", target_range=bounds))


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_channel_option(count):
    with pytest.raises(ValueError, match="positive integer"):
        preprocessing_from_args(Namespace(input_channels=count))


def test_minmax_scaler_roundtrip_vector_and_field():
    labels = np.linspace(1500, 4500, 120, dtype=np.float32).reshape(5, 1, 4, 6)
    scaler = MinMaxTargetScaler([1500, 4500], (1, 4, 6))
    expected = (((labels.astype(np.float64) - 1500.0) / 3000.0 - 0.5) * 2.0).astype(
        np.float32
    )
    np.testing.assert_array_equal(scaler.transform(labels), expected)
    np.testing.assert_allclose(scaler.inverse_transform(expected), labels, atol=3e-4)
    np.testing.assert_array_equal(scaler.mean_, [[[3000]]])
    np.testing.assert_array_equal(scaler.scale_, [[[1500]]])

    vec = np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32)
    vec_scaler = MinMaxTargetScaler([0.0, 10.0], (2,))
    np.testing.assert_allclose(
        vec_scaler.inverse_transform(vec_scaler.transform(vec)), vec, atol=1e-5
    )


def test_prepare_manifest_fixed_validation_no_cache_and_channel_inference(tmp_path):
    train, _, _ = make_field_manifest(tmp_path, "train", (2, 3), value=0)
    validation, _, labels = make_field_manifest(tmp_path, "validation", (4,), value=100)
    args = args_for(
        train,
        val_data_path=str(validation),
        input_transform="signed_log_minmax",
        input_range=[-25.0, 50.0],
        target_scaling="minmax",
        target_range=[1500.0, 4500.0],
    )
    train_dl, val_dl, scaler, in_shape, out_dim = prepare_data(
        args,
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(train_dl.dataset) == 5
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
    path, _, _ = make_field_manifest(tmp_path)
    args = args_for(path, target_scaling="minmax", target_range=[1500.0, 4500.0])
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


def test_reject_unsupported_input_channels_before_construction():
    scaler = MinMaxTargetScaler([0, 1], (1, 16, 16))
    with pytest.raises(ValueError, match="in_channels=3"):
        _build_model_for_targets("unet_field", (16, 16), 1, scaler, in_channels=3)


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


def test_checkpoint_metadata_restored_and_inference_applies_transform(
    tmp_path, monkeypatch
):
    config = preprocessing_from_args(
        Namespace(
            input_transform="signed_log_minmax",
            input_range=[-25.0, 50.0],
            target_scaling="minmax",
            target_range=[1500.0, 4500.0],
        )
    )
    model = TinyMultiChannelDense()
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
    import json

    checkpoint = tmp_path / "best_checkpoint"
    with (checkpoint / "training_meta.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata["in_channels"] == 5
    assert metadata["preprocessing"] == config
    assert not (checkpoint / "training_meta.pkl").exists()
    assert (checkpoint / "scaler.json").is_file()
    monkeypatch.setattr(inference, "get_model", lambda name: TinyMultiChannelDense)
    monkeypatch.setattr(
        inference,
        "build_model",
        lambda name, **kwargs: TinyMultiChannelDense(**kwargs),
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
    model = TinyMultiChannelDense().eval()
    model._wavedl_preprocessing = preprocessing_from_args(
        Namespace(
            input_transform="signed_log_minmax",
            input_range=[-25.0, 50.0],
            target_scaling="minmax",
            target_range=[1500.0, 4500.0],
        )
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
    assert '"input_transform": "signed_log_minmax"' in metadata["preprocessing"]


def test_train_to_test_manifest_smoke(tmp_path, monkeypatch, capsys):
    from wavedl import train
    from wavedl.models.registry import MODEL_REGISTRY

    monkeypatch.setitem(MODEL_REGISTRY, "tiny_multichannel_test", TinyMultiChannelDense)
    path, _, _ = make_field_manifest(tmp_path, lengths=(8, 8))
    validation, _, _ = make_field_manifest(tmp_path, "val", (3,), value=20)
    output = tmp_path / "run"
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-train",
            "--model",
            "tiny_multichannel_test",
            "--data_path",
            str(path),
            "--val_data_path",
            str(validation),
            "--input_transform",
            "signed_log_minmax",
            "--input_range",
            "-25",
            "50",
            "--target_scaling",
            "minmax",
            "--target_range",
            "1500",
            "4500",
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
    assert "OVERALL FIELD TEST RESULTS" in report


def test_manifest_explicit_channel_mismatch(tmp_path):
    path, _, _ = make_field_manifest(tmp_path)
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
    path, x, y = make_field_manifest(tmp_path)
    validation = tmp_path / "val.npz"
    np.savez(validation, input_train=x[:2], output_train=y[:2])
    train_dl, val_dl, _, _, _ = prepare_data(
        args_for(
            path,
            val_data_path=str(validation),
            target_scaling="minmax",
            target_range=[1500.0, 4500.0],
        ),
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(train_dl.dataset) == 5
    assert len(val_dl.dataset) == 2
    assert val_dl.dataset.in_channels == 5


def test_fixed_validation_one_channel_manifest_with_4d_npz(tmp_path):
    path, x, y = make_field_manifest(tmp_path, name="single_ch", channels=1)
    val_npz = tmp_path / "val_4d.npz"
    np.savez(val_npz, input_train=x[:2], output_train=y[:2])
    train_dl, val_dl, _, in_shape, _ = prepare_data(
        args_for(path, val_data_path=str(val_npz)),
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
    )
    assert in_shape == (8, 6)
    assert train_dl.dataset.in_channels == 1
    assert val_dl.dataset.in_channels == 1
    inputs, _ = next(iter(val_dl))
    assert inputs.shape == (2, 1, 8, 6)


def test_fixed_validation_rejects_sample_mismatch_and_nonfinite_inputs(tmp_path):
    train_npz = tmp_path / "train.npz"
    bad_len_val = tmp_path / "bad_len_val.npz"
    nan_train = tmp_path / "nan_train.npz"
    np.savez(
        train_npz,
        input_train=np.ones((4, 8, 6), dtype=np.float32),
        output_train=np.ones((4, 1), dtype=np.float32),
    )
    np.savez(
        bad_len_val,
        input_train=np.ones((2, 8, 6), dtype=np.float32),
        output_train=np.ones((5, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="mismatch"):
        prepare_data(
            args_for(train_npz, val_data_path=str(bad_len_val)),
            logging.getLogger(__name__),
            CPUAccelerator(),
            cache_dir=str(tmp_path / "cache1"),
        )
    bad_x = np.ones((4, 8, 6), dtype=np.float32)
    bad_x[0, 0, 0] = np.nan
    np.savez(nan_train, input_train=bad_x, output_train=np.ones((4, 1), np.float32))
    train_dl, _, _, _, _ = prepare_data(
        args_for(nan_train, val_data_path=str(train_npz)),
        logging.getLogger(__name__),
        CPUAccelerator(),
        cache_dir=str(tmp_path / "cache2"),
    )
    with pytest.raises(ValueError, match="finite real"):
        _ = train_dl.dataset[0]


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


def test_manifest_against_3d_checkpoint_raises_value_error(tmp_path, monkeypatch):
    path, _, _ = make_field_manifest(tmp_path, channels=1)
    ckpt = tmp_path / "ckpt3d"
    ckpt.mkdir()
    with (ckpt / "training_meta.pkl").open("wb") as file:
        pickle.dump(
            {
                "model_name": "resnet3d_18",
                "in_shape": (8, 8, 6),
                "in_channels": 1,
                "out_dim": 1,
                "target_shape": (1,),
            },
            file,
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["wavedl-test", "--checkpoint", str(ckpt), "--data_path", str(path)],
    )
    with pytest.raises(ValueError, match="does not match checkpoint in_shape"):
        inference.main()


def test_channels_last_detection_preserved_for_2d_single_channel_checkpoint(
    tmp_path, monkeypatch
):
    ckpt = tmp_path / "ckpt2d"
    ckpt.mkdir()
    with (ckpt / "training_meta.pkl").open("wb") as file:
        pickle.dump(
            {
                "model_name": "cnn",
                "in_shape": (32, 32),
                "in_channels": 1,
                "out_dim": 1,
                "target_shape": (1,),
            },
            file,
        )
    channels_last = tmp_path / "channels_last.npz"
    np.savez(
        channels_last,
        input_test=np.ones((2, 32, 32, 3), dtype=np.float32),
        output_test=np.ones((2, 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["wavedl-test", "--checkpoint", str(ckpt), "--data_path", str(channels_last)],
    )
    with pytest.raises(ValueError, match="channels-last"):
        inference.main()


def test_memmap_worker_init_clears_manifest_handles(tmp_path, monkeypatch):
    path, _, _ = make_field_manifest(tmp_path)
    source = ShardManifestDataset(path)
    _ = source[0]
    assert source._maps
    inputs = ManifestInputs(source)
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: Namespace(dataset=inputs, seed=12345),
    )
    memmap_worker_init_fn(0)
    assert len(source._maps) == 0
    # Subsequent reads in the worker must continue using OrderedDict LRU methods
    _ = source[0]
    _ = source[1]
    assert len(source._maps) > 0


def test_validate_config_accepts_preprocessing_keys_without_parser():
    warnings = validate_config(
        {
            "model": "unet_field",
            "data_path": "train.txt",
            "val_data_path": "val.txt",
            "input_channels": 5,
            "input_transform": "signed_log_minmax",
            "input_range": [-25.0, 50.0],
            "target_scaling": "minmax",
            "target_range": [1500.0, 4500.0],
            "precision": "no",
            "no_tf32": True,
        }
    )
    assert warnings == []


def test_configure_tf32_does_not_mix_legacy_and_new_matmul_apis():
    prev_prec = torch.get_float32_matmul_precision()
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        configure_tf32(Namespace(no_tf32=True, deterministic=False))
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.set_float32_matmul_precision(prev_prec)
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def test_minmax_target_scaler_preserves_float64_precision_for_large_offsets():
    scaler = MinMaxTargetScaler([1e9, 1e9 + 100.0], (1,))
    raw = np.array(
        [[1e9 + 25.0], [1e9 + 75.0], [1e9 + 50.0], [1e9 + 10.0]], dtype=np.float64
    )
    scaled = scaler.transform(raw)
    expected = np.array([[-0.5], [0.5], [0.0], [-0.8]], dtype=np.float32)
    np.testing.assert_allclose(scaled, expected, atol=1e-6)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, raw, rtol=0, atol=1e-4)


def test_inverse_transform_inputs_roundtrips_signed_log_minmax():
    cfg = preprocessing_from_args(
        Namespace(
            input_transform="signed_log_minmax",
            input_range=[-25.0, 50.0],
            target_scaling="standard",
            target_range=None,
        )
    )
    arr = np.array([-20.0, -1.5, 0.0, 3.25, 45.0], dtype=np.float32)
    transformed_np = transform_inputs(arr, cfg)
    recovered_np = inverse_transform_inputs(transformed_np, cfg)
    np.testing.assert_allclose(recovered_np, arr, rtol=1e-5, atol=1e-5)

    tensor = torch.from_numpy(arr)
    transformed_pt = transform_inputs(tensor, cfg)
    recovered_pt = inverse_transform_inputs(transformed_pt, cfg)
    torch.testing.assert_close(recovered_pt, tensor, rtol=1e-5, atol=1e-5)


def test_shard_manifest_lru_bounds_open_memmaps(tmp_path):
    path, _, _ = make_field_manifest(tmp_path, lengths=(2, 2, 2, 2, 2))
    ds = ShardManifestDataset(path, max_open_shards=2)
    try:
        for idx in range(len(ds)):
            _ = ds[idx]
            assert len(ds._maps) <= 2
    finally:
        ds.close()
        assert len(ds._maps) == 0


def test_prepare_data_does_not_reuse_stale_preprocessed_cache(tmp_path):
    from wavedl.utils.targets import SpatialTargetScaler

    data_file = tmp_path / "field.npz"
    x = np.random.default_rng(0).standard_normal((8, 12, 12)).astype(np.float32)
    y = (
        np.random.default_rng(1)
        .uniform(1500.0, 4500.0, size=(8, 12, 12))
        .astype(np.float32)
    )
    np.savez(data_file, input_train=x, output_train=y)

    cache_dir = tmp_path / "shared_cache"
    logger = logging.getLogger(__name__)
    acc = CPUAccelerator()

    # Step 1: Populate cache_dir via standard prepare_data (creates train_data_cache.dat)
    base_args = Namespace(
        data_path=str(data_file),
        batch_size=4,
        workers=0,
        seed=42,
        cache_validate="sha256",
        single_channel=False,
        allow_pickle=True,
    )
    prepare_data(base_args, logger, acc, cache_dir=str(cache_dir), val_split=0.25)

    # Step 2: Run prepare_preprocessed_data in same cache_dir (overwrites data_metadata.pkl + scaler.pkl)
    prep_cfg = preprocessing_from_args(
        Namespace(
            input_transform="none",
            input_range=None,
            target_scaling="minmax",
            target_range=[1500.0, 4500.0],
        )
    )
    prep_args = Namespace(
        data_path=str(data_file),
        val_data_path=None,
        batch_size=4,
        workers=0,
        seed=42,
        single_channel=False,
        input_channels=None,
        allow_pickle=True,
    )
    prepare_preprocessed_data(
        prep_args, prep_cfg, logger, acc, str(cache_dir), val_split=0.25
    )

    # Step 3: Subsequent default prepare_data must invalidate and rebuild with SpatialTargetScaler
    _, _, rebuilt_scaler, _, _ = prepare_data(
        base_args, logger, acc, cache_dir=str(cache_dir), val_split=0.25
    )
    assert isinstance(rebuilt_scaler, SpatialTargetScaler)


def test_3d_single_channel_manifest_with_npz_val_data_path(tmp_path):
    # Create 1-channel 3D shard manifest: (N, 1, 8, 10, 10) -> (N, 2)
    x_shard = tmp_path / "x0.npy"
    y_shard = tmp_path / "y0.npy"
    np.save(x_shard, np.ones((4, 1, 8, 10, 10), dtype=np.float32))
    np.save(y_shard, np.ones((4, 2), dtype=np.float32))
    manifest = tmp_path / "train.txt"
    manifest.write_text(f"{x_shard.name}\t{y_shard.name}\n", encoding="utf-8")

    # Shallow 3D validation array (M, 8, 10, 10) without explicit channel axis
    val_npz = tmp_path / "val.npz"
    np.savez(
        val_npz,
        input_train=np.ones((2, 8, 10, 10), dtype=np.float32),
        output_train=np.ones((2, 2), dtype=np.float32),
    )

    args = Namespace(
        data_path=str(manifest),
        val_data_path=str(val_npz),
        batch_size=2,
        workers=0,
        seed=42,
        single_channel=True,
        input_channels=None,
        allow_pickle=False,
    )
    cfg = preprocessing_from_args(args)
    train_dl, val_dl, _, in_shape, out_dim = prepare_preprocessed_data(
        args,
        cfg,
        logging.getLogger(__name__),
        CPUAccelerator(),
        str(tmp_path / "c"),
        0.2,
    )
    assert in_shape == (8, 10, 10)
    assert out_dim == 2
    assert len(train_dl.dataset) == 4
    assert len(val_dl.dataset) == 2


def test_prepare_preprocessed_data_rejects_empty_training_set_with_val_path(tmp_path):
    empty_train = tmp_path / "empty.npz"
    val_npz = tmp_path / "val.npz"
    np.savez(
        empty_train,
        input_train=np.ones((0, 2, 8, 8), dtype=np.float32),
        output_train=np.ones((0, 8, 8), dtype=np.float32),
    )
    np.savez(
        val_npz,
        input_train=np.ones((2, 2, 8, 8), dtype=np.float32),
        output_train=np.ones((2, 8, 8), dtype=np.float32),
    )
    args = Namespace(
        data_path=str(empty_train),
        val_data_path=str(val_npz),
        batch_size=2,
        workers=0,
        seed=42,
        single_channel=False,
        input_channels=None,
        allow_pickle=False,
        input_transform="none",
        input_range=None,
        target_scaling="minmax",
        target_range=[-1.0, 1.0],
    )
    cfg = preprocessing_from_args(args)
    with pytest.raises(ValueError, match="Training dataset is empty"):
        prepare_preprocessed_data(
            args,
            cfg,
            logging.getLogger(__name__),
            CPUAccelerator(),
            str(tmp_path / "c"),
            0.2,
        )


def test_3d_multichannel_inference_preserves_checkpoint_channel_count(
    tmp_path, monkeypatch
):
    # Create 3-channel 3D manifest: (N, 3, 6, 8, 8)
    x_shard = tmp_path / "x3d.npy"
    y_shard = tmp_path / "y3d.npy"
    np.save(x_shard, np.ones((2, 3, 6, 8, 8), dtype=np.float32))
    np.save(y_shard, np.ones((2, 2), dtype=np.float32))
    manifest = tmp_path / "test3d.txt"
    manifest.write_text(f"{x_shard.name}\t{y_shard.name}\n", encoding="utf-8")

    ckpt = tmp_path / "ckpt3d"
    ckpt.mkdir()
    with (ckpt / "training_meta.pkl").open("wb") as file:
        pickle.dump(
            {
                "model_name": "cnn",
                "in_shape": (6, 8, 8),
                "in_channels": 3,
                "out_dim": 2,
                "target_shape": (2,),
            },
            file,
        )

    captured = {}

    def fake_load_data_for_inference(*args, **kwargs):
        captured["input_channels"] = kwargs.get("input_channels")
        raise RuntimeError("sentinel_after_load")

    monkeypatch.setattr(
        inference, "load_data_for_inference", fake_load_data_for_inference
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["wavedl-test", "--checkpoint", str(ckpt), "--data_path", str(manifest)],
    )
    with pytest.raises(RuntimeError, match="sentinel_after_load"):
        inference.main()
    assert captured["input_channels"] == 3


def test_onnx_metadata_reports_input_transform_embedded_false_when_none(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = TinyMultiChannelDense().eval()
    model._wavedl_preprocessing = {"input_transform": "none"}
    raw = torch.randn(1, 5, 8, 6)
    path = tmp_path / "plain.onnx"
    assert inference.export_to_onnx(model, raw, str(path))
    metadata = {entry.key: entry.value for entry in onnx.load(path).metadata_props}
    assert metadata["input_transform_embedded"] == "false"


def test_mat_source_preserves_file_not_found_error(tmp_path):
    from wavedl.utils.data import MATSource

    missing = str(tmp_path / "nonexistent.mat")
    src = MATSource()
    with pytest.raises(FileNotFoundError):
        src.load(missing)
    with pytest.raises(FileNotFoundError):
        src.load_mmap(missing)
    with pytest.raises(FileNotFoundError):
        src.load_outputs_only(missing)


def test_launcher_honors_nested_training_precision_no(tmp_path, monkeypatch):
    from wavedl import launcher

    cfg = tmp_path / "nested.yaml"
    cfg.write_text("model: cnn\ntraining:\n  precision: no\n", encoding="utf-8")
    captured = {}

    class DummyProc:
        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher, "detect_gpus", lambda: 1)
    monkeypatch.setattr(sys, "argv", ["wavedl-train", "--config", str(cfg)])
    assert launcher.main() == 0
    assert "--mixed_precision=no" in captured["cmd"]
    prec_idx = captured["cmd"].index("--precision")
    assert captured["cmd"][prec_idx + 1] == "no"


def test_hpo_detect_target_shape_supports_shard_manifest(tmp_path):
    from wavedl.hpo import _detect_target_shape

    path, _, _ = make_field_manifest(tmp_path, lengths=(2,))
    assert _detect_target_shape(str(path)) == (1, 4, 6)


def test_1d_multichannel_array_dataset_and_load_test_data(tmp_path):
    from wavedl.utils.data import load_test_data
    from wavedl.utils.preprocessing import _ArrayDataset

    x = np.random.default_rng(3).normal(size=(4, 3, 32)).astype(np.float32)
    y = np.random.default_rng(4).normal(size=(4, 2)).astype(np.float32)
    path = tmp_path / "mc1d.npz"
    np.savez(path, input_test=x, output_test=y)

    x_loaded, y_loaded = load_test_data(str(path), input_channels=3)
    assert x_loaded.shape == (4, 3, 32)
    assert y_loaded.shape == (4, 2)

    from wavedl.utils.targets import create_target_scaler

    ds = _ArrayDataset(
        x,
        y,
        np.arange(len(x)),
        {"input_transform": "none"},
        create_target_scaler(y),
        input_channels=3,
    )
    sample_x, sample_y = ds[0]
    assert sample_x.shape == (3, 32)
    assert sample_y.shape == (2,)


def test_shard_manifest_preserves_float64_targets_and_manifest_inputs_context(tmp_path):
    from wavedl.utils.preprocessing import ManifestInputs, ShardManifestDataset

    x_shard = tmp_path / "x.npy"
    y_shard = tmp_path / "y.npy"
    y_f64 = 1e8 + np.arange(4, dtype=np.float64).reshape(4, 1)
    np.save(x_shard, np.ones((4, 1, 8, 8), dtype=np.float32))
    np.save(y_shard, y_f64)
    manifest = tmp_path / "m.txt"
    manifest.write_text(f"{x_shard.name}\t{y_shard.name}\n", encoding="utf-8")

    ds = ShardManifestDataset(str(manifest))
    try:
        batches = list(ds.target_batches(batch_size=2))
        stacked = np.concatenate(batches, axis=0)
        assert stacked.dtype == np.float64
        np.testing.assert_array_equal(stacked, y_f64)
    finally:
        ds.close()

    with ManifestInputs(ShardManifestDataset(str(manifest))) as inputs:
        assert inputs.shape == (4, 1, 8, 8)


def test_prepare_data_enables_drop_last_for_singleton_tail_batch(tmp_path):
    from types import SimpleNamespace

    from wavedl.utils.data import prepare_data

    x = np.ones((10, 8, 8), dtype=np.float32)
    y = np.arange(20, dtype=np.float32).reshape(10, 2)
    path = tmp_path / "ten.npz"
    np.savez(path, input_train=x, output_train=y)

    train_dl, _, _, _, _ = prepare_data(
        args=SimpleNamespace(
            data_path=str(path),
            batch_size=7,  # 8 train samples % 7 == 1 -> must drop_last
            workers=0,
            seed=42,
            deterministic=False,
            single_channel=False,
            allow_pickle=False,
            cache_validate="sha256",
        ),
        logger=logging.getLogger("test_drop_last"),
        accelerator=CPUAccelerator(),
        cache_dir=str(tmp_path / "cache"),
        val_split=0.2,
    )
    assert train_dl.drop_last is True


def test_transform_inputs_rejects_nonpositive_log_k_and_degenerate_range():
    from wavedl.utils.preprocessing import transform_inputs

    arr = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="log_k"):
        transform_inputs(
            arr,
            {
                "input_transform": "signed_log_minmax",
                "input_range": (0.0, 1.0),
                "log_k": 0.0,
            },
        )
    with pytest.raises(ValueError, match=r"input_range|lo < hi|strictly"):
        transform_inputs(
            arr,
            {"input_transform": "signed_log_minmax", "input_range": (1.0, 1.0)},
        )


class _ExactNormalizedVectorModel(BaseModel):
    """Returns exact normalized target encoded in input channel 0, pixel 0."""

    supports_spatial_output = False
    requires_spatial_output = False
    supported_input_channels = None

    def __init__(self, in_shape=(4, 4), out_size=1, in_channels=1, **kwargs):
        super().__init__(in_shape, out_size)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, 0, : self.out_size] * self.scale


class _ExactNormalizedFieldModel(BaseModel):
    """Returns exact normalized field encoded directly in input channel 0."""

    supports_spatial_output = True
    requires_spatial_output = True
    supported_input_channels = None

    def __init__(
        self, in_shape=(4, 4), out_size=1, in_channels=1, out_shape=(1, 4, 4), **kwargs
    ):
        super().__init__(in_shape, out_size)
        self.out_shape = tuple(out_shape)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x[:, : self.out_shape[0], : self.out_shape[1], : self.out_shape[2]]
            * self.scale
        )


@pytest.mark.parametrize("source_kind", ["npz", "manifest"])
@pytest.mark.parametrize("target_dtype", ["=f8", ">f8"])
def test_inference_preserves_float64_large_offset_targets_end_to_end(
    tmp_path, monkeypatch, capsys, source_kind, target_dtype
):
    """Loader through wavedl-test scoring and CSV export must preserve float64 precision."""
    import csv

    from wavedl.models.registry import MODEL_REGISTRY
    from wavedl.utils.checkpoint_io import save_scaler, save_training_meta
    from wavedl.utils.targets import create_target_scaler

    monkeypatch.setitem(MODEL_REGISTRY, "exact_vec_test", _ExactNormalizedVectorModel)

    # 12 distinct values around 1e8 collapse to <=2 values under float32 quantization
    y_true_f64 = (1e8 + np.arange(12, dtype=np.float64)).reshape(12, 1)
    assert len(np.unique(y_true_f64.astype(np.float32))) <= 2

    scaler = create_target_scaler(y_true_f64)
    y_norm_f32 = scaler.transform(y_true_f64).astype(np.float32)

    x_data = np.zeros((12, 1, 4, 4), dtype=np.float32)
    x_data[:, 0, 0, 0] = y_norm_f32[:, 0]
    stored_targets = y_true_f64.astype(target_dtype)

    if source_kind == "npz":
        data_path = tmp_path / "large_offset.npz"
        np.savez(data_path, input_test=x_data[:, 0], output_test=stored_targets)
    else:
        np.save(tmp_path / "x0.npy", x_data[:6])
        np.save(tmp_path / "y0.npy", stored_targets[:6])
        np.save(tmp_path / "x1.npy", x_data[6:])
        np.save(tmp_path / "y1.npy", stored_targets[6:])
        data_path = tmp_path / "test.txt"
        data_path.write_text("x0.npy\ty0.npy\nx1.npy\ty1.npy\n", encoding="utf-8")

    loaded_x, loaded_y = load_test_data(str(data_path))
    try:
        assert loaded_y is not None
        assert loaded_y.dtype == torch.float64
        assert len(np.unique(loaded_y.numpy())) == 12
        np.testing.assert_array_equal(loaded_y.numpy(), y_true_f64)
    finally:
        if hasattr(loaded_x, "close"):
            loaded_x.close()

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    model = _ExactNormalizedVectorModel(in_shape=(4, 4), out_size=1, in_channels=1)
    torch.save(model.state_dict(), ckpt_dir / "model.bin")
    save_scaler(scaler, ckpt_dir)
    save_training_meta(
        {
            "epoch": 1,
            "best_val_loss": 0.0,
            "model_name": "exact_vec_test",
            "in_shape": (4, 4),
            "in_channels": 1,
            "out_dim": 1,
            "target_shape": (1,),
        },
        ckpt_dir,
    )

    out_dir = tmp_path / f"results_{source_kind}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-test",
            "--checkpoint",
            str(ckpt_dir),
            "--data_path",
            str(data_path),
            "--batch_size",
            "4",
            "--output_dir",
            str(out_dir),
            "--save_predictions",
        ],
    )
    inference.main()
    stdout = capsys.readouterr().out
    assert "R² Score:         1.000000" in stdout
    rmse_line = next(line for line in stdout.splitlines() if line.startswith("RMSE:"))
    assert float(rmse_line.split(":", 1)[1].strip()) < 1e-4

    with (out_dir / "predictions.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    saved_true = np.array([float(r["True_P0"]) for r in rows], dtype=np.float64)
    saved_pred = np.array([float(r["Pred_P0"]) for r in rows], dtype=np.float64)
    assert len(np.unique(saved_true)) == 12
    np.testing.assert_array_equal(saved_true, y_true_f64[:, 0])
    np.testing.assert_allclose(saved_pred, y_true_f64[:, 0], rtol=0, atol=1e-4)


def test_dense_field_inference_preserves_float64_large_offset(
    tmp_path, monkeypatch, capsys
):
    """Dense 2D field evaluation on large-offset float64 targets scores near-zero RMSE."""
    from wavedl.models.registry import MODEL_REGISTRY
    from wavedl.utils.checkpoint_io import save_scaler, save_training_meta
    from wavedl.utils.targets import SpatialTargetScaler

    monkeypatch.setitem(MODEL_REGISTRY, "exact_field_test", _ExactNormalizedFieldModel)

    y_true_f64 = (1e8 + np.arange(8 * 16, dtype=np.float64)).reshape(8, 1, 4, 4)
    scaler = SpatialTargetScaler().fit(y_true_f64)
    y_norm_f32 = scaler.transform(y_true_f64).astype(np.float32)

    data_path = tmp_path / "dense_large.npz"
    np.savez(data_path, input_test=y_norm_f32[:, 0], output_test=y_true_f64)

    ckpt_dir = tmp_path / "ckpt_field"
    ckpt_dir.mkdir()
    model = _ExactNormalizedFieldModel(
        in_shape=(4, 4), out_size=1, in_channels=1, out_shape=(1, 4, 4)
    )
    torch.save(model.state_dict(), ckpt_dir / "model.bin")
    save_scaler(scaler, ckpt_dir)
    save_training_meta(
        {
            "epoch": 1,
            "best_val_loss": 0.0,
            "model_name": "exact_field_test",
            "in_shape": (4, 4),
            "in_channels": 1,
            "out_dim": 1,
            "target_shape": (1, 4, 4),
        },
        ckpt_dir,
    )

    out_dir = tmp_path / "field_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-test",
            "--checkpoint",
            str(ckpt_dir),
            "--data_path",
            str(data_path),
            "--batch_size",
            "4",
            "--output_dir",
            str(out_dir),
            "--save_predictions",
        ],
    )
    inference.main()
    stdout = capsys.readouterr().out
    assert "R² Score:         1.000000" in stdout
    rmse_line = next(line for line in stdout.splitlines() if line.startswith("RMSE:"))
    assert float(rmse_line.split(":", 1)[1].strip()) < 1e-4

    with np.load(out_dir / "predictions.npz") as saved:
        assert saved["target"].dtype == np.float64
        assert saved["prediction"].dtype == np.float64
        np.testing.assert_allclose(saved["prediction"], y_true_f64, rtol=0, atol=1e-4)


def test_export_only_skips_manifest_target_loading(tmp_path, monkeypatch):
    """wavedl-test --export onnx (export-only) must never load manifest targets."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from wavedl.models.registry import MODEL_REGISTRY
    from wavedl.utils.checkpoint_io import save_scaler, save_training_meta
    from wavedl.utils.targets import SpatialTargetScaler

    monkeypatch.setitem(
        MODEL_REGISTRY, "exact_field_export", _ExactNormalizedFieldModel
    )

    np.save(tmp_path / "x0.npy", np.ones((4, 1, 4, 4), dtype=np.float32))
    np.save(tmp_path / "y0.npy", np.ones((4, 1, 4, 4), dtype=np.float32))
    manifest = tmp_path / "test.txt"
    manifest.write_text("x0.npy\ty0.npy\n", encoding="utf-8")

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    model = _ExactNormalizedFieldModel(
        in_shape=(4, 4), out_size=1, in_channels=1, out_shape=(1, 4, 4)
    )
    scaler = SpatialTargetScaler().fit(
        np.arange(64, dtype=np.float32).reshape(4, 1, 4, 4)
    )
    torch.save(model.state_dict(), ckpt_dir / "model.bin")
    save_scaler(scaler, ckpt_dir)
    save_training_meta(
        {
            "epoch": 1,
            "best_val_loss": 0.0,
            "model_name": "exact_field_export",
            "in_shape": (4, 4),
            "in_channels": 1,
            "out_dim": 1,
            "target_shape": (1, 4, 4),
        },
        ckpt_dir,
    )

    monkeypatch.setattr(
        ShardManifestDataset,
        "target_batches",
        lambda *a, **k: pytest.fail("export-only must not read manifest targets"),
    )

    onnx_out = tmp_path / "out" / "exported.onnx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-test",
            "--checkpoint",
            str(ckpt_dir),
            "--data_path",
            str(manifest),
            "--export",
            "onnx",
            "--export_path",
            str(onnx_out),
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )
    inference.main()
    assert onnx_out.is_file()


def test_manifest_evaluation_streams_and_saves_without_materializing_all_arrays(
    tmp_path, monkeypatch, capsys
):
    """Manifest evaluation and saving should stream through the CLI."""
    from wavedl.models.registry import MODEL_REGISTRY
    from wavedl.utils.checkpoint_io import save_scaler, save_training_meta
    from wavedl.utils.preprocessing import ManifestTargets
    from wavedl.utils.targets import SpatialTargetScaler

    monkeypatch.setitem(
        MODEL_REGISTRY, "exact_field_stream", _ExactNormalizedFieldModel
    )

    rng = np.random.default_rng(23)
    y_all = (1500.0 + rng.normal(scale=200.0, size=(6, 1, 4, 4))).astype(np.float32)
    scaler = SpatialTargetScaler().fit(y_all)
    x_all = scaler.transform(y_all).astype(np.float32)

    np.save(tmp_path / "x0.npy", x_all[:3])
    np.save(tmp_path / "y0.npy", y_all[:3])
    np.save(tmp_path / "x1.npy", x_all[3:])
    np.save(tmp_path / "y1.npy", y_all[3:].astype(">f8"))
    manifest = tmp_path / "stream.txt"
    manifest.write_text("x0.npy\ty0.npy\nx1.npy\ty1.npy\n", encoding="utf-8")

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    model = _ExactNormalizedFieldModel(
        in_shape=(4, 4), out_size=1, in_channels=1, out_shape=(1, 4, 4)
    )
    torch.save(model.state_dict(), ckpt_dir / "model.bin")
    save_scaler(scaler, ckpt_dir)
    save_training_meta(
        {
            "epoch": 1,
            "best_val_loss": 0.0,
            "model_name": "exact_field_stream",
            "in_shape": (4, 4),
            "in_channels": 1,
            "out_dim": 1,
            "target_shape": (1, 4, 4),
        },
        ckpt_dir,
    )

    monkeypatch.setattr(
        inference,
        "run_inference",
        lambda *a, **k: pytest.fail(
            "manifest CLI evaluation must stream, not call run_inference"
        ),
    )
    monkeypatch.setattr(
        ManifestTargets,
        "materialize",
        lambda *a, **k: pytest.fail(
            "manifest CLI evaluation must not materialize all targets"
        ),
    )
    monkeypatch.setattr(
        inference,
        "save_predictions",
        lambda *a, **k: pytest.fail("dense manifest saving must stream its output"),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-test",
            "--checkpoint",
            str(ckpt_dir),
            "--data_path",
            str(manifest),
            "--batch_size",
            "2",
            "--save_predictions",
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )
    inference.main()
    out = capsys.readouterr().out
    assert "Samples:          6" in out
    assert "R² Score:         1.000000" in out
    with np.load(tmp_path / "out/predictions.npz", allow_pickle=False) as saved:
        assert saved["target"].dtype == np.float64
        assert saved["prediction"].dtype == np.float64
        np.testing.assert_array_equal(saved["target"], y_all)
        np.testing.assert_allclose(saved["prediction"], y_all, atol=1e-4)
        np.testing.assert_array_equal(saved["error"], saved["prediction"] - y_all)
        np.testing.assert_array_equal(saved["channel_names"], ["C0"])


@pytest.mark.parametrize("suffix", [".h5", ".mat"])
def test_preprocessed_pipeline_streams_hdf5_and_mat_v73_lazily(
    tmp_path, monkeypatch, suffix
):
    """Fixed validation / custom preprocessing on HDF5 & MAT v7.3 must not load eagerly."""
    import h5py

    import wavedl.utils.data as data_mod
    from wavedl.utils.preprocessing import _LazyH5MatDataset

    rng = np.random.default_rng(31)
    x_tr = rng.normal(size=(6, 2, 8, 8)).astype(np.float32)
    y_tr = rng.uniform(1500.0, 4500.0, size=(6, 1, 4, 4)).astype(np.float32)
    x_va = rng.normal(size=(3, 2, 8, 8)).astype(np.float32)
    y_va = rng.uniform(1500.0, 4500.0, size=(3, 1, 4, 4)).astype(np.float32)

    train_path = tmp_path / f"train{suffix}"
    val_path = tmp_path / f"val{suffix}"
    for p, x_arr, y_arr in ((train_path, x_tr, y_tr), (val_path, x_va, y_va)):
        with h5py.File(p, "w") as f:
            f.create_dataset("input_train", data=x_arr.T if suffix == ".mat" else x_arr)
            f.create_dataset(
                "output_train", data=y_arr.T if suffix == ".mat" else y_arr
            )

    monkeypatch.setattr(
        data_mod,
        "load_training_data",
        lambda *a, **k: pytest.fail("HDF5/MAT v7.3 must stream lazily, not eager load"),
    )

    args = Namespace(
        data_path=str(train_path),
        val_data_path=str(val_path),
        batch_size=2,
        workers=0,
        seed=42,
        single_channel=False,
        input_channels=2,
        allow_pickle=False,
        input_transform="signed_log_minmax",
        input_range=[-10.0, 10.0],
        target_scaling="minmax",
        target_range=[1500.0, 4500.0],
    )
    cfg = preprocessing_from_args(args)
    train_dl, val_dl, scaler, in_shape, out_dim = prepare_preprocessed_data(
        args,
        cfg,
        logging.getLogger(__name__),
        CPUAccelerator(),
        str(tmp_path / "cache"),
        0.2,
    )
    try:
        assert isinstance(train_dl.dataset, _LazyH5MatDataset)
        assert isinstance(val_dl.dataset, _LazyH5MatDataset)
        assert in_shape == (8, 8)
        assert out_dim == 1
        assert args.in_channels == 2
        bx, by = next(iter(val_dl))
        assert bx.shape == (2, 2, 8, 8)
        assert by.shape == (2, 1, 4, 4)
        np.testing.assert_allclose(by.numpy(), scaler.transform(y_va[:2]), atol=1e-6)
        restored = pickle.loads(pickle.dumps(train_dl.dataset))
        assert restored._handle is None
        rx, _ = restored[0]
        assert rx.shape == (2, 8, 8)
        restored.close()
    finally:
        train_dl.dataset.close()
        val_dl.dataset.close()


def test_safe_json_checkpoint_inference_never_invokes_pickle(tmp_path, monkeypatch):
    """JSON checkpoints load and run inference without calling pickle.load()."""
    from wavedl.models.registry import MODEL_REGISTRY
    from wavedl.utils.checkpoint_io import (
        load_scaler,
        load_training_meta,
        save_scaler,
        save_training_meta,
    )
    from wavedl.utils.targets import SpatialTargetScaler, create_target_scaler

    monkeypatch.setitem(MODEL_REGISTRY, "exact_json_test", _ExactNormalizedVectorModel)

    y_vals = (1e8 + np.arange(8, dtype=np.float64)).reshape(8, 1)
    scaler = create_target_scaler(y_vals)
    ckpt = tmp_path / "ckpt_json"
    ckpt.mkdir()
    model = _ExactNormalizedVectorModel(in_shape=(4, 4), out_size=1, in_channels=1)
    torch.save(model.state_dict(), ckpt / "model.bin")
    save_scaler(scaler, ckpt)
    save_training_meta(
        {
            "epoch": 3,
            "best_val_loss": float("inf"),
            "model_name": "exact_json_test",
            "in_shape": (4, 4),
            "in_channels": 1,
            "out_dim": 1,
            "target_shape": (1,),
        },
        ckpt,
    )
    assert (ckpt / "training_meta.json").is_file()
    assert (ckpt / "scaler.json").is_file()
    assert not (ckpt / "training_meta.pkl").exists()
    assert not (ckpt / "scaler.pkl").exists()

    for sc_dir, sc in (
        (tmp_path / "sc_std", scaler),
        (
            tmp_path / "sc_sp",
            SpatialTargetScaler().fit(
                np.arange(32, dtype=np.float64).reshape(2, 1, 4, 4) + 1e9
            ),
        ),
        (tmp_path / "sc_mm", MinMaxTargetScaler([1500.0, 4500.0], (1, 4, 4))),
    ):
        save_scaler(sc, sc_dir)
        loaded_sc, loaded_path = load_scaler(sc_dir)
        assert loaded_path.name == "scaler.json"
        np.testing.assert_allclose(loaded_sc.mean_, sc.mean_, rtol=0, atol=0)
        np.testing.assert_allclose(loaded_sc.scale_, sc.scale_, rtol=0, atol=0)

    monkeypatch.setattr(
        pickle,
        "load",
        lambda *a, **k: pytest.fail(
            "JSON checkpoint inference must not call pickle.load"
        ),
    )
    meta = load_training_meta(ckpt)
    assert meta["best_val_loss"] == float("inf")
    loaded_model, loaded_scaler = inference.load_checkpoint(ckpt, (4, 4), 1)
    assert loaded_model is not None
    np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_, rtol=0, atol=0)
