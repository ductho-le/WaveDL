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
    expected = ((labels - 1500) / 3000 - 0.5) * 2
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
    checkpoint = tmp_path / "best_checkpoint"
    with (checkpoint / "training_meta.pkl").open("rb") as file:
        metadata = pickle.load(file)
    assert metadata["in_channels"] == 5
    assert metadata["preprocessing"] == config
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
    assert source._maps == {}


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
