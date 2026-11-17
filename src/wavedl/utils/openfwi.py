"""OpenFWI shard loading, explicit preprocessing, and benchmark metrics.

Adapted from LANL/OpenFWI (BSD-3-Clause) at commit
48754806b7b4c5877259c6b958a87f4513fcdc0b, including the bundled bounds.
SSIM reimplements pytorch-ssim by Po-Hsun (Evan) Su (MIT).
See THIRD_PARTY_NOTICES.md. Transforms use the upstream
operation order, including log-transformed input bounds and no clipping.
Reproduction requires ``--precision no --no_tf32`` during training.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import pickle
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .distributed import wait_for_everyone
from .targets import (
    SpatialTargetScaler,
    canonical_target_shape,
    canonicalize_targets,
    create_target_scaler,
)


def dataset_config() -> dict:
    """Return the pinned upstream dataset configuration (including tutorial)."""
    return json.loads(files("wavedl").joinpath("dataset_config.json").read_text())


def _range(bounds, name):
    if bounds is None or len(bounds) != 2:
        raise ValueError(f"{name} requires MIN MAX bounds.")
    low, high = map(float, bounds)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(f"{name} requires finite MIN < MAX, got {bounds}.")
    return [low, high]


def preprocessing_from_args(args) -> dict:
    """Resolve presets and validate a serializable preprocessing contract.

    Explicit settings override preset values. Without a preset, the legacy
    defaults are identity inputs and standard target scaling.
    """
    input_channels = getattr(args, "input_channels", None)
    if input_channels is not None and (
        type(input_channels) is not int or input_channels < 1
    ):
        raise ValueError("--input_channels must be a positive integer.")
    if input_channels not in (None, 1) and getattr(args, "single_channel", False):
        raise ValueError("--single_channel conflicts with --input_channels > 1.")
    name = getattr(args, "openfwi_dataset", None)
    preset = {}
    if name:
        name = name.lower().replace("_", "-")
        configs = dataset_config()
        if name not in configs:
            raise ValueError(
                f"Unknown OpenFWI dataset {name!r}; choose {list(configs)}."
            )
        preset = configs[name]
    input_transform = getattr(args, "input_transform", None) or (
        "openfwi" if preset else "none"
    )
    target_scaling = getattr(args, "target_scaling", None) or (
        "minmax" if preset else "standard"
    )
    input_range = getattr(args, "input_range", None)
    target_range = getattr(args, "target_range", None)
    if preset:
        input_range = input_range or [preset["data_min"], preset["data_max"]]
        target_range = target_range or [preset["label_min"], preset["label_max"]]
    if input_transform == "openfwi":
        input_range = _range(input_range, "--input_range")
    if target_scaling == "minmax":
        target_range = _range(target_range, "--target_range")
    return {
        "input_transform": input_transform,
        "input_range": input_range,
        "log_k": 1.0,
        "target_scaling": target_scaling,
        "target_range": target_range,
        "openfwi_dataset": name,
    }


def uses_preprocessed_pipeline(args, preprocessing: dict) -> bool:
    """Whether training needs the manifest/preprocessing data path."""
    return bool(
        str(args.data_path).lower().endswith(".txt")
        or getattr(args, "val_data_path", None)
        or preprocessing["input_transform"] != "none"
        or preprocessing["target_scaling"] != "standard"
        or getattr(args, "input_channels", None) not in (None, 1)
    )


def add_preprocessing_arguments(parser):
    """Add train-time preprocessing options (persisted for inference/export)."""
    parser.add_argument(
        "--input_transform",
        choices=["none", "openfwi"],
        default=None,
        help="Input preprocessing (default: openfwi with a dataset preset, else none)",
    )
    parser.add_argument(
        "--input_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Physical input bounds for OpenFWI preprocessing (overrides preset)",
    )
    parser.add_argument(
        "--target_scaling",
        choices=["standard", "minmax"],
        default=None,
        help="Target scaling (default: minmax with a dataset preset, else standard)",
    )
    parser.add_argument(
        "--target_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Physical target bounds for min-max scaling (overrides preset)",
    )
    parser.add_argument(
        "--openfwi_dataset", help="Pinned OpenFWI dataset preset, e.g. flatvel-a"
    )


def transform_inputs(values, preprocessing: dict | None):
    """Copy and transform NumPy arrays or tensors, never changing a source mmap."""
    config = preprocessing or {}
    if config.get("input_transform", "none") == "none":
        return (
            values.clone() if torch.is_tensor(values) else np.array(values, copy=True)
        )
    low, high = _range(config.get("input_range"), "input_range")
    k = config.get("log_k", 1.0)
    low = np.log1p(abs(k * low)) * np.sign(low)
    high = np.log1p(abs(k * high)) * np.sign(high)
    if torch.is_tensor(values):
        result = torch.log1p(torch.abs(k * values)) * torch.sign(values)
        return ((result - float(low)) / float(high - low) - 0.5) * 2
    result = np.log1p(
        np.abs(k * np.array(values, dtype=np.float32, copy=True))
    ) * np.sign(values)
    result -= low
    result /= high - low
    return (result - 0.5) * 2


class MinMaxTargetScaler:
    """Fixed physical bounds mapped to [-1,1], with affine export attributes."""

    def __init__(self, bounds, target_shape):
        self.bounds = _range(bounds, "target_range")
        self.target_shape_ = tuple(target_shape)
        self.n_features_in_ = self.target_shape_[0]
        shape = (
            (self.n_features_in_, 1, 1)
            if len(self.target_shape_) == 3
            else (self.n_features_in_,)
        )
        low, high = self.bounds
        self.mean_ = np.full(shape, (low + high) / 2, dtype=np.float64)
        self.scale_ = np.full(shape, (high - low) / 2, dtype=np.float64)
        self.var_ = self.scale_**2

    def _array(self, values):
        values = canonicalize_targets(values)
        if tuple(values.shape[1:]) != self.target_shape_:
            raise ValueError(
                f"Expected target shape {self.target_shape_}, got {values.shape[1:]}."
            )
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError("Targets must contain finite real values.")
        return values

    def transform(self, values):
        result = self._array(values).astype(np.float32, copy=True)
        low, high = self.bounds
        result -= low
        result /= high - low
        return (result - 0.5) * 2

    def inverse_transform(self, values):
        values = self._array(values)
        low, high = self.bounds
        return (values / 2 + 0.5) * (high - low) + low


def validate_input_channels(model_class, count: int, name: str):
    """Reject incompatible channel counts before constructing a model."""
    supported = getattr(model_class, "supported_input_channels", (1,))
    if supported is not None and count not in supported:
        raise ValueError(
            f"Model '{name}' does not support in_channels={count}; supported_input_channels={supported}."
        )


class OpenFWIManifestDataset(Dataset):
    """Paired .npy shards addressed by cumulative lengths, opened read-only.

    Manifest lines are ``data.npy<TAB>label.npy``. Paths are relative to the
    manifest, blank lines are ignored, and each sample is copied before any
    transform. Workers reopen their own mappings after spawn; no data cache is
    created and differently sized shards are supported.
    """

    def __init__(self, path, preprocessing=None, scaler=None, indices=None):
        self.path = Path(path).resolve()
        self.preprocessing = preprocessing
        self.scaler = scaler
        self.paths = []
        self.ends = []
        self._maps = {}
        total = 0
        self.input_shape = None
        self.target_shape = None
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            pair = line.split("\t")
            if len(pair) != 2 or not all(part.strip() for part in pair):
                raise ValueError(
                    f"{self.path}:{line_number}: expected data.npy<TAB>label.npy."
                )
            paths = tuple(
                str((self.path.parent / part.strip()).resolve()) for part in pair
            )
            x, y = (np.load(p, mmap_mode="r", allow_pickle=False) for p in paths)
            if x.ndim != 4 or not all(x.shape) or x.dtype.kind not in "biuf":
                raise ValueError(
                    f"OpenFWI inputs require non-empty (N,C,H,W) real arrays, got {x.shape}."
                )
            if y.dtype.kind not in "biuf":
                raise ValueError("OpenFWI labels must contain finite real values.")
            shape = canonical_target_shape(y.shape)
            if len(shape) != 3 or len(x) != len(y):
                raise ValueError(
                    f"Input/target shard mismatch: {x.shape} and {y.shape}."
                )
            if self.input_shape is not None and (
                self.input_shape != x.shape[1:] or self.target_shape != shape
            ):
                raise ValueError(
                    "All OpenFWI shards must have the same input and target sample shapes."
                )
            self.input_shape = tuple(x.shape[1:])
            self.target_shape = shape
            total += len(x)
            self.paths.append(paths)
            self.ends.append(total)
        if total == 0:
            raise ValueError(f"Empty OpenFWI manifest: {self.path}")
        self.indices = np.arange(total) if indices is None else np.asarray(indices)
        self.in_channels = self.input_shape[0]
        self.shape = (total, *self.input_shape)

    def __len__(self):
        return len(self.indices)

    def _sample(self, index, target=False):
        real_index = int(self.indices[index])
        shard = bisect.bisect_right(self.ends, real_index)
        offset = real_index - (self.ends[shard - 1] if shard else 0)
        key = (shard, target)
        if key not in self._maps:
            self._maps[key] = np.load(
                self.paths[shard][int(target)], mmap_mode="r", allow_pickle=False
            )
        result = np.array(self._maps[key][offset], dtype=np.float32, copy=True)
        if not np.isfinite(result).all():
            raise ValueError("OpenFWI samples must contain finite real values.")
        return result

    def target_batches(self, indices=None, batch_size=64):
        indices = range(len(self)) if indices is None else indices
        for start in range(0, len(indices), batch_size):
            yield canonicalize_targets(
                np.stack(
                    [
                        self._sample(int(i), True)
                        for i in indices[start : start + batch_size]
                    ]
                )
            )

    def __getitem__(self, index):
        x = transform_inputs(self._sample(index), self.preprocessing)
        y = canonicalize_targets(self._sample(index, True)[None])
        if self.scaler is not None:
            y = self.scaler.transform(y)
        return torch.from_numpy(x), torch.from_numpy(y[0])

    def __getstate__(self):
        return {**self.__dict__, "_maps": {}}


class ManifestInputs(Dataset):
    """Tensor-like input view for bounded-memory inference and ONNX samples."""

    def __init__(self, source):
        self.source = source
        self.shape = source.shape
        self.ndim = len(self.shape)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return torch.stack([self[i] for i in range(*index.indices(len(self)))])
        return torch.from_numpy(self.source._sample(index))


class _ArrayDataset(Dataset):
    """Opt-in preprocessing/fixed-validation adapter for existing file formats."""

    def __init__(
        self,
        x,
        y,
        indices,
        preprocessing,
        scaler,
        input_channels=None,
        single_channel=False,
    ):
        self.x, self.y = x, canonicalize_targets(y)
        self.indices, self.preprocessing, self.scaler = indices, preprocessing, scaler
        channels_first = x.ndim >= 5 or (
            x.ndim == 4
            and not single_channel
            and input_channels != 1
            and (
                (input_channels is not None and input_channels > 1) or x.shape[1] <= 16
            )
        )
        self.add_channel = not channels_first
        self.in_channels = x.shape[1] if channels_first else 1
        self.input_shape = (
            self.in_channels,
            *(x.shape[2:] if channels_first else x.shape[1:]),
        )
        self.target_shape = tuple(self.y.shape[1:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = np.array(self.x[i], dtype=np.float32, copy=True)
        if self.add_channel:
            x = x[None]
        y = self.scaler.transform(self.y[i : i + 1])[0]
        return torch.from_numpy(
            transform_inputs(x, self.preprocessing)
        ), torch.from_numpy(np.asarray(y, dtype=np.float32))


def prepare_preprocessed_data(
    args, config, logger, accelerator, cache_dir, val_split, expected_scaler=None
):
    """Prepare opt-in sources without recaching shards; Accelerate shards loaders.

    ``config`` is the resolved :func:`preprocessing_from_args` contract. The
    legacy cached path stays unchanged for default single-channel workflows.
    Fixed validation also accepts NPZ/HDF5/MAT (eager loading for these opt-in
    formats); statistics are always fitted on training targets only.
    """
    from .data import _validate_scaler_compatibility, load_training_data

    val_path = getattr(args, "val_data_path", None)
    manifest = str(args.data_path).lower().endswith(".txt")
    if manifest:
        source = OpenFWIManifestDataset(args.data_path)
        n = len(source)
        targets = None
    else:
        x, targets = load_training_data(
            args.data_path, allow_pickle=getattr(args, "allow_pickle", False)
        )
        targets = canonicalize_targets(targets)
        n = len(x)
        if n != len(targets):
            raise ValueError("Input/output sample mismatch.")
    if n < 2 and not val_path:
        raise ValueError("Training requires at least two input samples.")
    tr_idx, val_idx = (
        (np.arange(n), None)
        if val_path
        else train_test_split(np.arange(n), test_size=val_split, random_state=args.seed)
    )
    shape = source.target_shape if manifest else tuple(targets.shape[1:])
    if config["target_scaling"] == "minmax":
        scaler = MinMaxTargetScaler(config["target_range"], shape)
    elif manifest:
        scaler = SpatialTargetScaler().fit_batches(source.target_batches(tr_idx))
    else:
        scaler = create_target_scaler(targets[tr_idx])
    if expected_scaler is not None:
        _validate_scaler_compatibility(scaler, expected_scaler)
    if manifest:
        train_ds = OpenFWIManifestDataset(args.data_path, config, scaler, tr_idx)
        if val_path and not str(val_path).lower().endswith(".txt"):
            vx, vy = load_training_data(
                val_path, allow_pickle=getattr(args, "allow_pickle", False)
            )
            val_ds = _ArrayDataset(
                vx,
                vy,
                np.arange(len(vx)),
                config,
                scaler,
                input_channels=train_ds.in_channels,
            )
        else:
            val_ds = OpenFWIManifestDataset(
                val_path or args.data_path, config, scaler, val_idx
            )
    else:
        kwargs = {
            "preprocessing": config,
            "scaler": scaler,
            "input_channels": getattr(args, "input_channels", None),
            "single_channel": getattr(args, "single_channel", False),
        }
        train_ds = _ArrayDataset(x, targets, tr_idx, **kwargs)
        if val_path:
            if str(val_path).lower().endswith(".txt"):
                val_ds = OpenFWIManifestDataset(val_path, config, scaler)
            else:
                vx, vy = load_training_data(
                    val_path, allow_pickle=getattr(args, "allow_pickle", False)
                )
                val_ds = _ArrayDataset(vx, vy, np.arange(len(vx)), **kwargs)
        else:
            val_ds = _ArrayDataset(x, targets, val_idx, **kwargs)
    if (
        train_ds.input_shape != val_ds.input_shape
        or train_ds.target_shape != val_ds.target_shape
    ):
        raise ValueError("Training and validation sample shapes must match.")
    requested_channels = getattr(args, "input_channels", None)
    if requested_channels is not None and requested_channels != train_ds.in_channels:
        raise ValueError(
            f"Requested --input_channels={requested_channels}, but data has {train_ds.in_channels} channels."
        )
    args.in_channels = train_ds.in_channels
    if accelerator.is_main_process:
        os.makedirs(cache_dir, exist_ok=True)
        for filename, value in (
            ("scaler.pkl", scaler),
            (
                "data_metadata.pkl",
                {
                    "source": "openfwi" if manifest else "preprocessed",
                    "shape": (n, *train_ds.input_shape),
                    "in_channels": args.in_channels,
                    "target_shape": shape,
                    "out_dim": shape[0],
                    "preprocessing": config,
                    "val_data_path": val_path,
                    "cache_complete": True,
                },
            ),
        ):
            path = Path(cache_dir) / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as file:
                pickle.dump(value, file)
            temporary.replace(path)
        logger.info(
            "Preprocessing: %s; train=%d validation=%d (fixed=%s); no shard data cache",
            config,
            len(train_ds),
            len(val_ds),
            bool(val_path),
        )
    wait_for_everyone(accelerator)
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": accelerator.device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_ds, shuffle=True, **options),
        DataLoader(val_ds, shuffle=False, **options),
        scaler,
        train_ds.input_shape[1:],
        shape[0],
    )


def ssim_map(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Upstream OpenFWI SSIM map for [0,1] images (11px, sigma 1.5).

    Zero padding and constants intentionally match pytorch_ssim rather than
    skimage's valid-window/default-data-range convention.
    """
    if img1.shape != img2.shape or img1.ndim != 4:
        raise ValueError("SSIM expects equal (N,C,H,W) tensors.")
    channels = img1.shape[1]
    gaussian = torch.tensor(
        [math.exp(-((x - 5) ** 2) / (2 * 1.5**2)) for x in range(11)],
        dtype=torch.float32,
    )
    gaussian /= gaussian.sum()
    window = (
        (gaussian[:, None] @ gaussian[None, :])[None, None]
        .expand(channels, 1, 11, 11)
        .contiguous()
        .to(img1)
    )
    mu1 = F.conv2d(img1, window, padding=5, groups=channels)
    mu2 = F.conv2d(img2, window, padding=5, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1 = F.conv2d(img1 * img1, window, padding=5, groups=channels) - mu1_sq
    sigma2 = F.conv2d(img2 * img2, window, padding=5, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=5, groups=channels) - mu12
    return ((2 * mu12 + 0.01**2) * (2 * sigma12 + 0.03**2)) / (
        (mu1_sq + mu2_sq + 0.01**2) * (sigma1 + sigma2 + 0.03**2)
    )


def ssim(img1, img2):
    """Global mean SSIM of [0,1] tensors, matching upstream pytorch_ssim."""
    return ssim_map(img1, img2).mean()


class StreamingNormalizedMetrics:
    """Pixel-weighted normalized MAE/RMSE/SSIM, invariant to batch boundaries."""

    def __init__(self):
        self.count = 0
        self.absolute_sum = self.square_sum = self.ssim_sum = 0.0

    def update(self, target, prediction):
        target = torch.as_tensor(target, dtype=torch.float32).cpu()
        prediction = torch.as_tensor(prediction, dtype=torch.float32).cpu()
        if target.shape != prediction.shape:
            raise ValueError("Normalized metric shapes must match.")
        error = prediction.double() - target.double()
        self.absolute_sum += error.abs().sum().item()
        self.square_sum += error.square().sum().item()
        self.ssim_sum += (
            ssim_map((target + 1) / 2, (prediction + 1) / 2).double().sum().item()
        )
        self.count += error.numel()

    def compute(self):
        if not self.count:
            raise ValueError("No normalized metric samples.")
        return {
            "normalized_mae": self.absolute_sum / self.count,
            "normalized_rmse": math.sqrt(self.square_sum / self.count),
            "normalized_ssim": self.ssim_sum / self.count,
        }


def configure_tf32(args):
    """Disable both TF32 backends for explicit FP32 reproduction/determinism."""
    if getattr(args, "no_tf32", False) or getattr(args, "deterministic", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
