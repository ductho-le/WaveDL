"""Domain-agnostic preprocessing, shard manifests, and precision helpers.

Provides opt-in data pipeline capabilities for general wave-based inverse
problems:

- Lazy paired-shard ``.txt`` manifests (``<input.npy><TAB><target.npy>``) for
  large 1D/2D/3D datasets without recaching shards to disk.
- Fixed validation datasets (``--val_data_path``) alongside standard random
  train/validation splits.
- Multi-channel input contracts (``--input_channels``) validated against
  ``BaseModel.supported_input_channels``.
- Serializable input transforms (``signed_log_minmax``) and fixed-bound
  min-max target scaling (``MinMaxTargetScaler``) persisted in checkpoints
  and embedded during ONNX export.
- Explicit TF32 control (``--no_tf32``) for FP32 precision studies.
"""

from __future__ import annotations

import bisect
import contextlib
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .distributed import wait_for_everyone
from .targets import (
    SpatialTargetScaler,
    canonical_target_shape,
    canonicalize_targets,
    create_target_scaler,
)


def _validate_range(bounds: Any, name: str) -> list[float]:
    """Validate finite ``[MIN, MAX]`` bounds with ``MIN < MAX``."""
    if bounds is None or len(bounds) != 2:
        raise ValueError(f"{name} requires MIN MAX bounds.")
    low, high = map(float, bounds)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(f"{name} requires finite MIN < MAX, got {bounds}.")
    return [low, high]


def preprocessing_from_args(args: Any) -> dict[str, Any]:
    """Resolve and validate a serializable preprocessing contract from CLI args.

    Defaults preserve WaveDL's legacy behavior: identity inputs (``none``) and
    fitted standard target scaling (``standard``).
    """
    input_channels = getattr(args, "input_channels", None)
    if input_channels is not None and (
        type(input_channels) is not int or input_channels < 1
    ):
        raise ValueError("--input_channels must be a positive integer.")
    if input_channels not in (None, 1) and getattr(args, "single_channel", False):
        raise ValueError("--single_channel conflicts with --input_channels > 1.")

    input_transform = getattr(args, "input_transform", None) or "none"
    if input_transform not in ("none", "signed_log_minmax"):
        raise ValueError(
            f"Unsupported --input_transform {input_transform!r}; "
            "choose 'none' or 'signed_log_minmax'."
        )

    target_scaling = getattr(args, "target_scaling", None) or "standard"
    if target_scaling not in ("standard", "minmax"):
        raise ValueError(
            f"Unsupported --target_scaling {target_scaling!r}; "
            "choose 'standard' or 'minmax'."
        )

    input_range = getattr(args, "input_range", None)
    target_range = getattr(args, "target_range", None)
    if input_transform == "signed_log_minmax" or input_range is not None:
        input_range = _validate_range(input_range, "--input_range")
    if target_scaling == "minmax" or target_range is not None:
        target_range = _validate_range(target_range, "--target_range")

    return {
        "input_transform": input_transform,
        "input_range": input_range,
        "log_k": 1.0,
        "target_scaling": target_scaling,
        "target_range": target_range,
    }


def uses_preprocessed_pipeline(args: Any, preprocessing: dict[str, Any]) -> bool:
    """Return True when training requires the manifest/preprocessing data path."""
    return bool(
        str(args.data_path).lower().endswith(".txt")
        or getattr(args, "val_data_path", None)
        or preprocessing["input_transform"] != "none"
        or preprocessing["target_scaling"] != "standard"
        or getattr(args, "input_channels", None) not in (None, 1)
    )


def add_preprocessing_arguments(parser: Any) -> None:
    """Register train-time preprocessing options (persisted for inference/export)."""
    parser.add_argument(
        "--input_transform",
        choices=["none", "signed_log_minmax"],
        default=None,
        help="Input preprocessing transform (default: none)",
    )
    parser.add_argument(
        "--input_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Physical input bounds for signed_log_minmax preprocessing",
    )
    parser.add_argument(
        "--target_scaling",
        choices=["standard", "minmax"],
        default=None,
        help="Target scaling strategy (default: standard)",
    )
    parser.add_argument(
        "--target_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Physical target bounds for min-max scaling to [-1, 1]",
    )


def transform_inputs(
    values: Any, preprocessing: dict[str, Any] | None, *, copy: bool = True
) -> Any:
    """Transform NumPy arrays or PyTorch tensors without mutating shared buffers."""
    config = preprocessing or {}
    mode = config.get("input_transform", "none")
    if mode == "none":
        if not copy:
            return values
        return (
            values.clone() if torch.is_tensor(values) else np.array(values, copy=True)
        )
    if mode != "signed_log_minmax":
        raise ValueError(f"Unsupported input_transform: {mode!r}.")

    low, high = _validate_range(config.get("input_range"), "input_range")
    k = float(config.get("log_k", 1.0))
    if not np.isfinite(k) or k <= 0:
        raise ValueError(f"log_k must be finite and positive, got {k!r}.")
    low_t = float(np.log1p(abs(k * low)) * np.sign(low))
    high_t = float(np.log1p(abs(k * high)) * np.sign(high))
    span = high_t - low_t
    if not np.isfinite(span) or span <= 0:
        raise ValueError(
            f"Invalid transformed input span ({span!r}) for input_range={[low, high]} and log_k={k}."
        )

    if torch.is_tensor(values):
        work = (
            values.float()
            if values.dtype in (torch.float16, torch.bfloat16)
            else values
        )
        result = torch.log1p(torch.abs(k * work)) * torch.sign(work)
        return ((result - low_t) / span - 0.5) * 2.0

    arr = np.asarray(values, dtype=np.float64)
    result = np.log1p(np.abs(k * arr)) * np.sign(arr)
    result = ((result - low_t) / span - 0.5) * 2.0
    return result.astype(np.float32, copy=False)


def inverse_transform_inputs(values: Any, preprocessing: dict[str, Any] | None) -> Any:
    """Invert a serializable input preprocessing transform back to physical units."""
    config = preprocessing or {}
    mode = config.get("input_transform", "none")
    if mode == "none":
        return values
    if mode != "signed_log_minmax":
        raise ValueError(f"Unsupported input_transform: {mode!r}.")

    low, high = _validate_range(config.get("input_range"), "input_range")
    k = float(config.get("log_k", 1.0))
    if not np.isfinite(k) or k <= 0:
        raise ValueError(f"log_k must be finite and positive, got {k!r}.")
    low_t = float(np.log1p(abs(k * low)) * np.sign(low))
    high_t = float(np.log1p(abs(k * high)) * np.sign(high))
    span = high_t - low_t
    if not np.isfinite(span) or span <= 0:
        raise ValueError(
            f"Invalid transformed input span ({span!r}) for input_range={[low, high]} and log_k={k}."
        )

    if torch.is_tensor(values):
        work = (
            values.float()
            if values.dtype in (torch.float16, torch.bfloat16)
            else values
        )
        signed_log = (work / 2.0 + 0.5) * span + low_t
        return torch.sign(signed_log) * torch.expm1(torch.abs(signed_log)) / k

    arr = np.asarray(values, dtype=np.float64)
    signed_log = (arr / 2.0 + 0.5) * span + low_t
    return np.sign(signed_log) * np.expm1(np.abs(signed_log)) / k


class MinMaxTargetScaler:
    """Map fixed physical target bounds linearly to ``[-1, 1]``.

    Exposes affine ``mean_``, ``scale_``, ``var_``, and ``target_shape_``
    attributes so physical-unit metrics, soft constraints, and ONNX
    de-normalization work identically to standard scalers.
    """

    def __init__(self, bounds: Any, target_shape: tuple[int, ...]) -> None:
        self.bounds = _validate_range(bounds, "target_range")
        self.target_shape_ = tuple(target_shape)
        self.n_features_in_ = self.target_shape_[0]
        shape = (
            (self.n_features_in_, 1, 1)
            if len(self.target_shape_) == 3
            else (self.n_features_in_,)
        )
        low, high = self.bounds
        self.mean_ = np.full(shape, (low + high) / 2.0, dtype=np.float64)
        self.scale_ = np.full(shape, (high - low) / 2.0, dtype=np.float64)
        self.var_ = self.scale_**2

    def _array(self, values: Any) -> np.ndarray:
        array = canonicalize_targets(values)
        if tuple(array.shape[1:]) != self.target_shape_:
            raise ValueError(
                f"Expected target shape {self.target_shape_}, got {array.shape[1:]}."
            )
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("Targets must contain finite real values.")
        return array

    def transform(self, values: Any) -> np.ndarray:
        array = self._array(values).astype(np.float64, copy=False)
        low, high = self.bounds
        scaled = ((array - low) / (high - low) - 0.5) * 2.0
        return scaled.astype(np.float32, copy=False)

    def inverse_transform(self, values: Any) -> np.ndarray:
        array = self._array(values).astype(np.float64, copy=False)
        low, high = self.bounds
        return (array / 2.0 + 0.5) * (high - low) + low


def validate_input_channels(model_class: type, count: int, name: str) -> None:
    """Reject incompatible input channel counts before constructing a model."""
    supported = getattr(model_class, "supported_input_channels", (1,))
    if supported is not None and count not in supported:
        raise ValueError(
            f"Model '{name}' does not support in_channels={count}; "
            f"supported_input_channels={supported}."
        )


def _close_memmap(array: Any) -> None:
    """Release the underlying mmap handle of a read-only ``np.memmap`` if present."""
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        with contextlib.suppress(Exception):
            mmap_obj.close()


class ShardManifestDataset(Dataset):
    """Lazy paired ``.npy`` shard dataset addressed by cumulative lengths.

    Each non-blank manifest line has the form ``<input.npy><TAB><target.npy>``.
    Relative paths resolve against the manifest directory. Input shards must
    store channels-first tensors ``(N, C, *spatial_dims)`` (rank 3–5 for 1D,
    2D, or 3D wave data); target shards may be vectors ``(N,)`` / ``(N, T)``
    or dense fields ``(N, H, W)`` / ``(N, C, H, W)``. Shards are opened
    read-only and each sample is copied before any transform.
    """

    DEFAULT_MAX_OPEN_SHARDS = 32

    def __init__(
        self,
        path: str | Path,
        preprocessing: dict[str, Any] | None = None,
        scaler: Any = None,
        indices: np.ndarray | None = None,
        *,
        max_open_shards: int = DEFAULT_MAX_OPEN_SHARDS,
        _validated_spec: tuple[
            list[tuple[str, str]],
            list[int],
            tuple[int, ...],
            tuple[int, ...],
        ]
        | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.preprocessing = preprocessing
        self.scaler = scaler
        self.max_open_shards = max(2, int(max_open_shards))
        self._maps: OrderedDict[tuple[int, bool], np.ndarray] = OrderedDict()

        if _validated_spec is not None:
            paths_list, ends_list, in_shape, tgt_shape = _validated_spec
            self.paths = list(paths_list)
            self.ends = list(ends_list)
            self.input_shape: tuple[int, ...] | None = in_shape
            self.target_shape: tuple[int, ...] | None = tgt_shape
            total = self.ends[-1]
        else:
            self.paths = []
            self.ends = []
            total = 0
            self.input_shape = None
            self.target_shape = None

            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                pair = line.split("\t")
                if len(pair) != 2 or not all(part.strip() for part in pair):
                    raise ValueError(
                        f"{self.path}:{line_number}: expected input.npy<TAB>target.npy."
                    )
                paths = (
                    str((self.path.parent / pair[0].strip()).resolve()),
                    str((self.path.parent / pair[1].strip()).resolve()),
                )
                x = np.load(paths[0], mmap_mode="r", allow_pickle=False)
                y = np.load(paths[1], mmap_mode="r", allow_pickle=False)
                try:
                    if (
                        not (3 <= x.ndim <= 5)
                        or not all(x.shape)
                        or x.dtype.kind not in "biuf"
                    ):
                        raise ValueError(
                            "Shard inputs require non-empty channels-first real arrays of "
                            f"rank 3-5, got {x.shape}."
                        )
                    if y.dtype.kind not in "biuf":
                        raise ValueError(
                            "Shard targets must contain finite real values."
                        )
                    shape = canonical_target_shape(y.shape)
                    if len(x) != len(y):
                        raise ValueError(
                            f"Input/target shard mismatch: {x.shape} and {y.shape}."
                        )
                    if self.input_shape is not None and (
                        self.input_shape != tuple(x.shape[1:])
                        or self.target_shape != shape
                    ):
                        raise ValueError(
                            "All manifest shards must share the same input and target sample shapes."
                        )
                    self.input_shape = tuple(x.shape[1:])
                    self.target_shape = shape
                    total += len(x)
                    self.paths.append(paths)
                    self.ends.append(total)
                finally:
                    _close_memmap(x)
                    _close_memmap(y)

        if total == 0 or self.input_shape is None or self.target_shape is None:
            raise ValueError(f"Empty shard manifest: {self.path}")
        self.indices = np.arange(total) if indices is None else np.asarray(indices)
        self.in_channels = self.input_shape[0]
        self.shape = (total, *self.input_shape)

    def clone_with(
        self,
        *,
        preprocessing: dict[str, Any] | None = None,
        scaler: Any = None,
        indices: np.ndarray | None = None,
    ) -> ShardManifestDataset:
        """Create a view sharing already-validated manifest metadata without re-reading headers."""
        assert self.input_shape is not None and self.target_shape is not None
        return ShardManifestDataset(
            self.path,
            preprocessing=preprocessing,
            scaler=scaler,
            indices=indices,
            max_open_shards=self.max_open_shards,
            _validated_spec=(
                self.paths,
                self.ends,
                self.input_shape,
                self.target_shape,
            ),
        )

    def close(self) -> None:
        """Close all cached read-only shard memmaps."""
        while self._maps:
            _, evicted = self._maps.popitem(last=False)
            _close_memmap(evicted)

    def __len__(self) -> int:
        return len(self.indices)

    def _open_shard(self, shard: int, target: bool) -> np.ndarray:
        key = (shard, target)
        if key in self._maps:
            self._maps.move_to_end(key)
        else:
            if len(self._maps) >= self.max_open_shards:
                _, evicted = self._maps.popitem(last=False)
                _close_memmap(evicted)
            self._maps[key] = np.load(
                self.paths[shard][int(target)], mmap_mode="r", allow_pickle=False
            )
        return self._maps[key]

    def _sample(self, index: int, target: bool = False) -> np.ndarray:
        real_index = int(self.indices[index])
        shard = bisect.bisect_right(self.ends, real_index)
        offset = real_index - (self.ends[shard - 1] if shard else 0)
        arr = self._open_shard(shard, target)
        # Preserve native dtype (e.g. float64) for targets so scalers do not
        # lose precision on large physical offsets before standardization.
        if target:
            result = np.array(arr[offset], copy=True)
        else:
            result = np.array(arr[offset], dtype=np.float32, copy=True)
        if not np.isfinite(result).all():
            raise ValueError("Shard samples must contain finite real values.")
        return result

    def target_batches(self, indices: Any = None, batch_size: int = 64):
        """Yield canonicalized target batches via contiguous per-shard slices."""
        if indices is None:
            real_indices = self.indices
        else:
            real_indices = self.indices[np.asarray(indices, dtype=np.int64)]
        for start in range(0, len(real_indices), batch_size):
            chunk = real_indices[start : start + batch_size]
            if len(chunk) == 0:
                continue
            parts: list[np.ndarray] = []
            pos = 0
            while pos < len(chunk):
                first_real = int(chunk[pos])
                shard = bisect.bisect_right(self.ends, first_real)
                shard_start = self.ends[shard - 1] if shard else 0
                shard_end = self.ends[shard]
                run_end = pos + 1
                while (
                    run_end < len(chunk)
                    and shard_start <= int(chunk[run_end]) < shard_end
                ):
                    run_end += 1
                arr = self._open_shard(shard, True)
                offsets = np.asarray(chunk[pos:run_end], dtype=np.int64) - shard_start
                if len(offsets) == 1 or (
                    offsets[-1] - offsets[0] == len(offsets) - 1
                    and np.all(np.diff(offsets) == 1)
                ):
                    sub = np.array(arr[offsets[0] : offsets[-1] + 1], copy=True)
                else:
                    sub = np.array(arr[offsets], copy=True)
                if not np.isfinite(sub).all():
                    raise ValueError("Shard samples must contain finite real values.")
                parts.append(sub)
                pos = run_end
            batch = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            yield canonicalize_targets(batch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = transform_inputs(self._sample(index), self.preprocessing, copy=False)
        y = canonicalize_targets(self._sample(index, True)[None])
        if self.scaler is not None:
            y = self.scaler.transform(y)
        return torch.from_numpy(x), torch.from_numpy(np.asarray(y[0], dtype=np.float32))

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_maps": OrderedDict()}


class ManifestInputs(Dataset):
    """Tensor-like lazy input view for bounded-memory inference and ONNX export."""

    def __init__(self, source: ShardManifestDataset) -> None:
        self.source = source
        self.shape = source.shape
        self.ndim = len(self.shape)

    def close(self) -> None:
        """Close all cached read-only shard memmaps owned by the underlying source."""
        self.source.close()

    def __enter__(self) -> ManifestInputs:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int | slice) -> torch.Tensor:
        if isinstance(index, slice):
            return torch.stack([self[i] for i in range(*index.indices(len(self)))])
        return torch.from_numpy(self.source._sample(index))


class _ArrayDataset(Dataset):
    """Opt-in preprocessing and fixed-validation adapter for array file formats."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        preprocessing: dict[str, Any],
        scaler: Any,
        input_channels: int | None = None,
        single_channel: bool = False,
    ) -> None:
        self.x, self.y = x, canonicalize_targets(y)
        if len(self.x) != len(self.y):
            raise ValueError(
                f"Input/output sample mismatch: {len(self.x)} inputs and {len(self.y)} targets."
            )
        if self.x.dtype.kind not in "biuf":
            raise ValueError("Inputs must contain finite real values.")
        self.indices, self.preprocessing, self.scaler = indices, preprocessing, scaler
        channels_first = (
            x.ndim >= 5
            or (
                x.ndim == 4
                and not single_channel
                and input_channels != 1
                and (
                    (input_channels is not None and input_channels > 1)
                    or x.shape[1] <= 16
                )
            )
            or (
                x.ndim == 3
                and not single_channel
                and input_channels is not None
                and input_channels > 1
            )
        )
        self.add_channel = not channels_first
        self.in_channels = x.shape[1] if channels_first else 1
        self.input_shape = (
            self.in_channels,
            *(x.shape[2:] if channels_first else x.shape[1:]),
        )
        self.target_shape = tuple(self.y.shape[1:])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[index]
        x = np.array(self.x[i], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError("Inputs must contain finite real values.")
        if self.add_channel:
            x = x[None]
        y = self.scaler.transform(self.y[i : i + 1])[0]
        return torch.from_numpy(
            transform_inputs(x, self.preprocessing, copy=False)
        ), torch.from_numpy(np.asarray(y, dtype=np.float32))


def prepare_preprocessed_data(
    args: Any,
    config: dict[str, Any],
    logger: Any,
    accelerator: Any,
    cache_dir: str,
    val_split: float,
    expected_scaler: Any = None,
):
    """Prepare opt-in manifest, fixed-validation, or preprocessed DataLoaders.

    ``config`` is the resolved :func:`preprocessing_from_args` contract. The
    legacy cached path stays unchanged for default single-channel workflows.
    Statistics are always fitted on training targets only.
    """
    from .data import (
        _iter_target_batches,
        _validate_scaler_compatibility,
        load_training_data,
        memmap_worker_init_fn,
    )

    val_path = getattr(args, "val_data_path", None)
    manifest = str(args.data_path).lower().endswith(".txt")
    source: ShardManifestDataset | None = None
    if manifest:
        source = ShardManifestDataset(args.data_path)
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
    try:
        if n == 0:
            raise ValueError("Training dataset is empty.")
        if n < 2 and not val_path:
            raise ValueError("Training requires at least two input samples.")

        tr_idx, val_idx = (
            (np.arange(n), None)
            if val_path
            else train_test_split(
                np.arange(n), test_size=val_split, random_state=args.seed
            )
        )
        shape = source.target_shape if manifest else tuple(targets.shape[1:])
        assert shape is not None

        def _fit_scaler() -> Any:
            if config["target_scaling"] == "minmax":
                return MinMaxTargetScaler(config["target_range"], shape)
            if manifest:
                assert source is not None
                if len(shape) == 3:
                    return SpatialTargetScaler().fit_batches(
                        source.target_batches(tr_idx)
                    )
                return create_target_scaler(
                    np.concatenate(list(source.target_batches(tr_idx)), axis=0)
                )
            if len(shape) == 3:
                return SpatialTargetScaler().fit_batches(
                    _iter_target_batches(targets, tr_idx)
                )
            return create_target_scaler(targets[tr_idx])

        is_distributed = (
            getattr(accelerator, "num_processes", 1) > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        scaler_path = Path(cache_dir) / "scaler.pkl"
        if is_distributed:
            fit_error: Exception | None = None
            scaler = None
            if accelerator.is_main_process:
                try:
                    scaler = _fit_scaler()
                    if expected_scaler is not None:
                        _validate_scaler_compatibility(scaler, expected_scaler)
                    os.makedirs(cache_dir, exist_ok=True)
                    tmp_scaler = scaler_path.with_suffix(".pkl.tmp")
                    with tmp_scaler.open("wb") as file:
                        pickle.dump(scaler, file)
                    tmp_scaler.replace(scaler_path)
                except Exception as exc:
                    fit_error = exc
            err_msg = [str(fit_error) if fit_error is not None else None]
            torch.distributed.broadcast_object_list(err_msg, src=0)
            if err_msg[0] is not None:
                if fit_error is not None:
                    raise fit_error
                raise ValueError(err_msg[0])
            if not accelerator.is_main_process:
                with scaler_path.open("rb") as file:
                    scaler = pickle.load(file)
        else:
            scaler = _fit_scaler()
            if expected_scaler is not None:
                _validate_scaler_compatibility(scaler, expected_scaler)

        if manifest:
            assert source is not None
            train_ds: Dataset = source.clone_with(
                preprocessing=config, scaler=scaler, indices=tr_idx
            )
            if val_path and not str(val_path).lower().endswith(".txt"):
                vx, vy = load_training_data(
                    val_path, allow_pickle=getattr(args, "allow_pickle", False)
                )
                if len(vx) == 0:
                    raise ValueError("Validation dataset is empty.")
                is_single_ch_3d = (
                    train_ds.in_channels == 1 and len(train_ds.input_shape) == 4
                )
                val_ds: Dataset = _ArrayDataset(
                    vx,
                    vy,
                    np.arange(len(vx)),
                    config,
                    scaler,
                    input_channels=(
                        train_ds.in_channels
                        if train_ds.in_channels > 1
                        else getattr(args, "input_channels", None)
                    ),
                    single_channel=(
                        getattr(args, "single_channel", False) or is_single_ch_3d
                    ),
                )
            elif val_path:
                val_ds = ShardManifestDataset(val_path, config, scaler, val_idx)
            else:
                val_ds = source.clone_with(
                    preprocessing=config, scaler=scaler, indices=val_idx
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
                    val_ds = ShardManifestDataset(val_path, config, scaler)
                else:
                    vx, vy = load_training_data(
                        val_path, allow_pickle=getattr(args, "allow_pickle", False)
                    )
                    if len(vx) == 0:
                        raise ValueError("Validation dataset is empty.")
                    val_ds = _ArrayDataset(vx, vy, np.arange(len(vx)), **kwargs)
            else:
                val_ds = _ArrayDataset(x, targets, val_idx, **kwargs)
    finally:
        if source is not None:
            source.close()

    if (
        train_ds.input_shape != val_ds.input_shape
        or train_ds.target_shape != val_ds.target_shape
    ):
        raise ValueError("Training and validation sample shapes must match.")
    requested_channels = getattr(args, "input_channels", None)
    if requested_channels is not None and requested_channels != train_ds.in_channels:
        raise ValueError(
            f"Requested --input_channels={requested_channels}, "
            f"but data has {train_ds.in_channels} channels."
        )
    args.in_channels = train_ds.in_channels

    if accelerator.is_main_process:
        os.makedirs(cache_dir, exist_ok=True)
        for filename, value in (
            ("scaler.pkl", scaler),
            (
                "data_metadata.pkl",
                {
                    "source": "manifest" if manifest else "preprocessed",
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
        "worker_init_fn": memmap_worker_init_fn if args.workers > 0 else None,
    }
    drop_last_train = len(train_ds) > args.batch_size and (
        len(train_ds) % args.batch_size == 1
    )
    return (
        DataLoader(train_ds, shuffle=True, drop_last=drop_last_train, **options),
        DataLoader(val_ds, shuffle=False, **options),
        scaler,
        train_ds.input_shape[1:],
        shape[0],
    )


def configure_tf32(args: Any) -> None:
    """Disable matmul and cuDNN TF32 for explicit FP32 precision or determinism."""
    if getattr(args, "no_tf32", False) or getattr(args, "deterministic", False):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
