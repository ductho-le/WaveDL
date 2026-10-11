"""Target-shape utilities for vector and dense 2D regression.

WaveDL historically treated every target as a two-dimensional ``(N, T)``
array.  Dense inverse problems need a different contract: a scalar field is
``(N, H, W)`` and a multi-property field is ``(N, C, H, W)``.  This module
keeps the old vector contract intact while providing one canonical form for
2D fields: ``(N, C, H, W)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import StandardScaler


if TYPE_CHECKING:
    from collections.abc import Iterable


TargetShape = tuple[int, ...]


def canonical_target_shape(raw_shape: tuple[int, ...]) -> TargetShape:
    """Return a canonical per-sample shape without loading target values."""

    if any(size <= 0 for size in raw_shape[1:]):
        raise ValueError(f"Target dimensions must be positive, got {raw_shape}.")
    if len(raw_shape) == 1:
        return (1,)
    if len(raw_shape) == 2:
        return tuple(raw_shape[1:])
    if len(raw_shape) == 3:
        return (1, *raw_shape[1:])
    if len(raw_shape) == 4:
        return tuple(raw_shape[1:])
    raise ValueError(
        "Unsupported target shape "
        f"{raw_shape}. Expected (N,), (N,T), (N,H,W), or (N,C,H,W)."
    )


def canonicalize_targets(values: Any) -> np.ndarray:
    """Return targets in WaveDL's canonical vector or field layout.

    Accepted layouts are:

    - ``(N,)`` -> ``(N, 1)`` for scalar regression
    - ``(N, T)`` for vector regression
    - ``(N, H, W)`` -> ``(N, 1, H, W)`` for a scalar 2D field
    - ``(N, C, H, W)`` for a multi-property 2D field

    Channels-last dense targets are intentionally not guessed because small
    spatial grids make that heuristic unsafe.  Callers should transpose them
    explicitly before saving the dataset.
    """

    array = np.asarray(values)
    canonical_target_shape(array.shape)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return np.expand_dims(array, axis=1)
    if array.ndim == 4:
        return array
    raise ValueError(
        "Unsupported target shape "
        f"{array.shape}. Expected (N,), (N,T), (N,H,W), or (N,C,H,W)."
    )


def get_target_shape(values: Any) -> TargetShape:
    """Return the per-sample canonical target shape."""

    return canonical_target_shape(tuple(np.shape(values)))


def is_spatial_target_shape(target_shape: TargetShape) -> bool:
    """Whether a canonical per-sample target shape represents a 2D field."""

    return len(target_shape) == 3


def flatten_targets_by_channel(values: Any) -> np.ndarray:
    """Flatten samples and pixels while retaining target channels.

    Vector targets are returned unchanged as ``(N, T)``.  Field targets are
    converted from ``(N, C, H, W)`` to ``(N*H*W, C)``.  This lets the existing
    R2/Pearson helpers compute channel-wise summary metrics without treating
    every pixel as a separate named output.
    """

    array = canonicalize_targets(values)
    if array.ndim == 2:
        return array
    return np.moveaxis(array, 1, -1).reshape(-1, array.shape[1])


class SpatialTargetScaler:
    """Channel-wise standardization for dense 2D target fields.

    Statistics are fitted over samples and both spatial axes, independently
    for each physical property channel.  The stored ``mean_`` and ``scale_``
    have shape ``(C, 1, 1)``, so they broadcast in NumPy, PyTorch, and ONNX.
    This is both substantially smaller and more physically meaningful than a
    separate scaler for every grid cell.
    """

    NORMALIZATION_VERSION = 1

    def __init__(self, eps: float = 1e-12):
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.eps = eps

    def fit(self, values: Any) -> SpatialTargetScaler:
        return self.fit_batches([values])

    @staticmethod
    def _dense_array(values: Any) -> np.ndarray:
        array = canonicalize_targets(values)
        if array.ndim != 4:
            raise ValueError(
                f"SpatialTargetScaler expects (N,H,W) or (N,C,H,W), got {array.shape}."
            )
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("Dense targets must contain only finite real values.")
        return array

    def fit_batches(self, batches: Iterable[Any]) -> SpatialTargetScaler:
        """Fit channel statistics from batches without materializing all maps."""

        running_mean = None
        running_m2 = None
        value_count = 0
        sample_count = 0
        target_shape = None

        for values in batches:
            array = self._dense_array(values)
            # Empty chunks can occur when streaming a filtered split. Skip
            # them before computing moments or dividing by the sample count.
            if array.shape[0] == 0:
                continue
            if target_shape is None:
                target_shape = tuple(array.shape[1:])
                running_mean = np.zeros(array.shape[1], dtype=np.float64)
                running_m2 = np.zeros(array.shape[1], dtype=np.float64)
            elif tuple(array.shape[1:]) != target_shape:
                raise ValueError(
                    f"Inconsistent field shapes: expected {target_shape}, "
                    f"got {tuple(array.shape[1:])}."
                )

            work = array.astype(np.float64, copy=False)
            reduce_axes = (0, 2, 3)
            batch_count = array.shape[0] * array.shape[2] * array.shape[3]
            batch_mean = work.mean(axis=reduce_axes)
            batch_m2 = np.square(work - batch_mean[None, :, None, None]).sum(
                axis=reduce_axes
            )

            # Parallel Welford update. Computing E[x^2] - E[x]^2 loses all
            # useful precision for physical fields with a large offset (for
            # example, modulus or velocity maps around 1e9 with small local
            # variations).
            total_count = value_count + batch_count
            delta = batch_mean - running_mean
            running_mean += delta * (batch_count / total_count)
            running_m2 += batch_m2 + np.square(delta) * (
                value_count * batch_count / total_count
            )
            value_count = total_count
            sample_count += array.shape[0]

        if (
            target_shape is None
            or value_count == 0
            or running_mean is None
            or running_m2 is None
        ):
            raise ValueError("Cannot fit SpatialTargetScaler on empty target batches.")

        variance = np.maximum(running_m2 / value_count, 0.0)
        self.mean_ = running_mean[:, None, None]
        self.var_ = variance[:, None, None]
        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        self.n_features_in_ = target_shape[0]
        self.n_samples_seen_ = sample_count
        self.target_shape_ = target_shape
        self.normalization_version_ = self.NORMALIZATION_VERSION
        return self

    def _validate(self, values: Any) -> np.ndarray:
        if not hasattr(self, "scale_"):
            raise RuntimeError("SpatialTargetScaler has not been fitted.")
        array = self._dense_array(values)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} target channel(s), "
                f"got {array.shape[1]}."
            )
        if tuple(array.shape[1:]) != self.target_shape_:
            raise ValueError(
                f"Expected target shape {self.target_shape_}, "
                f"got {tuple(array.shape[1:])}."
            )
        return array

    def transform(self, values: Any) -> np.ndarray:
        array = self._validate(values)
        dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
        result = np.empty_like(array, dtype=dtype)
        # Keep the fitted moments in float64 during arithmetic. Rounding a
        # large physical mean to float32 first can shift standardized values
        # by an entire standard deviation. An output buffer preserves the
        # input dtype without allocating a full float64 copy of dense maps.
        np.subtract(array, self.mean_, out=result, dtype=np.float64, casting="unsafe")
        np.divide(result, self.scale_, out=result, dtype=np.float64, casting="unsafe")
        return result

    def inverse_transform(self, values: Any) -> np.ndarray:
        array = self._validate(values)
        dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
        result = np.empty_like(array, dtype=dtype)
        np.multiply(array, self.scale_, out=result, dtype=np.float64, casting="unsafe")
        np.add(result, self.mean_, out=result, dtype=np.float64, casting="unsafe")
        return result

    def fit_transform(self, values: Any) -> np.ndarray:
        return self.fit(values).transform(values)


def create_target_scaler(values: Any) -> StandardScaler | SpatialTargetScaler:
    """Fit the appropriate scaler for vector or 2D field targets."""

    array = canonicalize_targets(values)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("Targets must contain only finite real values.")
    if array.ndim == 2:
        return StandardScaler().fit(array)
    return SpatialTargetScaler().fit(array)


def scaler_target_shape(scaler: Any, fallback_channels: int) -> TargetShape:
    """Recover a canonical target shape from new or legacy scaler objects."""

    target_shape = getattr(scaler, "target_shape_", None)
    if target_shape is not None:
        return tuple(target_shape)
    return (fallback_channels,)
