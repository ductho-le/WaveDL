"""Safe JSON serialization for WaveDL checkpoint metadata and target scalers.

New v2.0+ checkpoints persist ``training_meta.json`` and ``scaler.json`` so
inference, ONNX export, and checkpoint metadata inspection avoid ``pickle``
deserialization entirely. Legacy v1.x ``training_meta.pkl`` and ``scaler.pkl``
files remain readable as a backward-compatible fallback for trusted checkpoints.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


TRAINING_META_JSON = "training_meta.json"
TRAINING_META_PKL = "training_meta.pkl"
SCALER_JSON = "scaler.json"
SCALER_PKL = "scaler.pkl"

_NONFINITE_FLOAT_KEY = "__wavedl_nonfinite__"
_TUPLE_SHAPE_KEYS = frozenset({"in_shape", "target_shape"})


def _encode_json_value(value: Any) -> Any:
    """Convert Python/NumPy primitives into strict RFC-8259 JSON values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if math.isnan(fval):
            return {_NONFINITE_FLOAT_KEY: "nan"}
        if math.isinf(fval):
            return {_NONFINITE_FLOAT_KEY: "inf" if fval > 0 else "-inf"}
        return fval
    if isinstance(value, np.ndarray):
        return _encode_json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _encode_json_value(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _decode_json_value(value: Any) -> Any:
    """Restore encoded non-finite float markers from strict JSON."""
    if isinstance(value, dict):
        if len(value) == 1 and _NONFINITE_FLOAT_KEY in value:
            marker = value[_NONFINITE_FLOAT_KEY]
            if marker == "inf":
                return float("inf")
            if marker == "-inf":
                return float("-inf")
            if marker == "nan":
                return float("nan")
        return {k: _decode_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    return value


def _atomic_json_dump(payload: Any, path: str | Path) -> Path:
    """Atomically write strict UTF-8 JSON to ``path`` with ``fsync``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.tmp.{os.getpid()}"
    encoded = _encode_json_value(payload)
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(encoded, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def _atomic_pickle_dump(obj: Any, path: str | Path) -> Path:
    """Atomically serialize a legacy/custom object via pickle."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.tmp.{os.getpid()}"
    try:
        with tmp_path.open("wb") as file:
            pickle.dump(obj, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def save_training_meta(meta: dict[str, Any], directory: str | Path) -> Path:
    """Save checkpoint metadata safely as ``training_meta.json``.

    Removes any stale ``training_meta.pkl`` in the same directory so older
    pickle artifacts never shadow the JSON metadata.
    """
    target_dir = Path(directory)
    saved = _atomic_json_dump(meta, target_dir / TRAINING_META_JSON)
    with contextlib.suppress(OSError):
        (target_dir / TRAINING_META_PKL).unlink(missing_ok=True)
    return saved


def load_training_meta(directory: str | Path) -> dict[str, Any] | None:
    """Load checkpoint metadata, preferring safe ``training_meta.json``.

    Falls back to legacy ``training_meta.pkl`` only when ``training_meta.json``
    is absent.
    """
    target_dir = Path(directory)
    json_path = target_dir / TRAINING_META_JSON
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected JSON object in {json_path}, got {type(raw).__name__}."
            )
        meta = _decode_json_value(raw)
        for key in _TUPLE_SHAPE_KEYS:
            if isinstance(meta.get(key), list):
                meta[key] = tuple(int(v) for v in meta[key])
        return meta

    pkl_path = target_dir / TRAINING_META_PKL
    if pkl_path.is_file():
        with pkl_path.open("rb") as file:
            meta = pickle.load(file)
        if not isinstance(meta, dict):
            raise ValueError(f"Expected dict in {pkl_path}, got {type(meta).__name__}.")
        return meta
    return None


def _serialize_builtin_scaler(scaler: Any) -> dict[str, Any] | None:
    """Convert built-in WaveDL/sklearn target scalers to a strict JSON dict."""
    from .preprocessing import MinMaxTargetScaler
    from .targets import SpatialTargetScaler

    if isinstance(scaler, MinMaxTargetScaler):
        return {
            "format": "wavedl_scaler_v1",
            "type": "minmax",
            "bounds": [float(scaler.bounds[0]), float(scaler.bounds[1])],
            "target_shape_": [int(v) for v in scaler.target_shape_],
        }
    if isinstance(scaler, SpatialTargetScaler):
        if not hasattr(scaler, "scale_") or not hasattr(scaler, "mean_"):
            raise ValueError("Cannot serialize an unfitted SpatialTargetScaler.")
        return {
            "format": "wavedl_scaler_v1",
            "type": "spatial",
            "eps": float(scaler.eps),
            "mean_": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
            "scale_": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
            "var_": np.asarray(scaler.var_, dtype=np.float64).tolist(),
            "target_shape_": [int(v) for v in scaler.target_shape_],
            "n_features_in_": int(scaler.n_features_in_),
            "n_samples_seen_": int(getattr(scaler, "n_samples_seen_", 0)),
            "normalization_version_": int(
                getattr(
                    scaler,
                    "normalization_version_",
                    SpatialTargetScaler.NORMALIZATION_VERSION,
                )
            ),
        }
    if isinstance(scaler, StandardScaler):
        if not hasattr(scaler, "scale_") or not hasattr(scaler, "mean_"):
            raise ValueError("Cannot serialize an unfitted StandardScaler.")
        n_seen = getattr(scaler, "n_samples_seen_", 0)
        if isinstance(n_seen, np.ndarray):
            n_seen_out: Any = n_seen.astype(np.int64).tolist()
        else:
            n_seen_out = int(n_seen)
        payload: dict[str, Any] = {
            "format": "wavedl_scaler_v1",
            "type": "standard",
            "with_mean": bool(getattr(scaler, "with_mean", True)),
            "with_std": bool(getattr(scaler, "with_std", True)),
            "mean_": None
            if scaler.mean_ is None
            else np.asarray(scaler.mean_, dtype=np.float64).tolist(),
            "scale_": None
            if scaler.scale_ is None
            else np.asarray(scaler.scale_, dtype=np.float64).tolist(),
            "var_": None
            if getattr(scaler, "var_", None) is None
            else np.asarray(scaler.var_, dtype=np.float64).tolist(),
            "n_features_in_": int(
                getattr(
                    scaler,
                    "n_features_in_",
                    len(scaler.scale_) if scaler.scale_ is not None else 0,
                )
            ),
            "n_samples_seen_": n_seen_out,
        }
        return payload
    return None


def _deserialize_builtin_scaler(payload: dict[str, Any]) -> Any:
    """Reconstruct a WaveDL/sklearn target scaler from ``scaler.json``."""
    from .preprocessing import MinMaxTargetScaler
    from .targets import SpatialTargetScaler, _Float64StandardScaler

    if not isinstance(payload, dict) or payload.get("format") != "wavedl_scaler_v1":
        raise ValueError("Unsupported scaler.json format.")

    scaler_type = payload.get("type")
    if scaler_type == "minmax":
        bounds = payload["bounds"]
        target_shape = tuple(int(v) for v in payload["target_shape_"])
        return MinMaxTargetScaler(bounds, target_shape)

    if scaler_type == "spatial":
        scaler = SpatialTargetScaler(eps=float(payload.get("eps", 1e-12)))
        scaler.mean_ = np.asarray(payload["mean_"], dtype=np.float64)
        scaler.scale_ = np.asarray(payload["scale_"], dtype=np.float64)
        scaler.var_ = np.asarray(payload["var_"], dtype=np.float64)
        scaler.target_shape_ = tuple(int(v) for v in payload["target_shape_"])
        scaler.n_features_in_ = int(payload["n_features_in_"])
        scaler.n_samples_seen_ = int(payload.get("n_samples_seen_", 0))
        scaler.normalization_version_ = int(payload.get("normalization_version_", 0))
        return scaler

    if scaler_type == "standard":
        scaler = _Float64StandardScaler(
            with_mean=bool(payload.get("with_mean", True)),
            with_std=bool(payload.get("with_std", True)),
        )
        scaler.mean_ = (
            None
            if payload.get("mean_") is None
            else np.asarray(payload["mean_"], dtype=np.float64)
        )
        scaler.scale_ = (
            None
            if payload.get("scale_") is None
            else np.asarray(payload["scale_"], dtype=np.float64)
        )
        scaler.var_ = (
            None
            if payload.get("var_") is None
            else np.asarray(payload["var_"], dtype=np.float64)
        )
        scaler.n_features_in_ = int(payload["n_features_in_"])
        n_seen = payload.get("n_samples_seen_", 0)
        scaler.n_samples_seen_ = (
            np.asarray(n_seen, dtype=np.int64)
            if isinstance(n_seen, list)
            else int(n_seen)
        )
        return scaler

    raise ValueError(f"Unknown scaler type in scaler.json: {scaler_type!r}")


def save_scaler(scaler: Any, directory: str | Path) -> Path:
    """Save a target scaler as safe ``scaler.json`` (pickle fallback only for custom types)."""
    target_dir = Path(directory)
    serialized = _serialize_builtin_scaler(scaler)
    if serialized is not None:
        saved = _atomic_json_dump(serialized, target_dir / SCALER_JSON)
        with contextlib.suppress(OSError):
            (target_dir / SCALER_PKL).unlink(missing_ok=True)
        return saved
    saved = _atomic_pickle_dump(scaler, target_dir / SCALER_PKL)
    with contextlib.suppress(OSError):
        (target_dir / SCALER_JSON).unlink(missing_ok=True)
    return saved


def find_scaler_path(
    directory: str | Path, *, fallback_parent: bool = False
) -> Path | None:
    """Locate a scaler file, preferring local ``scaler.json`` over legacy ``scaler.pkl``."""
    target_dir = Path(directory)
    search_dirs = [target_dir]
    if fallback_parent:
        search_dirs.append(target_dir.parent)
    for folder in search_dirs:
        for filename in (SCALER_JSON, SCALER_PKL):
            candidate = folder / filename
            if candidate.is_file():
                return candidate
    return None


def load_scaler_from_path(path: str | Path) -> Any:
    """Load a target scaler from an explicit ``scaler.json`` or legacy ``scaler.pkl`` path."""
    target = Path(path)
    if target.suffix.lower() == ".json":
        with target.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return _deserialize_builtin_scaler(payload)
    with target.open("rb") as file:
        return pickle.load(file)


def load_scaler(
    directory: str | Path, *, fallback_parent: bool = False
) -> tuple[Any, Path] | tuple[None, None]:
    """Load a target scaler from ``directory`` (and optionally its parent)."""
    scaler_path = find_scaler_path(directory, fallback_parent=fallback_parent)
    if scaler_path is None:
        return None, None
    return load_scaler_from_path(scaler_path), scaler_path


def copy_or_save_scaler(
    src_dir: str | Path, dst_dir: str | Path, scaler: Any = None
) -> Path | None:
    """Persist a portable scaler inside ``dst_dir`` without leaving stale duplicates."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Prefer explicit in-memory built-in scaler -> always writes safe scaler.json
    if scaler is not None:
        return save_scaler(scaler, dst)

    # Otherwise copy existing scaler from src_dir (preferring scaler.json over scaler.pkl)
    for filename, stale_name in (
        (SCALER_JSON, SCALER_PKL),
        (SCALER_PKL, SCALER_JSON),
    ):
        src_path = src / filename
        if src_path.is_file():
            dst_path = dst / filename
            tmp_path = dst / f".{filename}.tmp.{os.getpid()}"
            try:
                shutil.copy2(src_path, tmp_path)
                os.replace(tmp_path, dst_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
                raise
            with contextlib.suppress(OSError):
                (dst / stale_name).unlink(missing_ok=True)
            return dst_path
    return None
