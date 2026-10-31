"""Bounded numeric dataset summaries for isolated daemon preview workers.

This module never imports the training loader or GUI. All decoding is performed
under explicit byte/element limits against one pinned ordinary file.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import sys
import zipfile
import zlib
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from . import dataset_headers as headers
from .data_keys import INPUT_KEYS, OUTPUT_KEYS
from .safe_files import open_directory


class PreviewRefused(ValueError):
    """Dataset preview violates the bounded numerical policy."""


@dataclass(frozen=True)
class PreviewLimits:
    max_file_bytes: int = 64 * 1024 * 1024
    max_decompressed_bytes: int = 64 * 1024 * 1024
    max_elements: int = 8 * 1024 * 1024
    max_sample_rows: int = 512
    max_input_features: int = 65536
    max_output_features: int = 256
    max_preview_elements: int = 16384
    max_output_bytes: int = 1024 * 1024
    max_zip_ratio: int = 100
    max_variables: int = 128
    max_header_bytes: int = 16 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Preview limits must be positive integers")


def _inspection_limits(limits):
    return headers.InspectionLimits(
        max_file_bytes=limits.max_file_bytes,
        max_header_bytes=limits.max_header_bytes,
        max_variables=limits.max_variables,
        max_declared_bytes=limits.max_decompressed_bytes,
    )


def _identity(stream, held=None):
    if held is not None and held.api:
        metadata = held.api.info(held.handle)
        ticks = metadata.write.dwHighDateTime << 32 | metadata.write.dwLowDateTime
        return {
            "st_dev": metadata.volume,
            "st_ino": metadata.index_high << 32 | metadata.index_low,
            "st_size": metadata.size_high << 32 | metadata.size_low,
            "st_mtime_ns": (ticks - 116444736000000000) * 100,
        }
    metadata = os.fstat(stream.fileno())
    return {
        name: getattr(metadata, name)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    }


def _validate_metadata(variables, limits):
    total_bytes = total_elements = 0
    for item in variables:
        import numpy as np

        if np.dtype(item["dtype"]).kind not in {"b", "i", "u", "f"}:
            raise PreviewRefused("Only real numeric arrays can be previewed")
        total_bytes += item["declared_bytes"]
        total_elements += math.prod(item["shape"])
    if (
        total_bytes > limits.max_decompressed_bytes
        or total_elements > limits.max_elements
    ):
        raise PreviewRefused("Declared dataset memory exceeds preview limit")


def _keys(variables, limits):
    names = {item["name"]: item for item in variables}
    selected = []
    for priorities, width_limit in (
        (INPUT_KEYS, limits.max_input_features),
        (OUTPUT_KEYS, limits.max_output_features),
    ):
        key = next((key for key in priorities if key in names), None)
        if key is None:
            raise PreviewRefused("Dataset lacks training input/output keys")
        shape = names[key]["shape"]
        if not shape or not all(shape) or math.prod(shape[1:]) > width_limit:
            raise PreviewRefused("Empty or excessive feature shape")
        selected.append(key)
    if names[selected[0]]["shape"][0] != names[selected[1]]["shape"][0]:
        raise PreviewRefused("Input and output sample counts differ")
    return selected


def _npz(stream, limits):
    policy = _inspection_limits(limits)
    headers._zip_directory(stream, policy)
    variables = headers._npz(stream, policy)
    _validate_metadata(variables, limits)
    keys = _keys(variables, limits)
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        total = 0
        for member in archive.infolist():
            name = member.filename
            mode = member.external_attr >> 16
            if "/" in name or "\\" in name or mode & 0o170000 == 0o120000:
                raise PreviewRefused("Unsafe archive member path")
            total += member.file_size
            if (
                total > limits.max_decompressed_bytes
                or member.file_size
                > max(member.compress_size, 1) * limits.max_zip_ratio
            ):
                raise PreviewRefused("Archive expansion exceeds preview limit")
        arrays = []
        import numpy as np

        for key in keys:
            with archive.open(key + ".npy") as member:
                # ZipExtFile enforces declared uncompressed length and CRC. The
                # directory and numeric headers have both been bounded first.
                arrays.append(np.lib.format.read_array(member, allow_pickle=False))
    return variables, keys, arrays


def _mat_expansion(stream, limits):
    """Drain every compressed MAT5 block before SciPy numerical decoding.

    Header-only inspection intentionally stops before payloads. Therefore its
    declared sizes alone cannot defend loadmat against trailing expansion.
    """
    stream.seek(0, 2)
    end = stream.tell()
    stream.seek(0)
    header = stream.read(128)
    if len(header) < 128 or header[126:128] not in {b"IM", b"MI"}:
        stream.seek(0)
        return
    endian = "<" if header[126:128] == b"IM" else ">"
    total = 128
    while stream.tell() < end:
        tag = stream.read(8)
        if len(tag) != 8:
            raise PreviewRefused("Truncated MAT tag")
        kind, size = struct.unpack(endian + "II", tag)
        if kind not in {14, 15} or stream.tell() + size > end:
            raise PreviewRefused("Invalid MAT extent")
        if kind == 14:
            total += size
            stream.seek(size, 1)
        else:
            decoder = zlib.decompressobj()
            remaining = size
            while remaining:
                chunk = stream.read(min(65536, remaining))
                remaining -= len(chunk)
                pending = chunk
                while pending:
                    expanded = decoder.decompress(
                        pending, min(65536, limits.max_decompressed_bytes - total + 1)
                    )
                    total += len(expanded)
                    if total > limits.max_decompressed_bytes:
                        raise PreviewRefused("MAT expansion exceeds preview limit")
                    pending = decoder.unconsumed_tail
                if decoder.unused_data:
                    raise PreviewRefused("Trailing compressed MAT data")
            if not decoder.eof:
                raise PreviewRefused("Truncated compressed MAT payload")
        if total > limits.max_decompressed_bytes:
            raise PreviewRefused("MAT expansion exceeds preview limit")
    stream.seek(0)


def _mat(stream, limits):
    variables = headers._mat(stream, _inspection_limits(limits))
    _validate_metadata(variables, limits)
    keys = _keys(variables, limits)
    _mat_expansion(stream, limits)
    from scipy.io import loadmat

    values = loadmat(stream, variable_names=keys, appendmat=False)
    return variables, keys, [values[key] for key in keys]


def _hdf(stream, limits, *, matlab):
    import h5py
    import numpy as np

    variables = headers._hdf(stream, _inspection_limits(limits), matlab=matlab)
    _validate_metadata(variables, limits)
    keys = _keys(variables, limits)
    stream.seek(0)
    with h5py.File(stream, "r", driver="fileobj") as file:
        # Check every dataset's storage contract before any numerical read.
        for item in variables:
            dataset = file[item["name"]]
            if not isinstance(dataset, h5py.Dataset):
                raise PreviewRefused("Sparse arrays are unsupported")
            if (
                dataset.chunks
                and math.prod(dataset.chunks) * dataset.dtype.itemsize
                > limits.max_decompressed_bytes
            ):
                raise PreviewRefused("HDF chunk expansion exceeds preview limit")
            plist = dataset.id.get_create_plist()
            for index in range(plist.get_nfilters()):
                if plist.get_filter(index)[0] not in {1, 2, 3}:
                    raise PreviewRefused("Unsupported HDF filter")
        shapes = [
            next(v["shape"] for v in variables if v["name"] == key) for key in keys
        ]
        indices = _indices(shapes, limits)
        arrays = []
        for key in keys:
            dataset = file[key]
            rows = []
            for index in indices:
                row = (
                    dataset[..., int(index)].transpose()
                    if matlab
                    else dataset[int(index)]
                )
                rows.append(np.asarray(row))
            arrays.append(np.stack(rows))
    return variables, keys, arrays, indices


def _indices(shapes, limits):
    import numpy as np

    width = sum(math.prod(shape[1:]) for shape in shapes)
    count = min(shapes[0][0], limits.max_sample_rows, limits.max_elements // width)
    if count < 1:
        raise PreviewRefused("Sample exceeds decoded element limit")
    return np.linspace(0, shapes[0][0] - 1, count, dtype=int)


def _finite(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _features(width, maximum):
    import numpy as np

    return np.linspace(0, width - 1, min(width, maximum), dtype=int)


def _histograms(array):
    import numpy as np

    result = []
    for index in _features(array.shape[1], 8):
        column = array[:, index]
        column = column[np.isfinite(column)]
        if column.size and np.max(np.abs(column)) > 1e150:
            raise PreviewRefused("Numeric magnitude exceeds histogram safety limit")
        counts, edges = np.histogram(column, bins=32)
        result.append({"feature_index": int(index), "counts": counts, "edges": edges})
    return result


def _summaries(x, y, indices, shapes, limits):
    import numpy as np

    x_flat = x.reshape(len(indices), -1).astype(np.float64)
    y_flat = y.reshape(len(indices), -1).astype(np.float64)
    stats = _features(x_flat.shape[1], 40)
    scatter = _features(x_flat.shape[1], 8)
    corr = _features(y_flat.shape[1], 30)
    with np.errstate(all="ignore"):
        matrix = (
            np.corrcoef(y_flat[:, corr], rowvar=False)
            if len(corr) > 1 and len(indices) > 1
            else None
        )
        means = x_flat[:, stats].mean(axis=0)
        stds = x_flat[:, stats].std(axis=0)
    samples = []
    remaining = limits.max_preview_elements
    sample_count = min(3, len(indices), limits.max_preview_elements)
    for position in range(sample_count):
        row = x[position].astype(np.float64, copy=False)
        # Preserve dimensionality; bound a leading rectangular slice.
        bounded_shape = list(row.shape) or [1]
        budget = max(1, remaining // (sample_count - position))
        while math.prod(bounded_shape) > budget:
            axis = max(range(len(bounded_shape)), key=bounded_shape.__getitem__)
            bounded_shape[axis] = max(1, bounded_shape[axis] // 2)
        value = (
            row[tuple(slice(0, size) for size in bounded_shape)]
            if row.ndim
            else row.reshape(1)
        )
        remaining -= value.size
        samples.append(
            {
                "sample_index": int(indices[position]),
                "feature_shape": shapes[0][1:],
                "shape": list(value.shape),
                "values": value,
            }
        )
    return _finite(
        {
            "sample_indices": indices,
            "input_histograms": _histograms(x_flat),
            "output_histograms": _histograms(y_flat),
            "correlation": {"feature_indices": corr, "matrix": matrix},
            "input_statistics": {
                "feature_indices": stats,
                "means": means,
                "stds": stds,
            },
            "scatter": {
                "feature_indices": scatter,
                "output_index": 0,
                "x": x_flat[:, scatter],
                "y": y_flat[:, 0],
            },
            "samples": samples,
        }
    )


def preview_dataset(path, *, expected_identity, limits=PreviewLimits()):
    """Return schema-v1 summaries, refusing changed files or unsafe decoding."""
    path = Path(path)
    if not path.is_absolute():
        raise PreviewRefused("Dataset path must be absolute")
    envelope = expected_identity
    if isinstance(envelope, dict) and "identity" in envelope:
        if set(envelope) - {
            "path",
            "identity",
            "root_id",
            "relative_path",
            "root_path",
            "root_identity",
            "ancestry",
        }:
            raise PreviewRefused("Unsupported dataset selection fields")
        if envelope.get("path") != str(path):
            raise PreviewRefused("Dataset selection changed")
        expected_identity = envelope["identity"]
    identity_keys = {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
    if (
        not isinstance(expected_identity, dict)
        or set(expected_identity) != identity_keys
        or any(
            type(expected_identity[key]) is not int or expected_identity[key] < 0
            for key in identity_keys
        )
    ):
        raise PreviewRefused("Expected dataset identity is required")
    expected = expected_identity
    with ExitStack() as stack:
        ancestry = []
        for entry in envelope.get("ancestry", ()):
            anchor = stack.enter_context(open_directory(Path(entry["path"])))
            if anchor.api:
                native = anchor.api.info(anchor.handle)
                actual = (native.volume, native.index_high << 32 | native.index_low)
            else:
                native = os.fstat(anchor.handle)
                actual = (native.st_dev, native.st_ino)
            if actual != (entry["st_dev"], entry["st_ino"]):
                raise PreviewRefused("Dataset ancestry changed")
            ancestry.append(anchor)
        directory = stack.enter_context(open_directory(path.parent))
        held = stack.enter_context(directory.open_file(path.name))
        if held.api:
            metadata = _identity(None, held)
            if metadata != expected or metadata["st_size"] > limits.max_file_bytes:
                raise PreviewRefused("Dataset identity changed or file exceeds limit")
            # Windows keeps immutable no-write/no-delete-sharing native handles.
            # A bounded snapshot supplies a portable seekable file object without
            # reopening a pathname or transferring ownership of the native handle.
            raw = bytearray()
            while chunk := held.read(min(65536, limits.max_file_bytes + 1 - len(raw))):
                raw.extend(chunk)
                if len(raw) > limits.max_file_bytes:
                    raise PreviewRefused("Dataset file exceeds limit")
            stream_context = io.BytesIO(raw)
        else:
            stream_context = os.fdopen(os.dup(held.handle), "rb")
        with stream_context as stream:
            before = _identity(stream, held)
            if before != expected:
                raise PreviewRefused("Dataset identity changed")
            if before["st_size"] > limits.max_file_bytes:
                raise PreviewRefused("Dataset file exceeds limit")
            held.guard()
            suffix = path.suffix.lower()
            sampled = False
            if suffix == ".npz":
                variables, keys, arrays = _npz(stream, limits)
            elif suffix in {".h5", ".hdf5", ".mat"}:
                hdf = suffix != ".mat" or headers._is_hdf(stream)
                stream.seek(0)
                if hdf:
                    variables, keys, arrays, indices = _hdf(
                        stream, limits, matlab=suffix == ".mat"
                    )
                    sampled = True
                else:
                    variables, keys, arrays = _mat(stream, limits)
            else:
                raise PreviewRefused("Unsupported dataset format")
            shapes = [
                next(v["shape"] for v in variables if v["name"] == key) for key in keys
            ]
            if not sampled:
                indices = _indices(shapes, limits)
                arrays = [array[indices] for array in arrays]
            result = {
                "schema_version": 1,
                "identity": before,
                "format": suffix[1:],
                "input_key": keys[0],
                "output_key": keys[1],
                "input_shape": shapes[0],
                "output_shape": shapes[1],
                **_summaries(*arrays, indices, shapes, limits),
                "limits": asdict(limits),
            }
            held.guard()
            for anchor in ancestry:
                anchor.guard()
            if _identity(stream, held) != before:
                raise PreviewRefused("Dataset changed during preview")
    if (
        len(json.dumps(result, allow_nan=False, separators=(",", ":")).encode())
        > limits.max_output_bytes
    ):
        raise PreviewRefused("Preview JSON exceeds output limit")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--limits")
    for name, default in asdict(PreviewLimits()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    arguments = vars(parser.parse_args(argv))
    path = arguments.pop("path")
    try:
        identity = json.loads(arguments.pop("expected_identity"))
        override = arguments.pop("limits")
        if override is not None:
            arguments.update(json.loads(override))
        result = preview_dataset(
            path, expected_identity=identity, limits=PreviewLimits(**arguments)
        )
    except Exception as exc:
        message = str(exc).lower()
        code = (
            "identity_changed"
            if "changed" in message
            else "too_large"
            if any(
                term in message
                for term in ("limit", "exceed", "excessive", "expansion")
            )
            else "unsupported"
        )
        print(json.dumps({"error": "Dataset preview refused", "code": code}))
        return 1
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
