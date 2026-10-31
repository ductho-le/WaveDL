"""Numerical preview uses pinned inputs and explicit allocation budgets."""

import json
import os
import struct
import subprocess
import sys
import zipfile
import zlib
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from wavedl.core.dataset_preview import PreviewLimits, PreviewRefused, preview_dataset
from wavedl.core.safe_files import open_directory


def identity(path):
    if os.name == "nt":
        # Python 3.13 stat() exposes a wider file ID than the Win32 identity
        # carried by dataset selections. Derive that public stamp independently.
        with open_directory(path.parent) as parent, parent.open_file(path.name) as file:
            info = file.api.info(file.handle)
            return {
                "st_dev": info.volume,
                "st_ino": info.index_high << 32 | info.index_low,
                "st_size": info.size_high << 32 | info.size_low,
                "st_mtime_ns": (
                    (info.write.dwHighDateTime << 32 | info.write.dwLowDateTime)
                    - 116444736000000000
                )
                * 100,
            }
    metadata = path.stat()
    return {
        key: getattr(metadata, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    }


def preview(path, **kwargs):
    return preview_dataset(path, expected_identity=identity(path), **kwargs)


def test_native_selection_identity_uses_handle_fields(tmp_path, monkeypatch):
    from contextlib import nullcontext

    native = SimpleNamespace(
        volume=7,
        index_high=2,
        index_low=3,
        size_high=1,
        size_low=5,
        write=SimpleNamespace(dwHighDateTime=27111902, dwLowDateTime=3577643008),
    )
    reads = []

    def info(handle):
        reads.append(handle)
        return native

    file = SimpleNamespace(handle=991, api=SimpleNamespace(info=info))
    parent = SimpleNamespace(open_file=lambda name: nullcontext(file))
    monkeypatch.setitem(identity.__globals__, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(
        identity.__globals__, "open_directory", lambda path: nullcontext(parent)
    )
    assert identity(tmp_path / "data.npz") == {
        "st_dev": 7,
        "st_ino": 8589934595,
        "st_size": 4294967301,
        "st_mtime_ns": 0,
    }
    assert reads == [991]


@pytest.mark.parametrize("suffix", [".npz", ".mat", ".h5", ".hdf5"])
def test_actual_numeric_formats_produce_six_bounded_views(tmp_path, suffix):
    path = tmp_path / ("data" + suffix)
    x = np.arange(6000, dtype=np.float32).reshape(1000, 2, 3)
    y = np.arange(2000, dtype=np.float64).reshape(1000, 2)
    if suffix == ".npz":
        np.savez(path, X=x, Y=y)
    elif suffix == ".mat":
        savemat(path, {"X": x, "Y": y}, do_compression=True)
    else:
        with h5py.File(path, "w") as file:
            file["X"], file["Y"] = x, y
    original_bytes = path.read_bytes()
    result = preview(path)
    assert result["input_shape"] == [1000, 2, 3]
    assert result["sample_indices"][0] == 0
    assert result["sample_indices"][-1] == 999
    assert len(result["sample_indices"]) == 512
    assert len(result["samples"]) == 3
    assert result["samples"][0]["values"] == x[0].tolist()
    assert len(result["input_histograms"][0]["counts"]) == 32
    assert set(result) == {
        "schema_version",
        "identity",
        "format",
        "input_key",
        "output_key",
        "input_shape",
        "output_shape",
        "sample_indices",
        "input_histograms",
        "output_histograms",
        "correlation",
        "input_statistics",
        "scatter",
        "samples",
        "limits",
    }
    assert preview(path) == result
    assert path.read_bytes() == original_bytes
    json.dumps(result, allow_nan=False)


def test_mat73_reverses_axes_and_sample_slices(tmp_path):
    path = tmp_path / "mat73.mat"
    x = np.arange(120).reshape(10, 3, 4)
    y = np.arange(20).reshape(10, 2)
    with h5py.File(path, "w") as file:
        file["X"], file["Y"] = x.transpose(), y.transpose()
    result = preview(path)
    assert result["input_shape"] == list(x.shape)
    assert result["samples"][0]["values"] == x[0].tolist()
    assert result["scatter"]["y"] == y[:, 0].tolist()


def test_key_priority_and_nonfinite_nulls(tmp_path):
    path = tmp_path / "keys.npz"
    x = np.array([[1, np.inf], [np.nan, 3], [4, 4]])
    np.savez(
        path,
        input_train=x,
        X=np.zeros((3, 1)),
        output_train=np.ones((3, 2)),
        Y=np.zeros((3, 1)),
    )
    result = preview(path)
    assert result["input_key"] == "input_train"
    assert result["output_key"] == "output_train"
    assert result["samples"][0]["values"][1] is None
    assert result["input_statistics"]["means"][0] is None
    assert result["correlation"]["matrix"] == [[None, None], [None, None]]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "array",
    [
        np.empty((0, 2)),
        np.ones((2, 2), dtype=complex),
        np.ones((2, 2), dtype=[("a", "f4")]),
        np.array([[object()]], dtype=object),
    ],
)
def test_invalid_array_types_rejected(tmp_path, array):
    path = tmp_path / "invalid.npz"
    np.savez(path, X=array, Y=np.ones((array.shape[0], 1)))
    with pytest.raises(ValueError):
        preview(path)


def test_mismatched_rows_feature_width_and_output_caps(tmp_path):
    path = tmp_path / "shape.npz"
    np.savez(path, X=np.ones((3, 10)), Y=np.ones((2, 1)))
    with pytest.raises(PreviewRefused, match="counts"):
        preview(path)
    np.savez(path, X=np.ones((3, 10)), Y=np.ones((3, 1)))
    with pytest.raises(PreviewRefused, match="feature"):
        preview(path, limits=replace(PreviewLimits(), max_input_features=9))
    with pytest.raises(PreviewRefused, match="JSON"):
        preview(path, limits=replace(PreviewLimits(), max_output_bytes=50))


def test_changed_identity_symlink_and_changed_ancestry_refused(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, X=np.ones((2, 2)), Y=np.ones((2, 1)))
    old = identity(path)
    np.savez(path, X=np.ones((3, 2)), Y=np.ones((3, 1)))
    with pytest.raises(PreviewRefused, match="identity"):
        preview_dataset(path, expected_identity=old)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        preview_dataset(link, expected_identity=identity(path))
    envelope = {
        "path": str(path),
        "identity": identity(path),
        "ancestry": [{"path": str(tmp_path), "st_dev": 0, "st_ino": 0}],
    }
    with pytest.raises(PreviewRefused, match="ancestry"):
        preview_dataset(path, expected_identity=envelope)


def test_file_and_zip_expansion_limits_precede_numeric_reads(tmp_path, monkeypatch):
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, X=np.zeros((10000, 10)), Y=np.zeros((10000, 1)))
    monkeypatch.setattr(
        np.lib.format,
        "read_array",
        lambda *a, **k: pytest.fail("No numeric decoding before cap"),
    )
    with pytest.raises(PreviewRefused, match="expansion"):
        preview(path)
    with pytest.raises(PreviewRefused, match="file"):
        preview(path, limits=replace(PreviewLimits(), max_file_bytes=1))


def test_mat_trailing_compressed_bomb_refused_before_loadmat(tmp_path, monkeypatch):
    path = tmp_path / "bomb.mat"
    savemat(path, {"X": np.ones((2, 2)), "Y": np.ones((2, 1))})
    original = path.read_bytes()
    # Preserve a legitimate bounded matrix header then add extreme expansion.
    first_size = struct.unpack("<I", original[132:136])[0]
    payload = original[128 : 136 + first_size] + b"\0" * 100000
    compressed = zlib.compress(payload)
    path.write_bytes(
        original[:128]
        + struct.pack("<II", 15, len(compressed))
        + compressed
        + original[136 + first_size :]
    )
    import scipy.io

    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda *a, **k: pytest.fail("No numeric load before expansion cap"),
    )
    with pytest.raises(PreviewRefused, match="expansion"):
        preview(path, limits=replace(PreviewLimits(), max_decompressed_bytes=10000))


def test_hdf_links_and_chunk_expansion_refused(tmp_path):
    path = tmp_path / "linked.hdf5"
    with h5py.File(path, "w") as file:
        file["X"] = h5py.ExternalLink("other.hdf5", "/X")
        file["Y"] = np.ones((2, 1))
    with pytest.raises(ValueError, match="links"):
        preview(path)
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "X", shape=(2, 1), maxshape=(None, 1), chunks=(10000, 1), dtype="f8"
        )
        file["Y"] = np.ones((2, 1))
    with pytest.raises(PreviewRefused, match="chunk"):
        preview(path, limits=replace(PreviewLimits(), max_decompressed_bytes=1000))


def test_zip_traversal_refused(tmp_path):
    path = tmp_path / "traversal.npz"
    np.savez(path, X=np.ones((2, 2)), Y=np.ones((2, 1)))
    with zipfile.ZipFile(path, "a") as file:
        file.writestr("../extra.npy", file.read("X.npy"))
    with pytest.raises(PreviewRefused, match="path"):
        preview(path)


def test_worker_stdout_is_finite_json_only(tmp_path):
    path = tmp_path / "cli.npz"
    np.savez(path, X=np.ones((2, 2)), Y=np.ones((2, 1)))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wavedl.core.dataset_preview",
            "--path",
            str(path),
            "--expected-identity",
            json.dumps(identity(path)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["schema_version"] == 1
    assert not completed.stderr


def test_worker_reports_identity_changed_for_mutated_input(tmp_path):
    path = tmp_path / "identity-change.npz"
    np.savez(path, X=np.ones((2, 2)), Y=np.ones((2, 1)))
    expected = identity(path)
    with path.open("ab") as stream:
        stream.write(b"mutation")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wavedl.core.dataset_preview",
            "--path",
            str(path),
            "--expected-identity",
            json.dumps(expected),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert json.loads(completed.stdout)["code"] == "identity_changed"


def test_content_mutation_during_numeric_read_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "mutation.npz"
    np.savez(path, X=np.ones((2, 2)), Y=np.ones((2, 1)))
    import wavedl.core.dataset_preview as module

    original = module._summaries
    original_bytes = path.read_bytes()
    attempts = []

    def mutate(*args):
        result = original(*args)
        attempts.append(True)
        if os.name == "nt":
            # The held Windows capability denies writes, rather than detecting
            # an allowed POSIX write after numerical decoding.
            with pytest.raises(PermissionError) as denied, path.open("ab") as file:
                file.write(b"changed")
            assert denied.value.errno == 13
            assert getattr(denied.value, "winerror", None) in (None, 32)
            assert path.read_bytes() == original_bytes
        else:
            with path.open("ab") as file:
                file.write(b"changed")
        return result

    monkeypatch.setattr(module, "_summaries", mutate)
    if os.name == "nt":
        result = preview(path)
        assert result["input_shape"] == [2, 2]
        assert result["output_shape"] == [2, 1]
        assert path.read_bytes() == original_bytes
    else:
        with pytest.raises(PreviewRefused, match="changed"):
            preview(path)
    assert attempts == [True]


def test_sample_slice_element_budget_and_scalar_features(tmp_path):
    path = tmp_path / "slices.npz"
    np.savez(path, X=np.arange(1000).reshape(10, 10, 10), Y=np.arange(10))
    result = preview(path, limits=replace(PreviewLimits(), max_preview_elements=2))
    assert len(result["samples"]) == 2
    assert sum(np.asarray(sample["values"]).size for sample in result["samples"]) <= 2
    assert result["samples"][0]["feature_shape"] == [10, 10]


def test_all_nonfinite_histograms_are_empty_and_stats_null(tmp_path):
    path = tmp_path / "nonfinite.npz"
    np.savez(path, X=np.array([[np.nan], [np.inf]]), Y=np.array([[np.inf], [-np.inf]]))
    result = preview(path)
    assert sum(result["input_histograms"][0]["counts"]) == 0
    assert result["input_statistics"]["means"] == [None]
    assert result["scatter"]["y"] == [None, None]


def test_one_sample_and_extreme_finite_numeric_values(tmp_path):
    path = tmp_path / "one.npz"
    np.savez(path, X=np.array([[1.0]]), Y=np.array([[2.0, 3.0]]))
    result = preview(path)
    assert result["sample_indices"] == [0]
    assert result["correlation"]["matrix"] is None
    np.savez(path, X=np.array([[1e300], [-1e300]]), Y=np.ones((2, 1)))
    with pytest.raises(PreviewRefused, match="magnitude"):
        preview(path)


def test_mat4_and_boolean_numeric_samples(tmp_path):
    path = tmp_path / "v4.mat"
    savemat(
        path,
        {"X": np.arange(12).reshape(4, 3), "Y": np.arange(4).reshape(4, 1)},
        format="4",
    )
    assert preview(path)["input_shape"] == [4, 3]
    path = tmp_path / "bool.npz"
    np.savez(
        path, X=np.array([[True, False], [False, True]]), Y=np.array([[True], [False]])
    )
    result = preview(path)
    assert result["samples"][0]["values"] == [1.0, 0.0]
    assert all(type(value) is float for value in result["samples"][0]["values"])
