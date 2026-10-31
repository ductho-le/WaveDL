"""Metadata inspection must never materialize numerical payloads."""

import io
import json
import os
import struct
import subprocess
import sys
import tracemalloc
import zipfile
import zlib
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from wavedl.core.dataset_headers import (
    InspectionLimits,
    InspectionRefused,
    inspect_dataset,
    npz_member_headers,
)


def header(shape=(3, 4), dtype="<f4", version=1):
    stream = io.BytesIO()
    writer = (
        np.lib.format.write_array_header_1_0
        if version == 1
        else np.lib.format.write_array_header_2_0
    )
    writer(stream, {"descr": dtype, "fortran_order": False, "shape": shape})
    return stream.getvalue()


def archive(path, contents):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as file:
        for name, data in contents:
            file.writestr(name, data)
    return path


@pytest.mark.parametrize("version", [1, 2])
def test_public_headers_without_materialization(tmp_path, monkeypatch, version):
    path = archive(
        tmp_path / "input.npz",
        [("inputs.npy", header(version=version) + b"\0" * 2000000)],
    )
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("np.load must not run")
    )
    monkeypatch.setattr(
        np.lib.format,
        "read_array",
        lambda *args, **kwargs: pytest.fail("array must not materialize"),
    )
    actual_reads = []
    original = zipfile.ZipExtFile.read

    def read(self, size=-1):
        actual_reads.append(size)
        return original(self, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", read)
    result = npz_member_headers(path)
    assert result[0]["shape"] == [3, 4]
    assert result[0]["dtype"] == "float32"
    assert sum(actual_reads) < 16384 and all(size >= 0 for size in actual_reads)


def test_npy_headers_do_not_require_payload(tmp_path):
    path = tmp_path / "empty.npy"
    path.write_bytes(header())
    assert inspect_dataset(path)["variables"][0]["sample_count"] == 3


def test_enormous_declared_zip_array_rejected(tmp_path):
    path = archive(tmp_path / "bomb.npz", [("inputs.npy", header((2**50, 2**50)))])
    with pytest.raises(InspectionRefused, match="array"):
        inspect_dataset(path)


def test_header_read_size_checked_before_requested_allocation(tmp_path):
    path = tmp_path / "bad.npy"
    path.write_bytes(b"\x93NUMPY\x02\x00" + (2**31).to_bytes(4, "little"))
    with pytest.raises(InspectionRefused, match="header"):
        inspect_dataset(path)


@pytest.mark.parametrize("dtype", ["|O", [("a", "<f4")]])
def test_unsafe_or_structured_dtype_is_refused_without_unpickle(tmp_path, dtype):
    path = tmp_path / "objects.npy"
    path.write_bytes(header(dtype=dtype))
    with pytest.raises(InspectionRefused, match="dtype"):
        inspect_dataset(path)


def test_file_member_and_directory_bounds(tmp_path):
    path = archive(
        tmp_path / "input.npz", [(f"{number}.npy", header()) for number in range(3)]
    )
    with pytest.raises(InspectionRefused):
        inspect_dataset(path, limits=replace(InspectionLimits(), max_file_bytes=4))
    with pytest.raises(InspectionRefused):
        inspect_dataset(path, limits=replace(InspectionLimits(), max_variables=2))
    with pytest.raises(InspectionRefused):
        inspect_dataset(path, limits=replace(InspectionLimits(), max_directory_bytes=4))


@pytest.mark.parametrize("suffix", [".h5", ".hdf5"])
def test_hdf_aliases_metadata_only(tmp_path, monkeypatch, suffix):
    path = tmp_path / ("data" + suffix)
    with h5py.File(path, "w") as file:
        file.create_dataset("nested/inputs", shape=(20, 3), dtype="f4")
    monkeypatch.setattr(
        h5py.Dataset, "__getitem__", lambda *args: pytest.fail("No payload reads")
    )
    result = inspect_dataset(path)
    assert result["variables"][0]["shape"] == [20, 3]


@pytest.mark.parametrize("kind", ["external", "soft", "storage", "virtual"])
def test_hdf_external_access_is_refused_before_dereference(tmp_path, kind):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as file:
        if kind == "external":
            file["inputs"] = h5py.ExternalLink("does-not-exist.h5", "/inputs")
        elif kind == "soft":
            file["inputs"] = h5py.SoftLink("/missing")
        elif kind == "storage":
            file.create_dataset(
                "inputs", shape=(3,), dtype="f4", external=[("external.bin", 0, 12)]
            )
        else:
            layout = h5py.VirtualLayout(shape=(3,), dtype="f4")
            layout[:] = h5py.VirtualSource("external.h5", "inputs", shape=(3,))
            file.create_virtual_dataset("inputs", layout)
    with pytest.raises(InspectionRefused, match=r"[Ee]xternal"):
        inspect_dataset(path)


def test_hdf_depth_nodes_and_cycle_bounds(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("a/b/c/inputs", shape=(2,), dtype="f4")
        file["loop"] = file["a"]
    with pytest.raises(InspectionRefused, match="depth"):
        inspect_dataset(path, limits=replace(InspectionLimits(), max_depth=1))
    assert len(inspect_dataset(path)["variables"]) == 1


def test_mat_uses_whosmat_not_loadmat(tmp_path, monkeypatch):
    import scipy.io

    path = tmp_path / "data.mat"
    savemat(path, {"inputs": np.zeros((5, 2), dtype="float32")}, do_compression=True)
    monkeypatch.setattr(
        scipy.io, "loadmat", lambda *args, **kwargs: pytest.fail("No MAT payload loads")
    )
    assert inspect_dataset(path)["variables"][0]["shape"] == [5, 2]


def test_cli_corrupt_header_reports_generic_json_no_trace(tmp_path):
    path = tmp_path / "private-secret.npy"
    path.write_bytes(b"broken")
    result = subprocess.run(
        [sys.executable, "-m", "wavedl.core.dataset_headers", "--path", str(path)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"error": "Dataset inspection refused"}
    assert result.stderr == b"" and str(path).encode() not in result.stdout


def test_import_stays_lightweight_without_site_packages():
    # -S bypasses editable-install .pth files, so expose only the checkout source.
    source_root = Path(__file__).resolve().parents[2] / "src"
    code = "import sys; import wavedl.core.dataset_headers; assert not any(x in sys.modules for x in ['numpy','torch','h5py','scipy','PySide6'])"
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        check=True,
        capture_output=True,
        timeout=10,
        env=os.environ | {"PYTHONPATH": str(source_root)},
    )


def test_total_declared_archive_limit_applies_to_public_headers(tmp_path):
    path = archive(tmp_path / "sum.npz", [("a.npy", header()), ("b.npy", header())])
    with pytest.raises(InspectionRefused, match="Total declared"):
        npz_member_headers(
            path, limits=replace(InspectionLimits(), max_declared_bytes=64)
        )


def test_hdf_member_count_checked_before_iteration(tmp_path):
    path = tmp_path / "many.h5"
    with h5py.File(path, "w") as file:
        for index in range(4):
            file.create_dataset(str(index), shape=(2,), dtype="f4")
    with pytest.raises(InspectionRefused, match="Too many"):
        inspect_dataset(path, limits=replace(InspectionLimits(), max_variables=3))


def test_mat_cumulative_header_work_is_bounded(tmp_path):
    path = tmp_path / "data.mat"
    savemat(path, {"inputs": np.zeros((5, 2), dtype="float32")})
    with pytest.raises(InspectionRefused, match="MAT header work"):
        inspect_dataset(
            path,
            limits=replace(InspectionLimits(), max_header_bytes=4, max_variables=1),
        )


def test_compressed_mat_large_zero_payload_does_not_inflate_for_metadata(tmp_path):
    path = tmp_path / "compressed.mat"
    array = np.zeros((4096, 1024), dtype="float64")
    savemat(path, {"inputs": array}, do_compression=True)
    del array
    # Imports and fixture creation precede allocation accounting; this measures
    # inspection itself, including SciPy's compressed header stream buffers.
    tracemalloc.start()
    try:
        result = inspect_dataset(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["variables"][0]["shape"] == [4096, 1024]
    assert result["variables"][0]["declared_bytes"] == 32 * 1024 * 1024
    assert peak < 2 * 1024 * 1024


@pytest.mark.parametrize("compressed", [True, False])
def test_mat_short_reads_preserve_variable_offsets_and_long_names(tmp_path, compressed):
    path = tmp_path / "variables.mat"
    name = "a" * 200
    savemat(
        path,
        {name: np.zeros((8, 4)), "outputs": np.ones((8, 2))},
        do_compression=compressed,
    )
    variables = inspect_dataset(path)["variables"]
    assert [(item["name"], item["shape"]) for item in variables] == [
        (name, [8, 4]),
        ("outputs", [8, 2]),
    ]


@pytest.mark.parametrize("field,count", [("dimensions", 2**30), ("name", 2**30)])
def test_hostile_compressed_mat_header_is_rejected_before_scipy(
    tmp_path, monkeypatch, field, count
):
    import scipy.io

    source = tmp_path / "source.mat"
    savemat(source, {"X": np.zeros((2, 2))})
    header = source.read_bytes()[:128]
    flags = struct.pack("<IIII", 6, 8, 6, 0)
    dims = struct.pack("<IIII", 5, 8, 2, 2)
    if field == "dimensions":
        metadata = flags + struct.pack("<II", 5, count)
    else:
        metadata = flags + dims + struct.pack("<II", 1, count)
    matrix = struct.pack("<II", 14, len(metadata)) + metadata
    compressed = zlib.compress(matrix)
    path = tmp_path / "hostile.mat"
    path.write_bytes(header + struct.pack("<II", 15, len(compressed)) + compressed)
    monkeypatch.setattr(
        scipy.io,
        "whosmat",
        lambda *args, **kwargs: pytest.fail("Unsafe header reached SciPy"),
    )
    with pytest.raises(InspectionRefused, match="header field"):
        inspect_dataset(path)


@pytest.mark.parametrize("version", ["4", "5"])
def test_mat_complex_metadata_counts_actual_storage(tmp_path, version):
    path = tmp_path / "complex.mat"
    savemat(path, {"X": np.ones((8, 4), dtype="complex128")}, format=version)
    variable = inspect_dataset(path)["variables"][0]
    assert variable["shape"] == [8, 4]
    assert variable["dtype"] == "complex128"
    assert variable["declared_bytes"] == 8 * 4 * 16


@pytest.mark.parametrize("userblock", [0, 512])
def test_mat_v73_dense_shape_matches_training_transpose(tmp_path, userblock):
    path = tmp_path / "v73.mat"
    with h5py.File(path, "w", userblock_size=userblock) as file:
        file.create_dataset("inputs", shape=(32, 20), dtype="float32")
        file.create_dataset("outputs", shape=(1, 20), dtype="float64")
    variables = inspect_dataset(path)["variables"]
    assert [
        (item["name"], item["shape"], item["sample_count"]) for item in variables
    ] == [("inputs", [20, 32], 20), ("outputs", [20, 1], 20)]


def test_mat_v73_sparse_shape_uses_attributes_not_numerical_arrays(
    tmp_path, monkeypatch
):
    path = tmp_path / "sparse.mat"
    with h5py.File(path, "w") as file:
        group = file.create_group("inputs")
        group.attrs["MATLAB_sparse"] = 32
        group.create_dataset("data", shape=(8,), dtype="float32")
        group.create_dataset("ir", shape=(8,), dtype="int64")
        group.create_dataset("jc", shape=(21,), dtype="int64")
    monkeypatch.setattr(
        h5py.Dataset,
        "__getitem__",
        lambda *args, **kwargs: pytest.fail("Numerical sparse data loaded"),
    )
    variable = inspect_dataset(path)["variables"][0]
    assert variable["shape"] == [20, 32]
    assert variable["sample_count"] == 20
    assert variable["declared_bytes"] == 20 * 32 * 4


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "uint8"])
def test_mat_v4_reports_storage_dtype_not_generic_double_class(tmp_path, dtype):
    path = tmp_path / "v4.mat"
    savemat(path, {"X": np.zeros((8, 4), dtype=dtype)}, format="4")
    variable = inspect_dataset(path)["variables"][0]
    assert variable["dtype"] == dtype
    assert variable["declared_bytes"] == 8 * 4 * np.dtype(dtype).itemsize


def test_expected_identity_is_checked_before_reading_metadata(tmp_path, monkeypatch):
    from wavedl.core import dataset_headers

    path = tmp_path / "selected.npy"
    np.save(path, np.zeros((3, 2), dtype="float32"))
    info = path.stat()
    expected = {
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_size": info.st_size,
        "st_mtime_ns": info.st_mtime_ns,
    }
    if os.name == "nt":
        with path.open("rb") as stream:
            expected = dataset_headers._held_identity(stream, os.fstat(stream.fileno()))
    assert inspect_dataset(path, expected_identity=expected)["format"] == "npy"
    monkeypatch.setattr(
        dataset_headers,
        "_npy_header",
        lambda *args: pytest.fail("Read metadata from an unverified file"),
    )
    with pytest.raises(InspectionRefused, match="identity changed"):
        inspect_dataset(
            path, expected_identity=expected | {"st_ino": expected["st_ino"] + 1}
        )


@pytest.mark.parametrize("field", [None, "st_dev", "st_ino", "st_size", "st_mtime_ns"])
def test_windows_header_identity_uses_borrowed_native_handle_not_wider_fstat(
    tmp_path, monkeypatch, field
):
    from types import SimpleNamespace

    from wavedl.core import dataset_headers, safe_files

    path = tmp_path / "native-fingerprint.npy"
    np.save(path, np.zeros((3, 2), dtype="float32"))
    info = path.stat()
    ticks = 116444736000000000 + 12345
    native = SimpleNamespace(
        volume=17,
        index_high=2,
        index_low=19,
        size_high=0,
        size_low=info.st_size,
        write=SimpleNamespace(
            dwHighDateTime=ticks >> 32, dwLowDateTime=ticks & 0xFFFFFFFF
        ),
    )
    handles = []

    def native_info(handle):
        handles.append(handle)
        assert handle == 777
        return native

    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: 777)
    )
    monkeypatch.setattr(
        safe_files, "_windows_api", lambda: SimpleNamespace(info=native_info)
    )
    monkeypatch.setattr(
        dataset_headers, "os", SimpleNamespace(**(vars(os) | {"name": "nt"}))
    )
    expected = {
        "st_dev": 17,
        "st_ino": (2 << 32) | 19,
        "st_size": info.st_size,
        "st_mtime_ns": 1234500,
    }
    # Deliberately unrelated fstat identity must not replace native authority.
    assert expected["st_ino"] != info.st_ino
    if field:
        monkeypatch.setattr(
            dataset_headers,
            "_npy_header",
            lambda *args: pytest.fail("Read unverified header"),
        )
        with pytest.raises(InspectionRefused, match="identity changed"):
            inspect_dataset(
                path, expected_identity=expected | {field: expected[field] + 1}
            )
    else:
        assert inspect_dataset(path, expected_identity=expected)["format"] == "npy"
    assert handles == [777]


def test_native_header_identity_failure_closes_owned_stream_without_closing_borrowed_handle(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from wavedl.core import dataset_headers, safe_files

    path = tmp_path / "native-query-failure.npy"
    np.save(path, np.zeros((3, 2)))
    streams = []
    original_fdopen = os.fdopen

    def fdopen(fd, mode):
        stream = original_fdopen(fd, mode)
        streams.append(stream)
        return stream

    def refused(handle):
        assert handle == 777
        raise OSError("Native identity query failed")

    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: 777)
    )
    monkeypatch.setattr(
        safe_files, "_windows_api", lambda: SimpleNamespace(info=refused)
    )
    monkeypatch.setattr(
        dataset_headers,
        "os",
        SimpleNamespace(**(vars(os) | {"name": "nt", "fdopen": fdopen})),
    )
    with pytest.raises(OSError, match="Native identity query failed"):
        inspect_dataset(path, expected_identity={})
    assert len(streams) == 1 and streams[0].closed
