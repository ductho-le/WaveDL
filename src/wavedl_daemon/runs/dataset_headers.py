"""Metadata-only dataset inspection, isolated from the training loader.

Defaults refuse files over 512 MiB, headers over 16 KiB, over 128 entries,
or arrays declaring over 8 GiB. Administrators may raise these explicit limits.
NPZ inspection reads public NumPy headers, never np.load or member payloads.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import stat
import struct
import sys
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path


class InspectionRefused(ValueError):
    """The dataset cannot safely be inspected under the metadata policy."""


@dataclass(frozen=True)
class InspectionLimits:
    max_file_bytes: int = 512 * 1024 * 1024
    max_header_bytes: int = 16 * 1024
    max_variables: int = 128
    max_declared_bytes: int = 8 * 1024 * 1024 * 1024
    max_dimensions: int = 16
    max_depth: int = 8
    max_name_bytes: int = 256
    max_directory_bytes: int = 1024 * 1024
    max_output_bytes: int = 64 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Inspection limits must be positive integers")


class _HeaderStream:
    def __init__(self, stream, limit):
        self.stream, self.remaining = stream, limit

    def read(self, size=-1):
        if size < 0 or size > self.remaining:
            raise InspectionRefused("Array header exceeds byte limit")
        value = self.stream.read(size)
        self.remaining -= len(value)
        return value


class _ZipStream:
    """Prevent a concurrent ZIP mutation from bypassing directory read bounds."""

    def __init__(self, stream, limit):
        self.stream, self.limit = stream, max(limit, 65557)

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()

    def seekable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            before = self.stream.tell()
            end = self.stream.seek(0, 2)
            self.stream.seek(before)
            size = end - before
        if size > self.limit:
            raise InspectionRefused("ZIP read exceeds metadata byte limit")
        return self.stream.read(size)


def _metadata(name, shape, dtype, limits):
    if not isinstance(name, str) or len(name.encode()) > limits.max_name_bytes:
        raise InspectionRefused("Variable name exceeds limit")
    if shape is None:
        shape = ()
    shape = tuple(shape)
    if len(shape) > limits.max_dimensions or any(
        type(size) is not int or not 0 <= size <= limits.max_declared_bytes
        for size in shape
    ):
        raise InspectionRefused("Invalid or excessive array dimensions")
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.kind not in {"b", "i", "u", "f", "c", "S", "U", "m", "M"}
    ):
        raise InspectionRefused("Object or structured dtype inspection is unsupported")
    size = math.prod(shape) * dtype.itemsize
    if size > limits.max_declared_bytes:
        raise InspectionRefused("Declared array size exceeds limit")
    return {
        "name": name,
        "shape": list(shape),
        "dtype": str(dtype),
        "sample_count": shape[0] if shape else None,
        "declared_bytes": size,
    }


def _npy_header(stream, name, limits):
    from numpy.lib import format

    bounded = _HeaderStream(stream, limits.max_header_bytes)
    version = format.read_magic(bounded)
    readers = {
        (1, 0): format.read_array_header_1_0,
        (2, 0): format.read_array_header_2_0,
    }
    if version not in readers:
        raise InspectionRefused("Unsupported NPY header version")
    shape, _, dtype = readers[version](bounded, max_header_size=limits.max_header_bytes)
    return _metadata(name, shape, dtype, limits)


def _zip_directory(stream, limits):
    """Bound central-directory allocation before ZipFile materializes entries."""
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(max(0, size - 65557))
    tail = stream.read(65557)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < 22:
        raise InspectionRefused("Invalid ZIP directory")
    _, disk, directory_disk, count_disk, count, directory_size, offset, comment = (
        struct.unpack("<4s4H2IH", tail[position : position + 22])
    )
    if (
        disk
        or directory_disk
        or count_disk != count
        or count == 65535
        or directory_size == 0xFFFFFFFF
        or offset == 0xFFFFFFFF
        or position + 22 + comment != len(tail)
        or count > limits.max_variables
        or directory_size > limits.max_directory_bytes
    ):
        raise InspectionRefused(
            "ZIP directory exceeds inspection limits or is unsupported"
        )
    stream.seek(0)


def _npz(stream, limits):
    _zip_directory(stream, limits)
    result, names = [], set()
    with zipfile.ZipFile(_ZipStream(stream, limits.max_directory_bytes)) as archive:
        members = archive.infolist()
        if len(members) > limits.max_variables:
            raise InspectionRefused("Too many archive members")
        for member in members:
            if (
                not member.filename.endswith(".npy")
                or member.filename in names
                or member.flag_bits & 1
                or member.compress_type
                not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            ):
                raise InspectionRefused("Unsupported archive member")
            names.add(member.filename)
            if len(member.filename.encode()) > limits.max_name_bytes:
                raise InspectionRefused("Archive member name exceeds limit")
            with archive.open(member) as inner:
                result.append(_npy_header(inner, member.filename[:-4], limits))
            if (
                sum(item["declared_bytes"] for item in result)
                > limits.max_declared_bytes
            ):
                raise InspectionRefused("Total declared array size exceeds limit")
    return result


def npz_member_headers(
    path: str | Path, *, limits: InspectionLimits = InspectionLimits()
) -> list[dict]:
    with _open_dataset(path, limits) as stream:
        return _npz(stream, limits)


def _hdf(stream, limits, *, matlab=False):
    import h5py

    result, seen, nodes = [], set(), 0
    with h5py.File(stream, "r") as file:

        def visit(group, depth):
            nonlocal nodes
            if depth > limits.max_depth:
                raise InspectionRefused("HDF group depth exceeds limit")
            identity = hash(group.id)
            if identity in seen:
                return
            seen.add(identity)
            if len(group) > limits.max_variables - nodes:
                raise InspectionRefused("Too many HDF objects")
            for name in group:
                nodes += 1
                if nodes > limits.max_variables:
                    raise InspectionRefused("Too many HDF objects")
                if len(name.encode()) > limits.max_name_bytes:
                    raise InspectionRefused("HDF name exceeds limit")
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise InspectionRefused(
                        "External and soft HDF links are unsupported"
                    )
                item = group[name]
                if isinstance(item, h5py.Group):
                    if matlab and {"data", "ir", "jc"}.issubset(item):
                        members = []
                        for key in ("data", "ir", "jc"):
                            if not isinstance(
                                item.get(key, getlink=True), h5py.HardLink
                            ):
                                raise InspectionRefused(
                                    "External sparse MAT links are unsupported"
                                )
                            member = item[key]
                            if (
                                not isinstance(member, h5py.Dataset)
                                or member.is_virtual
                                or member.external
                            ):
                                raise InspectionRefused("Invalid sparse MAT metadata")
                            members.append(member)
                        if "MATLAB_sparse" not in item.attrs:
                            raise InspectionRefused(
                                "Sparse MAT shape requires a bounded row attribute"
                            )
                        attribute = item.attrs.get_id("MATLAB_sparse")
                        if (
                            math.prod(attribute.shape) != 1
                            or attribute.dtype.itemsize > 8
                            or attribute.dtype.kind not in {"i", "u"}
                        ):
                            raise InspectionRefused("Invalid sparse MAT row attribute")
                        rows = int(item.attrs["MATLAB_sparse"])
                        columns = math.prod(members[2].shape) - 1
                        result.append(
                            _metadata(
                                item.name.lstrip("/"),
                                (columns, rows),
                                members[0].dtype,
                                limits,
                            )
                        )
                    else:
                        visit(item, depth + 1)
                elif isinstance(item, h5py.Dataset):
                    if item.is_virtual or item.external:
                        raise InspectionRefused(
                            "External and virtual HDF datasets are unsupported"
                        )
                    result.append(
                        _metadata(
                            item.name.lstrip("/"),
                            item.shape[::-1] if matlab else item.shape,
                            item.dtype,
                            limits,
                        )
                    )

        visit(file, 0)
    return result


def _mat(stream, limits):
    import numpy as np
    from scipy.io import whosmat

    metadata_stream, storage = _mat_metadata_snapshot(stream, limits)
    variables = whosmat(metadata_stream, appendmat=False)
    if len(variables) > limits.max_variables:
        raise InspectionRefused("Too many MAT variables")
    types = {
        "single": "float32",
        "double": "float64",
        "logical": "bool",
        **{
            name: name
            for name in (
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
            )
        },
    }
    result = []
    for index, (name, shape, kind) in enumerate(variables):
        if kind not in types:
            raise InspectionRefused("Unsupported MAT variable class")
        dtype = np.dtype(types[kind])
        complex_value, storage_dtype = storage[index]
        if storage_dtype is not None:
            dtype = np.dtype(storage_dtype)
        if complex_value:
            if dtype.kind != "f":
                raise InspectionRefused("Unsupported complex MAT variable class")
            dtype = np.dtype("complex64" if dtype.itemsize == 4 else "complex128")
        result.append(_metadata(name, shape, dtype, limits))
    return result


class _MatPrefix:
    """Decode only requested bounded header bytes, never an array payload."""

    def __init__(self, stream, size, limit, *, compressed):
        self.stream, self.remaining, self.limit = stream, size, limit
        self.compressed = zlib.decompressobj() if compressed else None
        self.pending = b""
        self.used = 0

    def read(self, count):
        if count < 0 or self.used + count > self.limit:
            raise InspectionRefused("MAT header work exceeds byte limit")
        self.used += count
        if self.compressed is None:
            if count > self.remaining:
                raise InspectionRefused("Truncated MAT header")
            self.remaining -= count
            result = self.stream.read(count)
        else:
            result = bytearray()
            while len(result) < count:
                if not self.pending:
                    if not self.remaining:
                        raise InspectionRefused("Truncated compressed MAT header")
                    self.pending = self.stream.read(min(self.remaining, 256))
                    self.remaining -= len(self.pending)
                    if not self.pending:
                        raise InspectionRefused("Truncated compressed MAT header")
                try:
                    result.extend(
                        self.compressed.decompress(self.pending, count - len(result))
                    )
                except zlib.error:
                    raise InspectionRefused("Invalid compressed MAT header") from None
                self.pending = self.compressed.unconsumed_tail
                if self.compressed.eof and len(result) < count:
                    raise InspectionRefused("Truncated compressed MAT header")
            result = bytes(result)
        if len(result) != count:
            raise InspectionRefused("Truncated MAT header")
        return result


def _mat_field(reader, endian, expected_type, maximum):
    tag = reader.read(8)
    word, count = struct.unpack(endian + "II", tag)
    if word > 65535:
        kind, small_count = struct.unpack(endian + "HH", tag[:4])
        if kind != expected_type or not 1 <= small_count <= min(4, maximum):
            raise InspectionRefused("Invalid small MAT header field")
        return tag[4 : 4 + small_count], tag
    if word != expected_type or count > maximum:
        raise InspectionRefused("MAT header field exceeds limit")
    value = reader.read(count)
    padding = reader.read((-count) % 8)
    return value, tag + value + padding


def _mat_metadata_snapshot(stream, limits):
    """Preflight Level-5 header fields before invoking public scipy.whosmat.

    A metadata-only snapshot prevents a mutated source or hostile dimension/name
    count from making SciPy inflate or allocate numerical data. The snapshot
    contains no payload bytes. Version-4 files use their bounded fixed header.
    """
    stream.seek(0, 2)
    end = stream.tell()
    stream.seek(0)
    header = stream.read(128)
    if len(header) < 128 or header[126:128] not in {b"IM", b"MI"}:
        return _mat4_metadata_snapshot(stream, end, limits)
    endian = "<" if header[126:128] == b"IM" else ">"
    if struct.unpack(endian + "H", header[124:126])[0] != 256:
        raise InspectionRefused("Unsupported MAT metadata version")
    output = bytearray(header)
    complex_flags = []
    while stream.tell() < end:
        if len(complex_flags) >= limits.max_variables:
            raise InspectionRefused("Too many MAT variables")
        tag = stream.read(8)
        if len(tag) != 8:
            raise InspectionRefused("Truncated MAT variable tag")
        kind, size = struct.unpack(endian + "II", tag)
        next_position = stream.tell() + size
        if kind not in {14, 15} or next_position > end:
            raise InspectionRefused("Invalid MAT variable extent")
        reader = _MatPrefix(
            stream, size, limits.max_header_bytes, compressed=kind == 15
        )
        if kind == 15:
            inner_type, _ = struct.unpack(endian + "II", reader.read(8))
            if inner_type != 14:
                raise InspectionRefused("Compressed MAT member is not a matrix")
        flags, raw_flags = _mat_field(reader, endian, 6, 8)
        if len(flags) != 8:
            raise InspectionRefused("Invalid MAT array flags")
        flag = struct.unpack(endian + "I", flags[:4])[0]
        if flag & 255 not in range(6, 16):
            raise InspectionRefused("Unsupported MAT variable class")
        dims, raw_dims = _mat_field(reader, endian, 5, limits.max_dimensions * 4)
        if len(dims) % 4 or not dims:
            raise InspectionRefused("Invalid MAT dimensions")
        shape = struct.unpack(endian + "i" * (len(dims) // 4), dims)
        if any(size < 0 or size > limits.max_declared_bytes for size in shape):
            raise InspectionRefused("Invalid MAT dimensions")
        name, raw_name = _mat_field(reader, endian, 1, limits.max_name_bytes)
        if b"\0" in name:
            raise InspectionRefused("Invalid MAT variable name")
        safe_header = raw_flags + raw_dims + raw_name
        output.extend(struct.pack(endian + "II", 14, len(safe_header)))
        output.extend(safe_header)
        complex_flags.append((bool(flag & 0x800), None))
        stream.seek(next_position)
    return io.BytesIO(output), complex_flags


class _Mat4Snapshot:
    """Sparse virtual metadata file; numerical payload ranges are never read."""

    def __init__(self, blocks, size, limit):
        self.blocks, self.size, self.remaining = blocks, size, limit
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, self.size)[whence]
        position = base + offset
        if position < 0:
            raise InspectionRefused("Invalid MAT metadata seek")
        self.position = position
        return position

    def read(self, count=-1):
        if count < 0 or count > self.remaining:
            raise InspectionRefused("MAT header work exceeds byte limit")
        if self.position >= self.size:
            return b""
        for start, content in self.blocks:
            if start <= self.position < start + len(content):
                result = content[self.position - start : self.position - start + count]
                self.position += len(result)
                self.remaining -= len(result)
                return result
        raise InspectionRefused("MAT metadata inspection attempted a payload read")


def _mat4_metadata_snapshot(stream, end, limits):
    # scipy.whosmat is safe only after these fixed-size variable headers and name
    # lengths are checked. Retain a tiny virtual file, not untrusted data reads.
    stream.seek(0)
    complex_flags = []
    blocks, virtual_position = [], 0
    endian = None
    while stream.tell() < end:
        if len(complex_flags) >= limits.max_variables:
            raise InspectionRefused("Too many MAT variables")
        header = stream.read(20)
        if len(header) != 20:
            raise InspectionRefused("Truncated MAT v4 header")
        if endian is None:
            endian = "<" if 0 <= struct.unpack("<i", header[:4])[0] <= 4052 else ">"
        kind, rows, columns, imaginary, name_size = struct.unpack(endian + "5i", header)
        precision, matrix_class = (kind // 10) % 10, kind % 10
        if (
            (kind // 1000) not in {0, 1}
            or precision not in range(6)
            or matrix_class != 0
            or imaginary not in {0, 1}
            or rows < 0
            or columns < 0
            or not 1 <= name_size <= limits.max_name_bytes + 1
        ):
            raise InspectionRefused("Unsupported or excessive MAT v4 header")
        name = stream.read(name_size)
        if len(name) != name_size or not name.endswith(b"\0"):
            raise InspectionRefused("Invalid MAT v4 name")
        item_size = (8, 4, 4, 2, 2, 1)[precision]
        payload_size = rows * columns * item_size * (1 + imaginary)
        if (
            payload_size > limits.max_declared_bytes
            or stream.tell() + payload_size > end
        ):
            raise InspectionRefused("MAT v4 extent exceeds limit")
        blocks.append((virtual_position, header + name))
        virtual_position += len(header) + len(name) + payload_size
        stream.seek(payload_size, 1)
        complex_flags.append(
            (bool(imaginary), ("f8", "f4", "i4", "i2", "u2", "u1")[precision])
        )
    return _Mat4Snapshot(
        blocks, virtual_position, limits.max_header_bytes * limits.max_variables
    ), complex_flags


def _held_identity(stream, metadata):
    """Match authority fingerprints using this already-open file, not its path.

    Python 3.12+ Windows fstat may expose FileIdInfo's wider volume/file IDs.
    Dataset authority uses BY_HANDLE_FILE_INFORMATION, so compare that same
    representation without truncating IDs or weakening the identity fence.
    """
    if os.name == "nt":
        import msvcrt

        from wavedl.core.safe_files import _windows_api

        # Borrow the CRT-owned HANDLE; stream remains its sole close owner.
        native = _windows_api().info(msvcrt.get_osfhandle(stream.fileno()))
        ticks = native.write.dwHighDateTime << 32 | native.write.dwLowDateTime
        return {
            "st_dev": int(native.volume),
            "st_ino": int(native.index_high << 32 | native.index_low),
            "st_size": int(native.size_high << 32 | native.size_low),
            "st_mtime_ns": (ticks - 116444736000000000) * 100,
        }
    return {
        name: int(getattr(metadata, name))
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    }


def _open_dataset(path, limits, expected_identity=None):
    path = Path(path)
    if not path.is_absolute():
        raise InspectionRefused("Dataset path must be absolute")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    stream = os.fdopen(descriptor, "rb")
    metadata = os.fstat(stream.fileno())
    if expected_identity is not None:
        try:
            actual_identity = _held_identity(stream, metadata)
        except BaseException:
            stream.close()
            raise
        if (
            type(expected_identity) is not dict
            or set(expected_identity) != set(actual_identity)
            or expected_identity != actual_identity
        ):
            stream.close()
            raise InspectionRefused("Dataset identity changed")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limits.max_file_bytes:
        stream.close()
        raise InspectionRefused("Dataset must be a bounded regular file")
    return stream


def _is_hdf(stream):
    for offset in (0, 512):
        stream.seek(offset)
        if stream.read(8) == b"\x89HDF\r\n\x1a\n":
            return True
    return False


def inspect_dataset(
    path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
    expected_identity=None,
) -> dict:
    suffix = Path(path).suffix.lower()
    with _open_dataset(path, limits, expected_identity) as stream:
        if suffix == ".npz":
            variables = _npz(stream, limits)
        elif suffix == ".npy":
            variables = [_npy_header(stream, "array", limits)]
        elif suffix in {".h5", ".hdf5"}:
            variables = _hdf(stream, limits)
        elif suffix == ".mat":
            hdf = _is_hdf(stream)
            stream.seek(0)
            variables = (
                _hdf(stream, limits, matlab=True) if hdf else _mat(stream, limits)
            )
        else:
            raise InspectionRefused("Unsupported dataset format")
    if sum(item["declared_bytes"] for item in variables) > limits.max_declared_bytes:
        raise InspectionRefused("Total declared array size exceeds limit")
    result = {"schema_version": 1, "format": suffix[1:], "variables": variables}
    if len(json.dumps(result, allow_nan=False).encode()) > limits.max_output_bytes:
        raise InspectionRefused("Metadata output exceeds limit")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-identity")
    for name in asdict(InspectionLimits()):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(InspectionLimits(), name),
        )
    args = vars(parser.parse_args(argv))
    path = args.pop("path")
    expected_identity = args.pop("expected_identity")
    try:
        result = inspect_dataset(
            path,
            limits=InspectionLimits(**args),
            expected_identity=(
                json.loads(expected_identity) if expected_identity is not None else None
            ),
        )
    except Exception:
        # The worker boundary deliberately hides library traces and local paths.
        print('{"error":"Dataset inspection refused"}')
        return 1
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
