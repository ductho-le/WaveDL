"""Bounded, no-follow reads and anchored directory traversal.

Windows opens reparse points themselves and rejects them. Directory ancestry is
held without write/delete sharing: pathname enumeration cannot be redirected by
renaming an ancestor or replacing its reparse data while the handles are alive.
No native handle is exposed to callers and no weaker pathname fallback is used.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


class UnsupportedSafeRead(OSError):
    """The filesystem cannot supply the required safe native primitives."""


class UnsafeOwnership(ValueError):
    """An opened object is not private to the current user."""


@dataclass(frozen=True)
class FileMetadata:
    st_size: int
    st_uid: int | None = None
    st_mode: int | None = None
    owner_sid: str | None = None

    @property
    def size(self):
        return self.st_size

    @property
    def uid(self):
        return self.st_uid

    @property
    def mode(self):
        return self.st_mode


def _private_owner(api, handle, *, confidential=False):
    if api:
        from .windows_ownership import verify_windows_private_owner

        verify_windows_private_owner(api, handle, confidential=confidential)
        return
    info = os.fstat(handle)
    forbidden = 0o077 if confidential else 0o022
    if info.st_uid != os.getuid() or info.st_mode & forbidden:
        raise UnsafeOwnership(
            "Object must be current-user owned with appropriately private permissions"
        )


def _identity(api, handle):
    if api:
        info = api.info(handle)
        return info.volume, info.index_high, info.index_low
    info = os.fstat(handle)
    return info.st_dev, info.st_ino


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    is_directory: bool
    is_regular: bool


def _member(name: str) -> str:
    if not name or name in (".", "..") or any(c in name for c in "/\\:\0"):
        raise ValueError("Expected a single ordinary directory member")
    return name


def _posix_open(path, *, directory=False, dir_fd=None):
    if not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_NONBLOCK")):
        raise UnsupportedSafeRead(
            errno.ENOTSUP, "Safe reads require no-follow/nonblocking support"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if directory:
        if not hasattr(os, "O_DIRECTORY"):
            raise UnsupportedSafeRead(errno.ENOTSUP, "No safe directory support")
        flags |= os.O_DIRECTORY
    try:
        return os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        if (directory or dir_fd is not None) and exc.errno in (
            errno.ELOOP,
            errno.ENOTDIR,
        ):
            raise ValueError("Member is not an ordinary no-follow file") from exc
        raise


def _read_descriptor(descriptor: int, limit: int) -> bytes:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("Metadata must be a bounded regular file")
    with os.fdopen(descriptor, "rb", closefd=False) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Metadata exceeds byte bound")
    return raw


def _posix_reopen(path: Path, *, directory=False):
    """Revalidate a configured path without following any replaced ancestor."""
    parts = path.absolute().parts
    descriptor = _posix_open(parts[0], directory=True)
    try:
        for index, part in enumerate(parts[1:], 1):
            child = _posix_open(
                part,
                directory=directory or index < len(parts) - 1,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        result, descriptor = descriptor, None
        return result
    finally:
        if descriptor is not None:
            os.close(descriptor)


class _Win32:
    """Lazy stdlib Win32 binding, deliberately separate for contract tests."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.types = wintypes
        self.dll = ctypes.WinDLL("kernel32", use_last_error=True)

        class Information(ctypes.Structure):
            _fields_: ClassVar = [
                ("attributes", wintypes.DWORD),
                ("creation", wintypes.FILETIME),
                ("access", wintypes.FILETIME),
                ("write", wintypes.FILETIME),
                ("volume", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD),
                ("index_low", wintypes.DWORD),
            ]

        self.Information = Information
        signatures = {
            "CreateFileW": (
                [
                    wintypes.LPCWSTR,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.LPVOID,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.HANDLE,
                ],
                wintypes.HANDLE,
            ),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "GetCurrentProcess": ([], wintypes.HANDLE),
            "DuplicateHandle": (
                [
                    wintypes.HANDLE,
                    wintypes.HANDLE,
                    wintypes.HANDLE,
                    ctypes.POINTER(wintypes.HANDLE),
                    wintypes.DWORD,
                    wintypes.BOOL,
                    wintypes.DWORD,
                ],
                wintypes.BOOL,
            ),
            "FlushFileBuffers": ([wintypes.HANDLE], wintypes.BOOL),
            "GetFileType": ([wintypes.HANDLE], wintypes.DWORD),
            "GetFileInformationByHandle": (
                [wintypes.HANDLE, ctypes.POINTER(Information)],
                wintypes.BOOL,
            ),
            "GetFinalPathNameByHandleW": (
                [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD],
                wintypes.DWORD,
            ),
            "ReadFile": (
                [
                    wintypes.HANDLE,
                    wintypes.LPVOID,
                    wintypes.DWORD,
                    ctypes.POINTER(wintypes.DWORD),
                    wintypes.LPVOID,
                ],
                wintypes.BOOL,
            ),
            "SetFilePointerEx": (
                [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes = arguments
            function.restype = result

    def error(self):
        code = self.ctypes.get_last_error()
        if code in (1, 50, 87, 120):
            return UnsupportedSafeRead(
                errno.ENOTSUP, f"Safe Win32 operation unsupported ({code})"
            )
        return self.ctypes.WinError(code)

    def open(self, path: Path, *, directory=False, share_write=False, writable=False):
        if writable and (not directory or share_write):
            raise ValueError("Writable directory anchors never share write access")
        # OPEN_EXISTING, FILE_SHARE_READ only; never mutate or follow reparse data.
        flags = 0x00200000 | (0x02000000 if directory else 0)
        sharing = 1 | (2 if share_write else 0)
        handle = self.dll.CreateFileW(
            str(path),
            0xC0000000 if writable else 0x80000000,
            sharing,
            None,
            3,
            flags,
            None,
        )
        if handle == self.ctypes.c_void_p(-1).value:
            raise self.error()
        try:
            info = self.info(handle)
            if bool(info.attributes & 0x10) != directory:
                raise ValueError("Unexpected file/directory handle")
        except BaseException:
            self.close(handle)
            raise
        return handle

    def duplicate(self, handle):
        """Retain an existing capability without a conflicting pathname reopen."""
        result = self.types.HANDLE()
        process = self.dll.GetCurrentProcess()
        if not self.dll.DuplicateHandle(
            process, handle, process, self.ctypes.byref(result), 0, False, 2
        ):
            raise self.error()
        return result.value

    def sync(self, handle):
        # FlushFileBuffers requires GENERIC_WRITE. Unsupported directory flushes
        # propagate; no volume flush or weaker sharing fallback is attempted.
        if not self.dll.FlushFileBuffers(handle):
            if self.ctypes.get_last_error() == 6:
                raise UnsupportedSafeRead(
                    errno.ENOTSUP, "Durable directory flush unsupported"
                )
            raise self.error()

    def link(self, source, directory, name):
        """Publish an exclusive hard link in the held source file's directory.

        FileLinkInformation (11), not the bypass-access-check or Ex variants.
        No pathname parent reopen, replacement flag, or weaker sharing fallback.
        The caller retains synchronous ordinary-file/directory handles throughout.
        RootDirectory must be NULL for the documented same-directory operation;
        the validated basename is relative to the source, not process cwd.
        """
        c = self.ctypes
        name = _member(name)
        encoded = name.encode("utf-16-le")
        if len(encoded) > 65534:
            raise ValueError("Hard-link member exceeds native name bound")

        class LinkInformation(c.Structure):
            _fields_: ClassVar = [
                ("replace", c.c_uint32),
                ("root", c.c_void_p),
                ("length", c.c_uint32),
                ("name", c.c_uint16 * 1),
            ]

        class StatusUnion(c.Union):
            _fields_: ClassVar = [("status", c.c_int32), ("pointer", c.c_void_p)]

        class StatusBlock(c.Structure):
            _fields_: ClassVar = [
                ("result", StatusUnion),
                ("information", c.c_size_t),
            ]

        native = c.WinDLL("ntdll", use_last_error=True)
        native.NtSetInformationFile.argtypes = [
            c.c_void_p,
            c.POINTER(StatusBlock),
            c.c_void_p,
            c.c_uint32,
            c.c_int,
        ]
        native.NtSetInformationFile.restype = c.c_int32
        native.RtlNtStatusToDosError.argtypes = [c.c_int32]
        native.RtlNtStatusToDosError.restype = c.c_uint32
        size = max(
            c.sizeof(LinkInformation), LinkInformation.name.offset + len(encoded)
        )
        buffer = c.create_string_buffer(size)
        information = c.cast(buffer, c.POINTER(LinkInformation)).contents
        information.replace = 0  # ReplaceIfExists FALSE; remaining union bytes zero.
        information.root = None
        information.length = len(encoded)
        c.memmove(
            c.addressof(buffer) + LinkInformation.name.offset, encoded, len(encoded)
        )
        status_block = StatusBlock()
        status = native.NtSetInformationFile(
            source, c.byref(status_block), buffer, size, 11
        )
        if status != 0:
            # Synchronous handles must not return pending/intermediate success.
            # Never claim publication when terminal completion is unknown.
            code = native.RtlNtStatusToDosError(status)
            if code in (1, 50, 87, 120):
                raise UnsupportedSafeRead(
                    errno.ENOTSUP, "Native hard links unsupported"
                )
            raise c.WinError(code)

    def info(self, handle):
        if self.dll.GetFileType(handle) != 1:  # FILE_TYPE_DISK, not pipe/device
            raise ValueError("Metadata handle is not a disk file")
        info = self.Information()
        if not self.dll.GetFileInformationByHandle(handle, self.ctypes.byref(info)):
            raise self.error()
        if info.attributes & (0x400 | 0x40):  # REPARSE_POINT or DEVICE
            raise ValueError("Reparse points and devices are not ordinary files")
        return info

    def final_path(self, handle) -> Path:
        buffer = self.ctypes.create_unicode_buffer(32768)
        length = self.dll.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if not length:
            raise self.error()
        if length >= len(buffer):
            raise UnsupportedSafeRead(errno.ENOTSUP, "Final path exceeds native bound")
        return Path(buffer.value)

    def read(self, handle, limit: int) -> bytes:
        info = self.info(handle)
        if info.attributes & 0x10 or (info.size_high << 32 | info.size_low) > limit:
            raise ValueError("Metadata must be a bounded regular file")
        chunks = []
        remaining = limit + 1
        while remaining:
            buffer = self.ctypes.create_string_buffer(min(remaining, 65536))
            count = self.types.DWORD()
            if not self.dll.ReadFile(
                handle, buffer, len(buffer), self.ctypes.byref(count), None
            ):
                raise self.error()
            if not count.value:
                break
            chunks.append(buffer.raw[: count.value])
            remaining -= count.value
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise ValueError("Metadata exceeds byte bound")
        return raw

    def close(self, handle):
        if not self.dll.CloseHandle(handle):
            raise self.error()

    def read_chunk(self, handle, size):
        if self.info(handle).attributes & 0x10:
            raise ValueError("File must be a regular file")
        buffer = self.ctypes.create_string_buffer(size)
        count = self.types.DWORD()
        if not self.dll.ReadFile(handle, buffer, size, self.ctypes.byref(count), None):
            raise self.error()
        return buffer.raw[: count.value]

    def seek(self, handle, offset):
        if not self.dll.SetFilePointerEx(handle, offset, None, 0):
            raise self.error()

    def read_first_line(self, handle, limit: int) -> bytes:
        info = self.info(handle)
        if info.attributes & 0x10:
            raise ValueError("Journal must be a regular file")
        chunks = []
        remaining = limit + 1
        while remaining:
            buffer = self.ctypes.create_string_buffer(min(remaining, 4096))
            count = self.types.DWORD()
            if not self.dll.ReadFile(
                handle, buffer, len(buffer), self.ctypes.byref(count), None
            ):
                raise self.error()
            if not count.value:
                break
            chunk = buffer.raw[: count.value]
            newline = chunk.find(b"\n")
            if newline >= 0:
                chunks.append(chunk[: newline + 1])
                break
            chunks.append(chunk)
            remaining -= count.value
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise ValueError("First journal record exceeds byte bound")
        return raw

    def chunks(self, handle):
        """Stream an ordinary file from its validated immutable native handle."""
        if self.info(handle).attributes & 0x10:
            raise ValueError("Checkpoint must be a regular file")
        while True:
            buffer = self.ctypes.create_string_buffer(65536)
            count = self.types.DWORD()
            if not self.dll.ReadFile(
                handle, buffer, len(buffer), self.ctypes.byref(count), None
            ):
                raise self.error()
            if not count.value:
                return
            yield buffer.raw[: count.value]


def _windows_api():
    return _Win32()


def read_regular_file(path: str | Path, *, limit: int = 1024 * 1024) -> bytes:
    """Read at most limit bytes from the opened ordinary file, never a link."""
    if type(limit) is not int or limit < 0:
        raise ValueError("Byte bound must be a nonnegative integer")
    if os.name == "nt":
        api = _windows_api()
        handle = api.open(Path(path))
        try:
            return api.read(handle, limit)
        finally:
            api.close(handle)
    descriptor = _posix_open(path)
    try:
        return _read_descriptor(descriptor, limit)
    finally:
        os.close(descriptor)


def read_first_line(path: str | Path, *, limit: int = 65536) -> bytes:
    """Read a bounded first record, including its newline if published.

    Unlike metadata reads, total journal size is unrestricted. An unterminated
    prefix within the bound is returned for callers to await completion; overflow
    raises rather than returning a truncated record. The handle permits writers
    on Windows because this function probes a live append-only journal.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError("Byte bound must be a nonnegative integer")
    if os.name == "nt":
        api = _windows_api()
        handle = api.open(Path(path), share_write=True)
        try:
            return api.read_first_line(handle, limit)
        finally:
            api.close(handle)
    descriptor = _posix_open(path)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Journal must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.readline(limit + 1)
        if len(raw) > limit:
            raise ValueError("First journal record exceeds byte bound")
        return raw
    finally:
        os.close(descriptor)


class SafeFile:
    """Owned ordinary-file handle with bounded streaming operations."""

    def __init__(self, path, handle, *, api=None, parent=None):
        self.path = path
        self.handle = handle
        self.api = api
        self.parent = parent
        self._offset = 0

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def _check(self):
        if self.handle is None:
            raise ValueError("File anchor is closed")

    def close(self):
        handle, self.handle = self.handle, None
        try:
            if handle is not None:
                if self.api:
                    self.api.close(handle)
                else:
                    os.close(handle)
        finally:
            if self.parent is not None:
                parent, self.parent = self.parent, None
                parent.close()

    def read(self, size=65536):
        self._check()
        if type(size) is not int or not 0 < size <= 1024 * 1024:
            raise ValueError("Read size must be a positive integer at most 1 MiB")
        if self.api:
            raw = self.api.read_chunk(self.handle, size)
        else:
            raw = os.read(self.handle, size)
        self._offset += len(raw)
        return raw

    def seek(self, offset, whence=0):
        self._check()
        if (
            type(offset) is not int
            or type(whence) is not int
            or whence not in (0, 1, 2)
        ):
            raise ValueError("Seek requires an integer offset and valid origin")
        base = (0, self._offset, self.stat().st_size)[whence]
        offset += base
        if not 0 <= offset <= (1 << 63) - 1:
            raise ValueError(
                "Resulting offset must be a nonnegative signed 64-bit integer"
            )
        if self.api:
            self.api.seek(self.handle, offset)
        else:
            os.lseek(self.handle, offset, os.SEEK_SET)
        self._offset = offset
        return offset

    def stat(self):
        self._check()
        if self.api:
            info = self.api.info(self.handle)
            return FileMetadata(info.size_high << 32 | info.size_low)
        info = os.fstat(self.handle)
        return FileMetadata(info.st_size, info.st_uid, info.st_mode)

    @property
    def identity(self):
        """Stable native identity of this opened ordinary file, never its path."""
        self._check()
        return _identity(self.api, self.handle)

    def read_first_line(self, limit=65536):
        self._check()
        if type(limit) is not int or not 0 <= limit <= 1024 * 1024:
            raise ValueError("Line bound must be an integer from zero to 1 MiB")
        raw = bytearray()
        while len(raw) <= limit:
            chunk = self.read(min(4096, limit + 1 - len(raw)))
            if not chunk:
                return bytes(raw)
            newline = chunk.find(b"\n")
            if newline >= 0:
                tail = len(chunk) - newline - 1
                if tail:
                    self.seek(self._offset - tail)
                chunk = chunk[: newline + 1]
            raw.extend(chunk)
            if len(raw) > limit:
                raise ValueError("First journal record exceeds byte bound")
            if newline >= 0:
                return bytes(raw)
        raise ValueError("First journal record exceeds byte bound")

    def verify_private_owner(self, *, confidential=False):
        """Optionally reject all group/world permissions for credential files."""
        self._check()
        _private_owner(self.api, self.handle, confidential=confidential)

    def guard(self):
        self._check()
        if self.parent is not None:
            self.parent.guard()
        if self.api:
            self.api.info(self.handle)
            return
        descriptor = _posix_reopen(self.path)
        try:
            if _identity(None, descriptor) != _identity(None, self.handle):
                raise ValueError(
                    "Configured file path no longer identifies pinned file"
                )
        finally:
            os.close(descriptor)


class SafeDirectory:
    """Owned directory anchor; each child retains independent ownership."""

    def __init__(
        self,
        path,
        handle,
        *,
        api=None,
        ancestors=None,
        writable=False,
        native_handles=(),
    ):
        self.path = path
        self.handle = handle
        self.api = api
        self.ancestors = ancestors
        self.writable = writable
        self._native_handles = native_handles

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if self.ancestors is not None:
                self.ancestors.close()
                self.ancestors = None
            elif self.api:
                self.api.close(handle)
            else:
                os.close(handle)
        if self.ancestors is not None:
            self.ancestors.close()
            self.ancestors = None

    def _check(self):
        if self.handle is None:
            raise ValueError("Directory anchor is closed")

    def verify_private_owner(self, *, confidential=False):
        self._check()
        _private_owner(self.api, self.handle, confidential=confidential)

    def sync(self):
        """Durably flush this held directory, never reopen its pathname."""
        self._check()
        if self.api:
            if not self.writable:
                raise PermissionError("Directory sync requires a writable capability")
            self.api.sync(self.handle)
        else:
            os.fsync(self.handle)

    def link_file(self, source: str, target: str):
        """Atomically publish a single regular member without replacing a target.

        Publication is not a durability acknowledgement; callers must sync this
        held writable directory before treating the new name as durable.
        """
        self._check()
        source, target = _member(source), _member(target)
        if not self.writable:
            raise PermissionError(
                "Hard-link publication requires a writable capability"
            )
        self.guard()
        with self.open_file(source) as opened:
            opened.verify_private_owner()
            opened.guard()
            if self.api:
                self.api.link(opened.handle, self.handle, target)
            else:
                os.link(
                    source,
                    target,
                    src_dir_fd=self.handle,
                    dst_dir_fd=self.handle,
                    follow_symlinks=False,
                )
            opened.guard()
        self.guard()

    def _duplicate_windows(self):
        self._check()
        stack = ExitStack()
        handles = []
        try:
            for original in self._native_handles or (self.handle,):
                handle = self.api.duplicate(original)
                stack.callback(self.api.close, handle)
                handles.append(handle)
            return SafeDirectory(
                self.path,
                handles[-1],
                api=self.api,
                ancestors=stack.pop_all(),
                writable=self.writable,
                native_handles=tuple(handles),
            )
        except BaseException:
            stack.close()
            raise

    def guard(self):
        self._check()
        if self.api:
            self.api.info(self.handle)
            return
        descriptor = _posix_reopen(self.path, directory=True)
        try:
            if _identity(None, descriptor) != _identity(None, self.handle):
                raise ValueError(
                    "Configured directory path no longer identifies pinned directory"
                )
        finally:
            os.close(descriptor)

    def open_file(self, name: str, *, share_write=False):
        self._check()
        name = _member(name)
        # Win32 pins every ancestor; POSIX duplicates the parent anchor so the
        # returned file remains owned after this directory is closed.
        parent = (
            self._duplicate_windows()
            if self._native_handles
            else open_directory(self.path)
            if self.api
            else SafeDirectory(self.path, os.dup(self.handle))
        )
        handle = None
        try:
            if self.api:
                handle = parent.api.open(parent.path / name, share_write=share_write)
                if parent.api.final_path(handle).parent != parent.path:
                    raise ValueError("File escaped pinned ancestry")
            else:
                handle = _posix_open(name, dir_fd=parent.handle)
                if not stat.S_ISREG(os.fstat(handle).st_mode):
                    raise ValueError("File must be a regular file")
            return SafeFile(parent.path / name, handle, api=parent.api, parent=parent)
        except BaseException:
            try:
                if handle is not None:
                    if parent.api:
                        parent.api.close(handle)
                    else:
                        os.close(handle)
            finally:
                parent.close()
            raise

    def open_directory(self, name: str, *, writable=False):
        self._check()
        name = _member(name)
        if type(writable) is not bool:
            raise ValueError("Writable capability must be boolean")
        if self.api:
            # Independently retain ancestry so a pending child outlives parent.
            if not self._native_handles:
                return open_directory(self.path / name, writable=writable)
            parent = self._duplicate_windows()
            stack = ExitStack()
            stack.callback(parent.close)
            try:
                handle = self.api.open(
                    self.path / name, directory=True, writable=writable
                )
                stack.callback(self.api.close, handle)
                final = self.api.final_path(handle)
                if final.parent != self.path:
                    raise ValueError("Directory escaped pinned ancestry")
                return SafeDirectory(
                    final,
                    handle,
                    api=self.api,
                    ancestors=stack.pop_all(),
                    writable=writable,
                    native_handles=(*parent._native_handles, handle),
                )
            except BaseException:
                stack.close()
                raise
        return SafeDirectory(
            self.path / name,
            _posix_open(name, directory=True, dir_fd=self.handle),
            writable=writable,
        )

    def read_file(self, name: str, *, limit: int = 1024 * 1024):
        self._check()
        name = _member(name)
        if type(limit) is not int or limit < 0:
            raise ValueError("Byte bound must be a nonnegative integer")
        if self.api:
            handle = self.api.open(self.path / name)
            try:
                if self.api.final_path(handle).parent != self.path:
                    raise ValueError("File escaped pinned ancestry")
                return self.api.read(handle, limit)
            finally:
                self.api.close(handle)
        descriptor = _posix_open(name, dir_fd=self.handle)
        try:
            return _read_descriptor(descriptor, limit)
        finally:
            os.close(descriptor)

    def entries(self):
        self._check()
        with os.scandir(self.path if self.api else self.handle) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if (
                    stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & 0x400
                ):
                    yield DirectoryEntry(entry.name, False, False)
                    continue
                yield DirectoryEntry(
                    entry.name, stat.S_ISDIR(info.st_mode), stat.S_ISREG(info.st_mode)
                )

    def digest_file(self, name: str, *, check=None, limit=None) -> tuple[int, str]:
        """Hash a no-follow ordinary file in fixed-size chunks on one handle."""
        self._check()
        name = _member(name)
        digest = hashlib.sha256()
        size = 0
        if self.api:
            handle = self.api.open(self.path / name)
            try:
                if self.api.final_path(handle).parent != self.path:
                    raise ValueError("File escaped pinned ancestry")
                for chunk in self.api.chunks(handle):
                    if check is not None:
                        check()
                    size += len(chunk)
                    if limit is not None and size > limit:
                        raise ValueError("File exceeds declared hash byte bound")
                    digest.update(chunk)
            finally:
                self.api.close(handle)
        else:
            descriptor = _posix_open(name, dir_fd=self.handle)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError("Checkpoint must be a regular file")
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    while chunk := stream.read(65536):
                        if check is not None:
                            check()
                        size += len(chunk)
                        if limit is not None and size > limit:
                            raise ValueError("File exceeds declared hash byte bound")
                        digest.update(chunk)
            finally:
                os.close(descriptor)
        return size, digest.hexdigest()


def open_directory(path: str | Path, *, writable=False) -> SafeDirectory:
    """Pin a directory for enumeration and no-follow child reads."""
    path = Path(path).absolute()
    if type(writable) is not bool:
        raise ValueError("Writable capability must be boolean")
    if os.name != "nt":
        if os.open not in os.supports_dir_fd or os.scandir not in os.supports_fd:
            raise UnsupportedSafeRead(
                errno.ENOTSUP, "Descriptor-relative directory reads unsupported"
            )
        return SafeDirectory(path, _posix_open(path, directory=True), writable=writable)
    api = _windows_api()
    stack = ExitStack()
    handles = []
    try:
        # Pin ancestors top-down; no junction, mount point, or symlink is accepted.
        current = Path(path.anchor)
        canonical = current
        parts = (None, *path.parts[1:])
        for index, part in enumerate(parts):
            if part is not None:
                current /= part
            if writable and index == len(parts) - 1:
                handle = api.open(current, directory=True, writable=True)
            else:
                handle = api.open(current, directory=True)
            stack.callback(api.close, handle)
            handles.append(handle)
            final = api.final_path(handle)
            if part is not None and final.parent != canonical:
                raise ValueError("Directory escaped pinned ancestry")
            canonical = final
            current = canonical
        # Stack owns all handles including the terminal directory.
        return SafeDirectory(
            canonical,
            handle,
            api=api,
            ancestors=stack.pop_all(),
            writable=writable,
            native_handles=tuple(handles) if writable else (),
        )
    except BaseException:
        stack.close()
        raise
