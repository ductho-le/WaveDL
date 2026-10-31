"""Exclusive publication of complete files on storage supporting durable sync."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def sync_directory(path: Path) -> None:
    """Flush a directory entry; unsupported storage raises, never degrades silently."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create.restype = wintypes.HANDLE
        flush = kernel.FlushFileBuffers
        flush.argtypes = [wintypes.HANDLE]
        flush.restype = wintypes.BOOL
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        # Safe readers briefly pin ancestry without write sharing. Admit this
        # short-lived flush handle after they leave; never relax their sharing
        # or retry the flush itself. The deadline bounds polls, not native I/O.
        deadline = time.monotonic() + 5
        while True:
            handle = create(str(path), 0xC0000000, 7, None, 3, 0x02000000, None)
            if handle != wintypes.HANDLE(-1).value:
                break
            error = ctypes.get_last_error()
            remaining = deadline - time.monotonic()
            if error != 32 or remaining <= 0:
                raise ctypes.WinError(error)
            time.sleep(min(0.02, remaining))
            if time.monotonic() >= deadline:
                raise ctypes.WinError(error)
        try:
            if not flush(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            close(handle)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path: Path, content: bytes) -> None:
    """Publish with atomic no-replace link after the temporary file is fsynced."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
