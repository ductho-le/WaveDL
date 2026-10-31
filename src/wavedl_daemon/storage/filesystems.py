# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Filesystem identification, using only the standard library.

``psutil`` is not a declared dependency of this project, and ``os.statvfs``
reports no filesystem type, so each platform needs its own mechanism. Every
function here answers "what kind of storage is this path on" on a best-effort
basis and says so honestly: ``filesystem_type`` returns ``None`` when the
platform cannot tell us, and callers must decide what to do about that rather
than being handed a confident wrong answer.
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path


# Filesystems whose locking is unsafe for SQLite. SQLite's own documentation
# warns that POSIX advisory locks are unreliable over NFS, and the same applies
# to the other network filesystems here.
NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afpfs",
        "afs",
        "beegfs",
        "ceph",
        "cifs",
        "fuse.glusterfs",
        "fuse.sshfs",
        "gpfs",
        "lustre",
        "nfs",
        "nfs3",
        "nfs4",
        "panfs",
        "smb2",
        "smb3",
        "smbfs",
        "webdav",
    }
)

# Filesystems that do not outlive the machine, so a shared run root on one is
# invisible to every other node no matter how the cluster is configured.
EPHEMERAL_FILESYSTEMS = frozenset({"devtmpfs", "ramfs", "tmpfs"})


class _MacStatfs(ctypes.Structure):
    """darwin ``struct statfs``; only the trailing name fields are of interest."""

    _fields_ = (
        ("f_bsize", ctypes.c_uint32),
        ("f_iosize", ctypes.c_int32),
        ("f_blocks", ctypes.c_uint64),
        ("f_bfree", ctypes.c_uint64),
        ("f_bavail", ctypes.c_uint64),
        ("f_files", ctypes.c_uint64),
        ("f_ffree", ctypes.c_uint64),
        ("f_fsid", ctypes.c_int32 * 2),
        ("f_owner", ctypes.c_uint32),
        ("f_type", ctypes.c_uint32),
        ("f_flags", ctypes.c_uint32),
        ("f_fssubtype", ctypes.c_uint32),
        ("f_fstypename", ctypes.c_char * 16),
        ("f_mntonname", ctypes.c_char * 1024),
        ("f_mntfromname", ctypes.c_char * 1024),
        ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    )


def _macos_filesystem_type(path: Path) -> str | None:
    library = ctypes.CDLL("libc.dylib", use_errno=True)
    buffer = _MacStatfs()
    if library.statfs(ctypes.c_char_p(bytes(path)), ctypes.byref(buffer)) != 0:
        return None
    return buffer.f_fstypename.decode("utf-8", "replace").lower() or None


def _linux_filesystem_type(path: Path) -> str | None:
    """The longest mount point in /proc/self/mountinfo that contains ``path``."""
    best_type, best_length = None, -1
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as stream:
            for line in stream:
                before, separator, after = line.partition(" - ")
                if not separator:
                    continue
                fields, rest = before.split(), after.split()
                if len(fields) < 5 or not rest:
                    continue
                mount_point = Path(
                    re.sub(
                        r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4]
                    )
                )
                if path != mount_point and mount_point not in path.parents:
                    continue
                if len(str(mount_point)) > best_length:
                    best_type, best_length = rest[0].lower(), len(str(mount_point))
    except OSError:
        return None
    return best_type


def _windows_filesystem_type(path: Path) -> str | None:
    """``GetDriveTypeW``: 4 is DRIVE_REMOTE. UNC paths are network by definition."""
    text = str(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return "unc"
    anchor = path.anchor or text[:3]
    drive_type = ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(anchor))
    return {2: "removable", 3: "local", 4: "remote", 5: "cdrom", 6: "ramdisk"}.get(
        drive_type
    )


def filesystem_type(path: Path) -> str | None:
    """The filesystem backing ``path``, or ``None`` when it cannot be determined.

    ``path`` need not exist; the nearest existing ancestor is probed, because a
    root is usually validated before it is created.
    """
    probe = Path(path).resolve()
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    try:
        if sys.platform == "darwin":
            return _macos_filesystem_type(probe)
        if sys.platform.startswith("linux"):
            return _linux_filesystem_type(probe)
        if sys.platform == "win32":
            return _windows_filesystem_type(probe)
    except OSError:
        return None
    return None


def is_network_filesystem(path: Path) -> bool | None:
    """Tri-state: True, False, or None when the platform will not say."""
    name = filesystem_type(path)
    if name is None:
        return None
    if sys.platform == "win32":
        return name in {"remote", "unc"}
    return name in NETWORK_FILESYSTEMS


def is_ephemeral_filesystem(path: Path) -> bool | None:
    name = filesystem_type(path)
    if name is None:
        return None
    if sys.platform == "win32":
        return name == "ramdisk"
    return name in EPHEMERAL_FILESYSTEMS
