"""
ONNX Export Safety
==================

Helpers that keep ``wavedl-test --export onnx`` outputs self-contained and
replace an existing export without ever leaving a damaged file behind.

Features:
    - Validation that an export is one bounded regular file whose tensors
      (including nested subgraphs) are all embedded, and whose standard-domain
      opset is exactly the requested one. External-data references are never
      loaded or followed.
    - Durable replacement: the staged file is fsynced, renamed over the
      destination and the directory is fsynced. If the commit fails after a
      prior export existed, the prior bytes are restored; if that rollback also
      fails, the prior export is kept in a recovery directory and
      ``ExportCommitUnknown`` is raised.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import stat
from pathlib import Path
from tempfile import mkdtemp


# Protobuf cannot serialize a single message of 2 GiB or more; larger ONNX
# models need external-data side files, which a single downloadable file lacks.
MAX_ONNX_BYTES = 2**31 - 1

# Directory fsync is unsupported on some filesystems (e.g. certain FUSE/network
# mounts); those report one of these and cannot be made more durable.
_DIRECTORY_SYNC_UNSUPPORTED = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}


class ExportCommitUnknown(OSError):
    """Commit and rollback both failed; the prior export is in a recovery dir.

    The destination may hold either the new or the prior export, so callers
    must not report the export as failed-without-change.
    """


def sync_directory(path: Path) -> None:
    """Flush a directory entry (rename/unlink) to stable storage.

    Windows cannot open directories for fsync; NTFS journals the metadata, so
    this is a no-op there.
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in _DIRECTORY_SYNC_UNSUPPORTED:
            raise
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    """Flush a regular file's data and metadata to stable storage."""
    # Windows flushes through FlushFileBuffers, which needs write access.
    flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
    descriptor = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _preserve(source: Path, target: Path) -> None:
    """Make ``target`` a durable copy of ``source`` (hardlink when possible)."""
    try:
        os.link(source, target)
    except OSError:
        # Filesystems without hardlinks (FAT/exFAT, some SMB shares).
        shutil.copyfile(source, target)
    fsync_file(target)


def replace_export(staged: Path, destination: Path) -> None:
    """Replace ``destination`` with ``staged``, rolling back on failure.

    The prior export (if any) is preserved in a recovery directory next to the
    destination until the replacement is durable. On failure the prior bytes
    are restored and the original error is re-raised; if restoring fails too,
    the recovery directory is retained and ``ExportCommitUnknown`` is raised.

    Symlink or non-regular destinations are refused rather than followed. The
    identity checks detect, but cannot atomically prevent, a concurrent writer;
    callers must not export to the same path concurrently.
    """
    staged_info = staged.lstat()
    if not stat.S_ISREG(staged_info.st_mode):
        raise ValueError("Export staging must be a regular file")
    try:
        prior = destination.lstat()
    except FileNotFoundError:
        prior = None
    if prior is not None and not stat.S_ISREG(prior.st_mode):
        raise ValueError("Export destination must be a regular file, not a symlink")

    def destination_unchanged() -> bool:
        try:
            current = destination.lstat()
        except FileNotFoundError:
            return prior is None
        return prior is not None and (
            _identity(current) == _identity(prior)
            and current.st_size == prior.st_size
            and current.st_mtime_ns == prior.st_mtime_ns
        )

    recovery = Path(mkdtemp(prefix=".onnx-recovery-", dir=destination.parent))
    backup = recovery / "prior.onnx"
    restore = recovery / "restore.onnx"
    committed = False
    retain = False
    try:
        if prior is not None:
            _preserve(destination, backup)
            sync_directory(recovery)
        sync_directory(destination.parent)
        if not destination_unchanged():
            raise OSError("Export destination changed before commit")
        os.replace(staged, destination)
        committed = True
        sync_directory(destination.parent)
    except BaseException as error:
        if committed:
            try:
                if _identity(destination.lstat()) != _identity(staged_info):
                    raise OSError("Export destination changed before rollback")
                if prior is None:
                    destination.unlink()
                else:
                    # Restore from a second copy so the backup survives a
                    # failed restore.
                    _preserve(backup, restore)
                    os.replace(restore, destination)
                sync_directory(destination.parent)
            except BaseException as rollback_error:
                retain = prior is not None
                location = (
                    f"prior export retained at {backup}"
                    if retain
                    else "no prior export existed"
                )
                raise ExportCommitUnknown(
                    f"ONNX export commit and rollback both failed; the state of "
                    f"{destination} is unknown ({location})"
                ) from rollback_error
        raise error
    finally:
        if not retain:
            try:
                shutil.rmtree(recovery)
            except OSError as cleanup_error:
                logging.warning(
                    f"   ⚠ Could not remove ONNX recovery directory {recovery}: "
                    f"{cleanup_error}"
                )


def reject_external_tensors(model) -> None:
    """Raise if any tensor in ``model`` (including subgraphs) is external data."""
    import onnx

    def visit(message) -> None:
        if isinstance(message, onnx.TensorProto) and (
            message.data_location == onnx.TensorProto.EXTERNAL or message.external_data
        ):
            raise ValueError(
                "ONNX external tensor data is unsupported; export a self-contained model"
            )
        for field, value in message.ListFields():
            if field.message_type is not None:
                # A singular message field, or a repeated container of them.
                if hasattr(value, "ListFields"):
                    visit(value)
                else:
                    for item in value:
                        visit(item)

    visit(model)


def validate_self_contained_onnx(
    path: Path, *, expected_opset: int | None = None
) -> None:
    """Check that ``path`` is a complete single-file ONNX model.

    The file must be a regular file below the protobuf 2 GiB limit, every
    tensor (including nested graphs) must be embedded, and, when
    ``expected_opset`` is given, the standard ONNX domain must declare exactly
    that version; an exporter's silent upgrade is not accepted. Companion
    external-data files are never read.

    Raises:
        ValueError: If any of the above does not hold.
    """
    import onnx

    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_ONNX_BYTES:
        raise ValueError("ONNX export must be a bounded regular single file")
    model = onnx.load(str(path), load_external_data=False)
    if expected_opset is not None:
        standard_opsets = [
            item.version
            for item in model.opset_import
            if item.domain in ("", "ai.onnx")
        ]
        if standard_opsets != [expected_opset]:
            raise ValueError(
                f"ONNX export requested opset {expected_opset}, "
                f"but produced standard-domain opsets {standard_opsets}"
            )
    reject_external_tensors(model)
    onnx.checker.check_model(model)
