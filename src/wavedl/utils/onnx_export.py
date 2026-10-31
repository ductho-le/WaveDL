"""Single-file ONNX acceptance: external paths are never loaded or followed."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from tempfile import mkdtemp

from wavedl.core.persistence import sync_directory


MAX_ONNX_BYTES = 2**31 - 1


class ExportCommitUnknown(OSError):
    """Commit/rollback durability failed; retain the reported recovery directory."""


def replace_export(staged: Path, destination: Path) -> None:
    """Commit a caller-owned quiescent target with durable prior-byte rollback.

    A recovery hardlink is outside the caller's temporary staging directory.
    Symlinks and detected precommit target changes fail closed. The inode checks
    are not portable atomic compare-and-swap: callers must quiesce other writers
    to this target (managed operations own their private run directory).
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
    recovery = Path(mkdtemp(prefix=".onnx-recovery-", dir=destination.parent))
    backup = recovery / "prior.onnx"
    committed = False
    retain = False

    def identity(info):
        return info.st_dev, info.st_ino

    try:
        if prior is not None:
            os.link(destination, backup, follow_symlinks=False)
            if not stat.S_ISREG(backup.lstat().st_mode) or identity(
                backup.lstat()
            ) != identity(prior):
                raise OSError("Export destination changed before backup")
            descriptor = os.open(backup, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                held = os.fstat(descriptor)
                if not stat.S_ISREG(held.st_mode) or identity(held) != identity(prior):
                    raise OSError("Export backup identity changed")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            sync_directory(recovery)
        sync_directory(destination.parent)
        try:
            current = destination.lstat()
        except FileNotFoundError:
            current = None
        if (current is None) != (prior is None) or (
            current is not None and identity(current) != identity(prior)
        ):
            raise OSError("Export destination changed before commit")
        os.replace(staged, destination)
        committed = True
        sync_directory(destination.parent)
    except BaseException as error:
        if committed:
            try:
                if identity(destination.lstat()) != identity(staged_info):
                    raise OSError("Export destination changed before rollback")
                if prior is None:
                    destination.unlink()
                else:
                    restore = recovery / "restore.onnx"
                    os.link(backup, restore, follow_symlinks=False)
                    os.replace(restore, destination)
                sync_directory(destination.parent)
            except BaseException as rollback_error:
                retain = True
                raise ExportCommitUnknown(
                    f"ONNX export durability/rollback is unknown; recovery retained at {recovery}"
                ) from rollback_error
        raise error
    finally:
        if not retain:
            try:
                if backup.exists():
                    backup.unlink()
                recovery.rmdir()
            except OSError as error:
                raise ExportCommitUnknown(
                    f"ONNX recovery cleanup failed; inspect {recovery}"
                ) from error


def validate_self_contained_onnx(
    path: Path, *, expected_opset: int | None = None
) -> None:
    """Check every tensor, including nested graphs, without reading companions.

    WaveDL publishes downloadable single-file models. Graphs requiring external
    weights or exceeding protobuf's 2GiB limit are unsupported, not partial
    success. This check is required even when numerical validation is disabled.
    If expected_opset is supplied, the standard ONNX domain must declare exactly
    that version; an exporter's silent upgrade is not an accepted substitute.
    """
    import onnx

    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_ONNX_BYTES:
        raise ValueError("ONNX export must be a bounded regular single file")
    model = onnx.load(path, load_external_data=False)
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

    def visit(message):
        if isinstance(message, onnx.TensorProto) and (
            message.data_location == onnx.TensorProto.EXTERNAL or message.external_data
        ):
            raise ValueError(
                "ONNX external tensor data is unsupported; export a self-contained model"
            )
        for field, value in message.ListFields():
            if field.message_type is not None:
                if field.is_repeated:
                    for item in value:
                        visit(item)
                else:
                    visit(value)

    visit(model)
    onnx.checker.check_model(model)
