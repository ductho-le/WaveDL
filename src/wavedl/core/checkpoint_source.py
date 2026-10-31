"""Explicit published inference sources, captured without loading ML or pickle."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from wavedl.core.checkpoint_selection import (
    _covered_inventory,
    read_selection_metadata,
    validate_selection_metadata,
)
from wavedl.core.generations import _manifest_digest, _read_open_generation
from wavedl.core.layout import RunLayout
from wavedl.core.safe_files import open_directory


def capture_checkpoint_source(
    layout: RunLayout, generation: str, *, manifest_sha256: str, check=None
) -> dict:
    """Capture an explicit publication, rejecting absent or malformed sources."""
    try:
        return _capture_checkpoint_source(
            layout, generation, manifest_sha256=manifest_sha256, check=check
        )
    except (FileNotFoundError, KeyError, TypeError) as error:
        raise ValueError("Checkpoint source is missing or malformed") from error


def _capture_checkpoint_source(
    layout: RunLayout, generation: str, *, manifest_sha256: str, check=None
) -> dict:
    """Hash one explicit generation against the registered immutable manifest.

    The caller resolves ownership before dispatching this filesystem-only work.
    Metadata is hash-covered; the daemon never deserializes training pickle.
    Aggregate CV results are not model checkpoints. Explicit CV folds are.
    """
    if type(generation) is not str or not re.fullmatch(r"gen-\d{6,}", generation):
        raise ValueError("Invalid checkpoint generation")
    check = check or (lambda: None)
    check()
    with ExitStack() as anchors:
        root = anchors.enter_context(open_directory(layout.run_dir))
        if _manifest_digest(layout, root) != manifest_sha256:
            raise ValueError("Checkpoint source manifest differs from approved plan")
        checkpoints = anchors.enter_context(root.open_directory("checkpoints"))
        directory = anchors.enter_context(checkpoints.open_directory(generation))
        raw = directory.read_file("COMPLETE.json", limit=1024 * 1024)
        inventory = json.loads(raw)
        files = inventory.get("files") if type(inventory) is dict else None
        if (
            type(files) is not dict
            or not 1 <= len(files) <= 2048
            or any(
                type(record) is not dict
                or type(record.get("size")) is not int
                or not 0 <= record["size"] <= 16 * 1024**3
                for record in files.values()
            )
            or sum(record["size"] for record in files.values()) > 16 * 1024**3
        ):
            raise ValueError("Checkpoint inventory exceeds inference bound")
        selected = _read_open_generation(
            layout,
            layout.checkpoints / generation,
            manifest_sha256,
            directory,
            check=check,
        )
        if selected is None or not _covered_inventory(selected, check=check):
            raise ValueError("Checkpoint generation is incomplete or changed")
        metadata = read_selection_metadata(selected)
        if metadata is None or metadata["kind"] == "cross_validation_summary":
            raise ValueError("Explicit source requires inference checkpoint metadata")
        weights = {
            "model.safetensors",
            "model.bin",
            "pytorch_model.bin",
            "model_weights.pth",
        }
        if not weights.intersection(files) or not {
            "training_meta.pkl",
            "scaler.pkl",
        }.issubset(files):
            raise ValueError("Checkpoint lacks published inference state")
        check()
        # Re-read the publication marker through the same anchor after hashing.
        if directory.read_file("COMPLETE.json", limit=1024 * 1024) != raw:
            raise ValueError("Checkpoint publication changed during selection")
        return {
            "run_id": layout.run_id,
            "generation": generation,
            "manifest_sha256": manifest_sha256,
            "complete_sha256": hashlib.sha256(raw).hexdigest(),
            "files": {name: asdict(record) for name, record in selected.files.items()},
            "selection": metadata,
        }


def validate_checkpoint_source(source: object) -> dict:
    """Validate the durable source pin without filesystem work."""
    if type(source) is not dict or set(source) != {
        "run_id",
        "generation",
        "manifest_sha256",
        "complete_sha256",
        "files",
        "selection",
    }:
        raise ValueError("Invalid checkpoint source pin")
    if (
        type(source["run_id"]) is not str
        or str(UUID(source["run_id"])) != source["run_id"]
    ):
        raise ValueError("Invalid checkpoint source identity")
    if type(source["generation"]) is not str or not re.fullmatch(
        r"gen-\d{6,}", source["generation"]
    ):
        raise ValueError("Invalid checkpoint generation")
    for name in ("manifest_sha256", "complete_sha256"):
        if type(source[name]) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", source[name]
        ):
            raise ValueError("Invalid checkpoint source digest")
    if type(source["files"]) is not dict or not 1 <= len(source["files"]) <= 2048:
        raise ValueError("Missing checkpoint source inventory")
    for name, record in source["files"].items():
        if (
            type(name) is not str
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or any(char in name for char in "\\\\:\0")
            or type(record) is not dict
            or set(record) != {"size", "sha256"}
            or type(record["size"]) is not int
            or not 0 <= record["size"] <= 16 * 1024**3
            or type(record["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError("Invalid checkpoint source inventory")
    if sum(record["size"] for record in source["files"].values()) > 16 * 1024**3:
        raise ValueError("Checkpoint source inventory exceeds bound")
    if (
        validate_selection_metadata(source["selection"])["kind"]
        == "cross_validation_summary"
    ):
        raise ValueError("Aggregate CV publications cannot be inference sources")
    return source


def verify_checkpoint_source(path: Path, source: dict) -> None:
    """Reject replacement publications after approval, before inference opens."""
    validate_checkpoint_source(source)
    if path.name != source["generation"] or path.parent.name != "checkpoints":
        raise ValueError("Inference path differs from approved checkpoint source")
    current = capture_checkpoint_source(
        RunLayout(path.parent.parent, source["run_id"]),
        source["generation"],
        manifest_sha256=source["manifest_sha256"],
    )
    if current != source:
        raise ValueError("Checkpoint source changed after plan approval")


@contextmanager
def snapshot_checkpoint_source(path: Path, source: dict):
    """Yield private inference inputs copied and hashed from held safe handles.

    Loading mutable source paths after verification would introduce a TOCTOU
    gap. Only files consumed by inference are copied; each exact copied byte is
    hashed against approval. The exclusive temporary directory is removed on
    both success and failure, and no existing user path is overwritten.
    """
    verify_checkpoint_source(path, source)
    names = {
        "model.safetensors",
        "model.bin",
        "pytorch_model.bin",
        "model_weights.pth",
        "training_meta.pkl",
        "scaler.pkl",
    } & source["files"].keys()
    with (
        TemporaryDirectory(prefix="wavedl-inference-") as temporary,
        open_directory(path) as directory,
    ):
        target = Path(temporary)
        for name in sorted(names):
            expected = source["files"][name]
            digest = hashlib.sha256()
            size = 0
            with (
                directory.open_file(name) as incoming,
                (target / name).open("xb") as outgoing,
            ):
                while chunk := incoming.read(65536):
                    size += len(chunk)
                    if size > expected["size"]:
                        raise ValueError(
                            "Checkpoint member grew during inference capture"
                        )
                    digest.update(chunk)
                    outgoing.write(chunk)
                incoming.guard()
            if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError("Checkpoint member changed during inference capture")
        directory.guard()
        yield target
