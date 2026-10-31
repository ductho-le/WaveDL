"""Checkpoint consumers shared by standalone training, inference, and clients.

Producers may put ``checkpoint_selection.json`` in their staging tree before
publication. Its supported contract is ``{"schema_version": 1, "kind":
"best" | "periodic" | "interrupted" | "cross_validation_fold" |
"cross_validation_summary"}``. COMPLETE.json must cover that file.
Only an explicit ``best`` kind qualifies for automatic plotting; filenames,
steps, and best-loss values do not identify best checkpoints. Existing T20a
generations without this optional metadata remain eligible for resume.
Optional ``source_seq`` is the nonnegative exact integer journal cursor whose
metrics are represented by the generation; missing cursors remain legacy data.

Latest means publication order, not training step: a newer attempt can rewind.
Completed emergency generations are resumable; interrupted publication is not.
CV outputs require an explicit generation choice and are never auto-resumed.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
from wavedl.core.manifest import read_manifest
from wavedl.core.safe_files import (
    UnsupportedSafeRead as UnsupportedSafeRead,
    open_directory,
)


if TYPE_CHECKING:
    from wavedl.core.generations import PublishedGeneration


CheckpointPurpose = Literal["resume", "best"]
SELECTION_METADATA = "checkpoint_selection.json"


def validate_selection_metadata(metadata: object) -> dict:
    """Validate the lightweight classification shared by checkpoint consumers.

    Older CV publications can omit fold identity; consumers exporting folds must
    require it separately. A supplied identity is never valid for another kind.
    """
    if (
        not isinstance(metadata, dict)
        or set(metadata) - {"schema_version", "kind", "source_seq", "fold"}
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or not isinstance(metadata.get("kind"), str)
        or metadata.get("kind")
        not in {
            "best",
            "periodic",
            "interrupted",
            "cross_validation_fold",
            "cross_validation_summary",
        }
        or (
            "fold" in metadata
            and (
                metadata["kind"] != "cross_validation_fold"
                or type(metadata["fold"]) is not int
                or not 1 <= metadata["fold"] <= 2**63 - 1
            )
        )
    ):
        raise ValueError("Invalid checkpoint selection metadata")
    if "source_seq" in metadata and (
        type(metadata["source_seq"]) is not int or metadata["source_seq"] < 0
    ):
        raise ValueError("Checkpoint source_seq must be a nonnegative integer")
    return metadata


def read_selection_metadata(generation: PublishedGeneration) -> dict | None:
    """Read bounded hash-covered classification and optional journal cutoff.

    Missing metadata or missing source_seq is supported for older publications.
    A present source_seq must be an exact nonnegative int, never a bool.
    """
    if SELECTION_METADATA not in generation.files:
        return None
    expected = generation.files[SELECTION_METADATA]
    if expected.size > 4096:
        raise ValueError("Checkpoint selection metadata exceeds byte bound")
    try:
        with open_directory(generation.path) as directory:
            raw = directory.read_file(SELECTION_METADATA, limit=4096)
    except FileNotFoundError as exc:
        raise ValueError(
            "Checkpoint selection metadata changed after validation"
        ) from exc
    if len(raw) != expected.size or hashlib.sha256(raw).hexdigest() != expected.sha256:
        raise ValueError("Checkpoint selection metadata changed after validation")
    return validate_selection_metadata(json.loads(raw))


def _kind(generation: PublishedGeneration) -> str | None:
    metadata = read_selection_metadata(generation)
    return metadata["kind"] if metadata is not None else None


def _covered_inventory(generation: PublishedGeneration, *, check=None) -> bool:
    # Discovery verifies listed hashes; reject added/unlisted files too, so an
    # inference loader cannot accidentally prefer an unverified weight format.
    check = check or (lambda: None)
    check()
    expected = set(generation.files) | {"COMPLETE.json"}
    directories = {
        parent.as_posix()
        for name in generation.files
        for parent in Path(name).parents
        if parent != Path(".")
    }
    seen = set()
    extra_directories = 0
    try:
        with ExitStack() as anchors:
            root = anchors.enter_context(open_directory(generation.path))
            pending = [(root, Path("."))]
            while pending:
                check()
                directory, relative = pending.pop()
                for entry in directory.entries():
                    check()
                    member = relative / entry.name
                    name = member.as_posix()
                    if entry.is_directory:
                        if name not in directories:
                            # Publication permits empty directories without
                            # file records. Bound additional traversal.
                            extra_directories += 1
                            if extra_directories > 1024:
                                return False
                        child = anchors.enter_context(
                            directory.open_directory(entry.name)
                        )
                        pending.append((child, member))
                    elif name in expected and entry.is_regular:
                        seen.add(name)
                    else:
                        return False
    except UnsupportedSafeRead:
        raise
    except (ValueError, FileNotFoundError):
        return False
    return seen == expected


def select_checkpoint(
    layout: RunLayout, *, purpose: CheckpointPurpose = "resume"
) -> PublishedGeneration | None:
    """Return the newest validated publication eligible for this purpose.

    No legacy fallback is inferred here. Callers retain their existing legacy
    selection when this returns None. Invalid optional metadata is ineligible.
    """
    if purpose not in ("resume", "best"):
        raise ValueError("Checkpoint purpose must be resume or best")
    for generation in reversed(published_generations(layout)):
        if not _covered_inventory(generation):
            continue
        try:
            kind = _kind(generation)
        except (ValueError, UnicodeError):
            continue
        if kind in {"cross_validation_fold", "cross_validation_summary"}:
            continue
        if purpose == "resume" or kind == "best":
            return generation
    return None


def resolve_checkpoint_path(
    path: str | Path, *, purpose: CheckpointPurpose = "best"
) -> Path:
    """Resolve a run/checkpoints root or validate an explicit generation.

    Explicit legacy checkpoint directories remain unchanged. Explicit generation
    paths must be valid publications, even when no automatic selection is needed.
    This works on either side of a declared storage map without daemon imports.
    """
    if purpose not in ("resume", "best"):
        raise ValueError("Checkpoint purpose must be resume or best")
    path = Path(path)
    explicit_generation = path.parent.name == "checkpoints" and re.fullmatch(
        r"gen-\d{6,}", path.name
    )
    if explicit_generation:
        run_dir = path.parent.parent
    elif path.name == "checkpoints" and (path.parent / "manifest.json").is_file():
        run_dir = path.parent
    elif (path / "manifest.json").is_file() and (path / "checkpoints").is_dir():
        run_dir = path
    else:
        if (path / "COMPLETE.json").exists():
            raise ValueError(f"Published checkpoint is outside a run layout: {path}")
        return path
    manifest = read_manifest(run_dir / "manifest.json")
    layout = RunLayout(run_dir, manifest.run_id)
    if explicit_generation:
        for generation in published_generations(layout):
            if generation.path == path.absolute() and _covered_inventory(generation):
                kind = _kind(generation)
                if purpose == "resume" and kind in {
                    "cross_validation_fold",
                    "cross_validation_summary",
                }:
                    raise ValueError(
                        "Cross-validation generations are inference-only, not resumable"
                    )
                return generation.path
        raise ValueError(f"Checkpoint generation is incomplete or invalid: {path}")
    selected = select_checkpoint(layout, purpose=purpose)
    if selected is None:
        # Legacy best compatibility is deliberately limited to automatic plotting.
        legacy = layout.run_dir / "best_checkpoint"
        if purpose == "best" and legacy.is_dir():
            return legacy
        raise FileNotFoundError(f"No published {purpose} checkpoint in {run_dir}")
    return selected.path
