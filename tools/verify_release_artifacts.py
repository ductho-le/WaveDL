"""Verify downloaded release bytes and provenance before external publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path


def regular_bytes(path: Path, *, limit: int) -> bytes:
    """Reject links, devices, FIFOs and oversized inputs before reading."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"Invalid bounded regular release file: {path}")
        data = source.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"Release input exceeds bound: {path}")
        return data


def distribution_hash(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024**3:
            raise ValueError(f"Invalid release distribution: {path}")
        digest = hashlib.sha256()
        total = 0
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > 8 * 1024**3:
                raise ValueError(f"Release distribution exceeds bound: {path}")
            digest.update(block)
        return digest.hexdigest()


def verify_artifacts(
    root: Path, *, commit: str, tag: str, expected_manifest_sha: str | None = None
) -> dict:
    manifest_bytes = regular_bytes(root / "release-artifacts.json", limit=65536)
    if (
        expected_manifest_sha
        and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha
    ):
        raise ValueError(
            "Release manifest digest does not match trusted package output"
        )
    manifest = json.loads(manifest_bytes)
    if manifest["source_commit"] != commit or manifest["tag"] != tag:
        raise ValueError("Release artifact provenance does not match this run")
    if not isinstance(manifest["prerelease"], bool):
        raise ValueError("Invalid release classification")
    files = manifest["files"]
    if (
        not isinstance(files, dict)
        or len(files) != 2
        or sum(name.endswith(".whl") for name in files) != 1
        or sum(name.endswith(".tar.gz") for name in files) != 1
        or any(Path(name).name != name for name in files)
    ):
        raise ValueError("Expected exactly one wheel and one source distribution")
    directory = root / "dist"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Invalid distribution directory")
    if {path.name for path in directory.iterdir()} != set(files):
        raise ValueError("Downloaded distribution file set differs from manifest")
    for name, expected in files.items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or distribution_hash(path) != expected
        ):
            raise ValueError(f"Release distribution hash mismatch: {name}")
    return manifest


def main() -> None:
    expected = os.environ.get("EXPECTED_RELEASE_MANIFEST_SHA")
    if not expected:
        raise ValueError("EXPECTED_RELEASE_MANIFEST_SHA is required")
    manifest = verify_artifacts(
        Path(sys.argv[1]),
        commit=os.environ["GITHUB_SHA"],
        tag=os.environ["GITHUB_REF_NAME"],
        expected_manifest_sha=expected,
    )
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"PRERELEASE={str(manifest['prerelease']).lower()}\n")
    print("Release artifact hashes and source commit verified")


if __name__ == "__main__":
    main()
