"""Stage a verified static build into the daemon package, retaining old output.

Run from any directory with ``python tools/vendor_web_assets.py`` after generating
the web client. Only the repository's explicit .output/public is accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "wavedl_daemon" / "web" / ".output" / "public"
TARGET = ROOT / "src" / "wavedl_daemon" / "web_assets"
WORK_AREA = ROOT / "src" / "wavedl_daemon" / "web" / ".vendor-backups"
MANIFEST = ".wavedl-assets.json"
LEGACY_IMMUTABLE = ".wavedl-immutable.json"
MAX_FILE = 32 * 1024 * 1024
ASSET_SUFFIXES = {
    ".css",
    ".html",
    ".ico",
    ".js",
    ".json",
    ".png",
    ".svg",
    ".webmanifest",
    ".woff2",
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate generated asset manifest key")
        result[key] = value
    return result


def inventory(directory: Path, *, managed: bool = False) -> dict[str, str]:
    """Reject links, backend output and unbounded files before packaging."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Static build directory is missing or is a symlink")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink is not a static asset: {path}")
        relative = path.relative_to(directory).as_posix()
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_size > MAX_FILE:
            raise ValueError(f"Invalid or oversized static asset: {relative}")
        if managed and relative == MANIFEST:
            continue
        if managed and relative == LEGACY_IMMUTABLE:
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            continue
        if path.suffix not in ASSET_SUFFIXES or any(
            part.startswith(".") for part in path.relative_to(directory).parts
        ):
            raise ValueError(f"Unexpected static asset: {relative}")
        if relative.split("/")[0] in {"api", "server", "node_modules"}:
            raise ValueError(f"Backend output is not a static asset: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "index.html" not in result:
        raise ValueError("Expected generated SPA entry point index.html")
    return result


def main() -> None:
    assets = inventory(SOURCE)
    if TARGET.exists() or TARGET.is_symlink():
        previous = inventory(TARGET, managed=True)
        manifest = TARGET / MANIFEST
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError("Refusing to replace an unmanaged web_assets directory")
        if (
            json.loads(manifest.read_text(), object_pairs_hook=unique_object)
            != previous
        ):
            raise ValueError("Existing packaged assets differ from their manifest")
    if WORK_AREA.is_symlink():
        raise ValueError("Packaging work area must not be a symlink")
    WORK_AREA.mkdir(parents=True, exist_ok=True)
    if WORK_AREA.stat().st_dev != TARGET.parent.stat().st_dev:
        raise ValueError("Packaging stage and target must share a filesystem")
    stage = Path(tempfile.mkdtemp(prefix="stage-", dir=WORK_AREA))
    backup = None
    try:
        for relative in assets:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / relative, destination)
        if inventory(stage) != assets:
            raise ValueError("Static build changed while being staged")
        (stage / MANIFEST).write_text(
            json.dumps(assets, indent=2, sort_keys=True) + "\n"
        )
        if inventory(stage, managed=True) != assets:
            raise ValueError("Generated asset manifest differs from staged assets")
        total_bytes = sum((stage / relative).stat().st_size for relative in assets)
        if TARGET.exists():
            backup = Path(tempfile.mkdtemp(prefix="backup-", dir=WORK_AREA))
            backup.rmdir()
            os.replace(TARGET, backup)
        try:
            os.replace(stage, TARGET)
        except BaseException:
            if backup is not None:
                os.replace(backup, TARGET)
            raise
        print(
            f"Packaged {len(assets)} verified static assets "
            f"({total_bytes} bytes) into {TARGET}"
        )
        if backup is not None:
            print(f"Previous generated output retained for recovery: {backup}")
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
