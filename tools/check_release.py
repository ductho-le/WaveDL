"""Validate release tags and extract notes without importing the package."""

import argparse
import ast
import json
import re
from pathlib import Path

from packaging.version import Version


def release_metadata(version):
    parsed = Version(version)
    return {"version": version, "prerelease": parsed.is_prerelease}


def check_release(root, tag):
    root = Path(root)
    tree = ast.parse((root / "src/wavedl/__init__.py").read_text())
    version = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )
    release_metadata(version)
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag} does not match package version {version}")
    changelog = (root / "CHANGELOG.md").read_text()
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise ValueError(f"Version {version} has no release notes")
    return match.group(1).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path)
    args = parser.parse_args()
    try:
        notes = check_release(Path(__file__).resolve().parents[1], args.tag)
    except ValueError as error:
        parser.error(str(error))
    args.notes_file.write_text(notes)
    if args.metadata_file:
        args.metadata_file.write_text(json.dumps(release_metadata(args.tag[1:])))


if __name__ == "__main__":
    main()
