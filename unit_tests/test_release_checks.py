"""Release gates must reject publishing a mismatched or undocumented version."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


release_helpers = runpy.run_path(
    str(Path(__file__).parents[1] / "tools/check_release.py")
)
check_release = release_helpers["check_release"]
release_metadata = release_helpers["release_metadata"]


@pytest.fixture
def release_tree(tmp_path):
    package = tmp_path / "src/wavedl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '__version__ = "1.2.3"\nraise RuntimeError("must not import")\n'
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\nPending\n\n## [1.2.3] - 2026-09-08\nFixed normalization.\n\n## [1.2.2]\nOlder notes.\n"
    )
    return tmp_path


def test_release_reads_version_without_importing_package(release_tree):
    assert check_release(release_tree, "v1.2.3") == "Fixed normalization.\n"


def test_release_rejects_wrong_tag(release_tree):
    with pytest.raises(ValueError, match="does not match"):
        check_release(release_tree, "v1.2.4")


@pytest.mark.parametrize(
    "changelog", ["## [Unreleased]\nPending", "## [1.2.3]\n\n## [1.2.2]\nOld"]
)
def test_release_rejects_missing_notes(release_tree, changelog):
    (release_tree / "CHANGELOG.md").write_text(changelog)
    with pytest.raises(ValueError, match="no release notes"):
        check_release(release_tree, "v1.2.3")


@pytest.mark.parametrize(
    "version,prerelease",
    [
        ("1.2.3a1", True),
        ("1.2.3b1", True),
        ("1.2.3rc1", True),
        ("1.2.3.dev1", True),
        ("1.2.3", False),
        ("1.2.3.post1", False),
        ("1.2.3.post1.dev1", True),
    ],
)
def test_release_prerelease_metadata_matches_python_versions(
    release_tree, version, prerelease
):
    (release_tree / "src/wavedl/__init__.py").write_text(f"__version__ = {version!r}\n")
    (release_tree / "CHANGELOG.md").write_text(f"## [{version}]\nRelease notes.\n")
    assert check_release(release_tree, f"v{version}") == "Release notes.\n"
    assert release_metadata(version) == {"version": version, "prerelease": prerelease}


def test_release_rejects_invalid_python_version(release_tree):
    (release_tree / "src/wavedl/__init__.py").write_text('__version__ = "invalid"\n')
    (release_tree / "CHANGELOG.md").write_text("## [invalid]\nRelease notes.\n")
    with pytest.raises(ValueError, match="Invalid version"):
        check_release(release_tree, "vinvalid")


def test_release_cli_writes_notes_and_metadata(release_tree, tmp_path):
    script = release_tree / "tools/check_release.py"
    script.parent.mkdir()
    script.write_text(
        (Path(__file__).parents[1] / "tools/check_release.py").read_text()
    )
    notes_file = tmp_path / "notes.md"
    metadata_file = tmp_path / "metadata.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tag",
            "v1.2.3",
            "--notes-file",
            str(notes_file),
            "--metadata-file",
            str(metadata_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert notes_file.read_text() == "Fixed normalization.\n"
    assert json.loads(metadata_file.read_text()) == {
        "version": "1.2.3",
        "prerelease": False,
    }
