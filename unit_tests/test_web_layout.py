"""Ensure the workstation source is colocated without becoming Python package data."""

import subprocess
import tomllib
from pathlib import Path

import setuptools


ROOT = Path(__file__).resolve().parents[1]


def tracked(path: str) -> list[str]:
    # Existing checkouts keep an untracked web/node_modules after the move.
    return subprocess.run(
        ["git", "ls-files", "--", path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def test_web_source_lives_beside_daemon():
    assert "src/wavedl_daemon/web/package.json" in tracked("src/wavedl_daemon/web")
    assert tracked("web") == []


def test_web_source_is_not_a_python_package():
    # pyproject discovery defaults to namespace packages, which match plain dirs.
    find = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"][
        "packages"
    ]["find"]
    packages = setuptools.find_namespace_packages(
        where=str(ROOT / find["where"][0]),
        include=find["include"],
        exclude=find.get("exclude", ()),
    )
    assert not [
        name
        for name in packages
        if name == "wavedl_daemon.web" or name.startswith("wavedl_daemon.web.")
    ]
    assert not any(
        Path(path).name == "__init__.py" for path in tracked("src/wavedl_daemon/web")
    )


def test_app_lib_typescript_is_not_gitignored():
    ts = subprocess.run(
        ["git", "check-ignore", "-q", "src/wavedl_daemon/web/app/lib/probe.ts"],
        cwd=ROOT,
        check=False,
    )
    js = subprocess.run(
        ["git", "check-ignore", "-q", "src/wavedl_daemon/web/app/lib/probe.js"],
        cwd=ROOT,
        check=False,
    )
    assert ts.returncode == 1
    assert js.returncode == 0


def test_manifest_prunes_web_source():
    assert (
        "prune src/wavedl_daemon/web" in (ROOT / "MANIFEST.in").read_text().splitlines()
    )
