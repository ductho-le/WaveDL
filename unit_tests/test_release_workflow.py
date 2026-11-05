"""Release publication must depend on gates and reuse one exact artifact."""

import os
from pathlib import Path

import pytest
import yaml

from tools.verify_release_artifacts import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def test_release_waits_for_reusable_acceptance_and_exact_package():
    release = workflow("release.yml")
    jobs = release["jobs"]
    assert set(jobs["release"]["needs"]) == {"test", "package"}
    assert jobs["test"]["uses"] == "./.github/workflows/test.yml"
    assert "if" not in jobs["test"]
    called = workflow("test.yml")
    # PyYAML uses YAML 1.1, where unquoted GitHub's `on` parses as True.
    assert "workflow_call" in called[True]
    assert "concurrency" not in called
    assert release["permissions"] == called["permissions"] == {"contents": "read"}
    for job in called["jobs"].values():
        assert job.get("permissions", called["permissions"]) == {"contents": "read"}
    for gate in ("test", "distribution", "web-client"):
        assert "if" not in called["jobs"][gate]
    assert "if" not in jobs["release"]
    assert jobs["publish"]["needs"] == ["release", "package"]


def test_release_static_client_gate_requires_complete_quality_checks():
    static = workflow("test.yml")["jobs"]["web-client"]
    assert static["defaults"]["run"]["working-directory"] == "src/wavedl_daemon/web"
    commands = [step["run"] for step in static["steps"] if "run" in step]
    required = [
        "pnpm install --frozen-lockfile",
        "pnpm run typecheck",
        "pnpm test",
        "pnpm run generate",
        "python ../../../tools/vendor_web_assets.py",
    ]
    assert all(command in commands for command in required)
    assert [commands.index(command) for command in required] == sorted(
        commands.index(command) for command in required
    )
    assert all("if" not in step for step in static["steps"])
    lint = workflow("lint.yml")["jobs"]["web-lint"]
    assert "if" not in lint
    assert lint["defaults"]["run"]["working-directory"] == "src/wavedl_daemon/web"
    assert [step["run"] for step in lint["steps"] if "run" in step] == [
        "pnpm install --frozen-lockfile",
        "pnpm run lint",
        "pnpm run format:check",
    ]
    assert all("if" not in step for step in lint["steps"])
    checkout = next(
        step for step in lint["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"]["persist-credentials"] is False


@pytest.mark.parametrize(
    ("filename", "job"),
    [
        ("lint.yml", "web-lint"),
        ("test.yml", "web-client"),
        ("test.yml", "test"),
        ("release.yml", "package"),
    ],
)
def test_web_build_jobs_set_up_pinned_pnpm_before_node_cache(filename, job):
    steps = workflow(filename)["jobs"][job]["steps"]
    setup = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("pnpm/action-setup@")
    )
    node = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/setup-node@")
    )
    assert setup < node
    assert steps[setup]["uses"] == "pnpm/action-setup@v6"
    assert steps[node]["uses"] == "actions/setup-node@v4"
    assert steps[node]["with"]["node-version"] == "24"
    assert steps[setup]["with"] == {
        "package_json_file": "src/wavedl_daemon/web/package.json"
    }
    options = steps[node]["with"]
    assert options["cache"] == "pnpm"
    assert options["cache-dependency-path"] == ("src/wavedl_daemon/web/pnpm-lock.yaml")
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "pnpm install --frozen-lockfile" in commands
    assert "npm ci" not in commands


def test_release_rejects_web_source_in_distributions():
    steps = workflow("release.yml")["jobs"]["package"]["steps"]
    release = next(
        step["run"]
        for step in steps
        if step.get("name") == "Verify version and wheel/sdist web asset identity"
    )
    assert 'not any(name.startswith("wavedl_daemon/web/")' in release
    assert 'not any("/src/wavedl_daemon/web/" in name' in release


def test_release_is_headless_without_reducing_shipped_test_gates():
    release = workflow("release.yml")
    jobs = release["jobs"]
    assert "gui" not in jobs
    assert jobs["test"]["uses"] == "./.github/workflows/test.yml"
    shared = workflow("test.yml")
    test = shared["jobs"]["test"]
    assert test["strategy"]["matrix"]["python-version"] == ["3.11", "3.12", "3.13"]
    commands = "\n".join(step.get("run", "") for step in test["steps"])
    assert 'python -m pip install -e ".[dev]"' in commands
    assert (
        'python -m pytest unit_tests/ -v --tb=short -m "not slow and not gpu"'
        in commands
    )
    assert "python -m pip check" in commands
    assert "QT_QPA_PLATFORM" not in test["env"]
    for retired in ("Qt", ",gui", "test_gui_", "verify_gui_ci_report"):
        assert retired not in (ROOT / ".github/workflows/release.yml").read_text()
    assert release["permissions"] == {"contents": "read"}
    assert jobs["release"]["permissions"] == {"contents": "write"}
    for job in [*jobs.values(), *shared["jobs"].values()]:
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


def test_retired_desktop_workflow_and_paths_are_not_active():
    assert not (ROOT / ".github/workflows/daemon-gui-integration.yml").exists()
    tests = workflow("test.yml")
    assert tests[True] == {
        "push": {"branches": ["main"]},
        "pull_request": {"branches": ["main"]},
        "workflow_dispatch": None,
        "workflow_call": None,
    }
    assert workflow("lint.yml")[True] == {
        "push": {"branches": ["main"]},
        "pull_request": {"branches": ["main"]},
    }
    assert set(tests["jobs"]) == {"test", "architecture", "distribution", "web-client"}
    lint = workflow("lint.yml")["jobs"]["lint"]
    lint_commands = "\n".join(step.get("run", "") for step in lint["steps"])
    assert "ruff check . --output-format=github" in lint_commands
    assert "ruff format . --check --diff" in lint_commands


def test_shared_test_matrix_has_exactly_one_windows_cell():
    job = workflow("test.yml")["jobs"]["test"]
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job["defaults"]["run"]["shell"] == "bash"
    assert job["strategy"]["fail-fast"] is False
    matrix = job["strategy"]["matrix"]
    assert matrix == {
        "python-version": ["3.11", "3.12", "3.13"],
        "os": ["ubuntu-latest"],
        "runtime": ["latest"],
        "include": [
            {"python-version": "3.11", "os": "ubuntu-latest", "runtime": "minimum"},
            {"python-version": "3.11", "os": "windows-latest", "runtime": "latest"},
            {"python-version": "3.11", "os": "macos-latest", "runtime": "latest"},
        ],
    }
    windows = [cell for cell in matrix["include"] if cell["os"] == "windows-latest"]
    assert len(windows) == 1
    cache = next(
        step for step in job["steps"] if step.get("name") == "Cache pip dependencies"
    )
    assert "${{ matrix.os }}" in cache["with"]["key"]
    upload = next(
        step for step in job["steps"] if step.get("name") == "Upload unit test evidence"
    )
    assert "${{ matrix.os }}" in upload["with"]["name"]


def test_release_and_pypi_consume_the_same_verified_distribution():
    jobs = workflow("release.yml")["jobs"]
    for name in ("release", "publish"):
        (download,) = (
            step
            for step in jobs[name]["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        )
        assert download["with"] == {"name": "wavedl-release-dist", "path": "verified"}
        steps = jobs[name]["steps"]
        check = next(
            index
            for index, step in enumerate(steps)
            if "verify_release_artifacts.py verified" in step.get("run", "")
        )
        publish = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses", "").startswith(("softprops/", "pypa/"))
        )
        assert check < publish and "if" not in steps[check]
        assert (
            steps[check]["env"]["EXPECTED_RELEASE_MANIFEST_SHA"]
            == "${{ needs.package.outputs.manifest_digest }}"
        )
        assert (
            steps[check]["run"] == "python tools/verify_release_artifacts.py verified"
        )
        assert any(
            step.get("uses", "").startswith("actions/checkout@")
            for step in steps[:check]
        )
        assert not any(
            "python -m build" in step.get("run", "") for step in jobs[name]["steps"]
        )
    package = jobs["package"]["steps"]
    upload = next(
        step
        for step in package
        if step.get("with", {}).get("name") == "wavedl-release-dist"
    )
    assert upload["with"]["if-no-files-found"] == "error"
    assert "if" not in upload
    assert "release-artifacts.json" in upload["with"]["path"]
    assert "tools/verify_release_artifacts.py" not in upload["with"]["path"]
    commands = "\n".join(step.get("run", "") for step in package)
    assert "WAVEDL_ACCEPTANCE_WHEEL=" in commands
    assert "WAVEDL_ACCEPTANCE_WHEELHOUSE=" in commands
    assert "--noconftest -k isolated_resolved" in commands
    assert "Wheel differs from verified generated web inventory" in commands
    assert "Sdist differs from verified generated web inventory" in commands
    assert "WAVEDL_ACCEPTANCE_SDIST=" in commands
    assert "--junitxml=release-sdist-installed.xml" in commands
    assert "'setuptools>=83.0.0,<85' wheel --only-binary=:all:" in commands
    sdist = next(
        index
        for index, step in enumerate(package)
        if "WAVEDL_ACCEPTANCE_SDIST=" in step.get("run", "")
    )
    upload_index = package.index(upload)
    guard = next(
        index
        for index, step in enumerate(package)
        if step.get("name") == "Require isolated exact-sdist acceptance with zero skips"
    )
    assert sdist < guard < upload_index
    assert package[guard]["if"] == "always()"
    assert 'ET.parse("release-sdist-installed.xml")' in package[guard]["run"]
    assert jobs["publish"]["environment"] == "pypi"
    assert jobs["publish"]["permissions"]["id-token"] == "write"


def test_release_generates_assets_before_tests_and_classifies_pep440():
    jobs = workflow("release.yml")["jobs"]
    assert jobs["test"]["uses"] == "./.github/workflows/test.yml"
    steps = workflow("test.yml")["jobs"]["test"]["steps"]
    vendor = next(
        index
        for index, step in enumerate(steps)
        if "vendor_web_assets.py" in step.get("run", "")
    )
    test = next(
        index
        for index, step in enumerate(steps)
        if "pytest unit_tests" in step.get("run", "")
    )
    assert vendor < test
    commands = "\n".join(step.get("run", "") for step in jobs["package"]["steps"])
    assert "version.is_prerelease or version.is_devrelease" in commands
    release = next(
        step
        for step in jobs["release"]["steps"]
        if step.get("uses", "").startswith("softprops/")
    )
    assert (
        release["with"]["prerelease"]
        == "${{ steps.verified.outputs.PRERELEASE == 'true' }}"
    )


def test_push_test_matrix_generates_and_vendors_browser_before_wheel_acceptance():
    jobs = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())["jobs"]
    steps = jobs["test"]["steps"]
    names = [step.get("name") for step in steps]
    required_names = [
        "Install locked browser dependencies",
        "Generate the browser client from the locked source",
        "Install dependencies",
        "Verify installed dependency contracts",
        "Vendor generated browser assets before package tests",
        "Prepare an isolated CPU wheelhouse",
        "Require dependency-resolved wheel installation outside the checkout",
    ]
    assert all(name in names for name in required_names)
    assert all(
        steps[names.index(name)].get("if") == "matrix.python-version == '3.11'"
        for name in required_names[-2:]
    )
    commands = [step["run"] for step in steps if "run" in step]
    required = [
        "pnpm install --frozen-lockfile",
        "pnpm run generate",
        "python tools/vendor_web_assets.py",
        'python -m pip wheel "."',
        "python -m pytest unit_tests/daemon/test_installed_cli.py",
    ]
    assert all(command in "\n".join(commands) for command in required)
    indices = [
        next(
            index for index, command in enumerate(commands) if required_item in command
        )
        for required_item in required
    ]
    assert indices == sorted(indices)
    step_indices = [names.index(name) for name in required_names]
    assert step_indices == sorted(step_indices)
    install = steps[names.index("Install dependencies")]
    assert 'python -m pip install -e ".[dev]"' in install["run"]
    assert (
        "python -m pip check"
        in steps[names.index("Verify installed dependency contracts")]["run"]
    )
    pnpm_steps = [
        step
        for step in steps
        if step.get("run") in {"pnpm install --frozen-lockfile", "pnpm run generate"}
    ]
    assert all(
        step["working-directory"] == "src/wavedl_daemon/web" for step in pnpm_steps
    )


@pytest.fixture
def release_artifacts(tmp_path):
    import hashlib
    import json

    directory = tmp_path / "dist"
    directory.mkdir()
    files = {}
    for name in ("wavedl-1.0.0-py3-none-any.whl", "wavedl-1.0.0.tar.gz"):
        path = directory / name
        path.write_bytes(name.encode())
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "source_commit": "abc",
        "tag": "v1.0.0",
        "prerelease": False,
        "files": files,
    }
    (tmp_path / "release-artifacts.json").write_text(json.dumps(manifest))
    return tmp_path


def test_release_hash_manifest_accepts_exact_bytes(release_artifacts):
    import hashlib

    digest = hashlib.sha256(
        (release_artifacts / "release-artifacts.json").read_bytes()
    ).hexdigest()
    assert (
        verify_artifacts(
            release_artifacts, commit="abc", tag="v1.0.0", expected_manifest_sha=digest
        )["prerelease"]
        is False
    )


def test_release_hash_manifest_works_without_posix_open_flags(
    release_artifacts, monkeypatch
):
    monkeypatch.delattr(os, "O_NONBLOCK", raising=False)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    assert (
        verify_artifacts(release_artifacts, commit="abc", tag="v1.0.0")["prerelease"]
        is False
    )


def test_manifest_anchor_rejects_joint_distribution_and_manifest_tampering(
    release_artifacts,
):
    import hashlib
    import json

    path = release_artifacts / "release-artifacts.json"
    trusted_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.read_text())
    wheel = next((release_artifacts / "dist").glob("*.whl"))
    wheel.write_bytes(b"tampered wheel")
    manifest["files"][wheel.name] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trusted package output"):
        verify_artifacts(
            release_artifacts,
            commit="abc",
            tag="v1.0.0",
            expected_manifest_sha=trusted_digest,
        )


def test_cli_requires_the_trusted_manifest_anchor(release_artifacts, monkeypatch):
    import sys

    from tools.verify_release_artifacts import main

    monkeypatch.delenv("EXPECTED_RELEASE_MANIFEST_SHA", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["verify_release_artifacts.py", str(release_artifacts)]
    )
    with pytest.raises(ValueError, match="EXPECTED_RELEASE_MANIFEST_SHA is required"):
        main()


@pytest.mark.parametrize("posix_open_flags", [True, False])
@pytest.mark.parametrize(
    "fault",
    [
        "symlink",
        "oversized",
        "directory",
        pytest.param(
            "fifo",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="Windows has no os.mkfifo"
            ),
        ),
    ],
)
def test_manifest_requires_a_bounded_regular_file(
    release_artifacts, fault, posix_open_flags, monkeypatch
):
    if not posix_open_flags:
        monkeypatch.delattr(os, "O_NONBLOCK", raising=False)
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    path = release_artifacts / "release-artifacts.json"
    if fault == "oversized":
        path.write_bytes(b" " * 65537)
    else:
        retained = release_artifacts / "manifest-retained"
        path.rename(retained)
        if fault == "symlink":
            path.symlink_to(retained)
        elif fault == "directory":
            path.mkdir()
        elif fault == "fifo":
            os.mkfifo(path)
    with pytest.raises((ValueError, OSError)):
        verify_artifacts(release_artifacts, commit="abc", tag="v1.0.0")


@pytest.mark.parametrize(
    "fault", ["tampered", "extra", "missing", "wrong_commit", "wrong_tag", "symlink"]
)
def test_release_hash_manifest_fails_closed(release_artifacts, fault):
    root = release_artifacts
    path = next((root / "dist").glob("*.whl"))
    if fault == "tampered":
        path.write_bytes(b"changed")
    elif fault == "extra":
        (root / "dist" / "unexpected.txt").write_bytes(b"extra")
    elif fault == "missing":
        path.unlink()
    elif fault == "symlink":
        content = root / "wheel-bytes"
        path.rename(content)
        path.symlink_to(content)
    with pytest.raises(ValueError):
        verify_artifacts(
            root,
            commit="wrong" if fault == "wrong_commit" else "abc",
            tag="wrong" if fault == "wrong_tag" else "v1.0.0",
        )
