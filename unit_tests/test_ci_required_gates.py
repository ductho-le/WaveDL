"""Every workflow exposes mandatory, fail-closed jobs and steps."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

# These are the jobs branch protection and release automation depend on.  Keep this
# list explicit: adding a workflow without a gate should be a deliberate change.
REQUIRED_JOBS = {
    ".github/workflows/daemon-foundations.yml": {"storage-enospc", "foundations"},
    ".github/workflows/lint.yml": {"lint"},
    ".github/workflows/release.yml": {
        "test",
        "package",
        "foundations",
        "web-client",
        "release",
        "publish",
    },
    ".github/workflows/test.yml": {"test", "scientific-cpu"},
    ".github/workflows/web-client.yml": {"static-client"},
    ".github/workflows/windows-cpu-acceptance.yml": {"installed-cpu"},
}


def _workflow(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text())
    assert isinstance(value, dict), path
    return value


def _assert_fail_closed(node: object, location: str) -> None:
    if isinstance(node, dict):
        assert node.get("continue-on-error") is not True, location
        for key, value in node.items():
            _assert_fail_closed(value, f"{location}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_fail_closed(value, f"{location}[{index}]")


def test_every_workflow_has_the_required_jobs() -> None:
    workflow_dir = ROOT / ".github/workflows"
    actual = {str(path.relative_to(ROOT)) for path in workflow_dir.glob("*.yml")}
    assert actual == set(REQUIRED_JOBS)
    for path, required in REQUIRED_JOBS.items():
        jobs = _workflow(path).get("jobs", {})
        assert required <= set(jobs), path


def test_required_workflows_and_steps_never_continue_on_error() -> None:
    for path in REQUIRED_JOBS:
        workflow = _workflow(path)
        for job_name, job in workflow["jobs"].items():
            _assert_fail_closed(job, f"{path}:jobs.{job_name}")
