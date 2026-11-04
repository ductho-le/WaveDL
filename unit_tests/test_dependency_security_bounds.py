"""Declared floors must not reintroduce the reviewed dependency advisories.

These are selected constraint regressions, not a full dependency vulnerability
scan or proof that every version admitted by a range passes runtime acceptance.
"""

import shlex
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def requirements(items):
    return {item.name: item for item in map(Requirement, items)}


@pytest.mark.parametrize(
    "group,package,affected,patched",
    [
        ("build", "setuptools", ["76.0.0", "78.1.1", "82.0.1"], "83.0.0"),
        ("core", "starlette", ["0.52.1", "1.0.0", "1.2.0"], "1.3.0"),
        ("core", "h11", ["0.14.0", "0.15.0"], "0.16.0"),
        # Upstream's 1.9.0 security baseline pairs torch 2.13 with torchvision 0.28.
        ("core", "torch", ["2.0.0", "2.5.1", "2.6.0"], "2.13.0"),
    ],
)
def test_reviewed_security_floors(group, package, affected, patched):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    items = (
        config["build-system"]["requires"]
        if group == "build"
        else config["project"]["dependencies"]
        if group == "core"
        else config["project"]["optional-dependencies"][group]
    )
    constraint = requirements(items)[package].specifier
    assert patched in constraint
    assert all(version not in constraint for version in affected)


@pytest.mark.parametrize(
    "workflow,job,package,patched,affected",
    [
        ("test.yml", "daemon-native", "setuptools", "83.0.0", ["78.1.1", "82.0.1"]),
        (
            "test.yml",
            "daemon-native",
            "starlette",
            "1.3.0",
            ["0.52.1", "1.0.0", "1.2.0"],
        ),
        ("test.yml", "daemon-native", "h11", "0.16.0", ["0.14.0", "0.15.0"]),
        (
            "test.yml",
            "web-client",
            "starlette",
            "1.3.0",
            ["0.52.1", "1.0.0", "1.2.0"],
        ),
    ],
)
def test_explicit_ci_install_floors_match_reviewed_constraints(
    workflow, job, package, patched, affected
):
    config = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    found = []
    for step in config["jobs"][job]["steps"]:
        command = step.get("run", "")
        if "pip install" not in command:
            continue
        for argument in shlex.split(command):
            if not argument.startswith(package):
                continue
            try:
                requirement = Requirement(argument)
            except ValueError:
                continue
            if requirement.name == package:
                found.append(requirement.specifier)
    assert found, f"Missing explicit {package} install in {workflow}:{job}"
    assert all(
        patched in spec and all(version not in spec for version in affected)
        for spec in found
    )
