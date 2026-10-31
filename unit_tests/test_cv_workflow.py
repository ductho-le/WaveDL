"""Keep the CPU CV cohort complete, successful, and mandatory in scientific CPU CI."""

import ast
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/test.yml"
FAMILY_COUNTS = {
    "unit_tests.test_cv_deterministic": 6,
    "unit_tests.test_cv_workers": 7,
    "unit_tests.test_cv_option_fidelity": 21,
    "unit_tests.test_cv_training_semantics": 20,
    "unit_tests.test_cv_physical_components": 11,
    "unit_tests.test_data_cv": 39,
}
REQUIRED_DATA_CASES = {
    "test_cross_validation_rejects_invalid_input_contract[X0-y0-2-samples and targets]",
    "test_cross_validation_rejects_invalid_input_contract[X1-y1-2-same number of samples]",
    "test_cross_validation_rejects_invalid_input_contract[X2-y2-2-at least 2 dimensions]",
    "test_cross_validation_rejects_invalid_input_contract[X3-y3-5-at least 5 samples]",
    "test_cross_validation_rejects_nonfinite_input_contract",
    "test_cross_validation_rejects_non_numeric_input_contract",
}
COHORT_PATHS = {f"{family.replace('.', '/')}.py" for family in FAMILY_COUNTS}


@pytest.fixture
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _cohort_step(workflow):
    return next(
        step
        for step in workflow["jobs"]["scientific-cpu"]["steps"]
        if "python -m pytest unit_tests/test_cv_option_fidelity.py"
        in step.get("run", "")
    )


def _guard_step(workflow):
    return next(
        step
        for step in workflow["jobs"]["scientific-cpu"]["steps"]
        if step.get("name") == "Require all CV semantics cases with zero skips"
    )


def _guard_script(workflow):
    run = _guard_step(workflow)["run"]
    prefix = "python - <<'PY'\n"
    assert run.startswith(prefix)
    script, terminator = run[len(prefix) :].rsplit("\nPY", 1)
    assert not terminator.strip()
    return script


def _report():
    """Mirror pytest's single-suite report with class-qualified data CV cases."""
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root, "testsuite", tests="104", skipped="0", failures="0", errors="0"
    )
    for family, count in FAMILY_COUNTS.items():
        classname = (
            f"{family}.TestDataCV" if family.endswith("test_data_cv") else family
        )
        for index in range(count - (6 if family.endswith("test_data_cv") else 0)):
            ET.SubElement(
                suite, "testcase", classname=classname, name=f"test_case[{index}]"
            )
    for name in sorted(REQUIRED_DATA_CASES):
        ET.SubElement(suite, "testcase", classname="unit_tests.test_data_cv", name=name)
    return root, suite


def _run_guard(workflow, tmp_path, root=None):
    if root is not None:
        ET.ElementTree(root).write(tmp_path / "cv-option-fidelity.xml")
    return subprocess.run(
        [sys.executable, "-c", _guard_script(workflow)],
        env={**os.environ, "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_cv_cohort_and_contract_trigger_paths_are_explicit(workflow, event):
    """Changes to any cohort file, helper, or contract test trigger this lane."""
    triggers = workflow.get("on", workflow.get(True))
    required = COHORT_PATHS | {
        "src/wavedl/utils/cv_components.py",
        "unit_tests/test_cv_workflow.py",
        ".github/workflows/test.yml",
    }
    assert "paths" not in triggers[event] or required.issubset(triggers[event]["paths"])


def test_cv_command_runs_all_six_files_without_skip_or_failure_escape(workflow):
    step = _cohort_step(workflow)
    tokens = shlex.split(step["run"])
    assert tokens[:3] == ["python", "-m", "pytest"]
    assert {token for token in tokens if token.endswith(".py")} == COHORT_PATHS
    assert "--junitxml=$TMPDIR/cv-option-fidelity.xml" in tokens
    assert not any(token in tokens[3:] for token in ("-k", "-m", "--collect-only"))
    assert "||" not in step["run"]
    assert not step.get("continue-on-error", False)
    assert not step.get("if")


def test_cv_guard_and_workflow_contract_tests_are_mandatory(workflow):
    steps = workflow["jobs"]["scientific-cpu"]["steps"]
    guard = _guard_step(workflow)
    assert guard["if"] == "always()"
    assert not guard.get("continue-on-error", False)
    assert "||" not in guard["run"]
    assert steps.index(_cohort_step(workflow)) < steps.index(guard)
    contract = next(
        step
        for step in steps
        if "pytest --noconftest unit_tests/test_cv_workflow.py" in step.get("run", "")
    )
    assert not contract.get("continue-on-error", False)
    assert not contract.get("if")
    assert "||" not in contract["run"]


def test_onecycle_update_and_resume_acceptance_is_mandatory(workflow):
    """The adjoining scientific-option gate cannot drop native CPU coverage."""
    path = "unit_tests/test_training_onecycle_accumulation.py"
    triggers = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert "paths" not in triggers[event] or path in triggers[event]["paths"]
    steps = workflow["jobs"]["scientific-cpu"]["steps"]
    test = next(step for step in steps if f"pytest {path}" in step.get("run", ""))
    assert shlex.split(test["run"]) == [
        "python",
        "-m",
        "pytest",
        path,
        "--junitxml=$TMPDIR/onecycle-updates.xml",
    ]
    assert not test.get("continue-on-error", False) and not test.get("if")
    guard = next(
        step
        for step in steps
        if step.get("name")
        == "Require CPU and native DDP OneCycle acceptance with zero skips"
    )
    assert guard["if"] == "always()"
    assert not guard.get("continue-on-error", False)
    assert steps.index(test) < steps.index(guard)
    assert "assert len(cases) == 7" in guard["run"]
    assert '("skipped", "failure", "error")' in guard["run"]
    assert "||" not in guard["run"]
    assert "onecycle-updates.xml" in steps[-1]["with"]["path"]


def test_cv_guard_accepts_complete_production_shaped_report(workflow, tmp_path):
    root, _suite = _report()
    result = _run_guard(workflow, tmp_path, root)
    assert result.returncode == 0, result.stderr


def test_cv_guard_tracks_real_source_contract_cases_without_importing_ml(workflow):
    """Read the actual parameter rows so a stale synthetic total cannot pass."""
    source = ast.parse(
        (ROOT / "unit_tests/test_data_cv.py").read_text(encoding="utf-8")
    )
    functions = {
        node.name: node for node in source.body if isinstance(node, ast.FunctionDef)
    }
    invalid = functions["test_cross_validation_rejects_invalid_input_contract"]
    (parameterization,) = invalid.decorator_list
    assert isinstance(parameterization, ast.Call)
    assert isinstance(parameterization.func, ast.Attribute)
    assert parameterization.func.attr == "parametrize"
    rows = parameterization.args[1]
    assert isinstance(rows, ast.List)
    expected = set()
    for index, row in enumerate(rows.elts):
        assert isinstance(row, ast.Tuple)
        folds, message = (ast.literal_eval(value) for value in row.elts[2:])
        expected.add(f"{invalid.name}[X{index}-y{index}-{folds}-{message}]")
    for name in (
        "test_cross_validation_rejects_nonfinite_input_contract",
        "test_cross_validation_rejects_non_numeric_input_contract",
    ):
        assert not functions[name].decorator_list
        expected.add(name)
    assert expected == REQUIRED_DATA_CASES
    tree = ast.parse(_guard_script(workflow))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "required_data_cases"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == expected


def test_cv_guard_rejects_absent_report(workflow, tmp_path):
    result = _run_guard(workflow, tmp_path)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "missing_deterministic_family",
        "duplicate_extra",
        "duplicate_replacement",
        "relabel_unknown_family",
        "relabel_known_family",
        "replace_required_data_case",
        "skipped",
        "error",
        "failure",
    ],
)
def test_cv_guard_rejects_incomplete_or_unsuccessful_report(
    workflow, tmp_path, mutation
):
    root, suite = _report()
    if mutation == "missing":
        suite.remove(suite[0])
    elif mutation == "missing_deterministic_family":
        for case in list(suite):
            if case.get("classname") == "unit_tests.test_cv_deterministic":
                suite.remove(case)
    elif mutation == "duplicate_extra":
        ET.SubElement(suite, "testcase", **suite[0].attrib)
    elif mutation == "duplicate_replacement":
        suite[1].attrib.update(suite[0].attrib)
    elif mutation == "relabel_unknown_family":
        suite[0].set("classname", "unit_tests.test_unrelated")
    elif mutation == "relabel_known_family":
        suite[0].set("classname", "unit_tests.test_cv_option_fidelity")
        suite[0].set("name", "test_relabelled_unique")
    elif mutation == "replace_required_data_case":
        case = next(case for case in suite if case.get("name") in REQUIRED_DATA_CASES)
        case.set("name", "test_unrelated_data_case")
    else:
        ET.SubElement(suite[0], mutation)
    # Keep aggregate counts plausible: the guard must inspect the actual cases.
    suite.set("tests", str(len(suite)))
    result = _run_guard(workflow, tmp_path, root)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
