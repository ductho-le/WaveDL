"""The daemon's option table matches upstream's command line and the browser's copy."""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl_daemon.backends.contracts import Runtime, Topology
from wavedl_daemon.backends.local import launch_argv
from wavedl_daemon.runs.layout import RESULT_OPTIONS
from wavedl_daemon.runs.model import Operation
from wavedl_daemon.runs.service import build_spec
from wavedl_daemon.runs.training_options import (
    OPTIONS,
    TEST,
    TRAIN,
    cli_args,
    flag,
    resolve,
)


ROOT = Path(__file__).resolve().parents[2]


def upstream_options(module: str) -> dict[str, tuple[str, str | None]]:
    """``--flag`` to its argument text and shown default, from ``--help``."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        env=os.environ | {"PYTHONPATH": str(ROOT / "src"), "COLUMNS": "10000"},
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    section = result.stdout.split("\noptions:\n", 1)[1].split("\n\n", 1)[0]
    options = {}
    for block in re.split(r"\n(?=  -)", section):
        invocation = re.split(r"\s{2,}", block.strip(), maxsplit=1)[0]
        shown = re.findall(r"\(default: (.*?)\)", block)
        for alias in invocation.split(", "):
            name, _, arguments = alias.partition(" ")
            options[name] = (arguments, shown[-1] if shown else None)
    return options


@pytest.fixture(scope="module")
def upstream():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("The trainer's --help needs torch")
    return {
        "launcher": upstream_options("wavedl.launcher"),
        Operation.TRAIN: upstream_options("wavedl.train"),
        Operation.TEST: upstream_options("wavedl.test"),
    }


@pytest.fixture
def upstream_required_options(upstream, monkeypatch):
    from wavedl import test

    required = {}

    def capture(parser: argparse.ArgumentParser) -> argparse.Namespace:
        for action in parser._actions:
            if action.required:
                for alias in action.option_strings:
                    required[alias] = (action.default, action.required)
        return argparse.Namespace()

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    test.parse_args()
    return required


@pytest.mark.parametrize("operation", [Operation.TRAIN, Operation.TEST])
def test_every_table_option_is_upstream_with_a_compatible_shape(
    upstream, upstream_required_options, operation
):
    for name, option in OPTIONS[operation].items():
        assert flag(name) in upstream[operation], name
        arguments, shown = upstream[operation][flag(name)]
        assert (option.type is bool) == (arguments == ""), name
        assert option.multiple == arguments.endswith("...]"), name
        if arguments.startswith("{"):
            choices = set(arguments.split(" ")[0][1:-1].split(","))
            assert option.choices and set(option.choices) <= choices, name
        if operation is Operation.TEST:
            assert option.required == (flag(name) in upstream_required_options), name
        if option.required:
            assert upstream_required_options[flag(name)] == (option.default, True), name
            # Python versions differ on whether required options show defaults.
            if shown is not None:
                assert str(option.default) == shown, name
        else:
            assert str(option.default) == shown, name


@pytest.mark.parametrize("operation", [Operation.TRAIN, Operation.TEST])
def test_every_flag_the_daemon_renders_is_upstream(upstream, tmp_path, operation):
    config = {"checkpoint": "best", "data_path": "data.npz"}
    spec = build_spec(
        tmp_path / "run",
        operation=operation,
        resolved_config=resolve(
            operation, {} if operation is Operation.TRAIN else config
        ),
        runtime=Runtime("python", "selected", "cpu", {}),
        submission_cwd=tmp_path,
    )
    accepted = upstream[operation] | (
        upstream["launcher"] if operation is Operation.TRAIN else {}
    )
    rendered = [
        item.partition("=")[0]
        for item in launch_argv(spec, Topology())
        if item[:2] == "--"
    ]
    for name in [*rendered, RESULT_OPTIONS[operation][0]]:
        assert name in accepted, name
    if operation is Operation.TRAIN:
        # Slurm adds these for a multi-machine launch.
        assert {"--main_process_ip", "--main_process_port"} <= set(accepted)


def test_browser_options_mirror_the_training_table():
    kinds = {bool: "boolean", int: "integer", float: "number", str: "string"}
    path = ROOT / "src/wavedl_daemon/web/shared/training-options.json"
    assert json.loads(path.read_text()) == [
        {
            "name": name,
            "kind": kinds[option.type],
            "multiple": option.multiple,
            "choices": list(option.choices),
            "default": option.default,
        }
        for name, option in TRAIN.items()
    ]


def test_resolution_fills_defaults_and_accepts_null_or_empty_only_as_defaults():
    resolved = resolve(Operation.TEST, {"checkpoint": "best", "data_path": "d.npz"})
    assert set(resolved) == set(TEST)
    assert resolved["plot_format"] == ["png"]
    assert resolve(Operation.TRAIN, {"constraint": [], "run_name": None})["lr"] == 1e-3
    assert (
        resolve(Operation.TRAIN, {"project_name": "", "run_name": ""})["run_name"] == ""
    )
    assert resolve(Operation.TRAIN, {"model": "--fresh"})["model"] == "--fresh"
    for config in (
        {"lr": None},
        {"constraint_weight": []},
        {"output_dir": "elsewhere"},
        {"allow_pickle": True},
        {"epochs": True},
        {"loss": "unknown"},
        {"lr": 0},
        {"constraint": ["-y0"]},
        {"import_modules": ["-m"]},
        {"resume": ""},
    ):
        with pytest.raises(ValueError):
            resolve(Operation.TRAIN, config)
    with pytest.raises(ValueError, match="required"):
        resolve(Operation.TEST, {"data_path": "d.npz"})
    with pytest.raises(ValueError):
        resolve(
            Operation.TEST,
            {"checkpoint": "best", "data_path": "d.npz", "param_names": ["-a"]},
        )


PARSE = """
import json, sys
from wavedl import launcher, test, train
operation, argv = json.loads(sys.argv[1])
sys.argv = ["wavedl", *argv]
if operation == "train":
    options, remaining = launcher.parse_args()
    sys.argv = ["wavedl.train", *remaining, "--precision", options.mixed_precision]
    parsed = train.parse_args()[0]
else:
    parsed = test.parse_args()
print(json.dumps(vars(parsed)))
"""


def upstream_parse(spec):
    argv = launch_argv(spec, Topology())
    result = subprocess.run(
        [sys.executable, "-c", PARSE, json.dumps([spec.operation.value, argv[4:]])],
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "operation,config",
    [
        (
            Operation.TRAIN,
            {
                "grad_clip": -1e-07,
                "min_lr": -2.5e-300,
                "seed": -7,
                "constraint_weight": [-1e-07, -2.0, 3e-30, -1e16, -0.0],
                "constraint": ["y0 - y1", "y0"],
                "run_name": "-x --fresh",
                "project_name": "",
            },
        ),
        (
            Operation.TEST,
            {
                "checkpoint": "best",
                "data_path": "data.npz",
                "input_key": "-X",
                "param_names": ["a b", "c"],
                "import_modules": ["models"],
            },
        ),
    ],
)
def test_upstream_parses_every_rendered_value_back(
    upstream, tmp_path, operation, config
):
    """Negative numbers and leading-dash text reach upstream as values, not flags."""
    spec = build_spec(
        tmp_path / "run",
        operation=operation,
        resolved_config=resolve(operation, config),
        runtime=Runtime("python", "selected", "cpu", {}),
        submission_cwd=tmp_path,
    )
    parsed = upstream_parse(spec)
    for name in config:
        assert parsed[name] == spec.resolved_config[name], name


def test_list_items_that_argparse_would_read_as_flags_are_refused():
    with pytest.raises(ValueError, match="Unsafe"):
        cli_args(Operation.TRAIN, {"constraint": ["-y0"]})
    assert cli_args(
        Operation.TRAIN, {"run_name": "-x", "constraint_weight": [-1e-07]}
    ) == [
        "--constraint_weight",
        "-0.0000001",
        "--run_name=-x",
    ]
