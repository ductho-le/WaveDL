"""The daemon's option table matches upstream's command line and the browser's copy."""

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
from wavedl_daemon.runs.training_options import OPTIONS, TEST, TRAIN, flag, resolve


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


@pytest.mark.parametrize("operation", [Operation.TRAIN, Operation.TEST])
def test_every_table_option_is_upstream_with_a_compatible_shape(upstream, operation):
    for name, option in OPTIONS[operation].items():
        assert flag(name) in upstream[operation], name
        arguments, shown = upstream[operation][flag(name)]
        assert (option.type is bool) == (arguments == ""), name
        assert option.multiple == arguments.endswith("...]"), name
        if arguments.startswith("{"):
            choices = set(arguments.split(" ")[0][1:-1].split(","))
            assert option.choices and set(option.choices) <= choices, name
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
    rendered = [item for item in launch_argv(spec, Topology()) if item[:2] == "--"]
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
    assert resolve(Operation.TRAIN, {"project_name": "", "run_name": ""})["run_name"] == ""
    for config in (
        {"lr": None},
        {"constraint_weight": []},
        {"output_dir": "elsewhere"},
        {"allow_pickle": True},
        {"model": "--fresh"},
        {"epochs": True},
        {"loss": "unknown"},
        {"lr": 0},
        {"run_name": "-x"},
        {"resume": ""},
    ):
        with pytest.raises(ValueError):
            resolve(Operation.TRAIN, config)
    with pytest.raises(ValueError, match="required"):
        resolve(Operation.TEST, {"data_path": "d.npz"})
