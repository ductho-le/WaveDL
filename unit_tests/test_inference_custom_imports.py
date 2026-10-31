"""Custom registration remains explicit across fresh training/inference processes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from wavedl.core.inference_cli import inference_defaults, parse_inference_args
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl_daemon.backends.contracts import UnsupportedPlan
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.runs.service import _test_config


def test_import_cli_defaults_are_independent_and_explicit():
    defaults = inference_defaults()
    assert defaults["import_modules"] == []
    defaults["import_modules"].append("changed")
    assert inference_defaults()["import_modules"] == []
    args = parse_inference_args(
        ["--checkpoint", "/checkpoint", "--data_path", "/data", "--import", "custom"]
    )
    assert args.import_modules == ["custom"]


@pytest.mark.parametrize(
    "modules", [None, "custom", [None], [1], [""], ["--device"], ["-x"], ["bad\0name"]]
)
def test_inference_imports_reject_malformed_or_option_looking_values(tmp_path, modules):
    config = {
        "checkpoint": "/checkpoint",
        "data_path": "/data",
        "import_modules": modules,
    }
    with pytest.raises(ValueError):
        _test_config(config)
    run_id = str(uuid4())
    with pytest.raises((ValueError, UnsupportedPlan)):
        manifest = build_manifest(
            RunLayout(tmp_path / run_id, run_id),
            operation=OperationKind.TEST,
            resolved_config=config,
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            submission_cwd=tmp_path,
        )
        LocalBackend().plan(manifest)


@pytest.mark.parametrize("modules", [[], ["custom.module", "/modules/custom model.py"]])
@pytest.mark.parametrize("operation", [OperationKind.TEST, OperationKind.PLOT])
def test_inference_plan_preserves_explicit_import_list(tmp_path, modules, operation):
    config = _test_config(
        {"checkpoint": "/checkpoint", "data_path": "/data", "import_modules": modules}
    )
    run_id = str(uuid4())
    manifest = build_manifest(
        RunLayout(tmp_path / run_id, run_id),
        operation=operation,
        resolved_config=config,
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    plan = LocalBackend().plan(manifest)
    assert plan.manifest.resolved_config["import_modules"] == modules
    if modules:
        start = plan.argv.index("--import")
        assert list(plan.argv[start + 1 : start + 1 + len(modules)]) == modules
    else:
        assert "--import" not in plan.argv


@pytest.fixture(scope="module")
def custom_training(tmp_path_factory):
    root = tmp_path_factory.mktemp("fresh-custom-registration")
    module = root / "custom model.py"
    module.write_text(
        "import math\n"
        "from torch import nn\n"
        "from wavedl.models import BaseModel, register_model\n"
        "@register_model('fresh_custom')\n"
        "class CustomModel(BaseModel):\n"
        "    def __init__(self, in_shape, out_size, **kwargs):\n"
        "        super().__init__(in_shape, out_size)\n"
        "        self.linear = nn.Linear(math.prod(in_shape), out_size)\n"
        "    def forward(self, inputs):\n"
        "        return self.linear(inputs.flatten(1))\n"
    )
    data = root / "data.npz"
    rng = np.random.default_rng(2)
    np.savez(
        data,
        X=rng.normal(size=(16, 32)).astype("float32"),
        y=rng.normal(size=(16, 2)).astype("float32"),
    )
    environment = dict(
        os.environ, PYTHONPATH=str(Path("src").resolve()), OMP_NUM_THREADS="1"
    )
    source = root / "training"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wavedl.train",
            "--model",
            "fresh_custom",
            "--import",
            module.name,
            "--data_path",
            str(data),
            "--output_dir",
            str(source),
            "--epochs",
            "1",
            "--workers",
            "0",
            "--batch_size",
            "4",
            "--precision",
            "no",
        ],
        env=environment,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return root, module, data, source, environment


@pytest.mark.parametrize("operation", ["test", "plot", "export"])
def test_fresh_process_consumes_explicit_custom_registration(
    custom_training, operation
):
    root, module, data, source, environment = custom_training
    if operation == "export":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    output = root / operation
    config = _test_config(
        {
            "checkpoint": str(source),
            "data_path": str(data),
            "model": "fresh_custom",
            "import_modules": [module.name],
            "workers": 0,
            "save_predictions": operation == "test",
            "plot": operation == "plot",
            "export": "onnx" if operation == "export" else None,
        }
    )
    run_id = str(uuid4())
    kind = OperationKind.PLOT if operation == "plot" else OperationKind.TEST
    manifest = build_manifest(
        RunLayout(output, run_id),
        operation=kind,
        resolved_config=config | {"device": "cpu", "operation": kind.value},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=root,
    )
    plan = LocalBackend().plan(manifest)
    # Local execution starts in its owned run, not the submission project.
    # A relative custom-file import must already be normalized in approval.
    output.mkdir()
    result = subprocess.run(
        plan.argv,
        env=environment,
        cwd=plan.working_directory,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert plan.manifest.resolved_config["import_modules"] == [str(module)]
    assert json.loads((output / "manifest.json").read_text())["operation"] == kind.value
    rows = [
        json.loads(line)
        for line in (output / "attempts/1/events.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["payload"]["status"] == "completed"
    artifacts = [row["payload"] for row in rows if row["type"] == "artifact"]
    assert artifacts
    if operation == "plot":
        assert len([item for item in artifacts if item["kind"] == "plot"]) == 10
    if operation == "export":
        assert any(item["path"].endswith(".onnx") for item in artifacts)


@pytest.mark.parametrize("module", [None, "missing_registration_module"])
def test_custom_source_never_supplies_implicit_registration(custom_training, module):
    root, _module_path, data, source, environment = custom_training
    output = root / f"failed-{module}"
    arguments = [
        sys.executable,
        "-m",
        "wavedl.test",
        "--checkpoint",
        str(source),
        "--data_path",
        str(data),
        "--output_dir",
        str(output),
        "--device",
        "cpu",
        "--workers",
        "0",
    ]
    if module is not None:
        arguments.extend(["--import", module])
    result = subprocess.run(
        arguments, env=environment, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1
    assert (
        "Model 'fresh_custom' not found" if module is None else module
    ) in result.stderr
    rows = [
        json.loads(line)
        for line in (output / "attempts/1/events.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["payload"]["status"] == "failed"
    assert any(row["type"] == "error" for row in rows)
    assert not any(row["type"] == "artifact" for row in rows)


def test_fresh_process_imports_named_module_only_when_explicit(custom_training):
    root, module, data, source, environment = custom_training
    (root / "named_registration.py").write_text(module.read_text())
    environment = environment | {
        "PYTHONPATH": os.pathsep.join([environment["PYTHONPATH"], str(root)])
    }
    output = root / "named-module"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wavedl.test",
            "--checkpoint",
            str(source),
            "--data_path",
            str(data),
            "--output_dir",
            str(output),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--import",
            "named_registration",
            "--save_predictions",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "predictions.csv").is_file()
