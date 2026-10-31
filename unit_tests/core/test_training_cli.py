"""Frozen argument contract for the dependency-light training schema."""

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl.core.training_cli import (
    build_training_parser,
    parse_training_args,
    training_defaults,
)


EXPECTED_DEFAULTS = {
    "model": "cnn",
    "list_models": False,
    "import_modules": [],
    "pretrained": False,
    "config": None,
    "batch_size": 128,
    "lr": 1e-3,
    "epochs": 1000,
    "patience": 20,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "grad_accum_steps": 1,
    "loss": "mse",
    "huber_delta": 1.0,
    "loss_weights": None,
    "selection_metric": "loss",
    "allow_pickle": False,
    "optimizer": "adamw",
    "momentum": 0.9,
    "nesterov": False,
    "betas": "0.9,0.999",
    "scheduler": "plateau",
    "scheduler_patience": 10,
    "min_lr": 1e-6,
    "scheduler_factor": 0.5,
    "warmup_epochs": 5,
    "step_size": 30,
    "milestones": None,
    "data_path": "train_data.npz",
    "workers": -1,
    "seed": 2025,
    "deterministic": False,
    "cache_validate": "sha256",
    "single_channel": False,
    "cv": 0,
    "cv_stratify": False,
    "cv_bins": 10,
    "resume": None,
    "save_every": 50,
    "output_dir": ".",
    "fresh": False,
    "compile": False,
    "precision": "bf16",
    "constraint": [],
    "constraint_file": None,
    "constraint_weight": [0.1],
    "constraint_reduction": "mse",
    "wandb": False,
    "wandb_watch": False,
    "project_name": "DL-Training",
    "run_name": None,
}


def test_defaults_preserve_all_original_destinations_and_values():
    args, parser = parse_training_args([])
    assert isinstance(parser, argparse.ArgumentParser)
    assert vars(args) == EXPECTED_DEFAULTS
    assert training_defaults() == EXPECTED_DEFAULTS


def test_defaults_and_parsers_do_not_share_mutable_values():
    defaults = training_defaults()
    defaults["import_modules"].append("custom")
    defaults["constraint"].append("y0")
    defaults["constraint_weight"].append(9.0)
    assert training_defaults() == EXPECTED_DEFAULTS
    assert vars(parse_training_args([])[0]) == EXPECTED_DEFAULTS


@pytest.mark.parametrize("flag", ["--precision", "--mixed_precision"])
@pytest.mark.parametrize("value", ["bf16", "fp16", "no"])
def test_precision_alias_uses_same_destination(flag, value):
    args, _ = parse_training_args([flag, value])
    assert args.precision == value
    assert "mixed_precision" not in vars(args)


def test_precision_duplicate_action_default_keeps_effective_bf16():
    parser = build_training_parser()
    actions = [action for action in parser._actions if action.dest == "precision"]
    assert [action.default for action in actions] == ["bf16", None]
    assert parser.parse_args([]).precision == "bf16"
    assert (
        parser.parse_args(["--precision", "fp16", "--mixed_precision", "no"]).precision
        == "no"
    )


def test_import_constraints_and_numeric_types():
    args, _ = parse_training_args(
        [
            "--import",
            "custom.one",
            "custom.two",
            "--constraint",
            "y0 - y1",
            "y2",
            "--constraint_weight",
            "0.2",
            "0.7",
            "--batch_size",
            "16",
            "--lr",
            "0.002",
            "--config",
            "settings.yaml",
        ]
    )
    assert args.import_modules == ["custom.one", "custom.two"]
    assert args.constraint == ["y0 - y1", "y2"]
    assert args.constraint_weight == [0.2, 0.7]
    assert type(args.batch_size) is int
    assert type(args.lr) is float
    assert args.config == "settings.yaml"


@pytest.mark.parametrize(
    "destination", [key for key, value in EXPECTED_DEFAULTS.items() if value is False]
)
def test_boolean_flags_remain_store_true(destination):
    args, _ = parse_training_args([f"--{destination}"])
    assert getattr(args, destination) is True


@pytest.mark.parametrize(
    ("flag", "choices"),
    [
        ("--loss", ["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"]),
        ("--optimizer", ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]),
        (
            "--scheduler",
            [
                "plateau",
                "cosine",
                "cosine_restarts",
                "onecycle",
                "step",
                "multistep",
                "exponential",
                "linear_warmup",
            ],
        ),
        ("--cache_validate", ["sha256", "fast", "size"]),
        ("--precision", ["bf16", "fp16", "no"]),
        ("--mixed_precision", ["bf16", "fp16", "no"]),
        ("--constraint_reduction", ["mse", "mae"]),
    ],
)
def test_fixed_choices_are_not_changed_by_registry_help(flag, choices):
    parser = build_training_parser(models=["custom"], losses=["custom"])
    action = next(action for action in parser._actions if flag in action.option_strings)
    assert action.choices == choices
    with pytest.raises(SystemExit) as error:
        parser.parse_args([flag, "not-a-choice"])
    assert error.value.code == 2


def test_registry_names_enrich_help_and_alias_stays_hidden(capsys):
    parser = build_training_parser(
        models=["cnn", "custom"],
        losses=["mse"],
        optimizers=["adamw"],
        schedulers=["plateau"],
    )
    help_text = parser.format_help()
    assert "Universal DDP Training Pipeline" in help_text
    for names in (["cnn", "custom"], ["mse"], ["adamw"], ["plateau"]):
        assert f"Available: {names}" in " ".join(help_text.split())
    assert "--mixed_precision" not in help_text
    assert "--precision" in help_text
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0
    assert capsys.readouterr().out == help_text


def test_training_wrapper_keeps_dependency_free_parser_contract(monkeypatch):
    """Execute just the wrapper AST, without import-time ML/cache side effects."""
    source = Path(__file__).resolve().parents[2] / "src/wavedl/train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    namespace = {"argparse": argparse, "parse_training_args": parse_training_args}
    for name in ("models", "losses", "optimizers", "schedulers"):
        namespace[f"list_{name}"] = lambda: ["runtime-registry"]
    exec(
        compile(ast.Module(body=[wrapper], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    monkeypatch.setattr(sys, "argv", ["wavedl.train", "--epochs", "3"])
    args, parser = namespace["parse_args"]()
    assert args.epochs == 3
    assert "Available: ['runtime-registry']" in parser.format_help()


def test_import_and_parse_without_site_packages_or_ml(tmp_path):
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "numpy", "PySide6", "wavedl_daemon", "wavedl_gui", "accelerate"
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from wavedl.core.training_cli import parse_training_args, training_defaults
assert parse_training_args(["--epochs", "2"])[0].epochs == 2
assert training_defaults()["precision"] == "bf16"
"""
    source_root = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(source_root)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
