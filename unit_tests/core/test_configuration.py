"""Shared YAML configuration behavior without importing the runtime stack."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl.core.configuration import (
    _flatten_config,
    load_config,
    merge_config_with_args,
)
from wavedl.core.training_cli import parse_training_args


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, {}),
        ({"lr": 0.001}, {"lr": 0.001}),
        ({"optimizer": {"params": {"beta1": 0.9}}}, {"optimizer_params_beta1": 0.9}),
        ({"a": {}}, {}),
    ],
)
def test_flatten_preserves_original_key_path(config, expected):
    assert _flatten_config(config) == expected


def test_flatten_custom_parent_and_separator():
    assert _flatten_config({"a": {"b": 1}}, "parent", ".") == {"parent.a.b": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", {}),
        ("null", {}),
        (
            "optimizer:\n  lr: 0.002\nbetas: [0.9, 0.99]\n",
            {"optimizer_lr": 0.002, "betas": [0.9, 0.99]},
        ),
    ],
)
def test_load_empty_and_nested_yaml(tmp_path, text, expected):
    pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(text)
    assert load_config(str(path)) == expected


def test_missing_yaml_preserves_error(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(str(path))


def test_malformed_yaml_preserves_yaml_exception(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text("a: [")
    with pytest.raises(yaml.YAMLError):
        load_config(str(path))


@pytest.mark.parametrize("cli", [["--workers", "-1"], ["--workers=-1"]])
def test_explicit_default_cli_wins_over_yaml(cli):
    args, parser = parse_training_args(cli)
    result = merge_config_with_args(
        {"workers": 8, "epochs": 12}, args, parser, argv=cli
    )
    assert result is args
    assert args.workers == -1
    assert args.epochs == 12


def test_explicit_tokens_override_ambient_sys_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launcher", "--workers=-1"])
    args, parser = parse_training_args([])
    merge_config_with_args({"workers": 8}, args, parser, argv=[])
    assert args.workers == 8


def test_default_tokens_still_use_sys_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--workers=-1"])
    args, parser = parse_training_args([])
    merge_config_with_args({"workers": 8}, args, parser)
    assert args.workers == -1


def test_precision_alias_cli_overrides_yaml():
    cli = ["--mixed_precision", "bf16"]
    args, parser = parse_training_args(cli)
    merge_config_with_args({"precision": "fp16"}, args, parser, argv=cli)
    assert args.precision == "bf16"


def test_nested_prefix_remapping_and_sequence_coercion(caplog):
    args, parser = parse_training_args([])
    with caplog.at_level(logging.WARNING):
        merge_config_with_args(
            {
                "optimizer_lr": 0.002,
                "betas": [0.8, 0.99],
                "milestones": [3, 9],
                "constraint_weight": [0.2, 0.5],
            },
            args,
            parser,
            argv=[],
        )
    assert args.lr == 0.002
    assert args.betas == "0.8,0.99"
    assert args.milestones == "3,9"
    assert args.constraint_weight == [0.2, 0.5]
    assert "mapped to 'lr'" in caplog.text


def test_no_parser_preserves_sequence_and_value_overwrite_behavior():
    args = argparse.Namespace(betas="0.9,0.999", workers=-1)
    merge_config_with_args(
        {"betas": (0.8, 0.9), "workers": 5}, args, argv=["--workers=-1"]
    )
    assert args.betas == "0.8,0.9"
    assert args.workers == 5


@pytest.mark.parametrize(
    "modules", [[], ["custom.models"], ["custom.models", "custom.constraints"]]
)
def test_yaml_training_sequence_options_remain_lists(modules):
    """YAML exports preserve nargs lists while CSV-valued options remain CSV."""
    args, parser = parse_training_args([])
    merge_config_with_args(
        {
            "model": "resnet18",
            "data_path": "/data/authored dataset.npz",
            "import_modules": modules,
            "constraint": ["custom.balance", "custom.boundary"],
            "constraint_weight": [0.125, 0.875],
            "betas": [0.8, 0.98],
            "milestones": [4, 8, 16],
            "loss_weights": [1.25, 2.5],
        },
        args,
        parser,
        argv=[],
    )
    assert args.model == "resnet18"
    assert args.data_path == "/data/authored dataset.npz"
    assert args.import_modules == modules
    assert args.constraint == ["custom.balance", "custom.boundary"]
    assert args.constraint_weight == [0.125, 0.875]
    assert args.betas == "0.8,0.98"
    assert args.milestones == "4,8,16"
    assert args.loss_weights == "1.25,2.5"


def test_explicit_import_sequence_still_overrides_yaml_sequence():
    cli = ["--import", "explicit.models", "explicit.constraints"]
    args, parser = parse_training_args(cli)
    merge_config_with_args({"import_modules": ["yaml.models"]}, args, parser, argv=cli)
    assert args.import_modules == ["explicit.models", "explicit.constraints"]


@pytest.mark.parametrize("ignore_unknown", [True, False])
def test_unknown_keys_preserve_logging_policy(caplog, ignore_unknown):
    args, parser = parse_training_args([])
    with caplog.at_level(logging.DEBUG):
        merge_config_with_args({"unknown": 1}, args, parser, ignore_unknown, argv=[])
    assert not hasattr(args, "unknown")
    assert caplog.records[-1].levelno == (
        logging.DEBUG if ignore_unknown else logging.WARNING
    )


def test_merge_does_not_add_validation_or_coerce_numeric_values():
    args, parser = parse_training_args([])
    merge_config_with_args(
        {"epochs": "12", "precision": "invalid"}, args, parser, argv=[]
    )
    assert args.epochs == "12"
    assert args.precision == "invalid"


def test_import_and_merge_are_stdlib_only(tmp_path):
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "numpy", "PySide6", "wavedl_daemon", "wavedl_gui", "yaml"
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from wavedl.core.configuration import merge_config_with_args
from wavedl.core.training_cli import parse_training_args
args, parser = parse_training_args([])
assert merge_config_with_args({"epochs": 12}, args, parser, argv=[]).epochs == 12
assert "yaml" not in sys.modules
"""
    source_root = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(source_root)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
