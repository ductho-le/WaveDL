# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""The upstream ``wavedl.train`` and ``wavedl.test`` options a plan may set.

The daemon never imports the trainer, so this table restates the part of its
command line the GUI exposes; ``test_training_options_drift`` checks it against
the installed ``--help``. Options the daemon owns (``--output_dir``, the metrics
files, ``--config``, ``--allow_pickle``, ``--selection_metric``) are absent, so a
request cannot set them. ``web/shared/training-options.json`` mirrors ``TRAIN``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wavedl_daemon.runs.model import Operation


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class Option:
    """One option. ``bool`` is a store-true flag; ``multiple`` is ``nargs="+"``.

    A request may send null only where the default is None, and an empty list
    only where the default is empty.
    """

    type: type[bool | int | float | str]
    default: object = None
    choices: tuple[str, ...] = ()
    multiple: bool = False
    valid: Callable[[int | float], bool] | None = None
    path: bool = False  # relative to the request's submission directory
    required: bool = False


def _at_least(low: int) -> Callable[[int | float], bool]:
    return lambda value: value >= low


FLAG = Option(bool, False)

TRAIN: dict[str, Option] = {
    "model": Option(str, "cnn"),
    "import_modules": Option(str, [], multiple=True),
    "pretrained": FLAG,
    "batch_size": Option(int, 128, valid=_at_least(1)),
    "lr": Option(float, 1e-3, valid=lambda value: value > 0),
    "epochs": Option(int, 1000, valid=_at_least(1)),
    "patience": Option(int, 20, valid=_at_least(0)),
    "weight_decay": Option(float, 1e-4, valid=_at_least(0)),
    "grad_clip": Option(float, 1.0),
    "grad_accum_steps": Option(int, 1, valid=_at_least(1)),
    "loss": Option(
        str,
        "mse",
        ("mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"),
    ),
    "huber_delta": Option(float, 1.0),
    "loss_weights": Option(str),
    "optimizer": Option(
        str, "adamw", ("adamw", "adam", "sgd", "nadam", "radam", "rmsprop")
    ),
    "momentum": Option(float, 0.9),
    "nesterov": FLAG,
    "betas": Option(str, "0.9,0.999"),
    "scheduler": Option(
        str,
        "plateau",
        (
            "plateau",
            "cosine",
            "cosine_restarts",
            "onecycle",
            "step",
            "multistep",
            "exponential",
            "linear_warmup",
        ),
    ),
    "scheduler_patience": Option(int, 10, valid=_at_least(0)),
    "min_lr": Option(float, 1e-6),
    "scheduler_factor": Option(float, 0.5),
    "warmup_epochs": Option(int, 5, valid=_at_least(0)),
    "step_size": Option(int, 30, valid=_at_least(1)),
    "milestones": Option(str),
    "data_path": Option(str, "train_data.npz", path=True),
    "workers": Option(int, -1, valid=_at_least(-1)),
    "seed": Option(int, 2025),
    "deterministic": FLAG,
    "cache_validate": Option(str, "sha256", ("sha256", "fast", "size")),
    "single_channel": FLAG,
    "cv": Option(int, 0, valid=_at_least(0)),
    "cv_stratify": FLAG,
    "cv_bins": Option(int, 10, valid=_at_least(1)),
    "resume": Option(str, path=True),
    "save_every": Option(int, 50, valid=_at_least(0)),
    "fresh": FLAG,
    "compile": FLAG,
    "precision": Option(str, "bf16", ("bf16", "fp16", "no")),
    "constraint": Option(str, [], multiple=True),
    "constraint_file": Option(str, path=True),
    "constraint_weight": Option(float, [0.1], multiple=True),
    "constraint_reduction": Option(str, "mse", ("mse", "mae")),
    "wandb": FLAG,
    "wandb_watch": FLAG,
    "project_name": Option(str, "DL-Training"),
    "run_name": Option(str),
}

# Upstream wavedl-test picks its device itself and has no flag to force one.
TEST: dict[str, Option] = {
    "checkpoint": Option(str, path=True, required=True),
    "data_path": Option(str, path=True, required=True),
    "model": Option(str),
    "import_modules": Option(str, [], multiple=True),
    "format": Option(str, "auto", ("auto", "npz", "mat", "hdf5")),
    "input_key": Option(str),
    "output_key": Option(str),
    "param_names": Option(str, multiple=True),
    "input_channels": Option(int, valid=_at_least(1)),
    "batch_size": Option(int, 128, valid=_at_least(1)),
    "workers": Option(int, 0, valid=_at_least(0)),
    "save_predictions": FLAG,
    "plot": FLAG,
    "plot_format": Option(
        str, ["png"], ("png", "pdf", "svg", "eps", "tiff"), multiple=True
    ),
    "verbose": FLAG,
    "export": Option(str, None, ("onnx",)),
    "export_opset": Option(int, 17, valid=_at_least(1)),
    "no_denorm": FLAG,
    "out_size": Option(int, valid=_at_least(1)),
}

OPTIONS: dict[Operation, dict[str, Option]] = {
    Operation.TRAIN: TRAIN,
    Operation.TEST: TEST,
    Operation.PLOT: TEST,
}

# Upstream's CV path publishes completed folds, never periodic checkpoints,
# and has no resume, compile or WandB support.
CV_UNSUPPORTED = (
    "resume",
    "save_every",
    "fresh",
    "compile",
    "wandb",
    "wandb_watch",
    "project_name",
    "run_name",
)


def flag(name: str) -> str:
    return "--import" if name == "import_modules" else f"--{name}"


def _item(option: Option, value: object) -> object:
    if option.type is float and type(value) is int:
        value = float(value)
    if type(value) is not option.type:
        raise ValueError("Option has the wrong type")
    # A leading dash would make upstream's argparse read the value as a flag.
    if option.type is str and (not value or value.startswith("-")):
        raise ValueError("Option requires a nonempty value that is not a flag")
    if option.choices and value not in option.choices:
        raise ValueError("Invalid option choice")
    if option.valid is not None and not option.valid(value):
        raise ValueError("Invalid config range")
    return value


def _value(option: Option, value: object) -> object:
    if option.type is bool:
        if type(value) is not bool:
            raise ValueError("Flag must be boolean")
        return value
    if value is None:
        if option.default is not None:
            raise ValueError("Option cannot be null")
        return None
    if not option.multiple:
        return _item(option, value)
    if type(value) is not list or (not value and option.default != []):
        raise ValueError("Option requires a nonempty list")
    return [_item(option, item) for item in value]


def resolve(operation: Operation, config: Mapping[str, object]) -> dict[str, object]:
    """Every option of ``operation``: its default, or the request's valid value."""
    table = OPTIONS[operation]
    if set(config) - set(table):
        raise ValueError("Unknown or daemon-owned config option")
    result = {name: deepcopy(option.default) for name, option in table.items()}
    for name, value in config.items():
        result[name] = _value(table[name], value)
    if any(result[name] is None for name, option in table.items() if option.required):
        raise ValueError("Checkpoint and dataset are required")
    if operation is Operation.TRAIN:
        check_cv(result)
    return result


def check_cv(config: Mapping[str, object]) -> None:
    """Refuse options upstream cross-validation cannot honour; others default."""
    values = {name: config.get(name, item.default) for name, item in TRAIN.items()}
    if values["cv"] <= 0:
        return
    if values["cv"] < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if values["nesterov"]:
        if values["optimizer"] != "sgd":
            raise ValueError(
                "Cross-validation Nesterov momentum requires --optimizer sgd"
            )
        if not values["momentum"] > 0:
            raise ValueError(
                "Cross-validation Nesterov momentum requires positive --momentum"
            )
    if values["scheduler"] == "plateau" and not 0 <= values["scheduler_factor"] < 1:
        raise ValueError(
            "Cross-validation plateau scheduler requires --scheduler_factor in [0, 1)"
        )
    unsupported = [
        flag(name) for name in CV_UNSUPPORTED if values[name] != TRAIN[name].default
    ]
    if unsupported:
        raise ValueError(
            "Cross-validation does not support nondefault options: "
            + ", ".join(unsupported)
        )


def cli_args(operation: Operation, config: Mapping[str, object]) -> list[str]:
    """``config`` as upstream arguments: a set flag, or the flag and its values."""
    table = OPTIONS[operation]
    argv = []
    for name, value in sorted(config.items()):
        option = table.get(name)
        if option is None:
            raise ValueError(f"Unsupported resolved option: {name}")
        if option.type is bool:
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            argv += [flag(name)] if value else []
            continue
        if value is None or value == []:
            continue
        if isinstance(value, list) and not option.multiple:
            raise ValueError(f"Option does not accept a list: {name}")
        items = value if isinstance(value, list) else [value]
        if not all(type(item) in {str, int, float} for item in items) or any(
            isinstance(item, str) and ("\0" in item or item.startswith("-"))
            for item in items
        ):
            raise ValueError(f"Unsafe option value: {name}")
        argv += [flag(name), *map(str, items)]
    return argv
