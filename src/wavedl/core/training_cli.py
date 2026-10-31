"""Dependency-light argument schema shared by training and its launcher.

Registry names enrich help only; importing this module never imports the ML stack.
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


def build_training_parser(
    *,
    models=(),
    losses=(),
    optimizers=(),
    schedulers=(),
) -> argparse.ArgumentParser:
    """Build the training CLI parser without loading optional runtime packages."""
    parser = argparse.ArgumentParser(
        description="Universal DDP Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model Selection
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        help=f"Model architecture to train. Available: {models}",
    )
    parser.add_argument(
        "--list_models", action="store_true", help="List all available models and exit"
    )
    parser.add_argument(
        "--import",
        dest="import_modules",
        type=str,
        nargs="+",
        default=[],
        help="Python modules to import before training (for custom models)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights (only for *_pretrained model variants). "
        "Default: train from scratch.",
    )

    # Configuration File
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI args override config values.",
    )

    # Hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size per GPU"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size x grad_accum_steps x num_gpus",
    )

    # Loss Function
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"],
        help=f"Loss function for training. Available: {losses}",
    )
    parser.add_argument(
        "--huber_delta", type=float, default=1.0, help="Delta for Huber loss"
    )
    parser.add_argument(
        "--loss_weights",
        type=str,
        default=None,
        help="Comma-separated weights for weighted_mse (e.g., '1.0,2.0,1.0')",
    )

    parser.add_argument(
        "--selection_metric",
        choices=["loss", "mse"],
        default="loss",
        help="Validation metric for checkpoint selection and early stopping. HPO uses mse to compare different training losses.",
    )
    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="Allow trusted legacy NPZ object arrays. Pickled data can execute arbitrary code.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"],
        help=f"Optimizer for training. Available: {optimizers}",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD/RMSprop"
    )
    parser.add_argument(
        "--nesterov", action="store_true", help="Use Nesterov momentum (SGD)"
    )
    parser.add_argument(
        "--betas",
        type=str,
        default="0.9,0.999",
        help="Betas for Adam variants (comma-separated)",
    )

    # Learning Rate Scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=[
            "plateau",
            "cosine",
            "cosine_restarts",
            "onecycle",
            "step",
            "multistep",
            "exponential",
            "linear_warmup",
        ],
        help=f"LR scheduler. Available: {schedulers}",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=10,
        help="Patience for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate"
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.5, help="LR reduction factor"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Warmup epochs for linear_warmup"
    )
    parser.add_argument(
        "--step_size", type=int, default=30, help="Step size for StepLR"
    )
    parser.add_argument(
        "--milestones",
        type=str,
        default=None,
        help="Comma-separated epochs for MultiStepLR (e.g., '30,60,90')",
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, default="train_data.npz", help="Path to NPZ dataset"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="DataLoader workers per GPU (-1=auto-detect based on CPU cores)",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility (slower, disables TF32/cuDNN benchmark)",
    )
    parser.add_argument(
        "--cache_validate",
        type=str,
        default="sha256",
        choices=["sha256", "fast", "size"],
        help="Cache validation mode: sha256 (full hash), fast (partial), size (quick)",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="Confirm data is single-channel (suppress ambiguous shape warnings for shallow 3D volumes)",
    )

    # Cross-Validation
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Enable K-fold cross-validation with K folds (0=disabled)",
    )
    parser.add_argument(
        "--cv_stratify",
        action="store_true",
        help="Use stratified splitting for cross-validation",
    )
    parser.add_argument(
        "--cv_bins",
        type=int,
        default=10,
        help="Number of bins for stratified CV (only with --cv_stratify)",
    )

    # Checkpointing & Resume
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint directory to resume from"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save checkpoint every N epochs (0=disable)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh training, ignore existing checkpoints",
    )

    # Performance
    parser.add_argument(
        "--compile", action="store_true", help="Enable torch.compile (PyTorch 2.x)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "no"],
        help="Mixed precision mode",
    )
    # Alias for consistency with wavedl-train (--mixed_precision is passed to accelerate)
    parser.add_argument(
        "--mixed_precision",
        dest="precision",
        type=str,
        choices=["bf16", "fp16", "no"],
        help=argparse.SUPPRESS,  # Hidden: use --precision instead
    )

    # Physical Constraints
    parser.add_argument(
        "--constraint",
        type=str,
        nargs="+",
        default=[],
        help="Soft constraint expressions: 'y0 - y1*y2' (penalize violations)",
    )

    parser.add_argument(
        "--constraint_file",
        type=str,
        default=None,
        help="Python file with constraint(pred, inputs) function",
    )
    parser.add_argument(
        "--constraint_weight",
        type=float,
        nargs="+",
        default=[0.1],
        help="Weight(s) for soft constraints (one per constraint, or single shared weight)",
    )
    parser.add_argument(
        "--constraint_reduction",
        type=str,
        default="mse",
        choices=["mse", "mae"],
        help="Reduction mode for constraint penalties",
    )

    # Logging
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_watch",
        action="store_true",
        help="Enable WandB gradient watching (adds overhead, useful for debugging)",
    )
    parser.add_argument(
        "--project_name", type=str, default="DL-Training", help="WandB project name"
    )
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    return parser


def parse_training_args(
    argv: list[str] | None = None,
    *,
    models=(),
    losses=(),
    optimizers=(),
    schedulers=(),
) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Parse training arguments, returning their parser for config merging."""
    parser = build_training_parser(
        models=models,
        losses=losses,
        optimizers=optimizers,
        schedulers=schedulers,
    )
    return parser.parse_args(argv), parser


def training_defaults() -> dict[str, object]:
    """Return an isolated snapshot of effective no-argument training defaults.

    Parse rather than collecting action defaults: the hidden precision alias has
    a None action default, but the effective shared destination defaults to bf16.
    """
    return deepcopy(vars(build_training_parser().parse_args([])))


CV_UNSUPPORTED_OPTIONS = (
    "resume",
    "save_every",
    "fresh",
    "compile",
    "wandb",
    "wandb_watch",
    "project_name",
    "run_name",
    "selection_metric",
)


def validate_cv_options(options: argparse.Namespace | Mapping[str, object]) -> None:
    """Refuse authored options that the CV algorithm cannot honour.

    Accept either effective CLI arguments (after YAML merging) or a flat resolved
    configuration. Missing values have CLI defaults. Non-CV training is unchanged.
    CV supports its model/import/data, optimisation, loss, scheduler, precision,
    pretrained, deterministic, physical-constraint and stratification options. It publishes each
    completed fold, not periodic checkpoints, so only the default ``save_every``
    placeholder is accepted. Resume/fresh, compilation and
    WandB options likewise require their defaults until implemented for CV, as
    does ``selection_metric``: folds always select using the configured loss.

    ``cache_validate`` is deliberately inapplicable, not unsupported: CV reads
    the source dataset directly, without the ordinary training cache. All cache
    validation choices are accepted and retained in the resolved configuration.
    Nesterov requires SGD with finite positive momentum; plateau reduction
    factors must be finite and in [0, 1). Other schedulers may increase the LR.
    Errors name only option keys, never caller-provided paths or logging values.
    """
    values = vars(options) if isinstance(options, argparse.Namespace) else options
    if values.get("cv", 0) <= 0:
        return
    if values["cv"] < 2:
        raise ValueError("Cross-validation requires at least two folds")
    defaults = training_defaults()
    if values.get("nesterov", defaults["nesterov"]):
        if str(values.get("optimizer", defaults["optimizer"])).lower() != "sgd":
            raise ValueError(
                "Cross-validation Nesterov momentum requires --optimizer sgd"
            )
        momentum = values.get("momentum", defaults["momentum"])
        if (
            not isinstance(momentum, (int, float))
            or not math.isfinite(momentum)
            or momentum <= 0
        ):
            raise ValueError(
                "Cross-validation Nesterov momentum requires positive --momentum"
            )
    if str(values.get("scheduler", defaults["scheduler"])).lower() == "plateau":
        factor = values.get("scheduler_factor", defaults["scheduler_factor"])
        if (
            not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or not 0 <= factor < 1
        ):
            raise ValueError(
                "Cross-validation plateau scheduler requires --scheduler_factor in [0, 1)"
            )
    unsupported = [
        name
        for name in CV_UNSUPPORTED_OPTIONS
        if values.get(name, defaults[name]) != defaults[name]
    ]
    if unsupported:
        flags = ", ".join(f"--{name}" for name in unsupported)
        raise ValueError(
            f"Cross-validation does not support nondefault options: {flags}"
        )
