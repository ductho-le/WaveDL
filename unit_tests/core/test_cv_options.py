"""CV options are implemented, inapplicable, or visibly refused before work."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl.core.training_cli import (
    CV_UNSUPPORTED_OPTIONS,
    parse_training_args,
    training_defaults,
    validate_cv_options,
)


UNSUPPORTED_ARGV = [
    ["--resume", "/private/secret-checkpoint"],
    ["--save_every", "1"],
    ["--save_every", "0"],
    ["--fresh"],
    ["--compile"],
    ["--wandb"],
    ["--wandb_watch"],
    ["--project_name", "private-project"],
    ["--run_name", "private-run"],
]


@pytest.mark.parametrize("argv", UNSUPPORTED_ARGV)
def test_parser_authored_unsupported_options_are_refused_without_values(argv):
    args, _ = parse_training_args(["--cv", "2", *argv])
    with pytest.raises(ValueError, match="Cross-validation") as error:
        validate_cv_options(args)
    assert argv[0] in str(error.value)
    assert "private" not in str(error.value)


@pytest.mark.parametrize("argv", UNSUPPORTED_ARGV)
def test_non_cv_parser_options_remain_accepted(argv):
    args, _ = parse_training_args(argv)
    validate_cv_options(args)


def test_effective_defaults_and_partial_configs_are_accepted_unchanged():
    defaults = training_defaults() | {"cv": 2}
    before = training_defaults() | {"cv": 2}
    validate_cv_options(defaults)
    validate_cv_options({"cv": 2})
    validate_cv_options({})
    assert defaults == before
    assert set(CV_UNSUPPORTED_OPTIONS) <= defaults.keys()


def test_every_parser_option_has_an_explicit_cv_compatibility_classification():
    """New CLI fields must receive a CV support decision, never silently pass."""
    implemented = {
        "model",
        "import_modules",
        "pretrained",
        "batch_size",
        "lr",
        "epochs",
        "patience",
        "weight_decay",
        "grad_clip",
        "grad_accum_steps",
        "loss",
        "huber_delta",
        "loss_weights",
        "optimizer",
        "momentum",
        "nesterov",
        "betas",
        "scheduler",
        "scheduler_patience",
        "min_lr",
        "scheduler_factor",
        "warmup_epochs",
        "step_size",
        "milestones",
        "data_path",
        "allow_pickle",
        "workers",
        "seed",
        "deterministic",
        "single_channel",
        "cv",
        "cv_stratify",
        "cv_bins",
        "precision",
        "constraint",
        "constraint_file",
        "constraint_weight",
        "constraint_reduction",
    }
    entrypoint = {"config", "list_models", "output_dir"}
    inapplicable = {"cache_validate"}
    assert set(training_defaults()) == (
        implemented | entrypoint | inapplicable | set(CV_UNSUPPORTED_OPTIONS)
    )


def test_one_fold_cv_is_refused():
    args, _ = parse_training_args(["--cv", "1"])
    with pytest.raises(ValueError, match="at least two folds"):
        validate_cv_options(args)


@pytest.mark.parametrize("optimizer", ["adamw", "adam", "nadam", "radam", "rmsprop"])
def test_cv_nesterov_requires_sgd(optimizer):
    args, _ = parse_training_args(["--cv", "2", "--optimizer", optimizer, "--nesterov"])
    with pytest.raises(ValueError, match="--optimizer"):
        validate_cv_options(args)


@pytest.mark.parametrize(
    "momentum", [0, -0.1, float("nan"), float("inf"), -float("inf")]
)
def test_cv_nesterov_requires_finite_positive_momentum(momentum):
    config = {"cv": 2, "optimizer": "sgd", "nesterov": True, "momentum": momentum}
    with pytest.raises(ValueError, match="--momentum"):
        validate_cv_options(config)


@pytest.mark.parametrize("factor", [-0.1, 1, 2, float("nan"), float("inf")])
def test_cv_plateau_requires_finite_reduction_factor(factor):
    with pytest.raises(ValueError, match="--scheduler_factor"):
        validate_cv_options(
            {"cv": 2, "scheduler": "plateau", "scheduler_factor": factor}
        )


@pytest.mark.parametrize("factor", [0, 0.1, 0.999])
def test_cv_valid_dependent_options_are_preserved(factor):
    args, _ = parse_training_args(
        [
            "--cv",
            "2",
            "--optimizer",
            "sgd",
            "--nesterov",
            "--momentum",
            "0.8",
            "--scheduler_factor",
            str(factor),
        ]
    )
    before = vars(args).copy()
    validate_cv_options(args)
    assert vars(args) == before


@pytest.mark.parametrize("scheduler", ["step", "multistep", "exponential"])
def test_cv_decay_schedulers_allow_lr_increases(scheduler):
    validate_cv_options({"cv": 2, "scheduler": scheduler, "scheduler_factor": 2})


def test_non_cv_dependent_option_validation_remains_unchanged():
    validate_cv_options(
        {"cv": 0, "nesterov": True, "optimizer": "adamw", "scheduler_factor": -1}
    )


@pytest.mark.parametrize("mode", ["sha256", "fast", "size"])
def test_cache_validation_is_inapplicable_to_direct_source_cv(mode):
    args, _ = parse_training_args(["--cv", "2", "--cache_validate", mode])
    validate_cv_options(args)
    assert args.cache_validate == mode


def test_all_implemented_authored_options_are_accepted_and_preserved():
    args, _ = parse_training_args(
        [
            "--cv",
            "3",
            "--model",
            "custom",
            "--import",
            "custom.model",
            "--data_path",
            "custom.npz",
            "--batch_size",
            "8",
            "--lr",
            "0.1",
            "--epochs",
            "3",
            "--patience",
            "2",
            "--weight_decay",
            "0.01",
            "--grad_clip",
            "2",
            "--grad_accum_steps",
            "2",
            "--loss",
            "weighted_mse",
            "--loss_weights",
            "1,2",
            "--huber_delta",
            "2",
            "--optimizer",
            "sgd",
            "--momentum",
            "0.8",
            "--nesterov",
            "--betas",
            "0.8,0.99",
            "--scheduler",
            "multistep",
            "--scheduler_patience",
            "2",
            "--scheduler_factor",
            "0.1",
            "--min_lr",
            "0.0001",
            "--step_size",
            "2",
            "--milestones",
            "1,2",
            "--warmup_epochs",
            "2",
            "--workers",
            "0",
            "--seed",
            "42",
            "--precision",
            "no",
            "--pretrained",
            "--constraint",
            "y0-y1",
            "--constraint_file",
            "physics.py",
            "--constraint_weight",
            "0.4",
            "--constraint_reduction",
            "mae",
            "--cv_stratify",
            "--cv_bins",
            "3",
            "--single_channel",
            "--output_dir",
            "custom-output",
        ]
    )
    before = vars(args).copy()
    validate_cv_options(args)
    assert vars(args) == before


def test_multiple_refusals_report_keys_not_authored_values():
    with pytest.raises(ValueError) as error:
        validate_cv_options({"cv": 2, "resume": "secret", "run_name": "secret"})
    assert str(error.value).endswith("--resume, --run_name")
    assert "secret" not in str(error.value)


def test_validator_remains_dependency_light():
    source = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; from wavedl.core.training_cli import validate_cv_options; "
        "validate_cv_options({'cv':2}); "
        "assert not any(name in sys.modules for name in "
        "['torch', 'numpy', 'PySide6', 'accelerate'])"
    )
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        env={**os.environ, "PYTHONPATH": str(source)},
        check=True,
        capture_output=True,
        timeout=10,
    )
