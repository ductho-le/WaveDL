"""
WaveDL - Hyperparameter Optimization with Optuna
=================================================
Automated hyperparameter search for finding optimal training configurations.

Usage:
    # Basic HPO (50 trials)
    wavedl-hpo --data_path train.npz --n_trials 50

    # Quick search (fewer parameters)
    wavedl-hpo --data_path train.npz --n_trials 30 --quick

    # Medium search (balanced)
    wavedl-hpo --data_path train.npz --n_trials 50 --medium

    # Full search with specific models
    wavedl-hpo --data_path train.npz --n_trials 100 --models cnn resnet18 efficientnet_b0

    # Parallel trials on multiple GPUs
    wavedl-hpo --data_path train.npz --n_trials 100 --n_jobs 4

    # In-process mode (enables pruning, faster, single-GPU)
    wavedl-hpo --data_path train.npz --n_trials 50 --inprocess

Execution Modes:
    --inprocess: Runs trials in the same Python process. Enables pruning
                 (MedianPruner) for early stopping of unpromising trials.
                 Faster due to no subprocess overhead, but trials share
                 GPU memory (no isolation between trials).

    Default (subprocess): Launches each trial as a separate process.
                          Provides GPU memory isolation but prevents pruning
                          (subprocess can't report intermediate results).

Author: Ductho Le (ductho.le@outlook.com)
"""

import argparse
import hashlib
import json
import math
import os
import queue
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Error: Optuna not installed. Run: pip install wavedl")
    sys.exit(1)


def _detect_gpu_count() -> int:
    """Return the number of GPUs via nvidia-smi (best effort, defaults to 1)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return max(
            1,
            len(
                [
                    value
                    for value in visible.split(",")
                    if value.strip() and value.strip() != "-1"
                ]
            ),
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip()
            return max(1, len(lines.split("\n"))) if lines else 1
    except Exception:
        pass
    return 1


# =============================================================================
# DEFAULT SEARCH SPACES
# =============================================================================

DEFAULT_MODELS = ["cnn", "resnet18", "resnet34"]
QUICK_MODELS = ["cnn"]
MEDIUM_MODELS = ["cnn", "resnet18"]
DENSE_MODELS = ["unet_field"]
DEFAULT_BATCH_SIZES = [16, 32, 64, 128]
DENSE_BATCH_SIZES = [1, 2, 4, 8]

# All 6 optimizers
DEFAULT_OPTIMIZERS = ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]
QUICK_OPTIMIZERS = ["adamw"]
MEDIUM_OPTIMIZERS = ["adamw", "adam", "sgd"]

# All 8 schedulers
DEFAULT_SCHEDULERS = [
    "plateau",
    "cosine",
    "cosine_restarts",
    "onecycle",
    "step",
    "multistep",
    "exponential",
    "linear_warmup",
]
QUICK_SCHEDULERS = ["plateau"]
MEDIUM_SCHEDULERS = ["plateau", "cosine", "onecycle"]

# All 6 losses
DEFAULT_LOSSES = ["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"]
QUICK_LOSSES = ["mse"]
MEDIUM_LOSSES = ["mse", "mae", "huber"]


def _detect_target_shape(
    data_path: str, *, allow_pickle: bool = False
) -> tuple[int, ...]:
    """Inspect labels and return WaveDL's canonical per-sample shape.

    HDF5 and MAT files expose dataset metadata lazily, so large target maps are
    not materialized just to configure a sweep. NPZ has no general lazy shape
    API, but only its output member is loaded rather than the input data.
    """
    from wavedl.utils import (
        DataSource,
        canonical_target_shape,
        get_data_source,
        load_outputs_only,
    )

    data_format = DataSource.detect_format(data_path)
    if data_format in ("hdf5", "mat"):
        handle = get_data_source(data_format, allow_pickle=allow_pickle).load_mmap(
            data_path
        )
        try:
            raw_shape = tuple(handle.outputs.shape)
            if data_format == "mat" and len(raw_shape) == 2:
                num_samples = handle.inputs.shape[0]
                if raw_shape[0] == 1 and raw_shape[1] == num_samples:
                    raw_shape = (num_samples, 1)
                elif raw_shape[1] == 1 and raw_shape[0] != num_samples:
                    raw_shape = (1, raw_shape[0])
        finally:
            handle.close()
        return canonical_target_shape(raw_shape)

    targets = load_outputs_only(data_path, allow_pickle=allow_pickle)
    try:
        return canonical_target_shape(tuple(targets.shape))
    finally:
        del targets


def _resolve_models_for_target(
    target_shape: tuple[int, ...],
    requested_models: list[str] | None = None,
    *,
    quick: bool = False,
    medium: bool = False,
) -> list[str]:
    """Choose an HPO model space and reject incompatible model/target pairs."""
    from wavedl.models import get_model

    target_shape = tuple(target_shape)
    if len(target_shape) not in (1, 3):
        raise ValueError(
            f"Unsupported canonical target shape {target_shape}. Expected a vector "
            "(T,) or dense 2D field (C,H,W)."
        )

    is_field = len(target_shape) == 3
    if requested_models:
        models = list(requested_models)
    elif is_field:
        models = DENSE_MODELS.copy()
    elif quick:
        models = QUICK_MODELS.copy()
    elif medium:
        models = MEDIUM_MODELS.copy()
    else:
        models = DEFAULT_MODELS.copy()

    incompatible = []
    for model_name in models:
        model_class = get_model(model_name)
        if (
            is_field and not getattr(model_class, "supports_spatial_output", False)
        ) or (not is_field and getattr(model_class, "requires_spatial_output", False)):
            incompatible.append(model_name)

    if incompatible:
        target_kind = "dense 2D field" if is_field else "vector"
        recommendation = (
            "Use --models unet_field or register a model with "
            "supports_spatial_output=True."
            if is_field
            else "Choose a vector-output model such as cnn or unet_regression."
        )
        raise ValueError(
            f"Model(s) {', '.join(incompatible)} are incompatible with {target_kind} "
            f"targets shaped {target_shape}. {recommendation}"
        )

    return models


def _resolve_batch_sizes(
    target_shape: tuple[int, ...] | None,
    requested_batch_sizes: list[int] | None = None,
) -> list[int]:
    """Use memory-conscious defaults for dense target maps."""
    if requested_batch_sizes:
        return list(requested_batch_sizes)
    if isinstance(target_shape, (list, tuple)) and len(target_shape) == 3:
        return DENSE_BATCH_SIZES.copy()
    return DEFAULT_BATCH_SIZES.copy()


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================


def create_objective(args):
    """Create Optuna objective function with configurable search space.

    Supports two execution modes:
    - Subprocess (default): Launches wavedl.train via subprocess. Provides GPU
      memory isolation but prevents pruning (MedianPruner has no effect).
    - In-process (--inprocess): Calls train_single_trial() directly. Enables
      pruning and reduces overhead, but trials share GPU memory.
    """

    target_shape = getattr(args, "target_shape", None)
    if not isinstance(target_shape, (list, tuple)):
        # Backward compatibility for callers that construct a Namespace and
        # invoke create_objective() directly instead of going through main().
        target_shape = (1,)
    models = _resolve_models_for_target(
        tuple(target_shape),
        args.models,
        quick=args.quick,
        medium=args.medium,
    )

    # Lease actual visible device IDs. Round-robin trial numbers can assign a
    # new trial to a GPU whose earlier (slower) trial is still running.
    gpu_pool = queue.Queue()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    n_gpus = getattr(args, "n_gpus", 1)
    n_gpus = n_gpus if isinstance(n_gpus, int) else 1
    gpu_ids = (
        [
            value.strip()
            for value in visible.split(",")
            if value.strip() and value.strip() != "-1"
        ]
        if visible is not None
        else [str(index) for index in range(max(1, n_gpus))]
    )
    for gpu_id in gpu_ids:
        gpu_pool.put(gpu_id)

    def objective(trial):
        # Select search space based on mode (quick < medium < full)
        # CLI arguments always take precedence over defaults
        if args.quick:
            optimizers = args.optimizers or QUICK_OPTIMIZERS
            schedulers = args.schedulers or QUICK_SCHEDULERS
            losses = args.losses or QUICK_LOSSES
        elif args.medium:
            optimizers = args.optimizers or MEDIUM_OPTIMIZERS
            schedulers = args.schedulers or MEDIUM_SCHEDULERS
            losses = args.losses or MEDIUM_LOSSES
        else:
            optimizers = args.optimizers or DEFAULT_OPTIMIZERS
            schedulers = args.schedulers or DEFAULT_SCHEDULERS
            losses = args.losses or DEFAULT_LOSSES

        # Suggest hyperparameters
        model = trial.suggest_categorical("model", models)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_sizes = _resolve_batch_sizes(
            getattr(args, "target_shape", None), args.batch_sizes
        )
        batch_size = trial.suggest_categorical("batch_size", batch_sizes)
        optimizer = trial.suggest_categorical("optimizer", optimizers)
        scheduler = trial.suggest_categorical("scheduler", schedulers)
        loss = trial.suggest_categorical("loss", losses)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        patience = trial.suggest_int("patience", 10, 30, step=5)

        # Conditional hyperparameters
        if loss == "huber":
            huber_delta = trial.suggest_float("huber_delta", 0.1, 2.0)
        else:
            huber_delta = 1.0  # default

        if optimizer == "sgd":
            momentum = trial.suggest_float("momentum", 0.8, 0.99)
        else:
            momentum = 0.9  # default

        # ==================================================================
        # IN-PROCESS MODE: Direct function call with pruning support
        # ==================================================================
        if args.inprocess:
            from wavedl.train import train_single_trial

            try:
                result = train_single_trial(
                    data_path=args.data_path,
                    model_name=model,
                    lr=lr,
                    batch_size=batch_size,
                    epochs=args.max_epochs,
                    patience=patience,
                    optimizer_name=optimizer,
                    scheduler_name=scheduler,
                    loss_name=loss,
                    weight_decay=weight_decay,
                    seed=args.seed,
                    huber_delta=huber_delta,
                    momentum=momentum,
                    trial=trial,  # Enable pruning via trial.report/should_prune
                    verbose=False,
                    selection_metric="mse",
                    timeout_seconds=args.timeout,
                    allow_pickle=getattr(args, "allow_pickle", False) is True,
                )

                if result["pruned"]:
                    print(
                        f"Trial {trial.number}: Pruned at epoch {result['epochs_trained']}"
                    )
                    raise optuna.TrialPruned()

                val_loss = result["best_val_loss"]
                if not math.isfinite(val_loss):
                    raise FloatingPointError(
                        "Trial returned a non-finite validation MSE."
                    )
                print(
                    f"Trial {trial.number}: val_loss={val_loss:.6f} ({result['epochs_trained']} epochs)"
                )
                return val_loss

            except TimeoutError as error:
                raise optuna.TrialPruned(str(error)) from error
            except optuna.TrialPruned:
                raise  # Re-raise for Optuna to handle
            except Exception as e:
                print(f"Trial {trial.number}: Error - {e}")
                raise

        # ==================================================================
        # SUBPROCESS MODE (default): GPU memory isolation, no pruning
        # ==================================================================
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "wavedl.train",
            "--data_path",
            str(args.data_path),
            "--model",
            model,
            "--lr",
            str(lr),
            "--batch_size",
            str(batch_size),
            "--optimizer",
            optimizer,
            "--scheduler",
            scheduler,
            "--loss",
            loss,
            "--weight_decay",
            str(weight_decay),
            "--patience",
            str(patience),
            "--epochs",
            str(args.max_epochs),
            "--seed",
            str(args.seed),
            "--selection_metric",
            "mse",
            "--workers",
            "0",
        ]
        if getattr(args, "allow_pickle", False) is True:
            cmd.append("--allow_pickle")

        modules = getattr(args, "import_modules", [])
        if isinstance(modules, (list, tuple)) and modules:
            cmd.extend(["--import", *modules])

        # Add conditional args
        if loss == "huber":
            cmd.extend(["--huber_delta", str(huber_delta)])
        if optimizer == "sgd":
            cmd.extend(["--momentum", str(momentum)])

        # Use temporary directory for trial output
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd.extend(["--output_dir", tmpdir])
            history_file = Path(tmpdir) / "training_history.csv"

            env = None
            gpu_id = None
            if args.n_jobs > 1 and gpu_ids:
                gpu_id = gpu_pool.get()
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

            # Run training
            # Note: We inherit the user's cwd instead of setting cwd=Path(__file__).parent
            # because site-packages may be read-only and train.py creates cache directories
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    env=env,
                )

                # Reject crashed trials before parsing any output
                if result.returncode != 0:
                    print(
                        f"Trial {trial.number}: Training subprocess failed "
                        f"(exit code {result.returncode})"
                    )
                    stderr_lines = result.stderr.strip().split("\n")[-3:]
                    for line in stderr_lines:
                        print(f"  stderr: {line}")
                    raise RuntimeError(
                        f"Trial subprocess failed with exit code {result.returncode}"
                    )

                # Read best val_loss from training_history.csv (reliable machine-readable)
                val_loss = None
                if history_file.exists():
                    try:
                        import csv

                        with open(history_file) as f:
                            reader = csv.DictReader(f)
                            val_losses = []
                            for row in reader:
                                if "val_loss" in row:
                                    try:
                                        val_losses.append(float(row["val_loss"]))
                                    except (ValueError, TypeError):
                                        pass
                            if val_losses:
                                val_loss = min(val_losses)  # Best (minimum) val_loss
                    except Exception as e:
                        print(f"Trial {trial.number}: Error reading history: {e}")

                if val_loss is None:
                    # Fallback: parse stdout for training log format
                    # Pattern: "epoch | train_loss | val_loss | ..."
                    # Use regex to avoid false positives from unrelated lines
                    import re

                    # Match lines like: "  42  | 0.0123   | 0.0156   | ..."
                    log_pattern = re.compile(
                        r"^\s*\d+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
                    )
                    val_losses_stdout = []
                    for line in result.stdout.split("\n"):
                        match = log_pattern.match(line)
                        if match:
                            try:
                                val_losses_stdout.append(float(match.group(1)))
                            except ValueError:
                                continue
                    if val_losses_stdout:
                        val_loss = min(val_losses_stdout)

                if val_loss is None:
                    raise RuntimeError(
                        f"Trial {trial.number}: Training completed but no val_loss found"
                    )

                if not math.isfinite(val_loss):
                    raise FloatingPointError(
                        "Trial returned a non-finite validation MSE."
                    )
                print(f"Trial {trial.number}: val_mse={val_loss:.6f}")
                return val_loss

            except subprocess.TimeoutExpired:
                print(f"Trial {trial.number}: Timeout after {args.timeout}s")
                raise optuna.TrialPruned(f"Timeout after {args.timeout}s")
            except Exception as e:
                print(f"Trial {trial.number}: Error - {e}")
                raise
            finally:
                if gpu_id is not None:
                    gpu_pool.put(gpu_id)

    return objective


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="WaveDL Hyperparameter Optimization with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    wavedl-hpo --data_path train.npz --n_trials 50
    wavedl-hpo --data_path train.npz --n_trials 30 --quick
    wavedl-hpo --data_path train.npz --n_trials 100 --models cnn resnet18
        """,
    )

    # Required
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data"
    )

    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="Allow trusted legacy NPZ object arrays (can execute arbitrary code).",
    )

    parser.add_argument(
        "--import",
        dest="import_modules",
        nargs="+",
        default=[],
        help="Trusted custom model modules or Python files to import.",
    )

    # HPO settings
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of HPO trials (default: 50)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel trials (-1 = auto-detect GPUs, default: -1)",
    )
    # --quick and --medium are mutually exclusive search-space presets.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: search fewer parameters (fastest, least thorough)",
    )
    mode_group.add_argument(
        "--medium",
        action="store_true",
        help="Medium mode: balanced parameter search (between --quick and full)",
    )
    parser.add_argument(
        "--inprocess",
        action="store_true",
        help="Run trials in-process (enables pruning, faster, but no GPU memory isolation)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per trial in seconds (default: 3600)",
    )

    # Search space customization
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            f"Models to search (vector default: {DEFAULT_MODELS}; "
            f"dense-field default: {DENSE_MODELS})"
        ),
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        choices=DEFAULT_OPTIMIZERS,
        help=f"Optimizers to search (default: {DEFAULT_OPTIMIZERS})",
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=None,
        choices=DEFAULT_SCHEDULERS,
        help=f"Schedulers to search (default: {DEFAULT_SCHEDULERS})",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        default=None,
        choices=DEFAULT_LOSSES,
        help=f"Losses to search (default: {DEFAULT_LOSSES})",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Batch sizes to search (vector default: 16 32 64 128; "
            "dense-field default: 1 2 4 8)"
        ),
    )

    # Training settings for each trial
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Max epochs per trial (default: 50, use early stopping)",
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed (default: 2025)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="hpo_results.json",
        help="Output file for best params (default: hpo_results.json)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="wavedl_hpo",
        help="Optuna study name (default: wavedl_hpo)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: sqlite:///{study_name}.db). "
        "Set to 'none' to disable persistence.",
    )

    args = parser.parse_args()
    for name in ("n_trials", "max_epochs", "timeout"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1.")
    if args.batch_sizes and any(value < 1 for value in args.batch_sizes):
        parser.error("--batch_sizes values must be >= 1.")

    from wavedl.models import import_model_modules

    args.import_modules = [
        str(Path(name).resolve()) if name.endswith(".py") else name
        for name in args.import_modules
    ]
    import_model_modules(args.import_modules)

    # Convert to absolute path (child processes may run in different cwd)
    args.data_path = str(Path(args.data_path).resolve())

    # Validate data path
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)

    # Resolve the model search space once before creating any trials. Without
    # this check, a dense dataset combined with the historical CNN/ResNet
    # defaults would make every HPO trial fail after training startup.
    try:
        args.target_shape = _detect_target_shape(
            args.data_path, allow_pickle=args.allow_pickle
        )
        args.models = _resolve_models_for_target(
            args.target_shape,
            args.models,
            quick=args.quick,
            medium=args.medium,
        )
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))

    # Detect GPU count once (reused by subprocess trials for round-robin pinning).
    args.n_gpus = _detect_gpu_count()

    # Auto-detect GPUs for n_jobs if not specified
    if args.n_jobs == -1:
        args.n_jobs = args.n_gpus
        print(f"Auto-detected {args.n_jobs} GPU(s) for parallel trials")
    elif args.n_jobs < 1:
        print(f"Error: --n_jobs must be -1 (auto-detect) or >= 1, got {args.n_jobs}.")
        sys.exit(1)

    # Create study
    print("=" * 60)
    print("WaveDL Hyperparameter Optimization")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    target_kind = "dense 2D field" if len(args.target_shape) == 3 else "vector"
    print(f"Targets: {target_kind} {args.target_shape}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Trials: {args.n_trials}")
    # Determine mode name for display
    if args.quick:
        mode_name = "Quick"
    elif args.medium:
        mode_name = "Medium"
    else:
        mode_name = "Full"

    print(
        f"Mode: {mode_name}"
        + (" (in-process, pruning enabled)" if args.inprocess else " (subprocess)")
    )
    print(f"Parallel jobs: {args.n_jobs}")
    print("=" * 60)

    # Use MedianPruner only for in-process mode (subprocess trials can't report)
    if args.inprocess:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    else:
        # NopPruner for subprocess mode - pruning has no effect there
        pruner = optuna.pruners.NopPruner()

    # Determine storage backend for study persistence
    if args.storage and args.storage.lower() == "none":
        storage = None  # Explicitly disabled
    elif args.storage:
        storage = args.storage  # User-specified
    else:
        # Default: SQLite file alongside results for resumability
        storage = f"sqlite:///{args.study_name}.db"

    if storage:
        print(f"Study persistence: {storage}")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    if storage:
        with open(args.data_path, "rb") as data_file:
            data_digest = hashlib.file_digest(data_file, "sha256").hexdigest()
        signature = {
            "data_sha256": data_digest,
            "target_shape": list(args.target_shape),
            "selection_metric": "normalized_mse_v1",
            "seed": args.seed,
            "models": args.models,
            "optimizers": args.optimizers,
            "schedulers": args.schedulers,
            "losses": args.losses,
            "batch_sizes": args.batch_sizes,
            "quick": args.quick,
            "medium": args.medium,
            "max_epochs": args.max_epochs,
            "inprocess": args.inprocess,
            "import_modules": args.import_modules,
        }
        prior_signature = study.user_attrs.get("wavedl_signature")
        if (prior_signature is not None and prior_signature != signature) or (
            study.trials and prior_signature is None
        ):
            parser.error(
                "Existing study belongs to different data/search settings or an older objective. Use a new --study_name or --storage none."
            )
        study.set_user_attr("wavedl_signature", signature)

    # Surface implicit resume: a persistent study with the same name is reused,
    # which appends to prior trials. Warn so a search-space change isn't silent.
    if storage and len(study.trials) > 0:
        print(
            f"⚠️  Resuming existing study '{args.study_name}' with "
            f"{len(study.trials)} prior trial(s). If you changed the search space "
            "(models/optimizers/batch sizes/mode), pass a new --study_name or "
            "--storage none to avoid distribution-mismatch errors."
        )

    # In-process mode: force single-threaded to avoid GPU memory contention
    if args.inprocess and args.n_jobs > 1:
        print(
            f"\u26a0\ufe0f  --inprocess mode: overriding n_jobs={args.n_jobs} \u2192 1 "
            "(multiple in-process trials would share GPU memory)"
        )
        args.n_jobs = 1

    # Run optimization
    objective = create_objective(args)
    # catch=(Exception,): a trial that errors (OOM, NaN, bad config combo, crashed
    # subprocess) is recorded as FAILED and the sweep continues, instead of one
    # bad trial aborting the entire run. TrialPruned is handled separately.
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    # Filter completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not completed_trials:
        print("No trials completed successfully.")
        sys.exit(1)

    print(f"\nCompleted trials: {len(completed_trials)}/{args.n_trials}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.6f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results = {
        "best_value": study.best_value,
        "objective_metric": "normalized_mse",
        "seed": args.seed,
        "best_params": study.best_params,
        "n_trials": len(completed_trials),
        "study_name": args.study_name,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print command to train with best params
    print("\n" + "=" * 60)
    print("TO TRAIN WITH BEST PARAMETERS:")
    print("=" * 60)
    cmd_parts = ["wavedl-train"]
    cmd_parts.append(f"--data_path {shlex.quote(args.data_path)}")
    cmd_parts.append("--selection_metric mse")
    cmd_parts.append(f"--seed {args.seed}")
    if args.import_modules:
        cmd_parts.append(
            "--import " + " ".join(shlex.quote(name) for name in args.import_modules)
        )
    if args.allow_pickle:
        cmd_parts.append("--allow_pickle")
    for key, value in study.best_params.items():
        cmd_parts.append(f"--{key} {value}")
    print(" \\\n    ".join(cmd_parts))


if __name__ == "__main__":
    main()
