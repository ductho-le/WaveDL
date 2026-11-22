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
import json
import math
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from wavedl.models.registry import validate_model_kwargs
from wavedl.utils import setup_hpc_cache_dirs
from wavedl.utils.config import parse_model_kwargs


setup_hpc_cache_dirs()


try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Error: Optuna not installed. Run: pip install wavedl")
    sys.exit(1)


def _detect_cuda_device_ids() -> list[str]:
    """Return visible CUDA device identifiers, or an empty list when no GPUs exist."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [
            value.strip()
            for value in visible.split(",")
            if value.strip() and value.strip() != "-1"
        ]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [
                line for line in result.stdout.splitlines() if line.startswith("GPU ")
            ]
            return [str(index) for index in range(len(lines))]
    except Exception:
        pass
    return []


def _detect_gpu_count() -> int:
    """Return the number of GPUs via nvidia-smi (best effort, defaults to 1)."""
    return max(1, len(_detect_cuda_device_ids()))


# =============================================================================
# DEFAULT SEARCH SPACES
# =============================================================================

DEFAULT_MODELS = ["cnn", "resnet18", "resnet34"]
QUICK_MODELS = ["cnn"]
MEDIUM_MODELS = ["cnn", "resnet18"]
DENSE_MODELS = ["unet_field", "inversionnet", "inversionnet_resize"]
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

# Unweighted losses for HPO sweeps (weighted_mse without --loss_weights duplicates mse)
ALL_LOSSES = ["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"]
DEFAULT_LOSSES = ["mse", "mae", "huber", "smooth_l1", "log_cosh"]
QUICK_LOSSES = ["mse"]
MEDIUM_LOSSES = ["mse", "mae", "huber"]


def _peek_array_input_shape(
    data_path: str, *, allow_pickle: bool = False
) -> tuple[int, ...] | None:
    """Inspect input tensor shape for NPZ/HDF5/MAT without loading full HDF5/MAT data."""
    import zipfile

    import numpy as np
    from numpy.lib import format as np_format

    from wavedl.utils.data import INPUT_KEYS, OUTPUT_KEYS, DataSource, get_data_source

    data_format = DataSource.detect_format(data_path)
    if data_format in ("hdf5", "mat"):
        handle = get_data_source(data_format, allow_pickle=allow_pickle).load_mmap(
            data_path
        )
        try:
            return tuple(int(d) for d in handle.inputs.shape)
        finally:
            handle.close()

    if data_format == "npz":
        # Fast header-only peek inside the NPZ zip archive (avoids materializing large arrays)
        try:
            with zipfile.ZipFile(data_path, "r") as zf:
                names = zf.namelist()
                keys = [n[:-4] for n in names if n.endswith(".npy")]
                inp_key, _ = DataSource._find_paired_keys(keys, INPUT_KEYS, OUTPUT_KEYS)
                if inp_key is not None and f"{inp_key}.npy" in names:
                    with zf.open(f"{inp_key}.npy") as member:
                        version = np_format.read_magic(member)
                        shape, _, dtype = np_format._read_array_header(member, version)
                    if dtype != object:
                        return tuple(int(d) for d in shape)
        except Exception:
            pass
        # Fallback for object/sparse NPZ archives
        inp, _ = get_data_source("npz", allow_pickle=allow_pickle).load(data_path)
        return tuple(int(d) for d in np.shape(inp))

    return None


def _detect_input_channels(
    data_path: str,
    *,
    input_channels: int | None = None,
    single_channel: bool = False,
    allow_pickle: bool = False,
) -> int:
    """Detect and validate input channel count for manifests and array datasets."""
    if input_channels is not None and (
        isinstance(input_channels, bool)
        or not isinstance(input_channels, int)
        or input_channels < 1
    ):
        raise ValueError(
            f"--input_channels must be a positive integer >= 1, got {input_channels!r}."
        )
    if input_channels not in (None, 1) and single_channel:
        raise ValueError("--single_channel conflicts with --input_channels > 1.")

    if str(data_path).lower().endswith(".txt"):
        from wavedl.utils.preprocessing import ShardManifestDataset

        dataset = ShardManifestDataset(data_path)
        try:
            detected = int(dataset.in_channels)
        finally:
            dataset.close()
        if input_channels is not None and input_channels != detected:
            raise ValueError(
                f"Requested --input_channels={input_channels}, "
                f"but manifest has {detected} channels."
            )
        return detected

    raw_shape = _peek_array_input_shape(data_path, allow_pickle=allow_pickle)
    if raw_shape is None:
        return input_channels if input_channels is not None else 1

    ndim = len(raw_shape)
    if input_channels is not None:
        if ndim >= 5 or (ndim in (3, 4) and input_channels > 1):
            if raw_shape[1] != input_channels:
                raise ValueError(
                    f"Requested --input_channels={input_channels}, "
                    f"but input data has shape {raw_shape} (axis 1 = {raw_shape[1]})."
                )
        return int(input_channels)

    if ndim >= 5:
        return int(raw_shape[1])
    if ndim == 4 and not single_channel:
        # Reject channels-last (N, H, W, C) mistake consistently with load_test_data
        if raw_shape[-1] <= 4 and raw_shape[1] > 16 and raw_shape[2] > 16:
            raise ValueError(
                f"Input appears to be channels-last format: {raw_shape}. "
                "WaveDL expects channels-first (N, C, H, W). "
                "If this is a shallow 3D volume, pass --single_channel."
            )
        if raw_shape[1] <= 16:
            return int(raw_shape[1])
    return 1


def _detect_target_shape(
    data_path: str, *, allow_pickle: bool = False
) -> tuple[int, ...]:
    """Inspect labels and return WaveDL's canonical per-sample shape."""
    from wavedl.utils.data import DataSource, get_data_source, load_outputs_only
    from wavedl.utils.targets import canonical_target_shape

    if str(data_path).lower().endswith(".txt"):
        from wavedl.utils.preprocessing import ShardManifestDataset

        dataset = ShardManifestDataset(data_path)
        try:
            assert dataset.target_shape is not None
            return dataset.target_shape
        finally:
            dataset.close()

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
    in_channels: int = 1,
) -> list[str]:
    """Choose an HPO model space and reject incompatible model/target/channel pairs."""
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
    channel_incompatible = []
    for model_name in models:
        model_class = get_model(model_name)
        if (
            is_field and not getattr(model_class, "supports_spatial_output", False)
        ) or (not is_field and getattr(model_class, "requires_spatial_output", False)):
            incompatible.append(model_name)
        supported_ch = getattr(model_class, "supported_input_channels", (1,))
        if supported_ch is not None and in_channels not in supported_ch:
            channel_incompatible.append(model_name)

    if incompatible:
        target_kind = "dense 2D field" if is_field else "vector"
        recommendation = (
            f"Use --models {' '.join(DENSE_MODELS)} or register a model with "
            "supports_spatial_output=True."
            if is_field
            else "Choose a vector-output model such as cnn or unet_regression."
        )
        raise ValueError(
            f"Model(s) {', '.join(incompatible)} are incompatible with {target_kind} "
            f"targets shaped {target_shape}. {recommendation}"
        )

    if channel_incompatible:
        if requested_models:
            raise ValueError(
                f"Model(s) {', '.join(channel_incompatible)} do not support "
                f"in_channels={in_channels}."
            )
        models = [m for m in models if m not in channel_incompatible]
        if not models:
            raise ValueError(
                f"No default HPO models support in_channels={in_channels}. "
                "Specify compatible models via --models."
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


def _trial_input_channels(args: argparse.Namespace) -> int | None:
    """Return the channel override to use for both trials and retraining."""
    explicit = getattr(args, "input_channels", None)
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return explicit
    detected = vars(args).get("in_channels", 1)
    return detected if isinstance(detected, int) and detected > 1 else None


def _trial_runtime_flags(args: argparse.Namespace) -> list[str]:
    flags = []
    precision = getattr(args, "precision", None)
    if isinstance(precision, str) and precision in {"bf16", "fp16", "no"}:
        flags.extend(["--precision", precision])
    input_channels = _trial_input_channels(args)
    if input_channels is not None:
        flags.extend(["--input_channels", str(input_channels)])
    if getattr(args, "single_channel", False) is True:
        flags.append("--single_channel")
    return flags


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
        in_channels=vars(args).get("in_channels", 1),
    )
    model_kwargs = vars(args).get("model_kwargs", {})
    if model_kwargs:
        for model in models:
            validate_model_kwargs(model, model_kwargs)

    # Lease actual visible device IDs. Round-robin trial numbers can assign a
    # new trial to a GPU whose earlier (slower) trial is still running.
    gpu_pool = queue.Queue()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    explicit_ids = getattr(args, "_cuda_device_ids", None)
    n_gpus = getattr(args, "n_gpus", None)
    if visible is not None:
        gpu_ids = [
            value.strip()
            for value in visible.split(",")
            if value.strip() and value.strip() != "-1"
        ]
    elif isinstance(explicit_ids, (list, tuple)):
        gpu_ids = [str(v) for v in explicit_ids]
    elif isinstance(n_gpus, int) and not isinstance(n_gpus, bool) and n_gpus > 1:
        gpu_ids = [str(index) for index in range(n_gpus)]
    else:
        gpu_ids = _detect_cuda_device_ids()

    n_jobs = getattr(args, "n_jobs", 1)
    n_jobs = n_jobs if isinstance(n_jobs, int) and not isinstance(n_jobs, bool) else 1
    # Exclusive GPU leasing: place each visible GPU ID into the pool once so two
    # concurrent trials can never share the same GPU even if n_jobs > len(gpu_ids).
    if gpu_ids:
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

        shared_cache_dir = getattr(args, "_shared_cache_dir", None)
        shared_cache_dir = (
            shared_cache_dir
            if isinstance(shared_cache_dir, str) and shared_cache_dir
            else None
        )

        # ==================================================================
        # IN-PROCESS MODE: Direct function call with pruning support
        # ==================================================================
        if args.inprocess:
            from wavedl.train import train_single_trial

            trial_kwargs = {}
            if model_kwargs:
                trial_kwargs["model_kwargs"] = model_kwargs
            if shared_cache_dir is not None:
                trial_kwargs["cache_dir"] = shared_cache_dir
            hpo_prec = getattr(args, "precision", None)
            if isinstance(hpo_prec, str) and hpo_prec in {"bf16", "fp16", "no"}:
                trial_kwargs["precision"] = hpo_prec
            input_channels = _trial_input_channels(args)
            if input_channels is not None:
                trial_kwargs["input_channels"] = input_channels
            if getattr(args, "single_channel", False) is True:
                trial_kwargs["single_channel"] = True

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
                    **trial_kwargs,
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
            "--scheduler_patience",
            str(max(1, patience // 2)),
            "--epochs",
            str(args.max_epochs),
            "--seed",
            str(args.seed),
            "--selection_metric",
            "mse",
            "--cache_validate",
            "fast",
            "--workers",
            "0",
        ]
        if model_kwargs:
            cmd.extend(["--model_kwargs", json.dumps(model_kwargs)])
        cmd.extend(_trial_runtime_flags(args))
        if shared_cache_dir is not None and (
            n_jobs == 1 or getattr(args, "_shared_cache_ready", False) is True
        ):
            cmd.extend(["--cache_dir", shared_cache_dir])
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
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            cmd.extend(["--output_dir", tmpdir])
            history_file = Path(tmpdir) / "training_history.csv"

            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            # Strip distributed environment variables inherited from parent
            # SLURM/torchrun jobs so single-GPU trial subprocesses never hang
            # trying to form a multi-rank DDP rendezvous.
            for dist_var in (
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            ):
                env.pop(dist_var, None)
            gpu_id = None
            if args.n_jobs > 1 and gpu_ids:
                gpu_id = gpu_pool.get()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

            # Run training
            # Note: We inherit the user's cwd instead of setting cwd=Path(__file__).parent
            # because site-packages may be read-only and train.py creates cache directories
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
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

                        with open(history_file, encoding="utf-8") as f:
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
                    # Fallback: parse stdout/stderr for training log format
                    # Pattern: "[HH:MM:SS | INFO | ] epoch | train_loss | val_loss | ..."
                    # Use regex to avoid false positives from unrelated lines
                    import re

                    log_pattern = re.compile(
                        r"^(?:\s*\d{2}:\d{2}:\d{2}\s*\|\s*INFO\s*\|\s*)?\s*\d+\s*\|\s*[-+eE0-9.]+\s*\|\s*([-+eE0-9.]+)\s*\|"
                    )
                    val_losses_stdout = []
                    combined_logs = f"{result.stdout or ''}\n{result.stderr or ''}"
                    for line in combined_logs.split("\n"):
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
        "--input_channels",
        type=int,
        default=None,
        help="Explicit input channel count for multi-channel arrays or shallow 3D volumes",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="Confirm 4D array data is a single-channel 3D volume (N, D, H, W)",
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
        "--model_kwargs",
        type=parse_model_kwargs,
        default={},
        help="Fixed architecture overrides as a JSON object, valid for all selected models.",
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
        choices=ALL_LOSSES,
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
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["bf16", "fp16", "no"],
        help="Mixed precision mode for trials (default: auto-detect bf16/fp16/no)",
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
        args.in_channels = _detect_input_channels(
            args.data_path,
            input_channels=args.input_channels,
            single_channel=args.single_channel,
            allow_pickle=args.allow_pickle,
        )
        args.models = _resolve_models_for_target(
            args.target_shape,
            args.models,
            quick=args.quick,
            medium=args.medium,
            in_channels=args.in_channels,
        )
        if args.model_kwargs:
            for model in args.models:
                validate_model_kwargs(model, args.model_kwargs)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))

    # Detect GPU count once (reused by subprocess trials for exclusive leasing).
    args._cuda_device_ids = _detect_cuda_device_ids()
    args.n_gpus = max(1, len(args._cuda_device_ids))

    # Auto-detect GPUs for n_jobs if not specified
    if args.n_jobs == -1:
        args.n_jobs = args.n_gpus
        print(f"Auto-detected {args.n_jobs} GPU(s) for parallel trials")
    elif args.n_jobs < 1:
        print(f"Error: --n_jobs must be -1 (auto-detect) or >= 1, got {args.n_jobs}.")
        sys.exit(1)
    elif (
        not args.inprocess
        and args._cuda_device_ids
        and args.n_jobs > len(args._cuda_device_ids)
    ):
        print(
            f"⚠️  --n_jobs={args.n_jobs} exceeds visible GPU count "
            f"({len(args._cuda_device_ids)}); capping n_jobs → "
            f"{len(args._cuda_device_ids)} to preserve exclusive per-GPU leasing."
        )
        args.n_jobs = len(args._cuda_device_ids)

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

    # In-process mode: force single-threaded to avoid GPU memory contention
    if args.inprocess and args.n_jobs > 1:
        print(
            f"\u26a0\ufe0f  --inprocess mode: overriding n_jobs={args.n_jobs} \u2192 1 "
            "(multiple in-process trials would share GPU memory)"
        )
        args.n_jobs = 1

    # Determine storage backend for study persistence
    storage_url: str | None
    if args.storage and args.storage.lower() == "none":
        storage_url = None  # Explicitly disabled
        storage = None
    elif args.storage:
        storage_url = args.storage  # User-specified
        storage = storage_url
    else:
        # Default: SQLite file alongside results for resumability
        safe_study = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", args.study_name).strip("._")
            or "wavedl_hpo"
        )
        out_parent = Path(args.output).resolve().parent
        out_parent.mkdir(parents=True, exist_ok=True)
        db_path = (out_parent / f"{safe_study}.db").as_posix()
        storage_url = f"sqlite:///{db_path}"
        storage = storage_url

    if storage_url:
        print(f"Study persistence: {storage_url}")
        if storage_url.startswith("sqlite:///") and args.n_jobs > 1:
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={"connect_args": {"timeout": 60}},
            )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    if storage_url:
        from wavedl.utils.data import _compute_dataset_fingerprint

        signature = {
            "data_fingerprint": _compute_dataset_fingerprint(args.data_path),
            "in_channels": args.in_channels,
            "input_channels": _trial_input_channels(args),
            "single_channel": args.single_channel,
            "precision": args.precision,
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
        if args.model_kwargs:
            signature["model_kwargs"] = args.model_kwargs
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
    prior_trial_count = len(study.trials)
    if storage_url and prior_trial_count > 0:
        print(
            f"⚠️  Resuming existing study '{args.study_name}' with "
            f"{prior_trial_count} prior trial(s). If you changed the search space "
            "(models/optimizers/batch sizes/mode), pass a new --study_name or "
            "--storage none to avoid distribution-mismatch errors."
        )

    # Run optimization with a shared temporary memmap cache.
    # For parallel subprocess sweeps (n_jobs > 1) on non-manifest datasets,
    # pre-populate the shared cache once in the parent so all workers reuse it
    # read-only without building duplicate multi-GB memmap files per trial.
    with tempfile.TemporaryDirectory(
        prefix="wavedl_hpo_cache_", ignore_cleanup_errors=True
    ) as shared_cache_dir:
        args._shared_cache_dir = shared_cache_dir
        args._shared_cache_ready = False
        if (
            not args.inprocess
            and args.n_jobs > 1
            and not str(args.data_path).lower().endswith(".txt")
            and args.in_channels == 1
        ):
            try:
                import logging as _logging

                import torch as _torch

                from wavedl.utils.data import prepare_data as _prepare_data

                class _CacheOnlyAccelerator:
                    is_main_process = True
                    device = _torch.device("cpu")
                    num_processes = 1

                    @staticmethod
                    def wait_for_everyone():
                        pass

                cache_args = argparse.Namespace(
                    data_path=args.data_path,
                    batch_size=16,
                    workers=0,
                    seed=args.seed,
                    cache_validate="fast",
                    single_channel=args.single_channel,
                    allow_pickle=args.allow_pickle,
                )
                _prepare_data(
                    cache_args,
                    _logging.getLogger("HPO"),
                    _CacheOnlyAccelerator(),
                    cache_dir=shared_cache_dir,
                    val_split=0.2,
                )
                args._shared_cache_ready = True
            except Exception as cache_err:
                print(
                    f"⚠️  Could not pre-populate shared HPO cache ({cache_err}); "
                    "trials will use per-trial caches."
                )
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

    total_trials = len(study.trials)
    print(f"\nCompleted trials: {len(completed_trials)}/{total_trials}")
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

    if args.model_kwargs:
        results["model_kwargs"] = args.model_kwargs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print command to train with best params
    print("\n" + "=" * 60)
    print("TO TRAIN WITH BEST PARAMETERS:")
    print("=" * 60)
    cmd_parts = ["wavedl-train"]
    cmd_parts.append(f"--data_path {shlex.quote(args.data_path)}")
    cmd_parts.append("--selection_metric mse")
    cmd_parts.append(f"--epochs {args.max_epochs}")
    cmd_parts.append(f"--seed {args.seed}")
    runtime_flags = _trial_runtime_flags(args)
    if runtime_flags:
        cmd_parts.append(shlex.join(runtime_flags))
    if args.import_modules:
        cmd_parts.append(
            "--import " + " ".join(shlex.quote(name) for name in args.import_modules)
        )
    if args.allow_pickle:
        cmd_parts.append("--allow_pickle")
    if args.model_kwargs:
        cmd_parts.append(f"--model_kwargs {shlex.quote(json.dumps(args.model_kwargs))}")
    for key, value in study.best_params.items():
        cmd_parts.append(f"--{key} {shlex.quote(str(value))}")
    if "patience" in study.best_params:
        sched_pat = max(1, int(study.best_params["patience"]) // 2)
        cmd_parts.append(f"--scheduler_patience {sched_pat}")
    print(" \\\n    ".join(cmd_parts))


if __name__ == "__main__":
    main()
