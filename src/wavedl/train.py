"""
WaveDL - Deep Learning for Wave-based Inverse Problems
=======================================================
Target Environment: NVIDIA HPC GPUs (Multi-GPU DDP) | PyTorch 2.x | Python 3.11+

A modular training framework for wave-based inverse problems and regression:
  1. HPC-Grade DDP Training: BF16/FP16 mixed precision with torch.compile support
  2. Dynamic Model Selection: Use --model flag to select any registered architecture
  3. Zero-Copy Data Engine: Memmap-backed datasets for large-scale training
  4. Physics-Aware Metrics: Real-time physical MAE with proper unscaling
  5. Robust Checkpointing: Resume training, periodic saves, and training curves
  6. Deep Observability: WandB integration with scatter analysis

Usage:
    # Recommended: Universal training command (works on local machines and HPC)
    wavedl-train --model cnn --batch_size 128 --compile

    # Multi-GPU with explicit config
    wavedl-train --num_gpus 4 --model cnn --output_dir results

    # Resume from checkpoint
    wavedl-train --model cnn --output_dir results  # auto-resumes if interrupted

    # List available models
    wavedl-train --list_models

Note:
    wavedl-train automatically detects your environment:
    - HPC clusters (SLURM, PBS, etc.): Uses local caching, offline WandB
    - Local machines: Uses standard cache locations (~/.cache)

Author: Ductho Le (ductho.le@outlook.com)
"""

from __future__ import annotations

# =============================================================================
# HPC Environment Setup (MUST be before any library imports)
# =============================================================================
# Auto-configure writable cache directories when home is not writable.
# Uses current working directory as fallback - works on HPC and local machines.
import os

# Import and call HPC cache setup before any library imports
from wavedl.utils import setup_hpc_cache_dirs


setup_hpc_cache_dirs()


def _setup_per_rank_compile_cache() -> None:
    """Set per-GPU Triton/Inductor cache to prevent multi-process race warnings.

    When using torch.compile with multiple GPUs, all processes try to write to
    the same cache directory, causing 'Directory is not empty - skipping!' warnings.
    This gives each GPU rank its own isolated cache subdirectory.
    """
    # Get local rank from environment (set by accelerate/torchrun)
    local_rank = os.environ.get("LOCAL_RANK", "0")

    # Get cache base from environment or use CWD
    cache_base = os.environ.get(
        "TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton_cache")
    )

    # Set per-rank cache directories
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_base, f"rank_{local_rank}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR", os.path.join(os.getcwd(), ".inductor_cache")
        ),
        f"rank_{local_rank}",
    )

    # Create directories
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)


# Setup per-rank compile caches (before torch imports)
_setup_per_rank_compile_cache()

# =============================================================================
# Standard imports (after environment setup)
# =============================================================================
import gc
import json
import logging
import math
import pickle
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import argparse

    import optuna


# Suppress Pydantic warnings from accelerate's internal Field() usage
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Force non-interactive Agg backend BEFORE importing pyplot.
# On HPC compute nodes there is no $DISPLAY / Wayland compositor, so the default
# Tk/Qt/Wx backend raises an error at import time. Agg renders to in-memory
# buffers and fully supports savefig(), so training curves are saved correctly.
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from wavedl.checkpointing import (
    publish_cv_outputs,
    save_emergency_checkpoint,
    save_normal_checkpoint,
    unwrap_checkpoint_model,
)
from wavedl.core.checkpoint_selection import (
    read_selection_metadata,
    resolve_checkpoint_path,
    select_checkpoint,
)
from wavedl.core.events import MetricPayload, StatePayload, SupersedePayload
from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
from wavedl.core.stdio import configure_utf8_stdio
from wavedl.core.training_cli import parse_training_args
from wavedl.journal import _read_manifest
from wavedl.models import build_model, get_model, list_models
from wavedl.runtime import open_training_runtime, training_lifecycle, wait_for_workers
from wavedl.utils import (
    FIGURE_DPI,
    MetricTracker,
    StreamingFieldMetrics,
    broadcast_early_stop,
    broadcast_value,
    calc_pearson,
    canonicalize_targets,
    create_training_curves,
    flatten_targets_by_channel,
    get_loss,
    get_lr,
    get_optimizer,
    get_scheduler,
    is_epoch_based,
    list_losses,
    list_optimizers,
    list_schedulers,
    plot_scientific_scatter,
    prepare_data,
    scaler_target_shape,
)


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ==============================================================================
# RUNTIME CONFIGURATION (post-import)
# ==============================================================================
# Note: matplotlib cache directory is already configured by setup_hpc_cache_dirs()
# called at module load time. No additional MPLCONFIGDIR setup needed here.

# Suppress warnings from known-noisy libraries, but preserve legitimate warnings
# from torch/numpy about NaN, dtype, and numerical issues.
# Scope suppression to known-noisy libraries instead of blanket suppression
for _mod in ("sklearn", "timm", "torchvision", "scipy"):
    warnings.filterwarnings("ignore", category=FutureWarning, module=_mod)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=_mod)
# Pydantic v1/v2 compatibility warnings
warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
# Transformer library warnings (loading configs, etc.)
warnings.filterwarnings("ignore", module="transformers")
# Accelerate verbose messages
warnings.filterwarnings("ignore", module="accelerate")
# torch.compile backend selection warnings
warnings.filterwarnings("ignore", message=".*TorchDynamo.*")
warnings.filterwarnings("ignore", message=".*Dynamo is not supported.*")
# Note: UserWarning from torch/numpy core is NOT suppressed to preserve
# legitimate warnings about NaN values, dtype mismatches, etc.

# ==============================================================================
# GPU PERFORMANCE OPTIMIZATIONS (Ampere/Hopper: A100, H100)
# ==============================================================================
# Enable TF32 for faster matmul (safe precision for training, ~2x speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # Use TF32 for float32 ops

# Enable cuDNN autotuning for fixed-size inputs (CNN-like models benefit most)
# Note: First few batches may be slower due to benchmarking
torch.backends.cudnn.benchmark = True

WANDB_SCATTER_INTERVAL = 5


# ==============================================================================
# LOGGING UTILITIES
# ==============================================================================
from contextlib import contextmanager


@contextmanager
def suppress_accelerate_logging():
    """Temporarily suppress accelerate's verbose checkpoint save messages."""
    accelerate_logger = logging.getLogger("accelerate.checkpointing")
    original_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(original_level)


def _build_model_for_targets(
    model_name: str,
    in_shape: tuple[int, ...],
    out_dim: int,
    scaler: Any,
    **kwargs,
) -> tuple[torch.nn.Module, tuple[int, ...]]:
    """Build a vector or field model from the fitted target scaler."""

    target_shape = scaler_target_shape(scaler, out_dim)
    model_kwargs = dict(kwargs)
    model_class = get_model(model_name)
    if len(target_shape) > 1 and not getattr(
        model_class, "supports_spatial_output", False
    ):
        raise ValueError(
            f"Model '{model_name}' produces vector outputs, but the dataset has "
            f"dense targets with shape {target_shape}. Use --model unet_field or "
            "register a model with supports_spatial_output=True."
        )
    if len(target_shape) == 1 and getattr(
        model_class, "requires_spatial_output", False
    ):
        raise ValueError(
            f"Model '{model_name}' produces dense fields, but the dataset has "
            f"vector targets with shape {target_shape}. Choose a vector model such "
            "as --model cnn, or provide (N,H,W)/(N,C,H,W) targets."
        )
    if len(target_shape) > 1:
        model_kwargs["out_shape"] = target_shape

    model = build_model(model_name, in_shape=in_shape, out_size=out_dim, **model_kwargs)
    return model, target_shape


def _validate_resume_metadata(
    metadata: dict[str, Any],
    model_name: str,
    in_shape: tuple[int, ...],
    out_dim: int,
    target_shape: tuple[int, ...],
) -> None:
    """Reject checkpoints that describe a different model or target contract."""

    mismatches = []
    if metadata.get("model_name") not in (None, model_name):
        mismatches.append(
            f"model_name={metadata['model_name']!r} (current {model_name!r})"
        )
    if metadata.get("in_shape") is not None and tuple(metadata["in_shape"]) != tuple(
        in_shape
    ):
        mismatches.append(
            f"in_shape={tuple(metadata['in_shape'])} (current {tuple(in_shape)})"
        )
    if metadata.get("out_dim") not in (None, out_dim):
        mismatches.append(f"out_dim={metadata['out_dim']} (current {out_dim})")
    if metadata.get("target_shape") is not None and tuple(
        metadata["target_shape"]
    ) != tuple(target_shape):
        mismatches.append(
            f"target_shape={tuple(metadata['target_shape'])} "
            f"(current {tuple(target_shape)})"
        )

    if mismatches:
        raise ValueError(
            "Checkpoint is incompatible with the current training run: "
            + "; ".join(mismatches)
            + ". Use the matching data/model configuration or start with --fresh."
        )


def _validate_emergency_resume(
    checkpoint_dir: str, scheduler_name: str, epochs: int | None = None
) -> bool:
    """Reject batch schedules that cannot safely replay a partial epoch.

    Emergency checkpoints lack a resumable data-loader position and partial
    gradients. Their optimizer state can be reused for approximate recovery,
    but replaying batches would overrun OneCycle's fixed update budget. An
    interruption exactly at an epoch boundary replays nothing: its schedule
    position equals the completed epochs' update count, so it resumes in place.
    """
    if not os.path.isfile(os.path.join(checkpoint_dir, "model_weights.pth")):
        return False

    state = None
    scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
    saved_batch_schedule = False
    if os.path.isfile(scheduler_path):
        state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
        # Identify legacy OneCycle checkpoints even when the caller changes
        # --scheduler; loading that state into an epoch scheduler is unsafe too.
        saved_batch_schedule = isinstance(state, dict) and "total_steps" in state

    if (
        not is_epoch_based(scheduler_name)
        and saved_batch_schedule
        and _emergency_at_epoch_boundary(checkpoint_dir, state, epochs)
    ):
        return True
    if not is_epoch_based(scheduler_name) or saved_batch_schedule:
        raise ValueError(
            f"Cannot safely resume the emergency checkpoint '{checkpoint_dir}' "
            "with a OneCycle/per-batch scheduler: replaying its interrupted epoch "
            "can exhaust the saved schedule. Use --resume with an "
            "epoch_*_checkpoint or best_checkpoint saved at an epoch boundary, "
            "or use --fresh if no epoch-boundary checkpoint is available. "
            "The emergency checkpoint has been preserved."
        )
    return True


def _emergency_at_epoch_boundary(
    checkpoint_dir: str, state: dict, epochs: int | None
) -> bool:
    """Whether a batch schedule was saved after exactly the completed epochs."""
    meta_path = os.path.join(checkpoint_dir, "training_meta.pkl")
    if not epochs or not os.path.isfile(meta_path):
        return False
    with open(meta_path, "rb") as f:
        completed = pickle.load(f).get("epoch")
    total = state.get("total_steps")
    if type(completed) is not int or type(total) is not int or total % epochs:
        return False
    return state.get("last_epoch") == completed * (total // epochs)


def _require_finite_loss(loss, phase, accelerator=None):
    valid = torch.isfinite(loss.detach()).all().to(dtype=torch.int32)
    if accelerator is not None:
        valid = accelerator.reduce(valid, reduction="sum")
        expected = accelerator.num_processes
    else:
        expected = 1
    if valid.item() != expected:
        raise FloatingPointError(f"{phase} loss is non-finite; training aborted.")


def _flag_nonfinite_loss(loss: torch.Tensor, nonfinite: torch.Tensor) -> torch.Tensor:
    """Record a non-finite micro-batch loss on device, without a host sync."""
    return nonfinite | ~torch.isfinite(loss.detach()).all()


def _finalize_accumulation_window(
    model: torch.nn.Module,
    samples: int,
    nonfinite: torch.Tensor,
    nominal_batch_size: int,
    accelerator: Accelerator,
) -> None:
    """Validate and rescale one optimizer update's accumulated gradients.

    Micro-batch losses are back-propagated with weight ``batch / nominal``, so
    a full window already holds the sample-mean gradient. A single collective
    per update checks every rank's losses and counts samples, before the
    optimizer can consume non-finite gradients. Only partial windows (e.g. the
    last batch of an epoch) need rescaling to the true sample mean.
    """
    stats = torch.stack(
        [
            torch.as_tensor(samples, device=accelerator.device, dtype=torch.int64),
            nonfinite.to(dtype=torch.int64),
        ]
    )
    total, nonfinite_count = accelerator.reduce(stats, reduction="sum").tolist()
    if nonfinite_count:
        raise FloatingPointError("Training loss is non-finite; training aborted.")
    expected = (
        accelerator.num_processes
        * accelerator.gradient_accumulation_steps
        * nominal_batch_size
    )
    if total != expected:
        factor = expected / total
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


def _archive_previous_run(output_dir):
    """Keep old artifacts recoverable while preventing stale auto-resume state."""
    import re
    from datetime import datetime
    from pathlib import Path

    output = Path(output_dir)
    names = {
        "best_checkpoint",
        "interrupted_checkpoint",
        "training_complete.flag",
        "best_model_weights.pth",
        "training_history.csv",
        "training_curves.png",
        "training_curves.svg",
    }
    artifacts = [
        path
        for path in output.iterdir()
        if path.name in names or re.fullmatch(r"epoch_\d+_checkpoint", path.name)
    ]
    if not artifacts:
        return None
    archive = output / "previous_runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive.mkdir(parents=True)
    for path in artifacts:
        path.rename(archive / path.name)
    if (output / "scaler.pkl").is_file():
        shutil.copy2(output / "scaler.pkl", archive / "scaler.pkl")
    return archive


# ==============================================================================
# STRUCTURED OUTPUT UTILITIES
# ==============================================================================
def _finite_or_none(value: Any) -> Any:
    """Recursively replace non-finite floats with None for strict JSON output.

    Python's json module emits bare ``Infinity``/``NaN`` tokens, which RFC 8259
    forbids and strict parsers reject -- including browser ``JSON.parse``, which
    consumes the structured metrics line. ``best_val_loss`` is infinity until the
    first checkpoint is written, so every fresh run would otherwise emit invalid
    JSON on epoch 1; diverged runs produce NaN in the losses and metrics.

    Tuples are normalised to lists so the result is always JSON-serialisable.
    """
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Parse command-line arguments with comprehensive options."""
    return parse_training_args(
        models=list_models(),
        losses=list_losses(),
        optimizers=list_optimizers(),
        schedulers=list_schedulers(),
    )


def _save_best_checkpoint(
    accelerator,
    model,
    args,
    epoch: int,
    best_val_loss: float,
    in_shape: tuple,
    out_dim: int,
    scaler,
    logger,
) -> None:
    """
    Save best checkpoint with metadata.

    Args:
        accelerator: Accelerator instance
        model: Model to save
        args: Command-line arguments
        epoch: Current epoch (0-indexed)
        best_val_loss: Best validation loss
        in_shape: Input shape
        out_dim: Number of vector targets or dense field channels
        scaler: Fitted vector or spatial target scaler
        logger: Logger instance
    """
    ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
    with suppress_accelerate_logging():
        accelerator.save_state(ckpt_dir, safe_serialization=False)

    if accelerator.is_main_process:
        with open(os.path.join(ckpt_dir, "training_meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "patience_ctr": 0,
                    "model_name": args.model,
                    "in_shape": in_shape,
                    "out_dim": out_dim,
                    "target_shape": scaler_target_shape(scaler, out_dim),
                },
                f,
            )

        # Save standalone weights
        try:
            unwrapped = accelerator.unwrap_model(model)
        except KeyError:
            unwrapped = model.module if hasattr(model, "module") else model
            if hasattr(unwrapped, "_orig_mod"):
                unwrapped = unwrapped._orig_mod

        torch.save(
            unwrapped.state_dict(),
            os.path.join(args.output_dir, "best_model_weights.pth"),
        )

        # Copy scaler for checkpoint portability
        scaler_src = os.path.join(args.output_dir, "scaler.pkl")
        scaler_dst = os.path.join(ckpt_dir, "scaler.pkl")
        if os.path.exists(scaler_src):
            shutil.copy2(scaler_src, scaler_dst)

        logger.info(f"   💾 Best model saved (val_loss: {best_val_loss:.6f})")


# ==============================================================================
# IN-PROCESS HPO TRAINING FUNCTION
# ==============================================================================


def train_single_trial(
    data_path: str,
    model_name: str = "cnn",
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 20,
    optimizer_name: str = "adamw",
    scheduler_name: str = "plateau",
    loss_name: str = "mse",
    weight_decay: float = 1e-4,
    seed: int = 2025,
    precision: str = "bf16",
    workers: int = 0,
    huber_delta: float = 1.0,
    momentum: float = 0.9,
    trial: optuna.trial.Trial | None = None,
    verbose: bool = False,
    selection_metric: str = "loss",
    allow_pickle: bool = False,
    timeout_seconds: float | None = None,
    *,
    grad_clip: float = 1.0,
) -> dict:
    """
    Single-trial training function for in-process HPO.

    This is a lightweight training loop designed for hyperparameter optimization
    that supports Optuna pruning callbacks. Unlike `main()`, this avoids
    Accelerator complexity for simpler single-GPU trials.

    Args:
        data_path: Path to training data (NPZ, HDF5, MAT)
        model_name: Model architecture name (from registry)
        lr: Learning rate
        batch_size: Batch size
        epochs: Maximum epochs
        patience: Early stopping patience
        optimizer_name: Optimizer name (from registry)
        scheduler_name: Scheduler name (from registry)
        loss_name: Loss function name (from registry)
        weight_decay: Weight decay for optimizer
        seed: Random seed
        precision: Mixed precision mode ("bf16", "fp16", "no")
        workers: DataLoader workers (0 for main process only)
        huber_delta: Delta for Huber loss
        momentum: Momentum for SGD optimizer
        grad_clip: Max gradient norm for clipping (0 to disable)
        trial: Optuna trial for pruning callbacks (None for standalone use)
        verbose: Print training progress

    Returns:
        dict with keys:
            - best_val_loss: Best validation loss achieved
            - epochs_trained: Number of epochs completed
            - final_val_loss: Validation loss at last epoch
            - pruned: Whether trial was pruned
    """
    import tempfile

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from wavedl.models import build_model
    from wavedl.utils import get_loss, get_optimizer, get_scheduler, prepare_data

    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds

    # Set seed for reproducibility
    set_seed(seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mixed precision setup
    use_amp = precision != "no" and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")

    # Load and prepare data using temporary directory.
    # ignore_cleanup_errors: on Windows the memmap cache (.dat) may still be held
    # open by the dataloaders when the block exits; tolerate the cleanup race
    # (the OS reclaims the temp dir) instead of crashing the trial.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        from types import SimpleNamespace

        args = SimpleNamespace(
            data_path=data_path,
            batch_size=batch_size,
            workers=workers,
            cache_validate="fast",
            single_channel=False,
            seed=seed,
            allow_pickle=allow_pickle,
        )

        # Create a dummy logger
        class DummyLogger:
            def info(self, msg):
                if verbose:
                    print(msg)

            def warning(self, msg):
                if verbose:
                    print(f"WARNING: {msg}")

            def error(self, msg):
                print(f"ERROR: {msg}")

        # Create a dummy accelerator for prepare_data compatibility
        # Note: explicit device capture needed since class body scope differs from function scope
        _device_for_accelerator = device

        class DummyAccelerator:
            is_main_process = True
            device = _device_for_accelerator
            num_processes = 1

            @staticmethod
            def wait_for_everyone():
                pass  # No-op for single-process

        train_dl, val_dl, target_scaler, in_shape, out_dim = prepare_data(
            args, DummyLogger(), DummyAccelerator(), cache_dir=tmpdir
        )

        # Build model
        model, _target_shape = _build_model_for_targets(
            model_name, in_shape, out_dim, target_scaler, pretrained=False
        )
        model = model.to(device)

        # Create optimizer
        optimizer = get_optimizer(
            name=optimizer_name,
            params=model.get_optimizer_groups(lr, weight_decay),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        # Create loss function
        criterion = get_loss(name=loss_name, delta=huber_delta)
        criterion = criterion.to(device)
        validation_criterion = (
            torch.nn.MSELoss() if selection_metric == "mse" else criterion
        )

        # Create scheduler
        scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            epochs=epochs,
            steps_per_epoch=len(train_dl),
            patience=patience // 2,  # Zero is valid: reduce after the first bad epoch
        )
        scheduler_step_per_batch = scheduler_name == "onecycle"

        # Training state
        best_val_loss = float("inf")
        patience_ctr = 0
        epochs_trained = 0
        final_val_loss = float("inf")

        # Training loop
        for epoch in range(epochs):
            epochs_trained = epoch + 1

            # === Training Phase ===
            model.train()
            train_loss_sum = 0.0
            train_samples = 0

            for x, y in train_dl:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Trial exceeded {timeout_seconds} seconds.")
                x, y = x.to(device), y.to(device)

                with torch.amp.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    pred = model(x)
                    if pred.shape != y.shape:
                        raise ValueError(
                            f"Model output shape {tuple(pred.shape)} does not match "
                            f"target shape {tuple(y.shape)}."
                        )
                    loss = criterion(pred, y)
                _require_finite_loss(loss, "Training")

                optimizer.zero_grad(set_to_none=True)

                if use_amp and precision == "fp16":
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_was_skipped = scaler.get_scale() < old_scale
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer_was_skipped = False

                if scheduler_step_per_batch and not optimizer_was_skipped:
                    scheduler.step()

                train_loss_sum += loss.item() * x.size(0)
                train_samples += x.size(0)

            avg_train_loss = train_loss_sum / train_samples

            # === Validation Phase ===
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0

            with torch.inference_mode():
                for x, y in val_dl:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(f"Trial exceeded {timeout_seconds} seconds.")
                    x, y = x.to(device), y.to(device)

                    with torch.amp.autocast(
                        device_type="cuda", dtype=amp_dtype, enabled=use_amp
                    ):
                        pred = model(x)
                        if pred.shape != y.shape:
                            raise ValueError(
                                f"Model output shape {tuple(pred.shape)} does not match "
                                f"target shape {tuple(y.shape)}."
                            )
                        loss = validation_criterion(pred, y)
                    _require_finite_loss(loss, "Validation")

                    val_loss_sum += loss.item() * x.size(0)
                    val_samples += x.size(0)

            avg_val_loss = val_loss_sum / val_samples
            final_val_loss = avg_val_loss

            # === Early Stopping Tracking (update best_val_loss before pruning check) ===
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_ctr = 0
            else:
                patience_ctr += 1

            # === Optuna Integration ===
            if trial is not None:
                # Report intermediate result
                trial.report(avg_val_loss, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    return {
                        "best_val_loss": best_val_loss,
                        "epochs_trained": epochs_trained,
                        "final_val_loss": final_val_loss,
                        "pruned": True,
                    }

            if patience_ctr >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            # === LR Scheduling ===
            if not scheduler_step_per_batch:
                if scheduler_name == "plateau":
                    scheduler.step(avg_val_loss)  # type: ignore[arg-type]
                else:
                    scheduler.step()

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}"
                )

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epochs_trained,
            "final_val_loss": final_val_loss,
            "pruned": False,
        }


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
def main():
    configure_utf8_stdio()
    args, parser = parse_args()

    # Load and merge config file if provided
    if args.config:
        from wavedl.utils.config import (
            load_config,
            merge_config_with_args,
            validate_config,
        )

        print(f"📄 Loading config from: {args.config}")
        config = load_config(args.config)

        # Merge config with CLI args (CLI takes precedence via parser defaults detection)
        args = merge_config_with_args(config, args, parser=parser)

    from wavedl.models import import_model_modules

    import_model_modules(args.import_modules)

    # Handle --list_models flag
    if args.list_models:
        print("Available models (output contract):")
        for name in list_models():
            ModelClass = get_model(name)
            supports_field = getattr(ModelClass, "supports_spatial_output", False)
            requires_field = getattr(ModelClass, "requires_spatial_output", False)
            if requires_field:
                output_contract = "dense 2D field"
            elif supports_field:
                output_contract = "vector + dense 2D field"
            else:
                output_contract = "vector"
            # Get first non-empty docstring line
            if ModelClass.__doc__:
                lines = [
                    l.strip() for l in ModelClass.__doc__.splitlines() if l.strip()
                ]
                doc_first_line = lines[0] if lines else "No description"
            else:
                doc_first_line = "No description"
            print(f"  - {name} [{output_contract}]: {doc_first_line}")
        sys.exit(0)

    from wavedl.utils.config import validate_config

    invalid_options = validate_config(vars(args), parser=parser)
    if invalid_options:
        parser.error("; ".join(invalid_options))
    if args.fresh and args.resume:
        parser.error("--fresh and --resume are mutually exclusive.")

    if args.grad_accum_steps < 1:
        raise ValueError(
            f"--grad_accum_steps must be >= 1, got {args.grad_accum_steps}"
        )
    accelerator = Accelerator(
        mixed_precision=args.precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        step_scheduler_with_optimizer=False,
        log_with="wandb" if args.wandb and WANDB_AVAILABLE else None,
    )
    cleanup_owned = False
    try:
        journal = open_training_runtime(args, accelerator)
        with training_lifecycle(journal):
            cleanup_owned = True
            _run_training_and_cleanup(args, accelerator, journal)
    finally:
        # Bootstrap has no prepared models. Once training starts, its cleanup
        # owner decides whether model release made backend destruction safe.
        if not cleanup_owned and dist.is_initialized():
            dist.destroy_process_group()


def _run_training_and_cleanup(args, accelerator, journal):
    """Release prepared models before destroying their distributed backend.

    A DDP reducer retains its process group. Destroying the default group first
    can leave the reducer as its final owner during interpreter teardown, where
    Gloo workers may need the GIL held by the reducer's destructor. Error
    tracebacks also retain the training frame, so release its inactive locals
    before clearing Accelerate's prepared-object references. Keep the original
    exception and traceback, and perform no cleanup collectives.
    """
    primary_error = None
    cleanup_errors = []
    try:
        _run_training(args, accelerator, journal)
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        primary_error = error
    try:
        accelerator.free_memory()
        # Accelerate collects before clearing its own prepared-object lists.
        # Collect remaining cycles while the default group still owns its backend.
        gc.collect()
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        cleanup_errors.append(("prepared-model release", error))
    else:
        try:
            accelerator.end_training()
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            cleanup_errors.append(("tracker/backend shutdown", error))
        # Tracker finalization can fail before end_training destroys the group.
        # Only attempt this fallback after prepared-model release succeeded.
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            cleanup_errors.append(("distributed backend shutdown", error))

    if cleanup_errors:
        failures = (
            [("training", primary_error)] if primary_error is not None else []
        ) + cleanup_errors
        details = "; ".join(
            f"{phase}: {type(error).__name__}: {str(error)[:256]}"
            for phase, error in failures
        )
        if cleanup_errors[0][0] == "prepared-model release":
            details += "; backend shutdown skipped: prepared-model release failed"
        raise BaseExceptionGroup(
            f"Training cleanup failed: {details}", [error for _, error in failures]
        )
    if primary_error is not None:
        raise primary_error


def _run_training(args, accelerator, journal):
    """Run the existing training algorithms under the actual runtime lifecycle."""
    journal.emit("state", StatePayload("running"))
    requested_device = journal.bootstrap.request_manifest.runtime.device
    if requested_device != "auto" and requested_device != accelerator.device.type:
        raise ValueError(
            f"Requested device {requested_device} differs from actual {accelerator.device.type}"
        )
    # Handle --cv flag (cross-validation mode)
    if args.cv > 0:
        from wavedl.core.training_cli import validate_cv_options

        validate_cv_options(args)
        if accelerator.num_processes != 1:
            raise ValueError("Cross-validation does not support distributed execution")
        print(f"🔄 Cross-Validation Mode: {args.cv} folds")
        from wavedl.utils.cross_validation import run_cross_validation

        # Load data for CV using memory-efficient loader
        from wavedl.utils.data import DataSource, get_data_source

        data_format = DataSource.detect_format(args.data_path)
        source = get_data_source(data_format, allow_pickle=args.allow_pickle)

        # Use memory-mapped loading when available (now returns LazyDataHandle for all formats)
        _cv_handle = None
        try:
            if hasattr(source, "load_mmap"):
                _cv_handle = source.load_mmap(args.data_path)
                X, y = _cv_handle.inputs, _cv_handle.outputs
            else:
                X, y = source.load(args.data_path)

            # Handle sparse matrices (must materialize for CV shuffling)
            if hasattr(X, "__getitem__") and len(X) > 0 and hasattr(X[0], "toarray"):
                X = np.stack([x.toarray() for x in X])

            y = canonicalize_targets(y)
            cv_target_shape = tuple(y.shape[1:])

            # Validate and determine input shape (consistent with prepare_data)
            # Check for ambiguous shapes that could be multi-channel or shallow 3D volume
            sample_shape = X.shape[1:]  # Per-sample shape

            # Same heuristic as prepare_data: detect ambiguous 3D shapes
            is_ambiguous_shape = (
                len(sample_shape) == 3  # Exactly 3D: could be (C, H, W) or (D, H, W)
                and sample_shape[0] <= 16  # First dim looks like channels
                and sample_shape[1] > 16
                and sample_shape[2] > 16  # Both spatial dims are large
            )

            if is_ambiguous_shape and not args.single_channel:
                raise ValueError(
                    f"Ambiguous input shape detected: sample shape {sample_shape}. "
                    f"This could be either:\n"
                    f"  - Multi-channel 2D data (C={sample_shape[0]}, H={sample_shape[1]}, W={sample_shape[2]})\n"
                    f"  - Single-channel 3D volume (D={sample_shape[0]}, H={sample_shape[1]}, W={sample_shape[2]})\n\n"
                    f"If this is single-channel 3D/shallow volume data, use --single_channel flag.\n"
                    f"If this is multi-channel 2D data, reshape to (N*C, H, W) with adjusted targets."
                )

            # in_shape = spatial dimensions for model registry (channel added during training)
            in_shape = sample_shape

            # Parse betas for CV (same logic as main training path)
            cv_betas = tuple(float(x.strip()) for x in args.betas.split(","))
            if len(cv_betas) != 2 or not all(0.0 <= beta < 1.0 for beta in cv_betas):
                raise ValueError("--betas must have exactly two values in [0, 1)")
            cv_loss_weights = (
                [float(x.strip()) for x in args.loss_weights.split(",")]
                if args.loss_weights
                else None
            )
            cv_milestones = (
                [int(x.strip()) for x in args.milestones.split(",")]
                if args.milestones
                else None
            )

            # Run cross-validation
            run_cross_validation(
                X=X,
                y=y,
                model_name=args.model,
                in_shape=in_shape,
                out_size=y.shape[1],
                out_shape=cv_target_shape,
                folds=args.cv,
                stratify=args.cv_stratify,
                stratify_bins=args.cv_bins,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                weight_decay=args.weight_decay,
                loss_name=args.loss,
                loss_weights=cv_loss_weights,
                huber_delta=args.huber_delta,
                pretrained=args.pretrained,
                optimizer_name=args.optimizer,
                scheduler_name=args.scheduler,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                min_lr=args.min_lr,
                betas=cv_betas,
                momentum=args.momentum,
                output_dir=args.output_dir,
                workers=args.workers,
                seed=args.seed,
                grad_clip=args.grad_clip,
                device=accelerator.device,
                nesterov=args.nesterov,
                scheduler_gamma=args.scheduler_factor,
                step_size=args.step_size,
                milestones=cv_milestones,
                warmup_epochs=args.warmup_epochs,
                grad_accum_steps=args.grad_accum_steps,
                precision=args.precision,
                constraint=args.constraint,
                constraint_file=args.constraint_file,
                constraint_weight=args.constraint_weight,
                constraint_reduction=args.constraint_reduction,
                deterministic=args.deterministic,
            )
            publish_cv_outputs(journal, args.output_dir, folds=args.cv)
        finally:
            # Clean up file handle if HDF5/MAT
            if _cv_handle is not None and hasattr(_cv_handle, "close"):
                try:
                    _cv_handle.close()
                except Exception as e:
                    logging.debug(f"Failed to close CV data handle: {e}")
        return

    # ==========================================================================
    # SYSTEM INITIALIZATION
    # ==========================================================================
    # Initialize Accelerator for DDP and mixed precision
    set_seed(args.seed)

    # Deterministic mode for scientific reproducibility
    # Disables TF32 and cuDNN benchmark for exact reproducibility (slower)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        accelerator.print("🔒 Deterministic mode enabled (slower but reproducible)")

    # Configure logging (rank 0 only prints to console)
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("Trainer")

    # Ensure output directory exists (critical for cache files, checkpoints, etc.)
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect optimal DataLoader workers if not specified
    if args.workers < 0:
        cpu_count = os.cpu_count() or 4
        num_gpus = accelerator.num_processes
        # Heuristic: 4-16 workers per GPU, bounded by available CPU cores
        # Increased cap from 8 to 16 for high-throughput GPUs (H100, A100)
        proposed = min(16, max(2, (cpu_count - 2) // num_gpus))

        # Probe shared memory: multi-worker DataLoaders use POSIX shm for
        # IPC. Environments with restricted /dev/shm (containers, CI, macOS
        # sandboxes) will crash with "torch_shm_manager ... Operation not
        # permitted".  Fall back to 0 workers (main-process loading) if shm
        # is unavailable.
        try:
            import multiprocessing.shared_memory as _shm

            test_block = _shm.SharedMemory(create=True, size=1)
            test_block.close()
            test_block.unlink()
            args.workers = proposed
        except (PermissionError, OSError):
            args.workers = 0
            if accelerator.is_main_process:
                logger.warning(
                    "⚠️  Shared memory unavailable — falling back to workers=0. "
                    "Multi-worker DataLoaders disabled."
                )

        if accelerator.is_main_process and args.workers > 0:
            logger.info(
                f"⚙️  Auto-detected workers: {args.workers} per GPU "
                f"(CPUs: {cpu_count}, GPUs: {num_gpus})"
            )

    if accelerator.is_main_process:
        logger.info(
            f"🚀 Training: {accelerator.num_processes} process(es) "
            f"on {accelerator.device.type.upper()}"
        )
        # Show the actual precision negotiated by Accelerator (may differ from
        # CLI default when launched via accelerate launch)
        actual_precision = accelerator.mixed_precision or "no"
        logger.info(
            f"   Model: {args.model} | Precision: {actual_precision} | Compile: {args.compile}"
        )
        logger.info(
            f"   Loss: {args.loss} | Optimizer: {args.optimizer} | Scheduler: {args.scheduler}"
        )
        logger.info(f"   Early Stopping Patience: {args.patience} epochs")
        if args.grad_accum_steps > 1:
            effective_bs = (
                args.batch_size * args.grad_accum_steps * accelerator.num_processes
            )
            logger.info(
                f"   Gradient Accumulation: {args.grad_accum_steps} steps "
                f"(effective batch size: {effective_bs})"
            )
        if args.save_every > 0:
            logger.info(f"   Periodic Checkpointing: Every {args.save_every} epochs")
        if args.resume:
            logger.info(f"   📂 Resuming from: {args.resume}")

        # Initialize WandB
        if args.wandb and WANDB_AVAILABLE:
            accelerator.init_trackers(
                project_name=args.project_name,
                config=vars(args),
                init_kwargs={"wandb": {"name": args.run_name or f"{args.model}_run"}},
            )

    # ==========================================================================
    # AUTO-RESUME / RESUME FROM CHECKPOINT
    # ==========================================================================
    start_epoch = 0
    best_val_loss = float("inf")
    patience_ctr = 0
    history: list[dict[str, Any]] = []

    # Define checkpoint paths
    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint")
    interrupted_ckpt_path = os.path.join(args.output_dir, "interrupted_checkpoint")
    complete_flag_path = os.path.join(args.output_dir, "training_complete.flag")

    if args.fresh:
        if accelerator.is_main_process:
            archived = _archive_previous_run(args.output_dir)
            if archived is not None:
                logger.info(f"Archived previous run to: {archived}")
        wait_for_workers(accelerator)
    elif (
        args.resume
        and accelerator.is_main_process
        and os.path.exists(complete_flag_path)
    ):
        os.remove(complete_flag_path)

    # Auto-resume logic (if not --fresh and no explicit --resume)
    # Priority: interrupted_checkpoint > latest periodic/best checkpoint
    # The interrupted checkpoint captures the latest state after Ctrl+C.
    # Periodic checkpoints may be newer than best_checkpoint after a
    # non-KeyboardInterrupt crash (e.g., OOM, hardware failure).
    training_already_complete = False
    restored_generation = None
    if not args.fresh and args.resume is None:
        if os.path.exists(complete_flag_path):
            # Training already completed — will exit after entering try block
            # so the finally block runs cleanup (WandB, DDP process group)
            training_already_complete = True
            if accelerator.is_main_process:
                logger.info(
                    "✅ Training already completed (early stopping). Use --fresh to retrain."
                )
        elif (
            journal.bootstrap.layout.checkpoints.is_dir()
            and (restored_generation := select_checkpoint(journal.bootstrap.layout))
            is not None
        ):
            args.resume = str(restored_generation.path)
        elif os.path.exists(interrupted_ckpt_path):
            # Prefer interrupted checkpoint (most recent optimizer/scheduler state)
            args.resume = interrupted_ckpt_path
            if accelerator.is_main_process:
                logger.info(
                    f"🔄 Auto-resuming from interrupted checkpoint: {interrupted_ckpt_path}"
                )
        else:
            # Find the latest checkpoint by epoch number across best and
            # periodic checkpoints.  A crash after a periodic save should
            # not rewind to an earlier best_checkpoint.
            latest_ckpt = None
            latest_epoch = -1

            # Check best_checkpoint
            if os.path.exists(best_ckpt_path):
                meta_file = os.path.join(best_ckpt_path, "training_meta.pkl")
                if os.path.exists(meta_file):
                    with open(meta_file, "rb") as f:
                        m = pickle.load(f)
                    ep = m.get("epoch", 0)
                    if ep > latest_epoch:
                        latest_epoch = ep
                        latest_ckpt = best_ckpt_path
                else:
                    # No metadata — treat as epoch 0 fallback
                    latest_ckpt = best_ckpt_path
                    latest_epoch = 0

            # Scan periodic epoch_*_checkpoint directories
            for entry in os.listdir(args.output_dir):
                if not entry.startswith("epoch_") or not entry.endswith("_checkpoint"):
                    continue
                ckpt_candidate = os.path.join(args.output_dir, entry)
                if not os.path.isdir(ckpt_candidate):
                    continue
                meta_file = os.path.join(ckpt_candidate, "training_meta.pkl")
                if os.path.exists(meta_file):
                    with open(meta_file, "rb") as f:
                        m = pickle.load(f)
                    ep = m.get("epoch", 0)
                    if ep > latest_epoch:
                        latest_epoch = ep
                        latest_ckpt = ckpt_candidate

            if latest_ckpt is not None:
                args.resume = latest_ckpt
                if accelerator.is_main_process:
                    logger.info(
                        f"🔄 Auto-resuming from latest checkpoint: {latest_ckpt} "
                        f"(epoch {latest_epoch})"
                    )

    if training_already_complete:
        return

    # Track whether we need to clean up a consumed interrupted checkpoint.
    # Deferred until the first new checkpoint is written so we don't lose
    # our only recovery point if the resumed process crashes during setup.
    _cleanup_interrupted_pending = (
        args.resume == interrupted_ckpt_path and os.path.exists(interrupted_ckpt_path)
    )

    if args.resume:
        args.resume = str(resolve_checkpoint_path(args.resume, purpose="resume"))
        if (Path(args.resume) / "COMPLETE.json").is_file():
            source_root = Path(args.resume).parent.parent.resolve()
            source_manifest = _read_manifest(source_root / "manifest.json")
            source_layout = RunLayout(source_root, source_manifest.run_id)
            restored_generation = next(
                (
                    item
                    for item in published_generations(source_layout)
                    if item.path == Path(args.resume).resolve()
                ),
                None,
            )
            if restored_generation is None:
                raise ValueError("Resume generation is not a validated publication")

    # Capture checkpoint normalization before prepare_data can invalidate a cache.
    # Legacy periodic/emergency checkpoints kept their scaler in the run folder.
    resume_scaler = None
    if args.resume:
        if not os.path.isdir(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        if _validate_emergency_resume(args.resume, args.scheduler, args.epochs):
            logger.warning(
                "Resuming an emergency checkpoint replays the interrupted epoch "
                "from mid-epoch model/optimizer state. This recovery is not exact; "
                "partial gradients, RNG state, and AMP scaler state are not restored. "
                "Use an epoch-boundary checkpoint for reproducible continuation."
            )
        scaler_candidates = [
            os.path.join(args.resume, "scaler.pkl"),
            os.path.join(os.path.dirname(os.path.abspath(args.resume)), "scaler.pkl"),
        ]
        for scaler_path in scaler_candidates:
            if os.path.isfile(scaler_path):
                with open(scaler_path, "rb") as f:
                    resume_scaler = pickle.load(f)
                break
        if resume_scaler is None:
            logger.warning(
                "Checkpoint scaler not found; normalization compatibility cannot "
                "be verified for this legacy checkpoint."
            )

    # ==========================================================================
    # DATA & MODEL LOADING
    # ==========================================================================
    train_dl, val_dl, scaler, in_shape, out_dim = prepare_data(
        args,
        logger,
        accelerator,
        cache_dir=args.output_dir,
        expected_scaler=resume_scaler,
    )

    # Build model using registry
    model, target_shape = _build_model_for_targets(
        args.model,
        in_shape,
        out_dim,
        scaler,
        pretrained=args.pretrained,
    )

    if accelerator.is_main_process:
        param_info = model.parameter_summary()
        logger.info(
            f"   Model Parameters: {param_info['trainable_parameters']:,} trainable"
        )
        logger.info(f"   Model Size: {param_info['total_mb']:.2f} MB")

    # Optional WandB model watching (opt-in due to overhead on large models)
    if (
        args.wandb
        and args.wandb_watch
        and WANDB_AVAILABLE
        and accelerator.is_main_process
    ):
        wandb.watch(model, log="gradients", log_freq=100)
        logger.info("   📊 WandB gradient watching enabled")

    # Torch 2.0 compilation (requires compatible Triton on GPU)
    if args.compile:
        try:
            # Test if Triton is available - just import the package
            # Different Triton versions have different internal APIs, so just check base import
            import triton

            model = torch.compile(model)
            if accelerator.is_main_process:
                logger.info("   ✔ torch.compile enabled (Triton backend)")
        except ImportError as e:
            if accelerator.is_main_process:
                if "triton" in str(e).lower():
                    logger.warning(
                        "   ⚠ Triton not installed or incompatible version - torch.compile disabled. "
                        "Training will proceed without compilation."
                    )
                else:
                    logger.warning(
                        f"   ⚠ torch.compile setup failed: {e}. Continuing without compilation."
                    )
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning(
                    f"   ⚠ torch.compile failed: {e}. Continuing without compilation."
                )

    # ==========================================================================
    # OPTIMIZER, SCHEDULER & LOSS CONFIGURATION
    # ==========================================================================
    # Parse comma-separated arguments with validation
    try:
        betas_list = [float(x.strip()) for x in args.betas.split(",")]
        if len(betas_list) != 2:
            raise ValueError(
                f"--betas must have exactly 2 values, got {len(betas_list)}"
            )
        if not all(0.0 <= b < 1.0 for b in betas_list):
            raise ValueError(f"--betas values must be in [0, 1), got {betas_list}")
        betas = tuple(betas_list)
    except ValueError as e:
        raise ValueError(
            f"Invalid --betas format '{args.betas}': {e}. Expected format: '0.9,0.999'"
        )

    loss_weights = None
    if args.loss_weights:
        loss_weights = [float(x.strip()) for x in args.loss_weights.split(",")]
    milestones = None
    if args.milestones:
        milestones = [int(x.strip()) for x in args.milestones.split(",")]

    # Create optimizer using factory
    optimizer = get_optimizer(
        name=args.optimizer,
        params=model.get_optimizer_groups(args.lr, args.weight_decay),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.nesterov,
        betas=betas,
    )

    # Create loss function using factory
    criterion = get_loss(
        name=args.loss,
        weights=loss_weights,
        delta=args.huber_delta,
    )
    # Move criterion to device (important for WeightedMSELoss buffer)
    criterion = criterion.to(accelerator.device)

    # ==========================================================================
    # PHYSICAL CONSTRAINTS INTEGRATION
    # ==========================================================================
    from wavedl.utils.constraints import (
        PhysicsConstrainedLoss,
        build_constraints,
    )

    # Build soft constraints
    soft_constraints = build_constraints(
        expressions=args.constraint,
        file_path=args.constraint_file,
        reduction=args.constraint_reduction,
    )

    # Wrap criterion with PhysicsConstrainedLoss if we have soft constraints
    if soft_constraints:
        # Pass output scaler so constraints can be evaluated in physical space
        output_mean = scaler.mean_ if hasattr(scaler, "mean_") else None
        output_std = scaler.scale_ if hasattr(scaler, "scale_") else None
        criterion = PhysicsConstrainedLoss(
            criterion,
            soft_constraints,
            weights=args.constraint_weight,
            output_mean=output_mean,
            output_std=output_std,
        )
        if accelerator.is_main_process:
            logger.info(
                f"   🔬 Physical constraints: {len(soft_constraints)} constraint(s) "
                f"with weight(s) {args.constraint_weight}"
            )
            if output_mean is not None:
                logger.info(
                    "   📐 Constraints evaluated in physical space (denormalized)"
                )

    validation_criterion = (
        torch.nn.MSELoss()
        if args.selection_metric == "mse"
        else criterion.base_loss
        if isinstance(criterion, PhysicsConstrainedLoss)
        else criterion
    )

    # Track if scheduler should step per batch (OneCycleLR) or per epoch
    scheduler_step_per_batch = not is_epoch_based(args.scheduler)

    # ==========================================================================
    # DDP Preparation Strategy:
    # - For batch-based schedulers (OneCycleLR): prepare DataLoaders first to get
    #   the correct sharded batch count, then create scheduler
    # - For epoch-based schedulers: create scheduler before prepare (no issue)
    # ==========================================================================
    if scheduler_step_per_batch:
        # BATCH-BASED SCHEDULER (e.g., OneCycleLR)
        # Prepare model, optimizer, dataloaders FIRST to get sharded loader length
        model, optimizer, train_dl, val_dl = accelerator.prepare(
            model, optimizer, train_dl, val_dl
        )

        # Count optimizer updates, including Accelerate's final partial flush.
        # The prepared loader is already sharded, so no world-size multiplier.
        steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum_steps)
        scheduler = get_scheduler(
            name=args.scheduler,
            optimizer=optimizer,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            min_lr=args.min_lr,
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            gamma=args.scheduler_factor,  # For Step/MultiStep/Exponential schedulers
            step_size=args.step_size,
            milestones=milestones,
            warmup_epochs=args.warmup_epochs,
        )
        # Step explicitly once per successful optimizer update. Accelerate's
        # scheduler wrapper multiplies steps by world size for non-split loaders,
        # which would double-count our already-sharded update budget.
        # Keep the prepared wrapper so existing scheduler checkpoint filenames
        # remain compatible; step_scheduler_with_optimizer=False disables its
        # implicit adjustment while our training loop handles skipped updates.
        scheduler = accelerator.prepare(scheduler)
        scheduler.step_with_optimizer = False
    else:
        # EPOCH-BASED SCHEDULER (plateau, cosine, step, etc.)
        # Create scheduler before prepare (no batch count dependency)
        scheduler = get_scheduler(
            name=args.scheduler,
            optimizer=optimizer,
            epochs=args.epochs,
            steps_per_epoch=None,
            min_lr=args.min_lr,
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            gamma=args.scheduler_factor,  # For Step/MultiStep/Exponential schedulers
            step_size=args.step_size,
            milestones=milestones,
            warmup_epochs=args.warmup_epochs,
        )

        # DON'T include scheduler in accelerator.prepare() for any epoch-based
        # scheduler. We handle stepping on rank 0 only, then broadcast LR.
        # This prevents Accelerate's wrapper from interfering with step counts.
        model, optimizer, train_dl, val_dl = accelerator.prepare(
            model, optimizer, train_dl, val_dl
        )
        # Register scheduler for checkpointing so state is saved/loaded on resume
        accelerator.register_for_checkpointing(scheduler)

    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"🔄 Loading checkpoint from: {args.resume}")

            meta_path = os.path.join(args.resume, "training_meta.pkl")
            resume_meta = None
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    resume_meta = pickle.load(f)
                _validate_resume_metadata(
                    resume_meta,
                    args.model,
                    in_shape,
                    out_dim,
                    target_shape,
                )

            # Detect checkpoint format: standalone (from KeyboardInterrupt)
            # vs accelerator-managed (from normal save_state)
            standalone_weights = os.path.join(args.resume, "model_weights.pth")
            if os.path.exists(standalone_weights):
                # Standalone format: model_weights.pth + optimizer.pt
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.load_state_dict(
                    torch.load(
                        standalone_weights, map_location="cpu", weights_only=True
                    )
                )
                standalone_optim = os.path.join(args.resume, "optimizer.pt")
                if os.path.exists(standalone_optim):
                    optimizer.load_state_dict(
                        torch.load(
                            standalone_optim, map_location="cpu", weights_only=True
                        )
                    )
                # Restore scheduler state (prevents LR schedule restart on resume)
                standalone_scheduler = os.path.join(args.resume, "scheduler.pt")
                if os.path.exists(standalone_scheduler):
                    scheduler.load_state_dict(
                        torch.load(
                            standalone_scheduler,
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
                    logger.info(
                        "   Loaded standalone checkpoint (emergency format, scheduler restored)"
                    )
                else:
                    logger.warning(
                        "   ⚠️ Loaded standalone checkpoint WITHOUT scheduler state — "
                        "LR schedule will restart from scratch"
                    )
            else:
                accelerator.load_state(args.resume)

            # Restore training metadata
            if resume_meta is not None:
                start_epoch = resume_meta.get("epoch", 0)
                best_val_loss = resume_meta.get("best_val_loss", float("inf"))
                patience_ctr = resume_meta.get("patience_ctr", 0)
                if (
                    restored_generation is not None
                    and start_epoch != restored_generation.step
                ):
                    raise ValueError("Generation step differs from training metadata")
                logger.info(
                    f"   ✅ Restored: Epoch {start_epoch}, Best Loss: {best_val_loss:.6f}"
                )
            else:
                if restored_generation is not None:
                    raise ValueError(
                        "Published resume generation requires training_meta.pkl"
                    )
                logger.warning(
                    "   ⚠️ training_meta.pkl not found, starting from epoch 0"
                )

            # Restore history, truncating to start_epoch to avoid duplicate
            # rows for epochs that will be re-run from the best checkpoint.
            history_path = os.path.join(
                args.resume if restored_generation is not None else args.output_dir,
                "training_history.csv",
            )
            if os.path.exists(history_path):
                full_history = pd.read_csv(history_path).to_dict("records")
                history = [
                    row for row in full_history if row.get("epoch", 0) <= start_epoch
                ]
                if len(full_history) > start_epoch:
                    logger.info(
                        f"   ✅ Loaded {len(history)} of {len(full_history)} history "
                        f"epochs (truncated to checkpoint epoch {start_epoch})"
                    )
                else:
                    logger.info(f"   ✅ Loaded {len(history)} epochs from history")
            if restored_generation is not None:
                if "training_history.csv" not in restored_generation.files:
                    raise ValueError(
                        "Published resume generation requires training history"
                    )
                if source_layout.run_id == journal.bootstrap.layout.run_id:
                    selection = read_selection_metadata(restored_generation)
                    source_seq = (
                        selection.get("source_seq") if selection is not None else None
                    )
                    if source_seq is None:
                        journal.log(
                            "Legacy generation has no causal source sequence; "
                            "restored metric lineage may be ambiguous",
                            stream="system",
                            level="warning",
                        )
                    journal.emit(
                        "supersede",
                        SupersedePayload(
                            restored_generation.attempt,
                            start_epoch,
                            restored_generation.generation,
                            source_seq=source_seq,
                        ),
                    )
                else:
                    journal.log(
                        f"Resumed foreign run {source_layout.run_id}, "
                        f"generation {restored_generation.generation}, "
                        f"attempt {restored_generation.attempt}, step {start_epoch}",
                        stream="system",
                    )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    # ==========================================================================
    # PHYSICAL METRIC SETUP
    # ==========================================================================
    # Physical MAE = normalized MAE * scaler.scale_
    phys_scale = torch.tensor(
        scaler.scale_, device=accelerator.device, dtype=torch.float32
    )
    is_field_output = len(target_shape) > 1

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    # Dynamic console header
    if accelerator.is_main_process:
        base_cols = ["Ep", "TrnLoss", "ValLoss", "R2", "PCC", "GradN", "LR", "MAE_Avg"]
        param_cols = [f"MAE_P{i}" for i in range(out_dim)]
        header = "{:<4} | {:<8} | {:<8} | {:<6} | {:<6} | {:<6} | {:<8} | {:<8}".format(
            *base_cols
        )
        header += " | " + " | ".join([f"{c:<8}" for c in param_cols])
        logger.info("=" * len(header))
        logger.info(header)
        logger.info("=" * len(header))

    # Initialize epoch before try block so the interrupt handler always has
    # a valid value, even if KeyboardInterrupt fires before the loop starts.
    epoch = start_epoch
    completed_epochs = start_epoch

    try:
        if training_already_complete:
            return

        total_training_time = 0.0

        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()

            # ==================== TRAINING PHASE ====================
            model.train()
            # Use GPU tensor for loss accumulation to avoid .item() sync per batch
            train_loss_sum = torch.tensor(0.0, device=accelerator.device)
            train_samples = 0
            accumulated_samples = 0
            window_nonfinite = torch.zeros(
                (), dtype=torch.bool, device=accelerator.device
            )
            grad_norm_tracker = MetricTracker()

            pbar = tqdm(
                train_dl,
                disable=not accelerator.is_main_process,
                leave=False,
                desc=f"Epoch {epoch + 1}",
            )

            for x, y in pbar:
                with accelerator.accumulate(model):
                    # Use mixed precision for forward pass (respects --precision flag)
                    with accelerator.autocast():
                        pred = model(x)
                        if pred.shape != y.shape:
                            raise ValueError(
                                f"Model output shape {tuple(pred.shape)} does not match "
                                f"target shape {tuple(y.shape)}."
                            )
                        # Pass inputs for input-dependent constraints (x_mean, x[...], etc.)
                        if isinstance(criterion, PhysicsConstrainedLoss):
                            loss = criterion(pred, y, x)
                        else:
                            loss = criterion(pred, y)

                    window_nonfinite = _flag_nonfinite_loss(loss, window_nonfinite)
                    accumulated_samples += x.size(0)
                    # Weight by batch size relative to the nominal size so a
                    # partial final batch contributes per sample, not per batch.
                    accelerator.backward(loss * (x.size(0) / args.batch_size))

                    if accelerator.sync_gradients:
                        _finalize_accumulation_window(
                            model,
                            accumulated_samples,
                            window_nonfinite,
                            args.batch_size,
                            accelerator,
                        )
                        accumulated_samples = 0
                        window_nonfinite = torch.zeros_like(window_nonfinite)
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.grad_clip if args.grad_clip > 0 else float("inf"),
                        )
                        if grad_norm is not None:
                            grad_norm_tracker.update(grad_norm.item())

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

                    # Per-update LR scheduling (e.g., OneCycleLR). An AMP
                    # overflow skips the optimizer update and must not spend a
                    # scheduler step either.
                    if (
                        scheduler_step_per_batch
                        and accelerator.sync_gradients
                        and not accelerator.optimizer_step_was_skipped
                    ):
                        scheduler.step()

                    # Accumulate as tensors to avoid .item() sync per batch
                    train_loss_sum += loss.detach() * x.size(0)
                    train_samples += x.size(0)

            # Single .item() call at end of epoch (reduces GPU sync overhead)
            train_loss_scalar = train_loss_sum.item()

            # Synchronize training metrics across GPUs
            global_loss = accelerator.reduce(
                torch.tensor([train_loss_scalar], device=accelerator.device),
                reduction="sum",
            ).item()
            global_samples = accelerator.reduce(
                torch.tensor([train_samples], device=accelerator.device),
                reduction="sum",
            ).item()
            avg_train_loss = global_loss / global_samples

            # ==================== VALIDATION PHASE ====================
            model.eval()
            # Accumulate predictions locally ON CPU to prevent GPU OOM
            local_preds = []
            local_targets = []
            field_metrics = (
                StreamingFieldMetrics(out_dim, np.asarray(scaler.scale_))
                if is_field_output and accelerator.is_main_process
                else None
            )
            field_loss_sum = 0.0
            field_samples = 0
            field_preview_limit = 4
            # DataLoader lengths match across ranks, so every rank fails
            # together here rather than deadlocking in a later collective.
            if len(val_dl) == 0:
                raise RuntimeError("Validation loader produced no samples.")

            # Collectives write their result tensors from backend worker threads.
            # Normal no-grad tensors permit those writes (inference tensors do not).
            # Every rank must join each gather; only rank 0 scores the gathered
            # data, and the scalar metric is broadcast below.
            is_main = accelerator.is_main_process
            with torch.no_grad():
                for x, y in val_dl:
                    # Use mixed precision for validation (consistent with training)
                    with accelerator.autocast():
                        pred = model(x)
                        if pred.shape != y.shape:
                            raise ValueError(
                                f"Model output shape {tuple(pred.shape)} does not match "
                                f"target shape {tuple(y.shape)}."
                            )

                    # Gather one batch at a time so Accelerate removes only the
                    # duplicated samples from DDP's final batch. Gathering a
                    # concatenated epoch makes its remainder logic trim valid
                    # samples from every earlier batch.
                    gathered_pred_batch, gathered_target_batch = (
                        accelerator.gather_for_metrics((pred.detach(), y.detach()))
                    )
                    if not is_main:
                        continue

                    if is_field_output:
                        # Dense maps are scored as they stream; only a few
                        # preview maps are retained, keeping memory bounded.
                        batch_samples = len(gathered_pred_batch)
                        field_loss_sum += (
                            validation_criterion(
                                gathered_pred_batch, gathered_target_batch
                            ).item()
                            * batch_samples
                        )
                        field_samples += batch_samples
                        pred_numpy = gathered_pred_batch.float().cpu().numpy()
                        target_numpy = gathered_target_batch.float().cpu().numpy()
                        field_metrics.update(target_numpy, pred_numpy)
                        preview_remaining = field_preview_limit - sum(
                            len(batch) for batch in local_preds
                        )
                        if preview_remaining > 0:
                            local_preds.append(pred_numpy[:preview_remaining])
                            local_targets.append(target_numpy[:preview_remaining])
                    else:
                        local_preds.append(gathered_pred_batch.float().cpu())
                        local_targets.append(gathered_target_batch.float().cpu())

            avg_val_loss = float("nan")
            if is_main and is_field_output:
                avg_val_loss = float(field_loss_sum / field_samples)
                r2, pcc, avg_mae_per_param = field_metrics.compute()
                avg_mae = float(avg_mae_per_param.mean())
                y_pred = np.concatenate(local_preds, axis=0)
                y_true = np.concatenate(local_targets, axis=0)
            elif is_main:
                # Vector targets are small enough to retain for diagnostics.
                # These batches are already globally gathered and de-padded.
                gathered_preds = torch.cat(local_preds)
                gathered_targets = torch.cat(local_targets)
                loss_sum = 0.0
                sample_count = len(gathered_preds)
                chunk_size = 2048
                for chunk_start in range(0, sample_count, chunk_size):
                    pred_chunk = gathered_preds[
                        chunk_start : chunk_start + chunk_size
                    ].to(accelerator.device)
                    target_chunk = gathered_targets[
                        chunk_start : chunk_start + chunk_size
                    ].to(accelerator.device)
                    loss_sum += validation_criterion(
                        pred_chunk, target_chunk
                    ).item() * len(pred_chunk)
                avg_val_loss = float(loss_sum / sample_count)
                avg_mae_per_param = (
                    torch.abs((gathered_preds - gathered_targets) * phys_scale.cpu())
                    .mean(dim=0)
                    .float()
                    .numpy()
                )
                avg_mae = avg_mae_per_param.mean()

            # One scalar broadcast keeps plateau schedulers and the finite
            # check identical on every rank (no rank-local recomputation).
            avg_val_loss = broadcast_value(avg_val_loss, accelerator)
            if not math.isfinite(avg_val_loss):
                raise FloatingPointError(
                    "Validation loss is non-finite; training aborted."
                )

            # ==================== LOGGING & CHECKPOINTING ====================
            # Compute the validation transition once on rank zero.  Keep the
            # collective below in its established position (after scheduling),
            # while allowing the structured metric to describe the state this
            # epoch will commit.
            is_best_candidate = False
            if accelerator.is_main_process:
                is_best_candidate = avg_val_loss < best_val_loss
                emitted_best_val_loss = (
                    avg_val_loss if is_best_candidate else best_val_loss
                )
                emitted_patience_ctr = 0 if is_best_candidate else patience_ctr + 1
                if not is_field_output:
                    # Scientific vector metrics; gathered tensors have already
                    # had DDP padding removed by Accelerate.
                    y_pred = gathered_preds.float().numpy()
                    y_true = gathered_targets.float().numpy()
                    metric_pred = flatten_targets_by_channel(y_pred)
                    metric_true = flatten_targets_by_channel(y_true)
                    if len(metric_true) >= 2:
                        r2 = r2_score(metric_true, metric_pred)
                        pcc = calc_pearson(metric_true, metric_pred)
                    else:
                        r2 = float("nan")
                        pcc = float("nan")
                current_lr = get_lr(optimizer)

                # Update history
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                total_training_time += epoch_time

                epoch_stats = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_r2": r2,
                    "val_pearson": pcc,
                    "val_mae_avg": avg_mae,
                    "grad_norm": grad_norm_tracker.avg,
                    "lr": current_lr,
                    "epoch_time": round(epoch_time, 2),
                    "total_time": round(total_training_time, 2),
                }
                for i, mae in enumerate(avg_mae_per_param):
                    epoch_stats[f"MAE_Phys_P{i}"] = mae

                history.append(epoch_stats)

                # Console display
                base_str = f"{epoch + 1:<4} | {avg_train_loss:<8.4f} | {avg_val_loss:<8.4f} | {r2:<6.4f} | {pcc:<6.4f} | {grad_norm_tracker.avg:<6.4f} | {current_lr:<8.2e} | {avg_mae:<8.4f}"
                param_str = " | ".join([f"{m:<8.4f}" for m in avg_mae_per_param])
                logger.info(f"{base_str} | {param_str}")

                # Structured metrics for GUI consumption
                # This line is parsed by the GUI's OutputParser and filtered
                # from the visible log output.
                gui_metrics = {
                    "epoch": epoch + 1,
                    "total_epochs": args.epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "best_val_loss": emitted_best_val_loss,
                    "r2": r2,
                    "pearson": pcc,
                    "grad_norm": grad_norm_tracker.avg,
                    "lr": current_lr,
                    "mae_avg": float(avg_mae),
                    "mae_per_param": [float(m) for m in avg_mae_per_param],
                    "epoch_time": round(epoch_time, 2),
                    "total_time": round(total_training_time, 2),
                    "patience_counter": emitted_patience_ctr,
                    "max_patience": args.patience,
                }
                # allow_nan=False turns any non-finite value that escapes the
                # sanitiser into a loud failure rather than silent corruption.
                print(
                    f"##METRICS##{json.dumps(_finite_or_none(gui_metrics), allow_nan=False)}",
                    flush=True,
                )
                journal.emit(
                    "metric",
                    MetricPayload(
                        "epoch",
                        avg_val_loss,
                        epoch + 1,
                        metrics=_finite_or_none(gui_metrics),
                    ),
                )

                # WandB logging
                if args.wandb and WANDB_AVAILABLE:
                    log_dict = {
                        "main/train_loss": avg_train_loss,
                        "main/val_loss": avg_val_loss,
                        "metrics/r2_score": r2,
                        "metrics/pearson_corr": pcc,
                        "metrics/mae_avg": avg_mae,
                        "system/grad_norm": grad_norm_tracker.avg,
                        "hyper/lr": current_lr,
                    }
                    for i, mae in enumerate(avg_mae_per_param):
                        log_dict[f"mae_detailed/P{i}"] = mae

                    # Periodic scatter plots (epoch+1 to match the displayed/logged
                    # 1-indexed epoch numbering used everywhere else in the loop)
                    if ((epoch + 1) % WANDB_SCATTER_INTERVAL == 0) or (
                        is_best_candidate
                    ):
                        real_true = scaler.inverse_transform(y_true)
                        real_pred = scaler.inverse_transform(y_pred)
                        real_true = flatten_targets_by_channel(real_true)
                        real_pred = flatten_targets_by_channel(real_pred)
                        fig = plot_scientific_scatter(real_true, real_pred)
                        log_dict["plots/scatter_analysis"] = wandb.Image(fig)
                        plt.close(fig)

                    accelerator.log(log_dict)

            # Learning rate scheduling (epoch-based schedulers only)
            # NOTE: All epoch-based schedulers must step only on main process
            # in DDP mode. Otherwise, each GPU process calls scheduler.step(),
            # consuming T_max N× faster (e.g., 4× with 4 GPUs).
            # After stepping on rank 0, we broadcast the updated LR to all.
            if not scheduler_step_per_batch:
                if accelerator.is_main_process:
                    if args.scheduler == "plateau":
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                # Sync LR across all processes after main process updates it
                wait_for_workers(accelerator)

                # Broadcast per-group LRs from rank 0 to all processes
                # (preserves multi-group ratios, e.g., Swin backbone 0.1× vs head 1×)
                if dist.is_initialized():
                    n_groups = len(optimizer.param_groups)
                    lr_tensor = torch.zeros(
                        n_groups, device=accelerator.device, dtype=torch.float32
                    )
                    if accelerator.is_main_process:
                        for i, pg in enumerate(optimizer.param_groups):
                            lr_tensor[i] = pg["lr"]
                    dist.broadcast(lr_tensor, src=0)
                    if not accelerator.is_main_process:
                        for i, pg in enumerate(optimizer.param_groups):
                            pg["lr"] = lr_tensor[i].item()

            completed_epochs = epoch + 1

            # Normal checkpoints remain collective; publication is rank-zero-only.
            is_best_epoch = broadcast_early_stop(
                is_best_candidate if accelerator.is_main_process else False,
                accelerator,
            )
            if accelerator.is_main_process:
                if is_best_epoch:
                    best_val_loss = avg_val_loss
                    patience_ctr = 0
                else:
                    patience_ctr += 1

            metadata = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "patience_ctr": patience_ctr,
                "model_name": args.model,
                "in_shape": in_shape,
                "out_dim": out_dim,
                "target_shape": target_shape,
            }
            kinds = []
            if is_best_epoch:
                kinds.append(("best", "best_checkpoint"))
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                kinds.append(("periodic", f"epoch_{epoch + 1}_checkpoint"))
            for kind, checkpoint_name in kinds:
                with suppress_accelerate_logging():
                    published = save_normal_checkpoint(
                        accelerator,
                        Path(args.output_dir) / checkpoint_name,
                        journal,
                        metadata=metadata,
                        history=history,
                        scaler_source=Path(args.output_dir) / "scaler.pkl",
                        kind=kind,
                    )
                if accelerator.is_main_process:
                    if kind == "best":
                        torch.save(
                            unwrap_checkpoint_model(accelerator, model).state_dict(),
                            os.path.join(args.output_dir, "best_model_weights.pth"),
                        )
                    pd.DataFrame(history).to_csv(
                        os.path.join(args.output_dir, "training_history.csv"),
                        index=False,
                    )
                    logger.info(
                        f"   💾 Published {kind} checkpoint: {published.generation}"
                    )
                    if _cleanup_interrupted_pending:
                        # Only a durable new publication permits recovery-source cleanup.
                        shutil.rmtree(interrupted_ckpt_path)
                        _cleanup_interrupted_pending = False

            # DDP-safe early stopping
            should_stop = (
                patience_ctr >= args.patience if accelerator.is_main_process else False
            )
            if broadcast_early_stop(should_stop, accelerator):
                if accelerator.is_main_process:
                    logger.info(
                        f"🛑 Early stopping at epoch {epoch + 1} (patience={args.patience})"
                    )
                    # Create completion flag to prevent auto-resume
                    with open(
                        os.path.join(args.output_dir, "training_complete.flag"), "w"
                    ) as f:
                        f.write(
                            f"Training completed via early stopping at epoch {epoch + 1}\n"
                        )
                break

    except KeyboardInterrupt:
        # Emergency publication deliberately has no distributed participation.
        if accelerator.is_main_process:
            logger.warning("Training interrupted. Saving emergency checkpoint...")
            save_emergency_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                interrupted_ckpt_path,
                journal,
                metadata={
                    "epoch": completed_epochs,
                    "best_val_loss": best_val_loss,
                    "patience_ctr": patience_ctr,
                    "model_name": args.model,
                    "in_shape": in_shape,
                    "out_dim": out_dim,
                    "target_shape": target_shape,
                },
                history=history,
                scaler_source=Path(args.output_dir) / "scaler.pkl",
            )
            logger.info(
                f"   💾 Emergency checkpoint published at step {completed_epochs}"
            )
        raise

    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        raise

    else:
        # Training completed normally (reached max epochs without early stopping)
        # Create completion flag to prevent auto-resume on re-run
        if accelerator.is_main_process:
            if not os.path.exists(complete_flag_path):
                with open(complete_flag_path, "w") as f:
                    f.write(f"Training completed normally after {args.epochs} epochs\n")
                logger.info(f"✅ Training completed after {args.epochs} epochs")

    finally:
        # Final CSV write to capture all epochs (handles non-multiple-of-10 endings)
        if accelerator.is_main_process and len(history) > 0:
            pd.DataFrame(history).to_csv(
                os.path.join(args.output_dir, "training_history.csv"),
                index=False,
            )

        # Generate training curves plot (PNG + SVG)
        if accelerator.is_main_process and len(history) > 0:
            try:
                fig = create_training_curves(history, show_lr=True)
                for fmt in ["png", "svg"]:
                    fig.savefig(
                        os.path.join(args.output_dir, f"training_curves.{fmt}"),
                        dpi=FIGURE_DPI,
                        bbox_inches="tight",
                    )
                plt.close(fig)
                logger.info("✔ Saved: training_curves.png, training_curves.svg")
            except Exception as e:
                logger.warning(f"Could not generate training curves: {e}")

        logger.info("Training completed.")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
