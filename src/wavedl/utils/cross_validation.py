"""
WaveDL - Cross-Validation Utilities
====================================

Internal module for K-fold cross-validation. Called by train.py when --cv flag is used.

This module provides:
    - SimpleDataset: In-memory dataset for CV
    - train_fold: Single fold training function
    - run_cross_validation: Main CV orchestration

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import gc
import json
import logging
import math
import os
import pickle
import random
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import wraps
from inspect import signature
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

from .data import _as_finite_float32
from .targets import (
    canonicalize_targets,
    create_target_scaler,
    flatten_targets_by_channel,
)


# ==============================================================================
# SIMPLE DATASET
# ==============================================================================
class CVDataset(torch.utils.data.Dataset):
    """Simple in-memory dataset for cross-validation."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, expected_spatial_ndim: int | None = None
    ):
        """
        Initialize CV dataset with explicit channel dimension handling.

        Args:
            X: Input data with shape (N, *spatial_dims) or (N, C, *spatial_dims)
            y: Vector targets ``(N,T)`` or dense fields ``(N,C,H,W)``
            expected_spatial_ndim: Expected number of spatial dimensions (1, 2, or 3).
                If provided, uses explicit logic instead of heuristics.
                If None, falls back to ndim-based inference (legacy behavior).

        Channel Dimension Logic:
            - If X.ndim == expected_spatial_ndim + 1: Add channel dim (N, *spatial) -> (N, 1, *spatial)
            - If X.ndim == expected_spatial_ndim + 2: Already has channel (N, C, *spatial)
            - If expected_spatial_ndim is None: Use legacy ndim-based inference

        Warning:
            Legacy mode (expected_spatial_ndim=None) may misinterpret multichannel
            3D data as single-channel 4D data. Always pass expected_spatial_ndim
            explicitly for 3D volumes with >1 channel.
        """
        if expected_spatial_ndim is not None:
            # Explicit mode: use expected_spatial_ndim to determine if channel exists
            if X.ndim == expected_spatial_ndim + 1:
                # Shape is (N, *spatial) - needs channel dimension
                X = np.expand_dims(X, axis=1)
            elif X.ndim == expected_spatial_ndim + 2:
                # Shape is (N, C, *spatial) - already has channel
                pass
            else:
                raise ValueError(
                    f"Input shape {X.shape} incompatible with expected_spatial_ndim={expected_spatial_ndim}. "
                    f"Expected ndim={expected_spatial_ndim + 1} or {expected_spatial_ndim + 2}, got {X.ndim}."
                )
        else:
            # Legacy mode: infer from ndim (for backwards compatibility)
            # Assumes single-channel data without explicit channel dimension
            if X.ndim == 2:  # 1D signals: (N, L) -> (N, 1, L)
                X = X[:, np.newaxis, :]
            elif X.ndim == 3:  # 2D images: (N, H, W) -> (N, 1, H, W)
                X = X[:, np.newaxis, :, :]
            elif X.ndim == 4:  # 3D volumes: (N, D, H, W) -> (N, 1, D, H, W)
                X = X[:, np.newaxis, :, :, :]
            # ndim >= 5 assumed to already have channel dimension

        if len(X) != len(y) or len(X) == 0:
            raise ValueError(
                "CV inputs and targets must have equal nonzero sample counts."
            )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("CV inputs and targets must be finite.")

        # Use from_numpy to share memory (zero-copy) instead of torch.tensor
        # which creates a full copy. Requires contiguous C-order arrays.
        X = np.ascontiguousarray(_as_finite_float32(X, "CV inputs"))
        y = np.ascontiguousarray(_as_finite_float32(y, "CV targets"))
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _take_samples(values: Any, indices: np.ndarray) -> np.ndarray:
    """Index array-like data while satisfying h5py's sorted-index contract."""

    indices = np.asarray(indices)
    if len(indices) < 2 or np.all(indices[:-1] <= indices[1:]):
        return np.asarray(values[indices])
    order = np.argsort(indices)
    sorted_values = np.asarray(values[indices[order]])
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    return sorted_values[inverse_order]


# ==============================================================================
# SINGLE FOLD TRAINING
# ==============================================================================
def _validate_training_options(
    device: torch.device, grad_accum_steps: int, precision: str
) -> None:
    """Reject unsupported execution modes before allocating fold models."""
    if (
        isinstance(grad_accum_steps, bool)
        or not isinstance(grad_accum_steps, int)
        or grad_accum_steps < 1
    ):
        raise ValueError("grad_accum_steps must be an integer >= 1")
    if precision not in {"no", "bf16", "fp16"}:
        raise ValueError(f"Unknown CV precision: {precision!r}")
    if precision == "no":
        return
    # Accelerate disables fp16 autocast on CPU; make that explicit for direct
    # callers instead of silently training in full precision.
    if device.type not in {"cpu", "cuda", "mps"} or (
        device.type == "cpu" and precision == "fp16"
    ):
        raise ValueError(f"CV precision={precision!r} is unsupported on {device.type}")
    if not torch.amp.autocast_mode.is_autocast_available(device.type):
        raise ValueError(f"Autocast is unavailable on {device.type}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA mixed precision requested but CUDA is unavailable")
        with torch.cuda.device(device):
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("CV bf16 precision is unsupported on this CUDA device")


def train_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    grad_clip: float,
    scaler: Any,
    logger: logging.Logger,
    *,
    grad_accum_steps: int = 1,
    precision: str = "no",
) -> dict[str, Any]:
    """
    Train and evaluate a single CV fold.

    Optimizer updates follow standard training: each update uses the
    sample-mean gradient of its accumulation window (including a partial final
    window), gradients are clipped only at update boundaries, and OneCycleLR
    advances only on updates not skipped by fp16 overflow. A
    ``PhysicsConstrainedLoss`` is used for training while its ``base_loss``
    alone drives validation and best-epoch selection.

    Args:
        fold: Fold index (0-based)
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Torch device
        epochs: Max epochs
        patience: Early stopping patience
        scaler: Target scaler (for physical units)
        logger: Logger instance
        grad_accum_steps: Micro-batches per optimizer update
        precision: ``"no"``, ``"bf16"`` (CPU/CUDA/MPS) or ``"fp16"`` (CUDA/MPS)

    Returns:
        Dictionary with fold results and metrics
    """
    device = torch.device(device)
    _validate_training_options(device, grad_accum_steps, precision)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    grad_scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None

    from wavedl.utils.constraints import PhysicsConstrainedLoss

    constrained = isinstance(criterion, PhysicsConstrainedLoss)
    validation_criterion = criterion.base_loss if constrained else criterion

    def autocast_context():
        if precision == "no":
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    # Same weighting as standard training: micro-batch losses are weighted by
    # batch / nominal size, and windows with fewer samples than nominal are
    # rescaled to the true sample mean at the update boundary.
    nominal_batch_size = train_loader.batch_size or 1
    window_capacity = grad_accum_steps * nominal_batch_size

    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None
    best_epoch = 0
    history = []

    # Determine if scheduler steps per batch (OneCycleLR) or per epoch
    # Use isinstance check since class name 'OneCycleLR' != 'onecycle' string in is_epoch_based
    from torch.optim.lr_scheduler import OneCycleLR

    step_per_batch = isinstance(scheduler, OneCycleLR)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        window_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with autocast_context():
                pred = model(x)
                if pred.shape != y.shape:
                    raise ValueError(
                        f"Model output shape {tuple(pred.shape)} does not match "
                        f"target shape {tuple(y.shape)}."
                    )
                # Input-dependent constraints (x_mean, x[...]) need the inputs
                loss = criterion(pred, y, x) if constrained else criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Cross-validation training loss is non-finite."
                )
            backward_loss = loss * (x.size(0) / nominal_batch_size) / grad_accum_steps
            if grad_scaler is None:
                backward_loss.backward()
            else:
                grad_scaler.scale(backward_loss).backward()
            window_samples += x.size(0)

            # A partial final window still produces an update (as in Accelerate)
            last_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % grad_accum_steps == 0 or last_batch:
                if window_samples != window_capacity:
                    factor = window_capacity / window_samples
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(factor)
                window_samples = 0
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer_updated = True
                if grad_scaler is None:
                    optimizer.step()
                else:
                    old_scale = grad_scaler.get_scale()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer_updated = grad_scaler.get_scale() >= old_scale
                optimizer.zero_grad(set_to_none=True)

                # Per-update LR scheduling (OneCycleLR), skipping fp16 overflows
                if step_per_batch and optimizer_updated:
                    scheduler.step()

            train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)

        avg_train_loss = train_loss / train_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast_context():
                    pred = model(x)
                    if pred.shape != y.shape:
                        raise ValueError(
                            f"Model output shape {tuple(pred.shape)} does not match "
                            f"target shape {tuple(y.shape)}."
                        )
                    loss = validation_criterion(pred, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Cross-validation validation loss is non-finite."
                    )

                val_loss += loss.item() * x.size(0)
                val_samples += x.size(0)

                all_preds.append(pred.detach().float().cpu())
                all_targets.append(y.detach().cpu())

        avg_val_loss = val_loss / val_samples

        # Compute metrics (guard for tiny datasets)
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        metric_true = flatten_targets_by_channel(y_true)
        metric_pred = flatten_targets_by_channel(y_pred)
        r2 = (
            r2_score(metric_true, metric_pred)
            if len(metric_true) >= 2
            else float("nan")
        )
        mae = np.abs((y_pred - y_true) * scaler.scale_).mean()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "r2": r2,
                "mae": mae,
            }
        )

        # LR scheduling (epoch-based only, not for per-batch schedulers)
        if not step_per_batch and hasattr(scheduler, "step"):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            logger.info(f"    Fold {fold + 1}: Early stopping at epoch {epoch + 1}")
            break

    # Restore best model and compute final metrics
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast_context():
                pred = model(x)
            if pred.shape != y.shape:
                raise ValueError(
                    f"Model output shape {tuple(pred.shape)} does not match "
                    f"target shape {tuple(y.shape)}."
                )
            all_preds.append(pred.detach().float().cpu())
            all_targets.append(y.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    # Inverse transform for physical units
    y_pred_phys = scaler.inverse_transform(y_pred)
    y_true_phys = scaler.inverse_transform(y_true)
    metric_pred = flatten_targets_by_channel(y_pred)
    metric_true = flatten_targets_by_channel(y_true)
    metric_pred_phys = flatten_targets_by_channel(y_pred_phys)
    metric_true_phys = flatten_targets_by_channel(y_true_phys)

    results = {
        "fold": fold + 1,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "r2": (
            r2_score(metric_true, metric_pred)
            if len(metric_true) >= 2
            else float("nan")
        ),
        "mae_normalized": mean_absolute_error(metric_true, metric_pred),
        "mae_physical": mean_absolute_error(metric_true_phys, metric_pred_phys),
        "epochs_trained": len(history),
        "history": history,
    }

    # Per-target metrics (guard for tiny folds)
    for i in range(metric_true.shape[1]):
        if len(metric_true) >= 2:
            results[f"r2_target_{i}"] = r2_score(metric_true[:, i], metric_pred[:, i])
        else:
            results[f"r2_target_{i}"] = float("nan")
        results[f"mae_target_{i}"] = mean_absolute_error(
            metric_true_phys[:, i], metric_pred_phys[:, i]
        )

    return results


# ==============================================================================
# MAIN CV ORCHESTRATION
# ==============================================================================
@contextmanager
def _deterministic_cv(seed: int):
    """Apply standard training's deterministic policy, then restore caller state."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cudnn.benchmark = flags[2]
        torch.backends.cudnn.deterministic = flags[3]
        torch.backends.cuda.matmul.allow_tf32 = flags[4]
        torch.backends.cudnn.allow_tf32 = flags[5]
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _with_cv_determinism(function):
    """Run ``function`` inside ``_deterministic_cv`` when ``deterministic=True``."""
    parameters = signature(function)

    @wraps(function)
    def scoped(*args, **kwargs):
        bound = parameters.bind_partial(*args, **kwargs).arguments
        context = (
            _deterministic_cv(bound.get("seed", parameters.parameters["seed"].default))
            if bound.get("deterministic", False)
            else nullcontext()
        )
        with context:
            return function(*args, **kwargs)

    return scoped


def _seed_cv_worker(_worker_id: int) -> None:
    """Seed Python/NumPy in DataLoader workers from their per-worker Torch seed."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _build_fold_criterion(
    loss_name: str,
    loss_weights: list[float] | None,
    huber_delta: float,
    training_scaler: Any,
    constraint: list[str] | None,
    constraint_file: str | None,
    constraint_weight: list[float] | None,
    constraint_reduction: str,
) -> nn.Module:
    """Build the fold loss; constraints use only this fold's training scaler."""
    from wavedl.utils import get_loss
    from wavedl.utils.constraints import PhysicsConstrainedLoss, build_constraints

    criterion = get_loss(loss_name, weights=loss_weights, delta=huber_delta)
    constraints = build_constraints(
        expressions=constraint,
        file_path=constraint_file,
        reduction=constraint_reduction,
    )
    if constraints:
        # Evaluate penalties in physical space, as standard training does
        criterion = PhysicsConstrainedLoss(
            criterion,
            constraints,
            weights=constraint_weight,
            output_mean=getattr(training_scaler, "mean_", None),
            output_std=getattr(training_scaler, "scale_", None),
        )
    return criterion


@_with_cv_determinism
def run_cross_validation(
    # Data
    X: np.ndarray,
    y: np.ndarray,
    # Model
    model_name: str,
    in_shape: tuple[int, ...],
    out_size: int,
    # CV settings
    folds: int = 5,
    stratify: bool = False,
    stratify_bins: int = 10,
    # Training settings
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 20,
    weight_decay: float = 1e-4,
    # Components
    loss_name: str = "mse",
    optimizer_name: str = "adamw",
    scheduler_name: str = "plateau",
    # Scheduler config
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    # Optimizer config
    betas: tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.9,
    # Output
    output_dir: str = "./cv_results",
    workers: int = 4,
    seed: int = 2025,
    grad_clip: float = 1.0,
    logger: logging.Logger | None = None,
    out_shape: tuple[int, ...] | None = None,
    loss_weights: list[float] | None = None,
    huber_delta: float = 1.0,
    pretrained: bool = False,
    nesterov: bool = False,
    step_size: int = 30,
    milestones: list[int] | None = None,
    warmup_epochs: int = 5,
    device: torch.device | str | None = None,
    grad_accum_steps: int = 1,
    precision: str = "no",
    constraint: list[str] | None = None,
    constraint_file: str | None = None,
    constraint_weight: list[float] | None = None,
    constraint_reduction: str = "mse",
    deterministic: bool = False,
) -> dict[str, Any]:
    """
    Run K-fold cross-validation.

    Args:
        X: Input data
        y: Target data
        model_name: Model architecture name
        in_shape: Input shape (excluding batch and channel)
        out_size: Number of vector targets or dense field channels
        out_shape: Canonical per-sample target shape, e.g. ``(C,H,W)``
        folds: Number of CV folds
        stratify: Use stratified splitting
        stratify_bins: Number of bins for stratification
        batch_size: Batch size
        lr: Learning rate
        epochs: Max epochs per fold
        patience: Early stopping patience
        weight_decay: Weight decay
        loss_name: Loss function name
        loss_weights: Per-target or per-field-channel weights for weighted MSE
        huber_delta: Transition threshold for Huber loss
        pretrained: Whether to initialize pretrained model weights
        optimizer_name: Optimizer name
        scheduler_name: Scheduler name
        output_dir: Output directory
        workers: DataLoader workers
        seed: Random seed
        logger: Logger instance
        device: Torch device; auto-selects CUDA, MPS, then CPU when None
        grad_accum_steps: Micro-batches per optimizer update
        precision: ``"no"``, ``"bf16"`` (CPU/CUDA/MPS) or ``"fp16"`` (CUDA/MPS)
        constraint: Soft constraint expressions (see ``--constraint``)
        constraint_file: Python file defining ``constraint(pred, inputs)``
        constraint_weight: Weight(s) for the soft constraints
        constraint_reduction: Constraint penalty reduction (``"mse"``/``"mae"``)
        deterministic: Apply standard training's deterministic mode; caller RNG
            and backend settings are restored afterwards

    Returns:
        Summary dictionary with aggregated results
    """
    # Setup
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive.")
    if len(X) != len(y):
        raise ValueError("CV inputs and targets must have equal sample counts.")
    os.makedirs(output_dir, exist_ok=True)
    y = canonicalize_targets(y)
    inferred_out_shape = tuple(y.shape[1:])
    if out_shape is None:
        out_shape = inferred_out_shape
    else:
        out_shape = tuple(out_shape)
        if out_shape != inferred_out_shape:
            raise ValueError(
                f"out_shape={out_shape} does not match targets {inferred_out_shape}."
            )
    if out_size != out_shape[0]:
        raise ValueError(
            f"out_size={out_size} does not match target channels {out_shape[0]}."
        )

    if folds < 2:
        raise ValueError(
            f"Cross-validation requires at least 2 folds, got {folds}. "
            f"Use standard training (without --cv) for single-split training."
        )

    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("CV-Trainer")

    # Set seeds for reproducibility
    # Note: sklearn KFold uses random_state parameter directly, not global numpy RNG
    rng = np.random.default_rng(seed)  # Local RNG for any numpy operations
    _ = rng  # Silence unused variable warning (available for future use)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is not None:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _validate_training_options(device, grad_accum_steps, precision)

    # Auto-detect optimal DataLoader workers if not specified (matches train.py behavior)
    if workers < 0:
        cpu_count = os.cpu_count() or 4
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # Heuristic: 4-16 workers per GPU, bounded by available CPU cores
        workers = min(16, max(2, (cpu_count - 2) // max(1, num_gpus)))
        logger.info(
            f"⚙️  Auto-detected workers: {workers} (CPUs: {cpu_count}, GPUs: {num_gpus})"
        )

    logger.info(f"🚀 K-Fold Cross-Validation ({folds} folds)")
    logger.info(f"   Model: {model_name} | Device: {device} | Precision: {precision}")
    if grad_accum_steps > 1:
        logger.info(f"   Gradient Accumulation: {grad_accum_steps} steps")
    logger.info(
        f"   Loss: {loss_name} | Optimizer: {optimizer_name} | Scheduler: {scheduler_name}"
    )
    logger.info(f"   Data shape: X={X.shape}, y={y.shape}")

    # Setup cross-validation
    if stratify and len(out_shape) > 1:
        raise ValueError(
            "Stratified cross-validation is not defined for dense fields. "
            "Disable --cv_stratify or provide specimen/model-level groups."
        )
    if stratify:
        if y.ndim > 1 and y.shape[1] > 1:
            logger.warning(
                "⚠️  Stratification bins on the first target only (y[:, 0]); "
                "other targets may be imbalanced across folds."
            )
        try:
            # Bin targets for stratification (regression)
            y_binned = np.digitize(
                y[:, 0], np.percentile(y[:, 0], np.linspace(0, 100, stratify_bins + 1))
            )
            y_binned = np.clip(y_binned, 1, stratify_bins)
            kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            splits = list(kfold.split(X, y_binned))
        except ValueError:
            logger.warning(
                "⚠️  StratifiedKFold failed (too few samples per bin). "
                "Falling back to standard KFold."
            )
            standard_kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
            splits = list(standard_kfold.split(X))
    else:
        standard_kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(standard_kfold.split(X))

    # Import factories
    from wavedl.models import build_model, get_model
    from wavedl.utils import get_optimizer, get_scheduler

    # Run folds
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 Fold {fold + 1}/{folds}")
        logger.info(f"   Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        # Split data
        X_train = _take_samples(X, train_idx)
        X_val = _take_samples(X, val_idx)
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit scaler on training data only
        scaler = create_target_scaler(y_train)
        y_train_scaled = scaler.transform(y_train)
        y_val_scaled = scaler.transform(y_val)

        # Create datasets and loaders with explicit spatial dimensionality
        spatial_ndim = len(in_shape)
        train_ds = CVDataset(
            X_train,
            y_train_scaled.astype(np.float32),
            expected_spatial_ndim=spatial_ndim,
        )
        val_ds = CVDataset(
            X_val,
            y_val_scaled.astype(np.float32),
            expected_spatial_ndim=spatial_ndim,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            generator=(
                torch.Generator().manual_seed(seed + fold) if deterministic else None
            ),
            worker_init_fn=_seed_cv_worker if deterministic else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=_seed_cv_worker if deterministic else None,
        )

        # Build model
        model_class = get_model(model_name)
        if len(out_shape) > 1 and not getattr(
            model_class, "supports_spatial_output", False
        ):
            raise ValueError(
                f"Model '{model_name}' cannot produce dense output {out_shape}. "
                "Use unet_field or another spatial-output model."
            )
        if len(out_shape) == 1 and getattr(
            model_class, "requires_spatial_output", False
        ):
            raise ValueError(
                f"Model '{model_name}' requires dense field targets, got {out_shape}."
            )
        model_kwargs = {"out_shape": out_shape} if len(out_shape) > 1 else {}
        model = build_model(
            model_name,
            in_shape=in_shape,
            out_size=out_size,
            pretrained=pretrained,
            **model_kwargs,
        )
        model = model.to(device)

        # Setup training components
        criterion = _build_fold_criterion(
            loss_name,
            loss_weights,
            huber_delta,
            scaler,
            constraint,
            constraint_file,
            constraint_weight,
            constraint_reduction,
        ).to(device)
        optimizer = get_optimizer(
            optimizer_name,
            model.get_optimizer_groups(lr, weight_decay),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            momentum=momentum,
            nesterov=nesterov,
        )
        scheduler = get_scheduler(
            scheduler_name,
            optimizer,
            epochs=epochs,
            # OneCycleLR counts optimizer updates, not micro-batches
            steps_per_epoch=(
                math.ceil(len(train_loader) / grad_accum_steps)
                if scheduler_name == "onecycle"
                else None
            ),
            patience=scheduler_patience,
            factor=scheduler_factor,
            min_lr=min_lr,
            gamma=scheduler_factor,
            step_size=step_size,
            milestones=milestones,
            warmup_epochs=warmup_epochs,
        )

        # Train fold
        results = train_fold(
            fold=fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            patience=patience,
            grad_clip=grad_clip,
            scaler=scaler,
            logger=logger,
            grad_accum_steps=grad_accum_steps,
            precision=precision,
        )

        fold_results.append(results)

        logger.info(
            f"    Fold {fold + 1} Results: R²={results['r2']:.4f}, MAE={results['mae_physical']:.4f}"
        )

        # Save fold model
        fold_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pth"))
        with open(os.path.join(fold_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(fold_dir, "training_meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    # The saved weights are the best epoch's, not the last one's
                    "epoch": int(results["best_epoch"]),
                    "epochs_trained": int(results["epochs_trained"]),
                    "best_val_loss": float(results["best_val_loss"]),
                    "model_name": model_name,
                    "in_shape": tuple(int(v) for v in in_shape),
                    "out_dim": int(out_size),
                    "target_shape": tuple(int(v) for v in out_shape),
                    "cv_fold": fold + 1,
                },
                f,
            )

        # Explicit cleanup to prevent OOM across folds
        del model, optimizer, scheduler, criterion
        del train_ds, val_ds, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ==============================================================================
    # AGGREGATE RESULTS
    # ==============================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("📈 Cross-Validation Summary")
    logger.info("=" * 60)

    r2_scores = [r["r2"] for r in fold_results]
    mae_scores = [r["mae_physical"] for r in fold_results]
    val_losses = [r["best_val_loss"] for r in fold_results]

    summary = {
        "config": {
            "model": model_name,
            "target_shape": out_shape,
            "folds": folds,
            "stratify": stratify,
            "stratify_bins": stratify_bins,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "patience": patience,
            "loss": loss_name,
            "loss_weights": loss_weights,
            "huber_delta": huber_delta,
            "pretrained": pretrained,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "grad_accum_steps": grad_accum_steps,
            "precision": precision,
            "constraints": list(constraint or []),
            "constraint_file": constraint_file,
            "deterministic": deterministic,
        },
        "timestamp": datetime.now().isoformat(),
        "folds": folds,
        "r2_mean": float(np.nanmean(r2_scores)),
        "r2_std": float(np.nanstd(r2_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "val_loss_mean": float(np.mean(val_losses)),
        "val_loss_std": float(np.std(val_losses)),
        "fold_results": fold_results,
    }

    logger.info(f"   R² Score:    {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    logger.info(f"   MAE (phys):  {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    logger.info(
        f"   Val Loss:    {summary['val_loss_mean']:.6f} ± {summary['val_loss_std']:.6f}"
    )

    # Per-target summary
    for i in range(out_size):
        r2_target = [r.get(f"r2_target_{i}", np.nan) for r in fold_results]
        mae_target = [r.get(f"mae_target_{i}", np.nan) for r in fold_results]
        logger.info(
            f"   Target {i}: R²={np.nanmean(r2_target):.4f}±{np.nanstd(r2_target):.4f}, "
            f"MAE={np.nanmean(mae_target):.4f}±{np.nanstd(mae_target):.4f}"
        )

    # Save summary (without bulky history to keep JSON small)
    with open(os.path.join(output_dir, "cv_summary.json"), "w") as f:
        summary_save = {
            **summary,
            "fold_results": [
                {k: v for k, v in r.items() if k != "history"} for r in fold_results
            ],
        }
        json.dump(summary_save, f, indent=2)

    # Save detailed results as CSV
    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "history"} for r in fold_results]
    )
    results_df.to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)

    logger.info(f"\n✅ Results saved to: {output_dir}")

    return summary
