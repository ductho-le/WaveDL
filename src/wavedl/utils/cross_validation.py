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
import os
import pickle
from datetime import datetime
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
) -> dict[str, Any]:
    """
    Train and evaluate a single CV fold.

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

    Returns:
        Dictionary with fold results and metrics
    """
    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None
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

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            if pred.shape != y.shape:
                raise ValueError(
                    f"Model output shape {tuple(pred.shape)} does not match "
                    f"target shape {tuple(y.shape)}."
                )
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Cross-validation training loss is non-finite."
                )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Per-batch LR scheduling (OneCycleLR)
            if step_per_batch:
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
                pred = model(x)
                if pred.shape != y.shape:
                    raise ValueError(
                        f"Model output shape {tuple(pred.shape)} does not match "
                        f"target shape {tuple(y.shape)}."
                    )
                loss = criterion(pred, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Cross-validation validation loss is non-finite."
                    )

                val_loss += loss.item() * x.size(0)
                val_samples += x.size(0)

                all_preds.append(pred.detach().cpu())
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
            pred = model(x)
            if pred.shape != y.shape:
                raise ValueError(
                    f"Model output shape {tuple(pred.shape)} does not match "
                    f"target shape {tuple(y.shape)}."
                )
            all_preds.append(pred.detach().cpu())
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
    logger.info(f"   Model: {model_name} | Device: {device}")
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
    from wavedl.utils import get_loss, get_optimizer, get_scheduler

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
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
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
        criterion = get_loss(loss_name, weights=loss_weights, delta=huber_delta).to(
            device
        )
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
            steps_per_epoch=len(train_loader) if scheduler_name == "onecycle" else None,
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
                    "epoch": results["epochs_trained"],
                    "best_val_loss": results["best_val_loss"],
                    "model_name": model_name,
                    "in_shape": in_shape,
                    "out_dim": out_size,
                    "target_shape": out_shape,
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
