"""
Loss Functions for Regression Tasks
====================================

Provides a comprehensive set of loss functions for regression problems,
with a factory function for easy selection via CLI arguments.

Supported Losses:
    - mse: Mean Squared Error (default)
    - mae: Mean Absolute Error (L1)
    - huber: Huber Loss (smooth blend of MSE and MAE)
    - smooth_l1: Smooth L1 Loss (PyTorch native Huber variant)
    - log_cosh: Log-Cosh Loss (smooth approximation to MAE)
    - weighted_mse: Per-vector-target or per-field-channel weighted MSE

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import math

import torch
import torch.nn as nn


# ==============================================================================
# CUSTOM LOSS FUNCTIONS
# ==============================================================================
class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss: A smooth approximation to Mean Absolute Error.

    The loss is defined as: loss = log(cosh(pred - target))

    Properties:
        - Smooth everywhere (differentiable)
        - Behaves like L2 for small errors, L1 for large errors
        - More robust to outliers than MSE

    Example:
        >>> criterion = LogCoshLoss()
        >>> loss = criterion(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Specifies the reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Log-Cosh loss.

        Args:
            pred: Predicted values of shape (N, *)
            target: Target values of shape (N, *)

        Returns:
            Loss value (scalar if reduction is 'mean' or 'sum')
        """
        if pred.shape != target.shape:
            raise ValueError("Prediction and target shapes must match.")
        # Promote low precision before subtraction and never exponentiate a
        # positive number: large negative errors otherwise overflow in fp16.
        work_pred = (
            pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred
        )
        work_target = target.to(work_pred.dtype)
        diff = (work_pred - work_target).abs()
        loss = diff + torch.nn.functional.softplus(-2.0 * diff) - math.log(2.0)

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            return loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss.

    Applies different weights to each vector target or dense property channel,
    allowing prioritization of specific outputs (e.g., prioritize thickness
    over velocity in NDE applications).

    Example:
        >>> # 3 targets, prioritize first target
        >>> criterion = WeightedMSELoss(weights=[2.0, 1.0, 1.0])
        >>> loss = criterion(predictions, targets)
    """

    def __init__(
        self, weights: list[float] | torch.Tensor | None = None, reduction: str = "mean"
    ):
        """
        Args:
            weights: Per-target/channel weights. If None, use standard MSE.
                     Length must match axis 1 of the model output.
            reduction: Specifies the reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

        if weights is not None:
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() == 0:
                raise ValueError("Loss weights must be a non-empty 1D sequence.")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Loss weights must be finite and non-negative.")
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            pred: Predictions shaped (N,T) or dense fields (N,C,H,W)
            target: Target tensor with the same shape as ``pred``

        Returns:
            Loss value (scalar if reduction is 'mean' or 'sum')

        Raises:
            ValueError: If weight dimension doesn't match target dimension
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match target shape "
                f"{tuple(target.shape)}."
            )
        mse = (pred - target) ** 2

        if self.weights is not None:
            # Targets/channels live on axis 1 for both (N,T) vectors and
            # channels-first dense fields (N,C,*spatial).
            if pred.ndim < 2:
                raise ValueError(
                    f"WeightedMSELoss expects at least 2D predictions, got {pred.shape}."
                )
            if self.weights.shape[0] != pred.shape[1]:
                raise ValueError(
                    f"Weight dimension ({self.weights.shape[0]}) must match "
                    f"output channel dimension ({pred.shape[1]}). "
                    f"Check your --loss_weights argument."
                )
            # Use local variable to avoid mutating registered buffer
            # (mutating self.weights breaks state_dict semantics)
            weights = self.weights.to(mse.device)
            # (T,) broadcasts directly for vectors.  Dense fields need
            # (1,C,1,1,...) so weights apply to physical channels, not width.
            if pred.ndim > 2:
                weights = weights.reshape(1, -1, *([1] * (pred.ndim - 2)))
            mse = mse * weights

        if self.reduction == "none":
            return mse
        elif self.reduction == "sum":
            return mse.sum()
        else:  # mean
            return mse.mean()


# ==============================================================================
# LOSS REGISTRY
# ==============================================================================
_LOSS_REGISTRY = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "l1": nn.L1Loss,  # Alias for mae
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "log_cosh": LogCoshLoss,
    "logcosh": LogCoshLoss,  # Alias
    "weighted_mse": WeightedMSELoss,
}


def list_losses() -> list[str]:
    """
    Return list of available loss function names.

    Returns:
        List of registered loss function names (excluding aliases)
    """
    # Return unique loss names (exclude aliases)
    primary_names = ["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"]
    return primary_names


def get_loss(
    name: str, weights: list[float] | None = None, delta: float = 1.0, **kwargs
) -> nn.Module:
    """
    Factory function to create loss function by name.

    Args:
        name: Loss function name (see list_losses())
        weights: Per-target/channel weights for weighted_mse
        delta: Delta parameter for Huber loss (default: 1.0)
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        Instantiated loss function (nn.Module)

    Raises:
        ValueError: If loss name is not recognized

    Example:
        >>> criterion = get_loss("mse")
        >>> criterion = get_loss("huber", delta=0.5)
        >>> criterion = get_loss("weighted_mse", weights=[2.0, 1.0, 1.0])
    """
    name_lower = name.lower().replace("-", "_")

    if name_lower not in _LOSS_REGISTRY:
        available = ", ".join(list_losses())
        raise ValueError(
            f"Unknown loss function: '{name}'. Available options: {available}"
        )

    loss_cls = _LOSS_REGISTRY[name_lower]

    # Special handling for specific loss types
    if name_lower == "huber":
        return loss_cls(delta=delta, **kwargs)
    elif name_lower == "weighted_mse":
        return loss_cls(weights=weights, **kwargs)
    else:
        return loss_cls(**kwargs)
