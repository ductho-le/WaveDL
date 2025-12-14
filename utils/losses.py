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
    - weighted_mse: Per-target weighted MSE

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import List, Optional, Union
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
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: Specifies the reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
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
        diff = pred - target
        # log(cosh(x)) = x + softplus(-2x) - log(2)
        # This formulation is numerically stable
        loss = diff + torch.nn.functional.softplus(-2.0 * diff) - 0.693147  # log(2)
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # mean
            return loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss.
    
    Applies different weights to each target dimension, allowing
    prioritization of specific outputs (e.g., prioritize thickness
    over velocity in NDE applications).
    
    Example:
        >>> # 3 targets, prioritize first target
        >>> criterion = WeightedMSELoss(weights=[2.0, 1.0, 1.0])
        >>> loss = criterion(predictions, targets)
    """
    
    def __init__(
        self, 
        weights: Optional[Union[List[float], torch.Tensor]] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            weights: Per-target weights. If None, equal weights (standard MSE).
                     Length must match number of output targets.
            reduction: Specifies the reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction
        
        if weights is not None:
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)
            self.register_buffer('weights', weights)
        else:
            self.weights = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            pred: Predicted values of shape (N, T) where T is number of targets
            target: Target values of shape (N, T)
            
        Returns:
            Loss value (scalar if reduction is 'mean' or 'sum')
            
        Raises:
            ValueError: If weight dimension doesn't match target dimension
        """
        mse = (pred - target) ** 2
        
        if self.weights is not None:
            # Validate weight dimension matches target dimension
            if self.weights.shape[0] != pred.shape[-1]:
                raise ValueError(
                    f"Weight dimension ({self.weights.shape[0]}) must match "
                    f"output dimension ({pred.shape[-1]}). "
                    f"Check your --loss_weights argument."
                )
            # Ensure weights are on same device
            if self.weights.device != mse.device:
                self.weights = self.weights.to(mse.device)
            # Apply per-target weights with correct broadcasting: (N, T) * (T,) -> (N, T)
            mse = mse * self.weights
        
        if self.reduction == 'none':
            return mse
        elif self.reduction == 'sum':
            return mse.sum()
        else:  # mean
            return mse.mean()


# ==============================================================================
# LOSS REGISTRY
# ==============================================================================
_LOSS_REGISTRY = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'l1': nn.L1Loss,  # Alias for mae
    'huber': nn.HuberLoss,
    'smooth_l1': nn.SmoothL1Loss,
    'log_cosh': LogCoshLoss,
    'logcosh': LogCoshLoss,  # Alias
    'weighted_mse': WeightedMSELoss,
}


def list_losses() -> List[str]:
    """
    Return list of available loss function names.
    
    Returns:
        List of registered loss function names (excluding aliases)
    """
    # Return unique loss names (exclude aliases)
    primary_names = ['mse', 'mae', 'huber', 'smooth_l1', 'log_cosh', 'weighted_mse']
    return primary_names


def get_loss(
    name: str,
    weights: Optional[List[float]] = None,
    delta: float = 1.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function by name.
    
    Args:
        name: Loss function name (see list_losses())
        weights: Per-target weights for weighted_mse
        delta: Delta parameter for Huber loss (default: 1.0)
        **kwargs: Additional arguments passed to loss constructor
    
    Returns:
        Instantiated loss function (nn.Module)
    
    Raises:
        ValueError: If loss name is not recognized
    
    Example:
        >>> criterion = get_loss('mse')
        >>> criterion = get_loss('huber', delta=0.5)
        >>> criterion = get_loss('weighted_mse', weights=[2.0, 1.0, 1.0])
    """
    name_lower = name.lower().replace('-', '_')
    
    if name_lower not in _LOSS_REGISTRY:
        available = ', '.join(list_losses())
        raise ValueError(
            f"Unknown loss function: '{name}'. "
            f"Available options: {available}"
        )
    
    loss_cls = _LOSS_REGISTRY[name_lower]
    
    # Special handling for specific loss types
    if name_lower == 'huber':
        return loss_cls(delta=delta, **kwargs)
    elif name_lower == 'weighted_mse':
        return loss_cls(weights=weights, **kwargs)
    else:
        return loss_cls(**kwargs)
