"""Build fold-local losses without leaking validation targets into constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wavedl.utils.constraints import PhysicsConstrainedLoss, build_constraints
from wavedl.utils.losses import get_loss


if TYPE_CHECKING:
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch import nn


def build_cv_criterion(
    *,
    loss_name: str,
    loss_weights: list[float] | None,
    huber_delta: float,
    training_scaler: StandardScaler,
    device: torch.device,
    constraint: list[str] | None = None,
    constraint_file: str | None = None,
    constraint_weight: list[float] | None = None,
    constraint_reduction: str = "mse",
) -> nn.Module:
    """Use only the current fold's training scaler for physical-space penalties.

    Callers use the full loss for training with model inputs, and ``base_loss``
    alone for validation and checkpoint selection, matching standard training.
    """
    criterion = get_loss(loss_name, weights=loss_weights, delta=huber_delta)
    constraints = build_constraints(
        expressions=constraint,
        file_path=constraint_file,
        reduction=constraint_reduction,
    )
    if constraints:
        criterion = PhysicsConstrainedLoss(
            criterion,
            constraints,
            weights=constraint_weight,
            output_mean=training_scaler.mean_,
            output_std=training_scaler.scale_,
        )
    return criterion.to(device)
