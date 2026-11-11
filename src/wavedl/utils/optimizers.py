"""
Optimizers for Deep Learning Training
=====================================

Provides a comprehensive set of optimizers with a factory function
for easy selection via CLI arguments.

Supported Optimizers:
    - adamw: AdamW (default, best for most cases)
    - adam: Adam (legacy)
    - sgd: SGD with momentum
    - nadam: NAdam (Adam + Nesterov momentum)
    - radam: RAdam (variance-adaptive Adam)
    - rmsprop: RMSprop (good for RNNs)

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from collections.abc import Iterator

import torch
import torch.optim as optim
from torch.nn import Parameter


# ==============================================================================
# OPTIMIZER REGISTRY
# ==============================================================================
def list_optimizers() -> list[str]:
    """
    Return list of available optimizer names.

    Returns:
        List of registered optimizer names
    """
    return ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]


def get_optimizer(
    name: str,
    params: Iterator[Parameter] | list[dict],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    # SGD-specific
    momentum: float = 0.9,
    nesterov: bool = False,
    dampening: float = 0.0,
    # Adam/AdamW-specific
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    amsgrad: bool = False,
    # RMSprop-specific
    alpha: float = 0.99,
    centered: bool = False,
    **kwargs,
) -> optim.Optimizer:
    """
    Factory function to create optimizer by name.

    Args:
        name: Optimizer name (see list_optimizers())
        params: Model parameters or parameter groups
        lr: Learning rate
        weight_decay: Weight decay (L2 penalty)
        momentum: Momentum factor (SGD, RMSprop)
        nesterov: Enable Nesterov momentum (SGD)
        dampening: Dampening for momentum (SGD)
        betas: Coefficients for computing running averages (Adam variants)
        eps: Term for numerical stability (Adam variants, RMSprop)
        amsgrad: Use AMSGrad variant (Adam variants)
        alpha: Smoothing constant (RMSprop)
        centered: Compute centered RMSprop
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Instantiated optimizer

    Raises:
        ValueError: If optimizer name is not recognized

    Example:
        >>> optimizer = get_optimizer("adamw", model.parameters(), lr=1e-3)
        >>> optimizer = get_optimizer("sgd", model.parameters(), lr=1e-2, momentum=0.9)
    """
    name_lower = name.lower()

    if name_lower == "adamw":
        return optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs,
        )

    elif name_lower == "adam":
        return optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs,
        )

    elif name_lower == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs,
        )

    elif name_lower == "nadam":
        return optim.NAdam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs
        )

    elif name_lower == "radam":
        return optim.RAdam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs
        )

    elif name_lower == "rmsprop":
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            **kwargs,
        )

    else:
        available = ", ".join(list_optimizers())
        raise ValueError(f"Unknown optimizer: '{name}'. Available options: {available}")


def get_optimizer_with_param_groups(
    name: str,
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    no_decay_keywords: list[str] | None = None,
    **kwargs,
) -> optim.Optimizer:
    """
    Create optimizer with automatic parameter grouping.

    Separates parameters into decay and no-decay groups based on
    parameter names. By default, bias and normalization layer parameters
    are excluded from weight decay.

    Args:
        name: Optimizer name
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay for applicable parameters
        no_decay_keywords: Keywords to identify no-decay parameters
                          Default: ['bias', 'norm', 'bn', 'ln', 'gamma', 'beta']
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer with parameter groups configured

    Example:
        >>> optimizer = get_optimizer_with_param_groups(
        ...     "adamw", model, lr=1e-3, weight_decay=1e-4
        ... )
    """
    use_default_norm_detection = no_decay_keywords is None
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "norm", "bn", "ln", "gamma", "beta"]

    norm_param_ids: set[int] = set()
    explicit_no_decay_names: set[str] = set()
    default_special_tokens = {
        "pos_embed",
        "cls_token",
        "layer_scale",
        "block_scale",
        "res_scale",
        "relative_position_bias_table",
        "attention_biases",
        "a_log",
        "log_dt",
        "a_real",
        "a_imag",
    }
    if use_default_norm_detection:
        no_wd_fn = getattr(model, "no_weight_decay", None)
        if callable(no_wd_fn):
            explicit_no_decay_names = set(no_wd_fn())

        norm_types = (
            torch.nn.modules.batchnorm._NormBase,
            torch.nn.GroupNorm,
            torch.nn.LayerNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        )
        if hasattr(torch.nn, "RMSNorm"):
            norm_types = (*norm_types, torch.nn.RMSNorm)
        for module in model.modules():
            cls_name = type(module).__name__
            if isinstance(module, norm_types) or cls_name.endswith(
                ("Norm", "Norm1d", "Norm2d", "Norm3d", "NormNd")
            ):
                for p in module.parameters(recurse=False):
                    norm_param_ids.add(id(p))

    decay_params = []
    no_decay_params = []

    for name_param, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lower_name = name_param.lower()
        leaf_name = lower_name.rsplit(".", 1)[-1]
        path_parts = set(lower_name.split("."))
        name_tokens = set(lower_name.replace(".", "_").split("_"))
        if use_default_norm_detection:
            is_no_decay = (
                param.ndim <= 1
                or id(param) in norm_param_ids
                or getattr(param, "_no_weight_decay", False)
                or name_param in explicit_no_decay_names
                or leaf_name in explicit_no_decay_names
                or bool(path_parts & default_special_tokens)
                or leaf_name in default_special_tokens
                or any(
                    kw.lower() in name_tokens or lower_name.endswith(f".{kw.lower()}")
                    for kw in no_decay_keywords
                )
            )
        else:
            is_no_decay = getattr(param, "_no_weight_decay", False) or any(
                kw.lower() in lower_name for kw in no_decay_keywords
            )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            }
        )

    # Fall back to trainable parameters if grouping produces no groups
    if not param_groups:
        trainable = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable, "weight_decay": 0.0}]

    return get_optimizer(name, param_groups, lr=lr, weight_decay=0.0, **kwargs)
