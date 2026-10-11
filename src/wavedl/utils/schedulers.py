"""
Learning Rate Schedulers
========================

Provides a comprehensive set of learning rate schedulers with a factory
function for easy selection via CLI arguments.

Supported Schedulers:
    - plateau: ReduceLROnPlateau (default, adaptive)
    - cosine: CosineAnnealingLR
    - cosine_restarts: CosineAnnealingWarmRestarts
    - onecycle: OneCycleLR
    - step: StepLR
    - multistep: MultiStepLR
    - exponential: ExponentialLR
    - linear_warmup: LinearLR (warmup phase)

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import warnings

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)


# ==============================================================================
# SCHEDULER REGISTRY
# ==============================================================================
def list_schedulers() -> list[str]:
    """
    Return list of available scheduler names.

    Returns:
        List of registered scheduler names
    """
    return [
        "plateau",
        "cosine",
        "cosine_restarts",
        "onecycle",
        "step",
        "multistep",
        "exponential",
        "linear_warmup",
    ]


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    # Common parameters
    epochs: int = 100,
    steps_per_epoch: int | None = None,
    min_lr: float = 1e-6,
    # ReduceLROnPlateau parameters
    patience: int = 10,
    factor: float = 0.5,
    # Cosine parameters
    T_max: int | None = None,
    T_0: int = 10,
    T_mult: int = 2,
    # OneCycleLR parameters
    max_lr: float | None = None,
    pct_start: float = 0.3,
    # Step/MultiStep parameters
    step_size: int = 30,
    milestones: list[int] | None = None,
    gamma: float | None = None,
    # Linear warmup parameters
    warmup_epochs: int = 5,
    start_factor: float = 0.1,
    **kwargs,
) -> LRScheduler:
    """
    Factory function to create learning rate scheduler by name.

    Args:
        name: Scheduler name (see list_schedulers())
        optimizer: Optimizer instance to schedule
        epochs: Total training epochs (for cosine, onecycle)
        steps_per_epoch: Steps per epoch (required for onecycle)
        min_lr: Minimum learning rate (eta_min for cosine)
        patience: Patience for ReduceLROnPlateau
        factor: Reduction factor for plateau/step
        T_max: Period for CosineAnnealingLR (default: epochs)
        T_0: Initial period for CosineAnnealingWarmRestarts
        T_mult: Period multiplier for warm restarts
        max_lr: Maximum LR for OneCycleLR (default: optimizer's initial LR)
        pct_start: Percentage of cycle spent increasing LR (OneCycleLR)
        step_size: Period for StepLR
        milestones: Epochs to decay LR for MultiStepLR
        gamma: Decay factor. If None, defaults to 0.1 for step/multistep and
            0.95 for exponential. ExponentialLR applies gamma every epoch, so
            small values (e.g. 0.5) decay the LR very fast.
        warmup_epochs: Number of warmup epochs for linear_warmup
        start_factor: Starting LR factor for warmup (LR * start_factor)
        **kwargs: Additional arguments passed to scheduler

    Returns:
        Instantiated learning rate scheduler

    Raises:
        ValueError: If scheduler name is not recognized

    Example:
        >>> scheduler = get_scheduler("plateau", optimizer, patience=15)
        >>> scheduler = get_scheduler("cosine", optimizer, epochs=100)
        >>> scheduler = get_scheduler(
        ...     "onecycle", optimizer, epochs=100, steps_per_epoch=1000, max_lr=1e-3
        ... )
    """
    name_lower = name.lower().replace("-", "_")

    # Get initial LR from optimizer
    base_lr = optimizer.param_groups[0]["lr"]

    if name_lower == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            **kwargs,
        )

    elif name_lower == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max if T_max is not None else epochs,
            eta_min=min_lr,
            **kwargs,
        )

    elif name_lower == "cosine_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr, **kwargs
        )

    elif name_lower == "onecycle":
        if steps_per_epoch is None:
            raise ValueError(
                "OneCycleLR requires 'steps_per_epoch'. "
                "Pass len(train_dataloader) as steps_per_epoch."
            )
        return OneCycleLR(
            optimizer,
            max_lr=max_lr if max_lr is not None else base_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            **kwargs,
        )

    elif name_lower == "step":
        step_gamma = gamma if gamma is not None else 0.1
        return StepLR(optimizer, step_size=step_size, gamma=step_gamma, **kwargs)

    elif name_lower == "multistep":
        if milestones is None:
            # Default milestones at 30%, 60%, 90% of epochs
            # Short runs must not decay several times during construction
            # because rounded milestones coincide at epoch zero.
            milestones = sorted(
                {max(1, int(epochs * fraction)) for fraction in (0.3, 0.6, 0.9)}
            )
        step_gamma = gamma if gamma is not None else 0.1
        return MultiStepLR(optimizer, milestones=milestones, gamma=step_gamma, **kwargs)

    elif name_lower == "exponential":
        exp_gamma = gamma if gamma is not None else 0.95
        if exp_gamma < 0.9:
            warnings.warn(
                f"ExponentialLR applies gamma every epoch; gamma={exp_gamma} decays "
                f"the LR by {(1 - exp_gamma) * 100:.0f}% per epoch, which is very "
                "aggressive. Use a value closer to 1.0 (e.g. 0.95) for gradual decay.",
                UserWarning,
                stacklevel=2,
            )
        return ExponentialLR(optimizer, gamma=exp_gamma, **kwargs)

    elif name_lower == "linear_warmup":
        return LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
            **kwargs,
        )

    else:
        available = ", ".join(list_schedulers())
        raise ValueError(f"Unknown scheduler: '{name}'. Available options: {available}")


def get_scheduler_with_warmup(
    name: str,
    optimizer: optim.Optimizer,
    warmup_epochs: int = 5,
    start_factor: float = 0.1,
    **kwargs,
) -> LRScheduler:
    """
    Create a scheduler with linear warmup phase.

    Combines LinearLR warmup with any other scheduler using SequentialLR.

    Args:
        name: Main scheduler name (after warmup)
        optimizer: Optimizer instance
        warmup_epochs: Number of warmup epochs
        start_factor: Starting LR factor for warmup
        **kwargs: Arguments for main scheduler (see get_scheduler)

    Returns:
        SequentialLR combining warmup and main scheduler

    Example:
        >>> scheduler = get_scheduler_with_warmup(
        ...     "cosine", optimizer, warmup_epochs=5, epochs=100
        ... )
    """
    # This SequentialLR combiner only supports epoch-based, epochs-parameterized
    # main schedulers. OneCycleLR steps per batch (incompatible with an epoch-count
    # milestone and has its own warmup via pct_start); cosine_restarts/step are
    # keyed on T_0/step_size rather than `epochs`, so the warmup budget adjustment
    # below would not shorten them.
    name_lower = name.lower().replace("-", "_")
    if name_lower in {"onecycle", "cosine_restarts", "plateau", "linear_warmup"}:
        raise ValueError(
            f"get_scheduler_with_warmup does not support '{name}'. Use OneCycleLR's "
            "built-in pct_start warmup, or a warmup-compatible scheduler "
            "(cosine, step, multistep, exponential)."
        )
    if type(warmup_epochs) is not int or warmup_epochs < 1:
        raise ValueError("warmup_epochs must be a positive integer.")
    if "epochs" in kwargs and kwargs["epochs"] <= warmup_epochs:
        raise ValueError(
            "epochs must exceed warmup_epochs to leave a main-scheduler phase."
        )

    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Create main scheduler with adjusted epoch budget
    if "epochs" in kwargs:
        kwargs["epochs"] = max(1, kwargs["epochs"] - warmup_epochs)
    main_scheduler = get_scheduler(name, optimizer, **kwargs)

    # Combine with SequentialLR
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )


def is_epoch_based(name: str) -> bool:
    """
    Check if scheduler should be stepped per epoch (True) or per batch (False).

    Args:
        name: Scheduler name

    Returns:
        True if scheduler should step per epoch, False for per batch
    """
    name_lower = name.lower().replace("-", "_")

    # OneCycleLR steps per batch, all others step per epoch
    per_batch_schedulers = {"onecycle"}

    return name_lower not in per_batch_schedulers
