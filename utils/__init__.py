"""
Utility Functions and Classes
=============================

Centralized exports for all utility modules.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from utils.metrics import (
    MetricTracker,
    calc_pearson,
    calc_per_target_r2,
    plot_scientific_scatter,
    get_lr,
)

from utils.data import (
    MemmapDataset,
    memmap_worker_init_fn,
    prepare_data,
    # Multi-format data loading
    DataSource,
    NPZSource,
    HDF5Source,
    MATSource,
    get_data_source,
    load_training_data,
)

from utils.distributed import (
    broadcast_early_stop,
    broadcast_value,
    sync_tensor,
)

from utils.losses import (
    get_loss,
    list_losses,
    LogCoshLoss,
    WeightedMSELoss,
)

from utils.schedulers import (
    get_scheduler,
    list_schedulers,
    get_scheduler_with_warmup,
    is_epoch_based,
)

from utils.optimizers import (
    get_optimizer,
    list_optimizers,
    get_optimizer_with_param_groups,
)

from utils.cross_validation import (
    run_cross_validation,
    CVDataset,
    train_fold,
)

from utils.config import (
    load_config,
    save_config,
    merge_config_with_args,
    validate_config,
    create_default_config,
)

__all__ = [
    # Metrics
    "MetricTracker",
    "calc_pearson",
    "calc_per_target_r2", 
    "plot_scientific_scatter",
    "get_lr",
    # Data
    "MemmapDataset",
    "memmap_worker_init_fn",
    "prepare_data",
    "DataSource",
    "NPZSource",
    "HDF5Source",
    "MATSource",
    "get_data_source",
    "load_training_data",
    # Distributed
    "broadcast_early_stop",
    "broadcast_value",
    "sync_tensor",
    # Losses
    "get_loss",
    "list_losses",
    "LogCoshLoss",
    "WeightedMSELoss",
    # Schedulers
    "get_scheduler",
    "list_schedulers",
    "get_scheduler_with_warmup",
    "is_epoch_based",
    # Optimizers
    "get_optimizer",
    "list_optimizers",
    "get_optimizer_with_param_groups",
    # Cross-Validation
    "run_cross_validation",
    "CVDataset",
    "train_fold",
    # Config
    "load_config",
    "save_config",
    "merge_config_with_args",
    "validate_config",
    "create_default_config",
]
