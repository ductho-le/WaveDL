"""
Utility Functions and Classes
=============================

Centralized exports for all utility modules.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import os


def setup_hpc_cache_dirs() -> None:
    """
    Configure cache directories for HPC environments with read-only home.

    Auto-configures writable cache directories when home is not writable.
    Uses current working directory as fallback - works on HPC and local machines.

    Call this BEFORE importing libraries that use cache directories:
        - torch (TORCH_HOME)
        - matplotlib (MPLCONFIGDIR)
        - fontconfig (FONTCONFIG_CACHE)

    Example:
        from wavedl.utils import setup_hpc_cache_dirs
        setup_hpc_cache_dirs()  # Must be before torch/matplotlib imports
    """

    def _setup_cache_dir(env_var: str, subdir: str) -> None:
        if env_var in os.environ:
            return  # User already set, respect their choice
        home = os.path.expanduser("~")
        if os.access(home, os.W_OK):
            return  # Home is writable, let library use defaults
        # Home not writable - use current working directory
        cache_path = os.path.join(os.getcwd(), f".{subdir}")
        os.makedirs(cache_path, exist_ok=True)
        os.environ[env_var] = cache_path

    _setup_cache_dir("TORCH_HOME", "torch_cache")
    _setup_cache_dir("HF_HOME", "hf_cache")
    # Force offline mode when home is not writable (HPC compute nodes)
    home = os.path.expanduser("~")
    if not os.access(home, os.W_OK):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _setup_cache_dir("MPLCONFIGDIR", "matplotlib")
    _setup_cache_dir("FONTCONFIG_CACHE", "fontconfig")

    # For matplotlib/fontconfig, also set up even when home IS writable
    # but we're on HPC (detected by scheduler env vars), because
    # ~/.cache/matplotlib/tex.cache often has permission issues on compute nodes.
    hpc_indicators = [
        "SLURM_JOB_ID",
        "PBS_JOBID",
        "LSB_JOBID",
        "SGE_TASK_ID",
        "COBALT_JOBID",
    ]
    if any(var in os.environ for var in hpc_indicators):
        for env_var, subdir in [
            ("MPLCONFIGDIR", "matplotlib"),
            ("FONTCONFIG_CACHE", "fontconfig"),
        ]:
            if env_var not in os.environ:
                cache_path = os.path.join(os.getcwd(), f".{subdir}")
                os.makedirs(cache_path, exist_ok=True)
                os.environ[env_var] = cache_path

    _setup_cache_dir("XDG_DATA_HOME", "local/share")
    _setup_cache_dir("XDG_STATE_HOME", "local/state")
    _setup_cache_dir("XDG_CACHE_HOME", "cache")


from .config import (  # noqa: E402
    create_default_config,
    load_config,
    merge_config_with_args,
    save_config,
    validate_config,
)
from .constraints import (  # noqa: E402
    ExpressionConstraint,
    FileConstraint,
    PhysicsConstrainedLoss,
    build_constraints,
)
from .cross_validation import (  # noqa: E402
    CVDataset,
    run_cross_validation,
    train_fold,
)
from .data import (  # noqa: E402
    # Multi-format data loading
    DataSource,
    HDF5Source,
    MATSource,
    MemmapDataset,
    NPZSource,
    get_data_source,
    load_outputs_only,
    load_test_data,
    load_training_data,
    memmap_worker_init_fn,
    prepare_data,
)
from .distributed import (  # noqa: E402
    broadcast_early_stop,
    broadcast_value,
    sync_tensor,
    wait_for_everyone,
)
from .losses import (  # noqa: E402
    LogCoshLoss,
    WeightedMSELoss,
    get_loss,
    list_losses,
)
from .metrics import (  # noqa: E402
    COLORS,
    FIGURE_DPI,
    FIGURE_WIDTH_CM,
    # Style constants
    FIGURE_WIDTH_INCH,
    FONT_SIZE_TEXT,
    FONT_SIZE_TICKS,
    MetricTracker,
    StreamingFieldMetrics,
    calc_pearson,
    calc_per_target_r2,
    configure_matplotlib_style,
    create_training_curves,
    get_lr,
    plot_bland_altman,
    plot_correlation_heatmap,
    plot_error_boxplot,
    plot_error_cdf,
    plot_error_histogram,
    plot_prediction_vs_index,
    plot_qq,
    plot_relative_error,
    plot_residuals,
    plot_scientific_scatter,
)
from .optimizers import (  # noqa: E402
    get_optimizer,
    get_optimizer_with_param_groups,
    list_optimizers,
)
from .schedulers import (  # noqa: E402
    get_scheduler,
    get_scheduler_with_warmup,
    is_epoch_based,
    list_schedulers,
)
from .targets import (  # noqa: E402
    SpatialTargetScaler,
    canonical_target_shape,
    canonicalize_targets,
    create_target_scaler,
    flatten_targets_by_channel,
    get_target_shape,
    is_spatial_target_shape,
    scaler_target_shape,
)


__all__ = [
    "COLORS",
    "FIGURE_DPI",
    "FIGURE_WIDTH_CM",
    "FIGURE_WIDTH_INCH",
    "FONT_SIZE_TEXT",
    "FONT_SIZE_TICKS",
    "CVDataset",
    "DataSource",
    "ExpressionConstraint",
    "FileConstraint",
    "HDF5Source",
    "LogCoshLoss",
    "MATSource",
    "MemmapDataset",
    "MetricTracker",
    "NPZSource",
    "PhysicsConstrainedLoss",
    "SpatialTargetScaler",
    "StreamingFieldMetrics",
    "WeightedMSELoss",
    "broadcast_early_stop",
    "broadcast_value",
    "build_constraints",
    "calc_pearson",
    "calc_per_target_r2",
    "canonical_target_shape",
    "canonicalize_targets",
    "configure_matplotlib_style",
    "create_default_config",
    "create_target_scaler",
    "create_training_curves",
    "flatten_targets_by_channel",
    "get_data_source",
    "get_loss",
    "get_lr",
    "get_optimizer",
    "get_optimizer_with_param_groups",
    "get_scheduler",
    "get_scheduler_with_warmup",
    "get_target_shape",
    "is_epoch_based",
    "is_spatial_target_shape",
    "list_losses",
    "list_optimizers",
    "list_schedulers",
    "load_config",
    "load_outputs_only",
    "load_test_data",
    "load_training_data",
    "memmap_worker_init_fn",
    "merge_config_with_args",
    "plot_bland_altman",
    "plot_correlation_heatmap",
    "plot_error_boxplot",
    "plot_error_cdf",
    "plot_error_histogram",
    "plot_prediction_vs_index",
    "plot_qq",
    "plot_relative_error",
    "plot_residuals",
    "plot_scientific_scatter",
    "prepare_data",
    "run_cross_validation",
    "save_config",
    "scaler_target_shape",
    "setup_hpc_cache_dirs",
    "sync_tensor",
    "train_fold",
    "validate_config",
    "wait_for_everyone",
]
