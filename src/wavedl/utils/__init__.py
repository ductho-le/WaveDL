"""
Utility Functions and Classes
=============================

Centralized exports for all utility modules.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import importlib
import os
import tempfile
from typing import TYPE_CHECKING, Any


def setup_hpc_cache_dirs() -> None:
    """
    Configure cache directories for HPC environments with read-only home.

    Auto-configures writable cache directories when home is not writable.
    Uses current working directory as fallback (or system temp directory if
    the current working directory is also read-only).

    Call this BEFORE importing libraries that use cache directories:
        - torch (TORCH_HOME)
        - matplotlib (MPLCONFIGDIR)
        - fontconfig (FONTCONFIG_CACHE)

    Example:
        from wavedl.utils import setup_hpc_cache_dirs
        setup_hpc_cache_dirs()  # Must be before torch/matplotlib imports
    """

    def _ensure_writable_cache_dir(subdir: str) -> str:
        uid_suffix = (
            str(os.getuid())
            if hasattr(os, "getuid")
            else os.environ.get("USERNAME") or str(os.getpid())
        )
        candidates = [
            os.path.join(os.getcwd(), f".{subdir}"),
            os.path.join(
                tempfile.gettempdir(), f"wavedl_cache_{uid_suffix}", f".{subdir}"
            ),
        ]
        for candidate in candidates:
            try:
                os.makedirs(candidate, exist_ok=True)
                fd, probe = tempfile.mkstemp(prefix=".wavedl_probe_", dir=candidate)
                os.close(fd)
                os.unlink(probe)
                return candidate
            except OSError:
                continue
        return candidates[-1]

    def _setup_cache_dir(env_var: str, subdir: str) -> None:
        if env_var in os.environ:
            return  # User already set, respect their choice
        home = os.path.expanduser("~")
        if os.access(home, os.W_OK):
            return  # Home is writable, let library use defaults
        # Home not writable - use current working directory (or temp fallback)
        os.environ[env_var] = _ensure_writable_cache_dir(subdir)

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
                os.environ[env_var] = _ensure_writable_cache_dir(subdir)

    _setup_cache_dir("XDG_DATA_HOME", "local/share")
    _setup_cache_dir("XDG_STATE_HOME", "local/state")
    _setup_cache_dir("XDG_CACHE_HOME", "cache")


if TYPE_CHECKING:
    from .checkpoint_io import (
        load_scaler,
        load_training_meta,
        save_scaler,
        save_training_meta,
    )
    from .config import (
        create_default_config,
        load_config,
        merge_config_with_args,
        save_config,
        validate_config,
    )
    from .constraints import (
        ExpressionConstraint,
        FileConstraint,
        PhysicsConstrainedLoss,
        build_constraints,
    )
    from .cross_validation import (
        CVDataset,
        run_cross_validation,
        train_fold,
    )
    from .data import (
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
    from .distributed import (
        broadcast_early_stop,
        broadcast_value,
        sync_tensor,
        wait_for_everyone,
    )
    from .losses import (
        LogCoshLoss,
        WeightedMSELoss,
        get_loss,
        list_losses,
    )
    from .metrics import (
        COLORS,
        FIGURE_DPI,
        FIGURE_WIDTH_CM,
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
    from .optimizers import (
        get_optimizer,
        get_optimizer_with_param_groups,
        list_optimizers,
    )
    from .preprocessing import (
        ManifestInputs,
        ManifestTargets,
        MinMaxTargetScaler,
        ShardManifestDataset,
        configure_tf32,
        inverse_transform_inputs,
        preprocessing_from_args,
        transform_inputs,
        validate_input_channels,
    )
    from .schedulers import (
        get_scheduler,
        get_scheduler_with_warmup,
        is_epoch_based,
        list_schedulers,
    )
    from .targets import (
        SpatialTargetScaler,
        canonical_target_shape,
        canonicalize_targets,
        create_target_scaler,
        flatten_targets_by_channel,
        get_target_shape,
        is_spatial_target_shape,
        scaler_target_shape,
    )


_LAZY_SUBMODULE_EXPORTS: dict[str, str] = {
    # checkpoint_io
    "load_scaler": ".checkpoint_io",
    "load_training_meta": ".checkpoint_io",
    "save_scaler": ".checkpoint_io",
    "save_training_meta": ".checkpoint_io",
    # config
    "create_default_config": ".config",
    "load_config": ".config",
    "merge_config_with_args": ".config",
    "save_config": ".config",
    "validate_config": ".config",
    # constraints
    "ExpressionConstraint": ".constraints",
    "FileConstraint": ".constraints",
    "PhysicsConstrainedLoss": ".constraints",
    "build_constraints": ".constraints",
    # cross_validation
    "CVDataset": ".cross_validation",
    "run_cross_validation": ".cross_validation",
    "train_fold": ".cross_validation",
    # data
    "DataSource": ".data",
    "HDF5Source": ".data",
    "MATSource": ".data",
    "MemmapDataset": ".data",
    "NPZSource": ".data",
    "get_data_source": ".data",
    "load_outputs_only": ".data",
    "load_test_data": ".data",
    "load_training_data": ".data",
    "memmap_worker_init_fn": ".data",
    "prepare_data": ".data",
    # distributed
    "broadcast_early_stop": ".distributed",
    "broadcast_value": ".distributed",
    "sync_tensor": ".distributed",
    "wait_for_everyone": ".distributed",
    # losses
    "LogCoshLoss": ".losses",
    "WeightedMSELoss": ".losses",
    "get_loss": ".losses",
    "list_losses": ".losses",
    # metrics
    "COLORS": ".metrics",
    "FIGURE_DPI": ".metrics",
    "FIGURE_WIDTH_CM": ".metrics",
    "FIGURE_WIDTH_INCH": ".metrics",
    "FONT_SIZE_TEXT": ".metrics",
    "FONT_SIZE_TICKS": ".metrics",
    "MetricTracker": ".metrics",
    "StreamingFieldMetrics": ".metrics",
    "calc_pearson": ".metrics",
    "calc_per_target_r2": ".metrics",
    "configure_matplotlib_style": ".metrics",
    "create_training_curves": ".metrics",
    "get_lr": ".metrics",
    "plot_bland_altman": ".metrics",
    "plot_correlation_heatmap": ".metrics",
    "plot_error_boxplot": ".metrics",
    "plot_error_cdf": ".metrics",
    "plot_error_histogram": ".metrics",
    "plot_prediction_vs_index": ".metrics",
    "plot_qq": ".metrics",
    "plot_relative_error": ".metrics",
    "plot_residuals": ".metrics",
    "plot_scientific_scatter": ".metrics",
    # optimizers
    "get_optimizer": ".optimizers",
    "get_optimizer_with_param_groups": ".optimizers",
    "list_optimizers": ".optimizers",
    # preprocessing
    "ManifestInputs": ".preprocessing",
    "ManifestTargets": ".preprocessing",
    "MinMaxTargetScaler": ".preprocessing",
    "ShardManifestDataset": ".preprocessing",
    "configure_tf32": ".preprocessing",
    "inverse_transform_inputs": ".preprocessing",
    "preprocessing_from_args": ".preprocessing",
    "transform_inputs": ".preprocessing",
    "validate_input_channels": ".preprocessing",
    # schedulers
    "get_scheduler": ".schedulers",
    "get_scheduler_with_warmup": ".schedulers",
    "is_epoch_based": ".schedulers",
    "list_schedulers": ".schedulers",
    # targets
    "SpatialTargetScaler": ".targets",
    "canonical_target_shape": ".targets",
    "canonicalize_targets": ".targets",
    "create_target_scaler": ".targets",
    "flatten_targets_by_channel": ".targets",
    "get_target_shape": ".targets",
    "is_spatial_target_shape": ".targets",
    "scaler_target_shape": ".targets",
}


def __getattr__(name: str) -> Any:
    """Lazily import utility symbols so setup_hpc_cache_dirs runs before heavy deps."""
    if name in _LAZY_SUBMODULE_EXPORTS:
        module = importlib.import_module(_LAZY_SUBMODULE_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
    "ManifestInputs",
    "ManifestTargets",
    "MemmapDataset",
    "MetricTracker",
    "MinMaxTargetScaler",
    "NPZSource",
    "PhysicsConstrainedLoss",
    "ShardManifestDataset",
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
    "configure_tf32",
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
    "inverse_transform_inputs",
    "is_epoch_based",
    "is_spatial_target_shape",
    "list_losses",
    "list_optimizers",
    "list_schedulers",
    "load_config",
    "load_outputs_only",
    "load_scaler",
    "load_test_data",
    "load_training_data",
    "load_training_meta",
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
    "preprocessing_from_args",
    "run_cross_validation",
    "save_config",
    "save_scaler",
    "save_training_meta",
    "scaler_target_shape",
    "setup_hpc_cache_dirs",
    "sync_tensor",
    "train_fold",
    "transform_inputs",
    "validate_config",
    "validate_input_channels",
    "wait_for_everyone",
]
