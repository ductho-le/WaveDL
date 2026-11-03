"""
Scientific Metrics and Visualization Utilities
===============================================

Provides metric tracking, statistical calculations, and publication-quality
visualization tools for deep learning experiments.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.1.0
"""

from typing import Any

import matplotlib


matplotlib.use("Agg")  # Headless-safe: no display required (HPC, CI, SSH)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from .targets import flatten_targets_by_channel


# ==============================================================================
# PUBLICATION-QUALITY PLOT CONFIGURATION
# ==============================================================================
# Width: 19 cm = 7.48 inches (for two-column journals)
FIGURE_WIDTH_CM = 19
FIGURE_WIDTH_INCH = FIGURE_WIDTH_CM / 2.54

# Font sizes (publication quality)
FONT_SIZE_TEXT = 10
FONT_SIZE_TICKS = 9
FONT_SIZE_TITLE = 11

# DPI for publication (300 for print, 150 for screen)
FIGURE_DPI = 300

# Color palette (accessible, print-friendly)
COLORS = {
    "primary": "#2E86AB",  # Steel blue
    "secondary": "#A23B72",  # Raspberry
    "accent": "#F18F01",  # Orange
    "neutral": "#6B717E",  # Slate gray
    "error": "#C73E1D",  # Red
    "success": "#3A7D44",  # Green
    "scatter": "#96C2D5",  # Light steel blue (simulates primary at 50% alpha on white)
}


def _is_latex_available() -> bool:
    """Check if LaTeX is available for matplotlib rendering."""
    import shutil

    return shutil.which("latex") is not None


def configure_matplotlib_style():
    """Configure matplotlib for publication-quality LaTeX-style plots.

    Falls back to standard fonts if LaTeX is not installed.
    """
    use_latex = _is_latex_available()

    if use_latex:
        latex_settings = {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        }
    else:
        # Fallback for systems without LaTeX (e.g., CI runners)
        latex_settings = {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        }

    plt.rcParams.update(
        {
            # LaTeX/font settings (conditional)
            **latex_settings,
            # Font sizes
            "font.size": FONT_SIZE_TEXT,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_TEXT,
            "xtick.labelsize": FONT_SIZE_TICKS,
            "ytick.labelsize": FONT_SIZE_TICKS,
            "legend.fontsize": FONT_SIZE_TICKS,
            # Line widths
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            # Grid style (use light gray instead of alpha for vector compatibility)
            "grid.color": "#CCCCCC",
            "grid.linestyle": ":",
            # Figure settings
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Remove top/right spines for cleaner look
            "axes.spines.top": False,
            "axes.spines.right": False,
            # SVG/vector export settings - prevent rasterization
            "svg.fonttype": "none",  # Embed fonts as text, not paths
            "image.composite_image": False,  # Don't composite images
        }
    )


# Lazy style initialization flag
_style_configured = False


def _ensure_style_configured():
    """Apply matplotlib style on first use (lazy initialization).

    This avoids modifying global plt.rcParams at import time, which could
    unexpectedly override user configurations.
    """
    global _style_configured
    if not _style_configured:
        configure_matplotlib_style()
        _style_configured = True


# ==============================================================================
# METRIC TRACKING
# ==============================================================================
class MetricTracker:
    """
    Tracks running averages of metrics during training.

    Useful for tracking loss, accuracy, or any scalar metric across batches.
    Handles division-by-zero safely by returning 0.0 when count is zero.

    Note:
        Not thread-safe. Intended for single-threaded use within training loops.

    Attributes:
        val: Most recent value
        avg: Running average
        sum: Cumulative sum
        count: Number of samples

    Example:
        tracker = MetricTracker()
        for batch in dataloader:
            loss = compute_loss(batch)
            tracker.update(loss.item(), n=batch_size)
        print(f"Average loss: {tracker.avg}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics to initial state."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: float = 0.0

    def update(self, val: float, n: int = 1):
        """
        Update tracker with new value(s).

        Args:
            val: New value (or mean of values if n > 1)
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"MetricTracker(val={self.val:.4f}, avg={self.avg:.4f}, count={self.count})"
        )


# ==============================================================================
# STATISTICAL METRICS
# ==============================================================================
def get_lr(optimizer) -> float:
    """
    Extract current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer instance

    Returns:
        Current learning rate (from first param group)
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def calc_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate average Pearson Correlation Coefficient across all targets.

    Handles edge cases where variance is near zero to avoid NaN values.
    This metric is important for physics-based signal regression papers.

    Args:
        y_true: Ground truth vectors ``(N,T)`` or fields ``(N,C,H,W)``
        y_pred: Predictions with the same shape as ``y_true``

    Returns:
        Mean Pearson correlation across all targets
    """
    y_true, y_pred = _validated_metric_arrays(y_true, y_pred)

    n_samples = y_true.shape[0]
    correlations = []
    for i in range(y_true.shape[1]):
        # Physical fields can have a large offset and small variation. Computing
        # the mean in float32 loses that variation and can turn anticorrelated
        # maps into a reported zero correlation. Promote one channel at a time.
        true_channel = np.asarray(y_true[:, i], dtype=np.float64)
        pred_channel = np.asarray(y_pred[:, i], dtype=np.float64)
        # Check for near-constant arrays to avoid NaN
        std_true = np.std(true_channel)
        std_pred = np.std(pred_channel)

        # Pearson is degenerate for < 3 samples (any 2 points are perfectly
        # collinear → a spurious ±1.0), so report 0.0 instead of a misleading 1.0.
        if n_samples < 3 or std_true == 0 or std_pred == 0:
            correlations.append(0.0)
        else:
            corr, _ = pearsonr(true_channel, pred_channel)
            # Handle NaN from pearsonr (shouldn't happen with std check, but safety)
            correlations.append(corr if not np.isnan(corr) else 0.0)

    return float(np.mean(correlations))


def calc_per_target_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate R² score for each target independently.

    Args:
        y_true: Ground truth vectors ``(N,T)`` or fields ``(N,C,H,W)``
        y_pred: Predictions with the same shape as ``y_true``

    Returns:
        Array of R² scores, one per target
    """
    y_true, y_pred = _validated_metric_arrays(y_true, y_pred)

    r2_scores = []
    for i in range(y_true.shape[1]):
        r2 = r2_score(
            np.asarray(y_true[:, i], dtype=np.float64),
            np.asarray(y_pred[:, i], dtype=np.float64),
        )
        r2_scores.append(r2)

    return np.array(r2_scores)


class StreamingFieldMetrics:
    """Accumulate dense-field statistics without retaining every map.

    Parallel centered-moment updates avoid cancellation when a channel has a
    large mean compared with its spatial variation. Statistics are computed
    per channel over samples and spatial positions, matching
    ``r2_score``/``calc_pearson`` applied to ``flatten_targets_by_channel``.

    Args:
        channels: Number of field channels ``C``.
        physical_scale: Optional per-channel factor applied to absolute
            errors (e.g. the target scaler's ``scale_``) when reporting MAE.
        track_relative_l2: Also accumulate per-sample relative L2 errors.
    """

    def __init__(
        self,
        channels: int,
        physical_scale: np.ndarray | None = None,
        *,
        track_relative_l2: bool = False,
    ):
        self.channels = channels
        self.physical_scale = (
            np.ones(channels) if physical_scale is None else np.asarray(physical_scale)
        )
        self.track_relative_l2 = track_relative_l2
        self.count = 0
        self.sample_count = 0
        self.mean_true = np.zeros(channels, dtype=np.float64)
        self.mean_pred = np.zeros(channels, dtype=np.float64)
        self.centered_true = np.zeros(channels, dtype=np.float64)
        self.centered_pred = np.zeros(channels, dtype=np.float64)
        self.centered_cross = np.zeros(channels, dtype=np.float64)
        self.sum_squared_error = np.zeros(channels, dtype=np.float64)
        self.sum_absolute_physical_error = np.zeros(channels, dtype=np.float64)
        self.sum_relative_l2 = np.zeros(channels, dtype=np.float64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        if target.shape != prediction.shape:
            raise ValueError("Targets and predictions must have matching shapes.")
        if self.track_relative_l2 and target.ndim > 2 and len(target):
            # Per-sample, per-channel relative L2 over the spatial axes.
            spatial_axes = tuple(range(2, target.ndim))
            target64 = target.astype(np.float64, copy=False)
            error64 = prediction.astype(np.float64, copy=False) - target64
            true_norm = np.sqrt(np.sum(np.square(target64), axis=spatial_axes))
            error_norm = np.sqrt(np.sum(np.square(error64), axis=spatial_axes))
            self.sum_relative_l2 += (error_norm / np.maximum(true_norm, 1e-12)).sum(
                axis=0
            )
            del target64, error64
        self.sample_count += len(target)

        target_flat = flatten_targets_by_channel(target).astype(np.float64, copy=False)
        prediction_flat = flatten_targets_by_channel(prediction).astype(
            np.float64, copy=False
        )
        if target_flat.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} field channel(s).")
        batch_count = len(target_flat)
        if batch_count == 0:
            return

        batch_true_mean = target_flat.mean(axis=0)
        batch_pred_mean = prediction_flat.mean(axis=0)
        true_centered = target_flat - batch_true_mean
        pred_centered = prediction_flat - batch_pred_mean
        delta_true = batch_true_mean - self.mean_true
        delta_pred = batch_pred_mean - self.mean_pred
        total_count = self.count + batch_count
        correction = self.count * batch_count / total_count
        self.centered_true += np.square(true_centered).sum(axis=0) + (
            np.square(delta_true) * correction
        )
        self.centered_pred += np.square(pred_centered).sum(axis=0) + (
            np.square(delta_pred) * correction
        )
        self.centered_cross += (true_centered * pred_centered).sum(axis=0) + (
            delta_true * delta_pred * correction
        )
        self.mean_true += delta_true * (batch_count / total_count)
        self.mean_pred += delta_pred * (batch_count / total_count)
        self.count = total_count

        error = prediction_flat - target_flat
        self.sum_squared_error += np.square(error).sum(axis=0)
        self.sum_absolute_physical_error += np.abs(
            error * self.physical_scale.reshape(-1)
        ).sum(axis=0)

    def per_channel(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-channel ``(r2, pearson)`` arrays (NaN if undefined)."""
        if self.count < 2:
            nan = np.full(self.channels, np.nan)
            return nan, nan.copy()
        # Match sklearn's finite convention for constant target channels:
        # perfect predictions score 1; any nonzero residual scores 0.
        r2_values = np.where(self.sum_squared_error == 0.0, 1.0, 0.0)
        variable = self.centered_true > 0.0
        r2_values[variable] = 1.0 - (
            self.sum_squared_error[variable] / self.centered_true[variable]
        )
        correlation_denominator = np.sqrt(self.centered_true) * np.sqrt(
            self.centered_pred
        )
        correlation_values = np.divide(
            self.centered_cross,
            correlation_denominator,
            out=np.zeros(self.channels, dtype=np.float64),
            where=correlation_denominator > 0.0,
        )
        return r2_values, np.clip(correlation_values, -1.0, 1.0)

    def compute(self) -> tuple[float, float, np.ndarray]:
        """Return mean ``R²``, mean Pearson, and per-channel physical MAE."""
        if self.count == 0:
            return float("nan"), float("nan"), np.full(self.channels, np.nan)
        mae_per_channel = self.sum_absolute_physical_error / self.count
        if self.count < 2:
            return float("nan"), float("nan"), mae_per_channel
        r2_values, correlation_values = self.per_channel()
        return (
            float(np.mean(r2_values)),
            float(np.mean(correlation_values)),
            mae_per_channel,
        )


# ==============================================================================
# VISUALIZATION HELPERS (internal)
# ==============================================================================
def _validated_metric_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired regression data before reporting numerical results."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Prediction shape {y_pred.shape} does not match target shape {y_true.shape}."
        )
    for array in (y_true, y_pred):
        if (
            array.size == 0
            or array.dtype.kind not in "biuf"
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "Metrics require non-empty arrays containing finite real values."
            )
    return flatten_targets_by_channel(y_true), flatten_targets_by_channel(y_pred)


def _prepare_plot_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Prepare data for plotting: reshape, sample, and generate param names."""
    y_true, y_pred = _validated_metric_arrays(y_true, y_pred)
    # Compute plotting statistics in double precision, matching reported metrics.
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    num_params = y_true.shape[1]

    # Subsample if needed. Use a seeded local generator so figures are
    # reproducible across runs and the global numpy RNG state is left untouched.
    if max_samples and len(y_true) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]

    # Generate default param names if needed
    if param_names is None or len(param_names) != num_params:
        param_names = [f"P{i}" for i in range(num_params)]

    return y_true, y_pred, param_names, num_params


def _create_subplot_grid(
    num_params: int,
    height_ratio: float = 1.0,
    max_cols: int = 4,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a subplot grid for multi-parameter plots."""
    cols = min(num_params, max_cols)
    rows = (num_params + cols - 1) // cols
    subplot_size = FIGURE_WIDTH_INCH / cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows * height_ratio),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    return fig, axes


def _add_unified_legend(
    fig: plt.Figure,
    axes: np.ndarray,
    ncol: int = 2,
    y_offset: float = -0.13,
    bottom_margin: float = 0.22,
) -> None:
    """Add a unified legend below the figure."""
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        fontsize=FONT_SIZE_TEXT,
        fancybox=False,
        framealpha=1.0,
        bbox_to_anchor=(0.5, y_offset),
    )
    fig.subplots_adjust(bottom=bottom_margin)


def _hide_unused_subplots(axes: np.ndarray, num_used: int) -> None:
    """Hide unused subplots in a grid."""
    for i in range(num_used, len(axes)):
        axes[i].axis("off")


# ==============================================================================
# VISUALIZATION - SCATTER PLOTS
# ==============================================================================
def plot_scientific_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    max_samples: int = 2000,
) -> plt.Figure:
    """
    Generate publication-quality scatter plots comparing predictions to ground truth.

    Creates a grid of scatter plots, one for each output target, with R² annotations
    and ideal diagonal reference lines.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for titles
        max_samples: Maximum samples to plot (downsamples if exceeded)

    Returns:
        Matplotlib Figure object (can be saved or logged to WandB)
    """
    _ensure_style_configured()
    # Compute R² on the FULL dataset BEFORE subsampling so the annotation matches
    # the reported (full-dataset) metric rather than a random 2000-point subset.
    full_r2 = calc_per_target_r2(y_true, y_pred)
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names, max_samples
    )
    fig, axes = _create_subplot_grid(num_params, height_ratio=1.0)

    for i in range(num_params):
        ax = axes[i]

        # R² computed on the full dataset above (not the plotted subsample)
        r2 = full_r2[i] if i < len(full_r2) else float("nan")

        # Scatter plot (using plot for vector SVG compatibility)
        ax.plot(
            y_true[:, i],
            y_pred[:, i],
            "o",
            markersize=5,
            markerfacecolor=COLORS["scatter"],
            markeredgecolor="none",
            label="Data",
        )

        # Ideal diagonal line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        margin = (max_val - min_val) * 0.05 or max(abs(min_val) * 0.05, 0.05)
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            "--",
            color=COLORS["error"],
            lw=1.2,
            label="Ideal",
        )

        # Labels and formatting
        ax.set_title(f"{param_names[i]}\n$R^2 = {r2:.4f}$")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.legend(fontsize=FONT_SIZE_TICKS, loc="best", fancybox=False, framealpha=1.0)

    _hide_unused_subplots(axes, num_params)
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - ERROR HISTOGRAM
# ==============================================================================
def plot_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    bins: int = 50,
) -> plt.Figure:
    """
    Generate publication-quality error distribution histograms.

    Shows the distribution of prediction errors (y_pred - y_true) for each target.
    Includes mean, std, and MAE annotations.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for titles
        bins: Number of histogram bins

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names
    )
    errors = y_pred - y_true
    fig, axes = _create_subplot_grid(num_params)

    for i in range(num_params):
        ax = axes[i]
        err = errors[:, i]
        mean_err, std_err, mae = np.mean(err), np.std(err), np.mean(np.abs(err))

        # Histogram - only label first subplot
        ax.hist(
            err,
            bins=bins,
            color=COLORS["primary"],
            edgecolor="white",
            linewidth=0.5,
            label="Errors" if i == 0 else None,
        )
        ax.axvline(
            x=0,
            color=COLORS["error"],
            linestyle="--",
            lw=1.2,
            label="Zero" if i == 0 else None,
        )
        ax.axvline(
            x=mean_err,
            color=COLORS["accent"],
            linestyle="-",
            lw=1.2,
            label="Mean" if i == 0 else None,
        )

        ax.set_title(f"{param_names[i]}\nMAE = {mae:.4f}, $\\sigma$ = {std_err:.4f}")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y")

    _hide_unused_subplots(axes, num_params)
    plt.tight_layout()
    _add_unified_legend(fig, axes, ncol=3)
    return fig


# ==============================================================================
# VISUALIZATION - RESIDUAL PLOT
# ==============================================================================
def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    max_samples: int = 2000,
) -> plt.Figure:
    """
    Generate publication-quality residual plots.

    Shows residuals (y_pred - y_true) vs predicted values. Useful for detecting
    systematic bias or heteroscedasticity in predictions.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for titles
        max_samples: Maximum samples to plot

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names, max_samples
    )
    residuals = y_pred - y_true
    fig, axes = _create_subplot_grid(num_params)

    for i in range(num_params):
        ax = axes[i]
        ax.plot(
            y_pred[:, i],
            residuals[:, i],
            "o",
            markersize=5,
            markerfacecolor=COLORS["scatter"],
            markeredgecolor="none",
            label="Data" if i == 0 else None,
        )
        ax.axhline(
            y=0,
            color=COLORS["error"],
            linestyle="--",
            lw=1.2,
            label="Zero" if i == 0 else None,
        )
        mean_res = np.mean(residuals[:, i])
        ax.axhline(
            y=mean_res,
            color=COLORS["accent"],
            linestyle="-",
            lw=1.0,
            label="Mean" if i == 0 else None,
        )

        ax.set_title(f"{param_names[i]}")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Pred - True")
        ax.grid(True)

    _hide_unused_subplots(axes, num_params)
    plt.tight_layout()
    _add_unified_legend(fig, axes, ncol=3)
    return fig


# ==============================================================================
# VISUALIZATION - TRAINING CURVES
# ==============================================================================
def create_training_curves(
    history: list[dict[str, Any]],
    metrics: list[str] | None = None,
    show_lr: bool = True,
) -> plt.Figure:
    """
    Create training curve visualization from history with optional learning rate.

    Plots training and validation loss over epochs. If learning rate data is
    available in history, it's plotted on a secondary y-axis.

    Args:
        history: List of epoch statistics dictionaries. Each dict should contain
                 'epoch', 'train_loss', 'val_loss', and optionally 'lr'.
        metrics: Metric names to plot on primary y-axis
        show_lr: If True and 'lr' is in history, show learning rate on secondary axis

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()

    if metrics is None:
        metrics = ["train_loss", "val_loss"]

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]

    fig, ax1 = plt.subplots(figsize=(FIGURE_WIDTH_INCH * 0.7, FIGURE_WIDTH_INCH * 0.4))

    colors = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["accent"],
        COLORS["neutral"],
    ]

    # Plot metrics on primary axis
    lines = []
    for idx, metric in enumerate(metrics):
        values = [h.get(metric, np.nan) for h in history]
        color = colors[idx % len(colors)]
        (line,) = ax1.plot(
            epochs,
            values,
            label=metric.replace("_", " ").title(),
            linewidth=1.5,
            color=color,
        )
        lines.append(line)

    def set_lr_ticks(ax: plt.Axes, data: list[float], n_ticks: int = 4) -> None:
        """Set n uniformly spaced ticks on LR axis with 10^n format labels."""
        valid_data = [v for v in data if v is not None and np.isfinite(v) and v > 0]
        if not valid_data:
            return
        vmin, vmax = min(valid_data), max(valid_data)
        # Snap to clean decade boundaries
        log_min = np.floor(np.log10(vmin))
        log_max = np.ceil(np.log10(vmax))
        # Generate n uniformly spaced ticks as powers of 10
        log_ticks = np.linspace(log_min, log_max, n_ticks)
        # Round to nearest integer power of 10 for clean numbers
        log_ticks = np.round(log_ticks)
        ticks = 10.0**log_ticks
        # Remove duplicates while preserving order
        ticks = list(dict.fromkeys(ticks))
        ax.set_yticks(ticks)
        # Format all tick labels as 10^n
        labels = [f"$10^{{{int(np.log10(t))}}}$" for t in ticks]
        ax.set_yticklabels(labels)
        ax.minorticks_off()

    def set_loss_ticks(ax: plt.Axes, data: list[float]) -> None:
        """Set ticks at powers of 10 that cover the data range."""
        valid_data = [v for v in data if v is not None and np.isfinite(v) and v > 0]
        if not valid_data:
            return
        vmin, vmax = min(valid_data), max(valid_data)
        # Get decade range that covers data (floor for min to include lowest data)
        log_min = int(np.floor(np.log10(vmin)))
        log_max = int(np.ceil(np.log10(vmax)))
        # Generate ticks at each power of 10
        ticks = [10.0**i for i in range(log_min, log_max + 1)]
        ax.set_yticks(ticks)
        # Format labels as 10^n
        labels = [f"$10^{{{i}}}$" for i in range(log_min, log_max + 1)]
        ax.set_yticklabels(labels)
        ax.minorticks_off()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")  # Log scale for loss
    ax1.grid(True)

    # Collect all loss values and set clean power of 10 ticks
    all_loss_values = []
    for metric in metrics:
        all_loss_values.extend([h.get(metric, np.nan) for h in history])
    set_loss_ticks(ax1, all_loss_values)

    # Check if learning rate data exists
    has_lr = show_lr and any("lr" in h for h in history)

    if has_lr:
        # Create secondary y-axis for learning rate
        ax2 = ax1.twinx()
        lr_values = [h.get("lr", np.nan) for h in history]
        (line_lr,) = ax2.plot(
            epochs,
            lr_values,
            "--",
            color=COLORS["neutral"],
            linewidth=1.0,
            label="Learning Rate",
        )
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")  # Log scale for LR
        set_lr_ticks(ax2, lr_values, n_ticks=4)
        # Ensure right spine (axis line) is visible
        ax2.spines["right"].set_visible(True)
        lines.append(line_lr)

    # Combined legend
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="best",
        fontsize=FONT_SIZE_TICKS,
        fancybox=False,
        framealpha=1.0,
    )

    ax1.set_title("Training Curves")

    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - BLAND-ALTMAN PLOT
# ==============================================================================
def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    max_samples: int = 2000,
) -> plt.Figure:
    """
    Generate Bland-Altman plots for method comparison.

    Plots the difference between predictions and ground truth against their mean.
    Includes mean difference line and ±1.96*SD limits of agreement.
    Standard for validating agreement in medical/scientific papers.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for titles
        max_samples: Maximum samples to plot

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names, max_samples
    )
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    fig, axes = _create_subplot_grid(num_params)

    for i in range(num_params):
        ax = axes[i]
        mean_diff = np.mean(diff_vals[:, i])
        std_diff = np.std(diff_vals[:, i])
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        ax.plot(
            mean_vals[:, i],
            diff_vals[:, i],
            "o",
            markersize=5,
            markerfacecolor=COLORS["scatter"],
            markeredgecolor="none",
            label="Data" if i == 0 else None,
        )
        ax.axhline(
            y=mean_diff,
            color=COLORS["accent"],
            linestyle="-",
            lw=1.2,
            label="Mean" if i == 0 else None,
        )
        ax.axhline(
            y=upper_loa,
            color=COLORS["error"],
            linestyle="--",
            lw=1.0,
            label=r"$\pm$1.96 SD" if i == 0 else None,
        )
        ax.axhline(y=lower_loa, color=COLORS["error"], linestyle="--", lw=1.0)
        ax.axhline(y=0, color="gray", linestyle=":", lw=0.8)

        ax.set_title(f"{param_names[i]}")
        ax.set_xlabel("Mean of True and Pred")
        ax.set_ylabel("Pred - True")
        ax.grid(True)

    _hide_unused_subplots(axes, num_params)
    plt.tight_layout()
    _add_unified_legend(fig, axes, ncol=3)
    return fig


# ==============================================================================
# VISUALIZATION - QQ PLOT
# ==============================================================================
def plot_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
) -> plt.Figure:
    """
    Generate Q-Q plots to check if prediction errors are normally distributed.

    Compares the quantiles of the error distribution to a theoretical normal
    distribution. Points on the diagonal indicate normally distributed errors.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for titles

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    from scipy import stats

    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names
    )

    # Calculate errors
    errors = y_pred - y_true

    fig, axes = _create_subplot_grid(num_params)

    for i in range(num_params):
        ax = axes[i]

        # Standardize errors for QQ plot
        err = errors[:, i]
        std_err = np.std(err)

        # Guard against zero variance (constant errors)
        if std_err < 1e-10:
            title = param_names[i]
            ax.text(
                0.5,
                0.5,
                "Zero variance\n(constant errors)",
                ha="center",
                va="center",
                fontsize=FONT_SIZE_TEXT,
                transform=ax.transAxes,
            )
            ax.set_title(f"{title}\n(zero variance)")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            continue

        standardized = (err - np.mean(err)) / std_err

        # Calculate theoretical quantiles and sample quantiles
        (osm, osr), (slope, intercept, r) = stats.probplot(standardized, dist="norm")

        # Scatter plot (using plot for vector SVG compatibility)
        ax.plot(
            osm,
            osr,
            "o",
            markersize=5,
            markerfacecolor=COLORS["scatter"],
            markeredgecolor="none",
            label="Data",
        )

        # Reference line
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, "--", color=COLORS["error"], lw=1.2, label="Normal")

        # Labels
        title = param_names[i]
        ax.set_title(f"{title}\n$R^2 = {r**2:.4f}$")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.grid(True)
        ax.legend(fontsize=FONT_SIZE_TICKS, loc="best", fancybox=False, framealpha=1.0)

    _hide_unused_subplots(axes, num_params)
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - CORRELATION HEATMAP
# ==============================================================================
def plot_correlation_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
) -> plt.Figure:
    """
    Generate correlation heatmap between predicted parameters.

    Shows the Pearson correlation between different output parameters,
    useful for understanding multi-output prediction relationships.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for labels

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()

    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names
    )

    if num_params < 2:
        # Need at least 2 params for correlation
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(
            0.5,
            0.5,
            "Correlation heatmap requires\nat least 2 parameters",
            ha="center",
            va="center",
            fontsize=FONT_SIZE_TEXT,
        )
        ax.axis("off")
        return fig

    # Calculate prediction error correlations.
    # Guard zero-variance error columns (a target predicted as a near-constant):
    # np.corrcoef divides by per-column std and would fill that row/col with NaN,
    # producing 'nan' annotations and 'bad' colors. Neutralize NaN to 0 (diag 1).
    errors = y_pred - y_true
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.corrcoef(errors.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # Create figure
    fig_size = min(FIGURE_WIDTH_INCH * 0.6, 2 + num_params * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Heatmap using Rectangle patches (vector-compatible, no imshow)
    cmap = plt.colormaps["RdBu_r"]
    norm = Normalize(vmin=-1, vmax=1)

    for i in range(num_params):
        for j in range(num_params):
            color = cmap(norm(corr_matrix[i, j]))
            rect = Rectangle(
                (j - 0.5, i - 0.5),  # bottom-left corner
                1,
                1,  # width, height
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    # Set axis limits and aspect
    ax.set_xlim(-0.5, num_params - 0.5)
    ax.set_ylim(num_params - 0.5, -0.5)  # Invert y-axis for matrix orientation
    ax.set_aspect("equal")

    # Vector colorbar using rectangles (no raster gradient)
    # Create a separate axes for colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Draw colorbar as discrete rectangles (20 segments for smooth gradient)
    n_segments = 20
    for k in range(n_segments):
        val = -1 + 2 * k / (n_segments - 1)  # -1 to 1
        color = cmap(norm(val))
        rect = Rectangle(
            (0, val - 1 / n_segments),
            1,
            2 / n_segments,
            facecolor=color,
            edgecolor="none",
        )
        cax.add_patch(rect)

    cax.set_xlim(0, 1)
    cax.set_ylim(-1, 1)
    cax.set_xticks([])
    cax.set_ylabel("Correlation", fontsize=FONT_SIZE_TEXT)
    cax.yaxis.set_label_position("right")
    cax.yaxis.tick_right()

    # Labels
    ax.set_xticks(range(num_params))
    ax.set_yticks(range(num_params))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticklabels(param_names)

    # Annotate with values
    for i in range(num_params):
        for j in range(num_params):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=FONT_SIZE_TICKS,
            )

    ax.set_title("Error Correlation Matrix")

    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - RELATIVE ERROR PLOT
# ==============================================================================
def plot_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    max_samples: int = 2000,
) -> plt.Figure:
    """
    Generate relative error plots (percentage error vs true value).

    Useful for detecting scale-dependent bias where errors increase
    proportionally with the magnitude of the true value.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for titles
        max_samples: Maximum samples to plot

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names, max_samples
    )

    # Calculate relative error (avoid division by zero)
    # Points where y_true ≈ 0 produce inf/nan — we keep them as NaN and
    # exclude from plotting/statistics rather than mapping to 0% (which
    # would understate error for near-zero targets).
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.abs((y_pred - y_true) / y_true) * 100
        rel_error = np.nan_to_num(rel_error, nan=np.nan, posinf=np.nan, neginf=np.nan)

    fig, axes = _create_subplot_grid(num_params)

    for i in range(num_params):
        ax = axes[i]
        # Filter out NaN (near-zero true values where relative error is undefined)
        mask = np.isfinite(rel_error[:, i])
        ax.plot(
            y_true[mask, i],
            rel_error[mask, i],
            "o",
            markersize=5,
            markerfacecolor=COLORS["scatter"],
            markeredgecolor="none",
            label="Data",
        )
        if mask.any():
            mean_rel = rel_error[mask, i].mean()
            ax.axhline(
                y=mean_rel, color=COLORS["accent"], linestyle="-", lw=1.2, label="Mean"
            )

        ax.set_title(f"{param_names[i]}")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Relative Error (\\%)")
        ax.grid(True)
        ax.legend(fontsize=FONT_SIZE_TICKS, loc="best", fancybox=False, framealpha=1.0)

    _hide_unused_subplots(axes, num_params)
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - CUMULATIVE ERROR DISTRIBUTION (CDF)
# ==============================================================================
def plot_error_cdf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    use_relative: bool = True,
) -> plt.Figure:
    """
    Generate cumulative distribution function (CDF) of prediction errors.

    Shows what percentage of predictions fall within a given error bound.
    Very useful for reporting: "95% of predictions have error < X%"

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for legend
        use_relative: If True, plot relative error (%), else absolute error

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names
    )

    # Calculate errors
    if use_relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.abs((y_pred - y_true) / y_true) * 100
            # Keep inf/nan as NaN — exclude from CDF rather than understating
            errors = np.nan_to_num(errors, nan=np.nan, posinf=np.nan, neginf=np.nan)
        xlabel = r"Relative Error\;$(\%)$"
    else:
        errors = np.abs(y_pred - y_true)
        xlabel = "Absolute Error"

    # Create figure
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCH * 0.6, FIGURE_WIDTH_INCH * 0.4))

    colors_list = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["accent"],
        COLORS["success"],
        COLORS["neutral"],
    ]

    for i in range(num_params):
        # Filter NaN values (near-zero targets where relative error is undefined)
        err_raw = errors[:, i]
        err = np.sort(err_raw[np.isfinite(err_raw)])
        if len(err) == 0:
            continue
        cdf = np.arange(1, len(err) + 1) / len(err)

        color = colors_list[i % len(colors_list)]
        ax.plot(err, cdf * 100, label=param_names[i], color=color, lw=1.5)

        # Find 95th percentile (use filtered err to avoid NaN from near-zero targets)
        p95_val = np.percentile(err, 95)
        ax.axvline(x=p95_val, color=color, linestyle=":")

    # Reference lines
    ax.axhline(y=95, color="gray", linestyle="--", lw=0.8, label=r"95\%")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Cumulative Percentage\;$(\%)$")
    ax.set_title("Cumulative Error Distribution")
    ax.legend(fontsize=FONT_SIZE_TICKS, loc="best", fancybox=False, framealpha=1.0)
    ax.grid(True)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - PREDICTION VS SAMPLE INDEX
# ==============================================================================
def plot_prediction_vs_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    max_samples: int = 500,
) -> plt.Figure:
    """
    Generate prediction vs sample index plots.

    Shows true and predicted values for each sample in sequence.
    Useful for time-series style visualization and spotting outliers.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for titles
        max_samples: Maximum samples to show

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    # This plot promises sample order. Random subsampling would display shuffled
    # samples under fabricated sequential indices and hide time-dependent errors.
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names
    )
    if max_samples is not None:
        y_true, y_pred = y_true[:max_samples], y_pred[:max_samples]
    indices = np.arange(len(y_true))
    fig, axes = _create_subplot_grid(num_params)

    for i in range(num_params):
        ax = axes[i]
        ax.plot(
            indices,
            y_true[:, i],
            "o",
            markersize=5,
            color=COLORS["primary"],
            label="True" if i == 0 else None,
        )
        ax.plot(
            indices,
            y_pred[:, i],
            "x",
            markersize=5,
            color=COLORS["error"],
            label="Predicted" if i == 0 else None,
        )

        ax.set_title(f"{param_names[i]}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(True)

    _hide_unused_subplots(axes, num_params)
    plt.tight_layout()
    _add_unified_legend(fig, axes, ncol=2)
    return fig


# ==============================================================================
# VISUALIZATION - ERROR BOX PLOT
# ==============================================================================
def plot_error_boxplot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list[str] | None = None,
    use_relative: bool = False,
) -> plt.Figure:
    """
    Generate box plots comparing error distributions across parameters.

    Provides a compact view of error statistics (median, quartiles, outliers)
    for all parameters side-by-side.

    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        param_names: Optional list of parameter names for x-axis
        use_relative: If True, plot relative error (%), else absolute error

    Returns:
        Matplotlib Figure object
    """
    _ensure_style_configured()
    y_true, y_pred, param_names, num_params = _prepare_plot_data(
        y_true, y_pred, param_names
    )

    # Calculate errors
    if use_relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.abs((y_pred - y_true) / y_true) * 100
            # Keep inf/nan as NaN — exclude from boxplot rather than understating
            errors = np.nan_to_num(errors, nan=np.nan, posinf=np.nan, neginf=np.nan)
        ylabel = (
            r"Relative Error (\%)"
            if plt.rcParams["text.usetex"]
            else "Relative Error (%)"
        )
    else:
        errors = y_pred - y_true  # Signed error for box plot
        ylabel = "Prediction Error"

    # Create figure
    fig_width = min(FIGURE_WIDTH_INCH * 0.5, 2 + num_params * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_WIDTH_INCH * 0.4))

    # Box plot (filter NaN for each parameter — near-zero targets excluded)
    box_data = [errors[:, i][np.isfinite(errors[:, i])] for i in range(num_params)]
    try:
        bp = ax.boxplot(
            box_data,
            tick_labels=param_names,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
    except TypeError:
        bp = ax.boxplot(
            box_data,
            labels=param_names,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )

    # Color the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.7)

    # Zero line for signed errors
    if not use_relative:
        ax.axhline(y=0, color=COLORS["error"], linestyle="--", lw=1.0)

    ax.set_ylabel(ylabel)
    ax.set_title("Error Distribution by Parameter")
    ax.grid(True, axis="y")

    # Rotate labels if needed
    if num_params > 4:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig
