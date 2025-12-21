"""
Scientific Metrics and Visualization Utilities
===============================================

Provides metric tracking, statistical calculations, and publication-quality
visualization tools for deep learning experiments.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.1.0
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ==============================================================================
# PUBLICATION-QUALITY PLOT CONFIGURATION
# ==============================================================================
# Width: 19 cm = 7.48 inches (for two-column journals)
FIGURE_WIDTH_CM = 19
FIGURE_WIDTH_INCH = FIGURE_WIDTH_CM / 2.54

# Font sizes (consistent 8pt for publication)
FONT_SIZE_TEXT = 8
FONT_SIZE_TICKS = 8
FONT_SIZE_TITLE = 9

# DPI for publication (300 for print, 150 for screen)
FIGURE_DPI = 300

# Color palette (accessible, print-friendly)
COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Raspberry
    'accent': '#F18F01',       # Orange
    'neutral': '#6B717E',      # Slate gray
    'error': '#C73E1D',        # Red
    'success': '#3A7D44',      # Green
}


def configure_matplotlib_style():
    """Configure matplotlib for publication-quality LaTeX-style plots."""
    plt.rcParams.update({
        # LaTeX-style fonts
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',  # Computer Modern for math
        
        # Font sizes
        'font.size': FONT_SIZE_TEXT,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_TEXT,
        'xtick.labelsize': FONT_SIZE_TICKS,
        'ytick.labelsize': FONT_SIZE_TICKS,
        'legend.fontsize': FONT_SIZE_TICKS,
        
        # Line widths
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        
        # Grid style
        'grid.alpha': 0.4,
        'grid.linestyle': ':',
        
        # Figure settings
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Remove top/right spines for cleaner look
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# Apply style on import
configure_matplotlib_style()


# ==============================================================================
# METRIC TRACKING
# ==============================================================================
class MetricTracker:
    """
    Tracks running averages of metrics with thread-safe accumulation.
    
    Useful for tracking loss, accuracy, or any scalar metric across batches.
    Handles division-by-zero safely by returning 0.0 when count is zero.
    
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
        return f"MetricTracker(val={self.val:.4f}, avg={self.avg:.4f}, count={self.count})"


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
        return param_group['lr']
    return 0.0


def calc_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate average Pearson Correlation Coefficient across all targets.
    
    Handles edge cases where variance is near zero to avoid NaN values.
    This metric is important for physics-based signal regression papers.
    
    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        
    Returns:
        Mean Pearson correlation across all targets
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    correlations = []
    for i in range(y_true.shape[1]):
        # Check for near-constant arrays to avoid NaN
        std_true = np.std(y_true[:, i])
        std_pred = np.std(y_pred[:, i])
        
        if std_true < 1e-9 or std_pred < 1e-9:
            correlations.append(0.0)
        else:
            corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            # Handle NaN from pearsonr (shouldn't happen with std check, but safety)
            correlations.append(corr if not np.isnan(corr) else 0.0)
    
    return float(np.mean(correlations))


def calc_per_target_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate R² score for each target independently.
    
    Args:
        y_true: Ground truth values of shape (N, num_targets)
        y_pred: Predicted values of shape (N, num_targets)
        
    Returns:
        Array of R² scores, one per target
    """
    if y_true.ndim == 1:
        return np.array([r2_score(y_true, y_pred)])
    
    r2_scores = []
    for i in range(y_true.shape[1]):
        try:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            r2_scores.append(r2)
        except ValueError:
            r2_scores.append(0.0)
    
    return np.array(r2_scores)


# ==============================================================================
# VISUALIZATION - SCATTER PLOTS
# ==============================================================================
def plot_scientific_scatter(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Downsample for visualization performance
    if len(y_true) > max_samples:
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    # Calculate grid dimensions
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    # Calculate figure size (19 cm width)
    subplot_size = FIGURE_WIDTH_INCH / cols
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        # Calculate R² for this target
        if len(y_true) >= 2:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
        else:
            r2 = float('nan')
        
        # Scatter plot
        ax.scatter(
            y_true[:, i], y_pred[:, i],
            alpha=0.5, 
            s=15, 
            c=COLORS['primary'], 
            edgecolors='none',
            rasterized=True,
            label='Data'
        )
        
        # Ideal diagonal line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], 
                [min_val - margin, max_val + margin], 
                '--', color=COLORS['error'], lw=1.2, alpha=0.8, label='Ideal')
        
        # Labels and formatting
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}\n$R^2 = {r2:.4f}$")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.legend(fontsize=6, loc='best')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - ERROR HISTOGRAM
# ==============================================================================
def plot_error_histogram(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate errors
    errors = y_pred - y_true
    
    # Calculate grid dimensions
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    # Calculate figure size
    subplot_size = FIGURE_WIDTH_INCH / cols
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows * 0.8),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        err = errors[:, i]
        
        # Statistics
        mean_err = np.mean(err)
        std_err = np.std(err)
        mae = np.mean(np.abs(err))
        
        # Histogram
        ax.hist(err, bins=bins, color=COLORS['primary'], alpha=0.7, 
                edgecolor='white', linewidth=0.5, label='Errors')
        
        # Vertical line at zero
        ax.axvline(x=0, color=COLORS['error'], linestyle='--', lw=1.2, alpha=0.8,
                   label='Zero')
        
        # Mean line
        ax.axvline(x=mean_err, color=COLORS['accent'], linestyle='-', lw=1.2, 
                   label=f'Mean = {mean_err:.4f}')
        
        # Labels and formatting
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}\nMAE = {mae:.4f}, $\\sigma$ = {std_err:.4f}")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")
        ax.grid(True, axis='y')
        ax.legend(fontsize=6, loc='best')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - RESIDUAL PLOT
# ==============================================================================
def plot_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate residuals
    residuals = y_pred - y_true
    
    # Downsample for visualization
    if len(y_true) > max_samples:
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_pred = y_pred[indices]
        residuals = residuals[indices]
    
    # Calculate grid dimensions
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    # Calculate figure size
    subplot_size = FIGURE_WIDTH_INCH / cols
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows * 0.8),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        # Scatter plot of residuals
        ax.scatter(
            y_pred[:, i], residuals[:, i],
            alpha=0.5, 
            s=15, 
            c=COLORS['primary'], 
            edgecolors='none',
            rasterized=True,
            label='Data'
        )
        
        # Zero line
        ax.axhline(y=0, color=COLORS['error'], linestyle='--', lw=1.2, alpha=0.8,
                   label='Zero')
        
        # Calculate and show mean residual
        mean_res = np.mean(residuals[:, i])
        ax.axhline(y=mean_res, color=COLORS['accent'], linestyle='-', lw=1.0, alpha=0.7,
                   label=f'Mean = {mean_res:.4f}')
        
        # Labels
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residual (Pred - True)")
        ax.grid(True)
        ax.legend(fontsize=6, loc='best')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - TRAINING CURVES
# ==============================================================================
def create_training_curves(
    history: List[Dict[str, Any]],
    metrics: List[str] = ['train_loss', 'val_loss'],
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
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    
    fig, ax1 = plt.subplots(figsize=(FIGURE_WIDTH_INCH * 0.7, FIGURE_WIDTH_INCH * 0.4))
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['neutral']]
    
    # Plot metrics on primary axis
    lines = []
    for idx, metric in enumerate(metrics):
        values = [h.get(metric, np.nan) for h in history]
        color = colors[idx % len(colors)]
        line, = ax1.plot(epochs, values, label=metric.replace('_', ' ').title(), 
                         linewidth=1.5, color=color)
        lines.append(line)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')  # Log scale for loss
    ax1.grid(True, alpha=0.3)
    
    # Check if learning rate data exists
    has_lr = show_lr and any('lr' in h for h in history)
    
    if has_lr:
        # Create secondary y-axis for learning rate
        ax2 = ax1.twinx()
        lr_values = [h.get('lr', np.nan) for h in history]
        line_lr, = ax2.plot(epochs, lr_values, '--', color=COLORS['neutral'], 
                            linewidth=1.0, alpha=0.7, label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color=COLORS['neutral'])
        ax2.tick_params(axis='y', labelcolor=COLORS['neutral'])
        ax2.set_yscale('log')  # Log scale for LR
        lines.append(line_lr)
    
    # Combined legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=6)
    
    ax1.set_title('Training Curves')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - BLAND-ALTMAN PLOT
# ==============================================================================
def plot_bland_altman(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate mean and difference
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    
    # Downsample for visualization
    if len(y_true) > max_samples:
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        mean_vals = mean_vals[indices]
        diff_vals = diff_vals[indices]
    
    # Calculate grid dimensions
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    # Calculate figure size
    subplot_size = FIGURE_WIDTH_INCH / cols
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows * 0.8),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        mean_diff = np.mean(diff_vals[:, i])
        std_diff = np.std(diff_vals[:, i])
        
        # Limits of agreement (95% CI = mean ± 1.96*SD)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff
        
        # Scatter plot
        ax.scatter(
            mean_vals[:, i], diff_vals[:, i],
            alpha=0.5, s=15, c=COLORS['primary'], 
            edgecolors='none', rasterized=True
        )
        
        # Mean difference line
        ax.axhline(y=mean_diff, color=COLORS['accent'], linestyle='-', lw=1.2,
                   label=f'Mean = {mean_diff:.3f}')
        
        # Limits of agreement
        ax.axhline(y=upper_loa, color=COLORS['error'], linestyle='--', lw=1.0,
                   label=f'+1.96 SD = {upper_loa:.3f}')
        ax.axhline(y=lower_loa, color=COLORS['error'], linestyle='--', lw=1.0,
                   label=f'-1.96 SD = {lower_loa:.3f}')
        
        # Zero line
        ax.axhline(y=0, color='gray', linestyle=':', lw=0.8, alpha=0.5)
        
        # Labels
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}")
        ax.set_xlabel("Mean of True and Predicted")
        ax.set_ylabel("Difference (Pred - True)")
        ax.grid(True)
        ax.legend(fontsize=6, loc='best')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - QQ PLOT
# ==============================================================================
def plot_qq(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    from scipy import stats
    
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate errors
    errors = y_pred - y_true
    
    # Calculate grid dimensions
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    # Calculate figure size
    subplot_size = FIGURE_WIDTH_INCH / cols
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        # Standardize errors for QQ plot
        err = errors[:, i]
        standardized = (err - np.mean(err)) / np.std(err)
        
        # Calculate theoretical quantiles and sample quantiles
        (osm, osr), (slope, intercept, r) = stats.probplot(standardized, dist="norm")
        
        # Scatter plot
        ax.scatter(osm, osr, alpha=0.5, s=15, c=COLORS['primary'], 
                   edgecolors='none', rasterized=True, label='Data')
        
        # Reference line
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, '--', color=COLORS['error'], lw=1.2, label='Normal')
        
        # Labels
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}\n$R^2 = {r**2:.4f}$")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.grid(True)
        ax.legend(fontsize=6, loc='best')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - CORRELATION HEATMAP
# ==============================================================================
def plot_correlation_heatmap(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    if num_params < 2:
        # Need at least 2 params for correlation
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Correlation heatmap requires\nat least 2 parameters",
                ha='center', va='center', fontsize=10)
        ax.axis('off')
        return fig
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    if param_names is None or len(param_names) != num_params:
        param_names = [f"P{i}" for i in range(num_params)]
    
    # Calculate prediction error correlations
    errors = y_pred - y_true
    corr_matrix = np.corrcoef(errors.T)
    
    # Create figure
    fig_size = min(FIGURE_WIDTH_INCH * 0.6, 2 + num_params * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=FONT_SIZE_TICKS)
    
    # Labels
    ax.set_xticks(range(num_params))
    ax.set_yticks(range(num_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Annotate with values
    for i in range(num_params):
        for j in range(num_params):
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                    color=color, fontsize=FONT_SIZE_TICKS)
    
    ax.set_title('Error Correlation Matrix')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - RELATIVE ERROR PLOT
# ==============================================================================
def plot_relative_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Downsample for visualization
    if len(y_true) > max_samples:
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    # Calculate relative error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs((y_pred - y_true) / y_true) * 100
        rel_error = np.nan_to_num(rel_error, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate grid dimensions
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    # Calculate figure size
    subplot_size = FIGURE_WIDTH_INCH / cols
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows * 0.8),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(
            y_true[:, i], rel_error[:, i],
            alpha=0.5, s=15, c=COLORS['primary'], 
            edgecolors='none', rasterized=True, label='Data'
        )
        
        # Mean relative error line
        mean_rel = np.mean(rel_error[:, i])
        ax.axhline(y=mean_rel, color=COLORS['accent'], linestyle='-', lw=1.2,
                   label=f'Mean = {mean_rel:.2f}%')
        
        # Labels
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Relative Error (%)")
        ax.grid(True)
        ax.legend(fontsize=6, loc='best')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - CUMULATIVE ERROR DISTRIBUTION (CDF)
# ==============================================================================
def plot_error_cdf(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    if param_names is None or len(param_names) != num_params:
        param_names = [f"P{i}" for i in range(num_params)]
    
    # Calculate errors
    if use_relative:
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.abs((y_pred - y_true) / y_true) * 100
            errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
        xlabel = "Relative Error (%)"
    else:
        errors = np.abs(y_pred - y_true)
        xlabel = "Absolute Error"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCH * 0.6, FIGURE_WIDTH_INCH * 0.4))
    
    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                   COLORS['success'], COLORS['neutral']]
    
    for i in range(num_params):
        err = np.sort(errors[:, i])
        cdf = np.arange(1, len(err) + 1) / len(err)
        
        color = colors_list[i % len(colors_list)]
        ax.plot(err, cdf * 100, label=param_names[i], color=color, lw=1.5)
        
        # Find 95th percentile (use np.percentile for accuracy)
        p95_val = np.percentile(errors[:, i], 95)
        ax.axvline(x=p95_val, color=color, linestyle=':', alpha=0.5)
    
    # Reference lines
    ax.axhline(y=95, color='gray', linestyle='--', lw=0.8, alpha=0.7, label='95%')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_title("Cumulative Error Distribution")
    ax.legend(fontsize=6, loc='best')
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
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Limit samples
    n_samples = min(len(y_true), max_samples)
    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]
    indices = np.arange(n_samples)
    
    # Calculate grid dimensions
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    # Calculate figure size
    subplot_size = FIGURE_WIDTH_INCH / cols
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(FIGURE_WIDTH_INCH, subplot_size * rows * 0.8),
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        # Plot true and predicted
        ax.plot(indices, y_true[:, i], 'o', markersize=3, alpha=0.6,
                color=COLORS['primary'], label='True')
        ax.plot(indices, y_pred[:, i], 'x', markersize=3, alpha=0.6,
                color=COLORS['error'], label='Predicted')
        
        # Labels
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(fontsize=6, loc='best')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# VISUALIZATION - ERROR BOX PLOT
# ==============================================================================
def plot_error_boxplot(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
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
    num_params = y_true.shape[1] if y_true.ndim > 1 else 1
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    if param_names is None or len(param_names) != num_params:
        param_names = [f"P{i}" for i in range(num_params)]
    
    # Calculate errors
    if use_relative:
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.abs((y_pred - y_true) / y_true) * 100
            errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
        ylabel = "Relative Error (%)"
    else:
        errors = y_pred - y_true  # Signed error for box plot
        ylabel = "Prediction Error"
    
    # Create figure
    fig_width = min(FIGURE_WIDTH_INCH * 0.5, 2 + num_params * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_WIDTH_INCH * 0.4))
    
    # Box plot
    bp = ax.boxplot([errors[:, i] for i in range(num_params)],
                    labels=param_names,
                    patch_artist=True,
                    showfliers=True,
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['primary'])
        patch.set_alpha(0.7)
    
    # Zero line for signed errors
    if not use_relative:
        ax.axhline(y=0, color=COLORS['error'], linestyle='--', lw=1.0, alpha=0.7)
    
    ax.set_ylabel(ylabel)
    ax.set_title("Error Distribution by Parameter")
    ax.grid(True, axis='y')
    
    # Rotate labels if needed
    if num_params > 4:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig
