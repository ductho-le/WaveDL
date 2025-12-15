"""
Scientific Metrics and Visualization Utilities
===============================================

Provides metric tracking, statistical calculations, and publication-quality
visualization tools for deep learning experiments.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
# VISUALIZATION
# ==============================================================================
def plot_scientific_scatter(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    param_names: Optional[List[str]] = None,
    max_samples: int = 2000,
    figsize_per_plot: Tuple[float, float] = (3.5, 3.5),
    dpi: int = 120,
    scatter_alpha: float = 0.6,
    scatter_size: int = 30,
    scatter_color: str = 'royalblue'
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
        figsize_per_plot: Size of each subplot in inches
        dpi: Figure resolution
        scatter_alpha: Transparency of scatter points
        scatter_size: Size of scatter points
        scatter_color: Color of scatter points
        
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
    cols = min(num_params, 5)
    rows = (num_params + cols - 1) // cols
    
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows),
        dpi=dpi
    )
    axes = np.array(axes).flatten() if num_params > 1 else [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        # Calculate R² for this target (guard for tiny datasets)
        if len(y_true) >= 2:
            r2 = r2_score(y_true[:, i], y_pred[:, i])
        else:
            r2 = float('nan')
        
        # Scatter plot
        ax.scatter(
            y_true[:, i], y_pred[:, i],
            alpha=scatter_alpha, 
            s=scatter_size, 
            c=scatter_color, 
            edgecolors='none'
        )
        
        # Ideal diagonal line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, alpha=0.8)
        
        # Labels and formatting
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"
        ax.set_title(f"{title}\n$R^2$={r2:.4f}", fontsize=10)
        ax.set_xlabel("Ground Truth", fontsize=9)
        ax.set_ylabel("Prediction", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Equal aspect ratio for proper visualization
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for i in range(num_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def create_training_curves(
    history: List[Dict[str, Any]],
    metrics: List[str] = ['train_loss', 'val_loss'],
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 100
) -> plt.Figure:
    """
    Create training curve visualization from history.
    
    Args:
        history: List of epoch statistics dictionaries
        metrics: Metric names to plot
        figsize: Figure size in inches
        dpi: Figure resolution
        
    Returns:
        Matplotlib Figure object
    """
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    for metric in metrics:
        values = [h.get(metric, np.nan) for h in history]
        ax.plot(epochs, values, label=metric, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Training Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
