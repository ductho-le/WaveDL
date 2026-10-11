"""
Unit Tests for Metrics Module
==============================

Comprehensive tests for:
- MetricTracker: running averages and accumulation
- calc_pearson: Pearson correlation coefficient
- calc_per_target_r2: per-target R² scores
- get_lr: learning rate extraction

Author: Ductho Le (ductho.le@outlook.com)
"""

import numpy as np
import pytest
import torch
from torch import optim

from wavedl.utils.metrics import (
    MetricTracker,
    calc_pearson,
    calc_per_target_r2,
    create_training_curves,
    get_lr,
    plot_scientific_scatter,
)


# ==============================================================================
# METRIC TRACKER TESTS
# ==============================================================================
class TestMetricTracker:
    """Tests for the MetricTracker class."""

    def test_initialization(self):
        """Test that MetricTracker initializes with zero values."""
        tracker = MetricTracker()

        assert tracker.val == 0.0
        assert tracker.avg == 0.0
        assert tracker.sum == 0.0
        assert tracker.count == 0.0

    def test_single_update(self):
        """Test single value update."""
        tracker = MetricTracker()
        tracker.update(5.0)

        assert tracker.val == 5.0
        assert tracker.avg == 5.0
        assert tracker.sum == 5.0
        assert tracker.count == 1.0

    def test_multiple_updates(self):
        """Test multiple value updates compute correct average."""
        tracker = MetricTracker()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            tracker.update(v)

        assert tracker.val == 5.0  # Last value
        assert tracker.avg == pytest.approx(3.0)  # Mean of 1-5
        assert tracker.sum == 15.0
        assert tracker.count == 5.0

    def test_weighted_update(self):
        """Test update with batch sizes (n > 1)."""
        tracker = MetricTracker()

        # Simulating batch averages
        tracker.update(2.0, n=10)  # Sum = 20
        tracker.update(4.0, n=10)  # Sum = 40

        assert tracker.sum == 60.0
        assert tracker.count == 20.0
        assert tracker.avg == pytest.approx(3.0)

    def test_reset(self):
        """Test reset clears all values."""
        tracker = MetricTracker()
        tracker.update(10.0, n=5)
        tracker.reset()

        assert tracker.val == 0.0
        assert tracker.avg == 0.0
        assert tracker.sum == 0.0
        assert tracker.count == 0.0

    def test_repr(self):
        """Test string representation."""
        tracker = MetricTracker()
        tracker.update(3.5, n=2)

        repr_str = repr(tracker)
        assert "MetricTracker" in repr_str
        assert "3.5" in repr_str

    def test_zero_division_safety(self):
        """Test that avg returns 0.0 when count is 0."""
        tracker = MetricTracker()

        # Directly check without any updates
        assert tracker.avg == 0.0

        # Also check after reset
        tracker.update(5.0)
        tracker.reset()
        assert tracker.avg == 0.0


# ==============================================================================
# PEARSON CORRELATION TESTS
# ==============================================================================
class TestCalcPearson:
    """Tests for Pearson correlation calculation."""

    def test_perfect_correlation(self):
        """Test perfect positive correlation returns 1.0."""
        y_true = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).T
        y_pred = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).T

        corr = calc_pearson(y_true, y_pred)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation returns -1.0."""
        y_true = np.array([[1, 2, 3, 4, 5]]).T
        y_pred = np.array([[5, 4, 3, 2, 1]]).T

        corr = calc_pearson(y_true, y_pred)
        assert corr == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation(self):
        """Test uncorrelated data returns near 0."""
        np.random.seed(42)
        y_true = np.random.randn(1000, 3)
        y_pred = np.random.randn(1000, 3)

        corr = calc_pearson(y_true, y_pred)
        assert abs(corr) < 0.1  # Should be close to 0

    def test_1d_input(self):
        """Test that 1D inputs are handled correctly."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        corr = calc_pearson(y_true, y_pred)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_constant_array_handling(self):
        """Test that constant arrays (zero variance) return 0."""
        y_true = np.array([[1, 1, 1, 1, 1]]).T
        y_pred = np.array([[2, 2, 2, 2, 2]]).T

        corr = calc_pearson(y_true, y_pred)
        assert corr == 0.0

    def test_mixed_targets(self):
        """Test with multiple targets of varying correlation."""
        y_true = np.array(
            [
                [1, 2, 3, 4, 5],  # Perfect correlation
                [1, 1, 1, 1, 1],  # Constant (should return 0)
            ]
        ).T
        y_pred = np.array(
            [
                [1, 2, 3, 4, 5],  # Perfect prediction
                [2, 3, 4, 5, 6],  # Different but constant base wouldn't correlate
            ]
        ).T

        corr = calc_pearson(y_true, y_pred)
        # Average of 1.0 and 0.0
        assert corr == pytest.approx(0.5, abs=1e-6)

    def test_dense_fields_are_flattened_per_channel(self):
        """Dense fields compute one correlation per property channel."""
        y_true = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
        y_pred = y_true.copy()

        assert calc_pearson(y_true, y_pred) == pytest.approx(1.0, abs=1e-6)


# ==============================================================================
# R² SCORE TESTS
# ==============================================================================
class TestCalcPerTargetR2:
    """Tests for per-target R² calculation."""

    def test_perfect_prediction(self):
        """Test perfect prediction returns R² = 1.0."""
        y_true = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        y_pred = y_true.copy()

        r2_scores = calc_per_target_r2(y_true, y_pred)

        assert len(r2_scores) == 3
        for r2 in r2_scores:
            assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_mean_prediction(self):
        """Test predicting mean returns R² = 0.0."""
        y_true = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        y_pred = np.array([[3, 4], [3, 4], [3, 4]]).astype(np.float32)  # Mean values

        r2_scores = calc_per_target_r2(y_true, y_pred)

        for r2 in r2_scores:
            assert r2 == pytest.approx(0.0, abs=1e-6)

    def test_negative_r2(self):
        """Test that worse-than-mean predictions can give negative R²."""
        y_true = np.array([[1], [2], [3]]).astype(np.float32)
        y_pred = np.array([[10], [10], [10]]).astype(np.float32)  # Bad predictions

        r2_scores = calc_per_target_r2(y_true, y_pred)

        assert r2_scores[0] < 0

    def test_1d_input(self):
        """Test 1D input handling."""
        y_true = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y_pred = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        r2_scores = calc_per_target_r2(y_true, y_pred)

        assert len(r2_scores) == 1
        assert r2_scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_multiple_targets(self):
        """Test with multiple targets of varying quality."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randn(n_samples, 4).astype(np.float32)

        # Target 0: perfect, Target 1: good, Target 2: poor, Target 3: random
        y_pred = np.zeros_like(y_true)
        y_pred[:, 0] = y_true[:, 0]  # Perfect
        y_pred[:, 1] = y_true[:, 1] + 0.1 * np.random.randn(n_samples)  # Good
        y_pred[:, 2] = y_true[:, 2] + 2.0 * np.random.randn(n_samples)  # Poor
        y_pred[:, 3] = np.random.randn(n_samples)  # Random

        r2_scores = calc_per_target_r2(y_true, y_pred)

        assert r2_scores[0] == pytest.approx(1.0, abs=1e-6)
        assert r2_scores[1] > 0.9  # Good prediction
        assert r2_scores[2] < r2_scores[1]  # Poor is worse than good
        assert r2_scores[3] < r2_scores[1]  # Random is worse than good

    def test_dense_fields_return_one_score_per_channel(self):
        """Spatial axes are observations, not separate named targets."""
        y_true = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        y_pred = y_true.copy()

        r2_scores = calc_per_target_r2(y_true, y_pred)

        np.testing.assert_allclose(r2_scores, np.ones(3))


# ==============================================================================
# LEARNING RATE EXTRACTION TESTS
# ==============================================================================
class TestGetLr:
    """Tests for learning rate extraction from optimizer."""

    def test_single_param_group(self):
        """Test LR extraction with single parameter group."""
        model = torch.nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        lr = get_lr(optimizer)
        assert lr == pytest.approx(0.001, abs=1e-8)

    def test_multiple_param_groups(self):
        """Test LR extraction returns first param group's LR."""
        model = torch.nn.Linear(10, 5)
        optimizer = optim.Adam(
            [
                {"params": model.weight, "lr": 0.01},
                {"params": model.bias, "lr": 0.001},
            ]
        )

        lr = get_lr(optimizer)
        assert lr == pytest.approx(0.01, abs=1e-8)

    def test_sgd_optimizer(self):
        """Test with SGD optimizer."""
        model = torch.nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        lr = get_lr(optimizer)
        assert lr == pytest.approx(0.1, abs=1e-8)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_after_scheduler_step(self):
        """Test LR extraction after scheduler modifies LR."""
        model = torch.nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        initial_lr = get_lr(optimizer)
        scheduler.step()
        updated_lr = get_lr(optimizer)

        assert initial_lr == pytest.approx(0.1, abs=1e-8)
        assert updated_lr == pytest.approx(0.05, abs=1e-8)


# ==============================================================================
# VISUALIZATION TESTS (Smoke Tests)
# ==============================================================================
class TestVisualization:
    """Smoke tests for visualization functions."""

    def test_plot_scientific_scatter_creates_figure(self):
        """Test that scatter plot function returns a figure."""
        y_true = np.random.randn(100, 3)
        y_pred = y_true + 0.1 * np.random.randn(100, 3)

        fig = plot_scientific_scatter(y_true, y_pred)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_scientific_scatter_with_param_names(self):
        """Test scatter plot with custom parameter names."""
        y_true = np.random.randn(50, 2)
        y_pred = y_true + 0.1 * np.random.randn(50, 2)

        fig = plot_scientific_scatter(
            y_true, y_pred, param_names=["Thickness", "Velocity"]
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_scientific_scatter_downsampling(self):
        """Test scatter plot downsamples large datasets."""
        y_true = np.random.randn(10000, 2)
        y_pred = y_true.copy()

        fig = plot_scientific_scatter(y_true, y_pred, max_samples=1000)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_create_training_curves(self):
        """Test training curves visualization."""
        history = [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.2},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.9},
            {"epoch": 3, "train_loss": 0.5, "val_loss": 0.6},
        ]

        fig = create_training_curves(history)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ==============================================================================
# EDGE CASES AND ROBUSTNESS
# ==============================================================================
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.parametrize("field", [False, True])
    def test_large_float32_offsets_preserve_r2_and_correlation(self, field):
        true = np.array([1e9, 1e9 + 64, 1e9, 1e9 + 64], dtype=np.float32)
        pred = true[::-1].copy()
        shape = (1, 1, 2, 2) if field else (4, 1)

        np.testing.assert_allclose(
            calc_per_target_r2(true.reshape(shape), pred.reshape(shape)), [-3.0]
        )
        assert calc_pearson(true.reshape(shape), pred.reshape(shape)) == pytest.approx(
            -1.0
        )

    def test_calc_pearson_small_sample(self):
        """Test Pearson with minimal sample size."""
        y_true = np.array([[1, 2], [3, 4]]).T
        y_pred = np.array([[1, 2], [3, 4]]).T

        # Should not raise, though correlation might be undefined
        corr = calc_pearson(y_true, y_pred)
        assert isinstance(corr, float)

    @pytest.mark.filterwarnings(
        "ignore:R\\^2 score is not well-defined:sklearn.exceptions.UndefinedMetricWarning"
    )
    def test_calc_per_target_r2_single_sample(self):
        """Test R² with very few samples returns NaN gracefully."""
        y_true = np.array([[1, 2, 3]]).astype(np.float32)
        y_pred = np.array([[1, 2, 3]]).astype(np.float32)

        # R² with single sample is undefined (returns NaN), but should not raise
        r2_scores = calc_per_target_r2(y_true, y_pred)
        assert len(r2_scores) == 3
        # Note: R² with 1 sample is undefined, so NaN is expected

    def test_metric_tracker_large_values(self):
        """Test MetricTracker with large values."""
        tracker = MetricTracker()

        for _ in range(1000):
            tracker.update(1e10, n=1000)

        assert tracker.count == 1000000
        assert tracker.avg == pytest.approx(1e10, rel=1e-6)

    def test_metric_tracker_small_values(self):
        """Test MetricTracker with very small values."""
        tracker = MetricTracker()

        for _ in range(100):
            tracker.update(1e-10)

        assert tracker.avg == pytest.approx(1e-10, rel=1e-6)
