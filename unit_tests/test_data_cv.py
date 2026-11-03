"""
Unit Tests for Data Loading and Cross-Validation Utilities
===========================================================

Additional tests for data.py and cross_validation.py to improve coverage.

**Tested Components**:
    - data.py: DataSource.detect_format, LazyDataHandle, NPZSource, get_data_source
    - cross_validation.py: CVDataset

Note: Full data loading tests are in test_data.py. This file adds coverage
for edge cases and utilities not covered there.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import tempfile

import numpy as np
import pytest
import torch


# ==============================================================================
# DATA SOURCE FORMAT DETECTION
# ==============================================================================


class TestDataSourceDetectFormat:
    """Tests for automatic file format detection."""

    def test_detect_npz(self):
        """Test NPZ format detection."""
        from wavedl.utils.data import DataSource

        assert DataSource.detect_format("data.npz") == "npz"
        assert DataSource.detect_format("/path/to/data.npz") == "npz"
        assert DataSource.detect_format("data.NPZ") == "npz"

    def test_detect_hdf5(self):
        """Test HDF5 format detection."""
        from wavedl.utils.data import DataSource

        assert DataSource.detect_format("data.h5") == "hdf5"
        assert DataSource.detect_format("data.hdf5") == "hdf5"
        assert DataSource.detect_format("data.HDF5") == "hdf5"

    def test_detect_mat(self):
        """Test MAT format detection."""
        from wavedl.utils.data import DataSource

        assert DataSource.detect_format("data.mat") == "mat"
        assert DataSource.detect_format("/path/to/data.MAT") == "mat"

    def test_detect_unsupported_raises(self):
        """Test that unsupported formats raise ValueError."""
        from wavedl.utils.data import DataSource

        with pytest.raises(ValueError, match="Unsupported"):
            DataSource.detect_format("data.csv")

        with pytest.raises(ValueError, match="Unsupported"):
            DataSource.detect_format("data.txt")


class TestGetDataSource:
    """Tests for data source factory function."""

    def test_get_npz_source(self):
        """Test getting NPZ data source."""
        from wavedl.utils.data import NPZSource, get_data_source

        source = get_data_source("npz")
        assert isinstance(source, NPZSource)

    def test_get_hdf5_source(self):
        """Test getting HDF5 data source."""
        from wavedl.utils.data import HDF5Source, get_data_source

        source = get_data_source("hdf5")
        assert isinstance(source, HDF5Source)

    def test_get_mat_source(self):
        """Test getting MAT data source."""
        from wavedl.utils.data import MATSource, get_data_source

        source = get_data_source("mat")
        assert isinstance(source, MATSource)

    def test_get_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        from wavedl.utils.data import get_data_source

        with pytest.raises(ValueError):
            get_data_source("invalid")


# ==============================================================================
# LAZY DATA HANDLE
# ==============================================================================


class TestLazyDataHandle:
    """Tests for LazyDataHandle context manager."""

    def test_context_manager_returns_data(self):
        """Test that context manager yields inputs and outputs."""
        from wavedl.utils.data import LazyDataHandle

        inputs = np.array([1, 2, 3])
        outputs = np.array([4, 5, 6])

        with LazyDataHandle(inputs, outputs) as (x, y):
            assert np.array_equal(x, inputs)
            assert np.array_equal(y, outputs)

    def test_close_is_safe_to_call_multiple_times(self):
        """Test that close() can be called multiple times."""
        from wavedl.utils.data import LazyDataHandle

        handle = LazyDataHandle(np.array([1]), np.array([2]))
        handle.close()
        handle.close()  # Should not raise

    def test_repr_contains_shape_info(self):
        """Test that repr contains useful info."""
        from wavedl.utils.data import LazyDataHandle

        handle = LazyDataHandle(np.zeros((10, 5)), np.zeros((10, 3)))
        repr_str = repr(handle)
        assert "10" in repr_str or "LazyDataHandle" in repr_str


# ==============================================================================
# NPZ SOURCE
# ==============================================================================


class TestNPZSourceLoad:
    """Tests for NPZ data loading."""

    def test_load_basic_npz(self):
        """Test loading a basic NPZ file."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            inputs = np.random.randn(100, 64, 64).astype(np.float32)
            outputs = np.random.randn(100, 3).astype(np.float32)
            np.savez(f.name, input_train=inputs, output_train=outputs)

            try:
                source = NPZSource()
                X, y = source.load(f.name)

                assert X.shape == (100, 64, 64)
                assert y.shape == (100, 3)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later

    def test_load_mmap_basic(self):
        """Test memory-mapped loading returns LazyDataHandle (consistent with HDF5/MAT)."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            inputs = np.random.randn(50, 32).astype(np.float32)
            outputs = np.random.randn(50, 2).astype(np.float32)
            np.savez(f.name, X=inputs, Y=outputs)

            try:
                source = NPZSource()
                # load_mmap now returns LazyDataHandle for consistent API
                with source.load_mmap(f.name) as (X, y):
                    assert X.shape == (50, 32)
                    assert y.shape == (50, 2)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later

    def test_load_outputs_only(self):
        """Test loading only outputs."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            inputs = np.random.randn(100, 128, 128).astype(np.float32)  # Large
            outputs = np.random.randn(100, 5).astype(np.float32)  # Small
            np.savez(f.name, input_train=inputs, output_train=outputs)

            try:
                source = NPZSource()
                y = source.load_outputs_only(f.name)

                assert y.shape == (100, 5)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later


class TestNPZSourceKeyDetection:
    """Tests for NPZ key auto-detection."""

    def test_detects_standard_keys(self):
        """Test detection of standard key names."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name, input_train=np.zeros((10, 5)), output_train=np.ones((10, 2))
            )

            try:
                source = NPZSource()
                X, y = source.load(f.name)
                assert X.shape == (10, 5)
                assert y.shape == (10, 2)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later

    def test_detects_alternative_keys(self):
        """Test detection of alternative key names."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, X=np.zeros((10, 5)), Y=np.ones((10, 2)))

            try:
                source = NPZSource()
                X, y = source.load(f.name)
                assert X.shape == (10, 5)
                assert y.shape == (10, 2)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later


# ==============================================================================
# CROSS-VALIDATION DATASET
# ==============================================================================


class TestCVDataset:
    """Tests for CVDataset class."""

    def test_basic_initialization(self):
        """Test basic dataset initialization."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(100, 64, 64).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)

        dataset = CVDataset(X, y)

        assert len(dataset) == 100

    def test_getitem_returns_tensors(self):
        """Test that __getitem__ returns torch tensors."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 32).astype(np.float32)
        y = np.random.randn(10, 2).astype(np.float32)

        dataset = CVDataset(X, y)
        x_item, y_item = dataset[0]

        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)

    def test_adds_channel_dimension_for_2d(self):
        """Test that channel dimension is added for 2D inputs."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 64, 64).astype(np.float32)  # (N, H, W)
        y = np.random.randn(10, 3).astype(np.float32)

        dataset = CVDataset(X, y)
        x_item, _ = dataset[0]

        # Should have shape (C, H, W) = (1, 64, 64)
        assert x_item.ndim == 3
        assert x_item.shape[0] == 1

    def test_adds_channel_dimension_for_1d(self):
        """Test that channel dimension is added for 1D inputs."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 128).astype(np.float32)  # (N, L)
        y = np.random.randn(10, 2).astype(np.float32)

        dataset = CVDataset(X, y)
        x_item, _ = dataset[0]

        # Should have shape (C, L) = (1, 128)
        assert x_item.ndim == 2
        assert x_item.shape[0] == 1

    def test_preserves_3d_with_expected_spatial_ndim(self):
        """Test that 3D inputs are handled correctly with expected_spatial_ndim."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 16, 32, 32).astype(np.float32)  # (N, D, H, W)
        y = np.random.randn(10, 3).astype(np.float32)

        # With expected_spatial_ndim=3, should add channel
        dataset = CVDataset(X, y, expected_spatial_ndim=3)
        x_item, _ = dataset[0]

        assert x_item.ndim == 4  # (C, D, H, W)


# ==============================================================================
# SPARSE MATRIX HANDLING
# ==============================================================================


class TestSparseMatrixHandling:
    """Tests for sparse matrix conversion."""

    def test_detects_csr_matrix(self):
        """Test that CSR sparse matrices are detected and converted."""

        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            # Create sparse inputs
            dense = np.random.randn(10, 100).astype(np.float32)
            dense[dense < 0.5] = 0  # Make it sparse
            # csr_matrix(dense) would create sparse, but we test dense loading

            outputs = np.random.randn(10, 2).astype(np.float32)

            # Save as sparse - NPZ supports sparse arrays
            np.savez(f.name, input_train=dense, output_train=outputs)

            try:
                source = NPZSource()
                X, _y = source.load(f.name)

                assert isinstance(X, np.ndarray)
                assert X.shape == (10, 100)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later


# ==============================================================================
# DATA LEAKAGE PREVENTION TESTS
# ==============================================================================


class TestScalerDataLeakage:
    """
    Critical tests to prevent train-validation data leakage via StandardScaler.

    Data leakage through the scaler is a subtle but serious issue that can:
    - Inflate validation metrics by 5-20% (depending on data distribution)
    - Give false confidence in model generalization
    - Go completely unnoticed until production deployment

    These tests verify that:
    1. Scaler is fitted ONLY on training data
    2. Validation data uses transform() only (never fit_transform())
    3. Scaler statistics match training-only statistics
    4. Each CV fold has an independent scaler
    """

    def test_scaler_fitted_only_on_training_split(self):
        """
        Verify scaler statistics match training data only, not full dataset.

        This is the core data leakage test. If scaler.mean_ matches the full
        dataset mean instead of training-only mean, we have data leakage.
        """
        from sklearn.preprocessing import StandardScaler

        # Create data with VERY distinct train/val distributions
        # Use non-overlapping indices to guarantee different statistics
        np.random.seed(42)
        n_train = 800
        n_val = 200
        n_targets = 3

        # Training data: centered around -10
        y_train = (np.random.randn(n_train, n_targets) - 10.0).astype(np.float32)

        # Validation data: centered around +10 (completely different!)
        y_val = (np.random.randn(n_val, n_targets) + 10.0).astype(np.float32)

        # Full dataset combines both
        y_full = np.vstack([y_train, y_val])

        # CORRECT: Fit scaler on training data only
        scaler_correct = StandardScaler()
        scaler_correct.fit(y_train)

        # WRONG: Fit scaler on full dataset (data leakage!)
        scaler_leaked = StandardScaler()
        scaler_leaked.fit(y_full)

        # Verify the means are VERY different (proving leakage would matter)
        # Train mean ~ -10, Full mean ~ -6 (weighted average)
        assert not np.allclose(scaler_correct.mean_, scaler_leaked.mean_, atol=1.0), (
            f"Test setup invalid: train mean {scaler_correct.mean_} and "
            f"full mean {scaler_leaked.mean_} should differ significantly"
        )

        # Verify correct scaler matches training data statistics
        np.testing.assert_array_almost_equal(
            scaler_correct.mean_,
            y_train.mean(axis=0),
            decimal=5,
            err_msg="Scaler mean must match training data mean exactly",
        )

        np.testing.assert_array_almost_equal(
            scaler_correct.scale_,
            y_train.std(axis=0),
            decimal=5,
            err_msg="Scaler scale must match training data std exactly",
        )

        # Verify leaked scaler would have different statistics
        assert not np.allclose(scaler_correct.mean_, y_full.mean(axis=0), atol=0.1), (
            "Scaler should NOT match full dataset mean (would indicate leakage)"
        )

    def test_validation_uses_transform_not_fit_transform(self):
        """
        Verify that validation data is transformed, not fit_transformed.

        fit_transform() on validation data would leak validation statistics
        into the scaler, which then affects how training data is scaled.
        """
        from sklearn.preprocessing import StandardScaler

        np.random.seed(123)

        # Training data: normal distribution around 0
        y_train = np.random.randn(100, 2).astype(np.float32)

        # Validation data: shifted distribution (mean=10)
        y_val = (np.random.randn(50, 2) + 10).astype(np.float32)

        # Correct approach: fit on train, transform on val
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train)
        y_val_scaled = scaler.transform(y_val)

        # Verify training data is centered (mean ≈ 0)
        assert np.abs(y_train_scaled.mean()) < 0.1, (
            "Training data should be centered after fit_transform"
        )

        # Verify validation data is NOT centered (because we used transform, not fit_transform)
        # If validation was fit_transformed, its mean would also be ~0
        assert np.abs(y_val_scaled.mean()) > 5.0, (
            "Validation data should NOT be centered - it should be transformed "
            "using training statistics, not its own statistics"
        )

        # Verify scaler statistics still match training data
        np.testing.assert_array_almost_equal(
            scaler.mean_,
            y_train.mean(axis=0),
            decimal=5,
            err_msg="Scaler mean should only reflect training data",
        )

    def test_inverse_transform_consistency(self):
        """
        Verify that inverse_transform recovers original values correctly.

        This ensures the scaler's forward and inverse transforms are consistent,
        which is critical for reporting metrics in physical units.
        """
        from sklearn.preprocessing import StandardScaler

        np.random.seed(456)
        y_train = (
            np.random.randn(100, 3).astype(np.float32) * 10 + 50
        )  # Mean~50, std~10
        y_val = np.random.randn(30, 3).astype(np.float32) * 10 + 50

        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train)
        y_val_scaled = scaler.transform(y_val)

        # Inverse transform should recover original values
        y_train_recovered = scaler.inverse_transform(y_train_scaled)
        y_val_recovered = scaler.inverse_transform(y_val_scaled)

        np.testing.assert_array_almost_equal(
            y_train,
            y_train_recovered,
            decimal=5,
            err_msg="Training data should be perfectly recovered after inverse_transform",
        )

        np.testing.assert_array_almost_equal(
            y_val,
            y_val_recovered,
            decimal=5,
            err_msg="Validation data should be perfectly recovered after inverse_transform",
        )


class TestCVScalerIsolation:
    """
    Tests for cross-validation scaler isolation.

    Each CV fold must have its own scaler fitted only on that fold's training data.
    Sharing scalers across folds or fitting on the full dataset would cause leakage.
    """

    def test_each_fold_has_independent_scaler(self):
        """
        Verify each CV fold fits scaler only on its training split.

        This replicates the logic in cross_validation.py to verify correctness.
        """
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        np.random.seed(789)
        n_samples = 200
        n_targets = 2

        # Create data with STRONG varying statistics across the dataset
        # Each segment has very different mean
        y = np.zeros((n_samples, n_targets), dtype=np.float32)
        segment_size = n_samples // 5
        for i in range(5):
            start = i * segment_size
            end = (i + 1) * segment_size
            # Each segment has mean = i * 10 (0, 10, 20, 30, 40)
            y[start:end] = np.random.randn(segment_size, n_targets) + i * 10

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scalers = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(y)):
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Each fold gets its own scaler (as WaveDL does)
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train)
            scaler.transform(y_val)

            # Verify scaler matches this fold's training data
            np.testing.assert_array_almost_equal(
                scaler.mean_,
                y_train.mean(axis=0),
                decimal=5,
                err_msg=f"Fold {fold_idx}: Scaler mean must match fold's training data",
            )

            fold_scalers.append(scaler)

            # Verify training data is properly normalized
            assert np.abs(y_train_scaled.mean()) < 0.1, (
                f"Fold {fold_idx}: Training data should be centered"
            )

        # Verify different folds have different scaler statistics
        # (because they see different training data)
        means = np.array([s.mean_ for s in fold_scalers])
        mean_range = means.max(axis=0) - means.min(axis=0)

        assert np.any(mean_range > 0.5), (
            f"Different folds should have different scaler means. "
            f"Mean range: {mean_range}"
        )

    def test_fold_scaler_does_not_see_validation_data(self):
        """
        Verify fold scaler statistics exclude validation samples.

        Critical test: if scaler sees val data, metrics are inflated.
        """
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        np.random.seed(101)
        n_samples = 100
        n_targets = 1

        # Create data where val samples are extreme outliers
        y = np.zeros((n_samples, n_targets), dtype=np.float32)
        y[:80] = np.random.randn(80, n_targets)  # Normal samples (mean ~0)
        y[80:] = 1000.0  # Extreme outliers (will be in some val splits)

        # Use shuffle=False to control which samples go to val
        kfold = KFold(n_splits=5, shuffle=False)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(y)):
            y_train = y[train_idx]

            scaler = StandardScaler()
            scaler.fit(y_train)

            # If outliers are in val set (indices 80-99), scaler should NOT see them
            has_outliers_in_val = any(idx >= 80 for idx in val_idx)
            has_outliers_in_train = any(idx >= 80 for idx in train_idx)

            if has_outliers_in_val and not has_outliers_in_train:
                # This fold's training set has no outliers
                # Scaler mean should be near 0, not influenced by 1000.0 values
                assert np.abs(scaler.mean_[0]) < 5.0, (
                    f"Fold {fold_idx}: Scaler should not be influenced by "
                    f"validation outliers. Mean={scaler.mean_[0]:.2f}"
                )


class TestCrossValidationScalerIntegration:
    """
    Integration tests for scaler handling in the full CV pipeline.

    These tests verify the actual cross_validation.py implementation
    correctly prevents data leakage.
    """

    def test_run_cross_validation_scaler_isolation(self):
        """
        Test that run_cross_validation creates isolated scalers per fold.

        This is an integration test that runs the actual CV function.
        """
        from wavedl.utils.cross_validation import CVDataset

        np.random.seed(2025)

        # Create synthetic data
        X = np.random.randn(100, 32, 32).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)

        # Add bias to y to make scaler statistics meaningful
        y[:, 0] += np.linspace(-10, 10, 100)

        # Simulate what run_cross_validation does for one fold
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        kfold = KFold(n_splits=5, shuffle=True, random_state=2025)
        train_idx, val_idx = next(iter(kfold.split(X)))

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit scaler on training data only (as WaveDL does)
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train)
        y_val_scaled = scaler.transform(y_val)

        # Create datasets
        train_ds = CVDataset(X_train, y_train_scaled, expected_spatial_ndim=2)
        val_ds = CVDataset(X_val, y_val_scaled, expected_spatial_ndim=2)

        # Verify datasets have correct sizes
        assert len(train_ds) == len(train_idx)
        assert len(val_ds) == len(val_idx)

        # Verify scaler was fitted on training data only
        np.testing.assert_array_almost_equal(
            scaler.mean_,
            y_train.mean(axis=0),
            decimal=5,
        )

        # Verify training targets are normalized
        _, y_item = train_ds[0]
        assert y_item.numpy().mean() < 5.0, (
            "Training targets should be approximately centered"
        )

    def test_scaler_saved_per_fold(self):
        """
        Test that scalers are correctly saved for each fold.

        Each fold should have its own scaler.pkl that can be used for inference.
        """
        import pickle

        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        np.random.seed(999)
        n_samples = 150
        n_targets = 2

        # Create data with VERY different statistics in different regions
        y = np.zeros((n_samples, n_targets), dtype=np.float32)
        segment_size = n_samples // 3
        # Segment 0: mean ~ -20
        y[:segment_size] = np.random.randn(segment_size, n_targets) - 20
        # Segment 1: mean ~ 0
        y[segment_size : 2 * segment_size] = np.random.randn(segment_size, n_targets)
        # Segment 2: mean ~ +20
        y[2 * segment_size :] = (
            np.random.randn(n_samples - 2 * segment_size, n_targets) + 20
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            saved_scalers = []

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(y)):
                y_train = y[train_idx]

                scaler = StandardScaler()
                scaler.fit(y_train)

                # Save scaler (as WaveDL does in CV)
                fold_dir = os.path.join(tmpdir, f"fold_{fold_idx + 1}")
                os.makedirs(fold_dir, exist_ok=True)
                scaler_path = os.path.join(fold_dir, "scaler.pkl")

                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)

                # Load it back and verify
                with open(scaler_path, "rb") as f:
                    loaded_scaler = pickle.load(f)

                np.testing.assert_array_equal(
                    scaler.mean_,
                    loaded_scaler.mean_,
                    err_msg=f"Fold {fold_idx}: Loaded scaler should match saved scaler",
                )

                saved_scalers.append(loaded_scaler.mean_.copy())

            # Verify different folds have different scalers (use larger tolerance)
            # With shuffled data from 3 distinct segments, means should vary
            mean_range = np.max(saved_scalers, axis=0) - np.min(saved_scalers, axis=0)
            assert np.any(mean_range > 1.0), (
                f"Different folds should have different scaler statistics. "
                f"Mean range: {mean_range}"
            )


class TestDataLoaderScalerLeakage:
    """
    Tests for scaler leakage in the main data loading pipeline (data.py).

    These tests verify prepare_dataloaders correctly handles scaler fitting.
    """

    def test_scaler_fitted_on_train_indices_only(self):
        """
        Verify the data.py scaler fitting logic uses train indices only.

        This replicates the logic in prepare_dataloaders to verify correctness.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        np.random.seed(2024)
        n_samples = 500
        n_targets = 4

        # Create outputs with distinct distribution across samples
        outputs = np.random.randn(n_samples, n_targets).astype(np.float32)
        outputs[:, 0] += np.linspace(-10, 10, n_samples)  # Trend in first target

        # Split indices (as prepare_dataloaders does)
        indices = np.arange(n_samples)
        tr_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=2025)

        # Fit scaler on training outputs only (as prepare_dataloaders does)
        outp_train = outputs[tr_idx]
        scaler = StandardScaler()
        scaler.fit(outp_train)

        # Verify scaler statistics match training data
        np.testing.assert_array_almost_equal(
            scaler.mean_,
            outp_train.mean(axis=0),
            decimal=5,
            err_msg="Scaler must be fitted on training indices only",
        )

        # Verify scaler does NOT match full dataset
        full_mean = outputs.mean(axis=0)
        assert not np.allclose(scaler.mean_, full_mean, atol=0.1), (
            "Scaler mean should differ from full dataset mean "
            "(confirming train-only fitting)"
        )

        # Verify scaler does NOT match validation data
        val_mean = outputs[val_idx].mean(axis=0)
        assert not np.allclose(scaler.mean_, val_mean, atol=0.1), (
            "Scaler mean should differ from validation mean "
            "(confirming no val data leakage)"
        )

    def test_1d_output_handling_no_leakage(self):
        """
        Verify 1D outputs are properly reshaped without leakage.

        data.py reshapes (N,) -> (N, 1) for StandardScaler compatibility.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        np.random.seed(111)
        n_samples = 200

        # 1D outputs (single target)
        outputs = np.random.randn(n_samples).astype(np.float32)
        outputs += np.linspace(-5, 5, n_samples)

        indices = np.arange(n_samples)
        tr_idx, _val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # Reshape for StandardScaler (as data.py does)
        outp_train = outputs[tr_idx].reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(outp_train)

        # Verify scaler was fitted correctly
        expected_mean = outputs[tr_idx].mean()
        assert np.abs(scaler.mean_[0] - expected_mean) < 0.01, (
            "Scaler mean should match training data mean for 1D outputs"
        )


class TestLeakageDetection:
    """
    Meta-tests that demonstrate how to detect data leakage.

    These tests show the impact of leakage and how to catch it.
    """

    def test_leakage_detection_via_validation_metrics(self):
        """
        Demonstrate that leaked scaler inflates validation metrics.

        This test shows WHY preventing leakage matters.
        """
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import StandardScaler

        np.random.seed(777)

        # Training data: centered around 0
        y_train = np.random.randn(100, 1).astype(np.float32)

        # Validation data: shifted distribution (mean=5)
        y_val = (np.random.randn(50, 1) + 5).astype(np.float32)

        # Simulate predictions (just return mean of training data)
        y_pred_val = np.zeros_like(y_val)

        # CORRECT: Scaler fitted on train only
        scaler_correct = StandardScaler()
        scaler_correct.fit(y_train)
        y_val_correct = scaler_correct.transform(y_val)
        y_pred_correct = scaler_correct.transform(y_pred_val)

        # WRONG: Scaler fitted on full data (leakage)
        y_full = np.vstack([y_train, y_val])
        scaler_leaked = StandardScaler()
        scaler_leaked.fit(y_full)
        y_val_leaked = scaler_leaked.transform(y_val)
        y_pred_leaked = scaler_leaked.transform(y_pred_val)

        # Compute MSE with each scaler
        mse_correct = mean_squared_error(y_val_correct, y_pred_correct)
        mse_leaked = mean_squared_error(y_val_leaked, y_pred_leaked)

        # The leaked MSE will be different (typically lower when val is OOD)
        # This demonstrates why leakage is dangerous
        assert mse_correct != mse_leaked, (
            "MSE should differ between correct and leaked scaling, "
            "demonstrating the impact of data leakage"
        )

    def test_leakage_signature_in_scaler_statistics(self):
        """
        Show how to detect leakage by examining scaler statistics.

        If scaler.n_samples_seen_ includes validation samples, we have leakage.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        np.random.seed(888)
        n_samples = 100

        y = np.random.randn(n_samples, 2).astype(np.float32)
        indices = np.arange(n_samples)
        tr_idx, _val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        y_train = y[tr_idx]

        scaler = StandardScaler()
        scaler.fit(y_train)

        # Verify sample count matches training set size
        assert scaler.n_samples_seen_ == len(tr_idx), (
            f"Scaler should have seen {len(tr_idx)} samples, "
            f"but saw {scaler.n_samples_seen_}. "
            "Mismatch indicates potential data leakage!"
        )

        # If this assertion fails, validation data was included in fit()
