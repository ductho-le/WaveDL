"""
Unit Tests for Data Loading Utilities
======================================

Comprehensive tests for:
- MemmapDataset: memory-mapped dataset functionality
- memmap_worker_init_fn: worker initialization
- Data format validation
- Multi-format data sources (NPZ, HDF5, MAT)

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from wavedl.utils.data import (
    DataSource,
    HDF5Source,
    LazyDataHandle,
    MATSource,
    MemmapDataset,
    NPZSource,
    get_data_source,
    load_test_data,
    load_training_data,
    memmap_worker_init_fn,
)


# ==============================================================================
# MEMMAP DATASET TESTS
# ==============================================================================
class TestMemmapDataset:
    """Tests for the MemmapDataset class."""

    @pytest.fixture
    def temp_memmap_setup(self):
        """Create temporary memmap file and targets for testing."""
        tmpdir = tempfile.mkdtemp(prefix="wavedl_memmap_test_")

        # Create sample data
        n_samples = 100
        channels = 1
        height, width = 32, 32
        n_targets = 5

        shape = (n_samples, channels, height, width)
        memmap_path = os.path.join(tmpdir, "test_data.dat")

        # Create and populate memmap
        fp = np.memmap(memmap_path, dtype="float32", mode="w+", shape=shape)
        fp[:] = np.random.randn(*shape).astype(np.float32)
        fp.flush()
        del fp

        # Create targets
        targets = torch.randn(n_samples, n_targets)
        indices = np.arange(n_samples)

        yield {
            "tmpdir": tmpdir,
            "memmap_path": memmap_path,
            "targets": targets,
            "shape": shape,
            "indices": indices,
            "n_samples": n_samples,
            "n_targets": n_targets,
        }

        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initialization(self, temp_memmap_setup):
        """Test MemmapDataset can be initialized."""
        setup = temp_memmap_setup

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )

        assert len(dataset) == setup["n_samples"]
        assert dataset.data is None  # Lazy initialization

    def test_len(self, temp_memmap_setup):
        """Test __len__ returns correct sample count."""
        setup = temp_memmap_setup

        # Full indices
        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )
        assert len(dataset) == setup["n_samples"]

        # Subset of indices
        subset_indices = np.array([0, 5, 10, 15])
        dataset_subset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=subset_indices,
        )
        assert len(dataset_subset) == 4

    def test_getitem(self, temp_memmap_setup):
        """Test __getitem__ returns correct data."""
        setup = temp_memmap_setup

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (1, 32, 32)  # (C, H, W)
        assert y.shape == (setup["n_targets"],)

    def test_lazy_memmap_initialization(self, temp_memmap_setup):
        """Test that memmap is lazily opened on first access."""
        setup = temp_memmap_setup

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )

        assert dataset.data is None

        # Access an item
        _ = dataset[0]

        # Now memmap should be opened
        assert dataset.data is not None

    def test_getitem_with_subset_indices(self, temp_memmap_setup):
        """Test __getitem__ works correctly with subset indices."""
        setup = temp_memmap_setup

        # Use only first 10 samples
        subset_indices = np.arange(10)

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=subset_indices,
        )

        # Dataset index 0 should map to memmap index 0
        _x, y = dataset[0]
        assert torch.allclose(y, setup["targets"][0])

        # Dataset index 5 should map to memmap index 5
        _x, y = dataset[5]
        assert torch.allclose(y, setup["targets"][5])

    def test_getitem_with_shuffled_indices(self, temp_memmap_setup):
        """Test __getitem__ with shuffled indices."""
        setup = temp_memmap_setup

        # Shuffled indices
        np.random.seed(42)
        shuffled_indices = np.random.permutation(setup["n_samples"])[:20]

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=shuffled_indices,
        )

        # Check that correct targets are returned
        for i in range(len(dataset)):
            _, y = dataset[i]
            expected_target = setup["targets"][shuffled_indices[i]]
            assert torch.allclose(y, expected_target)

    def test_repr(self, temp_memmap_setup):
        """Test string representation."""
        setup = temp_memmap_setup

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )

        repr_str = repr(dataset)

        assert "MemmapDataset" in repr_str
        assert str(setup["n_samples"]) in repr_str

    def test_with_dataloader(self, temp_memmap_setup):
        """Test MemmapDataset works with DataLoader."""
        setup = temp_memmap_setup

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Single process for simplicity
        )

        batch_count = 0
        for x_batch, y_batch in loader:
            assert x_batch.shape[0] <= 16
            assert x_batch.shape[1:] == (1, 32, 32)
            assert y_batch.shape[0] == x_batch.shape[0]
            assert y_batch.shape[1] == setup["n_targets"]
            batch_count += 1

        expected_batches = (setup["n_samples"] + 15) // 16
        assert batch_count == expected_batches

    def test_data_contiguity(self, temp_memmap_setup):
        """Test that returned tensors are contiguous."""
        setup = temp_memmap_setup

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )

        x, _y = dataset[0]

        assert x.is_contiguous()

    def test_data_copy_from_memmap(self, temp_memmap_setup):
        """Test that data is copied from memmap (not a view)."""
        setup = temp_memmap_setup

        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"],
        )

        x1, _ = dataset[0]
        x1.clone()

        # Modify x1
        x1[0, 0, 0] = 999.0

        # Get item again
        x2, _ = dataset[0]

        # x2 should match original values, not modified x1
        # (This verifies we're getting copies, not views)
        assert x2[0, 0, 0] != 999.0


# ==============================================================================
# WORKER INIT FUNCTION TESTS
# ==============================================================================
class TestMemmapWorkerInitFn:
    """Tests for the worker initialization function."""

    def test_resets_data_attribute(self):
        """Test that worker init resets dataset.data to None."""
        # Create a mock worker_info
        mock_dataset = MagicMock()
        mock_dataset.data = "some_memmap_handle"

        mock_worker_info = MagicMock()
        mock_worker_info.dataset = mock_dataset

        with patch("torch.utils.data.get_worker_info", return_value=mock_worker_info):
            memmap_worker_init_fn(0)

        assert mock_dataset.data is None

    def test_does_nothing_in_main_process(self):
        """Test that worker init does nothing when not in worker."""
        # When not in a worker, get_worker_info returns None
        with patch("torch.utils.data.get_worker_info", return_value=None):
            # Should not raise
            memmap_worker_init_fn(0)


# ==============================================================================
# DATA FORMAT VALIDATION TESTS
# ==============================================================================
class TestDataFormatValidation:
    """Tests for data format requirements."""

    def test_npz_format_with_required_keys(self, temp_dir):
        """Test that NPZ files must have required keys."""
        # Create valid NPZ
        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 5).astype(np.float32)

        valid_path = os.path.join(temp_dir, "valid.npz")
        np.savez(valid_path, input_train=X, output_train=y)

        # Load and verify
        with np.load(valid_path) as data:
            assert "input_train" in data
            assert "output_train" in data

    def test_input_output_sample_count_match(self, temp_dir):
        """Test that input and output must have matching sample counts."""
        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 5).astype(np.float32)

        # Valid case
        assert len(X) == len(y)

        # Would be invalid
        y_wrong = np.random.randn(40, 5).astype(np.float32)
        assert len(X) != len(y_wrong)

    def test_supports_1d_input(self, temp_dir):
        """Test that 1D input shapes are valid."""
        X = np.random.randn(50, 256).astype(np.float32)  # (N, L)
        y = np.random.randn(50, 3).astype(np.float32)

        path = os.path.join(temp_dir, "1d_data.npz")
        np.savez(path, input_train=X, output_train=y)

        with np.load(path) as data:
            assert data["input_train"].ndim == 2  # (N, L)

    def test_supports_2d_input(self, temp_dir):
        """Test that 2D input shapes are valid."""
        X = np.random.randn(50, 64, 64).astype(np.float32)  # (N, H, W)
        y = np.random.randn(50, 5).astype(np.float32)

        path = os.path.join(temp_dir, "2d_data.npz")
        np.savez(path, input_train=X, output_train=y)

        with np.load(path) as data:
            assert data["input_train"].ndim == 3  # (N, H, W)

    def test_supports_3d_input(self, temp_dir):
        """Test that 3D input shapes are valid."""
        X = np.random.randn(20, 16, 32, 32).astype(np.float32)  # (N, D, H, W)
        y = np.random.randn(20, 4).astype(np.float32)

        path = os.path.join(temp_dir, "3d_data.npz")
        np.savez(path, input_train=X, output_train=y)

        with np.load(path) as data:
            assert data["input_train"].ndim == 4  # (N, D, H, W)

    def test_float32_dtype(self, temp_dir):
        """Test that float32 dtype is preferred."""
        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 5).astype(np.float32)

        path = os.path.join(temp_dir, "float32_data.npz")
        np.savez(path, input_train=X, output_train=y)

        with np.load(path) as data:
            assert data["input_train"].dtype == np.float32
            assert data["output_train"].dtype == np.float32


# ==============================================================================
# EDGE CASES
# ==============================================================================
class TestDataEdgeCases:
    """Tests for edge cases in data handling."""

    def test_single_sample_dataset(self, temp_dir):
        """Test dataset with single sample."""
        # Create memmap with single sample
        shape = (1, 1, 32, 32)
        memmap_path = os.path.join(temp_dir, "single.dat")

        fp = np.memmap(memmap_path, dtype="float32", mode="w+", shape=shape)
        fp[:] = np.random.randn(*shape).astype(np.float32)
        fp.flush()
        del fp

        targets = torch.randn(1, 3)
        indices = np.array([0])

        dataset = MemmapDataset(
            memmap_path=memmap_path, targets=targets, shape=shape, indices=indices
        )

        assert len(dataset) == 1
        x, _y = dataset[0]
        assert x.shape == (1, 32, 32)

    def test_single_target_dimension(self, temp_dir):
        """Test dataset with single target dimension."""
        shape = (10, 1, 32, 32)
        memmap_path = os.path.join(temp_dir, "single_target.dat")

        fp = np.memmap(memmap_path, dtype="float32", mode="w+", shape=shape)
        fp[:] = np.random.randn(*shape).astype(np.float32)
        fp.flush()
        del fp

        targets = torch.randn(10, 1)  # Single target
        indices = np.arange(10)

        dataset = MemmapDataset(
            memmap_path=memmap_path, targets=targets, shape=shape, indices=indices
        )

        _, y = dataset[0]
        assert y.shape == (1,)

    def test_large_number_of_targets(self, temp_dir):
        """Test dataset with many target dimensions."""
        shape = (10, 1, 32, 32)
        memmap_path = os.path.join(temp_dir, "many_targets.dat")

        fp = np.memmap(memmap_path, dtype="float32", mode="w+", shape=shape)
        fp[:] = np.random.randn(*shape).astype(np.float32)
        fp.flush()
        del fp

        n_targets = 100
        targets = torch.randn(10, n_targets)
        indices = np.arange(10)

        dataset = MemmapDataset(
            memmap_path=memmap_path, targets=targets, shape=shape, indices=indices
        )

        _, y = dataset[0]
        assert y.shape == (n_targets,)


# ==============================================================================
# MULTI-FORMAT DATA SOURCE TESTS
# ==============================================================================
class TestFormatAutoDetection:
    """Tests for automatic file format detection."""

    def test_npz_extension_detection(self):
        """Test NPZ format is detected from .npz extension."""
        assert DataSource.detect_format("data.npz") == "npz"
        assert DataSource.detect_format("/path/to/data.NPZ") == "npz"

    def test_hdf5_extension_detection(self):
        """Test HDF5 format is detected from .h5 and .hdf5 extensions."""
        assert DataSource.detect_format("data.h5") == "hdf5"
        assert DataSource.detect_format("data.hdf5") == "hdf5"
        assert DataSource.detect_format("/path/to/data.H5") == "hdf5"

    def test_mat_extension_detection(self):
        """Test MAT format is detected from .mat extension."""
        assert DataSource.detect_format("data.mat") == "mat"
        assert DataSource.detect_format("/path/to/data.MAT") == "mat"

    def test_unknown_extension_raises_error(self):
        """Test unknown extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            DataSource.detect_format("data.xyz")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            DataSource.detect_format("data.txt")


class TestNPZSource:
    """Tests for NPZ data source."""

    def test_load_standard_keys(self, temp_dir):
        """Test loading with standard input_train/output_train keys."""
        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 5).astype(np.float32)

        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, input_train=X, output_train=y)

        source = NPZSource()
        X_loaded, y_loaded = source.load(path)

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)

    def test_load_alternate_keys_x_y(self, temp_dir):
        """Test loading with X/y keys."""
        X = np.random.randn(30, 32, 32).astype(np.float32)
        y = np.random.randn(30, 3).astype(np.float32)

        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, X=X, y=y)

        source = NPZSource()
        X_loaded, y_loaded = source.load(path)

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)

    def test_load_alternate_keys_data_labels(self, temp_dir):
        """Test loading with data/labels keys."""
        X = np.random.randn(20, 16).astype(np.float32)  # 1D data
        y = np.random.randn(20, 2).astype(np.float32)

        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, data=X, labels=y)

        source = NPZSource()
        X_loaded, y_loaded = source.load(path)

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)

    def test_raises_on_missing_keys(self, temp_dir):
        """Test that KeyError is raised when keys are not found."""
        X = np.random.randn(10, 32, 32).astype(np.float32)

        path = os.path.join(temp_dir, "invalid.npz")
        np.savez(path, unknown_key=X)

        source = NPZSource()
        with pytest.raises(KeyError):
            source.load(path)


class TestHDF5Source:
    """Tests for HDF5 data source."""

    def test_load_standard_keys(self, temp_dir):
        """Test loading HDF5 with standard keys."""
        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 5).astype(np.float32)

        path = os.path.join(temp_dir, "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("input_train", data=X)
            f.create_dataset("output_train", data=y)

        source = HDF5Source()
        X_loaded, y_loaded = source.load(path)

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)

    def test_load_alternate_keys(self, temp_dir):
        """Test loading HDF5 with X/y keys."""
        X = np.random.randn(30, 128).astype(np.float32)  # 1D
        y = np.random.randn(30, 4).astype(np.float32)

        path = os.path.join(temp_dir, "data.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)

        source = HDF5Source()
        X_loaded, y_loaded = source.load(path)

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)

    def test_raises_on_missing_keys(self, temp_dir):
        """Test that KeyError is raised when keys are not found."""
        X = np.random.randn(10, 32, 32).astype(np.float32)

        path = os.path.join(temp_dir, "invalid.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("unknown_key", data=X)

        source = HDF5Source()
        with pytest.raises(KeyError):
            source.load(path)


class TestLoadTrainingData:
    """Tests for the load_training_data convenience function."""

    def test_auto_detect_npz(self, temp_dir):
        """Test auto-detection for NPZ files."""
        X = np.random.randn(25, 32, 32).astype(np.float32)
        y = np.random.randn(25, 3).astype(np.float32)

        path = os.path.join(temp_dir, "train.npz")
        np.savez(path, input_train=X, output_train=y)

        X_loaded, y_loaded = load_training_data(path)

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)

    def test_auto_detect_hdf5(self, temp_dir):
        """Test auto-detection for HDF5 files."""
        X = np.random.randn(25, 64).astype(np.float32)
        y = np.random.randn(25, 2).astype(np.float32)

        path = os.path.join(temp_dir, "train.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)

        X_loaded, y_loaded = load_training_data(path)

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)

    def test_explicit_format_override(self, temp_dir):
        """Test explicit format specification."""
        X = np.random.randn(15, 16, 16).astype(np.float32)
        y = np.random.randn(15, 1).astype(np.float32)

        # Save as NPZ - np.savez adds .npz if not present
        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, input_train=X, output_train=y)

        # Explicitly specify format (even though auto-detect would work)
        X_loaded, y_loaded = load_training_data(path, format="npz")

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)


class TestGetDataSource:
    """Tests for the get_data_source factory function."""

    def test_returns_correct_source_types(self):
        """Test that correct source types are returned."""
        assert isinstance(get_data_source("npz"), NPZSource)
        assert isinstance(get_data_source("hdf5"), HDF5Source)
        assert isinstance(get_data_source("mat"), MATSource)

    def test_raises_on_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        with pytest.raises(ValueError):
            get_data_source("unsupported")


# ==============================================================================
# LAZY DATA HANDLE TESTS
# ==============================================================================
class TestLazyDataHandle:
    """Tests for the LazyDataHandle context manager."""

    def test_context_manager_returns_data(self, temp_dir):
        """Test context manager returns inputs and outputs."""
        X = np.random.randn(20, 32, 32).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)

        path = os.path.join(temp_dir, "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("input_train", data=X)
            f.create_dataset("output_train", data=y)

        source = HDF5Source()
        with source.load_mmap(path) as (inputs, outputs):
            assert inputs.shape == X.shape
            assert outputs.shape == y.shape
            # Verify we can read data
            np.testing.assert_array_equal(inputs[:], X)
            np.testing.assert_array_equal(outputs[:], y)

    def test_context_manager_closes_file(self, temp_dir):
        """Test file is closed after exiting context."""
        X = np.random.randn(10, 16, 16).astype(np.float32)
        y = np.random.randn(10, 2).astype(np.float32)

        path = os.path.join(temp_dir, "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)

        source = HDF5Source()
        handle = source.load_mmap(path)

        # Before close, handle should be open
        assert repr(handle) == "LazyDataHandle(status=open)"

        handle.close()

        # After close, handle should be closed
        assert repr(handle) == "LazyDataHandle(status=closed)"
        # Calling close again should be safe
        handle.close()

    def test_attributes_access(self, temp_dir):
        """Test accessing inputs and outputs via attributes."""
        X = np.random.randn(15, 24).astype(np.float32)
        y = np.random.randn(15, 4).astype(np.float32)

        path = os.path.join(temp_dir, "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=X)
            f.create_dataset("labels", data=y)

        source = HDF5Source()
        handle = source.load_mmap(path)

        try:
            assert handle.inputs.shape == X.shape
            assert handle.outputs.shape == y.shape
        finally:
            handle.close()

    def test_basic_handle_without_file(self):
        """Test LazyDataHandle works with plain numpy arrays (no file)."""
        inputs = np.array([1, 2, 3])
        outputs = np.array([4, 5, 6])

        handle = LazyDataHandle(inputs, outputs)

        with handle as (inp, outp):
            np.testing.assert_array_equal(inp, inputs)
            np.testing.assert_array_equal(outp, outputs)

        # Should be closed now but safe to access
        assert handle._closed is True


# ==============================================================================
# SINGLE-SAMPLE LOAD_TEST_DATA TESTS
# ==============================================================================
class TestLoadTestDataSingleSample:
    """Tests for single-sample edge cases in load_test_data.

    These tests verify that single-sample outputs are NOT incorrectly transposed.
    The bug manifested when (1, T) output was transposed to (T, 1).
    """

    def test_single_sample_npz_2d_input(self, temp_dir):
        """Single sample with 2D input and multiple targets."""
        X = np.random.randn(1, 64, 64).astype(np.float32)  # 1 sample
        y = np.random.randn(1, 5).astype(np.float32)  # 1 sample, 5 targets

        path = os.path.join(temp_dir, "single.npz")
        np.savez(path, input_test=X, output_test=y)

        X_loaded, y_loaded = load_test_data(path)

        # Input: (1, 64, 64) → (1, 1, 64, 64) with channel added
        assert X_loaded.shape == (1, 1, 64, 64)
        # Output: (1, 5) should remain (1, 5), NOT become (5, 1)
        assert y_loaded.shape == (1, 5)

    def test_single_sample_npz_1d_input(self, temp_dir):
        """Single sample with 1D input and multiple targets."""
        X = np.random.randn(1, 256).astype(np.float32)  # 1 sample, 256 points
        y = np.random.randn(1, 3).astype(np.float32)  # 1 sample, 3 targets

        path = os.path.join(temp_dir, "single_1d.npz")
        np.savez(path, input_test=X, output_test=y)

        X_loaded, y_loaded = load_test_data(path)

        # Input: (1, 256) → (1, 1, 256) with channel added
        assert X_loaded.shape == (1, 1, 256)
        # Output: (1, 3) should remain (1, 3)
        assert y_loaded.shape == (1, 3)

    def test_single_sample_hdf5(self, temp_dir):
        """Single sample HDF5 file preserves output shape."""
        X = np.random.randn(1, 32, 32).astype(np.float32)
        y = np.random.randn(1, 4).astype(np.float32)

        path = os.path.join(temp_dir, "single.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("input_test", data=X)
            f.create_dataset("output_test", data=y)

        X_loaded, y_loaded = load_test_data(path)

        assert X_loaded.shape == (1, 1, 32, 32)
        assert y_loaded.shape == (1, 4)

    def test_single_sample_mat(self, temp_dir):
        """Single sample MAT v7.3 file preserves output shape (critical test)."""
        # Create MAT v7.3 file with 1 sample, 2D input
        # MATLAB stores column-major: (N, H, W) becomes (W, H, N) in HDF5
        # For single sample (1, 32, 32): stored as (32, 32, 1), transposed back to (1, 32, 32)
        X = np.random.randn(1, 32, 32).astype(np.float32)  # (N=1, H=32, W=32)
        y = np.random.randn(1, 5).astype(np.float32)  # (N=1, T=5)

        path = os.path.join(temp_dir, "single.mat")
        with h5py.File(path, "w") as f:
            # MATLAB column-major: transpose all axes
            f.create_dataset("input_test", data=X.T)  # (W=32, H=32, N=1)
            f.create_dataset("output_test", data=y.T)  # (T=5, N=1)

        X_loaded, y_loaded = load_test_data(path)

        # After MATSource transpose and channel addition: (1, 1, 32, 32)
        assert X_loaded.shape[0] == 1  # 1 sample
        # Critical: output should be (1, 5), NOT (5, 1)
        assert y_loaded.shape == (1, 5)

    def test_multi_sample_still_works(self, temp_dir):
        """Verify fix doesn't break normal multi-sample case."""
        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 3).astype(np.float32)

        path = os.path.join(temp_dir, "multi.npz")
        np.savez(path, input_test=X, output_test=y)

        X_loaded, y_loaded = load_test_data(path)

        assert X_loaded.shape == (50, 1, 64, 64)
        assert y_loaded.shape == (50, 3)

    def test_single_sample_single_target(self, temp_dir):
        """Single sample with single target value."""
        X = np.random.randn(1, 64, 64).astype(np.float32)
        y = np.array([[3.14]], dtype=np.float32)  # (1, 1)

        path = os.path.join(temp_dir, "single_single.npz")
        np.savez(path, input_test=X, output_test=y)

        X_loaded, y_loaded = load_test_data(path)

        assert X_loaded.shape == (1, 1, 64, 64)
        assert y_loaded.shape == (1, 1)


# ==============================================================================
# FILE HASH AND CACHE VALIDATION TESTS
# ==============================================================================
class TestFileHash:
    """Tests for file hash computation used in cache validation."""

    def test_compute_file_hash_deterministic(self, temp_dir):
        """Test that file hash is deterministic."""
        from wavedl.utils.data import _compute_file_hash

        path = os.path.join(temp_dir, "test_file.bin")
        with open(path, "wb") as f:
            f.write(b"test content for hashing")

        hash1 = _compute_file_hash(path)
        hash2 = _compute_file_hash(path)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_compute_file_hash_changes_with_content(self, temp_dir):
        """Test that hash changes when file content changes."""
        from wavedl.utils.data import _compute_file_hash

        path = os.path.join(temp_dir, "test_file.bin")

        with open(path, "wb") as f:
            f.write(b"original content")
        hash1 = _compute_file_hash(path)

        with open(path, "wb") as f:
            f.write(b"modified content")
        hash2 = _compute_file_hash(path)

        assert hash1 != hash2

    def test_compute_file_hash_large_file(self, temp_dir):
        """Test hash computation on larger files."""
        from wavedl.utils.data import _compute_file_hash

        path = os.path.join(temp_dir, "large_file.bin")
        # Create a file larger than chunk size (8MB default)
        with open(path, "wb") as f:
            f.write(os.urandom(10 * 1024 * 1024))  # 10MB

        hash_val = _compute_file_hash(path)
        assert len(hash_val) == 64


# ==============================================================================
# LOAD OUTPUTS ONLY TESTS
# ==============================================================================
class TestLoadOutputsOnly:
    """Tests for memory-efficient output-only loading."""

    def test_load_outputs_only_npz(self, temp_dir):
        """Test loading only outputs from NPZ file."""
        from wavedl.utils.data import load_outputs_only

        X = np.random.randn(100, 128, 128).astype(np.float32)  # Large inputs
        y = np.random.randn(100, 5).astype(np.float32)  # Small outputs
        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, input_train=X, output_train=y)

        outputs = load_outputs_only(path)

        assert outputs.shape == (100, 5)
        np.testing.assert_array_equal(outputs, y)

    def test_load_outputs_only_hdf5(self, temp_dir):
        """Test loading only outputs from HDF5 file."""
        from wavedl.utils.data import load_outputs_only

        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 3).astype(np.float32)
        path = os.path.join(temp_dir, "data.h5")

        with h5py.File(path, "w") as f:
            f.create_dataset("input_train", data=X)
            f.create_dataset("output_train", data=y)

        outputs = load_outputs_only(path)

        assert outputs.shape == (50, 3)


# ==============================================================================
# TRANSPOSED H5 DATASET TESTS
# ==============================================================================
class TestTransposedH5Dataset:
    """Tests for _TransposedH5Dataset lazy transpose wrapper."""

    def test_integer_indexing(self, temp_dir):
        """Test integer indexing returns correctly transposed data."""
        from wavedl.utils.data import _TransposedH5Dataset

        # Create test data in MATLAB column-major order
        original = np.random.randn(32, 64, 10).astype(np.float32)  # (W, H, N)
        path = os.path.join(temp_dir, "test.h5")

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=original)

        with h5py.File(path, "r") as f:
            wrapper = _TransposedH5Dataset(f["data"], file_handle=f)

            assert wrapper.shape == (10, 64, 32)  # Transposed: (N, H, W)
            assert len(wrapper) == 10

            # Get single sample
            sample = wrapper[0]
            assert sample.shape == (64, 32)

    def test_slice_indexing(self, temp_dir):
        """Test slice indexing returns correctly transposed batch."""
        from wavedl.utils.data import _TransposedH5Dataset

        original = np.random.randn(16, 32, 20).astype(np.float32)  # (W, H, N)
        path = os.path.join(temp_dir, "test.h5")

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=original)

        with h5py.File(path, "r") as f:
            wrapper = _TransposedH5Dataset(f["data"], file_handle=f)

            batch = wrapper[0:5]
            assert batch.shape == (5, 32, 16)  # (batch, H, W)


# ==============================================================================
# INPUT CHANNELS OVERRIDE TESTS
# ==============================================================================
class TestInputChannelsOverride:
    """Tests for explicit input_channels parameter in load_test_data."""

    def test_input_channels_1_adds_channel(self, temp_dir):
        """Test input_channels=1 forces channel dimension addition."""
        # Create 4D data that could be (N, D, H, W) volume or (N, C, H, W) image
        X = np.random.randn(10, 8, 32, 32).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)
        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, input_test=X, output_test=y)

        # With input_channels=1, should add channel: (N, D, H, W) -> (N, 1, D, H, W)
        X_loaded, _ = load_test_data(path, input_channels=1)

        assert X_loaded.shape == (10, 1, 8, 32, 32)

    def test_input_channels_none_uses_heuristic(self, temp_dir):
        """Test that input_channels=None uses default heuristic."""
        # Create data with large first dim (>16) - should be treated as needing channel
        X = np.random.randn(10, 32, 32, 32).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)
        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, input_test=X, output_test=y)

        X_loaded, _ = load_test_data(path, input_channels=None)

        # Heuristic: dim 1 > 16, so add channel
        assert X_loaded.shape == (10, 1, 32, 32, 32)


# ==============================================================================
# EXPLICIT KEY VALIDATION TESTS
# ==============================================================================
class TestExplicitKeyValidation:
    """Tests for explicit --input_key and --output_key validation.

    These tests ensure that when users provide explicit key names that
    don't exist, the code raises helpful errors instead of silently
    falling back to auto-detected keys (which could load wrong data).
    """

    def test_explicit_input_key_not_found_raises(self, temp_dir):
        """Verify that an explicit --input_key that doesn't exist raises KeyError."""
        X = np.random.randn(10, 32, 32).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)

        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, input_test=X, output_test=y)

        # User provides mistyped key - should fail, not silently use 'input_test'
        with pytest.raises(KeyError, match=r"Explicit --input_key.*not found"):
            load_test_data(path, input_key="typo_input_key")

    def test_explicit_input_key_works_when_present(self, temp_dir):
        """Verify that a valid explicit --input_key loads correctly."""
        X = np.random.randn(10, 32, 32).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)

        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, my_custom_input=X, my_custom_output=y)

        # Explicit key that exists should work
        X_loaded, y_loaded = load_test_data(
            path, input_key="my_custom_input", output_key="my_custom_output"
        )

        assert X_loaded.shape[0] == 10
        assert y_loaded.shape == (10, 3)

    def test_explicit_output_key_not_found_raises(self, temp_dir):
        """Verify that an explicit --output_key that doesn't exist raises KeyError.

        This is a regression test to ensure the existing output_key strictness
        is preserved even after adding input_key strictness.
        """
        X = np.random.randn(10, 32, 32).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)

        path = os.path.join(temp_dir, "data.npz")
        np.savez(path, input_test=X, output_test=y)

        # User provides mistyped output_key - should fail with clear error
        with pytest.raises(KeyError, match=r"Explicit --output_key.*not found"):
            load_test_data(path, output_key="typo_output_key")

    def test_fallback_without_explicit_keys_works(self, temp_dir):
        """Verify that outputs-missing fallback still works when no explicit keys provided.

        This is a regression test to ensure the new strict validation
        doesn't break the legitimate fallback for inference-only mode
        (when test data has inputs but no ground truth outputs).
        """
        # Create file with only inputs (inference-only scenario)
        X = np.random.randn(10, 32, 32).astype(np.float32)

        path = os.path.join(temp_dir, "inference_only.npz")
        np.savez(path, input_test=X)

        # Without explicit keys, should fall back to inference-only mode
        X_loaded, y_loaded = load_test_data(path)

        assert X_loaded.shape[0] == 10
        assert y_loaded is None  # No outputs available
