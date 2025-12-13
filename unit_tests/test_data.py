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
import tempfile
import shutil
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from utils.data import (
    MemmapDataset,
    memmap_worker_init_fn,
    DataSource,
    NPZSource,
    HDF5Source,
    MATSource,
    get_data_source,
    load_training_data,
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
        fp = np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)
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
            indices=setup["indices"]
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
            indices=setup["indices"]
        )
        assert len(dataset) == setup["n_samples"]
        
        # Subset of indices
        subset_indices = np.array([0, 5, 10, 15])
        dataset_subset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=subset_indices
        )
        assert len(dataset_subset) == 4
    
    def test_getitem(self, temp_memmap_setup):
        """Test __getitem__ returns correct data."""
        setup = temp_memmap_setup
        
        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"]
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
            indices=setup["indices"]
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
            indices=subset_indices
        )
        
        # Dataset index 0 should map to memmap index 0
        x, y = dataset[0]
        assert torch.allclose(y, setup["targets"][0])
        
        # Dataset index 5 should map to memmap index 5
        x, y = dataset[5]
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
            indices=shuffled_indices
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
            indices=setup["indices"]
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
            indices=setup["indices"]
        )
        
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0  # Single process for simplicity
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
            indices=setup["indices"]
        )
        
        x, y = dataset[0]
        
        assert x.is_contiguous()
    
    def test_data_copy_from_memmap(self, temp_memmap_setup):
        """Test that data is copied from memmap (not a view)."""
        setup = temp_memmap_setup
        
        dataset = MemmapDataset(
            memmap_path=setup["memmap_path"],
            targets=setup["targets"],
            shape=setup["shape"],
            indices=setup["indices"]
        )
        
        x1, _ = dataset[0]
        x1_clone = x1.clone()
        
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
        
        with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
            memmap_worker_init_fn(0)
        
        assert mock_dataset.data is None
    
    def test_does_nothing_in_main_process(self):
        """Test that worker init does nothing when not in worker."""
        # When not in a worker, get_worker_info returns None
        with patch('torch.utils.data.get_worker_info', return_value=None):
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
        data = np.load(valid_path)
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
        
        data = np.load(path)
        assert data["input_train"].ndim == 2  # (N, L)
    
    def test_supports_2d_input(self, temp_dir):
        """Test that 2D input shapes are valid."""
        X = np.random.randn(50, 64, 64).astype(np.float32)  # (N, H, W)
        y = np.random.randn(50, 5).astype(np.float32)
        
        path = os.path.join(temp_dir, "2d_data.npz")
        np.savez(path, input_train=X, output_train=y)
        
        data = np.load(path)
        assert data["input_train"].ndim == 3  # (N, H, W)
    
    def test_supports_3d_input(self, temp_dir):
        """Test that 3D input shapes are valid."""
        X = np.random.randn(20, 16, 32, 32).astype(np.float32)  # (N, D, H, W)
        y = np.random.randn(20, 4).astype(np.float32)
        
        path = os.path.join(temp_dir, "3d_data.npz")
        np.savez(path, input_train=X, output_train=y)
        
        data = np.load(path)
        assert data["input_train"].ndim == 4  # (N, D, H, W)
    
    def test_float32_dtype(self, temp_dir):
        """Test that float32 dtype is preferred."""
        X = np.random.randn(50, 64, 64).astype(np.float32)
        y = np.random.randn(50, 5).astype(np.float32)
        
        path = os.path.join(temp_dir, "float32_data.npz")
        np.savez(path, input_train=X, output_train=y)
        
        data = np.load(path)
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
        
        fp = np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)
        fp[:] = np.random.randn(*shape).astype(np.float32)
        fp.flush()
        del fp
        
        targets = torch.randn(1, 3)
        indices = np.array([0])
        
        dataset = MemmapDataset(
            memmap_path=memmap_path,
            targets=targets,
            shape=shape,
            indices=indices
        )
        
        assert len(dataset) == 1
        x, y = dataset[0]
        assert x.shape == (1, 32, 32)
    
    def test_single_target_dimension(self, temp_dir):
        """Test dataset with single target dimension."""
        shape = (10, 1, 32, 32)
        memmap_path = os.path.join(temp_dir, "single_target.dat")
        
        fp = np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)
        fp[:] = np.random.randn(*shape).astype(np.float32)
        fp.flush()
        del fp
        
        targets = torch.randn(10, 1)  # Single target
        indices = np.arange(10)
        
        dataset = MemmapDataset(
            memmap_path=memmap_path,
            targets=targets,
            shape=shape,
            indices=indices
        )
        
        _, y = dataset[0]
        assert y.shape == (1,)
    
    def test_large_number_of_targets(self, temp_dir):
        """Test dataset with many target dimensions."""
        shape = (10, 1, 32, 32)
        memmap_path = os.path.join(temp_dir, "many_targets.dat")
        
        fp = np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)
        fp[:] = np.random.randn(*shape).astype(np.float32)
        fp.flush()
        del fp
        
        n_targets = 100
        targets = torch.randn(10, n_targets)
        indices = np.arange(10)
        
        dataset = MemmapDataset(
            memmap_path=memmap_path,
            targets=targets,
            shape=shape,
            indices=indices
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
    
    def test_unknown_extension_defaults_to_npz(self):
        """Test unknown extension defaults to NPZ."""
        assert DataSource.detect_format("data.xyz") == "npz"
        assert DataSource.detect_format("data.txt") == "npz"


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
        with h5py.File(path, 'w') as f:
            f.create_dataset('input_train', data=X)
            f.create_dataset('output_train', data=y)
        
        source = HDF5Source()
        X_loaded, y_loaded = source.load(path)
        
        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)
    
    def test_load_alternate_keys(self, temp_dir):
        """Test loading HDF5 with X/y keys."""
        X = np.random.randn(30, 128).astype(np.float32)  # 1D
        y = np.random.randn(30, 4).astype(np.float32)
        
        path = os.path.join(temp_dir, "data.hdf5")
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('y', data=y)
        
        source = HDF5Source()
        X_loaded, y_loaded = source.load(path)
        
        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)
    
    def test_raises_on_missing_keys(self, temp_dir):
        """Test that KeyError is raised when keys are not found."""
        X = np.random.randn(10, 32, 32).astype(np.float32)
        
        path = os.path.join(temp_dir, "invalid.h5")
        with h5py.File(path, 'w') as f:
            f.create_dataset('unknown_key', data=X)
        
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
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('y', data=y)
        
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
        X_loaded, y_loaded = load_training_data(path, format='npz')
        
        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(y_loaded, y)


class TestGetDataSource:
    """Tests for the get_data_source factory function."""
    
    def test_returns_correct_source_types(self):
        """Test that correct source types are returned."""
        assert isinstance(get_data_source('npz'), NPZSource)
        assert isinstance(get_data_source('hdf5'), HDF5Source)
        assert isinstance(get_data_source('mat'), MATSource)
    
    def test_raises_on_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        with pytest.raises(ValueError):
            get_data_source('unsupported')

