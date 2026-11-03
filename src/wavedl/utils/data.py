"""
Data Loading and Preprocessing Utilities
=========================================

Provides memory-efficient data loading for large-scale datasets with:
- Memory-mapped file support for datasets exceeding RAM
- DDP-safe data preparation with proper synchronization
- Thread-safe DataLoader worker initialization
- Multi-format support (NPZ, HDF5, MAT)

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import gc
import hashlib
import logging
import math
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from accelerate import Accelerator
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .distributed import wait_for_everyone
from .targets import (
    SpatialTargetScaler,
    canonical_target_shape,
    canonicalize_targets,
    create_target_scaler,
)


# Optional scipy.io for MATLAB files
try:
    import scipy.io

    HAS_SCIPY_IO = True
except ImportError:
    HAS_SCIPY_IO = False


# ==============================================================================
# DATA SOURCE ABSTRACTION
# ==============================================================================

# Supported key names for input/output arrays (priority order, pairwise aligned)
INPUT_KEYS = ["input_train", "input_test", "X", "data", "inputs", "features", "x"]
OUTPUT_KEYS = ["output_train", "output_test", "Y", "labels", "outputs", "targets", "y"]


def _as_finite_float32(values: Any, name: str) -> np.ndarray:
    """Reject invalid physical values before casting them into training tensors."""
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite real values.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains values outside the finite float32 range.")
    return result


def _bounded_batch_size(
    shape, dtype, max_items: int, max_bytes: int = 32 * 1024**2
) -> int:
    """Bound dense processing chunks by bytes as well as sample count."""
    sample_bytes = math.prod(shape[1:]) * max(np.dtype(dtype).itemsize, 8)
    return min(max_items, max(1, max_bytes // max(1, sample_bytes)))


def _compute_file_hash(
    path: str, mode: str = "sha256", chunk_size: int = 8 * 1024 * 1024
) -> str:
    """
    Compute hash of a file for cache validation.

    Uses chunked reading to handle large files efficiently without loading
    the entire file into memory. This is more reliable than mtime for detecting
    actual content changes, especially with cloud sync services (Dropbox, etc.)
    that may touch files without modifying content.

    Args:
        path: Path to file to hash
        mode: Validation mode:
            - 'sha256': Full content hash (default, most reliable)
            - 'fast': Partial hash (first+last 1MB + size, faster for large files)
            - 'size': File size only (fastest, least reliable)
        chunk_size: Read buffer size (default 8MB for fast I/O)

    Returns:
        Hash string for cache comparison
    """
    if mode == "size":
        return str(os.path.getsize(path))
    elif mode == "fast":
        # Hash first 1MB + last 1MB + file size for quick validation
        file_size = os.path.getsize(path)
        hasher = hashlib.sha256()
        hasher.update(str(file_size).encode())
        with open(path, "rb") as f:
            hasher.update(f.read(1024 * 1024))  # First 1MB
            if file_size > 1024 * 1024:
                f.seek(-1024 * 1024, 2)
                hasher.update(f.read())  # Last 1MB
        return hasher.hexdigest()
    else:  # sha256 (full)
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()


class LazyDataHandle:
    """
    Context manager wrapper for memory-mapped data handles.

    Provides proper cleanup of file handles returned by load_mmap() methods.
    Can be used either as a context manager (recommended) or with explicit close().

    Usage:
        # Context manager (recommended)
        with source.load_mmap(path) as (inputs, outputs):
            # Use inputs and outputs
            pass  # File automatically closed

        # Manual cleanup
        handle = source.load_mmap(path)
        inputs, outputs = handle.inputs, handle.outputs
        # ... use data ...
        handle.close()

    Attributes:
        inputs: Input data array/dataset
        outputs: Output data array/dataset
    """

    def __init__(self, inputs, outputs, file_handle=None):
        """
        Initialize the handle.

        Args:
            inputs: Input array or lazy dataset
            outputs: Output array or lazy dataset
            file_handle: Optional file handle to close on cleanup
        """
        self.inputs = inputs
        self.outputs = outputs
        self._file = file_handle
        self._closed = False

    def __enter__(self):
        """Return (inputs, outputs) tuple for unpacking."""
        return self.inputs, self.outputs

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file handle on context exit."""
        self.close()
        return False  # Don't suppress exceptions

    def close(self):
        """
        Close the underlying file handle.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        # Close the file handle if we have one
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        # Also close any _TransposedH5Dataset wrappers
        for data in (self.inputs, self.outputs):
            if hasattr(data, "close"):
                try:
                    data.close()
                except Exception:
                    pass

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"LazyDataHandle(status={status})"


class DataSource(ABC):
    """
    Abstract base class for data loaders supporting multiple file formats.

    Subclasses must implement the `load()` method to return input/output arrays,
    and optionally `load_outputs_only()` for memory-efficient target loading.
    """

    @abstractmethod
    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load input and output arrays from a file.

        Args:
            path: Path to the data file

        Returns:
            Tuple of (inputs, outputs) as numpy arrays
        """
        pass

    @abstractmethod
    def load_outputs_only(self, path: str) -> np.ndarray:
        """
        Load only output/target arrays from a file (memory-efficient).

        This avoids loading large input arrays when only targets are needed,
        which is critical for HPC environments with memory constraints.

        Args:
            path: Path to the data file

        Returns:
            Output/target array
        """
        pass

    @abstractmethod
    def load_mmap(self, path: str) -> LazyDataHandle:
        """
        Load arrays through a lazy data handle.

        Args:
            path: Path to the data file

        Returns:
            LazyDataHandle instance
        """
        pass

    @staticmethod
    def detect_format(path: str) -> str:
        """
        Auto-detect file format from extension.

        Args:
            path: Path to data file

        Returns:
            Format string: 'npz', 'hdf5', or 'mat'
        """
        ext = Path(path).suffix.lower()
        format_map = {
            ".npz": "npz",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".mat": "mat",
        }
        if ext not in format_map:
            raise ValueError(
                f"Unsupported file extension: '{ext}'. "
                f"Supported formats: .npz, .h5, .hdf5, .mat"
            )
        return format_map[ext]

    @staticmethod
    def _find_key(available_keys: list[str], candidates: list[str]) -> str | None:
        """Find first matching key from candidates in available keys."""
        for key in candidates:
            if key in available_keys:
                return key
        return None


class NPZSource(DataSource):
    """Load data from NumPy .npz archives."""

    def __init__(self, *, allow_pickle: bool = False):
        self.allow_pickle = allow_pickle

    @staticmethod
    def _safe_load(
        path: str,
        keys_to_probe: list[str],
        mmap_mode: str | None = None,
        *,
        allow_pickle: bool = False,
    ):
        """Load NPZ without executing pickle unless explicitly trusted.

        The error for object arrays happens at ACCESS time, not load time.
        So we need to probe the keys to detect if pickle is required.

        WARNING: When mmap_mode is not None, the returned NpzFile must be kept
        open for arrays to remain valid. Caller is responsible for closing.
        For non-mmap loading, use _load_and_copy() instead to avoid leaks.
        """
        data = np.load(path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        try:
            # Probe keys to trigger error if object arrays exist
            for key in keys_to_probe:
                if key in data:
                    _ = data[key]  # This raises ValueError for object arrays
            return data
        except Exception as e:
            data.close()
            if "allow_pickle=False" in str(e):
                raise ValueError(
                    "NPZ object arrays require pickle, which can execute arbitrary "
                    "code. Convert the dataset to numeric arrays or HDF5, or use "
                    "allow_pickle=True (--allow_pickle) only for a trusted file."
                ) from e
            raise

    @staticmethod
    def _load_and_copy(
        path: str, keys: list[str], *, allow_pickle: bool = False
    ) -> dict[str, np.ndarray]:
        """Load NPZ and copy arrays, ensuring file is properly closed.

        This prevents file descriptor leaks by copying arrays before closing.
        Use this for eager loading; use _safe_load for memory-mapped access.
        """
        data = NPZSource._safe_load(
            path, keys, mmap_mode=None, allow_pickle=allow_pickle
        )
        try:
            result = {}
            for key in keys:
                if key in data:
                    arr = data[key]
                    # Copy ensures we don't hold reference to mmap
                    result[key] = arr.copy() if hasattr(arr, "copy") else arr
            return result
        finally:
            if hasattr(data, "close"):
                data.close()

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load NPZ file (pickle enabled only for sparse matrices)."""
        # First pass to find keys without loading data
        with np.load(path, allow_pickle=False) as probe:
            keys = list(probe.keys())

        input_key = self._find_key(keys, INPUT_KEYS)
        output_key = self._find_key(keys, OUTPUT_KEYS)

        if input_key is None or output_key is None:
            raise KeyError(
                f"NPZ must contain input and output arrays. "
                f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                f"Found: {keys}"
            )

        data = self._load_and_copy(
            path, [input_key, output_key], allow_pickle=self.allow_pickle
        )
        inp = data[input_key]
        outp = data[output_key]

        # Handle object arrays (e.g., sparse matrices stored as objects)
        if inp.dtype == object:
            inp = np.array([x.toarray() if hasattr(x, "toarray") else x for x in inp])

        return inp, outp

    def load_mmap(self, path: str) -> LazyDataHandle:
        """
        Load NPZ arrays through the common lazy-handle interface.

        NumPy cannot memory-map members of a ZIP archive. Both requested
        arrays are materialized in RAM, even with mmap_mode='r'. Use HDF5 or
        MATLAB v7.3 files when datasets exceed available memory.

        Returns a LazyDataHandle for consistent API across all data sources.
        The NpzFile is kept open for lazy access.

        Usage:
            with source.load_mmap(path) as (inputs, outputs):
                # Use inputs and outputs
                pass  # File automatically closed

        Note: Returned arrays are independent of the original archive.
        """
        # First pass to find keys without loading data
        with np.load(path, allow_pickle=False) as probe:
            keys = list(probe.keys())

        input_key = self._find_key(keys, INPUT_KEYS)
        output_key = self._find_key(keys, OUTPUT_KEYS)

        if input_key is None or output_key is None:
            raise KeyError(
                f"NPZ must contain input and output arrays. "
                f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                f"Found: {keys}"
            )

        # Keep NpzFile open for lazy access (like HDF5/MATSource)
        data = self._safe_load(
            path, [input_key, output_key], mmap_mode="r", allow_pickle=self.allow_pickle
        )
        inp = data[input_key]
        outp = data[output_key]

        # Return LazyDataHandle for consistent API with HDF5Source/MATSource
        return LazyDataHandle(inp, outp, file_handle=data)

    def load_outputs_only(self, path: str) -> np.ndarray:
        """Load only targets from NPZ (avoids loading large input arrays)."""
        # First pass to find keys without loading data
        with np.load(path, allow_pickle=False) as probe:
            keys = list(probe.keys())

        output_key = self._find_key(keys, OUTPUT_KEYS)
        if output_key is None:
            raise KeyError(
                f"NPZ must contain output array. "
                f"Supported keys: {OUTPUT_KEYS}. Found: {keys}"
            )

        data = self._load_and_copy(path, [output_key], allow_pickle=self.allow_pickle)
        return data[output_key]


class HDF5Source(DataSource):
    """Load data from HDF5 (.h5, .hdf5) files."""

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())

            input_key = self._find_key(keys, INPUT_KEYS)
            output_key = self._find_key(keys, OUTPUT_KEYS)

            if input_key is None or output_key is None:
                raise KeyError(
                    f"HDF5 must contain input and output datasets. "
                    f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                    f"Found: {keys}"
                )

            # Load into memory (HDF5 datasets are lazy by default)
            inp = f[input_key][:]
            outp = f[output_key][:]

        return inp, outp

    def load_mmap(self, path: str) -> LazyDataHandle:
        """
        Load HDF5 file with lazy/memory-mapped access.

        Returns a LazyDataHandle that reads from disk on-demand,
        avoiding loading the entire file into RAM.

        Usage:
            with source.load_mmap(path) as (inputs, outputs):
                # Use inputs and outputs
                pass  # File automatically closed
        """
        f = h5py.File(path, "r")  # Keep file open for lazy access
        keys = list(f.keys())

        input_key = self._find_key(keys, INPUT_KEYS)
        output_key = self._find_key(keys, OUTPUT_KEYS)

        if input_key is None or output_key is None:
            f.close()
            raise KeyError(
                f"HDF5 must contain input and output datasets. "
                f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                f"Found: {keys}"
            )

        # Return wrapped handle for proper cleanup
        return LazyDataHandle(f[input_key], f[output_key], file_handle=f)

    def load_outputs_only(self, path: str) -> np.ndarray:
        """Load only targets from HDF5 (avoids loading large input arrays)."""
        with h5py.File(path, "r") as f:
            keys = list(f.keys())

            output_key = self._find_key(keys, OUTPUT_KEYS)
            if output_key is None:
                raise KeyError(
                    f"HDF5 must contain output dataset. "
                    f"Supported keys: {OUTPUT_KEYS}. Found: {keys}"
                )

            outp = f[output_key][:]

        return outp


class _TransposedH5Dataset:
    """
    Lazy transpose wrapper for h5py datasets.

    MATLAB stores arrays in column-major (Fortran) order, while Python/NumPy
    expects row-major (C) order. This wrapper provides a transposed view
    without loading the entire dataset into memory.

    Supports:
        - len(): Returns the transposed first dimension
        - []: Returns slices with automatic transpose
        - shape: Returns the transposed shape
        - dtype: Returns the underlying dtype

    This is critical for MATSource.load_mmap() to return consistent axis
    ordering with the eager loader (MATSource.load()).

    IMPORTANT: Holds a strong reference to the h5py.File to prevent
    premature garbage collection while datasets are in use.
    """

    def __init__(self, h5_dataset, file_handle=None):
        """
        Args:
            h5_dataset: The h5py dataset to wrap
            file_handle: Optional h5py.File reference to keep alive
        """
        self._dataset = h5_dataset
        self._file = file_handle  # Keep file alive to prevent GC
        # Transpose shape: MATLAB (cols, rows, ...) -> Python (rows, cols, ...)
        self.shape = tuple(reversed(h5_dataset.shape))
        self.dtype = h5_dataset.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions (derived from shape for numpy compatibility)."""
        return len(self.shape)

    @property
    def _transpose_axes(self) -> tuple[int, ...]:
        """Transpose axis order for reversing dimensions.

        For shape (A, B, C) -> reversed (C, B, A), transpose axes are (2, 1, 0).
        """
        return tuple(range(len(self._dataset.shape) - 1, -1, -1))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        """
        Fetch data with automatic full transpose.

        Handles integer indexing, slices, and fancy indexing.
        All operations return data with fully reversed axes to match .T behavior.
        """
        if isinstance(idx, int | np.integer):
            # Single sample: index into last axis of h5py dataset (column-major)
            # Result needs full transpose of remaining dimensions
            data = self._dataset[..., idx]
            if data.ndim == 0:
                return data
            elif data.ndim == 1:
                return data  # 1D doesn't need transpose
            else:
                # Full transpose: reverse all axes
                return np.transpose(data)

        elif isinstance(idx, slice):
            # Slice indexing: fetch from last axis, then fully transpose
            start, stop, step = idx.indices(self.shape[0])
            data = self._dataset[..., start:stop:step]

            # Handle special case: 1D result (e.g., row vector)
            if data.ndim == 1:
                return data

            # Full transpose: reverse ALL axes (not just moveaxis)
            # This matches the behavior of .T on a numpy array
            return np.transpose(data, axes=self._transpose_axes)

        elif isinstance(idx, list | np.ndarray):
            # Fancy indexing: load samples one at a time (h5py limitation)
            # This is slower but necessary for compatibility
            samples = [self[i] for i in idx]
            return np.stack(samples, axis=0)

        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def close(self):
        """Close the underlying file handle if we own it."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


class MATSource(DataSource):
    """
    Load data from MATLAB .mat files (v7.3+ only, which uses HDF5 format).

    Note: MAT v7.3 files are HDF5 files under the hood, so we use h5py for
    memory-efficient lazy loading. Save with: save('file.mat', '-v7.3')

    Supports MATLAB sparse matrices (automatically converted to dense).

    For older MAT files (v5/v7), convert to NPZ or save with -v7.3 flag.
    """

    @staticmethod
    def _is_sparse_dataset(dataset) -> bool:
        """Check if an HDF5 dataset/group represents a MATLAB sparse matrix."""
        # MATLAB v7.3 stores sparse matrices as groups with 'data', 'ir', 'jc' keys
        if hasattr(dataset, "keys"):
            keys = set(dataset.keys())
            return {"data", "ir", "jc"}.issubset(keys)
        return False

    @staticmethod
    def _load_sparse_to_dense(group) -> np.ndarray:
        """Convert MATLAB sparse matrix (CSC format in HDF5) to dense numpy array."""
        from scipy.sparse import csc_matrix

        data = np.array(group["data"])
        ir = np.array(group["ir"])  # row indices
        jc = np.array(group["jc"])  # column pointers

        # Get shape from MATLAB attributes or infer
        if "MATLAB_sparse" in group.attrs:
            nrows = group.attrs["MATLAB_sparse"]
        else:
            nrows = ir.max() + 1 if len(ir) > 0 else 0
        ncols = len(jc) - 1

        sparse_mat = csc_matrix((data, ir, jc), shape=(nrows, ncols))
        return sparse_mat.toarray()

    def _load_dataset(self, f, key: str) -> np.ndarray:
        """Load a dataset, handling sparse matrices automatically."""
        dataset = f[key]

        if self._is_sparse_dataset(dataset):
            # Sparse matrix: convert to dense
            arr = self._load_sparse_to_dense(dataset)
        else:
            # Regular dense array
            arr = np.array(dataset)

        # Transpose for MATLAB column-major -> Python row-major
        return arr.T

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load MAT v7.3 file using h5py."""
        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())

                input_key = self._find_key(keys, INPUT_KEYS)
                output_key = self._find_key(keys, OUTPUT_KEYS)

                if input_key is None or output_key is None:
                    raise KeyError(
                        f"MAT file must contain input and output arrays. "
                        f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                        f"Found: {keys}"
                    )

                # Load with sparse matrix support
                inp = self._load_dataset(f, input_key)
                outp = self._load_dataset(f, output_key)

                # Handle transposed outputs from MATLAB.
                # Case 1: (1, N) - N samples with 1 target → transpose to (N, 1)
                # Case 2: (T, 1) - 1 sample with T targets → transpose to (1, T)
                num_samples = inp.shape[0]  # inp is already transposed
                if outp.ndim == 2:
                    if outp.shape[0] == 1 and outp.shape[1] == num_samples:
                        # 1D vector: (1, N) → (N, 1)
                        outp = outp.T
                    elif outp.shape[1] == 1 and outp.shape[0] != num_samples:
                        # Single sample with multiple targets: (T, 1) → (1, T)
                        outp = outp.T

        except OSError as e:
            raise ValueError(
                f"Failed to load MAT file: {path}. "
                f"Ensure it's saved as v7.3: save('file.mat', '-v7.3'). "
                f"Original error: {e}"
            )

        return inp, outp

    def load_mmap(self, path: str) -> LazyDataHandle:
        """
        Load MAT v7.3 file with lazy/memory-mapped access.

        Returns a LazyDataHandle that reads from disk on-demand,
        avoiding loading the entire file into RAM.

        Note: For sparse matrices, this will load and convert them.
        For dense arrays, returns a transposed view wrapper for consistent axis ordering.

        Usage:
            with source.load_mmap(path) as (inputs, outputs):
                # Use inputs and outputs
                pass  # File automatically closed
        """
        try:
            f = h5py.File(path, "r")  # Keep file open for lazy access
            keys = list(f.keys())

            input_key = self._find_key(keys, INPUT_KEYS)
            output_key = self._find_key(keys, OUTPUT_KEYS)

            if input_key is None or output_key is None:
                f.close()
                raise KeyError(
                    f"MAT file must contain input and output arrays. "
                    f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                    f"Found: {keys}"
                )

            # Check for sparse matrices - must load them eagerly
            inp_dataset = f[input_key]
            outp_dataset = f[output_key]

            if self._is_sparse_dataset(inp_dataset):
                inp = self._load_sparse_to_dense(inp_dataset).T
            else:
                # Wrap h5py dataset with transpose view for consistent axis order
                # MATLAB stores column-major, Python expects row-major
                # Pass file handle to keep it alive
                inp = _TransposedH5Dataset(inp_dataset, file_handle=f)

            if self._is_sparse_dataset(outp_dataset):
                outp = self._load_sparse_to_dense(outp_dataset).T
            else:
                # Wrap h5py dataset with transpose view (shares same file handle)
                outp = _TransposedH5Dataset(outp_dataset, file_handle=f)

            # Return wrapped handle for proper cleanup
            return LazyDataHandle(inp, outp, file_handle=f)

        except OSError as e:
            raise ValueError(
                f"Failed to load MAT file: {path}. "
                f"Ensure it's saved as v7.3: save('file.mat', '-v7.3'). "
                f"Original error: {e}"
            )

    def load_outputs_only(self, path: str) -> np.ndarray:
        """Load only targets from MAT v7.3 file (avoids loading large input arrays)."""
        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())

                output_key = self._find_key(keys, OUTPUT_KEYS)
                if output_key is None:
                    raise KeyError(
                        f"MAT file must contain output array. "
                        f"Supported keys: {OUTPUT_KEYS}. Found: {keys}"
                    )

                # Load with sparse matrix support
                outp = self._load_dataset(f, output_key)

                # Handle 1D outputs that become (1, N) after transpose.
                # Note: This method has no input to compare against, so we can't
                # distinguish single-sample outputs. This is acceptable for training
                # data where single-sample is unlikely. For inference, use load_test_data.
                if outp.ndim == 2 and outp.shape[0] == 1:
                    outp = outp.T

        except OSError as e:
            raise ValueError(
                f"Failed to load MAT file: {path}. "
                f"Ensure it's saved as v7.3: save('file.mat', '-v7.3'). "
                f"Original error: {e}"
            )

        return outp


def get_data_source(format: str, *, allow_pickle: bool = False) -> DataSource:
    """
    Factory function to get the appropriate DataSource for a format.

    Args:
        format: One of 'npz', 'hdf5', 'mat'

    Returns:
        DataSource instance
    """
    sources = {
        "npz": NPZSource,
        "hdf5": HDF5Source,
        "mat": MATSource,
    }

    if format not in sources:
        raise ValueError(
            f"Unsupported format: {format}. Supported: {list(sources.keys())}"
        )

    if format == "npz":
        return NPZSource(allow_pickle=allow_pickle)
    return sources[format]()


def load_training_data(
    path: str, format: str = "auto", *, allow_pickle: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training data from file with automatic format detection.

    Supports:
        - NPZ: NumPy compressed archives (.npz)
        - HDF5: Hierarchical Data Format (.h5, .hdf5)
        - MAT: MATLAB files (.mat)

    Flexible key detection supports: input_train/X/data and output_train/y/labels.

    Args:
        path: Path to data file
        format: Format hint ('npz', 'hdf5', 'mat', or 'auto' for detection)

    Returns:
        Tuple of (inputs, outputs) arrays
    """
    if format == "auto":
        format = DataSource.detect_format(path)

    source = get_data_source(format, allow_pickle=allow_pickle)
    return source.load(path)


def load_outputs_only(
    path: str, format: str = "auto", *, allow_pickle: bool = False
) -> np.ndarray:
    """
    Load only output/target arrays from file (memory-efficient).

    This function avoids loading large input arrays when only targets are needed,
    which is critical for HPC environments with memory constraints during DDP.

    Args:
        path: Path to data file
        format: Format hint ('npz', 'hdf5', 'mat', or 'auto' for detection)

    Returns:
        Output/target array
    """
    if format == "auto":
        format = DataSource.detect_format(path)

    source = get_data_source(format, allow_pickle=allow_pickle)
    return source.load_outputs_only(path)


def _load_npz_for_test(
    path: str,
    input_keys: list[str],
    output_keys: list[str],
    explicit_input_key: str | None,
    explicit_output_key: str | None,
    *,
    allow_pickle: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load NPZ file for test/inference with key validation."""
    with np.load(path, allow_pickle=False) as probe:
        keys = list(probe.keys())

    inp_key = DataSource._find_key(keys, input_keys)
    out_key = DataSource._find_key(keys, output_keys)

    # Strict validation for explicit keys
    if explicit_input_key is not None and explicit_input_key not in keys:
        raise KeyError(
            f"Explicit --input_key '{explicit_input_key}' not found. Available: {keys}"
        )
    if inp_key is None:
        raise KeyError(f"Input key not found. Tried: {input_keys}. Found: {keys}")
    if explicit_output_key is not None and explicit_output_key not in keys:
        raise KeyError(
            f"Explicit --output_key '{explicit_output_key}' not found. Available: {keys}"
        )

    data = NPZSource._load_and_copy(
        path, [inp_key] + ([out_key] if out_key else []), allow_pickle=allow_pickle
    )
    inp = data[inp_key]
    if inp.dtype == object:
        inp = np.array([x.toarray() if hasattr(x, "toarray") else x for x in inp])
    outp = data[out_key] if out_key else None
    return inp, outp


def _load_hdf5_for_test(
    path: str,
    input_keys: list[str],
    output_keys: list[str],
    explicit_input_key: str | None,
    explicit_output_key: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load HDF5 file for test/inference with key validation and OOM guard."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        inp_key = DataSource._find_key(keys, input_keys)
        out_key = DataSource._find_key(keys, output_keys)

        # Strict validation
        if explicit_input_key is not None and explicit_input_key not in keys:
            raise KeyError(
                f"Explicit --input_key '{explicit_input_key}' not found. Available: {keys}"
            )
        if inp_key is None:
            raise KeyError(f"Input key not found. Tried: {input_keys}. Found: {keys}")
        if explicit_output_key is not None and explicit_output_key not in keys:
            raise KeyError(
                f"Explicit --output_key '{explicit_output_key}' not found. Available: {keys}"
            )

        # OOM guard
        n_samples = f[inp_key].shape[0]
        if n_samples > 100000:
            raise ValueError(
                f"Dataset has {n_samples:,} samples. load_test_data() loads "
                f"everything into RAM which may cause OOM. For large inference "
                f"sets, use a DataLoader with HDF5Source.load_mmap() instead."
            )

        inp = f[inp_key][:]
        outp = f[out_key][:] if out_key else None
    return inp, outp


def _load_mat_for_test(
    path: str,
    input_keys: list[str],
    output_keys: list[str],
    explicit_input_key: str | None,
    explicit_output_key: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load MAT v7.3 file for test/inference with sparse support."""
    mat_source = MATSource()
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        inp_key = DataSource._find_key(keys, input_keys)
        out_key = DataSource._find_key(keys, output_keys)

        # Strict validation
        if explicit_input_key is not None and explicit_input_key not in keys:
            raise KeyError(
                f"Explicit --input_key '{explicit_input_key}' not found. Available: {keys}"
            )
        if inp_key is None:
            raise KeyError(f"Input key not found. Tried: {input_keys}. Found: {keys}")
        if explicit_output_key is not None and explicit_output_key not in keys:
            raise KeyError(
                f"Explicit --output_key '{explicit_output_key}' not found. Available: {keys}"
            )

        # OOM guard
        input_dataset = f[inp_key]
        n_samples = (
            len(input_dataset["jc"]) - 1
            if mat_source._is_sparse_dataset(input_dataset)
            else input_dataset.shape[-1]
        )  # MAT is transposed
        if n_samples > 100000:
            raise ValueError(
                f"Dataset has {n_samples:,} samples. load_test_data() loads "
                f"everything into RAM which may cause OOM. For large inference "
                f"sets, use a DataLoader with MATSource.load_mmap() instead."
            )

        inp = mat_source._load_dataset(f, inp_key)
        outp = None
        if out_key:
            outp = mat_source._load_dataset(f, out_key)
            # Handle MATLAB transpose
            num_samples = inp.shape[0]
            if outp.ndim == 2:
                if (outp.shape[0] == 1 and outp.shape[1] == num_samples) or (
                    outp.shape[1] == 1 and outp.shape[0] != num_samples
                ):
                    outp = outp.T
    return inp, outp


def load_test_data(
    path: str,
    format: str = "auto",
    input_key: str | None = None,
    output_key: str | None = None,
    input_channels: int | None = None,
    *,
    allow_pickle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Load test/inference data and return PyTorch tensors ready for model input.

    This is the unified data loading function for inference. It:
    - Auto-detects file format from extension
    - Handles custom key names for non-standard datasets
    - Adds channel dimension if missing (dimension-agnostic)
    - Returns None for targets if not present in file

    Supports any input dimensionality:
        - 1D: (N, L) → (N, 1, L)
        - 2D: (N, H, W) → (N, 1, H, W)
        - 3D: (N, D, H, W) → (N, 1, D, H, W)
        - Already has channel: (N, C, ...) → unchanged

    Args:
        path: Path to data file (NPZ, HDF5, or MAT v7.3)
        format: Format hint ('npz', 'hdf5', 'mat', or 'auto' for detection)
        input_key: Custom key for input data (overrides auto-detection)
        output_key: Custom key for output data (overrides auto-detection)
        input_channels: Explicit number of input channels. If provided, bypasses
            the heuristic for 4D data. Use input_channels=1 for 3D volumes that
            look like multi-channel 2D (e.g., depth ≤16).

    Returns:
        Tuple of:
            - X: Input tensor with channel dimension (N, 1, *spatial_dims)
            - y: Target tensor ``(N,T)`` or ``(N,C,H,W)``, or None if absent

    Example:
        >>> X, y = load_test_data("test_data.npz")
        >>> X, y = load_test_data(
        ...     "data.mat", input_key="waveforms", output_key="params"
        ... )
    """
    if format == "auto":
        format = DataSource.detect_format(path)

    source = get_data_source(format, allow_pickle=allow_pickle)

    # Build custom key lists if provided
    if input_key:
        custom_input_keys = [input_key] + INPUT_KEYS
    else:
        # Prioritize test keys for inference
        custom_input_keys = ["input_test"] + [
            k for k in INPUT_KEYS if k != "input_test"
        ]

    if output_key:
        custom_output_keys = [output_key] + OUTPUT_KEYS
    else:
        custom_output_keys = ["output_test"] + [
            k for k in OUTPUT_KEYS if k != "output_test"
        ]

    # Load data using appropriate source with test-key priority
    # We detect keys first to ensure input_test/output_test are used when present
    try:
        if format == "npz":
            inp, outp = _load_npz_for_test(
                path,
                custom_input_keys,
                custom_output_keys,
                input_key,
                output_key,
                allow_pickle=allow_pickle,
            )
        elif format == "hdf5":
            inp, outp = _load_hdf5_for_test(
                path, custom_input_keys, custom_output_keys, input_key, output_key
            )
        elif format == "mat":
            inp, outp = _load_mat_for_test(
                path, custom_input_keys, custom_output_keys, input_key, output_key
            )
        else:
            # Fallback to default source.load() for unknown formats
            inp, outp = source.load(path)
    except KeyError as e:
        # IMPORTANT: Only fall back to inference-only mode if outputs are
        # genuinely missing (auto-detection failed). If user explicitly
        # provided --output_key, they expect it to exist - don't silently drop.
        if output_key is not None:
            raise KeyError(
                f"Explicit --output_key '{output_key}' not found in file. "
                f"Available keys depend on file format. Original error: {e}"
            ) from e

        # Also fail-fast if explicit input_key was provided but not found
        # This prevents silently loading a different tensor when user mistyped key
        if input_key is not None:
            raise KeyError(
                f"Explicit --input_key '{input_key}' not found in file. "
                f"Original error: {e}"
            ) from e

        # Legitimate fallback: no explicit keys, outputs just not present
        # Re-call helpers with None for explicit keys (validation already done above)
        if format == "npz":
            inp, outp = _load_npz_for_test(
                path,
                custom_input_keys,
                custom_output_keys,
                None,
                None,
                allow_pickle=allow_pickle,
            )
        elif format == "hdf5":
            inp, outp = _load_hdf5_for_test(
                path, custom_input_keys, custom_output_keys, None, None
            )
        elif format == "mat":
            inp, outp = _load_mat_for_test(
                path, custom_input_keys, custom_output_keys, None, None
            )
        else:
            raise

    # Handle sparse matrices
    if issparse(inp):
        inp = inp.toarray()
    if outp is not None and issparse(outp):
        outp = outp.toarray()

    # Convert to tensors
    inp = _as_finite_float32(inp, "Inputs")
    if not 2 <= inp.ndim <= 5 or any(size == 0 for size in inp.shape):
        raise ValueError(
            f"Inputs must be a non-empty array of rank 2-5, got {inp.shape}."
        )
    X = torch.from_numpy(np.ascontiguousarray(inp))

    if outp is not None:
        # Vector targets stay (N,T); scalar and multi-property fields use the
        # canonical channels-first layout (N,C,H,W).
        outp = canonicalize_targets(outp)
        if len(outp) != len(inp):
            raise ValueError(
                f"Input/output sample mismatch: {len(inp)} inputs but {len(outp)} targets."
            )
        y = torch.from_numpy(np.ascontiguousarray(_as_finite_float32(outp, "Targets")))
    else:
        y = None

    # Add channel dimension if needed (dimension-agnostic)
    # X.ndim == 2: 1D data (N, L) → (N, 1, L)
    # X.ndim == 3: 2D data (N, H, W) → (N, 1, H, W)
    # X.ndim == 4: Check if already has channel dim
    if X.ndim == 2:
        X = X.unsqueeze(1)  # 1D signal: (N, L) → (N, 1, L)
    elif X.ndim == 3:
        X = X.unsqueeze(1)  # 2D image: (N, H, W) → (N, 1, H, W)
    elif X.ndim == 4:
        # Could be 3D volume (N, D, H, W) or 2D with channel (N, C, H, W)
        if input_channels is not None:
            # Explicit override: user specifies channel count
            if input_channels == 1:
                X = X.unsqueeze(1)  # Add channel: (N, D, H, W) → (N, 1, D, H, W)
            # else: already has channels, leave as-is
        else:
            # Detect channels-last format: (N, H, W, C) where C is small (1-4)
            # and spatial dims are large (>16). This catches common mistakes.
            if X.shape[-1] <= 4 and X.shape[1] > 16 and X.shape[2] > 16:
                raise ValueError(
                    f"Input appears to be channels-last format: {tuple(X.shape)}. "
                    "WaveDL expects channels-first (N, C, H, W). "
                    "Convert your data using: X = X.permute(0, 3, 1, 2). "
                    "If this is actually a 3D volume with small depth, "
                    "use --input_channels 1 to add a channel dimension."
                )
            elif X.shape[1] > 16:
                # Heuristic fallback: large dim 1 suggests 3D volume needing channel
                X = X.unsqueeze(1)  # 3D volume: (N, D, H, W) → (N, 1, D, H, W)
            else:
                # Ambiguous case: shallow 3D volume (D <= 16) or multi-channel 2D
                # Default to treating as multi-channel 2D (no modification needed)
                # Log a warning so users know about the --input_channels option
                import warnings

                warnings.warn(
                    f"Ambiguous 4D input shape: {tuple(X.shape)}. "
                    f"Assuming {X.shape[1]} channels (multi-channel 2D). "
                    f"For 3D volumes with depth={X.shape[1]}, use --input_channels 1.",
                    UserWarning,
                    stacklevel=2,
                )
    # X.ndim >= 5: assume channel dimension already exists

    return X, y


# ==============================================================================
# WORKER INITIALIZATION
# ==============================================================================
def memmap_worker_init_fn(worker_id: int):
    """
    Worker initialization function for proper memmap handling in multi-worker DataLoader.

    Each DataLoader worker process runs this function after forking. It:
    1. Resets the memmap file handle to None, forcing each worker to open its own
       read-only handle (prevents file descriptor sharing issues and race conditions)
    2. Seeds numpy's random state per worker to ensure statistical diversity in
       random augmentations (prevents all workers from applying identical "random"
       transformations to their batches)

    Args:
        worker_id: Worker index (0 to num_workers-1), provided by DataLoader

    Usage:
        DataLoader(dataset, num_workers=8, worker_init_fn=memmap_worker_init_fn)
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Force re-initialization of memmap in each worker
        dataset.data = None
        if hasattr(dataset, "target_data"):
            dataset.target_data = None

        # Seed numpy RNG per worker using PyTorch's worker seed for reproducibility
        # This ensures random augmentations (noise, shifts, etc.) are unique per worker
        np.random.seed(worker_info.seed % (2**32))


# ==============================================================================
# MEMORY-MAPPED DATASET
# ==============================================================================
class MemmapDataset(Dataset):
    """
    Zero-copy memory-mapped dataset for large-scale training.

    Uses numpy memory mapping to load data directly from disk, allowing training
    on datasets that exceed available RAM. The memmap is only opened when first
    accessed (lazy initialization), and each DataLoader worker maintains its own
    file handle for thread safety.

    Args:
        memmap_path: Path to the memory-mapped data file
        targets: Pre-loaded target tensor (small enough to fit in memory)
        shape: Full shape of the memmap array (N, C, H, W)
        indices: Indices into the memmap for this split (train/val)

    Thread Safety:
        When using with DataLoader num_workers > 0, must use memmap_worker_init_fn
        as the worker_init_fn to ensure each worker gets its own file handle.

    Example:
        dataset = MemmapDataset("cache.dat", y_tensor, (10000, 1, 500, 500), train_indices)
        loader = DataLoader(dataset, num_workers=8, worker_init_fn=memmap_worker_init_fn)
    """

    def __init__(
        self,
        memmap_path: str,
        targets: torch.Tensor | None,
        shape: tuple[int, ...],
        indices: np.ndarray,
        target_memmap_path: str | None = None,
        target_shape: tuple[int, ...] | None = None,
    ):
        if targets is None and (target_memmap_path is None or target_shape is None):
            raise ValueError(
                "Provide either in-memory targets or target_memmap_path + target_shape."
            )
        self.memmap_path = memmap_path
        self.targets = targets
        self.shape = shape
        self.indices = indices
        self.data: np.memmap | None = None  # Lazy initialization
        self.target_memmap_path = target_memmap_path
        self.target_shape = target_shape
        self.target_data: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            # Mode 'r' = read-only, prevents accidental data modification
            self.data = np.memmap(
                self.memmap_path, dtype="float32", mode="r", shape=self.shape
            )

        real_idx = self.indices[idx]

        # .copy() detaches from mmap buffer - essential for PyTorch pinned memory
        x = torch.from_numpy(self.data[real_idx].copy()).contiguous()
        if self.target_memmap_path is not None:
            if self.target_data is None:
                target_full_shape = (self.shape[0], *self.target_shape)
                self.target_data = np.memmap(
                    self.target_memmap_path,
                    dtype="float32",
                    mode="r",
                    shape=target_full_shape,
                )
            y = torch.from_numpy(self.target_data[real_idx].copy()).contiguous()
        else:
            y = self.targets[real_idx]

        return x, y

    def __repr__(self) -> str:
        return (
            f"MemmapDataset(path='{self.memmap_path}', "
            f"samples={len(self)}, shape={self.shape})"
        )


# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def _iter_target_batches(targets: Any, indices: np.ndarray, batch_size: int = 256):
    """Yield canonical target batches restricted to ``indices``.

    Contiguous slices are read and then masked, so HDF5/MAT sources stream
    sequentially instead of issuing per-sample point selections. Chunks with
    no selected samples are skipped without being read.
    """

    batch_size = _bounded_batch_size(targets.shape, targets.dtype, batch_size)
    selected = np.zeros(len(targets), dtype=bool)
    selected[np.asarray(indices, dtype=np.int64)] = True
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        mask = selected[start:stop]
        if not mask.any():
            continue
        batch = np.asarray(targets[start:stop])
        yield canonicalize_targets(batch if mask.all() else batch[mask])


def _validate_scaler_compatibility(current: Any, expected: Any) -> None:
    """Reject a resume that changes the model's standardized target space."""
    current_shape = tuple(getattr(current, "target_shape_", (len(current.mean_),)))
    expected_shape = tuple(getattr(expected, "target_shape_", (len(expected.mean_),)))
    if (
        len(expected_shape) == 3
        and getattr(expected, "normalization_version_", 0)
        != SpatialTargetScaler.NORMALIZATION_VERSION
    ):
        raise ValueError(
            "The dense checkpoint uses an older target normalization version. "
            "Dense normalization now preserves large physical offsets accurately. "
            "Use --fresh or a separate output directory to retrain this field "
            "model; its original normalization cannot be reused safely."
        )
    compatible = current_shape == expected_shape
    for name in ("mean_", "scale_"):
        current_stat = np.asarray(getattr(current, name, None))
        expected_stat = np.asarray(getattr(expected, name, None))
        compatible = compatible and current_stat.shape == expected_stat.shape
        if compatible:
            compatible = bool(
                np.allclose(current_stat, expected_stat, rtol=1e-10, atol=1e-12)
            )
    if not compatible:
        raise ValueError(
            "Target normalization changed since the checkpoint was saved. "
            "Resuming with a different scaler would change the meaning of model "
            "outputs. Use --fresh or a separate output directory for this dataset "
            "and training split."
        )


def _fit_resume_scaler(args: Any, val_split: float) -> Any:
    """Check new split statistics before modifying an existing run's cache."""
    data_format = DataSource.detect_format(args.data_path)
    with get_data_source(
        data_format, allow_pickle=getattr(args, "allow_pickle", False)
    ).load_mmap(args.data_path) as (inputs, targets):
        num_samples = len(inputs)
        if data_format == "mat" and targets.ndim == 2:
            targets = np.asarray(targets[:])
            if (targets.shape[0] == 1 and targets.shape[1] == num_samples) or (
                targets.shape[1] == 1 and targets.shape[0] != num_samples
            ):
                targets = targets.T
        if len(targets) != num_samples:
            raise ValueError(
                f"Input/output sample mismatch: {num_samples} inputs but "
                f"{len(targets)} targets."
            )
        train_indices, _ = train_test_split(
            np.arange(num_samples), test_size=val_split, random_state=args.seed
        )
        target_shape = canonical_target_shape(tuple(targets.shape))
        if len(target_shape) == 3:
            return SpatialTargetScaler().fit_batches(
                _iter_target_batches(targets, train_indices)
            )
        return create_target_scaler(np.asarray(targets[:])[train_indices])


def prepare_data(
    args: Any,
    logger: logging.Logger,
    accelerator: Accelerator,
    cache_dir: str = ".",
    val_split: float = 0.2,
    *,
    expected_scaler: Any = None,
) -> tuple[DataLoader, DataLoader, Any, tuple[int, ...], int]:
    """
    Prepare DataLoaders with DDP synchronization guarantees.

    This function handles:
    1. Loading raw data and creating memmap cache (Rank 0 only)
    2. Fitting a vector or channel-wise field scaler on training data only
    3. Synchronizing all ranks before proceeding
    4. Creating thread-safe DataLoaders for DDP training

    Supports any input dimensionality:
        - 1D: (N, L) → returns in_shape = (L,)
        - 2D: (N, H, W) → returns in_shape = (H, W)
        - 3D: (N, D, H, W) → returns in_shape = (D, H, W)

    Args:
        args: Argument namespace with data_path, seed, batch_size, workers
        logger: Logger instance for status messages
        accelerator: Accelerator instance for DDP coordination
        cache_dir: Directory for cache files (default: current directory)
        val_split: Validation set fraction (default: 0.2)
        expected_scaler: Checkpoint scaler to preserve when resuming. Changed
            statistics are rejected before any existing cache files are removed.

    Returns:
        Tuple of:
            - train_dl: Training DataLoader
            - val_dl: Validation DataLoader
            - scaler: Fitted target scaler (for inverse transforms)
            - in_shape: Input spatial dimensions - (L,), (H, W), or (D, H, W)
            - out_dim: Number of vector targets or dense output channels

    Cache Files Created:
        - train_data_cache.dat: Memory-mapped input data
        - target_data_cache.dat: Memory-mapped scaled dense targets (field mode)
        - scaler.pkl: Fitted target scaler
        - data_metadata.pkl: Shape and dimension metadata
    """
    CACHE_FILE = os.path.join(cache_dir, "train_data_cache.dat")
    TARGET_CACHE_FILE = os.path.join(cache_dir, "target_data_cache.dat")
    SCALER_FILE = os.path.join(cache_dir, "scaler.pkl")
    META_FILE = os.path.join(cache_dir, "data_metadata.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    # ==========================================================================
    # PHASE 1: DATA GENERATION (Rank 0 Only)
    # ==========================================================================
    # Check cache existence and validity (data path must match)
    cache_exists = (
        os.path.exists(CACHE_FILE)
        and os.path.exists(SCALER_FILE)
        and os.path.exists(META_FILE)
    )

    # Validate cache using content hash (portable across folders/machines)
    # File size is a fast pre-check, content hash is definitive validation
    if cache_exists:
        try:
            with open(META_FILE, "rb") as f:
                meta = pickle.load(f)
            cached_file_size = meta.get("file_size", None)
            cached_content_hash = meta.get("content_hash", None)
            if meta.get("allow_pickle", True) != getattr(args, "allow_pickle", False):
                cache_exists = False

            if (
                meta.get("output_kind") == "field"
                and meta.get("target_normalization_version", 0)
                != SpatialTargetScaler.NORMALIZATION_VERSION
            ):
                cache_exists = False

            if meta.get("output_kind") == "field" and not os.path.exists(
                TARGET_CACHE_FILE
            ):
                cache_exists = False

            if cache_exists:
                expected_input_bytes = (
                    int(np.prod(meta["shape"])) * np.dtype(np.float32).itemsize
                )
                if os.path.getsize(CACHE_FILE) != expected_input_bytes:
                    cache_exists = False

            if cache_exists and meta.get("output_kind") == "field":
                target_cache_shape = tuple(meta.get("target_cache_shape", ()))
                expected_target_bytes = (
                    int(np.prod(target_cache_shape)) * np.dtype(np.float32).itemsize
                )
                if (
                    not target_cache_shape
                    or os.path.getsize(TARGET_CACHE_FILE) != expected_target_bytes
                ):
                    if accelerator.is_main_process:
                        logger.warning(
                            "⚠️  Dense target cache has an unexpected size; "
                            "regenerating it..."
                        )
                    cache_exists = False

            # Get current file stats
            current_stats = os.stat(args.data_path)
            current_size = current_stats.st_size

            # Check if file size changed (fast check before expensive hash)
            if cached_file_size is not None and cached_file_size != current_size:
                if accelerator.is_main_process:
                    logger.warning(
                        f"⚠️  Data file size changed!\n"
                        f"   Cached size: {cached_file_size:,} bytes\n"
                        f"   Current size: {current_size:,} bytes\n"
                        f"   Invalidating cache and regenerating..."
                    )
                cache_exists = False
            # Content hash check (robust against cloud sync mtime changes)
            elif cached_content_hash is not None:
                current_hash = _compute_file_hash(
                    args.data_path, mode=getattr(args, "cache_validate", "sha256")
                )
                if cached_content_hash != current_hash:
                    if accelerator.is_main_process:
                        logger.warning(
                            "⚠️  Data file content changed!\n"
                            "   Cache is stale, regenerating..."
                        )
                    cache_exists = False

            # The scaler is fit on the training split, which depends on val_split
            # and seed. If either changed since the cache was built, reusing the
            # cached scaler would fit it on samples now in the validation set
            # (leakage). Regenerate so the scaler matches the current split.
            if cache_exists:
                cached_val_split = meta.get("val_split", None)
                cached_seed = meta.get("seed", None)
                if (cached_val_split is not None and cached_val_split != val_split) or (
                    cached_seed is not None
                    and cached_seed != getattr(args, "seed", None)
                ):
                    if accelerator.is_main_process:
                        logger.warning(
                            "⚠️  val_split/seed changed since cache was built; "
                            "regenerating to refit the scaler on the correct "
                            "training split (avoids validation-set leakage)..."
                        )
                    cache_exists = False

            # Verify completion sentinel — catches interrupted cache writes
            if cache_exists and not meta.get("cache_complete", False):
                if accelerator.is_main_process:
                    logger.info(
                        "   Cache incomplete (interrupted write). Regenerating..."
                    )
                cache_exists = False

            # Validate scaler file integrity early (H4)
            if cache_exists:
                try:
                    with open(SCALER_FILE, "rb") as f:
                        _test_scaler = pickle.load(f)
                    if (
                        not hasattr(_test_scaler, "scale_")
                        or _test_scaler.scale_ is None
                    ):
                        raise ValueError("Scaler not fitted")
                    del _test_scaler
                except Exception:
                    if accelerator.is_main_process:
                        logger.warning(
                            "⚠️  Corrupt scaler file detected. Regenerating cache..."
                        )
                    cache_exists = False

        except Exception as e:
            logging.debug(f"Cache validation failed ({e}). Regenerating...")
            cache_exists = False

    if expected_scaler is not None:
        validation_error = None
        if accelerator.is_main_process:
            try:
                if cache_exists:
                    with open(SCALER_FILE, "rb") as f:
                        _validate_scaler_compatibility(pickle.load(f), expected_scaler)
                else:
                    # This extra streaming pass is restricted to a resume that
                    # needs a cache rebuild. Validate before invalidation, so a
                    # rejected resume preserves the original scaler and units.
                    _validate_scaler_compatibility(
                        _fit_resume_scaler(args, val_split), expected_scaler
                    )
            except Exception as exc:
                validation_error = exc
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            error_messages = [str(validation_error) if validation_error else None]
            torch.distributed.broadcast_object_list(error_messages, src=0)
            if error_messages[0] is not None and validation_error is None:
                validation_error = ValueError(error_messages[0])
        if validation_error is not None:
            raise validation_error

    # Invalidate the old completion sentinel before any rank starts waiting.
    # The barrier closes a race where non-main ranks could otherwise observe
    # stale metadata while rank 0 was recreating an interrupted dense cache.
    if not cache_exists and accelerator.is_main_process and os.path.exists(META_FILE):
        os.remove(META_FILE)
    wait_for_everyone(accelerator)

    if not cache_exists:
        if accelerator.is_main_process:
            # Delete stale cache files to force regeneration
            # This prevents silent reuse of old data when metadata invalidates cache
            for stale_file in [META_FILE, CACHE_FILE, TARGET_CACHE_FILE, SCALER_FILE]:
                if os.path.exists(stale_file):
                    try:
                        os.remove(stale_file)
                        logger.debug(f"   Removed stale cache: {stale_file}")
                    except OSError as e:
                        logger.warning(
                            f"   Failed to remove stale cache {stale_file}: {e}"
                        )

            # Fail explicitly if stale cache files couldn't be removed
            # This prevents silent reuse of outdated data
            remaining_stale = [
                f
                for f in [META_FILE, CACHE_FILE, TARGET_CACHE_FILE, SCALER_FILE]
                if os.path.exists(f)
            ]
            if remaining_stale:
                raise RuntimeError(
                    f"Cannot regenerate cache: stale files could not be removed. "
                    f"Please manually delete: {remaining_stale}"
                )

            # RANK 0: Create cache (can take a long time for large datasets)
            # Other ranks will wait at the barrier below

            # Detect format from extension
            data_format = DataSource.detect_format(args.data_path)
            logger.info(
                f"⚡ [Rank 0] Initializing Data Processing from: {args.data_path} (format: {data_format})"
            )

            # Validate data file exists
            if not os.path.exists(args.data_path):
                raise FileNotFoundError(
                    f"CRITICAL: Data file not found: {args.data_path}"
                )

            # HDF5/MAT stream slices; NPZ materializes its compressed arrays.
            # All load_mmap() methods return a common data handle.
            _lazy_handle = None
            try:
                source = get_data_source(
                    data_format, allow_pickle=getattr(args, "allow_pickle", False)
                )
                if hasattr(source, "load_mmap"):
                    _lazy_handle = source.load_mmap(args.data_path)
                    inp, outp = _lazy_handle.inputs, _lazy_handle.outputs
                    logger.info(
                        "   Loading NPZ arrays into host memory before caching"
                        if data_format == "npz"
                        else "   Using chunked loading before caching"
                    )
                else:
                    inp, outp = load_training_data(args.data_path, format=data_format)
                    logger.info("   Using eager loading (full data in memory)")
            except Exception as e:
                logger.error(f"Failed to load data file: {e}")
                raise

            try:
                # Detect shape (handle sparse matrices) - DIMENSION AGNOSTIC
                num_samples = len(inp)
                if num_samples < 2:
                    raise ValueError("Training requires at least two input samples.")

                # MAT files store outputs column-major. The eager MATSource.load() path
                # normalizes transposed MATLAB vectors, but the mmap path returns a raw
                # transposed view with NO normalization. Apply the same normalization
                # here so out_dim and the scaler are correct — otherwise a single target
                # saved as a (1, N) row vector yields out_dim=N and crashes scaler fitting.
                if data_format == "mat" and len(getattr(outp, "shape", ())) == 2:
                    outp = np.asarray(outp[:])
                    if outp.shape[0] == 1 and outp.shape[1] == num_samples:
                        outp = outp.T  # (1, N) -> (N, 1): N samples, 1 target
                    elif outp.shape[1] == 1 and outp.shape[0] != num_samples:
                        outp = outp.T  # (T, 1) -> (1, T): 1 sample, T targets

                if len(outp) != num_samples:
                    raise ValueError(
                        f"Input/output sample mismatch: {num_samples} inputs but "
                        f"{len(outp)} targets."
                    )

                # Canonical target contract:
                #   vector (N,) / (N,T) -> (N,T)
                #   field  (N,H,W)      -> (N,1,H,W)
                #   field  (N,C,H,W)    -> unchanged
                target_shape = canonical_target_shape(tuple(outp.shape))
                out_dim = target_shape[0]
                output_kind = "field" if len(target_shape) == 3 else "vector"

                sample_0 = inp[0]
                if issparse(sample_0) or hasattr(sample_0, "toarray"):
                    sample_0 = sample_0.toarray()

                # Detect dimensionality and validate single-channel assumption
                raw_shape = sample_0.shape
                if not 1 <= len(raw_shape) <= 3 or any(size == 0 for size in raw_shape):
                    raise ValueError(
                        f"Training inputs require 1D, 2D, or 3D non-empty samples, got {raw_shape}."
                    )

                # Heuristic for ambiguous multi-channel detection:
                # Trigger when shape is EXACTLY 3D (could be C,H,W or D,H,W) with:
                #   - First dim small (<=16) - looks like channels
                #   - BOTH remaining dims large (>16) - confirming it's an image, not a tiny patch
                # This distinguishes (3, 256, 256) from (8, 1024) or (128, 128)
                # Use --single_channel to confirm shallow 3D volumes like (8, 128, 128)
                is_ambiguous_shape = (
                    len(raw_shape) == 3  # Exactly 3D: could be (C, H, W) or (D, H, W)
                    and raw_shape[0] <= 16  # First dim looks like channels
                    and raw_shape[1] > 16
                    and raw_shape[2] > 16  # Both spatial dims are large
                )

                # Check for user confirmation via --single_channel flag
                user_confirmed_single_channel = getattr(args, "single_channel", False)

                if is_ambiguous_shape and not user_confirmed_single_channel:
                    raise ValueError(
                        f"Ambiguous input shape detected: sample shape {raw_shape}. "
                        f"This could be either:\n"
                        f"  - Multi-channel 2D data (C={raw_shape[0]}, H={raw_shape[1]}, W={raw_shape[2]})\n"
                        f"  - Single-channel 3D volume (D={raw_shape[0]}, H={raw_shape[1]}, W={raw_shape[2]})\n\n"
                        f"If this is single-channel 3D/shallow volume data, use --single_channel flag.\n"
                        f"If this is multi-channel 2D data, reshape to (N*C, H, W) with adjusted targets."
                    )

                spatial_shape = raw_shape
                full_shape = (
                    num_samples,
                    1,
                ) + spatial_shape  # Add channel dim: (N, 1, ...)

                dim_names = {1: "1D (L)", 2: "2D (H, W)", 3: "3D (D, H, W)"}
                dim_type = dim_names.get(len(spatial_shape), f"{len(spatial_shape)}D")
                logger.info(
                    f"   Shape Detected: {full_shape} [{dim_type}] | "
                    f"Output: {target_shape} [{output_kind}]"
                )

                # Compute metadata now; META_FILE is written LAST (after cache +
                # scaler) as the completion sentinel — see cache_complete flag.
                file_stats = os.stat(args.data_path)
                content_hash = _compute_file_hash(
                    args.data_path, mode=getattr(args, "cache_validate", "sha256")
                )
                _cache_meta = {
                    "shape": full_shape,
                    "out_dim": out_dim,
                    "target_shape": target_shape,
                    "output_kind": output_kind,
                    "target_normalization_version": (
                        SpatialTargetScaler.NORMALIZATION_VERSION
                        if output_kind == "field"
                        else None
                    ),
                    "target_cache_shape": (num_samples, *target_shape),
                    "target_cache_nbytes": int(np.prod((num_samples, *target_shape)))
                    * np.dtype(np.float32).itemsize,
                    "data_path": os.path.abspath(args.data_path),
                    "allow_pickle": getattr(args, "allow_pickle", False),
                    "file_size": file_stats.st_size,
                    "content_hash": content_hash,
                    # Split parameters the scaler fit depends on (see cache validation).
                    "val_split": val_split,
                    "seed": getattr(args, "seed", None),
                }

                # Create memmap cache
                if not os.path.exists(CACHE_FILE):
                    logger.info("   Writing Memmap Cache (one-time operation)...")
                    fp = np.memmap(
                        CACHE_FILE, dtype="float32", mode="w+", shape=full_shape
                    )

                    chunk_size = _bounded_batch_size(full_shape, np.float32, 2000)
                    pbar = tqdm(
                        range(0, num_samples, chunk_size),
                        desc="Caching",
                        disable=not accelerator.is_main_process,
                    )

                    for i in pbar:
                        end = min(i + chunk_size, num_samples)
                        batch = inp[i:end]

                        # Handle sparse/dense conversion
                        if issparse(batch[0]) or hasattr(batch[0], "toarray"):
                            data_chunk = np.stack([x.toarray() for x in batch])
                        else:
                            data_chunk = np.asarray(batch)
                        data_chunk = _as_finite_float32(data_chunk, "Inputs")

                        # Add channel dimension if needed (handles 1D, 2D, and 3D spatial data)
                        # data_chunk shape: (batch, *spatial) -> need (batch, 1, *spatial)
                        # full_shape is (N, 1, *spatial), so expected ndim = len(full_shape)
                        if data_chunk.ndim == len(full_shape) - 1:
                            # Missing channel dim: (batch, *spatial) -> (batch, 1, *spatial)
                            data_chunk = np.expand_dims(data_chunk, 1)

                        fp[i:end] = data_chunk

                        # Periodic flush to disk
                        if i % 10000 == 0:
                            fp.flush()

                    fp.flush()
                    del fp
                    gc.collect()

                # Train/Val split and scaler fitting
                indices = np.arange(num_samples)
                tr_idx, val_idx = train_test_split(
                    indices, test_size=val_split, random_state=args.seed
                )

                if not os.path.exists(SCALER_FILE):
                    logger.info("   Fitting target scaler (training set only)...")

                    if output_kind == "field":
                        scaler = SpatialTargetScaler().fit_batches(
                            _iter_target_batches(outp, tr_idx)
                        )
                    else:
                        # Vector labels are normally small; materialize them for the
                        # exact sklearn StandardScaler behavior used by legacy runs.
                        outp_all = np.asarray(outp[:])
                        if (
                            outp_all.dtype.kind not in "biuf"
                            or not np.isfinite(outp_all).all()
                        ):
                            raise ValueError(
                                "Targets must contain only finite real values."
                            )
                        outp_train = outp_all[tr_idx]
                        scaler = create_target_scaler(canonicalize_targets(outp_train))
                    with open(SCALER_FILE, "wb") as f:
                        pickle.dump(scaler, f)

                if output_kind == "field" and not os.path.exists(TARGET_CACHE_FILE):
                    logger.info("   Writing scaled target memmap cache...")
                    target_full_shape = (num_samples, *target_shape)
                    target_fp = np.memmap(
                        TARGET_CACHE_FILE,
                        dtype="float32",
                        mode="w+",
                        shape=target_full_shape,
                    )
                    target_chunk_size = _bounded_batch_size(outp.shape, outp.dtype, 256)
                    for start in range(0, num_samples, target_chunk_size):
                        end = min(start + target_chunk_size, num_samples)
                        target_batch = canonicalize_targets(outp[start:end])
                        target_fp[start:end] = _as_finite_float32(
                            scaler.transform(target_batch), "Scaled targets"
                        )
                    target_fp.flush()
                    del target_fp

                # Write META_FILE LAST as completion sentinel.
                # Order: CACHE_FILE -> SCALER_FILE -> META_FILE.
                # If any prior step is interrupted, META_FILE won't exist (or
                # will lack cache_complete), so the next run regenerates.
                _cache_meta["cache_complete"] = True
                meta_tmp_file = f"{META_FILE}.tmp"
                with open(meta_tmp_file, "wb") as f:
                    pickle.dump(_cache_meta, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(meta_tmp_file, META_FILE)

            finally:
                if _lazy_handle is not None:
                    _lazy_handle.close()
            del inp, outp
            gc.collect()

            logger.info("   ✔ Cache creation complete, synchronizing ranks...")
        else:
            # NON-MAIN RANKS: Wait for cache creation with timeout
            # Use monotonic clock (immune to system clock changes)
            import time

            wait_start = time.monotonic()

            # Robust env parsing with guards for invalid/non-positive values
            DEFAULT_CACHE_TIMEOUT = 3600  # 1 hour default
            try:
                env_timeout = os.environ.get("WAVEDL_CACHE_TIMEOUT", "")
                CACHE_TIMEOUT = (
                    int(env_timeout) if env_timeout else DEFAULT_CACHE_TIMEOUT
                )
                if CACHE_TIMEOUT <= 0:
                    CACHE_TIMEOUT = DEFAULT_CACHE_TIMEOUT
            except ValueError:
                CACHE_TIMEOUT = DEFAULT_CACHE_TIMEOUT

            while not (
                os.path.exists(CACHE_FILE)
                and os.path.exists(SCALER_FILE)
                and os.path.exists(META_FILE)
            ):
                time.sleep(5)  # Check every 5 seconds
                elapsed = time.monotonic() - wait_start

                if elapsed > CACHE_TIMEOUT:
                    raise RuntimeError(
                        f"[Rank {accelerator.process_index}] Timeout waiting for cache "
                        f"files after {CACHE_TIMEOUT}s. Rank 0 may have failed during "
                        f"cache generation. Check rank 0 logs for errors."
                    )

                if elapsed > 60 and int(elapsed) % 60 < 5:  # Log every ~minute
                    logger.info(
                        f"   [Rank {accelerator.process_index}] Waiting for cache "
                        f"creation... ({int(elapsed)}s / {CACHE_TIMEOUT}s max)"
                    )
            # Small delay to ensure files are fully written
            time.sleep(2)

    # ==========================================================================
    # PHASE 2: SYNCHRONIZED LOADING (All Ranks)
    # ==========================================================================
    wait_for_everyone(accelerator)

    # Load metadata
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
        full_shape = meta["shape"]
        out_dim = meta["out_dim"]
        target_shape = tuple(meta.get("target_shape", (out_dim,)))
        output_kind = meta.get("output_kind", "vector")

    if not meta.get("cache_complete", False):
        raise RuntimeError(
            f"Cache metadata is incomplete: {META_FILE}. Remove the output cache "
            "directory and rerun training."
        )
    expected_input_bytes = int(np.prod(full_shape)) * np.dtype(np.float32).itemsize
    if os.path.getsize(CACHE_FILE) != expected_input_bytes:
        raise RuntimeError(
            f"Input cache has an unexpected size: {CACHE_FILE}. Remove the output "
            "cache directory and rerun training."
        )
    if output_kind == "field":
        target_cache_shape = (full_shape[0], *target_shape)
        expected_target_bytes = (
            int(np.prod(target_cache_shape)) * np.dtype(np.float32).itemsize
        )
        if (
            not os.path.exists(TARGET_CACHE_FILE)
            or os.path.getsize(TARGET_CACHE_FILE) != expected_target_bytes
        ):
            raise RuntimeError(
                f"Dense target cache is missing or incomplete: {TARGET_CACHE_FILE}. "
                "Remove the output cache directory and rerun training."
            )

    # Load and validate scaler
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    if not hasattr(scaler, "scale_") or scaler.scale_ is None:
        raise RuntimeError("CRITICAL: Scaler is not properly fitted (scale_ is None)")

    y_tensor = None
    if output_kind == "vector":
        # Vector labels are small enough to keep in memory and retain the
        # established fast indexing behavior.
        outp = canonicalize_targets(
            load_outputs_only(
                args.data_path, allow_pickle=getattr(args, "allow_pickle", False)
            )
        )
        if tuple(outp.shape[1:]) != target_shape:
            raise ValueError(
                f"Target shape changed since cache creation: expected "
                f"{target_shape}, got {tuple(outp.shape[1:])}. "
                "Use --fresh or remove the output cache directory."
            )
        if outp.dtype.kind not in "biuf" or not np.isfinite(outp).all():
            raise ValueError("Targets must contain only finite real values.")
        y_scaled = _as_finite_float32(scaler.transform(outp), "Scaled targets")
        y_tensor = torch.tensor(y_scaled)
    elif not os.path.exists(TARGET_CACHE_FILE):
        raise RuntimeError(
            f"Dense target cache missing: {TARGET_CACHE_FILE}. Remove the output "
            "cache directory and rerun training."
        )

    # Regenerate indices (deterministic with same seed)
    indices = np.arange(full_shape[0])
    tr_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=args.seed
    )

    # Create datasets
    dataset_kwargs = {}
    if output_kind == "field":
        dataset_kwargs = {
            "target_memmap_path": TARGET_CACHE_FILE,
            "target_shape": target_shape,
        }
    tr_ds = MemmapDataset(CACHE_FILE, y_tensor, full_shape, tr_idx, **dataset_kwargs)
    val_ds = MemmapDataset(CACHE_FILE, y_tensor, full_shape, val_idx, **dataset_kwargs)

    # Create DataLoaders with thread-safe configuration
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": accelerator.device.type == "cuda",
        "persistent_workers": (args.workers > 0),
        "prefetch_factor": 2 if args.workers > 0 else None,
        "worker_init_fn": memmap_worker_init_fn if args.workers > 0 else None,
    }

    train_dl = DataLoader(tr_ds, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Return spatial shape (H, W) for model initialization
    in_shape = full_shape[2:]

    return train_dl, val_dl, scaler, in_shape, out_dim
