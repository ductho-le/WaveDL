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

import os
import gc
import pickle
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Any, List

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from accelerate import Accelerator

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
INPUT_KEYS = ['input_train', 'input_test', 'X', 'data', 'inputs', 'features', 'x']
OUTPUT_KEYS = ['output_train', 'output_test', 'Y', 'labels', 'outputs', 'targets', 'y']


class DataSource(ABC):
    """
    Abstract base class for data loaders supporting multiple file formats.
    
    Subclasses must implement the `load()` method to return input/output arrays,
    and optionally `load_outputs_only()` for memory-efficient target loading.
    """
    
    @abstractmethod
    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
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
            '.npz': 'npz',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.mat': 'mat',
        }
        return format_map.get(ext, 'npz')
    
    @staticmethod
    def _find_key(available_keys: List[str], candidates: List[str]) -> Optional[str]:
        """Find first matching key from candidates in available keys."""
        for key in candidates:
            if key in available_keys:
                return key
        return None


class NPZSource(DataSource):
    """Load data from NumPy .npz archives."""
    
    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        
        input_key = self._find_key(keys, INPUT_KEYS)
        output_key = self._find_key(keys, OUTPUT_KEYS)
        
        if input_key is None or output_key is None:
            raise KeyError(
                f"NPZ must contain input and output arrays. "
                f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                f"Found: {keys}"
            )
        
        inp = data[input_key]
        outp = data[output_key]
        
        # Handle object arrays (e.g., sparse matrices stored as objects)
        if inp.dtype == object:
            inp = np.array([x.toarray() if hasattr(x, 'toarray') else x for x in inp])
        
        return inp, outp
    
    def load_mmap(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data using memory-mapped mode for zero-copy access.
        
        This allows processing large datasets without loading them entirely
        into RAM. Critical for HPC environments with memory constraints.
        
        Note: Returns memory-mapped arrays - do NOT modify them.
        """
        data = np.load(path, allow_pickle=True, mmap_mode='r')
        keys = list(data.keys())
        
        input_key = self._find_key(keys, INPUT_KEYS)
        output_key = self._find_key(keys, OUTPUT_KEYS)
        
        if input_key is None or output_key is None:
            raise KeyError(
                f"NPZ must contain input and output arrays. "
                f"Supported keys: {INPUT_KEYS} / {OUTPUT_KEYS}. "
                f"Found: {keys}"
            )
        
        inp = data[input_key]
        outp = data[output_key]
        
        return inp, outp
    
    def load_outputs_only(self, path: str) -> np.ndarray:
        """Load only targets from NPZ (avoids loading large input arrays)."""
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        
        output_key = self._find_key(keys, OUTPUT_KEYS)
        if output_key is None:
            raise KeyError(
                f"NPZ must contain output array. "
                f"Supported keys: {OUTPUT_KEYS}. Found: {keys}"
            )
        
        return data[output_key]


class HDF5Source(DataSource):
    """Load data from HDF5 (.h5, .hdf5) files."""
    
    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(path, 'r') as f:
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
    
    def load_mmap(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load HDF5 file with lazy/memory-mapped access.
        
        Returns h5py datasets that read from disk on-demand,
        avoiding loading the entire file into RAM.
        """
        f = h5py.File(path, 'r')  # Keep file open for lazy access
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
        
        # Return h5py datasets (lazy - doesn't load into RAM)
        return f[input_key], f[output_key]
    
    def load_outputs_only(self, path: str) -> np.ndarray:
        """Load only targets from HDF5 (avoids loading large input arrays)."""
        with h5py.File(path, 'r') as f:
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
    """
    
    def __init__(self, h5_dataset):
        self._dataset = h5_dataset
        # Transpose shape: MATLAB (cols, rows, ...) -> Python (rows, cols, ...)
        self.shape = tuple(reversed(h5_dataset.shape))
        self.dtype = h5_dataset.dtype
        
        # Precompute transpose axis order for efficiency
        # For shape (A, B, C) -> reversed (C, B, A), transpose axes are (2, 1, 0)
        self._transpose_axes = tuple(range(len(h5_dataset.shape) - 1, -1, -1))
    
    def __len__(self) -> int:
        return self.shape[0]
    
    def __getitem__(self, idx):
        """
        Fetch data with automatic full transpose.
        
        Handles integer indexing, slices, and fancy indexing.
        All operations return data with fully reversed axes to match .T behavior.
        """
        if isinstance(idx, (int, np.integer)):
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
        
        elif isinstance(idx, (list, np.ndarray)):
            # Fancy indexing: load samples one at a time (h5py limitation)
            # This is slower but necessary for compatibility
            samples = [self[i] for i in idx]
            return np.stack(samples, axis=0)
        
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")


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
        if hasattr(dataset, 'keys'):
            keys = set(dataset.keys())
            return {'data', 'ir', 'jc'}.issubset(keys)
        return False
    
    @staticmethod
    def _load_sparse_to_dense(group) -> np.ndarray:
        """Convert MATLAB sparse matrix (CSC format in HDF5) to dense numpy array."""
        from scipy.sparse import csc_matrix
        
        data = np.array(group['data'])
        ir = np.array(group['ir'])  # row indices
        jc = np.array(group['jc'])  # column pointers
        
        # Get shape from MATLAB attributes or infer
        if 'MATLAB_sparse' in group.attrs:
            nrows = group.attrs['MATLAB_sparse']
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
    
    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load MAT v7.3 file using h5py."""
        try:
            with h5py.File(path, 'r') as f:
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
                
                # Handle 1D outputs that become (1, N) after transpose
                if outp.ndim == 2 and outp.shape[0] == 1:
                    outp = outp.T
                
        except OSError as e:
            raise ValueError(
                f"Failed to load MAT file: {path}. "
                f"Ensure it's saved as v7.3: save('file.mat', '-v7.3'). "
                f"Original error: {e}"
            )
        
        return inp, outp
    
    def load_mmap(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MAT v7.3 file with lazy/memory-mapped access.
        
        Returns h5py datasets that read from disk on-demand,
        avoiding loading the entire file into RAM.
        
        Note: For sparse matrices, this will load and convert them.
        For dense arrays, returns a transposed view wrapper for consistent axis ordering.
        """
        try:
            f = h5py.File(path, 'r')  # Keep file open for lazy access
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
                inp = _TransposedH5Dataset(inp_dataset)
            
            if self._is_sparse_dataset(outp_dataset):
                outp = self._load_sparse_to_dense(outp_dataset).T
            else:
                # Wrap h5py dataset with transpose view
                outp = _TransposedH5Dataset(outp_dataset)
            
            return inp, outp
            
        except OSError as e:
            raise ValueError(
                f"Failed to load MAT file: {path}. "
                f"Ensure it's saved as v7.3: save('file.mat', '-v7.3'). "
                f"Original error: {e}"
            )

    
    def load_outputs_only(self, path: str) -> np.ndarray:
        """Load only targets from MAT v7.3 file (avoids loading large input arrays)."""
        try:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                
                output_key = self._find_key(keys, OUTPUT_KEYS)
                if output_key is None:
                    raise KeyError(
                        f"MAT file must contain output array. "
                        f"Supported keys: {OUTPUT_KEYS}. Found: {keys}"
                    )
                
                # Load with sparse matrix support
                outp = self._load_dataset(f, output_key)
                
                # Handle 1D outputs
                if outp.ndim == 2 and outp.shape[0] == 1:
                    outp = outp.T
                
        except OSError as e:
            raise ValueError(
                f"Failed to load MAT file: {path}. "
                f"Ensure it's saved as v7.3: save('file.mat', '-v7.3'). "
                f"Original error: {e}"
            )
        
        return outp


def get_data_source(format: str) -> DataSource:
    """
    Factory function to get the appropriate DataSource for a format.
    
    Args:
        format: One of 'npz', 'hdf5', 'mat'
        
    Returns:
        DataSource instance
    """
    sources = {
        'npz': NPZSource,
        'hdf5': HDF5Source,
        'mat': MATSource,
    }
    
    if format not in sources:
        raise ValueError(f"Unsupported format: {format}. Supported: {list(sources.keys())}")
    
    return sources[format]()


def load_training_data(path: str, format: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
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
    if format == 'auto':
        format = DataSource.detect_format(path)
    
    source = get_data_source(format)
    return source.load(path)


def load_outputs_only(path: str, format: str = 'auto') -> np.ndarray:
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
    if format == 'auto':
        format = DataSource.detect_format(path)
    
    source = get_data_source(format)
    return source.load_outputs_only(path)





# ==============================================================================
# WORKER INITIALIZATION
# ==============================================================================
def memmap_worker_init_fn(worker_id: int):
    """
    Worker initialization function for proper memmap handling in multi-worker DataLoader.
    
    Each DataLoader worker process runs this function after forking. It resets the 
    memmap file handle to None, forcing each worker to open its own read-only handle.
    This prevents file descriptor sharing issues and race conditions.
    
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
        targets: torch.Tensor, 
        shape: Tuple[int, ...], 
        indices: np.ndarray
    ):
        self.memmap_path = memmap_path
        self.targets = targets
        self.shape = shape
        self.indices = indices
        self.data: Optional[np.memmap] = None  # Lazy initialization
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            # Mode 'r' = read-only, prevents accidental data modification
            self.data = np.memmap(
                self.memmap_path, 
                dtype='float32', 
                mode='r', 
                shape=self.shape
            )
        
        real_idx = self.indices[idx]
        
        # .copy() detaches from mmap buffer - essential for PyTorch pinned memory
        x = torch.from_numpy(self.data[real_idx].copy()).contiguous()
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
def prepare_data(
    args: Any,
    logger: logging.Logger,
    accelerator: Accelerator,
    cache_dir: str = ".",
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, StandardScaler, Tuple[int, ...], int]:
    """
    Prepare DataLoaders with DDP synchronization guarantees.
    
    This function handles:
    1. Loading raw data and creating memmap cache (Rank 0 only)
    2. Fitting StandardScaler on training set only (no data leakage)
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
    
    Returns:
        Tuple of:
            - train_dl: Training DataLoader
            - val_dl: Validation DataLoader
            - scaler: Fitted StandardScaler (for inverse transforms)
            - in_shape: Input spatial dimensions - (L,), (H, W), or (D, H, W)
            - out_dim: Number of output targets
    
    Cache Files Created:
        - train_data_cache.dat: Memory-mapped input data
        - scaler.pkl: Fitted StandardScaler
        - data_metadata.pkl: Shape and dimension metadata
    """
    CACHE_FILE = os.path.join(cache_dir, "train_data_cache.dat")
    SCALER_FILE = os.path.join(cache_dir, "scaler.pkl")
    META_FILE = os.path.join(cache_dir, "data_metadata.pkl")
    
    # ==========================================================================
    # PHASE 1: DATA GENERATION (Rank 0 Only)
    # ==========================================================================
    # Check cache existence on all ranks first (fast, no synchronization needed)
    cache_exists = (
        os.path.exists(CACHE_FILE) and 
        os.path.exists(SCALER_FILE) and 
        os.path.exists(META_FILE)
    )
    
    if not cache_exists:
        if accelerator.is_main_process:
            # RANK 0: Create cache (can take a long time for large datasets)
            # Other ranks will wait at the barrier below
            
            # Detect format from extension
            data_format = DataSource.detect_format(args.data_path)
            logger.info(f"⚡ [Rank 0] Initializing Data Processing from: {args.data_path} (format: {data_format})")
            
            # Validate data file exists
            if not os.path.exists(args.data_path):
                raise FileNotFoundError(f"CRITICAL: Data file not found: {args.data_path}")
            
            # Load raw data using memory-mapped mode for all formats
            # This avoids loading the entire dataset into RAM at once
            try:
                if data_format == 'npz':
                    source = NPZSource()
                    inp, outp = source.load_mmap(args.data_path)
                elif data_format == 'hdf5':
                    source = HDF5Source()
                    inp, outp = source.load_mmap(args.data_path)
                elif data_format == 'mat':
                    source = MATSource()
                    inp, outp = source.load_mmap(args.data_path)
                else:
                    inp, outp = load_training_data(args.data_path, format=data_format)
                logger.info("   Using memory-mapped loading (low memory mode)")
            except Exception as e:
                logger.error(f"Failed to load data file: {e}")
                raise
            
            # Detect shape (handle sparse matrices) - DIMENSION AGNOSTIC
            num_samples = len(inp)
            out_dim = outp.shape[1]
            sample_0 = inp[0]
            if issparse(sample_0) or hasattr(sample_0, 'toarray'):
                sample_0 = sample_0.toarray()
            
            # Automatically detect dimensionality from sample shape
            spatial_shape = sample_0.shape  # Could be (L,), (H, W), or (D, H, W)
            full_shape = (num_samples, 1) + spatial_shape  # Add channel dim: (N, 1, ...)
            
            dim_names = {1: "1D (L)", 2: "2D (H, W)", 3: "3D (D, H, W)"}
            dim_type = dim_names.get(len(spatial_shape), f"{len(spatial_shape)}D")
            logger.info(f"   Shape Detected: {full_shape} [{dim_type}] | Output Dim: {out_dim}")
            
            # Save metadata
            with open(META_FILE, 'wb') as f:
                pickle.dump({'shape': full_shape, 'out_dim': out_dim}, f)
            
            # Create memmap cache
            if not os.path.exists(CACHE_FILE):
                logger.info("   Writing Memmap Cache (one-time operation)...")
                fp = np.memmap(CACHE_FILE, dtype='float32', mode='w+', shape=full_shape)
                
                chunk_size = 2000
                pbar = tqdm(
                    range(0, num_samples, chunk_size),
                    desc="Caching",
                    disable=not accelerator.is_main_process
                )
                
                for i in pbar:
                    end = min(i + chunk_size, num_samples)
                    batch = inp[i:end]
                    
                    # Handle sparse/dense conversion
                    if issparse(batch[0]) or hasattr(batch[0], 'toarray'):
                        data_chunk = np.stack([x.toarray().astype(np.float32) for x in batch])
                    else:
                        data_chunk = np.array(batch).astype(np.float32)
                    
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
                indices, 
                test_size=val_split, 
                random_state=args.seed
            )
            
            if not os.path.exists(SCALER_FILE):
                logger.info("   Fitting StandardScaler (training set only)...")
                
                # Convert lazy datasets to numpy for reliable indexing
                # (h5py and _TransposedH5Dataset may not support fancy indexing)
                if hasattr(outp, '_dataset') or hasattr(outp, 'file'):
                    # Lazy h5py or _TransposedH5Dataset - load training subset
                    outp_train = np.array([outp[i] for i in tr_idx])
                else:
                    # Already numpy array
                    outp_train = outp[tr_idx]
                
                # Ensure 2D for StandardScaler: (N,) -> (N, 1)
                if outp_train.ndim == 1:
                    outp_train = outp_train.reshape(-1, 1)
                
                scaler = StandardScaler()
                scaler.fit(outp_train)
                with open(SCALER_FILE, 'wb') as f:
                    pickle.dump(scaler, f)
            
            # Cleanup
            del inp, outp
            gc.collect()
            
            logger.info("   ✔ Cache creation complete, synchronizing ranks...")
        else:
            # NON-MAIN RANKS: Wait for cache creation
            # Log that we're waiting (helps with debugging)
            import time
            wait_start = time.time()
            while not (os.path.exists(CACHE_FILE) and 
                      os.path.exists(SCALER_FILE) and 
                      os.path.exists(META_FILE)):
                time.sleep(5)  # Check every 5 seconds
                elapsed = time.time() - wait_start
                if elapsed > 60 and int(elapsed) % 60 < 5:  # Log every ~minute
                    logger.info(f"   [Rank {accelerator.process_index}] Waiting for cache creation... ({int(elapsed)}s)")
            # Small delay to ensure files are fully written
            time.sleep(2)
    
    # ==========================================================================
    # PHASE 2: SYNCHRONIZED LOADING (All Ranks)
    # ==========================================================================
    accelerator.wait_for_everyone()
    
    # Load metadata
    with open(META_FILE, 'rb') as f:
        meta = pickle.load(f)
        full_shape = meta['shape']
        out_dim = meta['out_dim']
    
    # Load and validate scaler
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    
    if not hasattr(scaler, 'scale_') or scaler.scale_ is None:
        raise RuntimeError("CRITICAL: Scaler is not properly fitted (scale_ is None)")
    
    # Load targets only (memory-efficient - avoids loading large input arrays)
    # This is critical for HPC environments with memory constraints during DDP
    outp = load_outputs_only(args.data_path)
    y_scaled = scaler.transform(outp).astype(np.float32)
    y_tensor = torch.tensor(y_scaled)
    
    # Regenerate indices (deterministic with same seed)
    indices = np.arange(full_shape[0])
    tr_idx, val_idx = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=args.seed
    )
    
    # Create datasets
    tr_ds = MemmapDataset(CACHE_FILE, y_tensor, full_shape, tr_idx)
    val_ds = MemmapDataset(CACHE_FILE, y_tensor, full_shape, val_idx)
    
    # Create DataLoaders with thread-safe configuration
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
        worker_init_fn=memmap_worker_init_fn if args.workers > 0 else None
    )
    
    train_dl = DataLoader(tr_ds, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    
    # Return spatial shape (H, W) for model initialization
    in_shape = full_shape[2:]
    
    return train_dl, val_dl, scaler, in_shape, out_dim
