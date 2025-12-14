"""
WaveDL - Testing & Inference Script
====================================
Target Environment: NVIDIA HPC GPUs (Single/Multi-GPU) | PyTorch 2.x | Python 3.11+

Production-grade inference script for evaluating trained WaveDL models:
  1. Model-Agnostic: Works with any registered architecture
  2. Flexible Data Loading: Supports NPZ (preferred) and legacy MAT formats
  3. Comprehensive Metrics: R², Pearson, per-parameter MAE with physical units
  4. Publication Plots: High-quality scatter plots with confidence intervals
  5. Batch Inference: Efficient GPU utilization for large test sets
  6. Model Export: ONNX format for deployment in production systems

Usage:
    # Basic inference (NPZ format - recommended)
    python test.py --checkpoint ./cnn_test/best_checkpoint --data_path test_data.npz
    
    # With visualization and detailed output
    python test.py --checkpoint ./cnn_test/best_checkpoint --data_path test_data.npz \
        --plot --output_dir ./test_results --save_predictions
    
    # Export model to ONNX for deployment
    python test.py --checkpoint ./cnn_test/best_checkpoint --data_path test_data.npz \
        --export onnx --export_path model.onnx
    
    # Legacy MAT file support
    python test.py --checkpoint ./cnn_run/best_checkpoint --data_path CS01.mat \
        --format mat --plot

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0 (WaveDL Testing Suite + ONNX Export)
"""

# ==============================================================================
# ENVIRONMENT CONFIGURATION (must be before matplotlib import)
# ==============================================================================
import os
os.environ.setdefault('MPLCONFIGDIR', os.getenv('TMPDIR', '/tmp') + '/matplotlib')
os.environ.setdefault('FONTCONFIG_PATH', '/etc/fonts')

import sys
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import r2_score, mean_absolute_error

# Local imports
from models import get_model, list_models, build_model
from utils import calc_pearson, plot_scientific_scatter

# Optional dependencies
try:
    import scipy.io
    from scipy.sparse import issparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for testing."""
    parser = argparse.ArgumentParser(
        description="WaveDL Testing & Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to checkpoint directory (e.g., ./cnn_test/best_checkpoint)")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to test data (NPZ or MAT format)")
    
    # Model specification
    parser.add_argument('--model', type=str, default=None,
                        help=f"Model architecture (auto-detect if not specified). Available: {list_models()}")
    
    # Data format
    parser.add_argument('--format', type=str, default='auto', choices=['auto', 'npz', 'mat', 'hdf5'],
                        help="Data format (auto-detect by extension)")
    parser.add_argument('--input_key', type=str, default=None,
                        help="Custom key name for input data in MAT/HDF5/NPZ files (e.g., 'X', 'waveforms')")
    parser.add_argument('--output_key', type=str, default=None,
                        help="Custom key name for output data in MAT/HDF5/NPZ files (e.g., 'Y', 'labels')")
    parser.add_argument('--param_names', type=str, nargs='+', default=None,
                        help="Parameter names for output (e.g., 'h' 'v11' 'v12')")
    
    # Inference options
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for inference")
    parser.add_argument('--workers', type=int, default=4,
                        help="DataLoader workers")
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='.',
                        help="Directory for saving results")
    parser.add_argument('--save_predictions', action='store_true',
                        help="Save predictions to CSV")
    parser.add_argument('--plot', action='store_true',
                        help="Generate scatter plots")
    parser.add_argument('--verbose', action='store_true',
                        help="Print per-sample predictions")
    
    # Export options
    parser.add_argument('--export', type=str, default=None, choices=['onnx'],
                        help="Export model format (onnx)")
    parser.add_argument('--export_path', type=str, default=None,
                        help="Output path for exported model (default: {model_name}.onnx)")
    parser.add_argument('--export_opset', type=int, default=17,
                        help="ONNX opset version (11-17, higher = newer ops)")
    parser.add_argument('--out_size', type=int, default=None,
                        help="Override output size (inferred from scaler.pkl if targets missing)")
    
    return parser.parse_args()


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_npz_data(
    file_path: str,
    input_key: Optional[str] = None,
    output_key: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data from NPZ format (WaveDL standard).
    
    Supports any input dimensionality:
        - 1D: (N, L) → (N, 1, L)
        - 2D: (N, H, W) → (N, 1, H, W)  
        - 3D: (N, D, H, W) → (N, 1, D, H, W)
        - Already has channel: (N, C, ...) → unchanged
    
    Args:
        file_path: Path to NPZ file
        input_key: Custom key for input data (auto-detect if None)
        output_key: Custom key for output data (auto-detect if None)
    """
    # Default key names (priority order - test keys first for inference)
    DEFAULT_INPUT_KEYS = ['input_test', 'input_train', 'X', 'data', 'inputs', 'features', 'x']
    DEFAULT_OUTPUT_KEYS = ['output_test', 'output_train', 'Y', 'labels', 'outputs', 'targets', 'y']
    
    data = np.load(file_path, allow_pickle=True)
    keys = list(data.keys())
    
    # Use custom key or auto-detect
    if input_key:
        if input_key not in keys:
            raise KeyError(f"Custom input key '{input_key}' not found. Available: {keys}")
        inp_key = input_key
    else:
        inp_key = next((k for k in DEFAULT_INPUT_KEYS if k in keys), None)
        if inp_key is None:
            raise KeyError(f"NPZ must contain input array. Supported keys: {DEFAULT_INPUT_KEYS}. Found: {keys}")
    
    if output_key:
        if output_key not in keys:
            raise KeyError(f"Custom output key '{output_key}' not found. Available: {keys}")
        out_key = output_key
    else:
        out_key = next((k for k in DEFAULT_OUTPUT_KEYS if k in keys), None)
    
    X = data[inp_key]
    
    if out_key:
        y = data[out_key]
    else:
        logging.warning(f"No target data found in NPZ. Proceeding with predictions only.")
        y = None  # None signals no targets available
    
    # Handle sparse matrices (requires scipy)
    if HAS_SCIPY:
        if issparse(X):
            X = X.toarray()
        if y is not None and issparse(y):
            y = y.toarray()
    
    X = torch.tensor(X, dtype=torch.float32)
    
    if y is not None:
        y = torch.tensor(y, dtype=torch.float32)
        # Normalize target shape: (N,) -> (N, 1)
        if y.ndim == 1:
            y = y.unsqueeze(1)
    
    # Add channel dimension if needed (dimension-agnostic)
    # X.ndim == 2: 1D data (N, L) → (N, 1, L)
    # X.ndim == 3: 2D data (N, H, W) → (N, 1, H, W)
    # X.ndim == 4: Could be 3D (N, D, H, W) → (N, 1, D, H, W) OR already has channel (N, C, H, W)
    # X.ndim == 5: 3D with channel (N, C, D, H, W) → unchanged
    
    if X.ndim == 2:
        # 1D signal: (N, L) → (N, 1, L)
        X = X.unsqueeze(1)
    elif X.ndim == 3:
        # 2D image: (N, H, W) → (N, 1, H, W)
        X = X.unsqueeze(1)
    elif X.ndim == 4:
        # Could be 3D volume (N, D, H, W) or 2D with channel (N, C, H, W)
        # Heuristic: if dim 1 is small (<=16), assume it's already a channel dim
        if X.shape[1] <= 16:
            logging.info(f"   Detected existing channel dimension (C={X.shape[1]})")
        else:
            # 3D volume without channel: (N, D, H, W) → (N, 1, D, H, W)
            X = X.unsqueeze(1)
    # X.ndim >= 5: assume channel dimension already exists
    
    return X, y


def load_mat_data(
    file_path: str,
    input_key: Optional[str] = None,
    output_key: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data from MATLAB .mat format (v7.3+ only, HDF5-based).
    
    Automatically handles sparse matrices (converted to dense).
    
    Args:
        file_path: Path to MAT file
        input_key: Custom key for input data (auto-detect if None)
        output_key: Custom key for output data (auto-detect if None)
    
    Note: MAT files must be saved with -v7.3 flag in MATLAB:
        save('data.mat', 'input_test', 'output_test', '-v7.3')
    """
    import h5py
    
    # Default key names (priority order - test keys first for inference)
    DEFAULT_INPUT_KEYS = ['input_test', 'input_train', 'X', 'data', 'inputs', 'features', 'x']
    DEFAULT_OUTPUT_KEYS = ['output_test', 'output_train', 'Y', 'labels', 'outputs', 'targets', 'y']
    
    def is_sparse_dataset(dataset) -> bool:
        """Check if HDF5 dataset/group represents a MATLAB sparse matrix."""
        if hasattr(dataset, 'keys'):
            keys = set(dataset.keys())
            return {'data', 'ir', 'jc'}.issubset(keys)
        return False
    
    def load_sparse_to_dense(group) -> np.ndarray:
        """Convert MATLAB sparse matrix (CSC format in HDF5) to dense array."""
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
    
    def load_dataset(f, key: str) -> np.ndarray:
        """Load a dataset, handling sparse matrices automatically."""
        dataset = f[key]
        
        if is_sparse_dataset(dataset):
            arr = load_sparse_to_dense(dataset)
        else:
            arr = np.array(dataset)
        
        # Transpose for MATLAB column-major -> Python row-major
        return arr.T
    
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            logging.info(f"   MAT file keys: {keys}")
            
            # Use custom input key if provided
            if input_key:
                if input_key not in keys:
                    raise KeyError(f"Custom input key '{input_key}' not found. Available: {keys}")
                inp_key = input_key
                logging.info(f"   Using custom input key: '{inp_key}'")
            else:
                # Auto-detect input key - must be numeric array
                inp_key = None
                for k in DEFAULT_INPUT_KEYS:
                    if k in keys:
                        dataset = f[k]
                        # Check if it's a valid numeric dataset or sparse matrix
                        if is_sparse_dataset(dataset):
                            inp_key = k
                            break
                        elif hasattr(dataset, 'dtype'):
                            # Skip string/object dtypes
                            if np.issubdtype(dataset.dtype, np.number):
                                inp_key = k
                                break
                            else:
                                logging.debug(f"   Skipping key '{k}' (dtype: {dataset.dtype})")
                
                if inp_key is None:
                    # Fallback: try to find any large numeric array
                    for k in keys:
                        if k.startswith('#') or k.startswith('_'):
                            continue  # Skip HDF5 metadata keys
                        dataset = f[k]
                        if is_sparse_dataset(dataset):
                            inp_key = k
                            logging.info(f"   Auto-detected input key: '{k}' (sparse)")
                            break
                        elif hasattr(dataset, 'dtype') and hasattr(dataset, 'shape'):
                            if np.issubdtype(dataset.dtype, np.number) and len(dataset.shape) >= 2:
                                inp_key = k
                                logging.info(f"   Auto-detected input key: '{k}' (shape: {dataset.shape})")
                                break
                
                if inp_key is None:
                    raise KeyError(
                        f"MAT file must contain numeric input array. "
                        f"Supported keys: {DEFAULT_INPUT_KEYS}. Found: {keys}. "
                        f"Use --input_key to specify your custom key name."
                    )
            
            # Load input data
            logging.info(f"   Loading input from key: '{inp_key}'")
            X_np = load_dataset(f, inp_key)
            
            # Handle 1D outputs that become (1, N) after transpose
            if X_np.ndim == 2 and X_np.shape[0] == 1:
                X_np = X_np.T
            
            # Use custom output key if provided
            if output_key:
                if output_key not in keys:
                    raise KeyError(f"Custom output key '{output_key}' not found. Available: {keys}")
                out_key = output_key
            else:
                out_key = next((k for k in DEFAULT_OUTPUT_KEYS if k in keys), None)
            
            if out_key is not None:
                y_np = load_dataset(f, out_key)
                
                # Handle 1D outputs
                if y_np.ndim == 2 and y_np.shape[0] == 1:
                    y_np = y_np.T
                
                # Ensure 2D target
                if y_np.ndim == 1:
                    y_np = y_np.reshape(-1, 1)
            else:
                logging.warning("No target data found. Proceeding with predictions only.")
                y_np = np.zeros((len(X_np), 1), dtype=np.float32)
    
    except OSError as e:
        raise ValueError(
            f"Failed to load MAT file: {file_path}. "
            f"Ensure it's saved as v7.3: save('file.mat', '-v7.3'). "
            f"Original error: {e}"
        )
    
    # Convert to tensors
    X = torch.tensor(X_np.astype(np.float32))
    y = torch.tensor(y_np.astype(np.float32))
    
    # Add channel dimension based on input dimensionality
    if X.ndim == 2:
        # 1D signal: (N, L) -> (N, 1, L)
        X = X.unsqueeze(1)
    elif X.ndim == 3:
        # 2D image: (N, H, W) -> (N, 1, H, W)
        X = X.unsqueeze(1)
    elif X.ndim == 4:
        # Could be 3D volume or 2D with channel
        if X.shape[1] <= 16:
            logging.info(f"   Detected existing channel dimension (C={X.shape[1]})")
        else:
            X = X.unsqueeze(1)
    
    return X, y


def load_hdf5_data(
    file_path: str,
    input_key: Optional[str] = None,
    output_key: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data from HDF5 (.h5, .hdf5) format.
    
    Args:
        file_path: Path to HDF5 file
        input_key: Custom key for input data (auto-detect if None)
        output_key: Custom key for output data (auto-detect if None)
    """
    import h5py
    
    # Default key names (priority order)
    DEFAULT_INPUT_KEYS = ['input_test', 'input_train', 'X', 'data', 'inputs', 'features', 'x']
    DEFAULT_OUTPUT_KEYS = ['output_test', 'output_train', 'Y', 'labels', 'outputs', 'targets', 'y']
    
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        
        # Use custom input key or auto-detect
        if input_key:
            if input_key not in keys:
                raise KeyError(f"Custom input key '{input_key}' not found. Available: {keys}")
            inp_key = input_key
        else:
            inp_key = next((k for k in DEFAULT_INPUT_KEYS if k in keys), None)
            if inp_key is None:
                raise KeyError(
                    f"HDF5 must contain input array. Supported keys: {DEFAULT_INPUT_KEYS}. "
                    f"Found: {keys}. Use --input_key to specify your custom key name."
                )
        
        # Use custom output key or auto-detect
        if output_key:
            if output_key not in keys:
                raise KeyError(f"Custom output key '{output_key}' not found. Available: {keys}")
            out_key = output_key
        else:
            out_key = next((k for k in DEFAULT_OUTPUT_KEYS if k in keys), None)
        
        X = f[inp_key][:]
        
        if out_key:
            y = f[out_key][:]
        else:
            logging.warning(f"No target data found in HDF5. Proceeding with predictions only.")
            y = np.zeros((len(X), 1))
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Add channel dimension if needed (same logic as load_npz_data)
    if X.ndim == 2:
        X = X.unsqueeze(1)
    elif X.ndim == 3:
        X = X.unsqueeze(1)
    elif X.ndim == 4:
        if X.shape[1] <= 16:
            logging.info(f"   Detected existing channel dimension (C={X.shape[1]})")
        else:
            X = X.unsqueeze(1)
    
    return X, y


def load_test_data(
    file_path: str,
    format: str = 'auto',
    input_key: Optional[str] = None,
    output_key: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data with automatic format detection.
    
    Supports NPZ, MAT, and HDF5 formats.
    
    Args:
        file_path: Path to data file
        format: Data format ('auto', 'npz', 'mat', 'hdf5')
        input_key: Custom key name for input data (overrides auto-detection)
        output_key: Custom key name for output data (overrides auto-detection)
    
    Returns:
        X: Input tensor (N, 1, H, W)
        y: Target tensor (N, T)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Auto-detect format
    if format == 'auto':
        suffix = file_path.suffix.lower()
        format_map = {'.npz': 'npz', '.mat': 'mat', '.h5': 'hdf5', '.hdf5': 'hdf5'}
        format = format_map.get(suffix, 'npz')
    
    logging.info(f"Loading test data from: {file_path} (format: {format})")
    if input_key:
        logging.info(f"   Using custom input key: '{input_key}'")
    if output_key:
        logging.info(f"   Using custom output key: '{output_key}'")
    
    if format == 'npz':
        X, y = load_npz_data(str(file_path), input_key=input_key, output_key=output_key)
    elif format == 'mat':
        X, y = load_mat_data(str(file_path), input_key=input_key, output_key=output_key)
    elif format == 'hdf5':
        X, y = load_hdf5_data(str(file_path), input_key=input_key, output_key=output_key)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: npz, mat, hdf5")
    
    logging.info(f"   ✔ Loaded {len(X)} samples | Input: {X.shape} | Target: {y.shape}")
    
    return X, y


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_checkpoint(
    checkpoint_dir: str,
    in_shape: Tuple[int, int],
    out_size: int,
    model_name: Optional[str] = None
) -> Tuple[nn.Module, any]:
    """
    Load model and scaler from Accelerate checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        in_shape: Input image shape (H, W)
        out_size: Number of output parameters
        model_name: Model architecture name (auto-detect if None)
    
    Returns:
        model: Loaded model in eval mode
        scaler: StandardScaler for inverse transform
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load training metadata
    meta_path = checkpoint_dir / "training_meta.pkl"
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        logging.info(f"   Checkpoint from epoch {meta.get('epoch', 'unknown')}, "
                     f"val_loss: {meta.get('best_val_loss', 'N/A'):.6f}")
    
    # Auto-detect model architecture if not specified
    if model_name is None:
        # Try to detect from parent directory name (e.g., 'cnn_test' -> 'cnn')
        parent_dir = checkpoint_dir.parent.name
        detected_name = parent_dir.split('_')[0] if '_' in parent_dir else parent_dir.split('-')[0]
        
        if detected_name in list_models():
            model_name = detected_name
            logging.info(f"   Auto-detected model: {model_name}")
        else:
            raise ValueError(
                f"Could not auto-detect model architecture from '{parent_dir}'.\n"
                f"Please specify --model explicitly. Available models: {list_models()}"
            )
    
    logging.info(f"   Building model: {model_name}")
    model = build_model(model_name, in_shape=in_shape, out_size=out_size)
    
    # Load weights (prefer safetensors)
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        weight_path = checkpoint_dir / "pytorch_model.bin"
    
    if not weight_path.exists():
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
    
    if HAS_SAFETENSORS and weight_path.suffix == '.safetensors':
        state_dict = load_safetensors(str(weight_path))
    else:
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    
    # Remove 'module.' prefix from DDP checkpoints
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    logging.info(f"   ✔ Loaded weights from: {weight_path.name}")
    
    # Load scaler
    scaler_path = checkpoint_dir.parent / "scaler.pkl"
    if not scaler_path.exists():
        # Try in checkpoint dir itself
        scaler_path = checkpoint_dir / "scaler.pkl"
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found. Expected at: {checkpoint_dir.parent}/scaler.pkl")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    logging.info(f"   ✔ Loaded scaler from: {scaler_path.name}")
    
    param_info = model.parameter_summary()
    logging.info(f"   Model: {param_info['trainable_parameters']:,} parameters ({param_info['total_mb']:.2f} MB)")
    
    return model, scaler


# ==============================================================================
# INFERENCE
# ==============================================================================
@torch.inference_mode()
def run_inference(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int = 128,
    device: torch.device = None,
    num_workers: int = 0
) -> np.ndarray:
    """
    Run batch inference on test data.
    
    Args:
        model: Trained model in eval mode
        X: Input tensor (N, C, *spatial_dims)
        batch_size: Batch size for inference
        device: Target device (auto-detect if None)
        num_workers: DataLoader workers (0 for single-threaded)
    
    Returns:
        predictions: Numpy array (N, out_size) - still in normalized space
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    dataset = TensorDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda'
    )
    
    predictions = []
    
    for (batch_x,) in tqdm(loader, desc="Inference", leave=False):
        batch_x = batch_x.to(device, non_blocking=True)
        preds = model(batch_x).cpu().numpy()
        predictions.append(preds)
    
    return np.vstack(predictions)


# ==============================================================================
# ONNX EXPORT
# ==============================================================================
def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17,
    validate: bool = True,
    model_name: str = "WaveDL_Model"
) -> bool:
    """
    Export PyTorch model to ONNX format for production deployment.
    
    Features:
        - Dynamic batch size support
        - Comprehensive validation with numerical comparison
        - Embedded metadata (input/output names, descriptions)
        - Compatibility testing with ONNX runtime
    
    Args:
        model: Trained PyTorch model in eval mode
        sample_input: Sample input tensor for tracing (N, C, *spatial_dims)
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (11-17, default 17)
        validate: Whether to validate the exported model
        model_name: Model name embedded in ONNX metadata
    
    Returns:
        True if export and validation successful, False otherwise
    
    Example:
        >>> success = export_to_onnx(model, X_test[:1], "model.onnx")
        >>> if success:
        ...     print("Model exported successfully!")
    
    Note:
        For deployment in MATLAB/LabVIEW/C++, use the exported .onnx file
        with the appropriate ONNX runtime for your target platform.
    """
    import warnings
    
    # Ensure model is in eval mode on CPU for consistent export
    model = model.cpu()
    model.eval()
    sample_input = sample_input.cpu()
    
    # Determine input/output names based on dimensions
    input_dims = sample_input.ndim - 2  # Exclude batch and channel
    if input_dims == 1:
        spatial_desc = "1D signal (length)"
    elif input_dims == 2:
        spatial_desc = "2D image (height, width)"
    else:
        spatial_desc = f"{input_dims}D volume"
    
    # Build dynamic axes for variable batch size
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'predictions': {0: 'batch_size'}
    }
    
    logging.info(f"📦 Exporting model to ONNX (opset {opset_version})...")
    logging.info(f"   Input shape: {tuple(sample_input.shape)} ({spatial_desc})")
    
    try:
        # Export with comprehensive settings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,  # Optimize for inference
                input_names=['input'],
                output_names=['predictions'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        logging.info(f"   ✔ Export completed: {output_path}")
        
        # Validate exported model
        if validate:
            return _validate_onnx_export(model, sample_input, output_path)
        
        return True
        
    except Exception as e:
        logging.error(f"   ✘ ONNX export failed: {e}")
        return False


def _validate_onnx_export(
    pytorch_model: nn.Module,
    sample_input: torch.Tensor,
    onnx_path: str,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> bool:
    """
    Validate ONNX model by comparing outputs with PyTorch model.
    
    Args:
        pytorch_model: Original PyTorch model
        sample_input: Input tensor for comparison
        onnx_path: Path to exported ONNX model
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logging.warning("   ⚠ ONNX validation skipped (install: pip install onnx onnxruntime)")
        return True
    
    logging.info("   Validating ONNX model...")
    
    try:
        # 1. Check ONNX model structure
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("   ✔ ONNX model structure valid")
        
        # 2. Compare numerical outputs
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(sample_input.cpu()).numpy()
        
        # ONNX Runtime inference
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        onnx_output = ort_session.run(
            None,
            {'input': sample_input.numpy()}
        )[0]
        
        # Numerical comparison
        max_diff = np.abs(pytorch_output - onnx_output).max()
        
        if np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
            logging.info(f"   ✔ Numerical validation passed (max diff: {max_diff:.2e})")
            return True
        else:
            logging.warning(f"   ⚠ Numerical mismatch (max diff: {max_diff:.2e})")
            return False
            
    except Exception as e:
        logging.warning(f"   ⚠ Validation error: {e}")
        return False


def get_onnx_model_info(onnx_path: str) -> Dict:
    """
    Get metadata from exported ONNX model.
    
    Returns:
        Dictionary with model information
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        input_info = model.graph.input[0]
        output_info = model.graph.output[0]
        
        return {
            'opset_version': model.opset_import[0].version,
            'input_name': input_info.name,
            'input_shape': [d.dim_value or 'dynamic' for d in input_info.type.tensor_type.shape.dim],
            'output_name': output_info.name,
            'output_shape': [d.dim_value or 'dynamic' for d in output_info.type.tensor_type.shape.dim],
            'file_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024),
        }
    except Exception as e:
        return {'error': str(e)}


# ==============================================================================
# METRICS & VISUALIZATION
# ==============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive regression metrics."""
    # Handle single-sample case
    if len(y_true) == 1:
        metrics = {
            'r2_score': float('nan'),  # R² undefined for single sample
            'pearson_corr': float('nan'),  # Correlation undefined for single sample
            'mae_avg': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2))
        }
    else:
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'pearson_corr': calc_pearson(y_true, y_pred),
            'mae_avg': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2))
        }
    
    # Per-parameter MAE
    for i in range(y_true.shape[1]):
        metrics[f'mae_p{i}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
    
    return metrics


def print_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    param_names: Optional[list] = None,
    verbose: bool = False
):
    """Print formatted test results."""
    n_params = y_true.shape[1]
    
    if param_names is None or len(param_names) != n_params:
        param_names = [f"P{i}" for i in range(n_params)]
    
    # Overall metrics
    print("\n" + "="*80)
    print("OVERALL TEST RESULTS")
    print("="*80)
    print(f"Samples:          {len(y_true)}")
    print(f"R² Score:         {metrics['r2_score']:.6f}")
    print(f"Pearson Corr:     {metrics['pearson_corr']:.6f}")
    print(f"RMSE:             {metrics['rmse']:.6f}")
    print(f"MAE (Avg):        {metrics['mae_avg']:.6f}")
    print("="*80)
    
    # Per-parameter MAE
    print("\nPER-PARAMETER MAE:")
    print("-"*80)
    for i, name in enumerate(param_names):
        print(f"  {name:12s}: {metrics[f'mae_p{i}']:.6f}")
    print("-"*80)
    
    # Sample-wise predictions (if verbose)
    if verbose:
        print(f"\nSAMPLE-WISE PREDICTIONS:")
        print("="*80)
        header = "ID   | " + " | ".join([f"{name:>8s}" for name in param_names])
        print(header)
        print("-"*80)
        
        for i in range(min(len(y_true), 20)):  # Limit to first 20 samples
            true_str = " | ".join([f"{val:8.4f}" for val in y_true[i]])
            pred_str = " | ".join([f"{val:8.4f}" for val in y_pred[i]])
            print(f"TRUE | {true_str}")
            print(f"PRED | {pred_str}")
            print("-"*80)
        
        if len(y_true) > 20:
            print(f"... ({len(y_true) - 20} more samples)")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    param_names: Optional[list] = None
):
    """Save predictions to CSV file."""
    n_params = y_true.shape[1]
    
    if param_names is None or len(param_names) != n_params:
        param_names = [f"P{i}" for i in range(n_params)]
    
    # Create DataFrame
    columns = [f"True_{name}" for name in param_names] + [f"Pred_{name}" for name in param_names]
    data = np.hstack([y_true, y_pred])
    df = pd.DataFrame(data, columns=columns)
    
    # Add error columns
    for i, name in enumerate(param_names):
        df[f"Error_{name}"] = df[f"Pred_{name}"] - df[f"True_{name}"]
        df[f"AbsError_{name}"] = np.abs(df[f"Error_{name}"])
    
    df.to_csv(output_path, index=False)
    logging.info(f"   ✔ Predictions saved to: {output_path}")


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    param_names: Optional[list] = None
):
    """Generate and save publication-quality plots."""
    n_params = y_true.shape[1]
    
    if param_names is None or len(param_names) != n_params:
        param_names = [f"P{i}" for i in range(n_params)]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Overall scatter plot (all parameters in grid)
    fig = plot_scientific_scatter(y_true, y_pred, param_names=param_names)
    fig.savefig(output_dir / "test_scatter_all.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"   ✔ Saved: test_scatter_all.png")
    
    # 2. Individual parameter plots (separate file for each)
    for i in range(n_params):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10, 
                   edgecolors='none', c='royalblue')
        
        # Ideal line
        lims = [
            min(y_true[:, i].min(), y_pred[:, i].min()),
            max(y_true[:, i].max(), y_pred[:, i].max())
        ]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=1.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        # Labels and metrics
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        
        ax.set_xlabel(f'True {param_names[i]}', fontsize=11)
        ax.set_ylabel(f'Predicted {param_names[i]}', fontsize=11)
        ax.set_title(f'{param_names[i]}\nR²={r2:.4f}, MAE={mae:.4f}', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Safe filename (replace spaces and special chars)
        safe_name = param_names[i].replace(' ', '_').replace('/', '_')
        filename = f"test_scatter_{safe_name}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"   ✔ Saved: {filename}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("Tester")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    X_test, y_test = load_test_data(
        args.data_path,
        format=args.format,
        input_key=args.input_key,
        output_key=args.output_key
    )
    in_shape = tuple(X_test.shape[2:])
    
    # Determine if we have ground truth targets
    has_targets = y_test is not None
    
    # Determine output size: from targets, from --out_size, or from scaler
    if args.out_size is not None:
        out_size = args.out_size
        logger.info(f"   Using explicit --out_size={out_size}")
    elif has_targets:
        out_size = y_test.shape[1]
    else:
        # Infer from scaler.pkl
        scaler_path = Path(args.checkpoint) / "scaler.pkl"
        if not scaler_path.exists():
            scaler_path = Path(args.checkpoint).parent / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                temp_scaler = pickle.load(f)
            out_size = len(temp_scaler.scale_)
            logger.info(f"   Inferred out_size={out_size} from scaler.pkl")
        else:
            raise ValueError(
                "Cannot determine output size. Provide targets, --out_size, "
                "or ensure scaler.pkl exists in checkpoint directory."
            )
    
    # Load model and scaler
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    model, scaler = load_checkpoint(args.checkpoint, in_shape, out_size, args.model)
    
    # Handle ONNX export if requested
    if args.export == 'onnx':
        # Determine output path
        if args.export_path:
            export_path = args.export_path
        else:
            model_name = args.model or Path(args.checkpoint).parent.name.split('_')[0]
            export_path = str(Path(args.output_dir) / f"{model_name}_model.onnx")
        
        # Export with sample input for tracing
        sample_input = X_test[:1]  # Single sample for tracing
        success = export_to_onnx(
            model=model,
            sample_input=sample_input,
            output_path=export_path,
            opset_version=args.export_opset,
            validate=True
        )
        
        if success:
            # Print model info
            info = get_onnx_model_info(export_path)
            if 'error' not in info:
                logger.info(f"   📊 Model size: {info['file_size_mb']:.2f} MB")
                logger.info(f"   📊 Input: {info['input_name']} {info['input_shape']}")
                logger.info(f"   📊 Output: {info['output_name']} {info['output_shape']}")
            logger.info(f"✅ ONNX export completed: {export_path}")
        else:
            logger.error("❌ ONNX export failed")
            return
        
        # If only export was requested (no other outputs), exit early
        if not args.save_predictions and not args.plot and not args.verbose:
            logger.info("Export-only mode. Use --save_predictions or --plot for inference.")
            return
    
    # Ensure output directory exists before any file operations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    logger.info(f"Running inference on {len(X_test)} samples...")
    y_pred_scaled = run_inference(
        model, X_test, args.batch_size, device, num_workers=args.workers
    )
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    if has_targets:
        # === MODE: With ground truth ===
        y_true = y_test.numpy()
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)
        
        # Print results
        print_results(y_true, y_pred, metrics, args.param_names, args.verbose)
        
        # Save predictions (with targets)
        if args.save_predictions:
            output_path = output_dir / "predictions.csv"
            save_predictions(y_true, y_pred, str(output_path), args.param_names)
        
        # Generate plots
        if args.plot:
            logger.info("Generating plots...")
            plot_results(y_true, y_pred, str(output_dir), args.param_names)
    else:
        # === MODE: Predictions only (no ground truth) ===
        logger.info("No ground truth targets - skipping metrics and scatter plots")
        
        # Print prediction summary
        n_params = y_pred.shape[1]
        param_names = args.param_names or [f"P{i}" for i in range(n_params)]
        
        print("\n" + "="*80)
        print("PREDICTION SUMMARY (No Ground Truth)")
        print("="*80)
        print(f"Samples:          {len(y_pred)}")
        print(f"Output params:    {n_params}")
        print("-"*80)
        for i, name in enumerate(param_names):
            print(f"  {name:12s}: mean={y_pred[:, i].mean():.6f}, std={y_pred[:, i].std():.6f}")
        print("="*80)
        
        # Save predictions (without targets)
        if args.save_predictions:
            output_path = output_dir / "predictions.csv"
            columns = [f"Pred_{name}" for name in param_names]
            df = pd.DataFrame(y_pred, columns=columns)
            df.to_csv(output_path, index=False)
            logger.info(f"   ✔ Predictions saved to: {output_path}")
        
        if args.plot:
            logger.warning("Cannot generate scatter plots without ground truth targets")
    
    logger.info("✅ Testing completed successfully!")


if __name__ == "__main__":
    main()
