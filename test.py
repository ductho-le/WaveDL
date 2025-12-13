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

import os
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
    
    return parser.parse_args()


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_npz_data(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data from NPZ format (WaveDL standard).
    
    Supports any input dimensionality:
        - 1D: (N, L) → (N, 1, L)
        - 2D: (N, H, W) → (N, 1, H, W)  
        - 3D: (N, D, H, W) → (N, 1, D, H, W)
        - Already has channel: (N, C, ...) → unchanged
    
    Supports flexible key names: input_test, input_train, X, data, etc.
    """
    # Supported key names (priority order - test keys first for inference, pairwise aligned)
    INPUT_KEYS = ['input_test', 'input_train', 'X', 'data', 'inputs', 'features', 'x']
    OUTPUT_KEYS = ['output_test', 'output_train', 'Y', 'labels', 'outputs', 'targets', 'y']
    
    data = np.load(file_path, allow_pickle=True)
    keys = list(data.keys())
    
    input_key = next((k for k in INPUT_KEYS if k in keys), None)
    output_key = next((k for k in OUTPUT_KEYS if k in keys), None)
    
    if input_key is None:
        raise KeyError(f"NPZ must contain input array. Supported keys: {INPUT_KEYS}. Found: {keys}")
    
    X = data[input_key]
    
    if output_key:
        y = data[output_key]
    else:
        logging.warning(f"No target data found in NPZ. Proceeding with predictions only.")
        y = np.zeros((len(X), 1))
    
    # Handle sparse matrices (requires scipy)
    if HAS_SCIPY:
        if issparse(X):
            X = X.toarray()
        if issparse(y):
            y = y.toarray()
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
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


def load_mat_data(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data from legacy MATLAB .mat format.
    
    Supports flexible key names: input_test, input_train, X, data, etc.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for MAT files: pip install scipy")
    
    # Supported key names (priority order - test keys first for inference, pairwise aligned)
    INPUT_KEYS = ['input_test', 'input_train', 'X', 'data', 'inputs', 'features', 'x']
    OUTPUT_KEYS = ['output_test', 'output_train', 'Y', 'labels', 'outputs', 'targets', 'y']
    
    mat = scipy.io.loadmat(file_path)
    valid_keys = [k for k in mat.keys() if not k.startswith('__')]
    
    # Try flexible key detection first, then fallback to legacy auto-detect
    input_key = next((k for k in INPUT_KEYS if k in valid_keys), None)
    
    # Fallback: legacy auto-detect for old-style MAT files
    if input_key is None:
        input_key = next(
            (k for k in valid_keys if any(s in k.lower() for s in ['input', 'mat', 'kf', 'image'])),
            valid_keys[0] if valid_keys else None
        )
    
    if input_key is None:
        raise KeyError(f"MAT must contain input array. Supported keys: {INPUT_KEYS}. Found: {valid_keys}")
    
    data = mat[input_key]
    
    # Handle MATLAB struct arrays
    if data.ndim == 2 and data.shape[0] == 1:
        data = data.T
    
    imgs = []
    for i in range(data.shape[0]):
        try:
            sample = data[i, 0] if data.ndim > 1 else data[i]
            
            # Handle struct fields
            if hasattr(sample, 'dtype') and sample.dtype.names:
                raw = sample['kf'] if 'kf' in sample.dtype.names else sample[sample.dtype.names[0]]
            else:
                raw = sample
            
            # Unpack nested arrays
            while isinstance(raw, np.ndarray) and raw.size == 1 and raw.ndim >= 2:
                raw = raw[0, 0]
            
            # Convert sparse to dense
            dense = (raw.toarray() if issparse(raw) else np.array(raw)).astype(np.float32)
            
            # Reshape if 1D
            if dense.ndim == 1:
                side = int(np.sqrt(dense.size))
                dense = dense.reshape(side, side)
            
            imgs.append(torch.tensor(dense))
        except Exception as e:
            logging.warning(f"Skipping sample {i}: {e}")
            continue
    
    if not imgs:
        raise ValueError("No valid samples found in MAT file")
    
    X = torch.stack(imgs).unsqueeze(1)  # Add channel dim
    
    # Try flexible key detection for output
    output_key = next((k for k in OUTPUT_KEYS if k in valid_keys), None)
    
    # Fallback: legacy auto-detect
    if output_key is None:
        output_key = next(
            (k for k in valid_keys if any(s in k.lower() for s in ['sample', 'target', 'output', 'label'])),
            None
        )
    
    if output_key and output_key in mat:
        y_np = mat[output_key]
        # Transpose if needed (parameters as rows -> parameters as columns)
        if y_np.shape[0] < y_np.shape[1] and y_np.shape[0] <= 10:
            y_np = y_np.T
        
        # Match length
        min_len = min(len(y_np), len(X))
        y = torch.tensor(y_np[:min_len].astype(np.float32))
    else:
        logging.warning(f"No target data found. Proceeding with predictions only.")
        y = torch.zeros((len(X), 1))  # Dummy targets
    
    return X, y


def load_hdf5_data(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data from HDF5 (.h5, .hdf5) format.
    
    Supports flexible key detection: input_test, input_train, X, data, etc.
    """
    import h5py
    
    # Supported key names (priority order)
    INPUT_KEYS = ['input_test', 'input_train', 'X', 'data', 'inputs', 'features', 'x']
    OUTPUT_KEYS = ['output_test', 'output_train', 'Y', 'labels', 'outputs', 'targets', 'y']
    
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        
        input_key = next((k for k in INPUT_KEYS if k in keys), None)
        output_key = next((k for k in OUTPUT_KEYS if k in keys), None)
        
        if input_key is None:
            raise KeyError(f"HDF5 must contain input array. Supported keys: {INPUT_KEYS}. Found: {keys}")
        
        X = f[input_key][:]
        
        if output_key:
            y = f[output_key][:]
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


def load_test_data(file_path: str, format: str = 'auto') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load test data with automatic format detection.
    
    Supports NPZ, MAT, and HDF5 formats.
    
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
    
    if format == 'npz':
        X, y = load_npz_data(str(file_path))
    elif format == 'mat':
        X, y = load_mat_data(str(file_path))
    elif format == 'hdf5':
        X, y = load_hdf5_data(str(file_path))
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
    device: torch.device = None
) -> np.ndarray:
    """
    Run batch inference on test data.
    
    Returns:
        predictions: Numpy array (N, out_size) - still in normalized space
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predictions = []
    
    for (batch_x,) in tqdm(loader, desc="Inference", leave=False):
        batch_x = batch_x.to(device)
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
    X_test, y_test = load_test_data(args.data_path, args.format)
    in_shape = tuple(X_test.shape[2:])
    out_size = y_test.shape[1]
    
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
    
    # Run inference
    logger.info(f"Running inference on {len(X_test)} samples...")
    y_pred_scaled = run_inference(model, X_test, args.batch_size, device)
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = y_test.numpy()
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Print results
    print_results(y_true, y_pred, metrics, args.param_names, args.verbose)
    
    # Save predictions
    if args.save_predictions:
        output_path = Path(args.output_dir) / "predictions.csv"
        save_predictions(y_true, y_pred, str(output_path), args.param_names)
    
    # Generate plots
    if args.plot:
        logger.info("Generating plots...")
        plot_results(y_true, y_pred, args.output_dir, args.param_names)
    
    logger.info("✅ Testing completed successfully!")


if __name__ == "__main__":
    main()
