# Troubleshooting Guide

This guide helps you resolve common issues when using WaveDL.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Loading Issues](#data-loading-issues)
- [Training Issues](#training-issues)
- [Multi-GPU / Distributed Training Issues](#multi-gpu--distributed-training-issues)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Inference / Testing Issues](#inference--testing-issues)
- [ONNX Export Issues](#onnx-export-issues)
- [General Debugging Tips](#general-debugging-tips)

---

## Installation Issues

### Issue: `pip install -e .` fails with dependency conflicts

**Solution:**
```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip first
pip install --upgrade pip

# Install WaveDL
pip install -e .
```

### Issue: PyTorch installation fails or wrong version installed

**Solution:**
```bash
# For CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install WaveDL
pip install -e .
```

### Issue: `ModuleNotFoundError: No module named 'models'` or `'utils'`

**Solution:**
```bash
# Make sure you're in the WaveDL root directory
cd /path/to/WaveDL

# Reinstall in editable mode
pip install -e .
```

### Issue: `triton` fails to install on Windows

**Solution:**
Triton is not supported on Windows. Install without the compile option:
```bash
pip install -e .  # Skip [all] which includes triton
```

---

## Data Loading Issues

### Issue: `KeyError: 'input_train'` or similar key errors

**Solution:**
WaveDL auto-detects common key pairs. Supported formats:
- `input_train`/`output_train`, `input_test`/`output_test` (recommended)
- `X`/`Y`, `x`/`y`
- `data`/`labels`, `inputs`/`outputs`, `features`/`targets`

Check your data keys:
```python
import numpy as np
data = np.load('your_data.npz')
print(data.files)  # List all keys

# Rename if needed
X = data['your_input_key']
Y = data['your_output_key']
np.savez('fixed_data.npz', input_train=X, output_train=Y)
```

### Issue: MATLAB `.mat` file won't load or "NotImplementedError"

**Solution:**
MAT files must be v7.3 format. In MATLAB:
```matlab
save('data.mat', 'input_train', 'output_train', '-v7.3')
```

For older MAT files, convert to NPZ:
```python
import numpy as np
from scipy.io import loadmat

data = loadmat('old_format.mat')
X = data['your_input_key'].astype(np.float32)
Y = data['your_output_key'].astype(np.float32)

np.savez('data.npz', input_train=X, output_train=Y)
```

### Issue: "Input must be 3D (N, H, W)" error

**Solution:**
Add channel dimension if missing:
```python
import numpy as np

data = np.load('data.npz')
X = data['input_train']

# If X is (N, H, W, C) -> transpose to (N, C, H, W)
if X.ndim == 4 and X.shape[-1] in [1, 3]:
    X = np.transpose(X, (0, 3, 1, 2))

# If X is (N, H, W) -> add channel dim -> (N, 1, H, W)
elif X.ndim == 3:
    X = X[:, np.newaxis, :, :]

np.savez('fixed_data.npz', input_train=X, output_train=data['output_train'])
```

### Issue: Sparse matrix warnings or errors

**Solution:**
Sparse matrices are automatically converted. To manually convert:
```python
import numpy as np
from scipy.sparse import issparse

data = np.load('data.npz', allow_pickle=True)
X = data['input_train']

if issparse(X):
    X = X.toarray().astype(np.float32)
    
np.savez('dense_data.npz', input_train=X, output_train=data['output_train'])
```

---

## Training Issues

### Issue: "CUDA out of memory" during training

**Solutions:**
1. Reduce batch size:
   ```bash
   accelerate launch train.py --model cnn --batch_size 64  # Try 32, 16, or 8
   ```

2. Use gradient accumulation:
   ```bash
   # Effective batch size = 32 * 4 = 128
   accelerate launch train.py --model cnn --batch_size 32 --gradient_accumulation_steps 4
   ```

3. Use mixed precision:
   ```bash
   accelerate launch train.py --model cnn --precision bf16
   ```

4. Reduce number of workers:
   ```bash
   accelerate launch train.py --model cnn --workers 0
   ```

### Issue: Training loss is NaN or explodes

**Solutions:**
1. Lower learning rate:
   ```bash
   accelerate launch train.py --model cnn --lr 1e-4  # Try 1e-5 if still fails
   ```

2. Enable gradient clipping (already default):
   ```bash
   accelerate launch train.py --model cnn --grad_clip 1.0
   ```

3. Check for invalid data:
   ```python
   import numpy as np
   
   data = np.load('train_data.npz')
   X = data['input_train']
   Y = data['output_train']
   
   print(f"X has NaN: {np.isnan(X).any()}")
   print(f"Y has NaN: {np.isnan(Y).any()}")
   print(f"X has Inf: {np.isinf(X).any()}")
   print(f"Y has Inf: {np.isinf(Y).any()}")
   ```

4. Try a different loss function:
   ```bash
   # Huber is more robust to outliers
   accelerate launch train.py --model cnn --loss huber
   ```

### Issue: Model not learning / Loss not decreasing

**Solutions:**
1. Check learning rate:
   ```bash
   # Try different learning rates
   accelerate launch train.py --model cnn --lr 1e-3  # Default
   accelerate launch train.py --model cnn --lr 5e-3  # Higher
   accelerate launch train.py --model cnn --lr 1e-4  # Lower
   ```

2. Disable early stopping temporarily:
   ```bash
   accelerate launch train.py --model cnn --patience 1000
   ```

3. Try a different optimizer:
   ```bash
   accelerate launch train.py --model cnn --optimizer adam
   accelerate launch train.py --model cnn --optimizer sgd --lr 0.01
   ```

4. Check data normalization - data should have reasonable scale

5. Try a simpler model first:
   ```bash
   accelerate launch train.py --model cnn  # Baseline
   ```

### Issue: "RuntimeError: DataLoader worker is killed by signal: Killed"

**Solution:**
Reduce number of workers:
```bash
accelerate launch train.py --model cnn --workers 0  # No multiprocessing
# or
accelerate launch train.py --model cnn --workers 2  # Fewer workers
```

### Issue: Training is very slow

**Solutions:**
1. Enable `torch.compile` (PyTorch 2.0+):
   ```bash
   accelerate launch train.py --model cnn --compile
   ```

2. Use mixed precision:
   ```bash
   accelerate launch train.py --model cnn --precision bf16
   ```

3. Increase batch size:
   ```bash
   accelerate launch train.py --model cnn --batch_size 256
   ```

4. Reduce number of workers if CPU-bound:
   ```bash
   accelerate launch train.py --model cnn --workers 4
   ```

---

## Multi-GPU / Distributed Training Issues

### Issue: "No GPUs detected" or training uses only one GPU

**Solution:**
Check GPU visibility:
```bash
# Check available GPUs
nvidia-smi

# Use run_training.sh (auto-detects GPUs)
./run_training.sh --model cnn --data_path train.npz

# Or manually set number of GPUs
NUM_GPUS=2 ./run_training.sh --model cnn --data_path train.npz

# Or configure accelerate
accelerate config  # Follow prompts
accelerate launch train.py --model cnn --data_path train.npz
```

### Issue: Hanging or deadlock in multi-GPU training

**Solutions:**
1. Reduce number of workers:
   ```bash
   NUM_GPUS=4 ./run_training.sh --model cnn --workers 2
   ```

2. Check for uneven data distribution:
   ```python
   # Ensure dataset size is divisible by (num_gpus * batch_size)
   # Or use drop_last=True in dataloader
   ```

3. Disable compile mode:
   ```bash
   NUM_GPUS=2 ./run_training.sh --model cnn  # Don't use --compile
   ```

### Issue: "RuntimeError: NCCL error" in multi-GPU training

**Solutions:**
1. Check CUDA/NCCL versions are compatible
2. Reduce batch size per GPU
3. Set environment variables:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_P2P_DISABLE=1  # Disable P2P if failing
   accelerate launch train.py --model cnn
   ```

---

## Memory Issues

### Issue: System runs out of RAM (not GPU memory)

**Solutions:**
1. Reduce DataLoader workers:
   ```bash
   accelerate launch train.py --model cnn --workers 0
   ```

2. WaveDL uses memory-mapped loading by default, but ensure your data is in the right format

3. Monitor memory usage:
   ```bash
   # Linux
   htop
   
   # macOS
   top
   
   # Windows
   Task Manager
   ```

### Issue: "Cannot allocate memory" during data loading

**Solution:**
```bash
# Use memory-mapped loading (default for NPZ/HDF5)
# Ensure data files are not corrupted
python -c "import numpy as np; data = np.load('train_data.npz'); print(data.files)"
```

---

## Performance Issues

### Issue: GPU utilization is low

**Solutions:**
1. Increase batch size:
   ```bash
   accelerate launch train.py --model cnn --batch_size 256
   ```

2. Reduce number of CPU workers (if I/O is not the bottleneck):
   ```bash
   accelerate launch train.py --model cnn --workers 2
   ```

3. Enable compile mode:
   ```bash
   accelerate launch train.py --model cnn --compile
   ```

### Issue: Training alternates between fast and slow epochs

**Solution:**
This is normal if using a scheduler like ReduceLROnPlateau. The validation phase runs every epoch and can be slower.

---

## Inference / Testing Issues

### Issue: "Checkpoint not found" or cannot load model

**Solution:**
```bash
# Check checkpoint structure
ls -la checkpoint_folder/

# Should contain:
# - config.json
# - model.safetensors (or pytorch_model.bin)
# - scaler_X.npz, scaler_Y.npz

# Specify checkpoint folder (not file)
python test.py --checkpoint ./output/best_checkpoint --data_path test.npz
```

### Issue: Test results show poor performance (but training was good)

**Solutions:**
1. Check data distribution:
   ```python
   import numpy as np
   
   train = np.load('train_data.npz')
   test = np.load('test_data.npz')
   
   print("Train output range:", train['output_train'].min(), train['output_train'].max())
   print("Test output range:", test['output_test'].min(), test['output_test'].max())
   ```

2. Ensure test data uses same normalization (handled automatically by scaler files)

3. Check for data leakage or overfitting

### Issue: Predictions are all the same value

**Solutions:**
1. Check if model converged during training
2. Verify test data format matches training data
3. Check scaler files are loaded correctly

---

## ONNX Export Issues

### Issue: "No module named 'onnx'" during export

**Solution:**
```bash
pip install -e ".[onnx]"
# or
pip install onnx onnxruntime
```

### Issue: ONNX export fails with operator errors

**Solutions:**
1. Some models may not be fully exportable. Try:
   ```bash
   python test.py --checkpoint ./checkpoint --data_path test.npz --export onnx --opset_version 14
   ```

2. For custom models, ensure all operations are ONNX-compatible

3. Check PyTorch version compatibility:
   ```bash
   pip install --upgrade torch onnx onnxruntime
   ```

### Issue: ONNX model gives different results than PyTorch

**Solution:**
This is checked automatically during export. If validation fails:
1. Check for non-deterministic operations
2. Ensure same precision (FP32) is used for both
3. Try different ONNX opset versions

---

## General Debugging Tips

### Enable verbose logging

```bash
# Python logging
export PYTHONVERBOSE=1

# PyTorch
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Accelerate
export ACCELERATE_DEBUG_MODE=1
```

### Check versions

```bash
python --version
pip show torch torchvision accelerate numpy
nvidia-smi  # Check CUDA version
```

### Test with minimal example

```python
# Create tiny dataset for quick testing
import numpy as np

X = np.random.randn(100, 64, 64).astype(np.float32)
Y = np.random.randn(100, 3).astype(np.float32)

np.savez('tiny_test.npz', input_train=X, output_train=Y)
```

```bash
# Quick training test
accelerate launch train.py --model cnn --data_path tiny_test.npz --epochs 5 --batch_size 16
```

### Get help

If none of these solutions work:

1. **Check the documentation**: [README](README.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/ductho-le/WaveDL/issues)
3. **Try the Colab demo**: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ductho-le/WaveDL/blob/main/notebooks/demo.ipynb)
4. **Ask in Discussions**: [GitHub Discussions](https://github.com/ductho-le/WaveDL/discussions)
5. **Open a new issue**: Use the [question template](https://github.com/ductho-le/WaveDL/issues/new/choose)

### Include this information when asking for help:

```bash
# System info
python --version
pip show wavedl torch torchvision accelerate
nvidia-smi  # If using GPU
uname -a  # Linux/macOS
# or
systeminfo  # Windows

# Your command
echo "Command: accelerate launch train.py --model cnn --data_path data.npz ..."

# Data info
python -c "import numpy as np; data = np.load('data.npz'); print('Keys:', data.files); print('Shapes:', {k: data[k].shape for k in data.files})"

# Error message (full traceback)
```

---

## Still Having Issues?

Create a new issue with:
- **Description**: What you're trying to do
- **Error message**: Full traceback
- **Environment**: OS, Python, PyTorch versions
- **Command**: Exact command you ran
- **Data info**: Shape and format of your data

Use the [bug report template](https://github.com/ductho-le/WaveDL/issues/new?template=bug_report.yml) for best results!
