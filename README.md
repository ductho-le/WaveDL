<div align="center">

<img src="logos/wavedl_logo.png" alt="WaveDL Logo" width="500">

### A Scalable Deep Learning Framework for Wave-Based Inverse Problems

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?style=plastic&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?style=plastic&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-Enabled-yellow.svg?style=plastic&logo=huggingface&logoColor=white)](https://huggingface.co/docs/accelerate/)
<br>
[![Tests](https://img.shields.io/github/actions/workflow/status/ductho-le/WaveDL/test.yml?branch=main&style=plastic&logo=githubactions&logoColor=white&label=Tests)](https://github.com/ductho-le/WaveDL/actions/workflows/test.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/ductho-le/WaveDL/lint.yml?branch=main&style=plastic&logo=ruff&logoColor=white&label=Lint)](https://github.com/ductho-le/WaveDL/actions/workflows/lint.yml)
[![Try it on Colab](https://img.shields.io/badge/Try_it_on_Colab-8E44AD?style=plastic&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ductho-le/WaveDL/blob/main/notebooks/demo.ipynb)
<br>
[![Downloads](https://img.shields.io/badge/dynamic/json?url=https://pypistats.org/api/packages/wavedl/recent?period=month%26mirrors=false&query=data.last_month&style=plastic&logo=pypi&logoColor=white&color=9ACD32&label=Downloads&suffix=/month)](https://pypistats.org/packages/wavedl)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg?style=plastic)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.cpc.2026.110400-008080.svg?style=plastic)](https://doi.org/10.1016/j.cpc.2026.110400)

**Production-ready • Multi-GPU DDP • Memory-Efficient • Plug-and-Play**

[Getting Started](#-getting-started) •
[Documentation](#-documentation) •
[Examples](#-examples) •
[Discussions](https://github.com/ductho-le/WaveDL/discussions) •
[Citation](#-citation)

---

 **Plug in your model, load your data, and let WaveDL do the heavy lifting 💪**

</div>

---

## 💡 What is WaveDL?

WaveDL is a **deep learning framework** built for **wave-based inverse problems** — from ultrasonic NDE and geophysics to biomedical tissue characterization. It provides a robust, scalable training pipeline for mapping multi-dimensional data (1D/2D/3D) to physical quantities.

```
Input: Waveforms, spectrograms, B-scans, dispersion curves, ...
   ↓
Output: Material properties, defect dimensions, damage locations, ...
```

The framework handles the engineering challenges of large-scale deep learning — big datasets, distributed training, and HPC deployment — so you can focus on the science, not the infrastructure.

**Built for researchers who need:**
- 📊 Multi-target regression with reproducibility and fair benchmarking
- 🚀 Seamless multi-GPU training on HPC clusters
- 💾 Memory-efficient handling of large-scale datasets
- 🔧 Easy integration of custom model architectures

---

## ✨ Features

<div align="center">
<table width="100%">
<tr>
<td width="50%" valign="top">

**⚡ Load All Data — No More Bottleneck**

Train on datasets larger than RAM:
- Memory-mapped, zero-copy streaming
- Full random shuffling at GPU speed
- Your GPU stays fed — always

</td>
<td width="50%" valign="top">

**🧠 Models? We've Got Options**

20+ architectures (70+ variants), ready to go:
- CNNs, ResNets, ViTs, SSMs, WaveNets...
- All adapted for regression
- [Add your own](#adding-custom-models) in one line

</td>
</tr>
<tr>
<td width="50%" valign="top">

**🛡️ DDP That Actually Works**

Multi-GPU training without the pain:
- Synchronized early stopping
- Deadlock-free checkpointing
- Correct metric aggregation

</td>
<td width="50%" valign="top">

**🔬 Physics-Constrained Training**

Make your model respect the laws:
- Enforce bounds, positivity, equations
- Simple expression syntax or Python
- [Custom constraints](#physical-constraints) for various laws

</td>
</tr>
<tr>
<td width="50%" valign="top">

**🖥️ HPC-Native Design**

Built for high-performance clusters:
- Automatic GPU detection
- WandB experiment tracking
- BF16/FP16 mixed precision

</td>
<td width="50%" valign="top">

**🔄 Crash-Proof Training**

Never lose your progress:
- Full state checkpoints
- Resume from any point
- Emergency saves on interrupt

</td>
</tr>
<tr>
<td width="50%" valign="top">

**🎛️ Flexible & Reproducible Training**

Fully configurable via CLI flags or YAML:
- Loss functions, optimizers, schedulers
- K-fold cross-validation
- See [Configuration](#️-configuration) for details

</td>
<td width="50%" valign="top">

**📦 ONNX Export**

Deploy models anywhere:
- One-command export to ONNX
- LabVIEW, MATLAB, C++ compatible
- Validated PyTorch↔ONNX outputs

</td>
</tr>
</table>
</div>

---

## 🚀 Getting Started

### Installation

#### From PyPI (recommended for all users)

```bash
pip install --upgrade wavedl
```

This installs everything you need: training, inference, HPO, ONNX export.

#### From Source (for development)

```bash
git clone https://github.com/ductho-le/WaveDL.git
cd WaveDL
pip install -e ".[dev]"
```

> [!NOTE]
> Python 3.11+ required. For contributor setup (pre-commit hooks), see [CONTRIBUTING.md](.github/CONTRIBUTING.md).

### Quick Start

> [!TIP]
> In all examples below, replace `<...>` placeholders with your values. See [Configuration](#️-configuration) for defaults and options.

### Training

```bash
# Basic training (auto-detects GPUs and environment)
wavedl-train --model <model_name> --data_path <train_data> --output_dir <output_folder>

# Detailed configuration
wavedl-train --model <model_name> --data_path <train_data> --batch_size <number> \
  --lr <number> --epochs <number> --patience <number> --compile --output_dir <output_folder>

# Multi-GPU is automatic (uses all available GPUs)
# Override with --num_gpus if needed
wavedl-train --model cnn --data_path train.npz --num_gpus 4 --output_dir results

# Resume training (automatic - just re-run with same output_dir)
wavedl-train --model <model_name> --data_path <train_data> --output_dir <output_folder>

# Force fresh start (ignores existing checkpoints)
wavedl-train --model <model_name> --data_path <train_data> --output_dir <output_folder> --fresh

# List available models
wavedl-train --list_models
```

> [!NOTE]
> `wavedl-train` automatically detects your environment:
> - **HPC clusters** (SLURM, PBS, etc.): Uses local caching, offline WandB
> - **Local machines**: Uses standard cache locations (~/.cache)
>
> **Auto-Resume**: If training crashes or is interrupted, simply re-run with the same `--output_dir`. The framework automatically detects incomplete training and resumes from the last checkpoint.

<details>
<summary><b>Advanced: Direct Accelerate Launch</b></summary>

For fine-grained control over distributed training, you can use `accelerate launch` directly:

```bash
# Custom accelerate configuration
accelerate launch -m wavedl.train --model <model_name> --data_path <train_data> --output_dir <output_folder>

# Multi-node training
accelerate launch --num_machines 2 --main_process_ip <ip> -m wavedl.train --model cnn --data_path train.npz
```

</details>

### Testing & Inference

```bash
# Basic inference
wavedl-test --checkpoint <checkpoint_folder> --data_path <test_data>

# With visualization, CSV export, and multiple file formats
wavedl-test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --plot --plot_format png pdf --save_predictions --output_dir <output_folder>

# With custom parameter names
wavedl-test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --param_names '$p_1$' '$p_2$' '$p_3$' --plot

# Export model to ONNX for deployment (LabVIEW, MATLAB, C++, etc.)
wavedl-test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --export onnx --export_path <output_file.onnx>

# For 3D volumes with small depth (e.g., 8×128×128), override auto-detection
wavedl-test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --input_channels 1
```

**Output:**
- **Console**: R², Pearson correlation, MAE per parameter
- **CSV** (with `--save_predictions`): True, predicted, error, and absolute error for all parameters
- **Plots** (with `--plot`): 10 publication-quality plots (scatter, histogram, residuals, Bland-Altman, Q-Q, correlation, relative error, CDF, index plot, box plot)
- **Format** (with `--plot_format`): Supported formats: `png` (default), `pdf` (vector), `svg` (vector), `eps` (LaTeX), `tiff`, `jpg`, `ps`

> [!NOTE]
> `wavedl-test` auto-detects the model architecture from checkpoint metadata. If unavailable, it falls back to folder name parsing. Use `--model` to override if needed.

### Adding Custom Models

<details>
<summary><b>Creating Your Own Architecture</b></summary>

**Requirements** (your model must):
1. Inherit from `BaseModel`
2. Accept `in_shape`, `out_size` in `__init__`
3. Return a tensor of shape `(batch, out_size)` from `forward()`

---

**Step 1: Create `my_model.py`**

```python
import torch.nn as nn
import torch.nn.functional as F
from wavedl.models import BaseModel, register_model

@register_model("my_model")  # This name is used with --model flag
class MyModel(BaseModel):
    def __init__(self, in_shape, out_size, **kwargs):
        # in_shape: spatial dimensions, e.g., (128,) or (64, 64) or (32, 32, 32)
        # out_size: number of parameters to predict (auto-detected from data)
        super().__init__(in_shape, out_size)

        # Define your layers (this is just an example for 2D)
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)  # Input always has 1 channel
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128, out_size)

    def forward(self, x):
        # Input x has shape: (batch, 1, *in_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=[-2, -1])  # Global average pooling
        return self.fc(x)  # Output shape: (batch, out_size)
```

**Step 2: Train**

```bash
wavedl-train --import my_model.py --model my_model --data_path train.npz
```

WaveDL handles everything else: training loop, logging, checkpoints, multi-GPU, early stopping, etc.

</details>

---

## 📁 Project Structure

```
WaveDL/
├── src/
│   └── wavedl/                   # Main package (namespaced)
│       ├── __init__.py           # Package init with __version__
│       ├── train.py              # Training script
│       ├── test.py               # Testing & inference script
│       ├── hpo.py                # Hyperparameter optimization
│       ├── launcher.py           # Training launcher (wavedl-train)
│       │
│       ├── models/               # Model Zoo (20+ architectures, 70+ variants)
│       │   ├── registry.py       # Model factory (@register_model)
│       │   ├── base.py           # Abstract base class
│       │   └── ...               # See "Available Models" section
│       │
│       └── utils/                # Utilities
│           ├── data.py           # Memory-mapped data pipeline
│           ├── metrics.py        # R², Pearson, visualization
│           ├── constraints.py    # Physical constraints for training
│           ├── distributed.py    # DDP synchronization
│           ├── losses.py         # Loss function factory
│           ├── optimizers.py     # Optimizer factory
│           ├── schedulers.py     # LR scheduler factory
│           └── config.py         # YAML configuration support
│
├── configs/                      # YAML config templates
├── examples/                     # Ready-to-run examples
├── notebooks/                    # Jupyter notebooks
├── unit_tests/                   # Pytest test suite
│
├── pyproject.toml                # Package config, dependencies
├── CHANGELOG.md                  # Version history
└── CITATION.cff                  # Citation metadata
```
---

## ⚙️ Configuration

> [!NOTE]
> All configuration options below work with `wavedl-train`. The wrapper script passes all arguments directly to `train.py`.

<details>
<summary><b>Available Models</b> — 20+ architectures (70+ variants)</summary>

| Model | Backbone Params | Dim |
|-------|-----------------|-----|
| **── Classic CNNs ──** |||
| **CNN** — Convolutional Neural Network |||
| `cnn` | 1.6M | 1D/2D/3D |
| **ResNet** — Residual Network |||
| `resnet18` | 11.2M | 1D/2D/3D |
| `resnet34` | 21.3M | 1D/2D/3D |
| `resnet50` | 23.5M | 1D/2D/3D |
| `resnet18_pretrained` ⭐ | 11.2M | 2D |
| `resnet50_pretrained` ⭐ | 23.5M | 2D |
| **DenseNet** — Densely Connected Network |||
| `densenet121` | 7.0M | 1D/2D/3D |
| `densenet169` | 12.5M | 1D/2D/3D |
| `densenet121_pretrained` ⭐ | 7.0M | 2D |
| **── Efficient/Mobile CNNs ──** |||
| **MobileNetV3** — Mobile Neural Network V3 |||
| `mobilenet_v3_small` ⭐ | 0.9M | 2D |
| `mobilenet_v3_large` ⭐ | 3.0M | 2D |
| **EfficientNet** — Efficient Neural Network |||
| `efficientnet_b0` ⭐ | 5.3M | 2D |
| `efficientnet_b2` ⭐ | 9.1M | 2D |
| `efficientnet_b4` ⭐ | 19M | 2D |
| `efficientnet_b7` ⭐ | 66M | 2D |
| **EfficientNetV2** — Efficient Neural Network V2 |||
| `efficientnet_v2_s` ⭐ | 20.2M | 2D |
| `efficientnet_v2_m` ⭐ | 52.9M | 2D |
| `efficientnet_v2_l` ⭐ | 117.2M | 2D |
| **RegNet** — Regularized Network |||
| `regnet_y_400mf` ⭐ | 3.9M | 2D |
| `regnet_y_800mf` ⭐ | 5.7M | 2D |
| `regnet_y_1_6gf` ⭐ | 10.3M | 2D |
| `regnet_y_3_2gf` ⭐ | 17.9M | 2D |
| `regnet_y_8gf` ⭐ | 37.4M | 2D |
| **── Modern CNNs ──** |||
| **ConvNeXt** — Convolutional Next |||
| `convnext_tiny` | 27.8M | 1D/2D/3D |
| `convnext_small` | 49.5M | 1D/2D/3D |
| `convnext_base` | 87.6M | 1D/2D/3D |
| `convnext_tiny_pretrained` ⭐ | 27.8M | 2D |
| **ConvNeXt V2** — ConvNeXt with GRN |||
| `convnext_v2_tiny` | 27.9M | 1D/2D/3D |
| `convnext_v2_small` | 49.6M | 1D/2D/3D |
| `convnext_v2_base` | 87.7M | 1D/2D/3D |
| `convnext_v2_tiny_pretrained` ⭐ | 27.9M | 2D |
| **UniRepLKNet** — Large-Kernel ConvNet |||
| `unireplknet_tiny` | 30.8M | 1D/2D/3D |
| `unireplknet_small` | 56.0M | 1D/2D/3D |
| `unireplknet_base` | 97.6M | 1D/2D/3D |
| **── Vision Transformers ──** |||
| **ViT** — Vision Transformer |||
| `vit_tiny` | 5.4M | 1D/2D |
| `vit_small` | 21.4M | 1D/2D |
| `vit_base` | 85.3M | 1D/2D |
| **Swin** — Shifted Window Transformer |||
| `swin_t` ⭐ | 27.5M | 2D |
| `swin_s` ⭐ | 48.8M | 2D |
| `swin_b` ⭐ | 86.7M | 2D |
| **MaxViT** — Multi-Axis ViT |||
| `maxvit_tiny` ⭐ | 30.1M | 2D |
| `maxvit_small` ⭐ | 67.6M | 2D |
| `maxvit_base` ⭐ | 119.1M | 2D |
| **── Hybrid CNN-Transformer ──** |||
| **FastViT** — Fast Hybrid CNN-ViT |||
| `fastvit_t8` ⭐ | 4.0M | 2D |
| `fastvit_t12` ⭐ | 6.8M | 2D |
| `fastvit_s12` ⭐ | 8.8M | 2D |
| `fastvit_sa12` ⭐ | 10.9M | 2D |
| **CAFormer** — MetaFormer with Attention |||
| `caformer_s18` ⭐ | 26.3M | 2D |
| `caformer_s36` ⭐ | 39.2M | 2D |
| `caformer_m36` ⭐ | 56.9M | 2D |
| `poolformer_s12` ⭐ | 11.9M | 2D |
| **EfficientViT** — Memory-Efficient ViT |||
| `efficientvit_m1` ⭐ | 2.6M | 2D |
| `efficientvit_b1` ⭐ | 7.5M | 2D |
| `efficientvit_b2` ⭐ | 21.8M | 2D |
| `efficientvit_l2` ⭐ | 60.5M | 2D |
| **── State Space Models ──** |||
| **Mamba** — State Space Model |||
| `mamba_1d` | 3.4M | 1D |
| **S4D** — Diagonal Structured State Space |||
| `s4d_small` | 0.8M | 1D |
| `s4d` | 3.2M | 1D |
| `s4d_large` | 11M | 1D |
| **Vision Mamba (ViM)** — 2D Mamba |||
| `vim_tiny` | 6.6M | 2D |
| `vim_small` | 51.1M | 2D |
| `vim_base` | 201.4M | 2D |
| **── Specialized Architectures ──** |||
| **TCN** — Temporal Convolutional Network |||
| `tcn_small` | 0.9M | 1D |
| `tcn` | 6.9M | 1D |
| `tcn_large` | 10.0M | 1D |
| **WaveNet** — Gated Dilated Conv Network |||
| `wavenet_small` | 1.0M | 1D |
| `wavenet` | 4.0M | 1D |
| `wavenet_large` | 15M | 1D |
| **ResNet3D** — 3D Residual Network |||
| `resnet3d_18` | 33.2M | 3D |
| `mc3_18` — Mixed Convolution 3D | 11.5M | 3D |
| **U-Net** — U-shaped Network |||
| `unet_regression` | 31.0M | 1D/2D/3D |


⭐ = **Pretrained on ImageNet** (recommended for smaller datasets). Weights are downloaded automatically on first use.
- **Cache location**: `~/.cache/torch/hub/checkpoints/` (or `./.torch_cache/` on HPC if home is not writable)
- **Use pretrained weights**: Pass `--pretrained` (default: train from scratch). Only effective for `*_pretrained` model variants marked with ⭐ above.

**💡 HPC Users**: If compute nodes block internet, pre-download weights on the login node:

```bash
# Run once on login node (with internet) — downloads ALL pretrained weights
# Uses download-only approach (no model instantiation) to avoid CPU time limits
python -c "
import os, torch, warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
os.environ['TORCH_HOME'] = '.torch_cache'  # Match WaveDL's HPC cache location
os.environ['HF_HOME'] = '.hf_cache'        # Match WaveDL's HPC cache for timm models

from torchvision import models as m
from torchvision.models import video as v

# === TorchVision + Video Models — download only, no model instantiation ===
urls = [
    ('ResNet18',         m.ResNet18_Weights.IMAGENET1K_V1.url),
    ('ResNet50',         m.ResNet50_Weights.IMAGENET1K_V1.url),
    ('EfficientNet_B0',  m.EfficientNet_B0_Weights.IMAGENET1K_V1.url),
    ('EfficientNet_B2',  m.EfficientNet_B2_Weights.IMAGENET1K_V1.url),
    ('EfficientNet_B4',  m.EfficientNet_B4_Weights.IMAGENET1K_V1.url),
    ('EfficientNet_B7',  m.EfficientNet_B7_Weights.IMAGENET1K_V1.url),
    ('EfficientNetV2_S', m.EfficientNet_V2_S_Weights.IMAGENET1K_V1.url),
    ('EfficientNetV2_M', m.EfficientNet_V2_M_Weights.IMAGENET1K_V1.url),
    ('EfficientNetV2_L', m.EfficientNet_V2_L_Weights.IMAGENET1K_V1.url),
    ('MobileNetV3_S',    m.MobileNet_V3_Small_Weights.IMAGENET1K_V1.url),
    ('MobileNetV3_L',    m.MobileNet_V3_Large_Weights.IMAGENET1K_V1.url),
    ('RegNet_Y_400MF',   m.RegNet_Y_400MF_Weights.IMAGENET1K_V1.url),
    ('RegNet_Y_800MF',   m.RegNet_Y_800MF_Weights.IMAGENET1K_V1.url),
    ('RegNet_Y_1_6GF',   m.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1.url),
    ('RegNet_Y_3_2GF',   m.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1.url),
    ('RegNet_Y_8GF',     m.RegNet_Y_8GF_Weights.IMAGENET1K_V1.url),
    ('Swin_T',           m.Swin_T_Weights.IMAGENET1K_V1.url),
    ('Swin_S',           m.Swin_S_Weights.IMAGENET1K_V1.url),
    ('Swin_B',           m.Swin_B_Weights.IMAGENET1K_V1.url),
    ('ConvNeXt_Tiny',    m.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url),
    ('DenseNet121',      m.DenseNet121_Weights.IMAGENET1K_V1.url),
    ('R3D_18',           v.R3D_18_Weights.KINETICS400_V1.url),
    ('MC3_18',           v.MC3_18_Weights.KINETICS400_V1.url),
]
cache = os.path.join(os.environ['TORCH_HOME'], 'hub', 'checkpoints')
os.makedirs(cache, exist_ok=True)
for name, url in urls:
    torch.hub.download_url_to_file(url, os.path.join(cache, os.path.basename(url)))
    print(f'  ✓ {name}')

# === Timm Models — download only via HuggingFace Hub ===
import timm
from huggingface_hub import hf_hub_download
timm_models = [
    'maxvit_tiny_tf_224', 'maxvit_small_tf_224', 'maxvit_base_tf_224',
    'fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12',
    'caformer_s18', 'caformer_s36', 'caformer_m36', 'poolformer_s12',
    'convnextv2_tiny',
    'efficientvit_m1', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_l2',
]
for name in timm_models:
    cfg = timm.get_pretrained_cfg(name)
    if cfg.hf_hub_id:
        hf_hub_download(cfg.hf_hub_id, 'model.safetensors')
    elif cfg.url:
        torch.hub.download_url_to_file(cfg.url, os.path.join(cache, os.path.basename(cfg.url)))
    print(f'  ✓ {name}')

print(f'\n✓ All {len(urls) + len(timm_models)} pretrained weight files cached!')
"
```


</details>

<details>
<summary><b>Training Parameters</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `cnn` | Model architecture |
| `--import` | - | Python file(s) to import for custom models (supports multiple) |
| `--batch_size` | `128` | Per-GPU batch size |
| `--lr` | `1e-3` | Learning rate |
| `--epochs` | `1000` | Maximum epochs |
| `--patience` | `20` | Early stopping patience |
| `--weight_decay` | `1e-4` | AdamW regularization |
| `--grad_clip` | `1.0` | Gradient clipping |
| `--grad_accum_steps` | `1` | Gradient accumulation steps (simulate larger batch sizes) |

</details>

<details>
<summary><b>Data & I/O</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `train_data.npz` | Dataset path |
| `--workers` | `-1` | DataLoader workers per GPU (-1=auto-detect) |
| `--seed` | `2025` | Random seed |
| `--output_dir` | `.` | Output directory for checkpoints |
| `--resume` | `None` | Checkpoint to resume (auto-detected if not set) |
| `--save_every` | `50` | Checkpoint frequency |
| `--fresh` | `False` | Force fresh training, ignore existing checkpoints |
| `--single_channel` | `False` | Confirm data is single-channel (for shallow 3D volumes like `(8, 128, 128)`) |

</details>

<details>
<summary><b>Performance</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--compile` | `False` | Enable `torch.compile` (recommended for long runs) |
| `--precision` | `bf16` | Mixed precision mode (`bf16`, `fp16`, `no`) |
| `--workers` | `-1` | DataLoader workers per GPU (-1=auto, up to 16) |
| `--wandb` | `False` | Enable W&B logging |
| `--wandb_watch` | `False` | Enable W&B gradient watching (adds overhead) |
| `--project_name` | `DL-Training` | W&B project name |
| `--run_name` | `None` | W&B run name (auto-generated if not set) |

**Automatic GPU Optimizations:**

WaveDL automatically enables performance optimizations for modern GPUs:

| Optimization | Effect | GPU Support |
|--------------|--------|-------------|
| **TF32 precision** | ~2x speedup for float32 matmul | A100, H100 (Ampere+) |
| **cuDNN benchmark** | Auto-tuned convolutions | All NVIDIA GPUs |
| **Worker scaling** | Up to 16 workers per GPU | All systems |

> [!NOTE]
> These optimizations are **backward compatible** — they have no effect on older GPUs (V100, T4, GTX) or CPU-only systems. No configuration needed.

**HPC Best Practices:**
- Stage data to `$SLURM_TMPDIR` (local NVMe) for maximum I/O throughput
- Use `--compile` for training runs > 50 epochs
- Increase `--workers` manually if auto-detection is suboptimal

</details>

<details>
<summary><b>Distributed Training Arguments</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_gpus` | **Auto-detected** | Number of GPUs to use. By default, automatically detected via `nvidia-smi`. Set explicitly to override |
| `--num_machines` | `1` | Number of machines in distributed setup |
| `--mixed_precision` | `bf16` | Precision mode: `bf16`, `fp16`, or `no` |
| `--dynamo_backend` | `no` | PyTorch Dynamo backend |

**Environment Variables (for logging):**

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_MODE` | `offline` | WandB mode: `offline` or `online` |

</details>

<details>
<summary><b>Loss Functions</b></summary>

| Loss | Flag | Best For | Notes |
|------|------|----------|-------|
| `mse` | `--loss mse` | Default, smooth gradients | Standard Mean Squared Error |
| `mae` | `--loss mae` | Outlier-robust, linear penalty | Mean Absolute Error (L1) |
| `huber` | `--loss huber --huber_delta 1.0` | Best of MSE + MAE | Robust, smooth transition |
| `smooth_l1` | `--loss smooth_l1` | Similar to Huber | PyTorch native implementation |
| `log_cosh` | `--loss log_cosh` | Smooth approximation to MAE | Differentiable everywhere |
| `weighted_mse` | `--loss weighted_mse --loss_weights "2.0,1.0,1.0"` | Prioritize specific targets | Per-target weighting |

**Example:**
```bash
# Use Huber loss for noisy NDE data
wavedl-train --model cnn --loss huber --huber_delta 0.5

# Weighted MSE: prioritize thickness (first target)
wavedl-train --model cnn --loss weighted_mse --loss_weights "2.0,1.0,1.0"
```

</details>

<details>
<summary><b>Optimizers</b></summary>

| Optimizer | Flag | Best For | Key Parameters |
|-----------|------|----------|----------------|
| `adamw` | `--optimizer adamw` | Default, most cases | `--betas "0.9,0.999"` |
| `adam` | `--optimizer adam` | Legacy compatibility | `--betas "0.9,0.999"` |
| `sgd` | `--optimizer sgd` | Better generalization | `--momentum 0.9 --nesterov` |
| `nadam` | `--optimizer nadam` | Adam + Nesterov | Faster convergence |
| `radam` | `--optimizer radam` | Variance-adaptive | More stable training |
| `rmsprop` | `--optimizer rmsprop` | RNN/LSTM models | `--momentum 0.9` |

**Example:**
```bash
# SGD with Nesterov momentum (often better generalization)
wavedl-train --model cnn --optimizer sgd --lr 0.01 --momentum 0.9 --nesterov

# RAdam for more stable training
wavedl-train --model cnn --optimizer radam --lr 1e-3
```

</details>

<details>
<summary><b>Learning Rate Schedulers</b></summary>

| Scheduler | Flag | Best For | Key Parameters |
|-----------|------|----------|----------------|
| `plateau` | `--scheduler plateau` | Default, adaptive | `--scheduler_patience 10 --scheduler_factor 0.5` |
| `cosine` | `--scheduler cosine` | Long training, smooth decay | `--min_lr 1e-6` |
| `cosine_restarts` | `--scheduler cosine_restarts` | Escape local minima | Warm restarts |
| `onecycle` | `--scheduler onecycle` | Fast convergence | Super-convergence |
| `step` | `--scheduler step` | Simple decay | `--step_size 30 --scheduler_factor 0.1` |
| `multistep` | `--scheduler multistep` | Custom milestones | `--milestones "30,60,90"` |
| `exponential` | `--scheduler exponential` | Continuous decay | `--scheduler_factor 0.95` |
| `linear_warmup` | `--scheduler linear_warmup` | Warmup phase | `--warmup_epochs 5` |

**Example:**
```bash
# Cosine annealing for 1000 epochs
wavedl-train --model cnn --scheduler cosine --epochs 1000 --min_lr 1e-7

# OneCycleLR for super-convergence
wavedl-train --model cnn --scheduler onecycle --lr 1e-2 --epochs 50

# MultiStep with custom milestones
wavedl-train --model cnn --scheduler multistep --milestones "100,200,300"
```

</details>

<details>
<summary><b>Cross-Validation</b></summary>

For robust model evaluation, simply add the `--cv` flag:

```bash
# 5-fold cross-validation
wavedl-train --model cnn --cv 5 --data_path train_data.npz

# Stratified CV (recommended for unbalanced data)
wavedl-train --model cnn --cv 5 --cv_stratify --loss huber --epochs 100

# Full configuration
wavedl-train --model cnn --cv 5 --cv_stratify \
    --loss huber --optimizer adamw --scheduler cosine \
    --output_dir ./cv_results
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--cv` | `0` | Number of CV folds (0=disabled, normal training) |
| `--cv_stratify` | `False` | Use stratified splitting (bins targets) |
| `--cv_bins` | `10` | Number of bins for stratified CV |

**Output:**
- `cv_summary.json`: Aggregated metrics (mean ± std)
- `cv_results.csv`: Per-fold detailed results
- `fold_*/`: Individual fold models and scalers

</details>

<details>
<summary><b>Configuration Files (YAML)</b></summary>

Use YAML files for reproducible experiments. CLI arguments can override any config value.

```bash
# Use a config file
wavedl-train --config configs/config.yaml --data_path train.npz

# Override specific values from config
wavedl-train --config configs/config.yaml --lr 5e-4 --epochs 500
```

**Example config (`configs/config.yaml`):**
```yaml
# Model & Training
model: cnn
batch_size: 128
lr: 0.001
epochs: 1000
patience: 20

# Loss, Optimizer, Scheduler
loss: mse
optimizer: adamw
scheduler: plateau

# Cross-Validation (0 = disabled)
cv: 0

# Performance
precision: bf16
compile: false
seed: 2025
```

> See [`configs/config.yaml`](configs/config.yaml) for the complete template with all available options documented.

</details>

<details>
<summary><b>Physical Constraints</b> — Enforce Physics During Training</summary>

Add penalty terms to the loss function to enforce physical laws:

```
Total Loss = Data Loss + weight × penalty(violation)
```

### Expression Constraints

```bash
# Positivity
--constraint "y0 > 0"

# Bounds
--constraint "y0 >= 0" "y0 <= 1"

# Equations (penalize deviations from zero)
--constraint "y2 - y0 * y1"

# Input-dependent constraints
--constraint "y0 - 2*x[0]"

# Multiple constraints with different weights
--constraint "y0 > 0" "y1 - y2" --constraint_weight 0.1 1.0
```

### Custom Python Constraints

For complex physics (matrix operations, implicit equations):

```python
# my_constraint.py
import torch

def constraint(pred, inputs=None):
    """
    Args:
        pred:   (batch, num_outputs)
        inputs: (batch, features) or (batch, C, H, W) or (batch, C, D, H, W)
    Returns:
        (batch,) — violation per sample (0 = satisfied)
    """
    # Outputs (same for all data types)
    y0, y1, y2 = pred[:, 0], pred[:, 1], pred[:, 2]

    # Inputs — Tabular: (batch, features)
    # x0 = inputs[:, 0]                    # Feature 0
    # x_sum = inputs.sum(dim=1)            # Sum all features

    # Inputs — Images: (batch, C, H, W)
    # pixel = inputs[:, 0, 3, 5]           # Pixel at (3,5), channel 0
    # img_mean = inputs.mean(dim=(1,2,3))  # Mean over C,H,W

    # Inputs — 3D Volumes: (batch, C, D, H, W)
    # voxel = inputs[:, 0, 2, 3, 5]        # Voxel at (2,3,5), channel 0

    # Example constraints:
    # return y2 - y0 * y1                                    # Wave equation
    # return y0 - 2 * inputs[:, 0]                           # Output = 2×input
    # return inputs[:, 0, 3, 5] * y0 + inputs[:, 0, 6, 7] * y1  # Mixed

    return y0 - y1 * y2
```

```bash
--constraint_file my_constraint.py --constraint_weight 1.0
```

---

### Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--constraint` | — | Expression(s): `"y0 > 0"`, `"y0 - y1*y2"` |
| `--constraint_file` | — | Python file with `constraint(pred, inputs)` |
| `--constraint_weight` | `0.1` | Penalty weight(s) |
| `--constraint_reduction` | `mse` | `mse` (squared) or `mae` (linear) |

#### Expression Syntax

| Variable | Meaning |
|----------|---------|
| `y0`, `y1`, ... | Model outputs |
| `x[0]`, `x[1]`, ... | Input values (1D tabular) |
| `x[i,j]`, `x[i,j,k]` | Input values (2D/3D: images, volumes) |
| `x_mean`, `x_sum`, `x_max`, `x_min`, `x_std` | Input aggregates |

**Operators:** `+`, `-`, `*`, `/`, `**`, `>`, `<`, `>=`, `<=`, `==`

**Functions:** `sin`, `cos`, `exp`, `log`, `sqrt`, `sigmoid`, `softplus`, `tanh`, `relu`, `abs`

</details>



<details>
<summary><b>Hyperparameter Search (HPO)</b></summary>

Automatically find the best training configuration using [Optuna](https://optuna.org/).

**Run HPO:**

```bash
# Basic HPO (50 trials, auto-detects GPUs)
wavedl-hpo --data_path train.npz --n_trials 50

# Quick search (minimal search space, fastest)
wavedl-hpo --data_path train.npz --n_trials 30 --quick

# Medium search (balanced between quick and full)
wavedl-hpo --data_path train.npz --n_trials 50 --medium

# Full search with specific models
wavedl-hpo --data_path train.npz --n_trials 100 --models cnn resnet18 efficientnet_b0

# In-process mode (enables pruning, faster, single-GPU)
wavedl-hpo --data_path train.npz --n_trials 50 --inprocess
```

> [!TIP]
> **GPU Detection**: HPO auto-detects GPUs and runs one trial per GPU in parallel. Use `--inprocess` for single-GPU with pruning support (early stopping of bad trials).

**Train with best parameters**

After HPO completes, it prints the optimal command:
```bash
wavedl-train --data_path train.npz --model cnn --lr 3.2e-4 --batch_size 128 ...
```

---

**What Gets Searched:**

| Parameter | Default | You Can Override With |
|-----------|---------|----------------------|
| Models | cnn, resnet18, resnet34 | `--models X Y Z` |
| Optimizers | [all 6](#optimizers) | `--optimizers X Y` |
| Schedulers | [all 8](#learning-rate-schedulers) | `--schedulers X Y` |
| Losses | [all 6](#loss-functions) | `--losses X Y` |
| Learning rate | 1e-5 → 1e-2 | (always searched) |
| Batch size | 16, 32, 64, 128 | (always searched) |

**Search Presets:**

| Mode | Models | Optimizers | Schedulers | Use Case |
|------|--------|------------|------------|----------|
| Full (default) | cnn, resnet18, resnet34 | all 6 | all 8 | Production search |
| `--medium` | cnn, resnet18 | adamw, adam, sgd | plateau, cosine, onecycle | Balanced exploration |
| `--quick` | cnn | adamw | plateau | Fast validation |

**Execution Modes:**

| Mode | Flag | Pruning | GPU Memory | Best For |
|------|------|---------|------------|----------|
| Subprocess (default) | — | ❌ No | Isolated | Multi-GPU parallel trials |
| In-process | `--inprocess` | ✅ Yes | Shared | Single-GPU with early stopping |

> [!TIP]
> Use `--inprocess` when running single-GPU trials. It enables MedianPruner to stop unpromising trials early, reducing total search time.

---

**All Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | (required) | Training data file |
| `--models` | 3 defaults | Models to search (specify any number) |
| `--n_trials` | `50` | Number of trials to run |
| `--quick` | `False` | Quick mode: minimal search space |
| `--medium` | `False` | Medium mode: balanced search space |
| `--inprocess` | `False` | Run trials in-process (enables pruning) |
| `--optimizers` | all 6 | Optimizers to search |
| `--schedulers` | all 8 | Schedulers to search |
| `--losses` | all 6 | Losses to search |
| `--n_jobs` | `-1` | Parallel trials (-1 = auto-detect GPUs) |
| `--max_epochs` | `50` | Max epochs per trial |
| `--output` | `hpo_results.json` | Output file |


> See [Available Models](#available-models) for all 20+ architectures (70+ variants) you can search.

</details>

---

## 📈 Data Preparation

WaveDL supports multiple data formats for training and inference:

| Format | Extension | Key Advantages |
|--------|-----------|----------------|
| **NPZ** | `.npz` | Native NumPy, fast loading, recommended |
| **HDF5** | `.h5`, `.hdf5` | Large datasets, hierarchical, cross-platform |
| **MAT** | `.mat` | MATLAB compatibility (**v7.3+ only**, saved with `-v7.3` flag) |

**The framework automatically detects file format and data dimensionality** (1D, 2D, or 3D) — you only need to provide the appropriate model architecture.

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `input_train` / `input_test` | `(N, L)`, `(N, H, W)`, or `(N, D, H, W)` | `float32` | N samples of 1D/2D/3D representations |
| `output_train` / `output_test` | `(N, T)` | `float32` | N samples with T regression targets |

> [!TIP]
> - **Flexible Key Names**: WaveDL auto-detects common key pairs:
>   - `input_train`/`output_train`, `input_test`/`output_test` (WaveDL standard)
>   - `X`/`Y`, `x`/`y` (ML convention)
>   - `data`/`labels`, `inputs`/`outputs`, `features`/`targets`
> - **Automatic Dimension Detection**: Channel dimension is added automatically. No manual reshaping required!
> - **Sparse Matrix Support**: NPZ and MAT v7.3 files with scipy/MATLAB sparse matrices are automatically converted to dense arrays.
> - **Auto-Normalization**: Target values are automatically standardized during training. MAE is reported in original physical units.

> [!IMPORTANT]
> **MATLAB Users**: MAT files must be saved with the `-v7.3` flag for memory-efficient loading:
> ```matlab
> save('data.mat', 'input_train', 'output_train', '-v7.3')
> ```
> Older MAT formats (v5/v7) are not supported. Convert to NPZ for best compatibility.

<details>
<summary><b>Example: Basic Preparation</b></summary>

```python
import numpy as np

X = np.array(images, dtype=np.float32)  # (N, H, W)
y = np.array(labels, dtype=np.float32)  # (N, T)

np.savez('train_data.npz', input_train=X, output_train=y)
```

</details>

<details>
<summary><b>Example: From Image Files + CSV</b></summary>

```python
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

# Load images
images = [np.array(Image.open(f).convert('L'), dtype=np.float32)
          for f in sorted(Path("images/").glob("*.png"))]
X = np.stack(images)

# Load labels
y = pd.read_csv("labels.csv").values.astype(np.float32)

np.savez('train_data.npz', input_train=X, output_train=y)
```

</details>

<details>
<summary><b>Example: From MATLAB (.mat)</b></summary>

```python
import numpy as np
from scipy.io import loadmat

data = loadmat('simulation_data.mat')
X = data['spectrograms'].astype(np.float32)  # Adjust key
y = data['parameters'].astype(np.float32)

# Transpose if needed: (H, W, N) → (N, H, W)
if X.ndim == 3 and X.shape[2] < X.shape[0]:
    X = np.transpose(X, (2, 0, 1))

np.savez('train_data.npz', input_train=X, output_train=y)
```

</details>

<details>
<summary><b>Example: Synthetic Test Data</b></summary>

```python
import numpy as np

X = np.random.randn(1000, 256, 256).astype(np.float32)
y = np.random.randn(1000, 5).astype(np.float32)

np.savez('test_data.npz', input_test=X, output_test=y)
```

</details>

<details>
<summary><b>Validation Script</b></summary>

```python
import numpy as np

data = np.load('train_data.npz')
assert data['input_train'].ndim >= 2, "Input must be at least 2D: (N, ...) "
assert data['output_train'].ndim == 2, "Output must be 2D: (N, T)"
assert len(data['input_train']) == len(data['output_train']), "Sample mismatch"

print(f"✓ Input:  {data['input_train'].shape} {data['input_train'].dtype}")
print(f"✓ Output: {data['output_train'].shape} {data['output_train'].dtype}")
```

</details>


---

## 📦 Examples [![Try it on Colab](https://img.shields.io/badge/Try_it_on_Colab-8E44AD?style=plastic&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ductho-le/WaveDL/blob/main/notebooks/demo.ipynb)

The `examples/` folder contains a **complete, ready-to-run example** for **material characterization of isotropic plates**. The pre-trained MobileNetV3 predicts three physical parameters from Lamb wave dispersion curves:

| Parameter | Unit | Description |
|-----------|------|-------------|
| $h$ | mm | Plate thickness |
| $\sqrt{E/\rho}$ | km/s | Square root of Young's modulus over density |
| $\nu$ | — | Poisson's ratio |

> [!NOTE]
> This example is based on our paper at **SPIE Smart Structures + NDE 2026**: [*"A lightweight deep learning model for ultrasonic assessment of plate thickness and elasticity"*](https://doi.org/10.1117/12.3090206) (Proc. SPIE 13951, 1395106).

**Sample Dispersion Data:**

<p align="center">
  <img src="examples/elasticity_prediction/dispersion_samples.png" alt="Dispersion curve samples" width="700"><br>
  <em>Test samples showing the wavenumber-frequency relationship for different plate properties</em>
</p>

**Try it yourself:**

```bash
# Run inference on the example data
wavedl-test --checkpoint ./examples/elasticity_prediction/best_checkpoint \
  --data_path ./examples/elasticity_prediction/Test_data_100.mat \
  --plot --save_predictions --output_dir ./examples/elasticity_prediction/test_results

# Export to ONNX (already included as model.onnx)
wavedl-test --checkpoint ./examples/elasticity_prediction/best_checkpoint \
  --data_path ./examples/elasticity_prediction/Test_data_100.mat \
  --export onnx --export_path ./examples/elasticity_prediction/model.onnx
```

**What's Included:**

| File | Description |
|------|-------------|
| `best_checkpoint/` | Pre-trained MobileNetV3 checkpoint |
| `Test_data_100.mat` | 100 sample test set (500×500 dispersion curves → $h$, $\sqrt{E/\rho}$, $\nu$) |
| `dispersion_samples.png` | Visualization of sample dispersion curves with material parameters |
| `model.onnx` | ONNX export with embedded de-normalization |
| `training_history.csv` | Epoch-by-epoch training metrics (loss, R², LR, etc.) |
| `training_curves.png` | Training/validation loss and learning rate plot |
| `test_results/` | Example predictions and diagnostic plots |
| `WaveDL_ONNX_Inference.m` | MATLAB script for ONNX inference |

**Training Progress:**

<p align="center">
  <img src="examples/elasticity_prediction/training_curves.png" alt="Training curves" width="600"><br>
  <em>Training and validation loss with <code>plateau</code> learning rate schedule</em>
</p>

**Inference Results:**

<p align="center">
  <img src="examples/elasticity_prediction/test_results/scatter_all.png" alt="Scatter plot" width="700"><br>
  <em>Figure 1: Predictions vs ground truth for all three elastic parameters</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/error_histogram.png" alt="Error histogram" width="700"><br>
  <em>Figure 2: Distribution of prediction errors showing near-zero mean bias</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/residuals.png" alt="Residual plot" width="700"><br>
  <em>Figure 3: Residuals vs predicted values (no heteroscedasticity detected)</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/bland_altman.png" alt="Bland-Altman plot" width="700"><br>
  <em>Figure 4: Bland-Altman analysis with ±1.96 SD limits of agreement</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/qq_plot.png" alt="Q-Q plot" width="700"><br>
  <em>Figure 5: Q-Q plots confirming normally distributed prediction errors</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/error_correlation.png" alt="Error correlation" width="300"><br>
  <em>Figure 6: Error correlation matrix between parameters</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/relative_error.png" alt="Relative error" width="700"><br>
  <em>Figure 7: Relative error (%) vs true value for each parameter</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/error_cdf.png" alt="Error CDF" width="500"><br>
  <em>Figure 8: Cumulative error distribution — 95% of predictions within indicated bounds</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/prediction_vs_index.png" alt="Prediction vs index" width="700"><br>
  <em>Figure 9: True vs predicted values by sample index</em>
</p>

<p align="center">
  <img src="examples/elasticity_prediction/test_results/error_boxplot.png" alt="Error box plot" width="400"><br>
  <em>Figure 10: Error distribution summary (median, quartiles, outliers)</em>
</p>

---

## 🔬 Broader Applications

Beyond the material characterization example above, the WaveDL pipeline can be adapted for a wide range of **wave-based inverse problems** across multiple domains:

### 🏗️ Non-Destructive Evaluation & Structural Health Monitoring

| Application | Input | Output |
|-------------|-------|--------|
| Defect Sizing | A-scans, phased array images, FMC/TFM, ... | Crack length, depth, ... |
| Corrosion Estimation | Thickness maps, resonance spectra, ... | Wall thickness, corrosion rate, ... |
| Weld Quality Assessment | Phased array images, TOFD, ... | Porosity %, penetration depth, ... |
| RUL Prediction | Acoustic emission (AE), vibration spectra, ... | Cycles to failure, ... |
| Damage Localization | Wavefield images, DAS/DVS data, ... | Damage coordinates (x, y, z) |

### 🌍 Geophysics & Seismology

| Application | Input | Output |
|-------------|-------|--------|
| Seismic Inversion | Shot gathers, seismograms, ... | Velocity models, density profiles, ... |
| Subsurface Characterization | Surface wave dispersion, receiver functions, ... | Layer thickness, shear modulus, ... |
| Earthquake Source Parameters | Waveforms, spectrograms, ... | Magnitude, depth, focal mechanism, ... |
| Reservoir Characterization | Reflection seismic, AVO attributes, ... | Porosity, fluid saturation, ... |

### 🩺 Biomedical Ultrasound & Elastography

| Application | Input | Output |
|-------------|-------|--------|
| Tissue Elastography | Shear wave data, strain images, ... | Shear modulus, Young's modulus, ... |
| Liver Fibrosis Staging | Elastography images, US RF data, ... | Stiffness (kPa), fibrosis score, ... |
| Tumor Characterization | B-mode + elastography, ARFI data, ... | Lesion stiffness, size, ... |
| Bone QUS | Axial-transmission signals, ... | Porosity, cortical thickness, elastic modulus ... |

> [!NOTE]
> Adapting WaveDL to these applications requires preparing your own dataset and choosing a suitable model architecture to match your input dimensionality.

---

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| Technical Paper | In-depth framework description *(coming soon)* |
| [`_template.py`](src/wavedl/models/_template.py) | Template for custom architectures |

---

## 📜 Citation

If you use WaveDL in your research, please cite our published paper:

```bibtex
@article{le2026wavedl,
  author = {Le, Ductho and Phan, Haidang and Sacchi, Mauricio D. and Le, Lawrence H.},
  title = {{WaveDL}: A Scalable Deep Learning Framework for Wave-Based Inverse Problems},
  journal = {Computer Physics Communications},
  year = {2026},
  doi = {10.1016/j.cpc.2026.110400},
  url = {https://doi.org/10.1016/j.cpc.2026.110400}
}
```

Or in APA format:
> Le, D., Phan, H., Sacchi, M. D., & Le, L. H. (2026). WaveDL: A scalable deep learning framework for wave-based inverse problems. *Computer Physics Communications*. https://doi.org/10.1016/j.cpc.2026.110400

Additional bibliographic details will be updated when assigned by the journal.

Please also consider to cite other related papers:

```bibtex
@inproceedings{le2026lightweight,
  author = {Le, Ductho and Sacchi, Mauricio D. and Zhang, Xu and Ta, Dean and Lou, Edmond and Le, Lawrence H.},
  title = {A lightweight deep learning model for ultrasonic assessment of plate thickness and elasticity},
  booktitle = {Health Monitoring of Structural and Biological Systems XX},
  volume = {13951},
  pages = {1395106},
  year = {2026},
  publisher = {SPIE},
  doi = {10.1117/12.3090206},
  url = {https://doi.org/10.1117/12.3090206}
}
```

Or in APA format:
> Le, D., Sacchi, M. D., Zhang, X., Ta, D., Lou, E., & Le, L. H. (2026). A lightweight deep learning model for ultrasonic assessment of plate thickness and elasticity. *Proc. SPIE 13951, Health Monitoring of Structural and Biological Systems XX*, 1395106. https://doi.org/10.1117/12.3090206

---

## 🙏 Acknowledgments

Ductho Le would like to acknowledge [NSERC](https://www.nserc-crsng.gc.ca/) and [Alberta Innovates](https://albertainnovates.ca/) for supporting his study and research by means of a research assistantship and a graduate doctoral fellowship.

This research was enabled in part by support provided by [Compute Ontario](https://www.computeontario.ca/), [Calcul Québec](https://www.calculquebec.ca/), and the [Digital Research Alliance of Canada](https://alliancecan.ca/).

<br>

<p align="center">
  <a href="https://www.ualberta.ca/"><img src="logos/ualberta.png" alt="University of Alberta" height="60"></a>
  &emsp;&emsp;
  <a href="https://albertainnovates.ca/"><img src="logos/alberta_innovates.png" alt="Alberta Innovates" height="60"></a>
  &emsp;&emsp;
  <a href="https://www.nserc-crsng.gc.ca/"><img src="logos/nserc.png" alt="NSERC" height="60"></a>
</p>

<p align="center">
  <a href="https://alliancecan.ca/"><img src="logos/drac.png" alt="Digital Research Alliance of Canada" height="50"></a>
</p>

---

<div align="center">

**[Ductho Le](mailto:ductho.le@outlook.com)** · University of Alberta

[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3073--1416-a6ce39?style=plastic&logo=orcid&logoColor=white)](https://orcid.org/0000-0002-3073-1416)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?style=plastic&logo=google-scholar&logoColor=white)](https://scholar.google.ca/citations?user=OlwMr9AAAAAJ)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=plastic&logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Ductho-Le)

<sub>May your signals be strong and your attenuation low 👋</sub>

</div>
