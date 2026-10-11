# Security Policy

WaveDL takes the security of its code and machine learning workflows seriously. This document outlines supported versions, vulnerability reporting procedures, and security considerations when running deep learning pipelines on research or production systems.

## Supported Versions

Security fixes are prioritized for the latest minor release line:

| Version | Supported          |
| ------- | ------------------ |
| 1.9.x   | :white_check_mark: |
| 1.8.x   | :white_check_mark: |
| < 1.8   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability or security bug in WaveDL, please report it responsibly:

- **Primary Contact**: Email the maintainer directly at [ductho.le@outlook.com](mailto:ductho.le@outlook.com).
- **GitHub Security Advisory**: Alternatively, open a private draft advisory via [GitHub Security Advisories](https://github.com/ductho-le/WaveDL/security/advisories/new).

Please **do not** open public GitHub issues for sensitive security vulnerabilities. We will acknowledge receipt within 48 hours and work with you on a timeline for verification and remediation prior to public disclosure.

---

## Security Boundaries & Safe Usage

### 1. Model Checkpoints & Pickle Deserialization
PyTorch model checkpoints (`.bin`, `.pth`, `.pkl`) and Python pickle files can execute arbitrary bytecode upon deserialization.
- **Untrusted Weights**: Never load checkpoints or model artifacts from untrusted or unverified third parties.
- **Safe Serialization**: For model distribution across platforms and edge deployment, export your trained model to ONNX format using `wavedl-test --export_onnx`.
- **Training Meta**: Checkpoint directories contain `training_meta.pkl` and `scaler.pkl` used for resuming and normalizing metrics. Maintain standard file system permissions on your checkpoint directories (`--output_dir`).

### 2. NumPy NPZ & Data Loading
WaveDL supports NumPy `.npz` and HDF5 `.h5` files:
- Starting in WaveDL 1.8+, NumPy array loading defaults to `allow_pickle=False` by default.
- If your workflow requires loading legacy NumPy archives containing pickled object arrays from a trusted source, you must explicitly opt in using the `--allow_pickle` CLI argument.

### 3. Safe Expression Evaluation for Physical Constraints
The `--constraint` CLI option accepts mathematical penalty expressions (such as `"y0 - y1 * y2"`) to enforce domain-specific physical laws:
- Expression parsing is implemented strictly via Python's Abstract Syntax Tree (`ast.parse`) with an explicit whitelist of safe tensor operations (e.g., trigonometric, exponential, clamping).
- Dynamic code evaluation (`eval()`, `exec()`, or arbitrary imports) is prohibited and blocked at parse time.

### 4. Experiment Tracking & Environment
- **WandB**: By default, WaveDL sets `WANDB_MODE=offline` unless configured otherwise, preventing accidental transmission of telemetry or model outputs to external services on sensitive HPC systems.
- **Multi-Process Communication**: Distributed Data Parallel (DDP) communication operates over standard PyTorch distributed backends (NCCL for NVIDIA GPUs, Gloo for CPU/macOS). Ensure that your compute nodes communicate across isolated, secure internal cluster networks.
