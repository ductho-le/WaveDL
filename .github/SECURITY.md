# Security Policy

WaveDL takes the security of its code and machine learning workflows seriously. This document outlines supported versions, vulnerability reporting procedures, and security considerations when running deep learning pipelines on research or production systems.

## Supported Versions

Security fixes are prioritized for the latest minor release line:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.9.x   | :white_check_mark: |
| < 1.9   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability or security bug in WaveDL, please report it responsibly:

- **Primary Contact**: Email the maintainer directly at [ductho.le@outlook.com](mailto:ductho.le@outlook.com).
- **GitHub Security Advisory**: Alternatively, open a private draft advisory via [GitHub Security Advisories](https://github.com/ductho-le/WaveDL/security/advisories/new).

Please **do not** open public GitHub issues for sensitive security vulnerabilities. We will acknowledge receipt within 48 hours and work with you on a timeline for verification and remediation prior to public disclosure.

---

## Security Boundaries & Safe Usage

### 1. Model Checkpoints & Pickle Deserialization
PyTorch training states (`.bin`, `.pth`, optimizer/scheduler checkpoints) and legacy Python pickle files (`.pkl`) can execute arbitrary bytecode upon deserialization.
- **Safe Checkpoint Metadata & Inference (v2.0+)**: Starting in WaveDL 2.0, checkpoints persist training metadata in `training_meta.json` and built-in target scalers in `scaler.json`, while `wavedl-test` loads model weights via `safetensors` or `torch.load(..., weights_only=True)`. Inference and ONNX export on v2.0+ checkpoints therefore avoid pickle deserialization. Legacy v1.x `training_meta.pkl` and `scaler.pkl` files remain supported as a fallback for trusted checkpoints.
- **Untrusted Checkpoints & Full Training Resume**: Never resume training (`--resume`) or load legacy `.pkl` artifacts from untrusted or unverified third parties, as full training state restoration (`accelerator.load_state`) deserializes optimizer, scheduler, and RNG state.
- **Safe Distribution**: For model distribution across platforms and edge deployment, export your trained model to ONNX format using `wavedl-test --export onnx --export_path model.onnx`.
- **Filesystem Permissions**: Maintain appropriate filesystem permissions on your checkpoint and cache directories (`--output_dir`, `--cache_dir`).

### 2. NumPy NPZ & Data Loading
WaveDL supports NumPy `.npz` and HDF5 `.h5` files:
- Starting in WaveDL 1.8+, NumPy array loading defaults to `allow_pickle=False` by default.
- If your workflow requires loading legacy NumPy archives containing pickled object arrays from a trusted source, you must explicitly opt in using the `--allow_pickle` CLI argument.

### 3. Safe Expression Evaluation for Physical Constraints
The `--constraint` CLI option accepts mathematical penalty expressions (such as `"y0 - y1 * y2"`) to enforce domain-specific physical laws:
- Expression parsing is implemented strictly via Python's Abstract Syntax Tree (`ast.parse`) with an explicit whitelist of safe tensor operations (e.g., trigonometric, exponential, clamping).
- Dynamic code evaluation (`eval()`, `exec()`, or arbitrary imports) is prohibited and blocked at parse time.

### 4. Experiment Tracking & Environment
- **WandB**: On HPC clusters and environments with non-writable home directories, `wavedl-train` automatically sets `WANDB_MODE=offline` unless configured otherwise, preventing accidental transmission of telemetry or model outputs on compute nodes. On local workstations, set `WANDB_MODE=offline` explicitly if you require air-gapped logging when `--wandb` is enabled.
- **Multi-Process Communication**: Distributed Data Parallel (DDP) communication operates over standard PyTorch distributed backends (NCCL for NVIDIA GPUs, Gloo for CPU/macOS). Ensure that your compute nodes communicate across isolated, secure internal cluster networks.

### 5. WaveDL Studio (`wavedl-gui`) Local Web Workspace
WaveDL Studio is designed as a **single-user local workstation tool**:
- **Loopback-Only Default**: `wavedl-gui` binds to `127.0.0.1` by default and refuses non-loopback `--host` values unless `--allow-remote` is explicitly passed.
- **Session Token Scope**: The per-session `X-WaveDL-Token` header (combined with `Host`, `Origin`, and `Sec-Fetch-Site` checks) mitigates cross-site browser requests (CSRF / DNS rebinding). Because the token is embedded in the served HTML page, it is **not** network-level authentication—any client that can directly reach a non-loopback listener can obtain it, read/write YAML configs, and launch jobs.
- **Remote Access**: For remote servers or HPC login nodes, keep the default loopback listener (`wavedl-gui --port 8080 --no-browser`) and connect via SSH port forwarding (`ssh -L 8080:127.0.0.1:8080 user@host`). Use `--restrict-workspace` to confine file browsing and YAML reads/writes to the launch directory.
