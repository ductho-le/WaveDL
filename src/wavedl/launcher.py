#!/usr/bin/env python
"""
WaveDL Training Launcher.

This module provides a universal training launcher that wraps accelerate
for distributed training. It works seamlessly on both:
- Local machines (uses standard cache locations)
- HPC clusters (uses local caching, offline WandB)

The environment is auto-detected based on scheduler variables (SLURM, PBS, etc.)
and home directory writability.

Usage:
    # Local machine or HPC - same command!
    wavedl-train --model cnn --data_path train.npz --output_dir results

    # Multi-GPU is automatic (uses all available GPUs)
    wavedl-train --model resnet18 --data_path train.npz --num_gpus 4

Example SLURM script:
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --gpus-per-node=4
    #SBATCH --time=12:00:00

    wavedl-train --model cnn --data_path /scratch/data.npz --compile

Author: Ductho Le (ductho.le@outlook.com)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def detect_gpus() -> int:
    """Auto-detect available GPUs using nvidia-smi."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [
            value.strip()
            for value in visible.split(",")
            if value.strip() and value.strip() != "-1"
        ]
        return max(1, len(devices))
    if shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi not found, defaulting to NUM_GPUS=1")
        return 1

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_output = result.stdout.strip()
        gpu_count = len(gpu_output.split("\n")) if gpu_output else 0
        if gpu_count > 0:
            print(f"Auto-detected {gpu_count} GPU(s)")
            return gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print("Warning: No GPUs detected, defaulting to NUM_GPUS=1")
    return 1


def is_hpc_environment() -> bool:
    """Detect if running on an HPC cluster.

    Checks for:
    1. Common HPC scheduler environment variables (SLURM, PBS, LSF, SGE, Cobalt)
    2. Non-writable home directory (common on HPC systems)

    Returns:
        True if HPC environment detected, False otherwise.
    """
    # Check for common HPC scheduler environment variables
    hpc_indicators = [
        "SLURM_JOB_ID",  # SLURM
        "PBS_JOBID",  # PBS/Torque
        "LSB_JOBID",  # LSF
        "SGE_TASK_ID",  # Sun Grid Engine
        "COBALT_JOBID",  # Cobalt
    ]
    if any(var in os.environ for var in hpc_indicators):
        return True

    # Check if home directory is not writable (common on HPC)
    home = os.path.expanduser("~")
    return not os.access(home, os.W_OK)


def setup_environment() -> None:
    """Configure environment for HPC or local machine.

    Automatically detects the environment and configures accordingly:
    - HPC: Uses CWD-based caching, offline WandB (compute nodes lack internet)
    - Local: Uses standard cache locations (~/.cache), doesn't override WandB
    """
    is_hpc = is_hpc_environment()

    if is_hpc:
        # HPC: use CWD-based caching (compute nodes lack internet)
        cache_base = os.getcwd()

        # TORCH_HOME set to CWD - compute nodes need pre-cached weights
        os.environ.setdefault("TORCH_HOME", f"{cache_base}/.torch_cache")
        Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

        # HuggingFace Hub cache (timm downloads pretrained weights from HF Hub)
        os.environ.setdefault("HF_HOME", f"{cache_base}/.hf_cache")
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # No HTTP on compute nodes

        # Triton/Inductor caches - prevents permission errors with --compile
        os.environ.setdefault("TRITON_CACHE_DIR", f"{cache_base}/.triton_cache")
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR", f"{cache_base}/.inductor_cache"
        )
        Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

        # Matplotlib/fontconfig caches - always redirect on HPC
        # Even with a writable home, ~/.cache/matplotlib/tex.cache often has
        # permission issues on compute nodes (shared filesystems, quotas).
        os.environ.setdefault("MPLCONFIGDIR", f"{cache_base}/.matplotlib")
        os.environ.setdefault("FONTCONFIG_CACHE", f"{cache_base}/.fontconfig")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["FONTCONFIG_CACHE"]).mkdir(parents=True, exist_ok=True)

        # XDG cache only if home is not writable
        home = os.path.expanduser("~")
        home_writable = os.access(home, os.W_OK)
        if not home_writable:
            os.environ.setdefault("XDG_CACHE_HOME", f"{cache_base}/.cache")
            Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

        # WandB configuration (offline by default for HPC)
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_DIR", f"{cache_base}/.wandb")
        os.environ.setdefault("WANDB_CACHE_DIR", f"{cache_base}/.wandb_cache")
        os.environ.setdefault("WANDB_CONFIG_DIR", f"{cache_base}/.wandb_config")

        print("🖥️  HPC environment detected - using local caching")
    else:
        # Local machine: use standard locations, don't override user settings
        # TORCH_HOME defaults to ~/.cache/torch (PyTorch default)
        # WANDB_MODE defaults to online (WandB default)
        print("💻 Local environment detected - using standard cache locations")


def handle_fast_path_args() -> int | None:
    """Handle utility flags that don't need accelerate launch.

    Returns:
        Exit code if handled (0 for success), None if should continue to full launch.
    """
    # --list_models: print models and exit immediately
    if "--list_models" in sys.argv:
        from wavedl.models import get_model, import_model_modules, list_models

        import_parser = argparse.ArgumentParser(add_help=False)
        import_parser.add_argument(
            "--import", dest="import_modules", nargs="+", default=[]
        )
        import_args, _ = import_parser.parse_known_args()
        import_model_modules(import_args.import_modules)

        print("Available models (output contract):")
        for name in list_models():
            model_class = get_model(name)
            supports_field = getattr(model_class, "supports_spatial_output", False)
            requires_field = getattr(model_class, "requires_spatial_output", False)
            if requires_field:
                output_contract = "dense 2D field"
            elif supports_field:
                output_contract = "vector + dense 2D field"
            else:
                output_contract = "vector"
            print(f"  {name:<32} {output_contract}")
        return 0

    return None  # Continue to full launch


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher-specific arguments, pass remaining to wavedl.train."""
    parser = argparse.ArgumentParser(
        description="WaveDL Training Launcher (works on local machines and HPC clusters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (auto-detects GPUs and environment)
  wavedl-train --model cnn --data_path train.npz --output_dir results

  # Specify GPU count explicitly
  wavedl-train --model cnn --data_path train.npz --num_gpus 4

  # Full configuration
  wavedl-train --model resnet18 --data_path train.npz --batch_size 256 \\
               --lr 1e-3 --epochs 100 --compile --output_dir ./results

  # List available models
  wavedl-train --list_models

Environment Detection:
  The launcher automatically detects your environment:
  - HPC (SLURM, PBS, etc.): Uses local caching, offline WandB
  - Local machine: Uses standard cache locations (~/.cache)

For full training options, see: python -m wavedl.train --help
""",
    )

    # HPC-specific arguments
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)",
    )
    parser.add_argument(
        "--num_machines",
        type=int,
        default=1,
        help="Number of machines for multi-node training (default: 1)",
    )
    parser.add_argument(
        "--machine_rank",
        type=int,
        default=0,
        help="Rank of this machine in multi-node setup (default: 0)",
    )
    parser.add_argument(
        "--main_process_ip",
        type=str,
        default=None,
        help="IP address of the main process for multi-node training",
    )
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=None,
        help="Port for multi-node communication (default: accelerate auto-selects)",
    )
    parser.add_argument(
        "--mixed_precision",
        "--precision",
        type=str,
        choices=["bf16", "fp16", "no"],
        default="bf16",
        help="Mixed precision mode (default: bf16)",
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="no",
        help="PyTorch dynamo backend (default: no)",
    )

    # Parse known args, pass rest to wavedl.train
    args, remaining = parser.parse_known_args()
    if args.num_gpus is not None and args.num_gpus < 1:
        parser.error("--num_gpus must be >= 1.")
    if args.num_machines < 1 or not 0 <= args.machine_rank < args.num_machines:
        parser.error(
            "--num_machines must be positive and --machine_rank must be within its range."
        )
    if args.num_machines > 1 and not args.main_process_ip:
        parser.error("--main_process_ip is required for multi-node training.")
    if args.main_process_port is not None and not 1 <= args.main_process_port <= 65535:
        parser.error("--main_process_port must be between 1 and 65535.")
    return args, remaining


def print_summary(
    exit_code: int, wandb_enabled: bool, wandb_mode: str, wandb_dir: str
) -> None:
    """Print post-training summary and instructions."""
    print()
    print("=" * 40)

    if exit_code == 0:
        print("✅ Training completed successfully!")
        print("=" * 40)

        # Only show WandB sync instructions if user enabled wandb
        if wandb_enabled and wandb_mode == "offline":
            print()
            print("📊 WandB Sync Instructions:")
            print("   From the login node, run:")
            print(f"   wandb sync {wandb_dir}/wandb/offline-run-*")
            print()
            print("   This will upload your training logs to wandb.ai")
    else:
        print(f"❌ Training failed with exit code: {exit_code}")
        print("=" * 40)
        print()
        print("Common issues:")
        print("  - Missing data file (check --data_path)")
        print("  - Insufficient GPU memory (reduce --batch_size)")
        print("  - Invalid model name (run: wavedl-train --list_models)")
        print()

    print("=" * 40)
    print()


def main() -> int:
    """Main entry point for wavedl-train command."""
    # Fast path for utility flags (avoid accelerate launch overhead)
    exit_code = handle_fast_path_args()
    if exit_code is not None:
        return exit_code

    # Parse arguments
    args, train_args = parse_args()

    # Setup environment (smart detection)
    setup_environment()

    # Check if wavedl package is importable
    try:
        import wavedl  # noqa: F401
    except ImportError:
        print("Error: wavedl package not found. Run: pip install -e .", file=sys.stderr)
        return 1

    # Auto-detect GPUs if not specified
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
        print(f"Using NUM_GPUS={num_gpus} (set via command line)")
    else:
        num_gpus = detect_gpus()

    # CV uses an independent single-process loop, so it must never be replicated.
    cv_parser = argparse.ArgumentParser(add_help=False)
    cv_parser.add_argument("--cv", type=int, default=None)
    cv_parser.add_argument("--config", default=None)
    cv_args, _ = cv_parser.parse_known_args(train_args)
    launcher_config = {}
    if cv_args.config:
        from wavedl.utils.config import load_config

        launcher_config = load_config(cv_args.config)
    cv_folds = cv_args.cv if cv_args.cv is not None else launcher_config.get("cv", 0)
    explicit_precision = any(
        argument.split("=", 1)[0] in {"--precision", "--mixed_precision"}
        for argument in sys.argv[1:]
    )
    if not explicit_precision:
        args.mixed_precision = launcher_config.get(
            "precision", launcher_config.get("mixed_precision", args.mixed_precision)
        )
    if args.mixed_precision not in {"no", "fp16", "bf16"}:
        print("Error: precision must be 'no', 'fp16', or 'bf16'.", file=sys.stderr)
        return 2
    if cv_folds and cv_folds > 0:
        if args.num_machines > 1:
            print("Error: Cross-validation requires one machine.", file=sys.stderr)
            return 2
        num_gpus = 1

    # Keep Accelerate and training precision consistent (both accept the aliases).
    train_args += ["--precision", args.mixed_precision]

    # Build accelerate launch command
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        f"--num_processes={num_gpus * args.num_machines}",
        f"--num_machines={args.num_machines}",
        f"--machine_rank={args.machine_rank}",
        f"--mixed_precision={args.mixed_precision}",
        f"--dynamo_backend={args.dynamo_backend}",
    ]

    # Explicitly set multi_gpu to suppress accelerate auto-detection warning
    if num_gpus * args.num_machines > 1:
        cmd.append("--multi_gpu")

    # Add multi-node networking args if specified (required for some clusters)
    if args.main_process_ip:
        cmd.append(f"--main_process_ip={args.main_process_ip}")
    if args.main_process_port:
        cmd.append(f"--main_process_port={args.main_process_port}")

    cmd += ["-m", "wavedl.train"] + train_args

    # Create output directory if specified
    for i, arg in enumerate(train_args):
        if arg == "--output_dir" and i + 1 < len(train_args):
            Path(train_args[i + 1]).mkdir(parents=True, exist_ok=True)
            break
        if arg.startswith("--output_dir="):
            Path(arg.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
            break

    # Launch training
    try:
        result = subprocess.run(cmd, check=False)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        exit_code = 130
    except OSError as error:
        print(f"Error: Could not launch training: {error}", file=sys.stderr)
        exit_code = 1

    # Print summary
    wandb_enabled = "--wandb" in train_args or bool(launcher_config.get("wandb", False))
    print_summary(
        exit_code,
        wandb_enabled,
        os.environ.get("WANDB_MODE", "online"),
        os.environ.get("WANDB_DIR", "/tmp/wandb"),
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
