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
import contextlib
import os
import select
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from wavedl.core.stdio import (
    configure_utf8_stdio as _configure_cli_stdio,
    utf8_stdio_environment as _training_stdio_environment,
)


_GRACE_SECONDS = 30.0
# A supervisor owns the inherited POSIX group (or Windows Job Object). This is
# an internal ownership handshake, not authentication: callers falsely setting
# it to "1" also take responsibility for whole-tree cleanup. Other values are
# standalone mode.
_SUPERVISED_ENV = "WAVEDL_SUPERVISED"


class _ChildExitObserver:
    """Observe our unreaped child; its PID reserves the owned group identity."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.exited = False
        self.queue = None
        if not hasattr(os, "waitid"):
            self.queue = select.kqueue()
            try:
                self.queue.control(
                    [
                        select.kevent(
                            proc.pid,
                            filter=select.KQ_FILTER_PROC,
                            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                            fflags=select.KQ_NOTE_EXIT,
                        )
                    ],
                    0,
                    0,
                )
            except ProcessLookupError:
                # A fast exit can precede kqueue registration. Never use poll:
                # it would release the leader while descendants need signals.
                try:
                    if not self.leader_is_zombie():
                        raise
                except BaseException:
                    self.close()
                    raise
                self.exited = True
            except BaseException:
                self.close()
                raise

    def snapshot(self, timeout: float = 1) -> list[tuple[int, str]]:
        """Bounded Darwin snapshot of this still-reserved process group."""
        probe = subprocess.Popen(
            ["/bin/ps", "-g", str(self.proc.pid), "-o", "pid=,pgid=,stat="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        captured = bytearray()
        deadline = time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as selector:
                os.set_blocking(probe.stdout.fileno(), False)
                selector.register(probe.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("owned process-group snapshot timed out")
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fd, 65537 - len(captured))
                        captured.extend(chunk)
                        if len(captured) > 65536:
                            raise OSError("owned process-group snapshot too large")
                        if not chunk:
                            selector.unregister(key.fileobj)
                if probe.wait(max(0, deadline - time.monotonic())) != 0:
                    raise OSError("owned process-group snapshot failed")
        finally:
            if probe.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    probe.kill()
                probe.wait()
            probe.stdout.close()
        members = []
        for line in captured.decode("ascii").splitlines():
            fields = line.split()
            if len(fields) != 3 or int(fields[1]) != self.proc.pid:
                raise OSError("invalid owned process-group snapshot")
            members.append((int(fields[0]), fields[2]))
        return members

    def leader_is_zombie(self) -> bool:
        return any(
            pid == self.proc.pid and state.startswith("Z")
            for pid, state in self.snapshot()
        )

    def group_is_dead(self) -> bool:
        if sys.platform != "darwin" or not self.exited:
            return False
        members = self.snapshot()
        return any(pid == self.proc.pid for pid, _ in members) and all(
            state.startswith("Z") for _, state in members
        )

    def confirm_group_is_dead(self) -> bool:
        """Resolve Darwin's exit/EPERM race without releasing group identity.

        killpg can reject a dying group before NOTE_EXIT or ps reports zombies.
        Only a nonempty zombie-only snapshot containing our unreaped leader is
        evidence of completion; an inaccessible live member must still fail.
        """
        if sys.platform != "darwin":
            return False
        deadline = time.monotonic() + 0.1
        while True:
            self.observe()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                members = self.snapshot(timeout=remaining)
            except (TimeoutError, subprocess.TimeoutExpired):
                return False
            leader_is_zombie = any(
                pid == self.proc.pid and state.startswith("Z") for pid, state in members
            )
            if leader_is_zombie:
                # The same owned-leader evidence is authoritative when kqueue
                # delivery lags, just as during fast-exit registration fallback.
                self.exited = True
                if all(state.startswith("Z") for _, state in members):
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))

    def observe(self) -> bool:
        if not self.exited:
            if self.queue is None:
                status = os.waitid(
                    os.P_PID, self.proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
                self.exited = status is not None and status.si_pid == self.proc.pid
            else:
                for event in self.queue.control(None, 1, 0):
                    if event.flags & select.KQ_EV_ERROR:
                        raise OSError(event.data, "child exit observation failed")
                    if event.fflags & select.KQ_NOTE_EXIT:
                        self.exited = True
        return self.exited

    def close(self) -> None:
        if self.queue is not None:
            self.queue.close()


def _signal_child(proc, signum: int, process_group: int | None = None) -> None:
    """Signal without waiting; owned groups require a still-unreaped leader."""
    try:
        if process_group is not None:
            os.killpg(process_group, signum)
        elif os.name == "nt" and signum == signal.SIGINT:
            # No console-group guarantee here: the Windows supervisor owns
            # graceful control events and whole-tree cleanup through its Job.
            proc.terminate()
        else:
            proc.send_signal(signum)
    except ProcessLookupError:
        pass


@dataclass
class _SignalForwarder:
    process: subprocess.Popen | None
    process_group: int | None = None
    received_signal: int | None = None
    previous_handlers: dict = field(default_factory=dict)

    def forward(self, signum, _frame) -> None:
        # Repeated delivery must not reset the grace deadline or interrupt reap.
        if self.received_signal is not None:
            return
        self.received_signal = signum
        if self.process is not None:
            try:
                _signal_child(self.process, signum, self.process_group)
            except PermissionError:
                if self.process_group is None:
                    raise
                # Darwin rejects zombie-only groups. Normal control flow must
                # verify this with the still-owned leader before suppressing it.

    def restore(self) -> None:
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)

    def disarm(self) -> None:
        # Clear process first: a pending Python signal cannot target a group
        # after its leader's PID is released by wait(). Keep recording signals.
        self.process = None
        self.process_group = None


def _install_sigterm_handler(
    proc: subprocess.Popen | None, process_group: int | None = None
) -> _SignalForwarder:
    """Install nonblocking SIGTERM/SIGINT forwarding; caller must restore it."""
    forwarder = _SignalForwarder(proc, process_group)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            forwarder.previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forwarder.forward)
    except BaseException:
        forwarder.restore()
        raise
    return forwarder


def _terminate_child(
    proc: subprocess.Popen,
    timeout: float = _GRACE_SECONDS,
    *,
    signum: int = signal.SIGTERM,
    process_group: int | None = None,
    forwarder: _SignalForwarder | None = None,
    observer: _ChildExitObserver | None = None,
) -> None:
    """Gracefully stop, escalate, and reap, only from normal control flow.

    Standalone POSIX launches own a child group. Under a supervisor children
    inherit its group instead: the supervisor, not this helper, kills the tree.
    Windows standalone fallback terminates/reaps the child only; whole-tree
    ownership requires the Windows supervisor's Job Object.
    """
    if process_group is not None:
        owned_observer = observer or _ChildExitObserver(proc)
        try:
            try:
                _signal_child(proc, signum, process_group)
            except PermissionError:
                if not owned_observer.confirm_group_is_dead():
                    raise
            deadline = time.monotonic() + max(0.0, timeout)
            while not owned_observer.observe():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))
            # A dead leader cannot prove that descendants are gone, even when
            # they closed their inherited pipes. Its unreaped PID remains ours
            # throughout grace and the last group signal.
            while time.monotonic() < deadline:
                if owned_observer.group_is_dead():
                    break
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            try:
                _signal_child(proc, signal.SIGKILL, process_group)
            except PermissionError:
                if not owned_observer.confirm_group_is_dead():
                    raise
            if forwarder is not None:
                forwarder.disarm()
            proc.wait()
        finally:
            if observer is None:
                owned_observer.close()
        return

    _signal_child(proc, signum)
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        child_done = proc.poll() is not None
        if child_done:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # poll/wait and monotonic deadlines belong here, never in the handler.
        time.sleep(min(0.05, remaining))
    if proc.poll() is None:
        proc.kill()
    proc.wait()


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
        "--cpu", action="store_true", help="Run training processes on CPU"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs, or CPU processes with --cpu (default: auto-detect)",
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
    parser.add_argument(
        "--launch_id", help="Fresh shared UUID for standalone multi-node execution"
    )

    # Parse known args, pass rest to wavedl.train
    args, remaining = parser.parse_known_args()
    args.original_argv = tuple(sys.argv[1:])
    if args.num_gpus is not None and args.num_gpus < 1:
        parser.error("--num_gpus must be positive")
    if args.num_machines < 1 or not 0 <= args.machine_rank < args.num_machines:
        parser.error(
            "--num_machines must be positive and --machine_rank must be within its range."
        )
    if args.num_machines > 1 and not args.main_process_ip:
        parser.error("--main_process_ip is required for multi-node training.")
    if args.main_process_port is not None and not 1 <= args.main_process_port <= 65535:
        parser.error("--main_process_port must be between 1 and 65535.")
    if args.cpu and args.num_gpus is not None and args.num_gpus > 64:
        parser.error("--cpu supports at most 64 processes")
    if args.cpu and (args.num_machines != 1 or args.machine_rank != 0):
        parser.error("--cpu supports only one machine with rank 0")
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
    _configure_cli_stdio()
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
    if args.cpu:
        num_gpus = args.num_gpus or 1
        print(f"Using CPU training ({num_gpus} process(es))")
    elif args.num_gpus is not None:
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
    if cv_folds and cv_folds > 0:
        if args.num_machines > 1:
            print("Error: Cross-validation requires one machine.", file=sys.stderr)
            return 2
        num_gpus = 1

    from wavedl.launch_runtime import prepare_runtime

    train_args, child_environment = prepare_runtime(
        train_args,
        args,
        python=sys.executable,
        environment=_training_stdio_environment(dict(os.environ)),
    )

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

    if args.cpu:
        cmd.append("--cpu")

    # Explicitly set multi_gpu to suppress accelerate auto-detection warning
    if num_gpus * args.num_machines > 1:
        cmd.append("--multi_gpu")

    # Add multi-node networking args if specified (required for some clusters)
    if args.main_process_ip:
        cmd.append(f"--main_process_ip={args.main_process_ip}")
    if args.main_process_port:
        cmd.append(f"--main_process_port={args.main_process_port}")

    cmd += ["-m", "wavedl.train"] + train_args

    if not args.cpu and num_gpus > 1:
        # Accelerate's binary torchrun workers create their own POSIX sessions.
        # Keep its configuration/environment handling, but use callable workers
        # so every rank remains inside the launcher's cancellation group.
        cmd[2] = "wavedl.distributed_launch"

    if args.cpu and num_gpus > 1:
        # Accelerate's --cpu launcher executes only one process. Native torchrun
        # supplies the real Gloo rank topology while retaining our pre-spawn
        # runtime identity and the supervising launcher's process tree.
        child_environment["ACCELERATE_USE_CPU"] = "true"
        child_environment["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "--local_addr=127.0.0.1",
            "--run-path",
            "--start-method=spawn",
            str(Path(__file__).with_name("train.py").resolve()),
            *train_args,
        ]

    # Create output directory if specified
    for i, arg in enumerate(train_args):
        if arg == "--output_dir" and i + 1 < len(train_args):
            Path(train_args[i + 1]).mkdir(parents=True, exist_ok=True)
            break
        if arg.startswith("--output_dir="):
            Path(arg.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
            break

    # Install before spawning so a signal during Popen is not lost. A standalone
    # POSIX child owns a session; supervised children MUST inherit the owner's
    # group so no training worker escapes the supervisor's cancellation scope.
    forwarder = _install_sigterm_handler(None)
    proc = None
    observer = None
    try:
        owns_group = os.name == "posix" and os.environ.get(_SUPERVISED_ENV) != "1"
        proc = subprocess.Popen(
            cmd, start_new_session=owns_group, env=child_environment
        )
        forwarder.process_group = proc.pid if owns_group else None
        forwarder.process = proc
        if owns_group:
            observer = _ChildExitObserver(proc)
        while forwarder.received_signal is None:
            if observer is not None:
                if observer.observe():
                    # Natural leader exit also leaves descendants behind. Close
                    # their group before releasing its identity; preserve status.
                    _terminate_child(
                        proc,
                        timeout=min(_GRACE_SECONDS, 0.2),
                        process_group=forwarder.process_group,
                        forwarder=forwarder,
                        observer=observer,
                    )
                    exit_code = proc.returncode
                    break
                time.sleep(0.05)
                continue
            try:
                exit_code = proc.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                continue
        if forwarder.received_signal is not None:
            _terminate_child(
                proc,
                timeout=_GRACE_SECONDS,
                signum=forwarder.received_signal,
                process_group=forwarder.process_group,
                forwarder=forwarder,
                observer=observer,
            )
            exit_code = 128 + forwarder.received_signal
        elif exit_code < 0:
            exit_code = 128 - exit_code
    except BaseException as error:
        if proc is not None:
            # Observation itself may have failed. We still own the unreaped
            # leader: immediately finish group signalling without retrying the
            # broken observer, then disarm before the only bounded reap. Never
            # mask the original failure with a cleanup/probe failure.
            try:
                _signal_child(proc, signal.SIGKILL, forwarder.process_group)
            except Exception as cleanup_error:
                error.add_note(f"launcher signal cleanup failed: {cleanup_error}")
            forwarder.disarm()
            try:
                proc.wait(timeout=1)
            except Exception as cleanup_error:
                error.add_note(f"launcher reap cleanup failed: {cleanup_error}")
        raise
    finally:
        forwarder.disarm()
        forwarder.restore()
        if observer is not None:
            observer.close()
    if forwarder.received_signal is not None:
        exit_code = 128 + forwarder.received_signal

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
