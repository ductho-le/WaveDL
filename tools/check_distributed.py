"""Launch a local two-process CPU training/checkpoint smoke check.

Run directly with --work-dir pointing to a new directory. The launcher uses a
static IPv4 rendezvous, avoiding macOS hostname/IPv6 lookup hangs in torchrun.
An existing torchrun launch is also supported; its rendezvous is caller-owned.
"""

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
from accelerate import Accelerator

from wavedl.utils import wait_for_everyone


def launch_local(work_dir: Path) -> int:
    """Configure rendezvous before torchrun starts, then propagate its exit code."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    env = os.environ.copy()
    env["ACCELERATE_USE_CPU"] = "true"
    env.setdefault("OMP_NUM_THREADS", "1")
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--rdzv-backend=static",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            str(Path(__file__).resolve()),
            "--work-dir",
            str(work_dir),
        ],
        env=env,
        check=False,
    ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if "RANK" not in os.environ:
        return launch_local(args.work_dir)
    accelerator = Accelerator(cpu=True)
    if accelerator.num_processes < 2:
        raise RuntimeError("Launch with torchrun and at least two processes")
    if accelerator.is_main_process:
        args.work_dir.mkdir(parents=True, exist_ok=False)
        rng = np.random.default_rng(42)
        np.savez(
            args.work_dir / "data.npz",
            X=rng.normal(size=(23, 32)).astype("float32"),
            Y=rng.normal(size=(23, 2)).astype("float32"),
        )
    wait_for_everyone(accelerator)
    os.environ["ACCELERATE_USE_CPU"] = "true"
    sys.argv = [
        "train",
        "--data_path",
        str(args.work_dir / "data.npz"),
        "--output_dir",
        str(args.work_dir / "run"),
        "--model",
        "cnn",
        "--epochs",
        "2",
        "--batch_size",
        "4",
        "--workers",
        "0",
        "--precision",
        "no",
        "--grad_accum_steps",
        "2",
        "--save_every",
        "1",
    ]
    from wavedl.train import main as train

    train()
    assert (args.work_dir / "run/epoch_2_checkpoint/training_meta.pkl").is_file()


if __name__ == "__main__":
    sys.exit(main())
