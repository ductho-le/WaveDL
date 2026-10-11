"""Run a real two-process CPU training/checkpoint smoke check under torchrun."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from accelerate import Accelerator

from wavedl.utils import wait_for_everyone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
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
    main()
