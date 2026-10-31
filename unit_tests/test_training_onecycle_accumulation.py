"""Real CPU normal-training OneCycle budgets, including native DDP sharding."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator

from wavedl.core.training_cli import parse_training_args
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.supervision.base import ProcessSpec


@pytest.mark.integration
@pytest.mark.parametrize("ranks,accumulation", [(1, 1), (1, 2), (1, 8), (2, 2)])
def test_normal_training_onecycle_counts_successful_optimizer_updates(
    tmp_path, ranks, accumulation
):
    """Five microbatches per rank yield ceil(5 / accumulation) updates/epoch."""
    if ranks > 1 and (not dist.is_available() or not dist.is_gloo_available()):
        pytest.skip("Native CPU distributed acceptance requires Gloo")
    source = Path(__file__).resolve().parents[1] / "src"
    data = tmp_path / "data.npz"
    generator = np.random.default_rng(8)
    samples = 50 * ranks  # Default 80% training split: five batches per rank.
    np.savez(
        data,
        X=generator.normal(size=(samples, 32)).astype("float32"),
        y=generator.normal(size=(samples, 1)).astype("float32"),
    )
    output = tmp_path / "run"
    command = [sys.executable]
    if ranks > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={ranks}",
                "--local_addr=127.0.0.1",
                "--start-method=spawn",
                "--run-path",
            ]
        )
    command.extend(
        [
            str(source / "wavedl/train.py"),
            "--data_path",
            str(data),
            "--output_dir",
            str(output),
            "--model",
            "cnn",
            "--epochs",
            "3",
            "--workers",
            "0",
            "--batch_size",
            "8",
            "--precision",
            "no",
            "--scheduler",
            "onecycle",
            "--grad_accum_steps",
            str(accumulation),
            "--save_every",
            "1",
        ]
    )
    environment = {
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": str(source)
        + (
            os.pathsep + os.environ["PYTHONPATH"]
            if os.environ.get("PYTHONPATH")
            else ""
        ),
    }
    result = run_command(
        ProcessSpec(tuple(command), tmp_path, environment),
        WorkBudget(timeout_seconds=60, max_output_bytes=128 * 1024),
    )
    diagnostics = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert not result.timed_out, diagnostics
    assert result.returncode == 0, diagnostics
    assert (output / "training_complete.flag").is_file()

    updates_per_epoch = math.ceil(5 / accumulation)
    total_updates = 3 * updates_per_epoch
    checkpoint = output / "epoch_3_checkpoint"
    state = torch.load(checkpoint / "scheduler.bin", weights_only=False)
    assert state["total_steps"] == total_updates
    assert state["last_epoch"] == total_updates
    optimizer_state = torch.load(checkpoint / "optimizer.bin", weights_only=False)
    assert all(
        int(parameter["step"]) == total_updates
        for parameter in optimizer_state["state"].values()
    )

    # Compare the saved LR to the exact PyTorch cycle, not an assumed final LR:
    # OneCycle's post-final-step LR follows its documented implementation.
    reference_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))], lr=1e-3
    )
    reference_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        reference_optimizer, max_lr=1e-3, total_steps=total_updates
    )
    for _ in range(total_updates):
        reference_optimizer.step()
        reference_scheduler.step()
    assert state["_last_lr"][0] == pytest.approx(reference_scheduler.get_last_lr()[0])
    assert optimizer_state["param_groups"][0]["lr"] == pytest.approx(
        reference_scheduler.get_last_lr()[0]
    )


def _cpu_training(tmp_path, monkeypatch, epochs=3):
    from wavedl import train

    generator = np.random.default_rng(8)
    data = tmp_path / "data.npz"
    np.savez(
        data,
        X=generator.normal(size=(50, 32)).astype("float32"),
        y=generator.normal(size=(50, 1)).astype("float32"),
    )
    args, parser = parse_training_args(
        [
            "--data_path",
            str(data),
            "--output_dir",
            str(tmp_path / "run"),
            "--model",
            "cnn",
            "--epochs",
            str(epochs),
            "--workers",
            "0",
            "--batch_size",
            "8",
            "--precision",
            "no",
            "--scheduler",
            "onecycle",
            "--grad_accum_steps",
            "2",
            "--save_every",
            "1",
        ]
    )
    monkeypatch.setattr(train, "parse_args", lambda: (args, parser))
    monkeypatch.setattr(train, "Accelerator", lambda **kw: Accelerator(cpu=True, **kw))
    torch.set_num_threads(1)
    return train, args


@pytest.mark.parametrize("checkpoint_kind", ["emergency", "normal"])
def test_onecycle_resume_retains_optimizer_update_position(
    tmp_path, monkeypatch, checkpoint_kind
):
    """A real epoch-one cancellation resumes the unchanged nine-update cycle."""
    train, args = _cpu_training(tmp_path, monkeypatch)
    calls = 0

    def accelerator_factory(**kwargs):
        accelerator = Accelerator(cpu=True, **kwargs)
        backward = accelerator.backward

        def cancel_after_epoch_one(loss, **backward_kwargs):
            nonlocal calls
            calls += 1
            if calls == 6:
                raise KeyboardInterrupt("after epoch one")
            return backward(loss, **backward_kwargs)

        monkeypatch.setattr(accelerator, "backward", cancel_after_epoch_one)
        return accelerator

    monkeypatch.setattr(train, "Accelerator", accelerator_factory)
    with pytest.raises(KeyboardInterrupt, match="after epoch one"):
        train.main()
    interrupted = Path(args.output_dir) / "interrupted_checkpoint"
    state = torch.load(interrupted / "scheduler.pt", weights_only=False)
    assert state["total_steps"] == 9 and state["last_epoch"] == 3
    resume_checkpoint = (
        interrupted
        if checkpoint_kind == "emergency"
        else Path(args.output_dir) / "epoch_1_checkpoint"
    )
    if checkpoint_kind == "normal":
        state = torch.load(resume_checkpoint / "scheduler.bin", weights_only=False)
        assert state["total_steps"] == 9 and state["last_epoch"] == 3
    args.resume = str(resume_checkpoint)
    monkeypatch.setattr(train, "Accelerator", lambda **kw: Accelerator(cpu=True, **kw))
    train.main()
    checkpoint = Path(args.output_dir) / "epoch_3_checkpoint"
    state = torch.load(checkpoint / "scheduler.bin", weights_only=False)
    optimizer = torch.load(checkpoint / "optimizer.bin", weights_only=False)
    assert state["total_steps"] == state["last_epoch"] == 9
    assert all(int(parameter["step"]) == 9 for parameter in optimizer["state"].values())
    reference = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        reference, max_lr=1e-3, total_steps=9
    )
    for _ in range(9):
        reference.step()
        scheduler.step()
    assert state["_last_lr"][0] == pytest.approx(scheduler.get_last_lr()[0])


def test_onecycle_controlled_scaler_skip_does_not_advance(tmp_path, monkeypatch):
    """CPU scaler control-flow probe, not real CUDA mixed-precision acceptance."""
    train, args = _cpu_training(tmp_path, monkeypatch, epochs=1)

    class ControlledScaler:
        updates = 0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer, closure=None):
            if self.updates:
                optimizer.step(closure)

        def update(self):
            self.updates += 1

        def state_dict(self):
            return {"updates": self.updates}

    scaler = ControlledScaler()

    def accelerator_factory(**kwargs):
        accelerator = Accelerator(cpu=True, **kwargs)
        accelerator.scaler = scaler
        return accelerator

    monkeypatch.setattr(train, "Accelerator", accelerator_factory)
    train.main()
    checkpoint = Path(args.output_dir) / "epoch_1_checkpoint"
    state = torch.load(checkpoint / "scheduler.bin", weights_only=False)
    optimizer = torch.load(checkpoint / "optimizer.bin", weights_only=False)
    assert scaler.updates == 3
    assert state["total_steps"] == 3 and state["last_epoch"] == 2
    assert all(int(parameter["step"]) == 2 for parameter in optimizer["state"].values())
