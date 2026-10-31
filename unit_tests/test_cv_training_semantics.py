"""CV optimizer updates and precision are exercised with real CPU gradients."""

import logging
import math

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from unit_tests import test_cv_option_fidelity as cv_fixtures
from wavedl.utils import cross_validation


cpu_cv = cv_fixtures.cpu_cv


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.updates = 0

    def step(self, closure=None):
        self.updates += 1
        return super().step(closure)


def fold_inputs(samples=6, batch_size=1):
    x = torch.arange(samples, dtype=torch.float32).reshape(-1, 1) / samples
    y = 2 * x + 1
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.1)
        model.bias.zero_()
    optimizer = CountingSGD(model.parameters())
    return {
        "fold": 0,
        "model": model,
        "train_loader": loader,
        "val_loader": loader,
        "criterion": torch.nn.MSELoss(),
        "optimizer": optimizer,
        "scheduler": None,
        "device": torch.device("cpu"),
        "epochs": 1,
        "patience": 5,
        "grad_clip": 0,
        "scaler": StandardScaler().fit(y.numpy()),
        "logger": logging.getLogger("cv-semantics"),
    }


@pytest.mark.parametrize("samples,accumulation", [(6, 1), (6, 3), (5, 3), (2, 4)])
def test_optimizer_steps_flush_partial_accumulation(samples, accumulation):
    inputs = fold_inputs(samples=samples)
    result = cross_validation.train_fold(**inputs, grad_accum_steps=accumulation)
    assert inputs["optimizer"].updates == math.ceil(samples / accumulation)
    assert result["epochs_trained"] == 1


def test_complete_accumulation_matches_large_batch_and_reports_unscaled_loss():
    micro = fold_inputs(batch_size=1)
    large = fold_inputs(batch_size=3)
    micro_result = cross_validation.train_fold(**micro, grad_accum_steps=3)
    large_result = cross_validation.train_fold(**large)
    for actual, expected in zip(
        micro["model"].parameters(), large["model"].parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    assert micro_result["history"][0]["train_loss"] == pytest.approx(
        large_result["history"][0]["train_loss"]
    )


def test_partial_accumulation_matches_accelerator_fixed_denominator():
    inputs = fold_inputs(samples=5)
    reference = fold_inputs(samples=5)
    model, optimizer = reference["model"], reference["optimizer"]
    optimizer.zero_grad()
    for index, (x, y) in enumerate(reference["train_loader"]):
        (reference["criterion"](model(x), y) / 3).backward()
        if index == 2 or index == 4:
            optimizer.step()
            optimizer.zero_grad()
    cross_validation.train_fold(**inputs, grad_accum_steps=3)
    for actual, expected in zip(
        inputs["model"].parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_clipping_occurs_only_at_optimizer_boundaries(monkeypatch):
    inputs = fold_inputs(samples=5)
    inputs["grad_clip"] = 0.2
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def clip(*args, **kwargs):
        calls.append(inputs["optimizer"].updates)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    cross_validation.train_fold(**inputs, grad_accum_steps=3)
    assert calls == [0, 1]


def test_onecycle_steps_only_on_updates_and_partial_flush():
    inputs = fold_inputs(samples=5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        inputs["optimizer"], max_lr=0.01, total_steps=2
    )
    inputs["scheduler"] = scheduler
    cross_validation.train_fold(**inputs, grad_accum_steps=3)
    assert scheduler.last_epoch == 2
    assert inputs["optimizer"].updates == 2


@pytest.mark.parametrize(
    "precision,expected_dtype", [("no", torch.float32), ("bf16", torch.bfloat16)]
)
def test_cpu_precision_is_executed_in_training_and_evaluation(
    precision, expected_dtype
):
    inputs = fold_inputs()
    dtypes = []
    inputs["model"].register_forward_hook(
        lambda _model, _inputs, output: dtypes.append(output.dtype)
    )
    result = cross_validation.train_fold(**inputs, precision=precision)
    assert dtypes and set(dtypes) == {expected_dtype}
    assert np.isfinite(result["best_val_loss"])


@pytest.mark.parametrize(
    "device,precision,error",
    [
        ("cpu", "fp16", "unsupported"),
        ("mps", "bf16", "unsupported"),
        ("cpu", "invalid", "Unknown"),
    ],
)
def test_unsupported_precision_fails_before_training(device, precision, error):
    inputs = fold_inputs()
    inputs["device"] = torch.device(device)
    with pytest.raises(ValueError, match=error):
        cross_validation.train_fold(**inputs, precision=precision)
    assert inputs["optimizer"].updates == 0


def test_cuda_mixed_precision_requires_available_hardware(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        cross_validation._validate_training_options(torch.device("cuda"), 1, "fp16")


def test_cuda_bf16_capability_is_checked_on_selected_device(monkeypatch):
    from contextlib import nullcontext

    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    def select(device):
        selected.append(device)
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "device", select)
    with pytest.raises(ValueError, match="unsupported on this CUDA device"):
        cross_validation._validate_training_options(torch.device("cuda:2"), 1, "bf16")
    assert selected == [torch.device("cuda:2")]


def test_fp16_controlled_overflow_does_not_advance_onecycle(monkeypatch):
    """Probe scaler control flow on CPU; this is not real CUDA acceptance."""
    inputs = fold_inputs(samples=5)
    inputs["scheduler"] = torch.optim.lr_scheduler.OneCycleLR(
        inputs["optimizer"], max_lr=0.01, total_steps=2
    )
    events = []

    class ControlledScaler:
        scale_value = 8
        updates = 0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            events.append("unscale")

        def get_scale(self):
            return self.scale_value

        def step(self, optimizer):
            events.append("step")
            if self.updates > 0:
                optimizer.step()

        def update(self):
            events.append("update")
            if self.updates == 0:
                self.scale_value /= 2
            self.updates += 1

    # Policy rejects CPU fp16 in production. Bypass only that guard to execute
    # the CUDA scaler branch with CPU tensors and a deterministic overflow probe.
    monkeypatch.setattr(cross_validation, "_validate_training_options", lambda *a: None)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda device: ControlledScaler())
    cross_validation.train_fold(**inputs, grad_accum_steps=3, precision="fp16")
    assert events == ["unscale", "step", "update"] * 2
    assert inputs["optimizer"].updates == 1
    assert inputs["scheduler"].last_epoch == 1


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_accumulation_fails_visibly(value):
    with pytest.raises(ValueError, match="grad_accum_steps"):
        cross_validation.train_fold(**fold_inputs(), grad_accum_steps=value)


def test_cli_threads_accumulation_and_precision_to_real_cpu_folds(cpu_cv, monkeypatch):
    original = cross_validation.train_fold
    options = []

    def observe(**kwargs):
        options.append((kwargs["grad_accum_steps"], kwargs["precision"]))
        return original(**kwargs)

    monkeypatch.setattr(cross_validation, "train_fold", observe)
    observed = cpu_cv(
        "--batch_size",
        "1",
        "--grad_accum_steps",
        "3",
        "--precision",
        "no",
        "--scheduler",
        "onecycle",
    )
    assert options == [(3, "no"), (3, "no")]
    assert all(fold["total_steps"] == 2 for fold in observed)
