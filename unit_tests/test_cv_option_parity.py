"""Cross-validation must honour the same training options as standard training."""

import logging
import math
import pickle
import random
import sys
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from wavedl.models.base import BaseModel
from wavedl.utils import cross_validation
from wavedl.utils.constraints import PhysicsConstrainedLoss


# ==============================================================================
# SINGLE-FOLD OPTIMIZER SEMANTICS
# ==============================================================================
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
    return {
        "fold": 0,
        "model": model,
        "train_loader": loader,
        "val_loader": loader,
        "criterion": torch.nn.MSELoss(),
        "optimizer": CountingSGD(model.parameters()),
        "scheduler": None,
        "device": torch.device("cpu"),
        "epochs": 1,
        "patience": 5,
        "grad_clip": 0,
        "scaler": StandardScaler().fit(y.numpy()),
        "logger": logging.getLogger("cv-parity"),
    }


def test_accumulated_micro_batches_match_one_large_batch():
    """Full and partial windows both use the window's sample-mean gradient."""
    micro = fold_inputs(samples=5, batch_size=1)
    large = fold_inputs(samples=5, batch_size=3)
    micro_result = cross_validation.train_fold(**micro, grad_accum_steps=3)
    large_result = cross_validation.train_fold(**large)
    for actual, expected in zip(
        micro["model"].parameters(), large["model"].parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    assert micro_result["history"][0]["train_loss"] == pytest.approx(
        large_result["history"][0]["train_loss"]
    )


def test_accumulation_matches_standard_training_accelerator_step():
    """Compare with the real Accelerate loop used by standard training."""
    from accelerate import Accelerator

    from wavedl.train import _finalize_accumulation_window

    inputs = fold_inputs(samples=5, batch_size=2)
    reference = fold_inputs(samples=5, batch_size=2)
    accelerator = Accelerator(
        cpu=True, mixed_precision="no", gradient_accumulation_steps=2
    )
    try:
        model, optimizer, loader = accelerator.prepare(
            reference["model"], reference["optimizer"], reference["train_loader"]
        )
        samples = 0
        for x, y in loader:
            with accelerator.accumulate(model):
                loss = reference["criterion"](model(x), y)
                samples += x.size(0)
                accelerator.backward(loss * (x.size(0) / 2))
                if accelerator.sync_gradients:
                    _finalize_accumulation_window(
                        model, samples, torch.tensor(False), 2, accelerator
                    )
                    samples = 0
                optimizer.step()
                optimizer.zero_grad()
    finally:
        accelerator.free_memory()

    cross_validation.train_fold(**inputs, grad_accum_steps=2)
    for actual, expected in zip(
        inputs["model"].parameters(), reference["model"].parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_clipping_occurs_only_at_update_boundaries(monkeypatch):
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


def test_onecycle_steps_once_per_optimizer_update():
    inputs = fold_inputs(samples=5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        inputs["optimizer"], max_lr=0.01, total_steps=2
    )
    inputs["scheduler"] = scheduler
    cross_validation.train_fold(**inputs, grad_accum_steps=3)
    assert scheduler.last_epoch == 2
    assert inputs["optimizer"].updates == 2


def test_bf16_applies_to_training_and_evaluation():
    inputs = fold_inputs()
    dtypes = []
    inputs["model"].register_forward_hook(
        lambda _model, _inputs, output: dtypes.append(output.dtype)
    )
    result = cross_validation.train_fold(**inputs, precision="bf16")
    assert dtypes and set(dtypes) == {torch.bfloat16}
    assert np.isfinite(result["best_val_loss"])


@pytest.mark.parametrize(
    "device,precision,accumulation,error",
    [
        ("cpu", "fp16", 1, "unsupported"),
        ("cpu", "invalid", 1, "Unknown"),
        ("cpu", "no", 0, "grad_accum_steps"),
    ],
)
def test_invalid_execution_options_fail_before_training(
    device, precision, accumulation, error
):
    inputs = fold_inputs()
    inputs["device"] = torch.device(device)
    with pytest.raises(ValueError, match=error):
        cross_validation.train_fold(
            **inputs, precision=precision, grad_accum_steps=accumulation
        )
    assert inputs["optimizer"].updates == 0


def test_fp16_overflow_skip_does_not_advance_onecycle(monkeypatch):
    """Drive the GradScaler branch on CPU with a scaler that skips one update."""
    inputs = fold_inputs(samples=5)
    inputs["scheduler"] = torch.optim.lr_scheduler.OneCycleLR(
        inputs["optimizer"], max_lr=0.01, total_steps=2
    )
    events = []

    class OverflowOnceScaler:
        scale_value = 8.0
        updates = 0

        def scale(self, loss):
            return loss

        def unscale_(self, _optimizer):
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

    # CPU fp16 is refused in production; bypass only that guard.
    monkeypatch.setattr(cross_validation, "_validate_training_options", lambda *a: None)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda _device: OverflowOnceScaler())
    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: nullcontext())
    cross_validation.train_fold(**inputs, grad_accum_steps=3, precision="fp16")
    assert events == ["unscale", "step", "update"] * 2
    assert inputs["optimizer"].updates == 1
    assert inputs["scheduler"].last_epoch == 1


def test_constraints_train_with_inputs_but_validate_with_base_loss():
    inputs = fold_inputs(samples=4, batch_size=2)
    criterion = cross_validation._build_fold_criterion(
        "mse", None, 1.0, inputs["scaler"], ["y0 - x_mean"], None, [0.5], "mse"
    )
    calls = []
    criterion.register_forward_pre_hook(lambda _m, args: calls.append(len(args)))
    inputs["criterion"] = criterion
    result = cross_validation.train_fold(**inputs)
    # Two training batches pass inputs; validation never evaluates penalties.
    assert calls == [3, 3]
    model = inputs["model"].eval()
    x, y = inputs["val_loader"].dataset.tensors
    with torch.no_grad():
        base = torch.nn.functional.mse_loss(model(x), y).item()
    assert result["best_val_loss"] == pytest.approx(base)


# ==============================================================================
# FOLD-LOCAL CONSTRAINTS
# ==============================================================================
def fold_criterion(targets, **options):
    scaler = StandardScaler().fit(np.asarray(targets, dtype=np.float32))
    return cross_validation._build_fold_criterion(
        "mse",
        None,
        1.0,
        scaler,
        options.get("constraint"),
        options.get("constraint_file"),
        options.get("constraint_weight", [0.1]),
        options.get("constraint_reduction", "mse"),
    )


def test_fold_training_scaler_defines_physical_penalty():
    """The same normalized prediction means different physical values per fold."""
    first = fold_criterion([[2], [6]], constraint=["y0"])
    second = fold_criterion([[20], [60]], constraint=["y0"])
    pred = torch.zeros((2, 1))
    target = torch.zeros_like(pred)
    assert isinstance(first, PhysicsConstrainedLoss)
    assert first(pred, target).item() == pytest.approx(0.1 * 4**2)
    assert second(pred, target).item() == pytest.approx(0.1 * 40**2)
    assert first.base_loss(pred, target).item() == 0


def test_constraint_file_and_unconstrained_criteria(tmp_path):
    path = tmp_path / "constraint.py"
    path.write_text(
        "def constraint(pred, inputs=None):\n"
        "    return pred[:, 0] - inputs.flatten(1).mean(1)\n"
    )
    loss = fold_criterion(
        [[2], [6]], constraint_file=str(path), constraint_weight=[0.5]
    )
    assert loss(
        torch.zeros((2, 1)), torch.zeros((2, 1)), torch.ones((2, 4))
    ).item() == pytest.approx(4.5)
    assert isinstance(fold_criterion([[2], [6]]), torch.nn.MSELoss)


# ==============================================================================
# DETERMINISM
# ==============================================================================
class StochasticModel(BaseModel):
    def __init__(self, in_shape, out_size, **kwargs):
        super().__init__(in_shape, out_size)
        self.linear = torch.nn.Linear(math.prod(in_shape), out_size)
        self.dropout = torch.nn.Dropout(0.25)
        self.register_buffer(
            "offset", torch.tensor(random.random() + np.random.random())
        )

    def forward(self, x):
        return self.linear(self.dropout(x.flatten(1))) + self.offset


def backend_flags():
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def assert_rng_state(before):
    python_state, numpy_state, torch_state = before
    assert random.getstate() == python_state
    current = np.random.get_state()
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def direct_options(output):
    rows = np.arange(12, dtype=np.float32)
    return {
        "X": np.stack([rows / 12, rows**2 / 144], axis=1),
        "y": (rows * 2 + 1).reshape(-1, 1),
        "model_name": "cnn",
        "in_shape": (2,),
        "out_size": 1,
        "folds": 2,
        "epochs": 2,
        "batch_size": 2,
        "workers": 0,
        "seed": 42,
        "device": "cpu",
        "output_dir": str(output),
        "logger": logging.getLogger("cv-parity"),
    }


def test_deterministic_cv_is_repeatable_and_restores_caller_state(
    tmp_path, monkeypatch
):
    from wavedl.models.registry import MODEL_REGISTRY

    monkeypatch.setitem(MODEL_REGISTRY, "cnn", StochasticModel)
    initial_flags = backend_flags()
    observed = []
    original = cross_validation.train_fold

    def train_fold(**kwargs):
        observed.append(backend_flags())
        return original(**kwargs)

    monkeypatch.setattr(cross_validation, "train_fold", train_fold)
    summaries = []
    for run in ("first", "second"):
        # Perturb caller RNGs: deterministic CV must not depend on them.
        random.random()
        np.random.random()
        torch.rand(3)
        before = rng_state()
        summaries.append(
            cross_validation.run_cross_validation(
                **direct_options(tmp_path / run), deterministic=True
            )
        )
        assert backend_flags() == initial_flags
        assert_rng_state(before)
    assert observed == [(True, True, False, True, False, False)] * 4
    assert summaries[0]["fold_results"] == summaries[1]["fold_results"]
    for fold in (1, 2):
        first, second = (
            torch.load(tmp_path / run / f"fold_{fold}" / "model.pth")
            for run in ("first", "second")
        )
        for key in first:
            assert torch.equal(first[key], second[key])


def test_deterministic_failure_restores_caller_state(tmp_path, monkeypatch):
    import wavedl.models

    initial_flags, initial_rng = backend_flags(), rng_state()

    def fail(*_args, **_kwargs):
        assert backend_flags() == (True, True, False, True, False, False)
        torch.rand(3)
        raise RuntimeError("injected fold failure")

    monkeypatch.setattr(wavedl.models, "build_model", fail)
    with pytest.raises(RuntimeError, match="injected fold failure"):
        cross_validation.run_cross_validation(
            **direct_options(tmp_path), deterministic=True
        )
    assert backend_flags() == initial_flags
    assert_rng_state(initial_rng)


def test_worker_seed_derives_from_torch_worker_seed(monkeypatch):
    before = rng_state()
    monkeypatch.setattr(torch, "initial_seed", lambda: 2**32 + 17)
    try:
        cross_validation._seed_cv_worker(0)
        assert random.random() == random.Random(17).random()
        assert np.random.random() == np.random.RandomState(17).random_sample()
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])


# ==============================================================================
# CLI END-TO-END
# ==============================================================================
class TinyCvModel(BaseModel):
    def __init__(self, in_shape, out_size, **kwargs):
        super().__init__(in_shape, out_size)
        self.linear = torch.nn.Linear(math.prod(in_shape), out_size)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x.flatten(1))


@pytest.fixture
def cli_cv(tmp_path, monkeypatch):
    """Run ``wavedl-train --cv 2`` in-process on CPU and record fold arguments."""
    from accelerate import Accelerator

    from wavedl import train
    from wavedl.models.registry import MODEL_REGISTRY

    rows = np.arange(8, dtype=np.float32)
    data = tmp_path / "data.npz"
    np.savez(
        data,
        input_train=np.stack([rows / 8, (rows + 1) / 8, rows**2 / 64], axis=1),
        output_train=np.stack([rows, rows**2], axis=1),
    )
    monkeypatch.setitem(MODEL_REGISTRY, "cnn", TinyCvModel)
    monkeypatch.setattr(train, "Accelerator", lambda **kw: Accelerator(cpu=True, **kw))
    original = cross_validation.train_fold
    observed = []

    def observe(**kwargs):
        observed.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cross_validation, "train_fold", observe)

    def run(*options):
        output = tmp_path / "cv"
        observed.clear()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "wavedl-train",
                "--data_path",
                str(data),
                "--output_dir",
                str(output),
                "--model",
                "cnn",
                "--cv",
                "2",
                "--epochs",
                "2",
                "--workers",
                "0",
                "--precision",
                "no",
                *options,
            ],
        )
        train.main()
        assert len(observed) == 2
        return output, list(observed)

    return run


def test_cli_threads_accumulation_precision_and_onecycle_updates(cli_cv):
    _output, folds = cli_cv(
        "--batch_size", "1", "--grad_accum_steps", "3", "--scheduler", "onecycle"
    )
    for fold in folds:
        assert fold["grad_accum_steps"] == 3
        assert fold["precision"] == "no"
        assert fold["device"] == torch.device("cpu")
        # 4 training samples -> 2 updates per epoch (one partial) x 2 epochs
        assert fold["scheduler"].total_steps == 4


def test_cli_constraints_use_each_folds_training_scaler(cli_cv):
    _output, folds = cli_cv("--constraint", "y1 - y0", "--constraint_weight", "0.5")
    means = []
    for fold in folds:
        criterion = fold["criterion"]
        assert isinstance(criterion, PhysicsConstrainedLoss)
        np.testing.assert_allclose(criterion.output_mean.cpu(), fold["scaler"].mean_)
        np.testing.assert_allclose(criterion.output_std.cpu(), fold["scaler"].scale_)
        means.append(criterion.output_mean.cpu())
    assert not torch.equal(means[0], means[1])


def test_cli_deterministic_reaches_scoped_policy(cli_cv, monkeypatch):
    seeds = []
    original = cross_validation._deterministic_cv

    def observe(seed):
        seeds.append(seed)
        return original(seed)

    monkeypatch.setattr(cross_validation, "_deterministic_cv", observe)
    with original(13):  # Keep the CLI's own global seeding scoped to this test
        cli_cv("--seed", "47")
        assert seeds == []
        cli_cv("--seed", "47", "--deterministic")
        assert seeds == [47]


def test_cli_fold_outputs_load_with_inference_loader(cli_cv):
    from wavedl.test import load_checkpoint

    output, _folds = cli_cv("--grad_accum_steps", "2")
    for fold in (1, 2):
        fold_dir = output / f"fold_{fold}"
        with open(fold_dir / "training_meta.pkl", "rb") as stream:
            meta = pickle.load(stream)
        assert meta["cv_fold"] == fold
        assert 1 <= meta["epoch"] <= meta["epochs_trained"] <= 2
        model, scaler = load_checkpoint(fold_dir, in_shape=(3,), out_size=2)
        assert isinstance(model, TinyCvModel)
        assert scaler.mean_.shape == (2,)
