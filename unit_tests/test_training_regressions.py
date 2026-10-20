"""Regression coverage for dense metrics and cross-validation configuration."""

import logging
import sys

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from wavedl.utils.metrics import StreamingFieldMetrics
from wavedl.utils.targets import flatten_targets_by_channel


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_streaming_field_metrics_match_full_metrics_with_large_offsets(offset):
    rng = np.random.default_rng(19)
    target = offset + rng.normal(size=(7, 2, 3, 5))
    prediction = target + rng.normal(scale=0.2, size=target.shape)
    scales = np.array([2.0, 7.0]).reshape(2, 1, 1)
    tracker = StreamingFieldMetrics(2, scales)
    for start in range(0, len(target), 3):
        tracker.update(target[start : start + 3], prediction[start : start + 3])
    r2, pcc, mae = tracker.compute()

    true_flat = flatten_targets_by_channel(target)
    pred_flat = flatten_targets_by_channel(prediction)
    expected_pcc = np.mean(
        [pearsonr(true_flat[:, c], pred_flat[:, c]).statistic for c in range(2)]
    )
    assert r2 == pytest.approx(r2_score(true_flat, pred_flat), abs=1e-8)
    assert pcc == pytest.approx(expected_pcc, abs=1e-8)
    np.testing.assert_allclose(
        mae, np.abs((prediction - target) * scales).mean((0, 2, 3))
    )


def test_streaming_field_metrics_constant_target_small_error_is_not_perfect():
    target = np.ones((3, 1, 2, 2))
    tracker = StreamingFieldMetrics(1, np.ones((1, 1, 1)))
    tracker.update(target, target + 1e-8)
    r2, pcc, _mae = tracker.compute()
    assert r2 == 0.0
    assert pcc == 0.0


def test_streaming_single_pixel_has_undefined_r2_and_correlation():
    tracker = StreamingFieldMetrics(1, np.ones((1, 1, 1)))
    tracker.update(np.ones((1, 1, 1, 1)), np.zeros((1, 1, 1, 1)))
    r2, pcc, mae = tracker.compute()
    assert np.isnan(r2)
    assert np.isnan(pcc)
    np.testing.assert_array_equal(mae, [1.0])


class _SmallFieldModel(torch.nn.Module):
    supports_spatial_output = True
    requires_spatial_output = True

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, 2, 1, 1))

    def forward(self, values):
        return self.bias.expand(len(values), 2, 3, 4)

    def get_optimizer_groups(self, _lr, _weight_decay):
        return self.parameters()


@pytest.mark.parametrize(
    ("loss_name", "loss_weights", "huber_delta", "expected_unit_loss"),
    [("weighted_mse", [1.0, 4.0], 1.0, 2.5), ("huber", None, 0.25, 0.21875)],
)
def test_dense_cross_validation_preserves_loss_configuration(
    tmp_path, monkeypatch, loss_name, loss_weights, huber_delta, expected_unit_loss
):
    from wavedl import models
    from wavedl.utils import cross_validation

    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(6, 4, 5)).astype(np.float32)
    targets = rng.normal(size=(6, 2, 3, 4)).astype(np.float32)
    forwarded_pretrained = []

    def build_model(_name, **kwargs):
        forwarded_pretrained.append(kwargs["pretrained"])
        assert kwargs["out_shape"] == (2, 3, 4)
        return _SmallFieldModel()

    monkeypatch.setattr(models, "get_model", lambda _name: _SmallFieldModel)
    monkeypatch.setattr(models, "build_model", build_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    original_train_fold = cross_validation.train_fold
    losses_checked = []

    def train_fold(**kwargs):
        ones = torch.ones((2, 2, 3, 4))
        actual_loss = kwargs["criterion"](ones, torch.zeros_like(ones)).item()
        losses_checked.append(actual_loss)
        return original_train_fold(**kwargs)

    monkeypatch.setattr(cross_validation, "train_fold", train_fold)
    summary = cross_validation.run_cross_validation(
        X=inputs,
        y=targets,
        model_name="test_field",
        in_shape=(4, 5),
        out_size=2,
        folds=2,
        batch_size=2,
        epochs=1,
        workers=0,
        loss_name=loss_name,
        loss_weights=loss_weights,
        huber_delta=huber_delta,
        pretrained=True,
        output_dir=str(tmp_path),
        logger=logging.getLogger(__name__),
    )
    assert losses_checked == pytest.approx([expected_unit_loss] * 2)
    assert forwarded_pretrained == [True, True]
    assert np.isfinite(summary["val_loss_mean"])
    assert summary["config"]["loss_weights"] == loss_weights
    assert summary["config"]["huber_delta"] == huber_delta


def test_inprocess_hpo_uses_same_pretrained_default_as_cli(tmp_path, monkeypatch):
    from wavedl import train

    rng = np.random.default_rng(12)
    path = tmp_path / "dense.npz"
    np.savez(
        path,
        input_train=rng.normal(size=(6, 16, 16)).astype(np.float32),
        output_train=rng.normal(size=(6, 2, 3, 4)).astype(np.float32),
    )
    initializations = []

    def build_model(_name, _shape, _channels, _scaler, **kwargs):
        initializations.append(kwargs["pretrained"])
        return _SmallFieldModel(), (2, 3, 4)

    monkeypatch.setattr(train, "_build_model_for_targets", build_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = train.train_single_trial(
        str(path), "unet_field", epochs=1, batch_size=2, workers=0, precision="no"
    )
    assert initializations == [False]
    assert np.isfinite(result["best_val_loss"])


@pytest.mark.parametrize(
    "unsupported_args",
    [
        ["--constraint", "y0 > 0"],
        ["--constraint_file", "constraint.py"],
        ["--grad_accum_steps", "2"],
        ["--selection_metric", "mse"],
    ],
)
def test_cv_rejects_unsupported_training_settings(
    monkeypatch, capsys, unsupported_args
):
    from wavedl.train import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["wavedl-train", "--cv", "2", "--data_path", "unused.npz", *unsupported_args],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Cross-validation does not support" in capsys.readouterr().err


class _TinyVectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 0.2)

    def forward(self, inputs):
        return self.linear(inputs)

    def get_optimizer_groups(self, _lr, _weight_decay):
        return self.parameters()

    def parameter_summary(self):
        return {"trainable_parameters": 1, "total_mb": 0.0}


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    """Exercise the production loop with real optimizer/checkpoint machinery."""
    import pickle

    from accelerate import Accelerator
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    from wavedl import train

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = x * 2
    scaler = StandardScaler().fit(y.numpy())
    model_instances = []

    def prepare_data(args, *_args, **_kwargs):
        with open(tmp_path / "scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)
        loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size)
        return loader, loader, scaler, (1,), 1

    def build_model(*_args, **_kwargs):
        model = _TinyVectorModel()
        model_instances.append(model)
        return model, (1,)

    monkeypatch.setattr(train, "prepare_data", prepare_data)
    monkeypatch.setattr(train, "_build_model_for_targets", build_model)
    monkeypatch.setattr(
        train,
        "create_training_curves",
        lambda *a, **k: __import__("matplotlib").pyplot.figure(),
    )

    def run(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--output_dir",
                str(tmp_path),
                "--workers",
                "0",
                "--precision",
                "no",
                "--epochs",
                "2",
                "--batch_size",
                "2",
                "--optimizer",
                "sgd",
                "--momentum",
                "0",
                "--weight_decay",
                "0",
                "--lr",
                "0.001",
                "--grad_clip",
                "0",
                "--save_every",
                "1",
                *extra,
            ],
        )
        args, _parser = train.parse_args()
        accelerator = Accelerator(
            cpu=True,
            mixed_precision="no",
            gradient_accumulation_steps=args.grad_accum_steps,
        )
        try:
            train._run_training(args, accelerator)
        finally:
            accelerator.free_memory()
        return model_instances[-1] if model_instances else None

    return run, x, y


def test_accumulation_partial_window_matches_sample_mean(tiny_training):
    run, x, y = tiny_training
    actual = run("--epochs", "1", "--grad_accum_steps", "4")
    expected = _TinyVectorModel()
    optimizer = torch.optim.SGD(expected.parameters(), lr=0.001)
    torch.nn.functional.mse_loss(expected(x), y).backward()
    optimizer.step()
    torch.testing.assert_close(actual.linear.weight, expected.linear.weight)


def test_epoch_scheduler_checkpoint_contains_next_epoch_state(tiny_training, tmp_path):
    run, _, _ = tiny_training
    run(
        "--epochs",
        "2",
        "--scheduler",
        "step",
        "--step_size",
        "1",
        "--scheduler_factor",
        "0.5",
    )
    state = torch.load(
        tmp_path / "epoch_2_checkpoint" / "custom_checkpoint_0.pkl", weights_only=False
    )
    assert state["last_epoch"] == 2
    assert state["_last_lr"] == pytest.approx([0.00025])


def test_onecycle_counts_optimizer_updates_with_accumulation(tiny_training, tmp_path):
    run, _, _ = tiny_training
    run("--epochs", "2", "--grad_accum_steps", "2", "--scheduler", "onecycle")
    state = torch.load(
        tmp_path / "epoch_2_checkpoint" / "scheduler.bin", weights_only=True
    )
    assert state["total_steps"] == 4
    assert state["last_epoch"] == 4


def test_fresh_archives_old_checkpoint_and_completion_state(tmp_path):
    from wavedl.train import _archive_previous_run

    (tmp_path / "epoch_100_checkpoint").mkdir()
    (tmp_path / "training_complete.flag").write_text("old completion")
    (tmp_path / "interrupted_checkpoint").mkdir()
    (tmp_path / "scaler.pkl").write_bytes(b"old scaler")
    (tmp_path / "user_notes.txt").write_text("keep")
    archive = _archive_previous_run(str(tmp_path))
    assert not (tmp_path / "training_complete.flag").exists()
    assert not (tmp_path / "epoch_100_checkpoint").exists()
    assert (archive / "epoch_100_checkpoint").is_dir()
    assert (archive / "scaler.pkl").read_bytes() == b"old scaler"
    assert (tmp_path / "scaler.pkl").exists()
    assert (tmp_path / "user_notes.txt").read_text() == "keep"


def test_nonfinite_training_fails_without_completion_flag(
    tiny_training, tmp_path, monkeypatch
):
    run, _, _ = tiny_training
    from wavedl import train

    class BadLoss(torch.nn.Module):
        def forward(self, pred, target):
            return pred.mean() * float("nan")

    monkeypatch.setattr(train, "get_loss", lambda **_: BadLoss())
    with pytest.raises(FloatingPointError, match="non-finite"):
        run()
    assert not (tmp_path / "training_complete.flag").exists()


def test_training_interrupt_propagates_and_saves_recovery(
    tiny_training, tmp_path, monkeypatch
):
    run, _, _ = tiny_training
    from wavedl import train

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "_flag_nonfinite_loss", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert (tmp_path / "interrupted_checkpoint" / "model_weights.pth").is_file()
    assert not (tmp_path / "training_complete.flag").exists()


def test_completed_run_returns_before_loading_data(
    tiny_training, tmp_path, monkeypatch
):
    run, _, _ = tiny_training
    from wavedl import train

    (tmp_path / "training_complete.flag").write_text("done")
    monkeypatch.setattr(
        train,
        "prepare_data",
        lambda *a, **k: pytest.fail("completed runs must not reload data"),
    )
    run()


def test_resuming_epoch_checkpoint_matches_uninterrupted_weights(
    tiny_training, tmp_path
):
    run, _, _ = tiny_training
    full = run("--scheduler", "step", "--step_size", "1")
    expected = full.linear.weight.detach().clone()
    resumed = run(
        "--scheduler",
        "step",
        "--step_size",
        "1",
        "--resume",
        str(tmp_path / "epoch_1_checkpoint"),
    )
    torch.testing.assert_close(resumed.linear.weight, expected, rtol=0, atol=0)


def test_training_cleanup_runs_when_setup_fails(monkeypatch):
    from unittest.mock import MagicMock

    from wavedl import train

    accelerator = MagicMock()
    monkeypatch.setattr(sys, "argv", ["train"])
    monkeypatch.setattr(train, "Accelerator", lambda **kwargs: accelerator)

    def fail(*args):
        raise RuntimeError("setup failure")

    monkeypatch.setattr(train, "_run_training", fail)
    with pytest.raises(RuntimeError, match="setup failure"):
        train.main()
    accelerator.end_training.assert_called_once()
    accelerator.free_memory.assert_called_once()


@pytest.mark.parametrize("loss_name", ["mae", "huber"])
def test_hpo_uses_comparable_mse_for_different_training_losses(
    tmp_path, monkeypatch, loss_name
):
    from wavedl import train

    rng = np.random.default_rng(12)
    path = tmp_path / "dense.npz"
    np.savez(
        path,
        input_train=rng.normal(size=(6, 16, 16)).astype(np.float32),
        output_train=rng.normal(size=(6, 2, 3, 4)).astype(np.float32),
    )
    monkeypatch.setattr(
        train,
        "_build_model_for_targets",
        lambda *a, **k: (_SmallFieldModel(), (2, 3, 4)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reference = train.train_single_trial(
        str(path), lr=0, epochs=1, batch_size=2, loss_name="mse", selection_metric="mse"
    )
    candidate = train.train_single_trial(
        str(path),
        lr=0,
        epochs=1,
        batch_size=2,
        loss_name=loss_name,
        selection_metric="mse",
    )
    assert candidate["best_val_loss"] == pytest.approx(reference["best_val_loss"])


def test_failed_onnx_validation_preserves_previous_export(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    from wavedl import test

    destination = tmp_path / "production.onnx"
    destination.write_bytes(b"previous validated artifact")
    monkeypatch.setattr(test, "_validate_onnx_export", lambda *a, **k: False)
    assert not test.export_to_onnx(
        _TinyVectorModel(), torch.ones(1, 1), str(destination)
    )
    assert destination.read_bytes() == b"previous validated artifact"
    assert not list(tmp_path.glob(".wavedl-export-*"))


def test_inference_rejects_nonfinite_predictions():
    from wavedl.test import run_inference

    model = _TinyVectorModel()
    model.linear.weight.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        run_inference(model, torch.ones(2, 1), device=torch.device("cpu"))


def test_failed_export_cli_has_nonzero_outcome(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    from wavedl import test

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--data_path",
            "unused.npz",
            "--checkpoint",
            str(tmp_path),
            "--out_size",
            "1",
            "--export",
            "onnx",
        ],
    )
    monkeypatch.setattr(
        test,
        "load_data_for_inference",
        lambda *a, **k: (torch.ones(2, 1, 1), torch.ones(2, 1)),
    )
    monkeypatch.setattr(
        test,
        "load_checkpoint",
        lambda *a, **k: (_TinyVectorModel(), StandardScaler().fit([[0.0], [1.0]])),
    )
    monkeypatch.setattr(test, "export_to_onnx", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="ONNX export failed"):
        test.main()


def test_launcher_cv_and_precision_use_single_consistent_process(monkeypatch):
    from types import SimpleNamespace

    from wavedl import launcher

    captured = []
    monkeypatch.setattr(sys, "argv", ["train", "--cv", "2", "--precision", "no"])
    monkeypatch.setattr(launcher, "detect_gpus", lambda: 4)
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda cmd, **kwargs: (captured.append(cmd) or SimpleNamespace(returncode=0)),
    )
    assert launcher.main() == 0
    assert "--num_processes=1" in captured[0]
    assert "--mixed_precision=no" in captured[0]
    assert captured[0][-2:] == ["--precision", "no"]
    assert "--multi_gpu" not in captured[0]


def test_gpu_detection_respects_allocated_visibility(monkeypatch):
    from wavedl import hpo, launcher

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,GPU-unique-id")
    assert launcher.detect_gpus() == 2
    assert hpo._detect_gpu_count() == 2


def test_custom_imports_same_filename_are_distinct_and_idempotent(tmp_path):
    from wavedl.models import MODEL_REGISTRY, import_model_modules

    saved = MODEL_REGISTRY.copy()
    try:
        modules = []
        for name in ("production_custom_a", "production_custom_b"):
            directory = tmp_path / name
            directory.mkdir()
            module = directory / "model.py"
            module.write_text(
                f"from wavedl.models import register_model\n@register_model('{name}')\nclass Model: pass\n"
            )
            modules.append(str(module))
        import_model_modules(modules)
        import_model_modules(modules)
        assert "production_custom_a" in MODEL_REGISTRY
        assert "production_custom_b" in MODEL_REGISTRY
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(saved)


def test_hpo_gpu_leases_never_overlap_and_preserve_visible_ids(monkeypatch):
    import csv
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from threading import Lock
    from types import SimpleNamespace

    import optuna

    from wavedl import hpo

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    active = set()
    lock = Lock()

    def subprocess_run(command, **kwargs):
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        with lock:
            assert gpu in {"3", "7"}
            assert gpu not in active
            active.add(gpu)
        try:
            time.sleep(0.02 if gpu == "3" else 0.001)
            output = Path(command[command.index("--output_dir") + 1])
            with (output / "training_history.csv").open("w") as file:
                writer = csv.writer(file)
                writer.writerow(["val_loss"])
                writer.writerow([0.5])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        finally:
            with lock:
                active.remove(gpu)

    monkeypatch.setattr(hpo.subprocess, "run", subprocess_run)
    args = SimpleNamespace(
        target_shape=(1,),
        models=["cnn"],
        quick=True,
        medium=False,
        optimizers=None,
        schedulers=None,
        losses=None,
        batch_sizes=[1],
        data_path="unused.npz",
        inprocess=False,
        n_jobs=2,
        n_gpus=2,
        max_epochs=1,
        seed=1,
        timeout=2,
        allow_pickle=False,
    )
    objective = hpo.create_objective(args)
    params = {
        "model": "cnn",
        "lr": 0.001,
        "batch_size": 1,
        "optimizer": "adamw",
        "scheduler": "plateau",
        "loss": "mse",
        "weight_decay": 0.0001,
        "patience": 10,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda index: objective(optuna.trial.FixedTrial(params, number=index)),
                range(6),
            )
        )
    assert results == [0.5] * 6


def test_inprocess_trial_enforces_timeout(tmp_path, monkeypatch):
    from wavedl import train

    path = tmp_path / "dense.npz"
    np.savez(
        path,
        input_train=np.zeros((6, 16, 16), dtype=np.float32),
        output_train=np.ones((6, 2, 3, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        train,
        "_build_model_for_targets",
        lambda *a, **k: (_SmallFieldModel(), (2, 3, 4)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(TimeoutError, match="Trial exceeded"):
        train.train_single_trial(str(path), epochs=1, timeout_seconds=0)


def test_launcher_preserves_precision_from_yaml(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from wavedl import launcher

    config = tmp_path / "config.yaml"
    config.write_text("precision: 'no'\n")
    captured = []
    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(config), "--num_gpus", "1"]
    )
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda cmd, **kwargs: (captured.append(cmd) or SimpleNamespace(returncode=0)),
    )
    assert launcher.main() == 0
    assert "--mixed_precision=no" in captured[0]
    assert captured[0][-2:] == ["--precision", "no"]


@pytest.mark.parametrize("explicit_resume", [False, True])
@pytest.mark.parametrize("requested_scheduler", ["onecycle", "step"])
def test_onecycle_emergency_resume_rejected_before_data_loading(
    tiny_training, tmp_path, monkeypatch, explicit_resume, requested_scheduler
):
    from wavedl import train

    run, _, _ = tiny_training
    original = train._flag_nonfinite_loss
    batches = 0

    def interrupt_after_one_update(loss, nonfinite):
        nonlocal batches
        batches += 1
        if batches == 2:
            raise KeyboardInterrupt
        return original(loss, nonfinite)

    monkeypatch.setattr(train, "_flag_nonfinite_loss", interrupt_after_one_update)
    with pytest.raises(KeyboardInterrupt):
        run("--scheduler", "onecycle")
    checkpoint = tmp_path / "interrupted_checkpoint"
    saved_files = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    scheduler = torch.load(checkpoint / "scheduler.pt", weights_only=True)
    assert scheduler["last_epoch"] == 1

    monkeypatch.setattr(train, "_flag_nonfinite_loss", original)
    monkeypatch.setattr(
        train,
        "prepare_data",
        lambda *a, **k: pytest.fail("unsafe recovery must fail before loading data"),
    )
    resume_options = ["--resume", str(checkpoint)] if explicit_resume else []
    with pytest.raises(ValueError, match=r"epoch_\*_checkpoint.*--fresh"):
        run("--scheduler", requested_scheduler, *resume_options)
    assert {
        path.name: path.read_bytes() for path in checkpoint.iterdir()
    } == saved_files
    assert not (tmp_path / "training_complete.flag").exists()


def test_onecycle_recovers_from_epoch_boundary_after_interruption(
    tiny_training, tmp_path, monkeypatch
):
    from wavedl import train

    run, _, _ = tiny_training
    original = train._flag_nonfinite_loss
    batches = 0

    def interrupt_during_second_epoch(loss, nonfinite):
        nonlocal batches
        batches += 1
        if batches == 5:
            raise KeyboardInterrupt
        return original(loss, nonfinite)

    monkeypatch.setattr(train, "_flag_nonfinite_loss", interrupt_during_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        run("--scheduler", "onecycle")
    assert (tmp_path / "epoch_1_checkpoint").is_dir()
    assert (tmp_path / "interrupted_checkpoint").is_dir()

    monkeypatch.setattr(train, "_flag_nonfinite_loss", original)
    run("--scheduler", "onecycle", "--resume", str(tmp_path / "epoch_1_checkpoint"))
    completed = torch.load(
        tmp_path / "epoch_2_checkpoint" / "scheduler.bin", weights_only=True
    )
    assert completed["last_epoch"] == completed["total_steps"] == 6
    assert (tmp_path / "training_complete.flag").is_file()


def test_epoch_scheduler_emergency_recovery_remains_usable_with_warning(
    tiny_training, tmp_path, monkeypatch, caplog
):
    from wavedl import train

    run, _, _ = tiny_training
    original = train._flag_nonfinite_loss
    batches = 0

    def interrupt_after_one_update(loss, nonfinite):
        nonlocal batches
        batches += 1
        if batches == 2:
            raise KeyboardInterrupt
        return original(loss, nonfinite)

    monkeypatch.setattr(train, "_flag_nonfinite_loss", interrupt_after_one_update)
    with pytest.raises(KeyboardInterrupt):
        run("--scheduler", "step")
    monkeypatch.setattr(train, "_flag_nonfinite_loss", original)
    caplog.clear()
    run("--scheduler", "step")
    assert "This recovery is not exact" in caplog.text
    assert (tmp_path / "training_complete.flag").is_file()
    assert (tmp_path / "epoch_2_checkpoint").is_dir()


@pytest.mark.parametrize("positional", [False, True])
def test_inprocess_pruned_trial_records_best_val_loss_and_supports_patience_one(
    tmp_path, monkeypatch, positional
):
    from wavedl import train

    path = tmp_path / "dense.npz"
    np.savez(
        path,
        input_train=np.zeros((6, 16, 16), dtype=np.float32),
        output_train=np.ones((6, 2, 3, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        train,
        "_build_model_for_targets",
        lambda *a, **k: (_SmallFieldModel(), (2, 3, 4)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _PruneFirstEpochTrial:
        def report(self, value, step):
            self.reported = (value, step)

        def should_prune(self):
            return True

    trial = _PruneFirstEpochTrial()
    if positional:
        # Preserve every positional parameter in the pre-grad_clip public API.
        result = train.train_single_trial(
            str(path),
            "cnn",
            1e-3,
            32,
            2,
            1,
            "adamw",
            "plateau",
            "mse",
            1e-4,
            2025,
            "no",
            0,
            1.0,
            0.9,
            trial,
            False,
            "mse",
            False,
            None,
            grad_clip=0.0,
        )
    else:
        result = train.train_single_trial(
            str(path),
            epochs=2,
            patience=1,
            scheduler_name="plateau",
            trial=trial,
        )
    assert result["pruned"] is True
    assert result["epochs_trained"] == 1
    assert np.isfinite(result["best_val_loss"])
    assert result["best_val_loss"] == pytest.approx(result["final_val_loss"])
    assert trial.reported == (result["final_val_loss"], 0)


def test_launcher_detects_wandb_from_yaml_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from wavedl import launcher

    config = tmp_path / "config.yaml"
    config.write_text("wandb: true\n")
    summary_calls = []
    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(config), "--num_gpus", "1"]
    )
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        launcher,
        "print_summary",
        lambda code, wandb_enabled, *a: summary_calls.append((code, wandb_enabled)),
    )
    assert launcher.main() == 0
    assert summary_calls == [(0, True)]
