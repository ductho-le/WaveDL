"""Real two-fold CPU execution must honour the approved component options."""

import json
import math
import pickle

import numpy as np
import pytest
import torch
from accelerate import Accelerator

from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import decode
from wavedl.core.training_cli import parse_training_args
from wavedl.models.base import BaseModel


class TinyCvModel(BaseModel):
    """Zero-initialised linear model retains genuine gradients and CPU steps."""

    def __init__(self, in_shape, out_size, **kwargs):
        super().__init__(in_shape, out_size, **kwargs)
        self.linear = torch.nn.Linear(math.prod(in_shape), out_size)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x.flatten(1))


@pytest.fixture
def cpu_cv(tmp_path, monkeypatch):
    from wavedl import train
    from wavedl.models.registry import MODEL_REGISTRY
    from wavedl.utils import cross_validation

    data = tmp_path / "data.npz"
    rows = np.arange(8, dtype=np.float32)
    np.savez(
        data,
        X=np.stack([rows / 8, (rows + 1) / 8, rows**2 / 64, -rows / 8], axis=1),
        y=np.stack([rows, rows**2], axis=1),
    )
    monkeypatch.setitem(MODEL_REGISTRY, "cnn", TinyCvModel)
    monkeypatch.setattr(train, "Accelerator", lambda **kw: Accelerator(cpu=True, **kw))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    real_train_fold = cross_validation.train_fold
    observed = []

    def observe_fold(**kwargs):
        criterion = kwargs["criterion"]
        base_criterion = getattr(criterion, "base_loss", criterion)
        optimizer = kwargs["optimizer"]
        scheduler = kwargs["scheduler"]
        initial = {
            "criterion": type(criterion).__name__,
            "weights": (
                None
                if getattr(base_criterion, "weights", None) is None
                else base_criterion.weights.detach().cpu().tolist()
            ),
            "delta": getattr(base_criterion, "delta", None),
            "nesterov": optimizer.param_groups[0].get("nesterov"),
            "scheduler": type(scheduler).__name__,
            "gamma": getattr(scheduler, "gamma", None),
            "step_size": getattr(scheduler, "step_size", None),
            "milestones": dict(getattr(scheduler, "milestones", {})),
            "total_iters": getattr(scheduler, "total_iters", None),
            "total_steps": getattr(scheduler, "total_steps", None),
        }
        result = real_train_fold(**kwargs)
        observed.append(
            {
                **initial,
                "loss": result["history"][0]["train_loss"],
                "final_lr": optimizer.param_groups[0]["lr"],
                "epochs": result["epochs_trained"],
            }
        )
        return result

    monkeypatch.setattr(cross_validation, "train_fold", observe_fold)
    count = 0

    def run(*options):
        nonlocal count
        count += 1
        output = tmp_path / f"run-{count}"
        argv = [
            "--data_path",
            str(data),
            "--output_dir",
            str(output),
            "--model",
            "cnn",
            "--cv",
            "2",
            "--epochs",
            "1",
            "--workers",
            "0",
            "--batch_size",
            "8",
            "--precision",
            "no",
            "--lr",
            "0.01",
            "--grad_clip",
            "0",
            *options,
        ]
        monkeypatch.setattr(train, "parse_args", lambda: parse_training_args(argv))
        observed.clear()
        train.main()
        journal_path = next((output / "attempts").glob("*/events.jsonl"))
        events = [decode(line) for line in journal_path.read_bytes().splitlines()]
        assert events[0].type == "hello"
        assert events[0].payload["capabilities"] == ["cross_validation", "lifecycle"]
        assert not any(event.type == "metric" for event in events)
        assert events[-1].payload["status"] == "completed"
        artifacts = [event.payload for event in events if event.type == "artifact"]
        assert [item["label"] for item in artifacts] == ["fold_1", "fold_2", "summary"]
        layout = RunLayout(output, events[0].run_id)
        generations = published_generations(layout)
        assert len(generations) == 3
        summary = json.loads((output / "cv_summary.json").read_text())
        assert summary["folds"] == 2
        assert [fold["fold"] for fold in summary["fold_results"]] == [1, 2]
        assert all("history" not in fold for fold in summary["fold_results"])
        for fold in (1, 2):
            with (output / f"fold_{fold}" / "training_meta.pkl").open("rb") as stream:
                metadata = pickle.load(stream)
            assert metadata["cross_validation_fold"] == fold
            assert metadata["epoch"] <= metadata["epochs_trained"]
        assert len(observed) == 2
        return [dict(item) for item in observed]

    yield run
    torch.set_num_threads(previous_threads)


def test_actual_cv_weighted_loss_nesterov_and_step_scheduler(cpu_cv):
    """Authored weights change actual first-epoch loss, not merely factory kwargs."""
    weighted = cpu_cv(
        "--loss",
        "weighted_mse",
        "--loss_weights",
        "100,1",
        "--optimizer",
        "sgd",
        "--momentum",
        "0.8",
        "--nesterov",
        "--scheduler",
        "step",
        "--step_size",
        "1",
        "--scheduler_factor",
        "0.5",
    )
    changed = cpu_cv(
        "--loss",
        "weighted_mse",
        "--loss_weights",
        "1,1",
        "--optimizer",
        "sgd",
        "--momentum",
        "0.8",
        "--scheduler",
        "step",
        "--step_size",
        "2",
        "--scheduler_factor",
        "0.2",
    )
    for first, second in zip(weighted, changed, strict=True):
        assert first["weights"] == [100, 1]
        assert first["nesterov"] is True
        assert first["step_size"] == 1 and first["gamma"] == 0.5
        assert first["loss"] == pytest.approx(50.5, rel=1e-6)
        assert first["final_lr"] == pytest.approx(0.005)
        assert second["weights"] == [1, 1]
        assert second["nesterov"] is False
        assert second["step_size"] == 2 and second["gamma"] == 0.2
        assert second["loss"] == pytest.approx(1, rel=1e-6)
        assert second["final_lr"] == pytest.approx(0.01)


def test_actual_cv_huber_delta_changes_training_loss(cpu_cv):
    small = cpu_cv("--loss", "huber", "--huber_delta", "0.25")
    normal = cpu_cv("--loss", "huber")
    for first, second in zip(small, normal, strict=True):
        assert first["delta"] == 0.25
        assert second["delta"] == 1
        assert first["loss"] < second["loss"]
        assert first["criterion"] == second["criterion"] == "HuberLoss"


@pytest.mark.parametrize(
    "name,options,expected",
    [
        ("step", [], {"scheduler": "StepLR", "step_size": 30, "gamma": 0.5}),
        (
            "multistep",
            ["--milestones", "1,2", "--scheduler_factor", "0.4"],
            {"scheduler": "MultiStepLR", "milestones": {1: 1, 2: 1}, "gamma": 0.4},
        ),
        (
            "exponential",
            ["--scheduler_factor", "0.9"],
            {"scheduler": "ExponentialLR", "gamma": 0.9},
        ),
        (
            "linear_warmup",
            ["--warmup_epochs", "2"],
            {"scheduler": "LinearLR", "total_iters": 2},
        ),
        ("plateau", [], {"scheduler": "ReduceLROnPlateau"}),
        ("cosine", [], {"scheduler": "CosineAnnealingLR"}),
        ("cosine_restarts", [], {"scheduler": "CosineAnnealingWarmRestarts"}),
        ("onecycle", [], {"scheduler": "OneCycleLR", "total_steps": 2}),
    ],
)
def test_actual_cv_scheduler_branches(cpu_cv, name, options, expected):
    observed = cpu_cv("--epochs", "2", "--scheduler", name, *options)
    for fold in observed:
        assert fold["epochs"] == 2
        for key, value in expected.items():
            assert fold[key] == value
        if name == "multistep":
            assert fold["final_lr"] == pytest.approx(0.01 * 0.4**2)
        elif name == "exponential":
            assert fold["final_lr"] == pytest.approx(0.01 * 0.9**2)
        elif name in {"step", "plateau", "linear_warmup"}:
            assert fold["final_lr"] == pytest.approx(0.01)


def test_actual_cv_default_loss_and_optimizer_remain_unchanged(cpu_cv):
    for fold in cpu_cv():
        assert fold["criterion"] == "MSELoss"
        assert fold["weights"] is None
        assert fold["scheduler"] == "ReduceLROnPlateau"
        assert fold["final_lr"] == pytest.approx(0.01)
        assert fold["loss"] == pytest.approx(1, rel=1e-6)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("step", {"step_size": 30, "gamma": 0.1}),
        ("exponential", {"gamma": 0.95}),
        ("linear_warmup", {"total_iters": 5}),
    ],
)
def test_direct_cv_helper_keeps_legacy_component_defaults(
    tmp_path, monkeypatch, name, expected
):
    """Existing helper callers retain factory defaults absent explicit new options."""
    from wavedl.models.registry import MODEL_REGISTRY
    from wavedl.utils import cross_validation

    monkeypatch.setitem(MODEL_REGISTRY, "cnn", TinyCvModel)
    original = cross_validation.train_fold
    observed = []

    def observe(**kwargs):
        scheduler = kwargs["scheduler"]
        observed.append({key: getattr(scheduler, key) for key in expected})
        assert type(kwargs["criterion"]).__name__ == "MSELoss"
        return original(**kwargs)

    monkeypatch.setattr(cross_validation, "train_fold", observe)
    rows = np.arange(8, dtype=np.float32)
    summary = cross_validation.run_cross_validation(
        X=np.stack([rows / 8, (rows + 1) / 8], axis=1),
        y=rows.reshape(-1, 1),
        model_name="cnn",
        in_shape=(2,),
        out_size=1,
        folds=2,
        epochs=1,
        batch_size=8,
        workers=0,
        scheduler_name=name,
        output_dir=str(tmp_path / "direct-cv"),
        device=torch.device("cpu"),
    )
    assert observed == [expected, expected]
    assert len(summary["fold_results"]) == 2


@pytest.mark.parametrize(
    "options,shape,message",
    [
        (["--betas", "broken,0.999"], (8, 4), "could not convert"),
        (["--betas", "0.9"], (8, 4), "exactly two"),
        (["--betas", "0.9,1"], (8, 4), "exactly two"),
        (["--betas", "nan,0.999"], (8, 4), "exactly two"),
        (["--loss_weights", "broken,1"], (8, 4), "could not convert"),
        (["--milestones", "broken,1"], (8, 4), "invalid literal"),
        ([], (8, 2, 17, 17), "Ambiguous input shape"),
    ],
)
def test_actual_hdf5_handle_closed_on_cv_setup_failure(
    tmp_path, monkeypatch, options, shape, message
):
    """Real open h5py files close before failed lifecycle publication returns."""
    import h5py

    from wavedl import train
    from wavedl.utils import cross_validation
    from wavedl.utils.data import HDF5Source

    data = tmp_path / "invalid-options.h5"
    with h5py.File(data, "w") as stream:
        stream.create_dataset("X", data=np.zeros(shape, dtype=np.float32))
        stream.create_dataset("y", data=np.arange(8, dtype=np.float32).reshape(8, 1))
    original_load = HDF5Source.load_mmap
    handles = []
    files = []
    close_counts = []

    def capture(source, path):
        handle = original_load(source, path)
        assert handle._file.id.valid
        handles.append(handle)
        files.append(handle._file)
        close_counts.append(0)
        index = len(close_counts) - 1
        original_close = handle.close

        def close():
            close_counts[index] += 1
            original_close()

        handle.close = close
        return handle

    def forbidden(**kwargs):
        raise AssertionError("Invalid setup must not enter CV training")

    monkeypatch.setattr(HDF5Source, "load_mmap", capture)
    monkeypatch.setattr(cross_validation, "run_cross_validation", forbidden)
    monkeypatch.setattr(train, "Accelerator", lambda **kw: Accelerator(cpu=True, **kw))
    output = tmp_path / "failed-run"
    argv = [
        "--data_path",
        str(data),
        "--output_dir",
        str(output),
        "--cv",
        "2",
        "--precision",
        "no",
        "--workers",
        "0",
        *options,
    ]
    monkeypatch.setattr(train, "parse_args", lambda: parse_training_args(argv))
    with pytest.raises(ValueError, match=message):
        train.main()
    assert len(handles) == 1
    assert handles[0]._closed and handles[0]._file is None
    assert not files[0].id.valid
    assert close_counts == [1]
    journal = next((output / "attempts").glob("*/events.jsonl"))
    events = [decode(line) for line in journal.read_bytes().splitlines()]
    assert events[-1].payload["status"] == "failed"
    assert events[-1].payload["code"] != 0
    assert not any(event.type in {"artifact", "metric"} for event in events)
    assert not list(output.rglob("COMPLETE.json"))
    assert not (output / "cv_summary.json").exists()
