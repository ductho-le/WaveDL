"""Bounded actual CPU training and rank-aware lifecycle integration."""

import json
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import Accelerator

from wavedl.core.protocol import decode
from wavedl.core.training_cli import parse_training_args
from wavedl.runtime import open_training_runtime, training_lifecycle


def events(output):
    path = next((output / "attempts").glob("*/events.jsonl"))
    return [decode(line) for line in path.read_bytes().splitlines()]


@pytest.fixture
def training(tmp_path, monkeypatch):
    from wavedl import train

    data = tmp_path / "data.npz"
    generator = np.random.default_rng(8)
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    output = tmp_path / "run"
    argv = [
        "--data_path",
        str(data),
        "--output_dir",
        str(output),
        "--model",
        "cnn",
        "--epochs",
        "1",
        "--workers",
        "0",
        "--batch_size",
        "8",
        "--precision",
        "no",
    ]
    monkeypatch.setattr(train, "parse_args", lambda: parse_training_args(argv))
    monkeypatch.setattr(train, "Accelerator", lambda **kw: Accelerator(cpu=True, **kw))
    torch.set_num_threads(1)
    return train, output


def test_actual_cpu_one_epoch(training, capsys):
    train, output = training
    train.main()
    journal = events(output)
    assert journal[0].type == "hello"
    assert journal[0].payload["device"] == "cpu"
    assert journal[0].payload["interpreter"] == sys.executable
    metrics = [event for event in journal if event.type == "metric"]
    assert len(metrics) == 1 and metrics[0].payload["step"] == 1
    epoch_metrics = metrics[0].payload["metrics"]
    assert epoch_metrics["best_val_loss"] == pytest.approx(epoch_metrics["val_loss"])
    assert epoch_metrics["patience_counter"] == 0
    assert journal[-1].payload["status"] == "completed"
    assert "##METRICS##" in capsys.readouterr().out
    assert (output / "best_checkpoint/training_meta.pkl").is_file()


def test_actual_cpu_plateau_metrics_match_checkpoint_and_early_stop(
    training, monkeypatch, capsys
):
    """A constant-output model exercises real loss/backward/optimizer epochs."""
    from wavedl.models.base import BaseModel
    from wavedl.models.registry import MODEL_REGISTRY

    class PlateauModel(BaseModel):
        def __init__(self, in_shape, out_size, **kwargs):
            super().__init__(in_shape, out_size, **kwargs)
            self.weight = torch.nn.Parameter(torch.zeros(out_size))

        def forward(self, x):
            # The trainable parameter retains a real autograd/optimizer path,
            # but zero sensitivity makes validation loss exactly stationary.
            return (self.weight * 0).expand(len(x), self.out_size)

    monkeypatch.setitem(MODEL_REGISTRY, "cnn", PlateauModel)
    train, output = training
    args, parser = train.parse_args()
    args.epochs = 5
    args.patience = 2
    args.save_every = 1
    monkeypatch.setattr(train, "parse_args", lambda: (args, parser))
    train.main()
    journal = events(output)
    metrics = [event.payload["metrics"] for event in journal if event.type == "metric"]
    assert [item["epoch"] for item in metrics] == [1, 2, 3]
    assert [item["patience_counter"] for item in metrics] == [0, 1, 2]
    assert all(item["val_loss"] == metrics[0]["val_loss"] for item in metrics)
    assert all(item["best_val_loss"] == metrics[0]["val_loss"] for item in metrics)
    for item in metrics:
        metadata_path = (
            output / f"epoch_{item['epoch']}_checkpoint" / "training_meta.pkl"
        )
        with metadata_path.open("rb") as stream:
            metadata = pickle.load(stream)
        assert metadata["best_val_loss"] == item["best_val_loss"]
        assert metadata["patience_ctr"] == item["patience_counter"]
    console_metrics = [
        json.loads(line.removeprefix("##METRICS##"))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("##METRICS##")
    ]
    assert console_metrics == metrics
    assert journal[-1].payload["status"] == "completed"


@pytest.mark.parametrize(
    "exception,status,code",
    [
        (RuntimeError("setup failed"), "failed", 1),
        (KeyboardInterrupt("cancelled"), "cancelled", 130),
    ],
)
def test_setup_failure_and_cancel_are_terminal(
    training, monkeypatch, exception, status, code
):
    train, output = training

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(train, "prepare_data", fail)
    with pytest.raises(type(exception)):
        train.main()
    assert events(output)[-1].payload == {
        "code": code,
        "status": status,
        "message": str(exception),
        "artifacts": [],
    }


def test_non_main_runtime_uses_identity_only(tmp_path, monkeypatch):
    args, _ = parse_training_args(["--output_dir", str(tmp_path), "--workers", "0"])
    main = SimpleNamespace(
        process_index=0,
        is_main_process=True,
        num_processes=1,
        device=torch.device("cpu"),
        wait_for_everyone=lambda: None,
    )
    journal = open_training_runtime(args, main)
    identity = journal.bootstrap.environment
    journal.close()
    for name, value in identity.items():
        monkeypatch.setenv(name, value)
    follower = SimpleNamespace(
        process_index=1,
        is_main_process=False,
        num_processes=2,
        device=torch.device("cpu"),
        wait_for_everyone=lambda: None,
    )
    before = (tmp_path / "attempts/1/events.jsonl").read_bytes()
    follower_journal = open_training_runtime(args, follower)
    assert follower_journal.last_sequence is None
    with training_lifecycle(follower_journal):
        pass
    assert (tmp_path / "attempts/1/events.jsonl").read_bytes() == before
    assert (tmp_path / "attempts/1/rank/1.log").exists()
    assert [path.name for path in (tmp_path / "attempts").iterdir()] == ["1"]


def test_list_models_does_not_bootstrap(tmp_path, monkeypatch):
    from wavedl import train

    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: parse_training_args(
            ["--list_models", "--output_dir", str(tmp_path / "unused")]
        ),
    )
    with pytest.raises(SystemExit) as result:
        train.main()
    assert result.value.code == 0
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("foreign", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_generation_resume_restores_history_and_supersedes(
    training, monkeypatch, tmp_path, foreign, legacy
):
    from wavedl.core.generations import publish_generation
    from wavedl.core.layout import RunLayout
    from wavedl.core.manifest import read_manifest

    train, output = training
    train.main()
    staging = tmp_path / "staging"
    shutil.copytree(output / "best_checkpoint", staging)
    shutil.copy2(output / "training_history.csv", staging / "training_history.csv")
    # Restore a previously accumulated patience counter and force the next
    # finite actual validation result to improve.  The published generation
    # includes this metadata, rather than faking epoch metrics or loop results.
    metadata_path = staging / "training_meta.pkl"
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    metadata.update(best_val_loss=float("inf"), patience_ctr=3)
    with metadata_path.open("wb") as stream:
        pickle.dump(metadata, stream)
    if legacy:
        selection_path = staging / "checkpoint_selection.json"
        selection = json.loads(selection_path.read_text())
        selection.pop("source_seq")
        selection_path.write_text(json.dumps(selection))
    layout = RunLayout(output.resolve(), read_manifest(output / "manifest.json").run_id)
    layout.checkpoints.mkdir(exist_ok=True)
    generation = publish_generation(layout, staging, attempt=1, step=1)
    args, parser = train.parse_args()
    args.epochs = 2
    if foreign:
        args.output_dir = str(tmp_path / "foreign-target")
        args.resume = str(generation.path)
    else:
        (output / "training_complete.flag").unlink()
    monkeypatch.setattr(train, "parse_args", lambda: (args, parser))
    train.main()
    resumed_journal_path = (
        Path(args.output_dir) / "attempts" / ("1" if foreign else "2") / "events.jsonl"
    )
    resumed_journal = [
        decode(line) for line in resumed_journal_path.read_bytes().splitlines()
    ]
    resumed_metric = next(
        event.payload["metrics"] for event in resumed_journal if event.type == "metric"
    )
    assert resumed_metric["best_val_loss"] == resumed_metric["val_loss"]
    assert resumed_metric["patience_counter"] == 0
    if foreign:
        output = tmp_path / "foreign-target"
        journal = events(output)
        assert not any(event.type == "supersede" for event in journal)
        assert any(
            "Resumed foreign run" in event.payload.get("message", "")
            for event in journal
        )
        import pandas as pd

        assert pd.read_csv(output / "training_history.csv")["epoch"].tolist() == [1, 2]
        return
    journal = [decode(line) for line in layout.journal(2).read_bytes().splitlines()]
    rollback = next(event for event in journal if event.type == "supersede")
    metric = next(event for event in journal if event.type == "metric")
    assert rollback.payload == {
        "source_attempt": 1,
        "step": 1,
        "generation": generation.generation,
        "reason": "resume",
        "source_seq": json.loads(
            (generation.path / "checkpoint_selection.json").read_text()
        ).get("source_seq"),
    }
    assert rollback.seq < metric.seq and metric.payload["step"] == 2
    if legacy:
        assert any(
            "Legacy generation has no causal" in event.payload.get("message", "")
            for event in journal
        )
    import pandas as pd

    assert pd.read_csv(output / "training_history.csv")["epoch"].tolist() == [1, 2]


def test_actual_cv_lifecycle_without_fake_epoch_metrics(training, monkeypatch):
    train, output = training
    args, parser = train.parse_args()
    args.cv = 2
    monkeypatch.setattr(train, "parse_args", lambda: (args, parser))
    train.main()
    journal = events(output)
    assert "epoch_metrics" not in journal[0].payload["capabilities"]
    assert not any(event.type == "metric" for event in journal)
    assert journal[-1].payload["status"] == "completed"


def test_loop_cancel_saves_emergency_and_does_not_report_success(training, monkeypatch):
    train, output = training
    accelerator = Accelerator(cpu=True, mixed_precision="no")

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt("batch cancelled")

    monkeypatch.setattr(accelerator, "backward", cancel)
    monkeypatch.setattr(train, "Accelerator", lambda **kwargs: accelerator)
    with pytest.raises(KeyboardInterrupt):
        train.main()
    assert events(output)[-1].payload["status"] == "cancelled"
    assert (output / "interrupted_checkpoint/model_weights.pth").is_file()
    assert not (output / "training_complete.flag").exists()


def test_normal_generations_preserve_scheduler_resume_boundary(training, monkeypatch):
    from wavedl.core.checkpoint_selection import select_checkpoint
    from wavedl.core.layout import RunLayout
    from wavedl.core.manifest import read_manifest

    train, output = training
    args, parser = train.parse_args()
    args.scheduler = "step"
    args.step_size = 1
    args.scheduler_factor = 0.5
    args.save_every = 1
    monkeypatch.setattr(train, "parse_args", lambda: (args, parser))
    train.main()
    layout = RunLayout(output.resolve(), read_manifest(output / "manifest.json").run_id)
    generation = select_checkpoint(layout)
    scheduler = torch.load(
        generation.path / "custom_checkpoint_0.pkl", weights_only=False
    )
    assert scheduler["last_epoch"] == 1
    assert all(value == pytest.approx(args.lr * 0.5) for value in scheduler["_last_lr"])
    (output / "training_complete.flag").unlink()
    args.epochs = 2
    train.main()
    generation = select_checkpoint(layout)
    scheduler = torch.load(
        generation.path / "custom_checkpoint_0.pkl", weights_only=False
    )
    assert scheduler["last_epoch"] == 2
    assert all(
        value == pytest.approx(args.lr * 0.25) for value in scheduler["_last_lr"]
    )


def test_periodic_every_two_epochs_publishes_only_on_boundary(training, monkeypatch):
    train, output = training
    args, parser = train.parse_args()
    args.epochs = 2
    args.save_every = 2
    monkeypatch.setattr(train, "parse_args", lambda: (args, parser))
    train.main()
    artifacts = [event.payload for event in events(output) if event.type == "artifact"]
    periodic = [item for item in artifacts if item["label"] == "periodic"]
    assert len(periodic) == 1 and periodic[0]["step"] == 2
    assert (output / "epoch_2_checkpoint/training_history.csv").exists()
    assert (output / "best_checkpoint/scaler.pkl").exists()


def test_emergency_publication_failure_reports_failed_not_cancelled(
    training, monkeypatch
):
    from wavedl import checkpointing

    train, output = training
    accelerator = Accelerator(cpu=True, mixed_precision="no")

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt("cancelled batch")

    def disk_full(*args, **kwargs):
        raise OSError("publication disk full")

    monkeypatch.setattr(accelerator, "backward", cancel)
    monkeypatch.setattr(train, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(checkpointing, "publish_generation", disk_full)
    with pytest.raises(OSError, match="publication disk full"):
        train.main()
    assert events(output)[-1].payload["status"] == "failed"
    assert not any(event.type == "artifact" for event in events(output))
