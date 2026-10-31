"""Production-default CPU CV publishes fold checkpoints through public HTTP."""

import json
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings

from .test_writer_http_acceptance import DisconnectAfterExit
from .test_writer_operation_parity import submit_operation, verify_artifacts


@pytest.fixture
def cv_http(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")

    def forbidden(*args, **kwargs):
        raise AssertionError("The daemon must not allocate writer attempts")

    monkeypatch.setattr(
        "wavedl_daemon.runs.control.dispatch.allocate_attempt", forbidden
    )
    data = tmp_path / "data.npz"
    generator = np.random.default_rng(47)
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(settings)
    app = create_app(
        settings,
        token="cv-test",
        daemon_id="test",
        build_id="test",
        services=services,
    )
    runtime = {
        "python": sys.executable,
        "mode": "existing",
        "device": "cpu",
        "environment": {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    }
    with TestClient(
        DisconnectAfterExit(app),
        base_url="http://localhost:8420",
        headers={"Authorization": "Bearer cv-test"},
    ) as client:
        yield client, settings, runtime, data
    assert not services.host._thread.is_alive()


def test_actual_cpu_writer_cv_fold_publication_and_inference(cv_http):
    client, _, _, data = cv_http
    url, root, events = submit_operation(
        cv_http,
        "train",
        {
            "model": "cnn",
            "epochs": 1,
            "cv": 2,
            "data_path": str(data),
            "precision": "no",
            "workers": 0,
            "batch_size": 8,
        },
    )
    assert events[0]["payload"]["capabilities"] == ["cross_validation", "lifecycle"]
    assert not any(event["type"] == "metric" for event in events)
    artifacts = [event["payload"] for event in events if event["type"] == "artifact"]
    assert [(item["label"], item["step"]) for item in artifacts] == [
        ("fold_1", 1),
        ("fold_2", 2),
        ("summary", 2),
    ]
    generations = verify_artifacts(client, url)
    assert len(generations) == 3
    summary = next(item for item in generations if "cv_summary.json" in item["files"])
    response = client.get(f"{url}/artifacts/{summary['generation']}/cv_summary.json")
    assert response.json()["folds"] == 2
    assert [item["fold"] for item in response.json()["fold_results"]] == [1, 2]
    fold_generations = [
        item for item in generations if "model_weights.pth" in item["files"]
    ]
    assert len(fold_generations) == 2
    for fold, generation in enumerate(fold_generations, 1):
        assert (root / f"fold_{fold}" / "model.pth").is_file()
        assert {
            "model.pth",
            "model_weights.pth",
            "scaler.pkl",
            "training_meta.pkl",
        } <= set(generation["files"])
        marker = json.loads(
            (
                root / "checkpoints" / generation["generation"] / "COMPLETE.json"
            ).read_text()
        )
        assert marker["attempt"] == 1 and marker["step"] == fold
        _, _, inference = submit_operation(
            cv_http,
            "test",
            {
                "checkpoint": str(root / "checkpoints" / generation["generation"]),
                "data_path": str(data),
                "batch_size": 8,
                "workers": 0,
            },
        )
        metrics = [event for event in inference if event["type"] == "metric"]
        assert len(metrics) == 1 and metrics[0]["payload"]["metrics"]["samples"] == 20


def test_actual_cv_single_sample_folds_publish_strict_undefined_metrics(cv_http):
    import numpy as np

    from wavedl.core.protocol import strict_loads

    client, _, _, data = cv_http
    generator = np.random.default_rng(53)
    np.savez(
        data,
        X=generator.normal(size=(2, 32)).astype("float32"),
        y=generator.normal(size=(2, 1)).astype("float32"),
    )
    url, _, events = submit_operation(
        cv_http,
        "train",
        {
            "model": "cnn",
            "epochs": 1,
            "cv": 2,
            "data_path": str(data),
            "precision": "no",
            "workers": 0,
            "batch_size": 1,
        },
    )
    assert not any(event["type"] == "metric" for event in events)
    generations = verify_artifacts(client, url)
    summary = next(item for item in generations if "cv_summary.json" in item["files"])
    response = client.get(f"{url}/artifacts/{summary['generation']}/cv_summary.json")
    payload = strict_loads(response.content)
    assert payload["r2_mean"] is None and payload["r2_std"] is None
    assert [fold["r2"] for fold in payload["fold_results"]] == [None, None]
    assert [fold["r2_target_0"] for fold in payload["fold_results"]] == [None, None]
    assert payload["mae_mean"] is not None and payload["val_loss_mean"] is not None


def test_cv_saved_metadata_describes_best_weights_not_last_epoch(
    cv_http, tmp_path, monkeypatch
):
    """Producer metadata must describe restored best tensors, not total work."""
    import pickle

    import numpy as np
    import torch

    from wavedl.utils import cross_validation

    def best_at_first_epoch(**kwargs):
        # Controlled fold result isolates metadata accounting; model tensors,
        # scaler, fold split and checkpoint serialization are real.
        with torch.no_grad():
            for parameter in kwargs["model"].parameters():
                parameter.fill_(1)
        return {
            "fold": kwargs["fold"] + 1,
            "best_epoch": 1,
            "epochs_trained": 2,
            "best_val_loss": 0.5,
            "r2": 0.25,
            "r2_target_0": 0.25,
            "mae_normalized": 0.5,
            "mae_physical": 1.0,
            "mae_target_0": 1.0,
            "history": [],
        }

    monkeypatch.setattr(cross_validation, "train_fold", best_at_first_epoch)
    _, _, _, data = cv_http
    with np.load(data) as arrays:
        cross_validation.run_cross_validation(
            X=arrays["X"],
            y=arrays["y"],
            model_name="cnn",
            in_shape=(32,),
            out_size=1,
            folds=2,
            epochs=2,
            workers=0,
            batch_size=8,
            output_dir=str(tmp_path / "metadata-cv"),
            device=torch.device("cpu"),
        )
    for fold in (1, 2):
        root = tmp_path / "metadata-cv" / f"fold_{fold}"
        with (root / "training_meta.pkl").open("rb") as stream:
            metadata = pickle.load(stream)
        assert metadata["epoch"] == 1 and metadata["epochs_trained"] == 2
        assert metadata["cross_validation_fold"] == fold
        assert metadata["in_shape"] == (32,) and metadata["out_dim"] == 1


@pytest.mark.parametrize("publication_fault", [False, True])
def test_actual_cv_ignores_stale_folds_and_preserves_publication_failure(
    cv_http, tmp_path, monkeypatch, publication_fault
):
    """Real fold training cannot report completion after a publication fault."""
    import torch
    from accelerate import Accelerator

    from wavedl import checkpointing, train
    from wavedl.core.generations import published_generations
    from wavedl.core.layout import RunLayout
    from wavedl.core.manifest import read_manifest
    from wavedl.core.protocol import decode
    from wavedl.core.training_cli import parse_training_args

    _, _, _, data = cv_http
    root = tmp_path / "standalone-cv"
    stale = root / "fold_99"
    stale.mkdir(parents=True)
    (stale / "model_weights.pth").write_bytes(b"old unrelated fold")
    first = root / "fold_1"
    first.mkdir()
    (first / "old_unrelated.dat").write_bytes(b"not this CV output")
    argv = [
        "--data_path",
        str(data),
        "--output_dir",
        str(root),
        "--model",
        "cnn",
        "--epochs",
        "1",
        "--cv",
        "2",
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
    original = checkpointing.publish_generation
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if publication_fault and calls == 2:
            raise OSError("CV publication fault")
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpointing, "publish_generation", fail_second)
    if publication_fault:
        with pytest.raises(OSError, match="CV publication fault"):
            train.main()
    else:
        train.main()
    manifest = read_manifest(root / "manifest.json")
    layout = RunLayout(root, manifest.run_id)
    publications = published_generations(layout)
    assert len(publications) == (1 if publication_fault else 3)
    assert all("old_unrelated.dat" not in item.files for item in publications)
    assert [item.step for item in publications] == (
        [1] if publication_fault else [1, 2, 2]
    )
    events = [decode(line) for line in layout.journal(1).read_bytes().splitlines()]
    assert events[-1].type == "exit"
    assert events[-1].payload["status"] == (
        "failed" if publication_fault else "completed"
    )
    assert not any(event.type == "metric" for event in events)
    assert len([event for event in events if event.type == "artifact"]) == len(
        publications
    )
    assert (stale / "model_weights.pth").read_bytes() == b"old unrelated fold"
