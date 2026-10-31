"""Quiescent collective publication versus rank-only emergency publication."""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
import torch

from wavedl import checkpointing
from wavedl.core import generations
from wavedl.core.events import HelloPayload
from wavedl.core.generations import published_generations
from wavedl.core.manifest import RuntimeSelection
from wavedl.core.protocol import decode
from wavedl.journal import RunJournal, bootstrap_run


@pytest.fixture
def publication(tmp_path):
    bootstrap = bootstrap_run(
        tmp_path / "run",
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        environment={},
    )
    journal = RunJournal(
        bootstrap,
        hello=HelloPayload(1, 1, "test", "3.11", "cpu", ["train"], sys.executable),
    )
    scaler = tmp_path / "scaler.pkl"
    scaler.write_bytes(b"scaler")
    metadata = {"epoch": 1, "best_val_loss": 1.0, "patience_ctr": 0}
    yield journal, scaler, metadata
    journal.close()


def test_two_rank_normal_save_quiesces_before_publication(publication, monkeypatch):
    journal, scaler, metadata = publication
    path = journal.bootstrap.layout.run_dir / "best_checkpoint"
    barrier = Barrier(2, timeout=5)
    lock = Lock()
    order = []
    actual_publish = checkpointing.publish_generation

    def publish(*args, **kwargs):
        with lock:
            assert {"save0", "save1"}.issubset(order)
            order.append("publish")
        return actual_publish(*args, **kwargs)

    monkeypatch.setattr(checkpointing, "publish_generation", publish)

    def worker(rank):
        def save_state(directory, **kwargs):
            path.mkdir(exist_ok=True)
            (path / f"rank-{rank}.pt").write_bytes(b"rank state")
            with lock:
                order.append(f"save{rank}")

        accelerator = SimpleNamespace(
            is_main_process=rank == 0,
            save_state=save_state,
            wait_for_everyone=barrier.wait,
        )
        result = checkpointing.save_normal_checkpoint(
            accelerator,
            path,
            journal,
            metadata=metadata,
            history=[{"epoch": 1, "train_loss": 1.0}],
            scaler_source=scaler,
            kind="best",
        )
        with lock:
            order.append(f"return{rank}")
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, [0, 1]))
    generation = results[0]
    assert results[1] is None
    assert {"rank-0.pt", "rank-1.pt", "scaler.pkl", "training_history.csv"}.issubset(
        generation.files
    )
    assert order.index("publish") < order.index("return1")
    event = decode(journal.bootstrap.layout.journal(1).read_bytes().splitlines()[-1])
    marker = generation.path / "COMPLETE.json"
    assert event.type == "artifact"
    assert event.payload["sha256"] == hashlib.sha256(marker.read_bytes()).hexdigest()
    assert event.payload["size"] == marker.stat().st_size
    selection = json.loads((generation.path / "checkpoint_selection.json").read_text())
    assert selection["source_seq"] == event.seq - 1 == 0
    assert journal.last_sequence == event.seq


def test_partial_copy_keeps_previous_and_announces_nothing(publication, monkeypatch):
    journal, scaler, metadata = publication
    path = journal.bootstrap.layout.run_dir / "best_checkpoint"

    def save_state(directory, **kwargs):
        path.mkdir(exist_ok=True)
        (path / "model.pt").write_bytes(b"model")

    accelerator = SimpleNamespace(
        is_main_process=True, save_state=save_state, wait_for_everyone=lambda: None
    )
    options = {
        "metadata": metadata,
        "history": [{"epoch": 1}],
        "scaler_source": scaler,
        "kind": "best",
    }
    prior = checkpointing.save_normal_checkpoint(accelerator, path, journal, **options)
    before = journal.bootstrap.layout.journal(1).read_bytes()
    original_copy = generations._copy_file
    copies = 0

    def fail_copy(source, target):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError("disk full")
        return original_copy(source, target)

    monkeypatch.setattr(generations, "_copy_file", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        checkpointing.save_normal_checkpoint(accelerator, path, journal, **options)
    assert journal.bootstrap.layout.journal(1).read_bytes() == before
    assert [
        item.generation for item in published_generations(journal.bootstrap.layout)
    ] == [prior.generation]
    assert not (
        journal.bootstrap.layout.checkpoints / "gen-000002/COMPLETE.json"
    ).exists()


def test_emergency_has_no_collectives_and_truncates_history(publication):
    journal, scaler, metadata = publication

    def forbidden(*args, **kwargs):
        pytest.fail("Emergency save must never use collectives")

    accelerator = SimpleNamespace(
        is_main_process=True,
        unwrap_model=lambda model: model,
        save_state=forbidden,
        wait_for_everyone=forbidden,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = journal.bootstrap.layout.run_dir / "interrupted_checkpoint"
    generation = checkpointing.save_emergency_checkpoint(
        accelerator,
        model,
        optimizer,
        scheduler,
        path,
        journal,
        metadata=metadata,
        history=[{"epoch": 1}, {"epoch": 2}],
        scaler_source=scaler,
    )
    import pandas as pd

    assert pd.read_csv(generation.path / "training_history.csv")["epoch"].tolist() == [
        1
    ]
    assert {"model_weights.pth", "optimizer.pt", "scheduler.pt"}.issubset(
        generation.files
    )
    assert (
        json.loads((generation.path / "checkpoint_selection.json").read_text())[
            "source_seq"
        ]
        == 0
    )
    accelerator.is_main_process = False
    assert (
        checkpointing.save_emergency_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            path,
            journal,
            metadata=metadata,
            history=[],
            scaler_source=scaler,
        )
        is None
    )
