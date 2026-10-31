"""Separate collective normal and rank-zero emergency checkpoint producers."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path

from wavedl.core.events import ArtifactPayload
from wavedl.core.generations import publish_generation
from wavedl.core.persistence import sync_directory
from wavedl.runtime import wait_for_workers


def unwrap_checkpoint_model(accelerator, model):
    """Retain compatibility with distributed and compiled model wrappers."""
    try:
        return accelerator.unwrap_model(model)
    except KeyError:
        unwrapped = model.module if hasattr(model, "module") else model
        return getattr(unwrapped, "_orig_mod", unwrapped)


def _metadata(path, *, metadata, history, scaler_source, kind, source_seq):
    import pandas as pd

    with (path / "training_meta.pkl").open("wb") as stream:
        pickle.dump(metadata, stream)
    shutil.copy2(scaler_source, path / "scaler.pkl")
    rows = [row for row in history if row.get("epoch", 0) <= metadata["epoch"]]
    pd.DataFrame(rows if rows else {"epoch": []}).to_csv(
        path / "training_history.csv", index=False
    )
    if type(source_seq) is not int or source_seq < 0:
        raise ValueError("Checkpoint source sequence must be a nonnegative integer")
    (path / "checkpoint_selection.json").write_text(
        json.dumps({"schema_version": 1, "kind": kind, "source_seq": source_seq})
        + "\n",
        encoding="utf-8",
    )


def _publish(path, journal, *, step, kind):
    layout = journal.bootstrap.layout
    layout.checkpoints.mkdir(exist_ok=True)
    # The first publication must persist the new checkpoints parent entry too.
    sync_directory(layout.run_dir)
    generation = publish_generation(
        layout, path, attempt=journal.bootstrap.attempt, step=step
    )
    marker = generation.path / "COMPLETE.json"
    content = marker.read_bytes()
    journal.emit(
        "artifact",
        ArtifactPayload(
            "checkpoint",
            marker.relative_to(layout.run_dir).as_posix(),
            hashlib.sha256(content).hexdigest(),
            len(content),
            label=kind,
            generation=generation.generation,
            step=step,
        ),
    )
    return generation


def save_normal_checkpoint(
    accelerator, path, journal, *, metadata, history, scaler_source, kind
):
    """All ranks save; rank zero publishes only after every writer is quiescent."""
    import torch.distributed as dist

    if kind not in {"best", "periodic"}:
        raise ValueError("Normal checkpoint kind must be best or periodic")
    path = Path(path)
    accelerator.save_state(str(path), safe_serialization=False)
    wait_for_workers(accelerator)
    result = None
    failure = None
    if accelerator.is_main_process:
        try:
            _metadata(
                path,
                metadata=metadata,
                history=history,
                scaler_source=scaler_source,
                kind=kind,
                source_seq=journal.last_sequence,
            )
            result = _publish(path, journal, step=metadata["epoch"], kind=kind)
        except BaseException as exc:
            failure = exc
    # Share publication failure before the final barrier so a storage failure
    # cannot strand followers while rank zero exits its lifecycle.
    status = [None if failure is None else str(failure)]
    if dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        if failure is not None:
            raise failure
        raise RuntimeError(f"Checkpoint publication failed on rank zero: {status[0]}")
    wait_for_workers(accelerator)
    return result


def save_emergency_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    path,
    journal,
    *,
    metadata,
    history,
    scaler_source,
):
    """Rank-zero-only save: never call save_state, barriers, or collectives."""
    import torch

    if not accelerator.is_main_process:
        return None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        unwrap_checkpoint_model(accelerator, model).state_dict(),
        path / "model_weights.pth",
    )
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    _metadata(
        path,
        metadata=metadata,
        history=history,
        scaler_source=scaler_source,
        kind="interrupted",
        source_seq=journal.last_sequence,
    )
    return _publish(path, journal, step=metadata["epoch"], kind="interrupted")


def publish_cv_outputs(journal, output_dir, *, folds):
    """Publish cross-validation fold outputs and summary as immutable generations."""
    layout = journal.bootstrap.layout
    root = Path(output_dir)
    layout.checkpoints.mkdir(exist_ok=True)
    sync_directory(layout.run_dir)
    published = []
    for number in range(1, folds + 1):
        fold = root / f"fold_{number}"
        if not fold.is_dir() or not {
            "model_weights.pth",
            "scaler.pkl",
            "training_meta.pkl",
        } <= {item.name for item in fold.iterdir() if item.is_file()}:
            raise ValueError("Cross-validation fold is incomplete")
        # A reused legacy fold directory may contain obsolete or unrelated
        # files. Only the outputs produced by this CV operation are authority.
        with tempfile.TemporaryDirectory(dir=root) as staging:
            staging_path = Path(staging)
            for name in (
                "model.pth",
                "model_weights.pth",
                "scaler.pkl",
                "training_meta.pkl",
            ):
                shutil.copy2(fold / name, staging_path / name)
            (staging_path / "checkpoint_selection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "cross_validation_fold",
                        "fold": number,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            generation = publish_generation(
                layout, staging_path, attempt=journal.bootstrap.attempt, step=number
            )
        marker = generation.path / "COMPLETE.json"
        content = marker.read_bytes()
        journal.emit(
            "artifact",
            ArtifactPayload(
                "checkpoint",
                marker.relative_to(layout.run_dir).as_posix(),
                hashlib.sha256(content).hexdigest(),
                len(content),
                label=fold.name,
                generation=generation.generation,
                step=number,
            ),
        )
        published.append(generation)
    summary_files = [root / "cv_summary.json", root / "cv_results.csv"]
    if all(path.is_file() for path in summary_files):
        with tempfile.TemporaryDirectory(dir=root) as staging:
            staging_path = Path(staging)
            for path in summary_files:
                shutil.copy2(path, staging_path / path.name)
            (staging_path / "checkpoint_selection.json").write_text(
                json.dumps({"schema_version": 1, "kind": "cross_validation_summary"})
                + "\n",
                encoding="utf-8",
            )
            generation = publish_generation(
                layout,
                staging_path,
                attempt=journal.bootstrap.attempt,
                step=len(published),
            )
            marker = generation.path / "COMPLETE.json"
            content = marker.read_bytes()
            journal.emit(
                "artifact",
                ArtifactPayload(
                    "history",
                    marker.relative_to(layout.run_dir).as_posix(),
                    hashlib.sha256(content).hexdigest(),
                    len(content),
                    label="summary",
                    generation=generation.generation,
                    step=len(published),
                ),
            )
            published.append(generation)
    if len(published) != folds + 1:
        raise ValueError("Cross-validation summary is incomplete")
    return published
