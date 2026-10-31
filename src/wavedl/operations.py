"""Dependency-light provenance and lifecycle for test and plot execution."""

from __future__ import annotations

import os
import platform
import signal
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from wavedl import __version__
from wavedl.core.events import (
    ArtifactPayload,
    ErrorPayload,
    ExitPayload,
    HelloPayload,
    StatePayload,
)
from wavedl.core.generations import publish_generation
from wavedl.core.inference_cli import inference_defaults
from wavedl.core.layout import OperationKind
from wavedl.core.live_push import LivePushClient, LivePushConfig
from wavedl.core.manifest import RuntimeSelection
from wavedl.core.persistence import sync_directory
from wavedl.core.safe_files import open_directory
from wavedl.journal import RunJournal, _read_manifest, bootstrap_run


def operation_selection(args):
    """Read declared selection without treating it as actual execution evidence."""
    output = Path(args.output_dir).resolve()
    keys = ("WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT")
    supplied = [key in os.environ for key in keys]
    if any(supplied):
        if not all(supplied):
            raise ValueError("Incomplete launcher runtime identity")
        attempt = os.environ[keys[2]]
        if not attempt.isdecimal() or str(int(attempt)) != attempt:
            raise ValueError("Invalid WAVEDL_ATTEMPT")
        request = output / "attempts" / attempt / "runtime_request.json"
        manifest = _read_manifest(
            request if request.exists() else output / "manifest.json"
        )
        if manifest.operation not in {OperationKind.TEST, OperationKind.PLOT}:
            raise ValueError("Inference requires a test or plot operation")
        return manifest.operation, manifest.runtime
    operation = OperationKind(getattr(args, "operation", "test"))
    # Existing training artifacts are inputs, never an inference output run.
    if (output / "manifest.json").exists():
        manifest = _read_manifest(output / "manifest.json")
        if manifest.operation != operation:
            output = output / "operations" / f"{operation}-{uuid4()}"
            args.output_dir = str(output)
            print(f"Operation output directory: {output}", flush=True)
    return operation, RuntimeSelection(
        sys.executable, "existing", getattr(args, "device", "auto"), {}
    )


def open_operation_runtime(args, *, operation, runtime, device, allocated_devices):
    """Bootstrap a single-process operation and announce actual execution."""
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) != 0
    ):
        raise ValueError("Test and plot operations require single-process execution")
    bootstrap = bootstrap_run(
        Path(args.output_dir),
        resolved_config=dict(vars(args)),
        runtime=runtime,
        operation=operation,
        approved_defaults=inference_defaults(),
    )
    client = None
    try:
        config = LivePushConfig.from_environment(os.environ)
        if config is not None:
            client = LivePushClient(config)
    except (ValueError, RuntimeError):
        pass
    journal = RunJournal(
        bootstrap,
        hello=HelloPayload(
            1,
            1,
            __version__,
            platform.python_version(),
            device,
            [operation.value, "evaluation_metrics", "generation_checkpoints"],
            sys.executable,
            container=runtime.container,
            allocated_devices=allocated_devices,
            prefix=sys.prefix,
            base_prefix=sys.base_prefix,
        ),
        push_client=client,
    )
    journal.emit("state", StatePayload("starting"))
    journal._operation_artifacts = {}
    return journal


@contextmanager
def operation_lifecycle(journal):
    previous = None

    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    try:
        previous = signal.signal(signal.SIGTERM, cancelled)
    except ValueError:
        pass
    try:
        journal.emit("state", StatePayload("running"))
        yield
        publish_operation_outputs(journal)
    except KeyboardInterrupt as exc:
        journal.emit("state", StatePayload("cancelled", str(exc)))
        journal.emit("exit", ExitPayload(130, "cancelled", str(exc)))
        raise
    except BaseException as exc:
        journal.emit("error", ErrorPayload("OPERATION_FAILED", str(exc)))
        journal.emit("state", StatePayload("failed", str(exc)))
        journal.emit("exit", ExitPayload(1, "failed", str(exc)))
        raise
    else:
        journal.emit("state", StatePayload("completed"))
        journal.emit("exit", ExitPayload(0, "completed"))
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        journal.close()


def emit_operation_artifact(journal, path, *, kind, label=""):
    """Register operation-owned output; publication precedes announcements."""
    path = Path(path)
    path = path.parent.resolve() / path.name
    root = journal.bootstrap.layout.run_dir
    if not path.is_relative_to(root):
        journal.log(f"External output saved: {path}", stream="system")
        return
    journal._operation_artifacts[path.relative_to(root).as_posix()] = (kind, label)


def publish_operation_outputs(journal):
    """Publish only registered quiescent outputs, retaining legacy flat copies."""
    artifacts = journal._operation_artifacts
    if not artifacts:
        return
    layout = journal.bootstrap.layout
    layout.checkpoints.mkdir(exist_ok=True)
    sync_directory(layout.run_dir)
    with TemporaryDirectory(
        prefix=".operation-staging-", dir=layout.run_dir
    ) as temporary:
        staging = Path(temporary)
        with open_directory(layout.run_dir) as root:
            for name in artifacts:
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid registered output path")
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with ExitStack() as anchors:
                    source_directory = root
                    for member in relative.parts[:-1]:
                        source_directory = anchors.enter_context(
                            source_directory.open_directory(member)
                        )
                    source = anchors.enter_context(
                        source_directory.open_file(relative.name)
                    )
                    with destination.open("xb") as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                if relative.suffix.lower() == ".onnx":
                    from wavedl.utils.onnx_export import validate_self_contained_onnx

                    validate_self_contained_onnx(destination)
        published = publish_generation(
            layout, staging, attempt=journal.bootstrap.attempt, step=0
        )
    for name, (kind, label) in artifacts.items():
        record = published.files[name]
        path = (published.path / name).relative_to(layout.run_dir).as_posix()
        journal.emit(
            "artifact",
            ArtifactPayload(
                kind,
                path,
                record.sha256,
                record.size,
                label,
                generation=published.generation,
                step=0,
            ),
        )
