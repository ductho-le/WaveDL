"""Actual training runtime provenance and lifecycle, with no daemon dependency."""

from __future__ import annotations

import os
import platform
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

from wavedl import __version__
from wavedl.core.events import ErrorPayload, ExitPayload, HelloPayload, StatePayload
from wavedl.core.live_push import LivePushClient, LivePushConfig
from wavedl.core.manifest import RuntimeSelection
from wavedl.core.training_cli import training_defaults
from wavedl.journal import RunJournal, _read_manifest, bootstrap_run


def wait_for_workers(accelerator):
    """Keep CPU barriers device-free even on hosts with an MPS accelerator."""
    import torch.distributed as dist

    # Accelerate passes device_ids for MULTI_CPU too. Recent PyTorch interprets
    # those IDs using the host accelerator, selecting MPS on Apple Silicon.
    if dist.is_initialized() and accelerator.device.type == "cpu":
        dist.barrier()
    else:
        accelerator.wait_for_everyone()


def open_training_runtime(args, accelerator):
    """Bootstrap once; direct distributed execution broadcasts rank zero identity."""
    import torch.distributed as dist

    rank = accelerator.process_index
    main = accelerator.is_main_process
    output = Path(args.output_dir).resolve()
    environment = dict(os.environ)
    injected = any(
        name in environment
        for name in ("WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT")
    )
    if injected:
        # Selection describes the approved request, not an inferred replacement.
        attempt = environment.get("WAVEDL_ATTEMPT", "")
        request = output / "attempts" / attempt / "runtime_request.json"
        runtime = _read_manifest(
            request if request.exists() else output / "manifest.json"
        ).runtime
    else:
        runtime = RuntimeSelection(sys.executable, "existing", "auto", {})
    kwargs = {
        "resolved_config": dict(vars(args)),
        "runtime": runtime,
        "approved_defaults": training_defaults(),
    }
    if not injected and accelerator.num_processes > 1:
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed identity requires an initialized process group"
            )
        identity = [None]
        if main:
            bootstrap = bootstrap_run(
                output, environment=environment | {"WORLD_SIZE": "1"}, **kwargs
            )
            identity[0] = bootstrap.environment
        dist.broadcast_object_list(identity, src=0)
        environment.update(identity[0])
        # All ranks validate the same immutable request and attempt.
    wait_for_workers(accelerator)
    bootstrap = bootstrap_run(
        output, environment=environment, is_main_process=main, rank=rank, **kwargs
    )
    device = accelerator.device.type
    allocation = []
    if device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            allocation = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            import torch

            allocation = [str(index) for index in range(torch.cuda.device_count())]
    elif device == "mps":
        allocation = ["mps"]
    hello = HelloPayload(
        1,
        1,
        __version__,
        platform.python_version(),
        device,
        (
            ["cross_validation", "lifecycle"]
            if args.cv > 0
            else ["train", "epoch_metrics", "generation_resume"]
        ),
        sys.executable,
        container=runtime.container,
        allocated_devices=allocation,
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
    )
    push_client = None
    if main:
        try:
            push_config = LivePushConfig.from_environment(os.environ)
            if push_config is not None:
                push_client = LivePushClient(push_config)
        except (ValueError, RuntimeError):
            # Misconfigured optional delivery never prevents authoritative IO.
            pass
    journal = None
    try:
        journal = RunJournal(
            bootstrap,
            hello=hello,
            is_main_process=main,
            rank=rank,
            push_client=push_client,
        )
        journal.log(
            f"Training runtime rank {rank}: {device}, {sys.executable}", stream="system"
        )
        journal.emit("state", StatePayload("starting"))
    except BaseException as primary:
        # Lifecycle ownership transfers only after all startup writes succeed.
        # RunJournal cleans up a failed constructor, but an optional push client
        # can also exist before construction reaches that cleanup scope.
        try:
            if journal is not None:
                journal.close()
            elif push_client is not None:
                push_client.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "Runtime startup and cleanup failed", [primary, cleanup]
            ) from None
        raise
    return journal


@contextmanager
def training_lifecycle(journal):
    """Restore signal handlers and persist truthful terminal events on all paths."""
    previous = None

    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    try:
        previous = signal.signal(signal.SIGTERM, cancelled)
    except ValueError:
        # Embedders may call main in a non-main Python thread.
        pass
    try:
        yield
    except KeyboardInterrupt as exc:
        journal.log(str(exc), stream="stderr", level="warning")
        journal.emit("state", StatePayload("cancelled", str(exc)))
        journal.emit("exit", ExitPayload(130, "cancelled", str(exc)))
        raise
    except BaseException as exc:
        journal.log(str(exc), stream="stderr", level="error")
        journal.emit("error", ErrorPayload("TRAINING_FAILED", str(exc)))
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
