"""Resolve and propagate one runtime identity before Accelerate starts ranks."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from wavedl.core.configuration import load_config, merge_config_with_args
from wavedl.core.manifest import RuntimeSelection
from wavedl.core.persistence import write_once
from wavedl.core.safe_files import open_directory, read_regular_file
from wavedl.core.scheduler_attempt import (
    _publish,
    _strict_object,
    coordinate_attempt,
    read_scheduler_binding,
    scheduler_environment_identity,
    validate_scheduler_view,
)
from wavedl.core.training_cli import parse_training_args, training_defaults
from wavedl.journal import _manifest_from_data, _read_manifest, bootstrap_run


if TYPE_CHECKING:
    from argparse import Namespace


def prepare_runtime(
    train_args: list[str],
    launcher_args: Namespace,
    *,
    python: str,
    environment: dict[str, str],
    timeout_seconds: float = 30,
) -> tuple[list[str], dict[str, str]]:
    """Return child argv/environment without changing the launcher's globals.

    Standalone multi-node launches require a shared output directory and either
    a scheduler execution identity or a fresh caller-supplied launch UUID. A
    durable rendezvous is never overwritten or reused for a new execution.
    """
    forwarded = list(train_args)
    # The launcher consumes both spellings of its precision option.
    mixed_explicit = any(
        value.split("=", 1)[0] in {"--mixed_precision", "--precision"}
        for value in launcher_args.original_argv
    )
    precision_explicit = any(
        value == "--precision" or value.startswith("--precision=")
        for value in forwarded
    )
    if mixed_explicit and not precision_explicit:
        forwarded += ["--precision", launcher_args.mixed_precision]
    effective, parser = parse_training_args(forwarded)
    if effective.config:
        effective = merge_config_with_args(
            load_config(effective.config), effective, parser, argv=forwarded
        )
    if mixed_explicit and launcher_args.mixed_precision != effective.precision:
        raise ValueError("Launcher and training precision selections conflict")
    launcher_args.mixed_precision = effective.precision
    output = Path(effective.output_dir).expanduser().resolve()
    env = dict(environment)
    scheduler = scheduler_environment_identity(env)
    if scheduler is not None:
        identity, initial_attempt = scheduler
        binding = read_scheduler_binding(
            output, identity, initial_attempt=initial_attempt
        )
        if env.get("WAVEDL_ATTEMPT") != str(binding.attempt):
            raise ValueError("Launcher attempt differs from scheduler startup binding")
    identity_keys = ("WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT")
    if any(key in env for key in identity_keys):
        if not all(key in env for key in identity_keys):
            raise ValueError("Incomplete injected execution identity")
        for key in identity_keys[:2]:
            if str(UUID(env[key])) != env[key]:
                raise ValueError("Execution identity requires canonical UUIDs")
        attempt = env[identity_keys[2]]
        if not attempt.isdecimal() or str(int(attempt)) != attempt or int(attempt) < 1:
            raise ValueError("Invalid injected attempt")
    injected = all(key in env for key in identity_keys)
    rank = launcher_args.machine_rank
    coordination = None
    if not injected and launcher_args.num_machines > 1:
        launch_id = launcher_args.launch_id
        if not launch_id and env.get("SLURM_JOB_ID"):
            launch_id = str(
                uuid5(
                    NAMESPACE_URL,
                    "wavedl:"
                    + ":".join(
                        env.get(name, "")
                        for name in (
                            "SLURM_CLUSTER_NAME",
                            "SLURM_JOB_ID",
                            "SLURM_RESTART_COUNT",
                        )
                    ),
                )
            )
        if not launch_id or str(UUID(launch_id)) != launch_id:
            raise ValueError(
                "Standalone multi-node launch requires a fresh --launch_id UUID"
            )
        coordination = output / "launchers" / f"{launch_id}.json"
        if rank == 0:
            if os.path.lexists(coordination):
                raise ValueError(
                    "Launch identity has already been reserved; never reuse it"
                )
        else:
            started = time.monotonic()
            while not os.path.lexists(coordination):
                if time.monotonic() - started >= timeout_seconds:
                    raise TimeoutError(
                        "Main-node bootstrap did not publish runtime identity"
                    )
                time.sleep(0.02)
            receipt = json.loads(read_regular_file(coordination, limit=4096))
            if set(receipt) != set(identity_keys) or not all(
                type(value) is str for value in receipt.values()
            ):
                raise ValueError("Malformed launcher rendezvous identity")
            env.update(receipt)
            injected = True
    if injected:
        manifest = _read_manifest(output / "manifest.json")
        request = output / "attempts" / env["WAVEDL_ATTEMPT"] / "runtime_request.json"
        if os.path.lexists(request):
            manifest = _read_manifest(request)
        selection = manifest.runtime
        if rank != 0:
            receipt = (
                output / "attempts" / env["WAVEDL_ATTEMPT"] / "runtime_identity.json"
            )
            started = time.monotonic()
            while not os.path.lexists(receipt):
                if time.monotonic() - started >= timeout_seconds:
                    raise TimeoutError("Main-node runtime identity was not published")
                time.sleep(0.02)
    else:
        selection = RuntimeSelection(
            python, "existing", "cpu" if launcher_args.cpu else "auto", {}
        )
    bootstrap = bootstrap_run(
        output,
        resolved_config=vars(effective),
        runtime=selection,
        environment=env,
        is_main_process=rank == 0,
        rank=rank,
        approved_defaults=training_defaults(),
        submission_cwd=Path.cwd(),
    )
    if coordination is not None and rank == 0:
        coordination.parent.mkdir(parents=True, exist_ok=True)
        write_once(coordination, (json.dumps(bootstrap.environment) + "\n").encode())
    env.update(bootstrap.environment)
    env["WAVEDL_RUN_DIR"] = str(bootstrap.layout.run_dir)
    return forwarded, env


def _writer_scheduler_request(output, identity, environment):
    """Verify immutable writer plan and manifest before any attempt allocation."""
    from wavedl.core.submission import (
        identity_for_submission,
        read_submission_file,
    )

    if "WAVEDL_ATTEMPT" in environment:
        raise ValueError("Writer startup cannot consume an injected attempt")
    submission = identity_for_submission(
        output,
        request_id=identity.request_id,
        backend="slurm",
        owner=identity.owner,
        plan_sha256=environment.get("WAVEDL_PLAN_SHA256"),
    )
    if submission.run_id != identity.run_id or submission.plan_sha256 is None:
        raise ValueError("Writer startup differs from submission identity")
    raw = read_submission_file(output, submission, "plan.json")
    if hashlib.sha256(raw).hexdigest() != submission.plan_sha256:
        raise ValueError("Writer startup plan digest differs")
    plan = _strict_object(raw)
    if (
        not isinstance(plan, dict)
        or plan.get("target") != "slurm"
        or plan.get("allocation_mode") != "writer"
        or type(plan.get("manifest_json")) is not str
    ):
        raise ValueError("Invalid scheduler writer plan")
    if (
        json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        != raw
    ):
        raise ValueError("Scheduler writer plan is not canonical")
    try:
        resources = dict(plan.get("resources", ()))
        data = _strict_object(plan["manifest_json"])
        manifest = _manifest_from_data(data)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid scheduler writer manifest") from error
    if resources.get("cluster") != identity.cluster:
        raise ValueError("Scheduler cluster differs from approved plan")
    validate_scheduler_view(manifest, output)
    request = (
        json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    if hashlib.sha256(request).hexdigest() != submission.manifest_sha256:
        raise ValueError("Scheduler plan manifest differs from approved file")
    return request


def bind_scheduler_startup(run_directory: Path, environment: dict[str, str]):
    """Allocation coordinator runs this once before launching any rank workers."""
    scheduler = scheduler_environment_identity(environment)
    if scheduler is None:
        raise ValueError("Scheduler startup requires explicit opt-in")
    identity, initial_attempt = scheduler
    output = Path(run_directory)
    writer_request = (
        _writer_scheduler_request(output, identity, environment)
        if initial_attempt is None
        else None
    )
    binding = coordinate_attempt(output, identity, initial_attempt=initial_attempt)
    if binding.attempt != initial_attempt:
        # Keep the private directory ancestry pinned through read and publication.
        with (
            open_directory(output) as root,
            root.open_directory("attempts") as attempts,
        ):
            root.verify_private_owner()
            attempts.verify_private_owner()
            with root.open_file("manifest.json") as approved:
                approved.verify_private_owner()
                if approved.stat().st_size > 1024 * 1024:
                    raise ValueError("Scheduler manifest exceeds bound")
                original = _manifest_from_data(
                    _strict_object(approved.read(1024 * 1024))
                )
                approved.guard()
            validate_scheduler_view(original, output)
            raw = writer_request
            if initial_attempt is not None:
                with attempts.open_directory(str(initial_attempt)) as initial:
                    initial.verify_private_owner()
                    try:
                        with initial.open_file("runtime_request.json") as source:
                            source.verify_private_owner()
                            if source.stat().st_size > 1024 * 1024:
                                raise ValueError(
                                    "Initial runtime request exceeds bound"
                                )
                            raw = source.read(1024 * 1024)
                            source.guard()
                    except FileNotFoundError:
                        raw = None
            if raw is not None:
                try:
                    manifest = _manifest_from_data(_strict_object(raw))
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Invalid initial runtime request") from exc
                if manifest.run_id != identity.run_id or manifest != original:
                    raise ValueError("Initial request differs from scheduler identity")
                validate_scheduler_view(manifest, output)
                with attempts.open_directory(str(binding.attempt)) as selected:
                    selected.verify_private_owner()
                    try:
                        with selected.open_file("runtime_request.json") as target:
                            target.verify_private_owner()
                            if (
                                target.stat().st_size > 1024 * 1024
                                or target.read(1024 * 1024) != raw
                            ):
                                raise ValueError(
                                    "Scheduler request differs from initial request"
                                )
                            target.guard()
                    except FileNotFoundError:
                        try:
                            _publish(selected, "runtime_request.json", raw)
                        except FileExistsError:
                            # Another caller of the same immutable binding may
                            # publish first. Accept only identical approved bytes.
                            with selected.open_file("runtime_request.json") as target:
                                target.verify_private_owner()
                                if (
                                    target.stat().st_size > 1024 * 1024
                                    or target.read(1024 * 1024) != raw
                                ):
                                    raise ValueError(
                                        "Concurrent scheduler request differs"
                                    ) from None
                                target.guard()
                    selected.guard()
            attempts.guard()
            root.guard()
    return binding


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["scheduler-bind"])
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        binding = bind_scheduler_startup(args.run_directory, dict(os.environ))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Scheduler startup refused: {type(exc).__name__}\n")
    print(binding.attempt, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
