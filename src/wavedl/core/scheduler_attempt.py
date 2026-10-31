"""Fail-closed startup rendezvous for one scheduler execution generation.

This foundation does not enable scheduler requeue or modify daemon execution
state. Empty/incomplete claims are never reclaimed, even after their owner dies.
Host and execution views require an exact declared storage mapping;
working_directory alone is not authority. This is not mount-visibility proof.
Timeouts bound rendezvous polling, not the wall-clock duration of filesystem
syscalls. A hard syscall deadline requires external process supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from uuid import UUID, uuid4

from wavedl.core.layout import (
    OperationKind,
    RunLayout,
    allocate_attempt,
    allocate_attempt_in_directory,
)
from wavedl.core.manifest import Manifest, RuntimeSelection, StorageMapping
from wavedl.core.protocol import strict_loads
from wavedl.core.safe_files import open_directory


class IncompleteSchedulerClaim(TimeoutError):
    """A startup claim has no complete binding; do not recycle its attempt."""


def _integer(value, minimum):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError("Invalid scheduler integer")


def _windows_scheduler_path(path: str | Path) -> PureWindowsPath:
    """Compare only equivalent DOS/UNC spellings, never arbitrary devices."""
    value = PureWindowsPath(path)
    drive = value.drive
    if drive.startswith("\\\\?\\"):
        if re.fullmatch(r"\\\\\?\\[A-Za-z]:", drive):
            drive = drive[4:]
        elif drive.startswith("\\\\?\\UNC\\"):
            drive = "\\\\" + drive[8:]
        else:
            raise ValueError("Unsupported scheduler Windows namespace")
        value = PureWindowsPath(drive + "\\", *value.parts[1:])
    if not value.is_absolute():
        raise ValueError("Scheduler Windows path must be absolute")
    if re.fullmatch(r"[A-Za-z]:", value.drive):
        return value
    # A UNC root contains exactly a server and share. Device namespaces and
    # incomplete roots must not become aliases for an approved storage view.
    parts = value.drive.split("\\")
    if (
        len(parts) == 4
        and parts[:2] == ["", ""]
        and all(part not in {"", ".", "..", "?"} for part in parts[2:])
    ):
        return value
    raise ValueError("Unsupported scheduler Windows namespace")


def _scheduler_path(path: str | Path) -> Path | PureWindowsPath:
    resolved = Path(path).resolve()
    return _windows_scheduler_path(resolved) if os.name == "nt" else resolved


def validate_scheduler_view(manifest: Manifest, path: Path) -> str:
    """Validate the declared host-to-execution relationship before using a view."""
    host = _scheduler_path(manifest.run_directory)
    execution = _scheduler_path(manifest.working_directory)
    roots = {}
    for mapping in manifest.storage_map:
        root = _scheduler_path(mapping.host_root)
        if root in roots:
            raise ValueError("Ambiguous normalized scheduler storage mapping")
        roots[root] = _scheduler_path(mapping.execution_root)
    expected = host
    for root in sorted(roots, key=lambda item: len(item.parts), reverse=True):
        if host.is_relative_to(root):
            expected = roots[root] / host.relative_to(root)
            break
    if os.name != "nt":
        expected = expected.resolve()
    if expected != execution:
        raise ValueError("Scheduler execution view lacks an exact declared mapping")
    actual = _scheduler_path(path)
    if actual == host:
        return "host"
    if actual == execution:
        return "execution"
    raise ValueError("Scheduler directory is outside the approved storage views")


@dataclass(frozen=True)
class SchedulerIdentity:
    run_id: str
    request_id: str
    cluster: str
    job_id: str
    owner: str
    restart_count: int

    def __post_init__(self):
        for value in (self.run_id, self.request_id):
            if not isinstance(value, str) or str(UUID(value)) != value:
                raise ValueError("Scheduler identity requires canonical UUIDs")
        for value in (self.cluster, self.owner):
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", value
            ):
                raise ValueError("Invalid scheduler cluster/owner token")
        if not isinstance(self.job_id, str) or not re.fullmatch(
            r"[1-9][0-9]{0,18}", self.job_id
        ):
            raise ValueError("Invalid scheduler job identity")
        _integer(int(self.job_id), 1)
        _integer(self.restart_count, 0)

    @property
    def claim_name(self):
        return hashlib.sha256(_encode(asdict(self))).hexdigest()


@dataclass(frozen=True)
class SchedulerBinding:
    run_id: str
    request_id: str
    cluster: str
    job_id: str
    owner: str
    restart_count: int
    attempt: int

    def __post_init__(self):
        SchedulerIdentity(
            **{key: value for key, value in asdict(self).items() if key != "attempt"}
        )
        _integer(self.attempt, 1)


def _encode(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )


def _strict_object(raw):
    # Retain the shared finite-number decoder, then reject duplicate keys at
    # every object depth rather than accepting json.loads' last-value rule.
    value = strict_loads(raw)

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate scheduler JSON field")
            result[key] = item
        return result

    json.loads(raw, object_pairs_hook=unique)
    return value


def _read(directory, name, limit=4096):
    with directory.open_file(name) as file:
        file.verify_private_owner()
        if file.stat().st_size > limit:
            raise ValueError("Scheduler receipt exceeds byte bound")
        raw = file.read(limit)
        file.guard()
    directory.guard()
    value = _strict_object(raw)
    if not isinstance(value, dict):
        raise ValueError("Scheduler receipt must be an object")
    return value


def _mkdir(directory, name):
    directory.guard()
    if os.name == "posix":
        os.mkdir(name, mode=0o700, dir_fd=directory.handle)
        os.fsync(directory.handle)
    else:
        raise OSError("Scheduler publication requires POSIX")
    directory.guard()


def _publish(directory, name, content):
    """No-replace publication relative to a private pinned parent."""
    directory.guard()
    if os.name != "posix":
        raise OSError("Scheduler publication requires POSIX")
    else:
        temporary = f".{name}-{uuid4()}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory.handle,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(
                temporary,
                name,
                src_dir_fd=directory.handle,
                dst_dir_fd=directory.handle,
                follow_symlinks=False,
            )
            os.fsync(directory.handle)
        finally:
            os.unlink(temporary, dir_fd=directory.handle)
    directory.guard()


def _validate_binding(value, identity, initial_attempt):
    if set(value) != set(SchedulerBinding.__dataclass_fields__):
        raise ValueError("Invalid scheduler binding fields")
    binding = SchedulerBinding(**value)
    if {
        key: value for key, value in asdict(binding).items() if key != "attempt"
    } != asdict(identity):
        raise ValueError("Scheduler binding identity mismatch")
    if (
        initial_attempt is not None
        and identity.restart_count == 0
        and binding.attempt != initial_attempt
    ):
        raise ValueError("Scheduler binding differs from reserved initial attempt")
    return binding


@contextmanager
def _claims(root, attempts, manifest, identity, initial_attempt, *, create):
    """Hold the exact submission authority and all claim ancestry until done."""
    with ExitStack() as stack:
        ancestry = []
        if initial_attempt is None:
            from wavedl.core.submission import (
                open_submission_in_directory,
            )

            request = _verify_writer_submission(root, manifest, identity)
            selected = stack.enter_context(
                open_submission_in_directory(root, request, writable=create)
            )
            parent, name = selected, "launchers"
            ancestry = [selected]
        else:
            with attempts.open_directory(str(initial_attempt)) as initial:
                initial.verify_private_owner()
                submission = _read(initial, "submission.json")
                if (
                    set(submission) != {"request_id", "attempt", "owner", "created_at"}
                    or submission["request_id"] != identity.request_id
                    or type(submission["attempt"]) is not int
                    or submission["attempt"] != initial_attempt
                    or submission["owner"] != identity.owner
                    or not isinstance(submission["created_at"], str)
                    or not submission["created_at"]
                ):
                    raise ValueError(
                        "Scheduler identity differs from durable submission intent"
                    )
            parent, name = root, "scheduler_attempts"
        if create:
            try:
                _mkdir(parent, name)
            except FileExistsError:
                pass
        claims = stack.enter_context(
            parent.open_directory(name, writable=create and initial_attempt is None)
        )
        claims.verify_private_owner()
        yield claims
        for directory in ancestry:
            if create:
                directory.sync()
            directory.guard()


def _verify_writer_submission(root, manifest, identity):
    """Verify writer authority before creating even the empty attempts parent."""
    from wavedl.core.submission import (
        MAX_REQUEST_BYTES,
        SubmissionIdentity,
        open_submission_in_directory,
    )

    with root.open_directory("submissions") as submissions:
        submissions.verify_private_owner(confidential=True)
        with submissions.open_directory(identity.request_id) as selected:
            selected.verify_private_owner(confidential=True)
            value = _read(selected, "request.json", MAX_REQUEST_BYTES)
            if (
                set(value)
                != {"schema_version", *SubmissionIdentity.__dataclass_fields__}
                or type(value["schema_version"]) is not int
                or value["schema_version"] != 1
            ):
                raise ValueError("Invalid writer submission fields")
            try:
                request = SubmissionIdentity(
                    **{
                        key: item
                        for key, item in value.items()
                        if key != "schema_version"
                    }
                )
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid writer submission identity") from error
            if (
                request.run_id != identity.run_id
                or request.request_id != identity.request_id
                or request.backend != "slurm"
                or request.owner != identity.owner
                or request.run_directory != manifest.run_directory
                or request.plan_sha256 is None
            ):
                raise ValueError("Scheduler identity differs from writer submission")
            with open_submission_in_directory(root, request):
                root.guard()
            return request


def read_scheduler_binding(
    run_directory: Path, identity: SchedulerIdentity, *, initial_attempt: int | None
) -> SchedulerBinding:
    """Read complete matching private startup proof without allocating or writing.

    A matching binding is authority only for the verified declared POSIX view.
    Missing/incomplete proof is refused, never inferred from scheduler counters.
    ``initial_attempt=None`` explicitly selects writer allocation and requires
    its immutable submission namespace; it cannot reinterpret legacy intent.
    """
    if initial_attempt is not None:
        _integer(initial_attempt, 1)
    if os.name != "posix":
        raise OSError("Scheduler startup binding requires POSIX")
    path = Path(run_directory)
    if not path.is_absolute():
        raise ValueError("Scheduler run path must be absolute")
    with open_directory(path) as root:
        return read_scheduler_binding_in_directory(
            root, identity, initial_attempt=initial_attempt
        )


def read_scheduler_binding_in_directory(
    root, identity: SchedulerIdentity, *, initial_attempt: int | None
) -> SchedulerBinding:
    """Verify startup proof under the caller's held root without reopening it."""
    if not isinstance(identity, SchedulerIdentity):
        raise ValueError("Typed scheduler identity required")
    if initial_attempt is not None:
        _integer(initial_attempt, 1)
    if os.name != "posix":
        raise OSError("Scheduler startup binding requires POSIX")
    root.verify_private_owner()
    manifest = _read_manifest(root)
    validate_scheduler_view(manifest, root.path)
    if manifest.run_id != identity.run_id:
        raise ValueError("Scheduler identity differs from anchored manifest")
    with root.open_directory("attempts") as attempts:
        attempts.verify_private_owner()
        with _claims(
            root, attempts, manifest, identity, initial_attempt, create=False
        ) as claims:
            claims.verify_private_owner()
            with claims.open_directory(identity.claim_name) as claim:
                claim.verify_private_owner()
                value = _read(claim, "binding.json")
                binding = _validate_binding(value, identity, initial_attempt)
            with attempts.open_directory(str(binding.attempt)) as selected:
                selected.verify_private_owner()
                if (
                    _validate_binding(
                        _read(selected, "scheduler_identity.json"),
                        identity,
                        initial_attempt,
                    )
                    != binding
                ):
                    raise ValueError("Attempt scheduler binding mismatch")
        attempts.guard()
    root.guard()
    return binding


def scheduler_environment_identity(environment):
    """Parse opt-in full scheduler identity; ambient Slurm alone changes nothing."""
    gate = environment.get("WAVEDL_SCHEDULER_BINDING")
    if gate is None:
        return None
    if gate != "1":
        raise ValueError("Invalid scheduler startup opt-in")
    writer = environment.get("WAVEDL_ATTEMPT_ALLOCATION") == "writer"
    allocation = environment.get("WAVEDL_ATTEMPT_ALLOCATION")
    if allocation not in {None, "writer"}:
        raise ValueError("Invalid scheduler allocation mode")
    initial = environment.get("WAVEDL_SCHEDULER_INITIAL_ATTEMPT", "")
    if writer and initial != "":
        raise ValueError("Writer allocation cannot consume a reserved attempt")
    restart = environment.get("SLURM_RESTART_COUNT", "0")
    numbers = [("restart", restart, 0)]
    if not writer:
        numbers.insert(0, ("initial attempt", initial, 1))
    for name, value, minimum in numbers:
        if (
            not isinstance(value, str)
            or not value.isdecimal()
            or str(int(value)) != value
        ):
            raise ValueError(f"Invalid scheduler {name}")
        _integer(int(value), minimum)
    identity = SchedulerIdentity(
        environment.get("WAVEDL_RUN_ID"),
        environment.get("WAVEDL_REQUEST_ID"),
        environment.get("WAVEDL_SCHEDULER_CLUSTER"),
        environment.get("SLURM_JOB_ID"),
        environment.get("WAVEDL_SCHEDULER_OWNER"),
        int(restart),
    )
    return identity, None if writer else int(initial)


def coordinate_attempt(
    run_directory: Path,
    identity: SchedulerIdentity,
    *,
    initial_attempt: int | None,
    coordinator: bool = True,
    timeout_seconds: float = 30,
    poll_seconds: float = 0.02,
) -> SchedulerBinding:
    """Elect one allocator; all other participants await its immutable receipt.

    A positive initial attempt retains legacy reservation semantics. ``None``
    requires immutable writer submission authority and allocates even at restart
    zero. Neither mode enables scheduler requeue or activates daemon state.
    """
    if initial_attempt is not None:
        _integer(initial_attempt, 1)
    if os.name != "posix":
        raise OSError("Scheduler startup coordination requires POSIX")
    if type(coordinator) is not bool:
        raise ValueError("Invalid coordinator role")
    for value, maximum in ((timeout_seconds, 300), (poll_seconds, 1)):
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or not 0 < value <= maximum
        ):
            raise ValueError("Invalid scheduler rendezvous deadline")
    path = Path(run_directory)
    if not path.is_absolute():
        raise ValueError("Scheduler run path must be absolute")
    deadline = time.monotonic() + timeout_seconds
    with open_directory(path, writable=initial_attempt is None) as root:
        root.verify_private_owner()
        manifest = _read_manifest(root)
        validate_scheduler_view(manifest, root.path)
        if manifest.run_id != identity.run_id:
            raise ValueError("Scheduler identity differs from anchored manifest")
        if initial_attempt is None:
            _verify_writer_submission(root, manifest, identity)
            try:
                _mkdir(root, "attempts")
            except FileExistsError:
                pass
        with root.open_directory(
            "attempts", writable=initial_attempt is None
        ) as attempts:
            attempts.verify_private_owner()
            with _claims(
                root, attempts, manifest, identity, initial_attempt, create=True
            ) as claims:
                claims.verify_private_owner()
                elected = False
                if coordinator:
                    try:
                        _mkdir(claims, identity.claim_name)
                        elected = True
                    except FileExistsError:
                        pass
                while True:
                    try:
                        with claims.open_directory(
                            identity.claim_name, writable=initial_attempt is None
                        ) as claim:
                            claim.verify_private_owner()
                            if elected:
                                root.guard()
                                if initial_attempt is None:
                                    attempt = allocate_attempt_in_directory(root)
                                elif identity.restart_count == 0:
                                    attempt = initial_attempt
                                else:
                                    attempt = allocate_attempt(
                                        RunLayout(root.path, identity.run_id),
                                        attempts_anchor=attempts,
                                    )
                                attempts.guard()
                                root.guard()
                                binding = SchedulerBinding(
                                    **asdict(identity), attempt=attempt
                                )
                                raw = _encode(asdict(binding))
                                with attempts.open_directory(
                                    str(attempt), writable=initial_attempt is None
                                ) as selected:
                                    selected.verify_private_owner()
                                    _publish(selected, "scheduler_identity.json", raw)
                                _publish(claim, "binding.json", raw)
                                elected = False
                            value = _read(claim, "binding.json")
                            binding = _validate_binding(
                                value, identity, initial_attempt
                            )
                            with attempts.open_directory(
                                str(binding.attempt)
                            ) as selected:
                                selected.verify_private_owner()
                                if (
                                    _validate_binding(
                                        _read(selected, "scheduler_identity.json"),
                                        identity,
                                        initial_attempt,
                                    )
                                    != binding
                                ):
                                    raise ValueError(
                                        "Attempt scheduler binding mismatch"
                                    )
                                # Readers can see link publication before the
                                # publisher's directory fsync. Each participant
                                # independently flushes all binding ancestry.
                                os.fsync(selected.handle)
                                selected.guard()
                            for directory in (attempts, claim, claims, root):
                                os.fsync(directory.handle)
                                directory.guard()
                            root.guard()
                            return binding
                    except FileNotFoundError:
                        if elected:
                            raise
                    if time.monotonic() >= deadline:
                        raise IncompleteSchedulerClaim(
                            "Scheduler startup binding unavailable; claim retained"
                        )
                    time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def _read_manifest(root):
    with root.open_file("manifest.json") as file:
        file.verify_private_owner()
        if file.stat().st_size > 1024 * 1024:
            raise ValueError("Manifest exceeds byte bound")
        value = _strict_object(file.read(1024 * 1024))
        file.guard()
    root.guard()
    if not isinstance(value, dict):
        raise ValueError("Invalid scheduler manifest")
    try:
        return Manifest(
            **(
                value
                | {
                    "operation": OperationKind(value["operation"]),
                    "runtime": RuntimeSelection(**value["runtime"]),
                    "storage_map": tuple(
                        StorageMapping(**mapping) for mapping in value["storage_map"]
                    ),
                    "path_fields": tuple(value["path_fields"]),
                }
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid scheduler manifest") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    for name in ("run-id", "request-id", "cluster", "job-id", "owner"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--restart-count", type=int, required=True)
    allocation = parser.add_mutually_exclusive_group(required=True)
    allocation.add_argument("--initial-attempt", type=int)
    allocation.add_argument(
        "--writer",
        action="store_true",
        help="Allocate at startup from immutable submission-scoped authority",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--follower", action="store_true")
    args = parser.parse_args(argv)
    try:
        identity = SchedulerIdentity(
            args.run_id,
            args.request_id,
            args.cluster,
            args.job_id,
            args.owner,
            args.restart_count,
        )
        binding = coordinate_attempt(
            args.run_directory,
            identity,
            initial_attempt=None if args.writer else args.initial_attempt,
            coordinator=not args.follower,
            timeout_seconds=args.timeout_seconds,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"Scheduler startup refused: {type(error).__name__}\n")
    print(_encode(asdict(binding)).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
