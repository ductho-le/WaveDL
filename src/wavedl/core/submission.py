"""Immutable pre-runtime receipts, independent of runtime attempt allocation.

Submission intent lives in submissions/<request_id>, never attempts/<n>.
All filesystem ancestry remains anchored during reads and publication. Filesystem
syscalls have no hard deadline; unsupported durable publication fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from wavedl.core.layout import OperationKind
from wavedl.core.manifest import Manifest, RuntimeSelection, StorageMapping
from wavedl.core.protocol import strict_loads
from wavedl.core.safe_files import open_directory
from wavedl.core.scheduler_attempt import validate_scheduler_view


if TYPE_CHECKING:
    from collections.abc import Iterator

    from wavedl.core.safe_files import SafeDirectory


MAX_REQUEST_BYTES = 10 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SUBMISSION_FILE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SubmissionIdentity:
    run_id: str
    request_id: str
    backend: str
    owner: str
    run_directory: str
    manifest_sha256: str
    plan_sha256: str | None = None

    def __post_init__(self):
        for value in (self.run_id, self.request_id):
            if not isinstance(value, str) or str(UUID(value)) != value:
                raise ValueError("Submission IDs must be canonical UUIDs")
        if self.backend not in {"local", "slurm"}:
            raise ValueError("Unsupported submission backend")
        if (
            not isinstance(self.owner, str)
            or re.fullmatch(
                r"[A-Za-z0-9_.-]{1,128}"
                if self.backend == "local"
                else r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}",
                self.owner,
            )
            is None
        ):
            raise ValueError("Invalid submission owner token")
        if (
            not isinstance(self.run_directory, str)
            or not Path(self.run_directory).is_absolute()
            or "\0" in self.run_directory
            or str(Path(self.run_directory)) != self.run_directory
            or ".." in Path(self.run_directory).parts
        ):
            raise ValueError("Submission run directory must be normalized and absolute")
        for name, value in (
            ("manifest", self.manifest_sha256),
            ("plan", self.plan_sha256),
        ):
            if value is None and name == "plan":
                continue
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError("Invalid submission identity digest")


def _object(raw: bytes) -> dict:
    value = strict_loads(raw)

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate submission JSON field")
            result[key] = item
        return result

    json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Submission metadata must be a JSON object")
    return value


def _read(directory: SafeDirectory, name: str, limit: int) -> bytes:
    with directory.open_file(name) as file:
        file.verify_private_owner()
        if file.stat().st_size > limit:
            raise ValueError("Submission file exceeds byte bound")
        raw = file.read(limit)
        file.guard()
    directory.guard()
    return raw


def _manifest(directory: SafeDirectory) -> tuple[Manifest, str]:
    raw = _read(directory, "manifest.json", MAX_MANIFEST_BYTES)
    data = _object(raw)
    try:
        manifest = Manifest(
            **(
                data
                | {
                    "operation": OperationKind(data["operation"]),
                    "runtime": RuntimeSelection(**data["runtime"]),
                    "storage_map": tuple(
                        StorageMapping(**item) for item in data["storage_map"]
                    ),
                    "path_fields": tuple(data["path_fields"]),
                }
            )
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Invalid submission manifest") from exc
    validate_scheduler_view(manifest, directory.path)
    return manifest, hashlib.sha256(raw).hexdigest()


def _verify(directory: SafeDirectory, identity: SubmissionIdentity) -> None:
    if not isinstance(identity, SubmissionIdentity):
        raise ValueError("Typed submission identity required")
    manifest, digest = _manifest(directory)
    if (manifest.run_id, manifest.run_directory, digest) != (
        identity.run_id,
        identity.run_directory,
        identity.manifest_sha256,
    ):
        raise ValueError("Submission identity differs from anchored manifest")


def _mkdir(directory: SafeDirectory, name: str, *, exclusive=False) -> None:
    directory.guard()
    if directory.api is not None and not directory.writable:
        raise PermissionError("Submission publication requires a writable directory")
    try:
        if os.name == "posix":
            os.mkdir(name, mode=0o700, dir_fd=directory.handle)
            directory.sync()
        else:
            # Windows SafeDirectory holds all ancestors without delete/write
            # sharing; guarded path publication cannot swap those ancestors.
            (directory.path / name).mkdir(mode=0o700)
            directory.sync()
    except FileExistsError:
        if exclusive:
            raise
    directory.guard()


def _publish(directory: SafeDirectory, name: str, content: bytes) -> None:
    directory.guard()
    if directory.api is not None and not directory.writable:
        raise PermissionError("Submission publication requires a writable directory")
    if os.name == "posix":
        temporary = f".{name}-{uuid4()}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory.handle,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            directory.guard()
            os.link(
                temporary,
                name,
                src_dir_fd=directory.handle,
                dst_dir_fd=directory.handle,
                follow_symlinks=False,
            )
            directory.sync()
        finally:
            os.unlink(temporary, dir_fd=directory.handle)
            directory.sync()
    else:
        temporary = directory.path / f".{name}-{uuid4()}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            directory.guard()
            directory.link_file(temporary.name, name)
            directory.sync()
        finally:
            temporary.unlink(missing_ok=True)
            directory.sync()
    directory.guard()


def _root_path(run_directory: Path) -> Path:
    path = Path(run_directory)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Submission access requires an absolute ordinary path")
    return path


def identity_for_submission(
    run_directory: Path,
    *,
    request_id: str,
    backend: str,
    owner: str,
    plan_sha256: str | None = None,
) -> SubmissionIdentity:
    """Derive the exact manifest-byte identity from a held declared storage view."""
    with open_directory(_root_path(run_directory)) as root:
        return identity_for_submission_directory(
            root,
            request_id=request_id,
            backend=backend,
            owner=owner,
            plan_sha256=plan_sha256,
        )


def identity_for_submission_directory(
    root: SafeDirectory,
    *,
    request_id: str,
    backend: str,
    owner: str,
    plan_sha256: str | None = None,
) -> SubmissionIdentity:
    """Derive identity using the caller's held root, never reopen its pathname."""
    root.verify_private_owner()
    manifest, digest = _manifest(root)
    root.guard()
    return SubmissionIdentity(
        manifest.run_id,
        request_id,
        backend,
        owner,
        manifest.run_directory,
        digest,
        plan_sha256,
    )


@contextmanager
def _namespace(
    run_directory, identity, *, create=False, writable=False
) -> Iterator[SafeDirectory]:
    if type(writable) is not bool:
        raise ValueError("Writable submission capability must be boolean")
    with ExitStack() as stack:
        writable = create or writable
        root = stack.enter_context(
            _open_writable_root(
                _root_path(run_directory), deadline=time.monotonic() + 5
            )
            if writable
            else open_directory(_root_path(run_directory))
        )
        root.verify_private_owner()
        _verify(root, identity)
        if create:
            _mkdir(root, "submissions")
        submissions = stack.enter_context(
            root.open_directory("submissions", writable=writable)
        )
        submissions.verify_private_owner(confidential=True)
        if create:
            _mkdir(submissions, identity.request_id, exclusive=True)
        selected = stack.enter_context(
            submissions.open_directory(identity.request_id, writable=writable)
        )
        selected.verify_private_owner(confidential=True)
        yield selected
        for directory in (selected, submissions, root):
            directory.guard()


def _open_writable_root(path, *, deadline, poll_seconds=0.02, opener=None):
    """Admit a writer before mutation; retry only native Windows sharing denial.

    The absolute monotonic deadline is never reset. This bounds admission polls,
    not filesystem syscalls. Sharing remains restrictive; after admission every
    claim/proof/publication executes exactly once under the retained capability.
    Submission/outcome writers allow five seconds for admission; Local startup
    supplies its existing overall coordination deadline instead.
    """
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
        or isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not math.isfinite(poll_seconds)
        or not 0 < poll_seconds <= 60
    ):
        raise ValueError("Finite writer admission deadline and positive poll required")
    opener = open_directory if opener is None else opener
    denial = None
    while True:
        if time.monotonic() >= deadline:
            if denial is not None:
                raise denial
            raise TimeoutError(
                "Writer root admission deadline exceeded before mutation"
            )
        try:
            return opener(path, writable=True)
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) != 32:
                raise
            denial = error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(poll_seconds, remaining))


def create_submission(run_directory: Path, identity: SubmissionIdentity) -> Path:
    """Exclusively publish request.json; an existing claim never authorizes retry."""
    raw = (
        json.dumps(
            {"schema_version": 1, **asdict(identity)}, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode()
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Submission identity exceeds byte bound")
    with _namespace(run_directory, identity, create=True) as selected:
        _publish(selected, "request.json", raw)
        return selected.path


def _verify_request(selected: SafeDirectory, identity: SubmissionIdentity) -> None:
    data = _object(_read(selected, "request.json", MAX_REQUEST_BYTES))
    if type(data.get("schema_version")) is not int or data != {
        "schema_version": 1,
        **asdict(identity),
    }:
        raise ValueError("Submission receipt differs from exact identity")


def read_submission(
    run_directory: Path, identity: SubmissionIdentity
) -> SubmissionIdentity:
    """Verify complete receipt identity, manifest digest and declared storage view."""
    with _namespace(run_directory, identity) as selected:
        _verify_request(selected, identity)
    return identity


@contextmanager
def open_submission(
    run_directory: Path, identity: SubmissionIdentity, *, writable=False
) -> Iterator[SafeDirectory]:
    """Hold verified private submission ancestry through writer coordination.

    The yielded directory is owned by this context. Consumers must retain this
    context while using anchored descendants and never transfer its handles.
    """
    with _namespace(run_directory, identity, writable=writable) as selected:
        _verify_request(selected, identity)
        yield selected


@contextmanager
def open_submission_in_directory(
    root: SafeDirectory, identity: SubmissionIdentity, *, writable: bool = False
) -> Iterator[SafeDirectory]:
    """Open a verified namespace from one caller-held root capability.

    Only descendants are owned by this context; the caller retains the root.
    Writable Windows children duplicate their held ancestry instead of opening
    a conflicting second root handle. Guards detect a substituted POSIX path.
    """
    if type(writable) is not bool:
        raise ValueError("Writable submission capability must be boolean")
    if writable and root.api is not None and not root.writable:
        raise PermissionError("Writable submission requires a writable root")
    root.verify_private_owner()
    _verify(root, identity)
    with (
        root.open_directory("submissions", writable=writable) as submissions,
        submissions.open_directory(identity.request_id, writable=writable) as selected,
    ):
        submissions.verify_private_owner(confidential=True)
        selected.verify_private_owner(confidential=True)
        _verify_request(selected, identity)
        yield selected
        for directory in (selected, submissions, root):
            directory.guard()


def _file_arguments(name: str, limit: int) -> None:
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None
    ):
        raise ValueError("Expected an ordinary submission file name")
    if name in {"request.json", "manifest.json"}:
        raise ValueError("Reserved submission file name")
    if type(limit) is not int or not 1 <= limit <= MAX_SUBMISSION_FILE_BYTES:
        raise ValueError("Invalid submission file byte bound")


def mkdir_submission_member(
    directory: SafeDirectory, name: str, *, exclusive: bool = True
) -> None:
    """Create a private claim under ancestry held by open_submission's caller."""
    _file_arguments(name, 1)
    if type(exclusive) is not bool:
        raise ValueError("exclusive must be a boolean")
    directory.verify_private_owner(confidential=True)
    _mkdir(directory, name, exclusive=exclusive)


def publish_submission_member(
    directory: SafeDirectory,
    name: str,
    content: bytes,
    *,
    limit: int = MAX_SUBMISSION_FILE_BYTES,
) -> Path:
    """Publish once in a held verified submission or its private claim child."""
    _file_arguments(name, limit)
    if not isinstance(content, bytes) or len(content) > limit:
        raise ValueError("Submission content must be bounded bytes")
    directory.verify_private_owner(confidential=True)
    _publish(directory, name, content)
    return directory.path / name


def publish_submission_file(
    run_directory: Path,
    identity: SubmissionIdentity,
    name: str,
    content: bytes,
    *,
    limit: int = MAX_SUBMISSION_FILE_BYTES,
) -> Path:
    """Publish script/fence bytes once under the verified immutable submission."""
    _file_arguments(name, limit)
    if not isinstance(content, bytes) or len(content) > limit:
        raise ValueError("Submission content must be bounded bytes")
    with _namespace(run_directory, identity, writable=True) as selected:
        _verify_request(selected, identity)
        _publish(selected, name, content)
        return selected.path / name


def read_submission_file(
    run_directory: Path,
    identity: SubmissionIdentity,
    name: str,
    *,
    limit: int = MAX_SUBMISSION_FILE_BYTES,
) -> bytes:
    """Read bounded ordinary bytes without following substituted members."""
    _file_arguments(name, limit)
    with _namespace(run_directory, identity) as selected:
        _verify_request(selected, identity)
        return _read(selected, name, limit)
