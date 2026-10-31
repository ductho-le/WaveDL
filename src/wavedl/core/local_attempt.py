"""Runtime-side Local startup allocation with immutable supervisor binding.

Submission receipts authorize allocation, not saved PIDs. Incomplete election
claims are never reclaimed. Polling deadlines do not bound filesystem syscalls;
native Windows filesystem acceptance remains an independent readiness gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from wavedl.core.layout import allocate_attempt_in_directory
from wavedl.core.protocol import strict_loads
from wavedl.core.safe_files import open_directory
from wavedl.core.submission import (
    SubmissionIdentity,
    _open_writable_root,
    mkdir_submission_member as _mkdir,
    open_submission_in_directory,
    publish_submission_member as _publish,
)


class IncompleteLocalClaim(TimeoutError):
    """A writer claim lacks complete proof; retain it without allocating again."""


@dataclass(frozen=True)
class LocalStartupIdentity:
    run_id: str
    request_id: str
    owner: str
    supervisor_id: str
    run_directory: str
    manifest_sha256: str
    plan_sha256: str

    def __post_init__(self):
        _ = self.submission_identity
        if (
            not isinstance(self.supervisor_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.supervisor_id) is None
        ):
            raise ValueError("Invalid Local supervisor identity")
        if self.plan_sha256 is None:
            raise ValueError("Local startup requires immutable plan digest")

    @property
    def submission_identity(self) -> SubmissionIdentity:
        return SubmissionIdentity(
            self.run_id,
            self.request_id,
            "local",
            self.owner,
            self.run_directory,
            self.manifest_sha256,
            self.plan_sha256,
        )


@dataclass(frozen=True)
class LocalStartupBinding(LocalStartupIdentity):
    attempt: int

    def __post_init__(self):
        super().__post_init__()
        if type(self.attempt) is not int or not 1 <= self.attempt <= 2**63 - 1:
            raise ValueError("Local startup attempt must be a positive integer")


def _encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _binding(raw, identity):
    value = strict_loads(raw)

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate Local startup field")
            result[key] = item
        return result

    json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Local startup proof must be an object")
    if set(value) != set(LocalStartupBinding.__dataclass_fields__):
        raise ValueError("Invalid Local startup binding fields")
    binding = LocalStartupBinding(**value)
    if {key: item for key, item in value.items() if key != "attempt"} != asdict(
        identity
    ):
        raise ValueError("Local startup binding identity mismatch")
    return binding


def _read(directory, name, limit):
    with directory.open_file(name) as file:
        file.verify_private_owner(confidential=True)
        if file.stat().st_size > limit:
            raise ValueError("Local startup proof exceeds byte bound")
        raw = file.read(limit)
        file.guard()
    directory.guard()
    return raw


def _verify_fence(namespace, identity):
    raw = _read(namespace, "local-submission.json", 16384)
    value = strict_loads(raw)
    expected = {
        "schema_version": 1,
        "run_id": identity.run_id,
        "request_id": identity.request_id,
        "owner": identity.owner,
        "supervisor_id": identity.supervisor_id,
        "plan_sha256": identity.plan_sha256,
    }
    if isinstance(value, dict) and "runtime_probe_sha256" in value:
        digest = value["runtime_probe_sha256"]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("Invalid Local runtime probe digest")
        if (
            hashlib.sha256(_read(namespace, "runtime-probe.json", 32768)).hexdigest()
            != digest
        ):
            raise ValueError("Local runtime probe differs from immutable fence")
        expected["runtime_probe_sha256"] = digest

    # Canonical byte form is not required, but duplicate keys and type coercion
    # cannot turn a forged receipt into authority.
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate Local submission field")
            result[key] = item
        return result

    json.loads(raw, object_pairs_hook=unique)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value != expected
    ):
        raise ValueError("Local startup differs from exact supervisor fence")


def _proof(root, namespace, identity):
    with namespace.open_directory("startup-claim") as claim:
        claim.verify_private_owner(confidential=True)
        binding = _binding(_read(namespace, "startup.json", 16384), identity)
        with root.open_directory("attempts") as attempts:
            attempts.verify_private_owner()
            with attempts.open_directory(str(binding.attempt)) as attempt:
                attempt.verify_private_owner(confidential=True)
                with attempt.open_directory("rank") as rank:
                    rank.verify_private_owner()
                    rank.guard()
                if (
                    _binding(_read(attempt, "local_identity.json", 16384), identity)
                    != binding
                ):
                    raise ValueError("Local attempt proof differs from startup binding")
                attempt.guard()
            attempts.guard()
        claim.guard()
    return binding


def read_local_binding(
    run_directory: Path, identity: LocalStartupIdentity
) -> LocalStartupBinding:
    """Read complete private startup proof; never allocate or infer PID authority."""
    if not isinstance(identity, LocalStartupIdentity) or isinstance(
        identity, LocalStartupBinding
    ):
        raise ValueError("Typed Local startup identity required")
    with (
        open_directory(_root_path(run_directory)) as root,
        open_submission_in_directory(root, identity.submission_identity) as namespace,
    ):
        namespace.guard()
        _verify_fence(namespace, identity)
        result = _proof(root, namespace, identity)
        root.guard()
        return result


def _root_path(run_directory):
    path = Path(run_directory)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Local startup requires an absolute ordinary run path")
    return path


def coordinate_local_attempt(
    run_directory: Path,
    identity: LocalStartupIdentity,
    *,
    timeout_seconds: float = 30,
    poll_seconds: float = 0.02,
) -> LocalStartupBinding:
    """Elect one writer, allocate at startup, and publish full-identity proof.

    Same complete identity is idempotent. Any torn or abandoned claim remains a
    permanent reservation; a timeout does not authorize a replacement writer.
    """
    if not isinstance(identity, LocalStartupIdentity) or isinstance(
        identity, LocalStartupBinding
    ):
        raise ValueError("Typed Local startup identity required")
    for name, value in (("timeout", timeout_seconds), ("poll", poll_seconds)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 60
        ):
            raise ValueError(f"Invalid Local startup {name}")
    deadline = time.monotonic() + timeout_seconds
    with (
        _open_writable_root(
            _root_path(run_directory),
            deadline=deadline,
            poll_seconds=poll_seconds,
            opener=open_directory,
        ) as root,
        open_submission_in_directory(
            root, identity.submission_identity, writable=True
        ) as namespace,
    ):
        namespace.guard()
        _verify_fence(namespace, identity)
        try:
            _mkdir(namespace, "startup-claim", exclusive=True)
        except FileExistsError:
            while True:
                try:
                    return _proof(root, namespace, identity)
                except FileNotFoundError as error:
                    if time.monotonic() >= deadline:
                        raise IncompleteLocalClaim(
                            "Local startup claim incomplete; claim retained"
                        ) from error
                    time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
        attempt = allocate_attempt_in_directory(root)
        binding = LocalStartupBinding(**asdict(identity), attempt=attempt)
        raw = _encode(asdict(binding))
        with (
            root.open_directory("attempts", writable=True) as attempts,
            attempts.open_directory(str(attempt), writable=True) as selected,
        ):
            selected.verify_private_owner(confidential=True)
            _publish(selected, "local_identity.json", raw)
        _publish(namespace, "startup.json", raw)
        return _proof(root, namespace, identity)
