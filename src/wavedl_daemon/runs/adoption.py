# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Explicit standalone provenance verification; never adopts a PID or handle."""

from __future__ import annotations

import json
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import (
    Manifest,
    RuntimeSelection,
    StorageMapping,
    describes_layout,
)
from wavedl.core.protocol import Envelope, decode
from wavedl.core.safe_files import open_directory
from wavedl_daemon.runs.negotiation import checked_hello


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate adoption metadata key")
        result[key] = value
    return result


def _object(raw: bytes) -> dict:
    def reject(value):
        raise ValueError("Nonfinite adoption metadata")

    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("Adoption metadata must be an object")
    return value


def _manifest(raw: bytes) -> Manifest:
    data = _object(raw)
    return Manifest(
        **data
        | {
            "operation": OperationKind(data["operation"]),
            "runtime": RuntimeSelection(**data["runtime"]),
            "storage_map": tuple(
                StorageMapping(**item) for item in data["storage_map"]
            ),
            "path_fields": tuple(data["path_fields"]),
        }
    )


@dataclass
class VerifiedStandaloneRun:
    layout: RunLayout
    manifest: Manifest
    request_manifest: Manifest
    request_id: str
    attempt: int
    hello: Envelope
    protocol_version: int
    _anchors: ExitStack
    _directories: tuple

    def guard(self) -> None:
        """Recheck retained ownership and pathname identity before registration."""
        for directory in self._directories:
            directory.guard()
            directory.verify_private_owner()

    def close(self) -> None:
        self._anchors.close()

    def __enter__(self):
        self.guard()
        return self

    def __exit__(self, *_):
        self.close()


def verify_standalone_run(
    path: str | Path, *, attempt: int | None = None
) -> VerifiedStandaloneRun:
    """Return retained verified evidence, without registry or backend mutations.

    Missing backend evidence must remain supervision_lost/unknown in the caller;
    runtime journal exit records are not independent backend terminal evidence.
    """
    path = Path(path).absolute()
    anchors = ExitStack()
    try:
        root = anchors.enter_context(open_directory(path))
        root.verify_private_owner()
        retained = [root]

        def read(directory, name, limit=1024 * 1024):
            stream = anchors.enter_context(directory.open_file(name))
            stream.verify_private_owner()
            retained.append(stream)
            raw = bytearray()
            while chunk := stream.read(min(65536, limit + 1 - len(raw))):
                raw.extend(chunk)
                if len(raw) > limit:
                    raise ValueError("Adoption metadata exceeds bound")
            return bytes(raw)

        manifest = _manifest(read(root, "manifest.json"))
        layout = RunLayout(path, manifest.run_id)
        if not describes_layout(manifest, layout):
            raise ValueError("Adopted manifest/layout mismatch")
        attempts = anchors.enter_context(root.open_directory("attempts"))
        attempts.verify_private_owner()
        retained.append(attempts)
        if attempt is None:
            candidates = [
                int(entry.name)
                for entry in attempts.entries()
                if entry.is_directory and re.fullmatch(r"[1-9][0-9]*", entry.name)
            ]
            if not candidates:
                raise ValueError("Adopted run has no runtime attempt")
            attempt = max(candidates)
        if type(attempt) is not int or attempt < 1:
            raise ValueError("Invalid adoption attempt")
        execution = anchors.enter_context(attempts.open_directory(str(attempt)))
        execution.verify_private_owner()
        retained.append(execution)
        identity = _object(read(execution, "runtime_identity.json", 4096))
        if (
            set(identity) != {"run_id", "request_id", "attempt", "restart_count"}
            or identity["run_id"] != manifest.run_id
            or type(identity["attempt"]) is not int
            or identity["attempt"] != attempt
            or type(identity["restart_count"]) is not int
            or identity["restart_count"] != 0
            or not isinstance(identity["request_id"], str)
            or str(UUID(identity["request_id"])) != identity["request_id"]
        ):
            raise ValueError("Adopted runtime identity mismatch")
        request = _manifest(read(execution, "runtime_request.json"))
        if (
            not describes_layout(request, layout)
            or request.operation != manifest.operation
        ):
            raise ValueError("Adopted attempt request mismatch")
        stream = anchors.enter_context(
            execution.open_file("events.jsonl", share_write=True)
        )
        stream.verify_private_owner()
        retained.append(stream)
        raw = stream.read_first_line(limit=65536)
        if not raw.endswith(b"\n"):
            raise ValueError("Adopted runtime hello is unpublished")
        event = decode(raw)
        _, hello, version = checked_hello(event)
        if event.run_id != manifest.run_id or event.attempt != attempt:
            raise ValueError("Adopted hello identity mismatch")
        if (
            not hello.runtime_version.strip()
            or not Path(hello.interpreter).is_absolute()
            or os.path.normcase(os.path.normpath(hello.interpreter))
            != os.path.normcase(os.path.normpath(request.runtime.python))
            or request.runtime.device not in {"auto", hello.device}
            or request.runtime.container != hello.container
        ):
            raise ValueError("Adopted actual runtime differs from request")
        python_version = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\.[0-9]+)?", hello.python)
        if python_version is None or tuple(map(int, python_version.group(1, 2))) < (
            3,
            11,
        ):
            raise ValueError("Adopted runtime requires Python >=3.11")
        result = VerifiedStandaloneRun(
            layout,
            manifest,
            request,
            identity["request_id"],
            attempt,
            event,
            version,
            anchors.pop_all(),
            tuple(retained),
        )
        try:
            result.guard()
        except BaseException:
            result.close()
            raise
        return result
    finally:
        anchors.close()
