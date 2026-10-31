"""Allocate a Local runtime attempt before launching its approved module.

The daemon retains native supervisor ownership; this bootstrap proves only the
writer's attempt binding. It runs inside the supervised process, including on
Windows, rather than replacing the process with platform-dependent exec rules.
Existing reserved-attempt launch commands are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from wavedl.core.local_attempt import LocalStartupIdentity, coordinate_local_attempt
from wavedl.core.protocol import strict_loads
from wavedl.core.runtime_identity import (
    checked_python_identity,
    load_runtime_identity_json,
)
from wavedl.core.submission import identity_for_submission, read_submission_file


def prepare_local_startup(run_directory: Path, environment: dict[str, str]):
    """Verify immutable authority before allocating; return argv and identity."""
    if environment.get("WAVEDL_ATTEMPT_ALLOCATION") != "writer":
        raise ValueError("Local writer allocation requires explicit selection")
    if "WAVEDL_ATTEMPT" in environment:
        raise ValueError("Writer allocation cannot consume an injected attempt")
    identity = identity_for_submission(
        run_directory,
        request_id=environment.get("WAVEDL_REQUEST_ID"),
        backend="local",
        owner=environment.get("WAVEDL_OWNER"),
        plan_sha256=environment.get("WAVEDL_PLAN_SHA256"),
    )
    if environment.get("WAVEDL_RUN_ID") != identity.run_id:
        raise ValueError("Local startup run differs from submission")
    startup = LocalStartupIdentity(
        run_id=identity.run_id,
        request_id=identity.request_id,
        owner=identity.owner,
        supervisor_id=environment.get("WAVEDL_SUPERVISOR_ID"),
        run_directory=identity.run_directory,
        manifest_sha256=identity.manifest_sha256,
        plan_sha256=identity.plan_sha256,
    )
    raw = read_submission_file(run_directory, identity, "plan.json")
    if hashlib.sha256(raw).hexdigest() != startup.plan_sha256:
        raise ValueError("Local startup plan digest differs")
    plan = strict_loads(raw)
    if (
        type(plan) is not dict
        or plan.get("target") != "local"
        or plan.get("allocation_mode") != "writer"
        or type(plan.get("manifest_json")) is not str
        or type(plan.get("argv")) is not list
    ):
        raise ValueError("Invalid Local writer plan")
    # The digest binds the exact canonical plan, including nested fields. Reject
    # duplicate-key or alternative encodings rather than interpreting them twice.
    canonical = json.dumps(
        plan, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    if canonical != raw:
        raise ValueError("Local startup plan is not canonical")
    manifest = strict_loads(plan["manifest_json"])
    if (
        type(manifest) is not dict
        or manifest.get("run_id") != identity.run_id
        or manifest.get("run_directory") != identity.run_directory
        or manifest.get("working_directory") != identity.run_directory
        or type(manifest.get("runtime")) is not dict
    ):
        raise ValueError("Local startup manifest differs from submission")
    manifest_bytes = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != identity.manifest_sha256:
        raise ValueError("Local plan manifest differs from the approved file")
    argv = plan["argv"]
    operation = manifest.get("operation")
    expected_module = (
        "wavedl.launcher"
        if operation == "train"
        else "wavedl.test"
        if operation in {"test", "plot"}
        else None
    )
    if (
        len(argv) < 4
        or len(argv) > 4096
        or any(type(value) is not str or "\0" in value for value in argv)
        or sum(len(value.encode()) + 1 for value in argv) > 65536
        or argv[:4] != [manifest["runtime"].get("python"), "-u", "-m", expected_module]
        or expected_module is None
    ):
        raise ValueError("Local startup command differs from the approved module")
    fence = load_runtime_identity_json(
        read_submission_file(run_directory, identity, "local-submission.json")
    )
    if type(fence) is not dict:
        raise ValueError("Invalid Local supervisor fence")
    if "runtime_probe_sha256" in fence:
        probe_raw = read_submission_file(run_directory, identity, "runtime-probe.json")
        if hashlib.sha256(probe_raw).hexdigest() != fence["runtime_probe_sha256"]:
            raise ValueError("Local runtime probe differs from immutable fence")
        attestation = load_runtime_identity_json(probe_raw)
        expected = {
            "schema_version": 1,
            "selected_python": manifest["runtime"].get("python"),
            "plan_sha256": identity.plan_sha256,
            "run_id": identity.run_id,
            "request_id": identity.request_id,
            "owner": identity.owner,
            "supervisor_id": startup.supervisor_id,
        }
        if (
            type(attestation) is not dict
            or type(attestation.get("schema_version")) is not int
            or set(attestation) != {*expected, "identity"}
            or any(attestation[key] != value for key, value in expected.items())
        ):
            raise ValueError("Local runtime probe differs from submission")
        actual = {
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        }
        if checked_python_identity(attestation["identity"]) != actual:
            raise ValueError("Local startup Python environment differs from probe")
    binding = coordinate_local_attempt(run_directory, startup)
    selected = dict(environment)
    selected.update(
        WAVEDL_ATTEMPT=str(binding.attempt),
        WAVEDL_RUN_DIR=str(run_directory),
    )
    return tuple(argv), selected, binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        command, environment, _binding = prepare_local_startup(
            args.run_directory, dict(os.environ)
        )
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(2, f"Local startup refused: {type(error).__name__}\n")
    os.environ.update(environment)
    os.chdir(args.run_directory)
    sys.argv = [command[3], *command[4:]]
    runpy.run_module(command[3], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
