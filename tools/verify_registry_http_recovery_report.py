"""Validate POSIX-only F04 evidence; require all four crash/recovery cases.

This gate does not certify native Windows process termination or ACL behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
from pathlib import Path
from uuid import UUID


LIMITATION = (
    "POSIX-only local ASGI acceptance with controlled numerical-startup backend and injected "
    "reverse writer order; no browser, remote HPC, or network transport."
)
ORDERS = {"projection-cursor", "cursor-projection"}
RECOVERIES = {"rebuild", "ingest"}
MAX_BYTES = 64 * 1024


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _read(path):
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("F04 report validation requires POSIX no-follow access")
    if ".." in path.parts:
        raise ValueError("report path cannot traverse parents")
    path = Path(os.path.abspath(path))
    ancestors = []
    for parent in reversed(path.parents):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("report ancestor must not be a symlink")
        ancestors.append((parent, _snapshot(info)))
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
        raise ValueError("report must be an ordinary bounded nonempty file")
    if before.st_mode & 0o077:
        raise ValueError("report must be private")
    if hasattr(os, "getuid") and before.st_uid != os.getuid():
        raise ValueError("report must be owned")
    fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        if _snapshot(os.fstat(fd)) != _snapshot(before):
            raise ValueError("report identity changed")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_BYTES + 1)
        if not raw or len(raw) > MAX_BYTES:
            raise ValueError("report exceeds byte bound")
        if _snapshot(os.fstat(fd)) != _snapshot(before) or _snapshot(
            path.lstat()
        ) != _snapshot(before):
            raise ValueError("report changed during read")
        for parent, snapshot in ancestors:
            # Directory mtime can change for unrelated siblings; identity/mode
            # is the safety invariant for ancestor replacement.
            if _snapshot(parent.lstat())[:3] != snapshot[:3]:
                raise ValueError("report ancestry changed")
        return raw
    finally:
        os.close(fd)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _typed_equal(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def verify_report(path: Path) -> dict:
    report = json.loads(_read(path), object_pairs_hook=_pairs)
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    expected = {
        "schema_version": 1,
        "case": "registry-http-recovery",
        "event_count": 1,
        "cursor": 0,
        "replay_rows": 1,
        "rollback": {"event_count": 0, "cursor": -1, "replay_rows": 0},
        "before_recovery": {"event_count": 0, "cursor": -1, "replay_rows": 0},
        "authenticated_status": 200,
        "unauthenticated_status": 401,
        "stable_polls": 3,
        "child_reaped": True,
        "host_closed": True,
        "host_outstanding": 0,
        "limitation": LIMITATION,
    }
    if set(report) != set(expected) | {
        "order",
        "recovery",
        "run_id",
        "child_returncode",
        "kill_platform",
    }:
        raise ValueError("unexpected report fields")
    for key, value in expected.items():
        if not _typed_equal(report[key], value):
            raise ValueError("incomplete recovery evidence")
    if type(report["order"]) is not str or report["order"] not in ORDERS:
        raise ValueError("unexpected crash order")
    if type(report["recovery"]) is not str or report["recovery"] not in RECOVERIES:
        raise ValueError("unexpected recovery path")
    if report["kill_platform"] != "posix":
        raise ValueError("F04 only certifies POSIX SIGKILL recovery")
    returncode = -signal.SIGKILL
    if not _typed_equal(report["child_returncode"], returncode):
        raise ValueError("child was not killed at the barrier")
    if (
        type(report["run_id"]) is not str
        or str(UUID(report["run_id"])) != report["run_id"]
    ):
        raise ValueError("noncanonical run identity")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=4)
    args = parser.parse_args()
    records = [verify_report(path) for path in args.reports]
    if {(record["order"], record["recovery"]) for record in records} != {
        (order, recovery) for order in ORDERS for recovery in RECOVERIES
    }:
        raise ValueError("all four crash/recovery cases are required")
    print("F04 registry HTTP recovery: all four crash/recovery cases passed")


if __name__ == "__main__":
    main()
