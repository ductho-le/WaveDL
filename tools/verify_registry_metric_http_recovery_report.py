"""Strict additive F04 metric transaction crash/recovery evidence gate (POSIX)."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from uuid import UUID


if __package__ in {None, ""}:
    from verify_registry_http_recovery_report import (
        LIMITATION,
        ORDERS,
        RECOVERIES,
        _pairs,
        _read,
    )
else:
    from tools.verify_registry_http_recovery_report import (
        LIMITATION,
        ORDERS,
        RECOVERIES,
        _pairs,
        _read,
    )


METRIC = {
    "name": "loss",
    "value": 0.375,
    "step": 7,
    "unit": "mse",
    "metrics": {
        "epoch": 7,
        "train_loss": 0.375,
        "val_loss": None,
        "mae_per_param": [0.125, None],
    },
}


def snapshot(run_id, attempt, *, recovered):
    """Expected shape is independent of the transaction under observation."""
    return {
        "event_count": 2 if recovered else 1,
        "cursor": 1 if recovered else 0,
        "last_type": "metric" if recovered else "hello",
        "state": "running",
        "events": [{"type": "hello", "seq": 0}]
        + ([{"type": "metric", "seq": 1}] if recovered else []),
        "history": (
            [
                {
                    "run_id": run_id,
                    "attempt": attempt,
                    "seq": 1,
                    **METRIC,
                    "source_generation": None,
                    "active": True,
                }
            ]
            if recovered
            else []
        ),
    }


def _equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def verify_report(path: Path) -> dict:
    report = json.loads(_read(path), object_pairs_hook=_pairs)
    if type(report) is not dict:
        raise ValueError("report must be an object")
    run_id, attempt = report.get("run_id"), report.get("attempt")
    if type(run_id) is not str or str(UUID(run_id)) != run_id:
        raise ValueError("noncanonical run identity")
    if type(attempt) is not int or not 1 <= attempt <= 2**63 - 1:
        raise ValueError("invalid attempt")
    if type(report.get("order")) is not str or report["order"] not in ORDERS:
        raise ValueError("unexpected crash order")
    if type(report.get("recovery")) is not str or report["recovery"] not in RECOVERIES:
        raise ValueError("unexpected recovery path")
    before = snapshot(run_id, attempt, recovered=False)
    final = snapshot(run_id, attempt, recovered=True)
    expected = {
        "schema_version": 1,
        "case": "registry-metric-http-recovery",
        "run_id": run_id,
        "attempt": attempt,
        "order": report["order"],
        "recovery": report["recovery"],
        "rollback": before,
        "before_recovery": before,
        "recovered": final,
        "reopened": final,
        "http_metric": METRIC,
        "hello_preserved": True,
        "authenticated_status": 200,
        "unauthenticated_status": 401,
        "stable_polls": 3,
        "child_reaped": True,
        "child_returncode": -signal.SIGKILL,
        "kill_platform": "posix",
        "host_closed": True,
        "host_outstanding": 0,
        "limitation": LIMITATION,
    }
    if not _equal(report, expected):
        raise ValueError("incomplete metric recovery evidence")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=4)
    args = parser.parse_args()
    records = [verify_report(path) for path in args.reports]
    if {(row["order"], row["recovery"]) for row in records} != {
        (order, recovery) for order in ORDERS for recovery in RECOVERIES
    }:
        raise ValueError("all four metric crash/recovery cases are required")
    print("F04 metric HTTP recovery: all four crash/recovery cases passed")


if __name__ == "__main__":
    main()
