"""Require native ownership/supervision evidence for the current CI platform.

Other-platform and optional ML/site skips remain visible, not waived as native
proof. This gate covers local filesystem and process supervision only; it does
not certify GPU trees, shared filesystems or scheduler behavior.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


POSIX_REQUIRED = {
    "core": {
        "unit_tests.core.test_safe_files": {
            "test_owned_stream_replacement_is_anchored_but_guard_rejects",
            "test_owned_directory_replacement_rejects_guard",
            "test_guard_rejects_symlink_ancestor_even_when_identity_matches",
            "test_owned_objects_reject_insecure_mode_and_foreign_uid",
            "test_fifo_rejected_without_blocking",
        },
    },
    "daemon": {
        "unit_tests.daemon.test_posix_supervisor": {
            "test_large_separate_output_is_drained_but_bounded",
            "test_zero_capture_and_stdin_closed",
            "test_exited_leader_still_escalates_ignoring_grandchild",
            "test_exited_leader_closed_pipes_do_not_hide_descendant",
            "test_ignoring_leader_requires_forceful_phase",
            "test_repeated_cancel_and_concurrent_wait_share_result",
            "test_timeout_kills_actual_tree",
            "test_cancellation_drains_continuous_output_with_bounded_capture",
        },
    },
}
WINDOWS_REQUIRED = {
    "core": {
        "unit_tests.core.test_journal_writer": {
            "test_first_flush_waits_for_native_journal_reader_release",
        },
        "unit_tests.core.test_windows_ownership": {
            "test_native_current_user_file_owner_and_dacl",
            "test_native_mkdir_700_owner_rights_is_private_to_actual_user",
        },
        "unit_tests.core.test_safe_files": {
            "test_native_windows_pinned_directory_cannot_be_renamed",
            "test_native_windows_junction_rejected",
            "test_held_directory_exclusive_link_preserves_bytes_and_existing_target",
        },
    },
    "daemon": {
        "unit_tests.daemon.test_inspection": {
            "test_native_windows_catalog_header_worker_preserves_identity_binding",
        },
        "unit_tests.daemon.test_slurm_observation": {
            "test_native_windows_pipe_is_rejected_before_reading",
        },
        "unit_tests.daemon.test_windows_supervisor": {
            "test_native_large_independent_streams_and_closed_stdin",
            "test_native_assignment_failure_never_executes_child",
            "test_native_suspended_assignment_and_descendant_inherits_job",
            "test_native_ignoring_descendant_killed_even_after_leader_exit[False]",
            "test_native_ignoring_descendant_killed_even_after_leader_exit[True]",
            "test_native_leader_exit_finalizes_without_caller_timeout",
            "test_native_daemon_death_closes_job_and_kills_descendant",
        },
    },
}
MAX_REPORT_BYTES = 8 * 1024 * 1024


def _cases(path: Path) -> list[ET.Element]:
    # Parse a bounded snapshot, not an unbounded artifact or XML entity graph.
    if path.is_symlink():
        raise ValueError("Foundations report must not be a symlink")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Foundations report must be an ordinary file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError("Foundations report identity changed while opening")
        content = stream.read(MAX_REPORT_BYTES + 1)
    if not content or len(content) > MAX_REPORT_BYTES:
        raise ValueError("Foundations report is empty or exceeds its byte bound")
    try:
        xml = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Foundations report must use UTF-8 XML") from error
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise ValueError("Foundations report cannot declare XML entities")
    root = ET.fromstring(content)
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Foundations report has an unsupported root")
    aggregates = [
        element for element in root.iter() if element.tag in {"testsuite", "testsuites"}
    ]
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("Foundations report has no testcases")

    def count(aggregate: ET.Element, attribute: str) -> int | None:
        value = aggregate.get(attribute)
        if value is None:
            return None
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"Invalid foundations {attribute} aggregate")
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError(f"Invalid foundations {attribute} aggregate") from error
        return parsed

    for aggregate in aggregates:
        aggregate_cases = list(aggregate.iter("testcase"))
        expected_counts = {
            "tests": len(aggregate_cases),
            "skipped": sum(
                case.find("skipped") is not None for case in aggregate_cases
            ),
            "failures": sum(
                case.find("failure") is not None for case in aggregate_cases
            ),
            "errors": sum(case.find("error") is not None for case in aggregate_cases),
        }
        for attribute, expected in expected_counts.items():
            reported = count(aggregate, attribute)
            if reported is not None and reported != expected:
                raise ValueError(
                    f"Foundations {attribute} count contradicts its aggregate"
                )
        if expected_counts["failures"] or expected_counts["errors"]:
            raise ValueError("Foundations report contains unsuccessful outcomes")
    return cases


def verify_reports(core: Path, daemon: Path, *, platform: str) -> dict[str, int]:
    """Reject missing/skipped native cases while retaining other skip counts."""
    if platform == "win32":
        required = WINDOWS_REQUIRED
    elif platform in {"linux", "darwin"}:
        required = POSIX_REQUIRED
    else:
        raise ValueError("Unsupported native acceptance platform")
    counts = {"required": 0, "passed": 0, "other_skipped": 0}
    for slice_name, path in (("core", core), ("daemon", daemon)):
        expected = {
            (module, name)
            for module, names in required[slice_name].items()
            for name in names
        }
        seen = set()
        for case in _cases(path):
            key = (case.get("classname"), case.get("name"))
            # pytest.importorskip at module scope emits a skipped collection
            # item with no classname and the dotted module as its name. This
            # is unverified optional coverage, never a passed native case.
            if (
                key[0] in {None, ""}
                and isinstance(key[1], str)
                and re.fullmatch(
                    rf"unit_tests\.{slice_name}\.test_[A-Za-z0-9_]+", key[1]
                )
                and case.find("skipped") is not None
                and not any(
                    case.find(outcome) is not None for outcome in ("failure", "error")
                )
            ):
                if key[1] in required[slice_name]:
                    raise ValueError("Required native module collection was skipped")
                if key in seen:
                    raise ValueError("Duplicate foundations testcase identity")
                seen.add(key)
                counts["other_skipped"] += 1
                continue
            if not all(isinstance(value, str) and value for value in key):
                raise ValueError("Foundations testcase identity is missing")
            if key in seen:
                raise ValueError("Duplicate foundations testcase identity")
            seen.add(key)
            if any(case.find(outcome) is not None for outcome in ("failure", "error")):
                raise ValueError("Foundations testcase failed")
            if case.find("skipped") is not None:
                if key in expected:
                    raise ValueError("Required native acceptance testcase was skipped")
                counts["other_skipped"] += 1
            else:
                counts["passed"] += 1
        if expected - seen:
            raise ValueError("Missing required native acceptance testcases")
        counts["required"] += len(expected)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core", type=Path)
    parser.add_argument("daemon", type=Path)
    args = parser.parse_args()
    try:
        result = verify_reports(args.core, args.daemon, platform=sys.platform)
    except (OSError, ValueError, ET.ParseError) as error:
        parser.exit(1, f"Native foundations gate failed: {error}\n")
    print(
        f"Native foundations accepted on {sys.platform}: "
        f"{result['required']} required native cases passed; "
        f"{result['other_skipped']} other skips remain explicitly unverified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
