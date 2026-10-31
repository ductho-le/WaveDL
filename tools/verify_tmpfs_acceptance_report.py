"""Require exactly one successful real-tmpfs acceptance case, never a skip.

This gate validates evidence, not filesystem capacity or mount identity; the
required test performs those checks itself. It intentionally accepts no broad
suite or optional/platform skip waiver.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REQUIRED_CASE = (
    "unit_tests.daemon.test_tmpfs_storage_acceptance",
    "test_real_tmpfs_enospc_preserves_authority_and_recovers",
)
MAX_REPORT_BYTES = 1024 * 1024
COUNTS = {"tests": "1", "errors": "0", "failures": "0", "skipped": "0"}


def _read_report(path: Path) -> bytes:
    """Read a bounded snapshot through pinned, no-follow POSIX ancestors."""
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("tmpfs report validation requires POSIX no-follow access")
    if ".." in path.parts:
        raise ValueError("tmpfs report path must not traverse parent aliases")
    path = Path(os.path.abspath(path))
    uid = os.getuid()
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    identities = []
    current = Path(path.anchor)
    try:
        for index, part in enumerate(path.parts[1:]):
            current /= part
            final = index == len(path.parts) - 2
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if not final:
                flags |= os.O_DIRECTORY
            opened = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = opened
            metadata = os.fstat(descriptor)
            identities.append((current, metadata))
            if final:
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("tmpfs report must be an ordinary file")
                if metadata.st_uid != uid or metadata.st_mode & 0o022:
                    raise ValueError("tmpfs report must be user-owned and protected")
                if metadata.st_size > MAX_REPORT_BYTES:
                    raise ValueError("tmpfs report exceeds its byte bound")
            else:
                # Root-owned sticky temporary roots may contain private children;
                # other ancestors must not permit another user to replace entries.
                shared_tmp = metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX
                if metadata.st_uid not in {0, uid} or (
                    metadata.st_mode & 0o022 and not shared_tmp
                ):
                    raise ValueError("tmpfs report has an unsafe ancestor")
        if not identities:
            raise ValueError("tmpfs report must name an ordinary file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(MAX_REPORT_BYTES + 1)
        if not content or len(content) > MAX_REPORT_BYTES:
            raise ValueError("tmpfs report is empty or exceeds its byte bound")
        for checked, before in identities:
            now = checked.lstat()
            if stat.S_ISLNK(now.st_mode) or (
                now.st_dev,
                now.st_ino,
                now.st_uid,
                now.st_mode,
            ) != (before.st_dev, before.st_ino, before.st_uid, before.st_mode):
                raise ValueError("tmpfs report path changed while reading")
        if (now.st_size, now.st_mtime_ns) != (metadata.st_size, metadata.st_mtime_ns):
            raise ValueError("tmpfs report changed while reading")
        return content
    finally:
        os.close(descriptor)


def verify_report(path: Path) -> int:
    """Return one only for the exact, non-skipped pytest acceptance report."""
    try:
        xml = _read_report(path).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("tmpfs report must use UTF-8 XML") from error
    if "\x00" in xml or "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise ValueError("tmpfs report cannot declare XML entities or use UTF-16")
    encoding = re.match(
        r"\s*<\?xml\b[^?]*\bencoding\s*=\s*(['\"])(.*?)\1", xml, re.IGNORECASE
    )
    if encoding is not None and encoding[2].lower() not in {"utf-8", "utf8"}:
        raise ValueError("tmpfs report must declare UTF-8 XML")
    root = ET.fromstring(xml)
    if root.tag == "testsuites":
        if len(root) != 1 or root[0].tag != "testsuite":
            raise ValueError("tmpfs report must contain exactly one testsuite")
        suite = root[0]
        for attribute, expected in COUNTS.items():
            if attribute in root.attrib and root.get(attribute) != expected:
                raise ValueError("tmpfs report root aggregate contradicts acceptance")
    elif root.tag == "testsuite":
        suite = root
    else:
        raise ValueError("tmpfs report has an unsupported root")
    for attribute, expected in COUNTS.items():
        if suite.get(attribute) != expected:
            raise ValueError(
                "tmpfs report requires tests=1 and zero error/failure/skip"
            )
    if len(suite) != 1 or suite[0].tag != "testcase":
        raise ValueError("tmpfs report must contain exactly one testcase")
    case = suite[0]
    if (case.get("classname"), case.get("name")) != REQUIRED_CASE:
        raise ValueError("tmpfs report lacks the exact required acceptance testcase")
    if any(
        element.tag in {"skipped", "failure", "error", "testcase", "testsuite"}
        for child in case
        for element in child.iter()
    ):
        raise ValueError("tmpfs acceptance must pass without skips or other outcomes")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        verify_report(args.report)
    except (OSError, ValueError, ET.ParseError):
        # Never echo untrusted XML, paths, exception messages or testcase names.
        print(
            "tmpfs acceptance gate failed: provide a protected UTF-8 JUnit report "
            "containing exactly the required real-tmpfs test passed, with zero skips, "
            "failures or errors.",
            file=sys.stderr,
        )
        return 1
    print("tmpfs acceptance accepted: 1 required case passed; 0 skips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
