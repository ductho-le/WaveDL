"""Dependency-light UTF-8 stream contract for WaveDL command entry points."""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Use UTF-8 CLI logs, including redirected Windows daemon captures.

    Entry points call this before parsing or logging; importing this module does
    not change streams. Caller streams without reconfiguration stay intact.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def utf8_stdio_environment(environment: dict[str, str]) -> dict[str, str]:
    """Return a copy aligning Python child logs with the CLI UTF-8 contract."""
    return environment | {"PYTHONIOENCODING": "utf-8:backslashreplace"}
