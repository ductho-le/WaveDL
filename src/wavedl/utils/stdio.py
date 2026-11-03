"""UTF-8 console output for WaveDL command-line entry points.

WaveDL CLIs print Unicode (emoji status markers, ``R²``, ``✔``). Python encodes
``sys.stdout``/``sys.stderr`` with the console or locale encoding, so a Windows
console (cp1252) or a pipe under a non-UTF-8 locale raises ``UnicodeEncodeError``
and aborts the run. Entry points call :func:`configure_utf8_stdio` before any
output; importing this module changes nothing.
"""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Switch ``sys.stdout``/``sys.stderr`` to UTF-8 where supported.

    Unencodable characters are backslash-escaped instead of raising. Streams
    without ``reconfigure`` (e.g. ``io.StringIO`` substitutes) are left intact.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def utf8_stdio_environment(environment: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``environment`` that makes Python children write UTF-8."""
    return environment | {"PYTHONIOENCODING": "utf-8:backslashreplace"}
