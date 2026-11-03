"""Explicit bounded credentials; importing this module performs no IO."""

from __future__ import annotations

import os
import re
from pathlib import Path

from wavedl_daemon import paths


_TOKEN = re.compile(rb"[A-Za-z0-9._~+/=-]+\Z")


def credential_bytes(token: str) -> bytes:
    """Validate without including a credential in an error or representation."""
    try:
        if not isinstance(token, str):
            raise ValueError
        value = token.encode("ascii")
        if not _TOKEN.fullmatch(value) or len(value) > 4096:
            raise ValueError
        return value
    except (ValueError, UnicodeError):
        raise ValueError("Invalid daemon credential") from None


def read_credential(path: str | Path) -> str:
    """Read a bounded token from a regular file no other user can read."""
    try:
        with paths.open_regular(Path(path).expanduser()) as stream:
            if not paths.is_private(os.fstat(stream.fileno())):
                raise ValueError
            raw = stream.read(4098)
        if len(raw) > 4097:
            raise ValueError
        token = raw.decode("ascii").removesuffix("\n")
        credential_bytes(token)
        return token
    except (OSError, ValueError, UnicodeError):
        raise ValueError("Credential file must be private and bounded") from None
