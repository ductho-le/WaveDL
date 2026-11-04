"""Explicit bounded credentials; importing this module performs no IO."""

from __future__ import annotations

import re
from pathlib import Path

from wavedl.core.safe_files import open_directory


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
    """Read a private, owned, regular no-follow file with a hard byte bound."""
    path = Path(path).expanduser().absolute()
    try:
        with (
            open_directory(path.parent) as directory,
            directory.open_file(path.name) as stream,
        ):
            stream.verify_private_owner(confidential=True)
            if stream.stat().size > 4097:
                raise ValueError
            raw = stream.read(4098)
            stream.guard()
        token = raw.decode("ascii").removesuffix("\n")
        credential_bytes(token)
        return token
    except (OSError, ValueError, UnicodeError):
        raise ValueError("Credential file must be private, owned and bounded") from None
