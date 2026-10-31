# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Dispatch-lock liveness and private, credential-free serving metadata."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from wavedl.core.protocol import strict_loads
from wavedl.core.safe_files import open_directory
from wavedl_daemon.storage.ownership import DispatchHeld, DispatchLock


_RUN_URL = re.compile(r"http://(127\.0\.0\.1|\[::1\]):\d{1,5}/\Z")


class AlreadyRunning(RuntimeError):
    """The state directory already has a dispatch owner."""

    def __init__(self, url: str | None):
        super().__init__("WaveDL is already running")
        self.url = url


def check_not_running(state_root: Path) -> None:
    path = state_root / "dispatch.lock"
    if not path.exists():
        return
    lock = DispatchLock(path)
    try:
        lock.acquire()
    except DispatchHeld:
        raise AlreadyRunning(read_run_url(state_root)) from None
    finally:
        lock.release()


def read_run_url(state_root: Path) -> str | None:
    try:
        with (
            open_directory(state_root) as directory,
            directory.open_file("serve.json") as stream,
        ):
            stream.verify_private_owner(confidential=True)
            if stream.stat().size > 4096:
                return None
            value = strict_loads(stream.read(4097))
            stream.guard()
        url = value.get("url") if type(value) is dict else None
        if isinstance(url, str) and _RUN_URL.fullmatch(url):
            return url
    except (OSError, ValueError, RuntimeError):
        pass
    return None


def write_run_url(state_root: Path, url: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".serve-", dir=state_root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"url": url}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_root / "serve.json")
    finally:
        temporary.unlink(missing_ok=True)


def remove_run_url(state_root: Path) -> None:
    (state_root / "serve.json").unlink(missing_ok=True)
