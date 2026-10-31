# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Describe the daemon host so the client can prefill a runnable Local runtime.

The daemon never imports numerical packages: accelerators are probed in a reaped
child of the daemon interpreter, the same interpreter dataset inspection uses.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.supervision.base import ProcessSpec


if TYPE_CHECKING:
    from wavedl_daemon.execution.budget import WorkBudget
    from wavedl_daemon.execution.pool import BoundedThreadPool


# Prints one JSON line; any failure leaves the accelerators unknown, not the host.
_PROBE = """
import json
try:
    import torch
except Exception:
    print(json.dumps({"torch": None, "cuda": [], "mps": False}))
else:
    cuda = []
    if torch.cuda.is_available():
        for index in range(min(torch.cuda.device_count(), 64)):
            props = torch.cuda.get_device_properties(index)
            cuda.append({"name": str(props.name)[:200], "memory_bytes": int(props.total_memory)})
    backend = getattr(torch.backends, "mps", None)
    mps = bool(backend is not None and backend.is_available())
    print(json.dumps({"torch": str(torch.__version__)[:64], "cuda": cuda, "mps": mps}))
"""
_MAX_PROBE_BYTES = 32768


def validate_accelerators(content: bytes) -> dict | None:
    """Accept only the probe's exact shape; anything else is unknown."""
    if len(content) > _MAX_PROBE_BYTES:
        return None
    try:
        value = json.loads(content)
    except (ValueError, RecursionError):
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"torch", "cuda", "mps"}
        or not (value["torch"] is None or isinstance(value["torch"], str))
        or type(value["mps"]) is not bool
        or not isinstance(value["cuda"], list)
        or len(value["cuda"]) > 64
    ):
        return None
    for device in value["cuda"]:
        if (
            not isinstance(device, dict)
            or set(device) != {"name", "memory_bytes"}
            or not isinstance(device["name"], str)
            or type(device["memory_bytes"]) is not int
            or device["memory_bytes"] < 0
        ):
            return None
    return value


def _memory_bytes() -> int | None:
    try:
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return pages * size if pages > 0 and size > 0 else None


class HostProfiler:
    """Profile once per service session; hardware does not change under a daemon."""

    def __init__(
        self,
        python: str,
        cwd: Path,
        pool: BoundedThreadPool,
        backends,
        *,
        budget: WorkBudget | None = None,
    ):
        if not isinstance(python, str) or not Path(python).is_absolute():
            raise ValueError(
                "Host profile requires a server-selected absolute interpreter"
            )
        self.python, self.cwd, self.pool = python, Path(cwd), pool
        self.backends = tuple(sorted(backends))
        self.budget = budget or pool.budget
        self._profile: dict | None = None

    async def profile(self) -> dict:
        if self._profile is not None:
            return self._profile
        accelerators = await self._accelerators()
        profile = {
            "hostname": platform.node()[:255],
            "os": platform.system()[:64],
            "os_release": platform.release()[:128],
            "arch": platform.machine()[:64],
            "cpu_count": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
            "python": self.python,
            "python_version": platform.python_version(),
            "accelerators": accelerators,
            "backends": list(self.backends),
        }
        # A failed probe may be transient (a saturated pool); ask again next time.
        if accelerators is not None:
            self._profile = profile
        return profile

    async def _accelerators(self) -> dict | None:
        def work(token):
            result = run_command(
                ProcessSpec((self.python, "-c", _PROBE), self.cwd),
                self.budget,
                token=token,
            )
            if (
                result.returncode
                or result.timed_out
                or result.cancelled
                or result.stdout_truncated
            ):
                return None
            return validate_accelerators(result.stdout)

        handle = self.pool.submit(work)
        try:
            return await handle.wait()
        except asyncio.CancelledError:
            handle.cancel()
            try:
                await asyncio.shield(handle.wait())
            except Exception:
                pass
            raise
        except Exception:
            return None
