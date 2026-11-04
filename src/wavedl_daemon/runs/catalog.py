"""
Bundled model descriptions and scoped directory browsing, with few
dependencies.

Descriptions are not evidence of runtime capability. Browsing never reads file
contents, leaves its root, or loads an unbounded directory into memory.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon import paths
from wavedl_daemon.execution.budget import BudgetExhausted


if TYPE_CHECKING:
    from collections.abc import Mapping

    from wavedl_daemon.execution.service_host import ServiceHost


class BrowseForbidden(ValueError):
    """The requested location is outside the configured browse authority."""


class BrowseMissing(ValueError):
    """An allowed directory no longer exists."""


class CatalogService:
    """Call on the service owner thread, never directly on the HTTP event loop."""

    def __init__(
        self,
        roots: Mapping[str, Path],
        *,
        model_metadata_path: Path | None = None,
        max_entries: int = 1000,
    ):
        if type(max_entries) is not int or not 1 <= max_entries <= 10000:
            raise ValueError("Invalid directory enumeration bound")
        self.roots = {}
        if len(roots) > 32:
            raise ValueError("Too many browse roots")
        for key, value in roots.items():
            if (
                type(key) is not str
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) is None
            ):
                raise ValueError("Invalid browse root identifier")
            path = Path(value)
            if not path.is_absolute():
                raise ValueError("Browse roots must be absolute")
            # Explicit administrator selection, not user-controlled resolution.
            self.roots[key] = path.resolve()
        self.max_entries = max_entries
        self._metadata_path = model_metadata_path or Path(__file__).with_name(
            "model_metadata.json"
        )
        self._models_json = None

    def models(self) -> dict:
        """Load static JSON metadata without importing GUI or the ML stack."""
        if self._models_json is None:
            metadata = json.loads(paths.read_regular(self._metadata_path, 256 * 1024))
            categories = metadata.get("categories") if type(metadata) is dict else None
            models = metadata.get("models") if type(metadata) is dict else None
            if (
                type(categories) is not dict
                or type(models) is not list
                or len(models) > 1000
            ):
                raise ValueError("Invalid bundled model metadata")
            names = set()
            for model in models:
                if (
                    type(model) is not dict
                    or type(model.get("id")) is not str
                    or model["id"] in names
                ):
                    raise ValueError("Invalid bundled model identity")
                names.add(model["id"])
            result = {
                "source": "bundled_metadata",
                "runtime_verified": False,
                "categories": categories,
                "models": sorted(models, key=lambda item: item["id"]),
            }
            encoded = json.dumps(result, allow_nan=False, sort_keys=True)
            if len(encoded.encode()) > 256 * 1024:
                raise ValueError("Model catalog exceeds bound")
            self._models_json = encoded
        return json.loads(self._models_json)

    def browse(
        self,
        root: str | None = None,
        path: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        if (
            type(limit) is not int
            or not 1 <= limit <= 200
            or type(offset) is not int
            or not 0 <= offset <= self.max_entries
        ):
            raise ValueError("Invalid browse page bounds")
        if type(path) is not str or len(path) > 4096 or "\0" in path:
            raise ValueError("Invalid browse path")
        if root is None:
            if path:
                raise ValueError("Path requires an explicit browse root")
            return {"roots": [{"id": name} for name in sorted(self.roots)]}
        if type(root) is not str or root not in self.roots:
            raise BrowseForbidden("Unknown browse root")
        try:
            directory = paths.inside(self.roots[root], path.split("/") if path else ())
        except paths.UnsafePath:
            raise BrowseForbidden(
                "Browse path must remain within its configured root"
            ) from None
        entries = []
        try:
            with os.scandir(directory) as listing:
                for index, entry in enumerate(listing):
                    if index >= self.max_entries:
                        raise BudgetExhausted("Directory listing capacity exhausted")
                    if entry.is_dir(follow_symlinks=False):
                        entries.append({"name": entry.name, "kind": "directory"})
                    elif entry.is_file(follow_symlinks=False):
                        entries.append({"name": entry.name, "kind": "file"})
        except NotADirectoryError:
            raise BrowseForbidden("Browse location is not a directory") from None
        entries.sort(key=lambda item: (item["kind"] != "directory", item["name"]))
        return {
            "root": root,
            "path": path,
            "entries": entries[offset : offset + limit],
            "total": len(entries),
            "limit": limit,
            "offset": offset,
        }

    def resolve(self, root: str, path: str = "") -> dict:
        """The selected host path, a file or directory inside its root."""
        if type(root) is not str or root not in self.roots:
            raise BrowseForbidden("Unknown browse root")
        if type(path) is not str or len(path.encode("utf-8")) > 4096 or "\0" in path:
            raise ValueError("Invalid selected browse path")
        parts = path.split("/") if path else []
        if len(parts) > 128:
            raise BrowseForbidden("Selected path is too deep")
        try:
            info = paths.inside(self.roots[root], parts).stat()
        except paths.UnsafePath:
            raise BrowseForbidden(
                "Selected path must remain within its configured root"
            ) from None
        except PermissionError:
            raise BrowseForbidden("Selected browse location is unreadable") from None
        except FileNotFoundError:
            raise BrowseMissing("Selected browse location does not exist") from None
        if stat.S_ISDIR(info.st_mode):
            kind = "directory"
        elif stat.S_ISREG(info.st_mode):
            kind = "file"
        else:
            raise BrowseForbidden("Selected location is not a file or directory")
        return {
            "path": str(self.roots[root].joinpath(*parts)),
            "kind": kind,
            "root": root,
            "relative_path": path,
        }


class AsyncCatalogService:
    """Owner admission plus bounded filesystem workers, never ASGI or SQL loop."""

    def __init__(self, host: ServiceHost):
        self.host = host

    async def models(self) -> dict:
        return await self._work("models")

    async def browse(self, root=None, path="", limit=100, offset=0) -> dict:
        return await self._work("browse", root, path, limit, offset)

    async def resolve(self, root, path="") -> dict:
        return await self._work("resolve", root, path)

    async def _work(self, method, *args):
        async def invoke(session):
            def work(token):
                try:
                    return getattr(session.catalog, method)(*args)
                except PermissionError:
                    raise BrowseForbidden("Browse directory is unreadable") from None
                except FileNotFoundError:
                    raise BrowseMissing("Browse directory does not exist") from None

            handle = session.pool.submit(work)
            try:
                return await handle.wait()
            except asyncio.CancelledError:
                # The admission slot stays owned until the real worker
                # finishes, including a blocked filesystem that ignores expiry.
                handle.cancel()
                try:
                    await asyncio.shield(handle.wait())
                except Exception:
                    pass
                raise

        return await self.host.call(invoke, concurrent_async=True)
