# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Dependency-light bundled model descriptions and explicitly scoped browsing.

Bundled descriptions are not runtime capability evidence. Browsing never reads
file payloads, follows child symlinks, or expands an unbounded directory into memory.
"""

from __future__ import annotations

import asyncio
import errno
import io
import json
import re
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.safe_files import open_directory, read_regular_file
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
            raw = read_regular_file(self._metadata_path, limit=256 * 1024)
            metadata = json.load(io.BytesIO(raw))
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
        parts = path.split("/") if path else []
        if any(
            not part or part in {".", ".."} or any(char in part for char in "\\:")
            for part in parts
        ):
            raise BrowseForbidden("Browse path must remain within its configured root")
        with ExitStack() as stack:
            anchor = stack.enter_context(open_directory(self.roots[root]))
            self._guard(anchor)
            current = anchor
            for part in parts:
                try:
                    current = stack.enter_context(current.open_directory(part))
                except ValueError:
                    raise BrowseForbidden(
                        "Browse directory is not an ordinary contained directory"
                    ) from None
            self._guard(current)
            entries = []
            for index, entry in enumerate(current.entries()):
                if index >= self.max_entries:
                    raise BudgetExhausted("Directory enumeration capacity exhausted")
                if entry.is_directory or entry.is_regular:
                    entries.append(
                        {
                            "name": entry.name,
                            "kind": "directory" if entry.is_directory else "file",
                        }
                    )
            self._guard(anchor)
            self._guard(current)
        entries.sort(key=lambda item: (item["kind"] != "directory", item["name"]))
        return {
            "root": root,
            "path": path,
            "entries": entries[offset : offset + limit],
            "total": len(entries),
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    def _guard(anchor):
        try:
            anchor.guard()
        except ValueError:
            raise BrowseForbidden("Browse directory ancestry changed") from None

    def resolve(self, root: str, path: str = "") -> dict:
        """Return an explicit selected host path only after no-follow metadata proof."""
        if type(root) is not str or root not in self.roots:
            raise BrowseForbidden("Unknown browse root")
        if type(path) is not str or len(path.encode("utf-8")) > 4096 or "\0" in path:
            raise ValueError("Invalid selected browse path")
        parts = path.split("/") if path else []
        if len(parts) > 128 or any(
            not part or part in {".", ".."} or any(char in part for char in "\\:")
            for part in parts
        ):
            raise BrowseForbidden(
                "Selected path must remain within its configured root"
            )
        try:
            with ExitStack() as stack:
                anchor = stack.enter_context(open_directory(self.roots[root]))
                retained = [anchor]
                self._guard(anchor)
                current = anchor
                for part in parts[:-1]:
                    current = stack.enter_context(current.open_directory(part))
                    retained.append(current)
                    self._guard(current)
                selected, kind = current, "directory"
                if parts:
                    try:
                        selected = stack.enter_context(
                            current.open_directory(parts[-1])
                        )
                        retained.append(selected)
                    except ValueError:
                        selected = stack.enter_context(current.open_file(parts[-1]))
                        kind = "file"
                    self._guard(selected)
                for directory in retained:
                    self._guard(directory)
                self._guard(selected)
                resolved = self.roots[root].joinpath(*parts)
                return {
                    "path": str(resolved),
                    "kind": kind,
                    "root": root,
                    "relative_path": path,
                }
        except PermissionError:
            raise BrowseForbidden("Selected browse location is unreadable") from None
        except FileNotFoundError:
            raise BrowseMissing("Selected browse location does not exist") from None
        except ValueError:
            raise BrowseForbidden(
                "Selected location is not an ordinary contained member"
            ) from None
        except OSError as error:
            if error.errno in {errno.ENXIO, errno.ENODEV}:
                raise BrowseForbidden(
                    "Selected location is not an ordinary contained member"
                ) from None
            raise


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
                # Admission/ancestry handles stay owned until the real worker
                # finishes, including a blocked filesystem that ignores expiry.
                handle.cancel()
                try:
                    await asyncio.shield(handle.wait())
                except Exception:
                    pass
                raise

        return await self.host.call(invoke, concurrent_async=True)
