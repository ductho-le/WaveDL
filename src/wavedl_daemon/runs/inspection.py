# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded supervised metadata workers; no numerical imports in the daemon."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.dataset_headers import InspectionLimits, InspectionRefused
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.runs.preview_authority import (
    PreviewAuthority,
    PreviewChanged,
    PreviewForbidden,
    PreviewMissing,
    verify_selection,
)
from wavedl_daemon.supervision.base import ProcessSpec


if TYPE_CHECKING:
    from wavedl_daemon.execution.budget import WorkBudget
    from wavedl_daemon.execution.pool import BoundedThreadPool
    from wavedl_daemon.runs.catalog import CatalogService


def _pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError("Duplicate metadata key")
        result[name] = value
    return result


def _constant(value):
    raise ValueError("Non-finite metadata")


def validate_metadata(content: bytes, limits: InspectionLimits) -> dict:
    if len(content) > limits.max_output_bytes:
        raise InspectionRefused("Metadata output exceeds limit")
    try:
        value = json.loads(content, object_pairs_hook=_pairs, parse_constant=_constant)
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "format", "variables"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["format"] not in {"npy", "npz", "mat", "h5", "hdf5"}
            or not isinstance(value["variables"], list)
            or len(value["variables"]) > limits.max_variables
        ):
            raise ValueError("Invalid metadata schema")
        total = 0
        for item in value["variables"]:
            if not isinstance(item, dict) or set(item) != {
                "name",
                "shape",
                "dtype",
                "sample_count",
                "declared_bytes",
            }:
                raise ValueError("Invalid variable schema")
            for field in ("name", "dtype"):
                if (
                    not isinstance(item[field], str)
                    or len(item[field].encode()) > limits.max_name_bytes
                    or "\0" in item[field]
                ):
                    raise ValueError("Invalid variable text")
            shape = item["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) > limits.max_dimensions
                or any(
                    type(size) is not int or not 0 <= size <= limits.max_declared_bytes
                    for size in shape
                )
                or type(item["declared_bytes"]) is not int
                or item["declared_bytes"] < 0
                or (
                    shape
                    and (
                        type(item["sample_count"]) is not int
                        or item["sample_count"] != shape[0]
                    )
                )
                or (not shape and item["sample_count"] is not None)
            ):
                raise ValueError("Invalid variable dimensions")
            total += item["declared_bytes"]
        if total > limits.max_declared_bytes:
            raise ValueError("Declared arrays exceed limit")
        return value
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise InspectionRefused("Invalid dataset metadata") from error


class DatasetInspector:
    def __init__(
        self,
        python: str,
        cwd: Path,
        pool: BoundedThreadPool,
        *,
        budget: WorkBudget | None = None,
        limits: InspectionLimits | None = None,
        catalog: CatalogService | None = None,
    ):
        if (
            not isinstance(python, str)
            or not Path(python).is_absolute()
            or "\0" in python
        ):
            raise ValueError(
                "Inspection requires a server-selected absolute interpreter"
            )
        self.python, self.cwd, self.pool, self.authority = (
            python,
            Path(cwd),
            pool,
            (
                PreviewAuthority(
                    dict(catalog.roots),
                    extensions={".npy", ".npz", ".mat", ".h5", ".hdf5"},
                )
                if catalog is not None
                else None
            ),
        )
        self.budget, self.limits = budget or pool.budget, limits or InspectionLimits()
        if self.budget.max_output_bytes < 2 * self.limits.max_output_bytes:
            raise ValueError("Command output budget must cover inspection metadata")

    async def inspect(
        self, root_or_path: str, relative_path: str | None = None
    ) -> dict:
        if self.authority is None and relative_path is None:
            path = root_or_path
            if (
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or "\0" in path
                or len(path.encode()) > 8192
            ):
                raise InspectionRefused("Invalid dataset path")
        elif self.authority is not None and relative_path is not None:
            if not isinstance(root_or_path, str) or not isinstance(relative_path, str):
                raise InspectionRefused("Invalid dataset selection")
        else:
            raise InspectionRefused("Invalid dataset selection")

        def work(token):
            if self.authority is not None:
                try:
                    with self.authority.open_preview_selection(
                        root_or_path, relative_path
                    ) as selection:
                        token.check()
                        result = self._inspect_path(
                            str(selection.path),
                            token,
                            expected_identity=selection.identity,
                        )
                        verify_selection(selection)
                        return result
                except (
                    PreviewChanged,
                    PreviewForbidden,
                    PreviewMissing,
                    OSError,
                ):
                    raise InspectionRefused("Dataset selection refused") from None
            return self._inspect_path(path, token)

        handle = self.pool.submit(work)
        try:
            return await handle.wait()
        except asyncio.CancelledError:
            # Keep request cancellation from abandoning a live inspection child.
            handle.cancel()
            try:
                await asyncio.shield(handle.wait())
            except Exception:
                pass
            raise

    def _inspect_path(self, path: str, token, *, expected_identity=None) -> dict:
        argv = [self.python, "-m", "wavedl.core.dataset_headers", "--path", path]
        if expected_identity is not None:
            argv.extend(
                (
                    "--expected-identity",
                    json.dumps(expected_identity, separators=(",", ":")),
                )
            )
        for name, value in asdict(self.limits).items():
            argv.extend(("--" + name.replace("_", "-"), str(value)))
        result = run_command(
            ProcessSpec(tuple(argv), self.cwd), self.budget, token=token
        )
        if (
            result.returncode
            or result.timed_out
            or result.cancelled
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            raise InspectionRefused("Dataset inspection refused or unavailable")
        return validate_metadata(result.stdout, self.limits)
