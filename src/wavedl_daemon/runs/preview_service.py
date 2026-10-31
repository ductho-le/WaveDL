# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded daemon-owned dataset preview execution; no numerical imports."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.dataset_preview import PreviewLimits
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.runs.preview_authority import PreviewAuthority
from wavedl_daemon.supervision.base import ProcessSpec


if TYPE_CHECKING:
    from wavedl_daemon.execution.pool import BoundedThreadPool
    from wavedl_daemon.runs.catalog import CatalogService


class PreviewBadRequest(ValueError):
    """Invalid fixed preview request."""


class PreviewRefused(ValueError):
    """The selected dataset cannot be previewed."""


class PreviewIdentityChanged(PreviewRefused):
    """The selected file changed before or during preview."""


class PreviewTooLarge(PreviewRefused):
    """The dataset or preview exceeds a configured cap."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate preview key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Nonfinite preview value")


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("Invalid preview fields")


def _indices(value, width, maximum):
    if (
        type(value) is not list
        or len(value) > maximum
        or any(type(item) is not int or not 0 <= item < width for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError("Invalid preview indices")
    return value


def _number(value):
    if value is not None and (
        type(value) not in {int, float} or not math.isfinite(value)
    ):
        raise ValueError("Invalid preview number")


def _numbers(value, size):
    if type(value) is not list or len(value) != size:
        raise ValueError("Invalid preview vector")
    for item in value:
        _number(item)


def _shape(value, limits):
    if (
        type(value) is not list
        or not 1 <= len(value) <= 16
        or any(type(size) is not int or size < 1 for size in value)
        or math.prod(value) > limits.max_elements
    ):
        raise ValueError("Invalid preview shape")
    return value


def _nested(value, shape):
    if not shape:
        _number(value)
        return
    if type(value) is not list or len(value) != shape[0]:
        raise ValueError("Invalid sample shape")
    for item in value:
        _nested(item, shape[1:])


def _validate_value(value, limits):
    _keys(
        value,
        {
            "schema_version",
            "identity",
            "format",
            "input_key",
            "output_key",
            "input_shape",
            "output_shape",
            "sample_indices",
            "input_histograms",
            "output_histograms",
            "correlation",
            "input_statistics",
            "scatter",
            "samples",
            "limits",
        },
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["format"] not in {"npz", "mat", "h5", "hdf5"}
    ):
        raise ValueError("Invalid preview schema")
    _keys(value["identity"], {"st_dev", "st_ino", "st_size", "st_mtime_ns"})
    if (
        any(type(item) is not int or item < 0 for item in value["identity"].values())
        or value["identity"]["st_size"] > limits.max_file_bytes
    ):
        raise ValueError("Invalid preview identity")
    _keys(value["limits"], asdict(limits))
    if value["limits"] != asdict(limits) or any(
        type(item) is not int for item in value["limits"].values()
    ):
        raise ValueError("Invalid preview limits")
    for key in ("input_key", "output_key"):
        if (
            type(value[key]) is not str
            or not value[key]
            or "\0" in value[key]
            or len(value[key].encode()) > 1024
        ):
            raise ValueError("Invalid preview variable")
    xs, ys = (_shape(value[key], limits) for key in ("input_shape", "output_shape"))
    if xs[0] != ys[0]:
        raise ValueError("Unaligned preview shapes")
    xwidth, ywidth = math.prod(xs[1:]), math.prod(ys[1:])
    if xwidth > limits.max_input_features or ywidth > limits.max_output_features:
        raise ValueError("Preview feature limit")
    indices = _indices(value["sample_indices"], xs[0], limits.max_sample_rows)
    if not indices:
        raise ValueError("Empty preview")
    count = len(indices)
    for name, width in (("input_histograms", xwidth), ("output_histograms", ywidth)):
        histograms = value[name]
        if type(histograms) is not list or not 1 <= len(histograms) <= min(8, width):
            raise ValueError("Invalid histograms")
        features = []
        for histogram in histograms:
            _keys(histogram, {"feature_index", "counts", "edges"})
            features.append(histogram["feature_index"])
            counts, edges = histogram["counts"], histogram["edges"]
            if (
                type(counts) is not list
                or len(counts) != 32
                or any(type(item) is not int or item < 0 for item in counts)
                or sum(counts) > count
            ):
                raise ValueError("Invalid histogram counts")
            _numbers(edges, 33)
            if any(edge is None for edge in edges) or edges != sorted(edges):
                raise ValueError("Invalid histogram edges")
        _indices(features, width, 8)
    correlation = value["correlation"]
    _keys(correlation, {"feature_indices", "matrix"})
    features = _indices(correlation["feature_indices"], ywidth, 30)
    if len(features) <= 1 or count <= 1:
        if correlation["matrix"] is not None:
            raise ValueError("Invalid correlation")
    else:
        _nested(correlation["matrix"], [len(features), len(features)])
    stats = value["input_statistics"]
    _keys(stats, {"feature_indices", "means", "stds"})
    features = _indices(stats["feature_indices"], xwidth, 40)
    _numbers(stats["means"], len(features))
    _numbers(stats["stds"], len(features))
    if any(number is not None and number < 0 for number in stats["stds"]):
        raise ValueError("Invalid standard deviation")
    scatter = value["scatter"]
    _keys(scatter, {"feature_indices", "output_index", "x", "y"})
    features = _indices(scatter["feature_indices"], xwidth, 8)
    if type(scatter["output_index"]) is not int or scatter["output_index"] != 0:
        raise ValueError("Invalid scatter output")
    _nested(scatter["x"], [count, len(features)])
    _numbers(scatter["y"], count)
    samples = value["samples"]
    if type(samples) is not list or not 1 <= len(samples) <= min(3, count):
        raise ValueError("Invalid sample previews")
    elements = 0
    for sample in samples:
        _keys(sample, {"sample_index", "feature_shape", "shape", "values"})
        if (
            type(sample["sample_index"]) is not int
            or sample["sample_index"] not in indices
            or sample["feature_shape"] != xs[1:]
        ):
            raise ValueError("Invalid sample identity")
        shape = _shape(sample["shape"], limits)
        if len(shape) != max(1, len(xs) - 1) or any(
            size > original for size, original in zip(shape, xs[1:] or [1], strict=True)
        ):
            raise ValueError("Invalid sample dimensions")
        elements += math.prod(shape)
        _nested(sample["values"], shape)
    if elements > limits.max_preview_elements:
        raise ValueError("Preview sample limit")


def validate_preview(content: bytes, limits: PreviewLimits) -> dict:
    if len(content) > limits.max_output_bytes:
        raise PreviewTooLarge("Preview output exceeds limit")
    try:
        value = json.loads(content, object_pairs_hook=_pairs, parse_constant=_constant)
        _validate_value(value, limits)
        return value
    except (ValueError, TypeError, RecursionError, UnicodeDecodeError, OverflowError):
        raise PreviewRefused("Invalid dataset preview") from None


class DatasetPreviewService:
    """Owner-loop service; numerical and filesystem work stays in bounded workers."""

    ALLOWED_VIEWS = frozenset(
        {"distribution", "sample", "correlation", "statistics", "scatter", "all"}
    )

    def __init__(
        self,
        python: str,
        cwd: Path,
        pool: BoundedThreadPool,
        catalog: CatalogService,
        *,
        limits: PreviewLimits | None = None,
    ):
        if type(python) is not str or not Path(python).is_absolute() or "\0" in python:
            raise ValueError("Preview requires a server-selected absolute interpreter")
        self.python, self.cwd, self.pool = python, Path(cwd), pool
        self.authority = PreviewAuthority(dict(catalog.roots))
        self.limits = limits or replace(
            PreviewLimits(),
            max_output_bytes=min(
                PreviewLimits().max_output_bytes,
                pool.budget.max_output_bytes // 2 - 1024,
            ),
        )
        if pool.budget.max_output_bytes // 2 < self.limits.max_output_bytes + 1024:
            raise ValueError("Command output budget must cover preview response")

    async def preview(self, request: dict) -> dict:
        if (
            type(request) is not dict
            or set(request) != {"root", "path", "view"}
            or type(request["root"]) is not str
            or type(request["path"]) is not str
            or type(request["view"]) is not str
            or request["view"] not in self.ALLOWED_VIEWS
        ):
            raise PreviewBadRequest("Invalid preview request")

        def work(token):
            with self.authority.open_preview_selection(
                request["root"], request["path"]
            ) as selected:
                token.check()
                expected_identity = selected.identity
                if expected_identity["st_size"] > self.limits.max_file_bytes:
                    raise PreviewTooLarge("Dataset file exceeds limit")
                argv = [
                    self.python,
                    "-m",
                    "wavedl.core.dataset_preview",
                    "--path",
                    str(selected.path),
                    "--expected-identity",
                    json.dumps(selected.expected, separators=(",", ":")),
                ]
                for name, value in asdict(self.limits).items():
                    argv.extend(("--" + name.replace("_", "-"), str(value)))
                result = run_command(
                    ProcessSpec(tuple(argv), self.cwd), self.pool.budget, token=token
                )
                if result.timed_out or result.cancelled:
                    raise BudgetExhausted("Preview deadline or cancellation")
                if result.stdout_truncated or result.stderr_truncated:
                    raise PreviewTooLarge("Preview process output exceeds limit")
                if result.returncode:
                    try:
                        code = json.loads(result.stdout).get("code")
                    except (ValueError, AttributeError):
                        code = None
                    if code == "identity_changed":
                        raise PreviewIdentityChanged("Dataset identity changed")
                    if code == "too_large":
                        raise PreviewTooLarge("Preview exceeds limit")
                    raise PreviewRefused("Dataset preview refused")
                value = validate_preview(result.stdout, self.limits)
                if value.get("identity") != expected_identity:
                    raise PreviewIdentityChanged("Dataset identity changed")
                value.update(
                    selection={"root": request["root"], "path": request["path"]},
                    view=request["view"],
                )
                if (
                    len(
                        json.dumps(
                            value, allow_nan=False, separators=(",", ":")
                        ).encode()
                    )
                    > self.limits.max_output_bytes
                ):
                    raise PreviewTooLarge("Preview response exceeds limit")
                return value

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
