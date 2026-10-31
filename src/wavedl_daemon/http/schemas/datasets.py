# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Dataset selections and validated child-process response views."""

from __future__ import annotations

from pydantic import JsonValue

from wavedl_daemon.http.schemas.common import WireModel


class InspectRequest(WireModel):
    root: str
    path: str


class PreviewRequest(InspectRequest):
    # Domain view selection remains in DatasetPreviewService.
    view: str


class DatasetVariable(WireModel):
    name: str
    shape: list[int]
    dtype: str
    sample_count: int | None
    declared_bytes: int


class InspectionResponse(WireModel):
    schema_version: int
    format: str
    variables: list[DatasetVariable]


class DatasetIdentity(WireModel):
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int


class Histogram(WireModel):
    feature_index: int
    counts: list[int]
    edges: list[int | float]


class Correlation(WireModel):
    feature_indices: list[int]
    matrix: list[list[int | float | None]] | None


class InputStatistics(WireModel):
    feature_indices: list[int]
    means: list[int | float | None]
    stds: list[int | float | None]


class Scatter(WireModel):
    feature_indices: list[int]
    output_index: int
    x: list[list[int | float | None]]
    y: list[int | float | None]


class PreviewSample(WireModel):
    sample_index: int
    feature_shape: list[int]
    shape: list[int]
    values: JsonValue


class PreviewResponse(WireModel):
    schema_version: int
    identity: DatasetIdentity
    format: str
    input_key: str
    output_key: str
    input_shape: list[int]
    output_shape: list[int]
    sample_indices: list[int]
    input_histograms: list[Histogram]
    output_histograms: list[Histogram]
    correlation: Correlation
    input_statistics: InputStatistics
    scatter: Scatter
    samples: list[PreviewSample]
    limits: dict[str, int]
    selection: InspectRequest
    view: str
