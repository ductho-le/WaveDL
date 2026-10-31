# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bundled descriptions and explicitly scoped filesystem selections."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator

from wavedl_daemon.http.schemas.common import WireModel


def _integer_query(value: str) -> int:
    # Preserve int(query) syntax: Pydantic also accepts "1.0", unlike the route
    # before migration. Keep pagination acceptance unchanged.
    return int(value)


BrowseInteger = Annotated[int, BeforeValidator(_integer_query)]


class ResolveRequest(WireModel):
    root: str
    path: str


class BrowseRoot(WireModel):
    id: str


class BrowseRoots(WireModel):
    roots: list[BrowseRoot]


class BrowseEntry(WireModel):
    name: str
    kind: Literal["file", "directory"]


class BrowsePage(WireModel):
    root: str
    path: str
    entries: list[BrowseEntry]
    total: int
    limit: int
    offset: int


class ResolvedSelection(WireModel):
    path: str
    kind: Literal["file", "directory"]
    root: str
    relative_path: str


class ModelCategory(WireModel):
    short_description: str
    models: list[str]


class ModelDescription(WireModel):
    id: str
    display_name: str
    architecture_family: str
    author_institution: str
    year_published: int
    params_m: int | float
    size_mb: int | float
    short_description: str
    description: str
    best_for: str
    pros: list[str]
    cons: list[str]
    compute_tier: str
    tags: list[str]
    supported_dims: list[int]
    is_pretrained: bool


class ModelCatalog(WireModel):
    source: Literal["bundled_metadata"]
    runtime_verified: Literal[False]
    categories: dict[str, ModelCategory]
    models: list[ModelDescription]
