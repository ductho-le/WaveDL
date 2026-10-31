# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Authenticated bounded model/browse routes assembled from explicit services."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.responses import Response

from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.http.app import strict_json_route
from wavedl_daemon.http.schemas.catalog import (
    BrowseInteger,
    BrowsePage,
    BrowseRoots,
    ModelCatalog,
    ResolvedSelection,
    ResolveRequest,
)
from wavedl_daemon.http.schemas.common import (
    API_ERRORS,
    JSON_BODY_ERRORS,
    NO_STORE,
    ErrorResponse,
    exact_query,
)
from wavedl_daemon.runs.catalog import BrowseForbidden, BrowseMissing


if TYPE_CHECKING:
    from wavedl_daemon.runs.catalog import AsyncCatalogService, CatalogService


async def _call(callback, *args):
    try:
        result = callback(*args)
        return await result if inspect.isawaitable(result) else result
    except BrowseForbidden:
        raise HTTPException(403, "Browse location is not allowed") from None
    except BrowseMissing:
        raise HTTPException(404, "Browse location does not exist") from None
    except PermissionError:
        raise HTTPException(403, "Browse location is unreadable") from None
    except FileNotFoundError:
        raise HTTPException(404, "Browse location does not exist") from None
    except (BudgetExhausted, WorkCancelled, StorageWorkError, OSError):
        raise HTTPException(503, "Catalog service unavailable") from None
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid catalog request") from None


def catalog_routes(service: CatalogService | AsyncCatalogService) -> list[APIRoute]:
    async def models(response: Response):
        try:
            result = await _call(service.models)
        except HTTPException:
            # Metadata belongs to the installed package, not the browse caller.
            raise HTTPException(503, "Bundled model metadata unavailable") from None
        response.headers.update(NO_STORE)
        return result

    async def browse(
        response: Response,
        root: str | None = None,
        path: str = "",
        limit: Annotated[BrowseInteger, Query(ge=1, le=200)] = 100,
        offset: Annotated[BrowseInteger, Query(ge=0)] = 0,
    ):
        result = await _call(service.browse, root, path, limit, offset)
        response.headers.update(NO_STORE)
        return result

    async def resolve(body: ResolveRequest, response: Response):
        result = await _call(service.resolve, body.root, body.path)
        response.headers.update(NO_STORE)
        return result

    router = APIRouter()
    router.add_api_route(
        "/api/models",
        models,
        operation_id="list_models",
        responses=API_ERRORS,
        methods=["GET", "HEAD"],
        response_model=ModelCatalog,
        dependencies=[Depends(exact_query())],
    )
    router.add_api_route(
        "/api/browse",
        browse,
        operation_id="browse_files",
        responses=API_ERRORS | {403: {"model": ErrorResponse}},
        methods=["GET", "HEAD"],
        response_model=BrowseRoots | BrowsePage,
        dependencies=[Depends(exact_query("root", "path", "limit", "offset"))],
    )
    router.add_api_route(
        "/api/browse/resolve",
        resolve,
        operation_id="resolve_browse",
        responses=JSON_BODY_ERRORS | {403: {"model": ErrorResponse}},
        methods=["POST"],
        response_model=ResolvedSelection,
        route_class_override=strict_json_route(8192),
    )
    return [route for route in router.routes if isinstance(route, APIRoute)]
