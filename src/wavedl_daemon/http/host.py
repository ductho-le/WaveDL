# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Authenticated host profile used to prefill an automatic Local runtime."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.responses import Response

from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.http.schemas.common import API_ERRORS, NO_STORE, exact_query
from wavedl_daemon.http.schemas.host import HostProfile


def host_routes(services):
    async def host(response: Response):
        try:
            result = await services.host_profile()
        except (BudgetExhausted, StorageWorkError, WorkCancelled, OSError):
            raise HTTPException(503, "Host profile unavailable") from None
        response.headers.update(NO_STORE)
        return result

    router = APIRouter()
    router.add_api_route(
        "/api/host",
        host,
        operation_id="get_host",
        responses=API_ERRORS,
        methods=["GET", "HEAD"],
        response_model=HostProfile,
        dependencies=[Depends(exact_query())],
    )
    return [route for route in router.routes if isinstance(route, APIRoute)]
