# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Inspection routes live inside the app's existing authentication boundary."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from wavedl.core.dataset_headers import InspectionRefused
from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.http.app import strict_json_route
from wavedl_daemon.http.schemas.common import JSON_BODY_ERRORS, NO_STORE
from wavedl_daemon.http.schemas.datasets import InspectionResponse, InspectRequest


def inspection_routes(inspector):
    async def inspect_dataset(
        body: InspectRequest, request: Request, response: Response
    ):
        async def disconnected():
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    return

        operation = asyncio.create_task(inspector.inspect(body.root, body.path))
        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait(
                {operation, disconnect}, return_when=asyncio.FIRST_COMPLETED
            )
            if operation not in done:
                operation.cancel()
                try:
                    await operation
                except asyncio.CancelledError:
                    pass
                raise HTTPException(503, "Dataset inspection unavailable")
            result = await operation
        except InspectionRefused:
            raise HTTPException(400, "Dataset inspection refused") from None
        except (BudgetExhausted, StorageWorkError, WorkCancelled, OSError):
            raise HTTPException(503, "Dataset inspection unavailable") from None
        finally:
            disconnect.cancel()
            if not operation.done():
                operation.cancel()
            await asyncio.gather(disconnect, operation, return_exceptions=True)
        response.headers.update(NO_STORE)
        return result

    router = APIRouter()
    router.add_api_route(
        "/api/datasets/inspect",
        inspect_dataset,
        operation_id="inspect_dataset",
        responses=JSON_BODY_ERRORS,
        methods=["POST"],
        response_model=InspectionResponse,
        route_class_override=strict_json_route(16384),
    )
    return [route for route in router.routes if isinstance(route, APIRoute)]
