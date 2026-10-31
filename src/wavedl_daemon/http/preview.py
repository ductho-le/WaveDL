# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Authenticated dataset preview endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.http.app import strict_json_route
from wavedl_daemon.http.schemas.common import JSON_BODY_ERRORS, NO_STORE
from wavedl_daemon.http.schemas.datasets import PreviewRequest, PreviewResponse
from wavedl_daemon.runs.preview_authority import (
    PreviewChanged,
    PreviewForbidden,
    PreviewMissing,
)
from wavedl_daemon.runs.preview_service import (
    PreviewBadRequest,
    PreviewIdentityChanged,
    PreviewRefused,
    PreviewTooLarge,
)


def preview_routes(previewer):
    async def preview_dataset(
        body: PreviewRequest, request: Request, response: Response
    ):
        async def disconnected():
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    return

        operation = asyncio.create_task(previewer.preview(body.model_dump(mode="json")))
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
                raise HTTPException(503, "Dataset preview unavailable")
            result = await operation
        except (PreviewIdentityChanged, PreviewChanged):
            raise HTTPException(409, "Dataset identity changed") from None
        except PreviewTooLarge:
            raise HTTPException(413, "Preview exceeds limit") from None
        except (PreviewBadRequest, PreviewForbidden, PreviewMissing):
            raise HTTPException(400, "Invalid preview request") from None
        except PreviewRefused:
            raise HTTPException(400, "Dataset preview refused") from None
        except (BudgetExhausted, StorageWorkError, WorkCancelled, OSError):
            raise HTTPException(503, "Dataset preview unavailable") from None
        finally:
            disconnect.cancel()
            if not operation.done():
                operation.cancel()
            await asyncio.gather(disconnect, operation, return_exceptions=True)
        response.headers.update(NO_STORE)
        return result

    router = APIRouter()
    router.add_api_route(
        "/api/datasets/preview",
        preview_dataset,
        operation_id="preview_dataset",
        responses=JSON_BODY_ERRORS,
        methods=["POST"],
        response_model=PreviewResponse,
        route_class_override=strict_json_route(65536),
    )
    return [route for route in router.routes if isinstance(route, APIRoute)]
