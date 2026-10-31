# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Strict authenticated-app adoption endpoint, with no filesystem disclosures."""

from __future__ import annotations

import inspect
import sqlite3

from fastapi import APIRouter
from fastapi.routing import APIRoute
from pydantic import MISSING
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from wavedl.core.safe_files import UnsafeOwnership
from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.http.app import strict_json_route
from wavedl_daemon.http.schemas.adoption import AdoptionResponse, AdoptRequest
from wavedl_daemon.http.schemas.common import JSON_BODY_ERRORS, NO_STORE, ErrorResponse
from wavedl_daemon.storage.ownership import DispatchRequired
from wavedl_daemon.storage.plans import IdempotencyConflict, PlanRecoveryRequired


def adoption_routes(service):
    """Enclosing SecurityBoundary must authenticate this explicit operation."""

    async def adopt(body: AdoptRequest, request: Request, response: Response):
        if len(request.headers.getlist("idempotency-key")) != 1:
            raise HTTPException(400, "Invalid adoption request")
        try:
            result = service.adopt(
                body.path,
                attempt=None if body.attempt is MISSING else body.attempt,
                idempotency_key=request.headers["idempotency-key"],
            )
            if inspect.isawaitable(result):
                result = await result
        except IdempotencyConflict:
            raise HTTPException(409, "Adoption conflict") from None
        except (UnsafeOwnership, PermissionError):
            raise HTTPException(403, "Forbidden") from None
        except StorageWorkError as error:
            status = 403 if isinstance(error.__cause__, PermissionError) else 503
            raise HTTPException(status, "Adoption storage unavailable") from None
        except (
            BudgetExhausted,
            WorkCancelled,
            DispatchRequired,
            PlanRecoveryRequired,
            sqlite3.DatabaseError,
            OSError,
        ):
            raise HTTPException(503, "Adoption service unavailable") from None
        except (ValueError, TypeError, KeyError):
            raise HTTPException(400, "Invalid adoption request") from None
        response.headers.update(NO_STORE)
        return result.to_dict()

    router = APIRouter()
    router.add_api_route(
        "/api/runs/adopt",
        adopt,
        operation_id="adopt_run",
        responses=JSON_BODY_ERRORS | {403: {"model": ErrorResponse}},
        methods=["POST"],
        response_model=AdoptionResponse,
        route_class_override=strict_json_route(8192),
    )
    return [route for route in router.routes if isinstance(route, APIRoute)]
