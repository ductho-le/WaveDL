# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Authenticated session routes and the explicit launch-capability exception."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from wavedl_daemon.http.app import strict_json_route
from wavedl_daemon.http.bootstrap import (
    BootstrapCapacityError,
    BootstrapCredentials,
    BootstrapExchangeError,
)
from wavedl_daemon.http.runs import _call
from wavedl_daemon.http.schemas.common import API_ERRORS, JSON_BODY_ERRORS, NO_STORE
from wavedl_daemon.http.schemas.session import (
    BootstrapRequest,
    HealthResponse,
    SessionGrantResponse,
)
from wavedl_daemon.http.services import DaemonServices


def session_routes(
    credential_pool: BootstrapCredentials,
    services: DaemonServices | None,
    *,
    bootstrap_enabled: bool,
    build_id: str,
) -> list[APIRoute]:
    async def health(request: Request, response: Response):
        status = {} if services is None else await _call(services.health)
        response.headers.update(NO_STORE)
        return {
            **status,
            "daemon_id": request.app.state.daemon_id,
            "build_id": build_id,
            "protocol_min": 1,
            "protocol_max": 1,
        }

    def grant(exchanged, request, response):
        response.headers.update(NO_STORE | {"Pragma": "no-cache"})
        return {
            "access_token": exchanged.token,
            "token_type": "Bearer",
            "expires_at": exchanged.expires_at,
            "daemon_id": request.app.state.daemon_id,
        }

    async def bootstrap_exchange(
        body: BootstrapRequest, request: Request, response: Response
    ):
        try:
            exchanged = credential_pool.exchange(body.code)
        except BootstrapExchangeError:
            raise HTTPException(401, "Unauthorized") from None
        except BootstrapCapacityError:
            raise HTTPException(503, "Session capacity reached") from None
        return grant(exchanged, request, response)

    async def rotate_session(request: Request, response: Response):
        try:
            exchanged = credential_pool.rotate(request.scope.get("wavedl.auth_token"))
        except BootstrapExchangeError:
            raise HTTPException(401, "Unauthorized") from None
        except BootstrapCapacityError:
            raise HTTPException(503, "Session capacity reached") from None
        return grant(exchanged, request, response)

    async def revoke_session(request: Request):
        credential_pool.revoke(request.scope.get("wavedl.auth_token"))
        return Response(status_code=204)

    router = APIRouter()
    router.add_api_route(
        "/api/health",
        health,
        operation_id="get_health",
        responses=API_ERRORS,
        methods=["GET", "HEAD"],
        response_model=HealthResponse,
        response_model_exclude_unset=True,
    )
    if bootstrap_enabled:
        router.add_api_route(
            "/api/bootstrap",
            bootstrap_exchange,
            operation_id="exchange_bootstrap",
            responses=JSON_BODY_ERRORS,
            methods=["POST"],
            response_model=SessionGrantResponse,
            route_class_override=strict_json_route(4096),
        )
    router.add_api_route(
        "/api/session",
        rotate_session,
        operation_id="rotate_session",
        responses=API_ERRORS,
        methods=["POST"],
        response_model=SessionGrantResponse,
    )
    router.add_api_route(
        "/api/session",
        revoke_session,
        operation_id="revoke_session",
        responses=API_ERRORS,
        methods=["DELETE"],
        response_model=None,
        status_code=204,
    )
    return [route for route in router.routes if isinstance(route, APIRoute)]
