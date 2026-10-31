# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Construct an inert, authenticated ASGI app from explicitly owned services."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse

from wavedl_daemon.http.auth import (
    DEFAULT_ALLOWED_HOSTS,
    authority,
    bearer_token,
    credential_bytes,
    require_auth,
    validate_origin_host,
)
from wavedl_daemon.http.bootstrap import (
    BootstrapCredentials,
    install_access_log_redaction,
)
from wavedl_daemon.http.services import DaemonServices, require_services


MAX_JSON_BODY = 1024 * 1024
JSON_BODY_TIMEOUT_SECONDS = 10.0


class SecurityBoundary:
    """Pure ASGI checks also cover routing errors, without buffering streams."""

    def __init__(
        self,
        app,
        *,
        token: bytes,
        allowed_hosts: tuple[str, ...],
        bootstrap_enabled: bool = False,
        bootstrap_credentials=None,
    ):
        self.app, self.allowed_hosts = app, allowed_hosts
        self.bootstrap_enabled = bootstrap_enabled
        self.bootstrap_credentials = bootstrap_credentials
        self.token = b"" if bootstrap_credentials is not None else token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        try:
            validate_origin_host(request, self.allowed_hosts)
            if scope["path"] == "/api" or scope["path"].startswith("/api/"):
                identities = request.headers.getlist("x-wavedl-daemon-id")
                if identities:
                    state = getattr(scope.get("app"), "state", None)
                    expected = getattr(state, "daemon_id", None)
                    if len(identities) != 1 or identities[0] != expected:
                        raise HTTPException(
                            409,
                            "Daemon identity changed",
                            headers={"X-WaveDL-Error-Code": "daemon_identity_changed"},
                        )
                if not (
                    self.bootstrap_enabled
                    and scope["path"] == "/api/bootstrap"
                    and scope["method"] == "POST"
                ):
                    master = require_auth(
                        request, self.token, sessions=self.bootstrap_credentials
                    )
                    if (
                        scope["method"] == "POST"
                        and scope["path"].startswith("/api/runs/")
                        and scope["path"].rstrip("/").endswith("/events")
                        and not master
                    ):
                        raise HTTPException(403, "Forbidden")
                    scope["wavedl.auth_master"] = master
                    scope["wavedl.auth_token"] = bearer_token(request)
        except HTTPException as exc:
            await JSONResponse(
                {"error": exc.detail}, exc.status_code, headers=exc.headers
            )(scope, receive, send)
            return

        async def without_cookies(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"set-cookie"
                ]
            await send(message)

        await self.app(scope, receive, without_cookies)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _invalid_constant(value):
    raise ValueError("Non-finite JSON number")


def _decode_json_bytes(body: bytes) -> dict:
    """Decode one strict JSON object without exposing parser diagnostics."""
    try:
        result = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_float=_finite_float,
            parse_constant=_invalid_constant,
        )
        if not isinstance(result, dict):
            raise ValueError("Expected object")
    except (ValueError, RecursionError):
        raise HTTPException(400, "Invalid JSON") from None
    return result


async def read_json_bytes(
    request: Request,
    *,
    max_bytes: int,
    timeout_seconds: float = JSON_BODY_TIMEOUT_SECONDS,
) -> bytes:
    """Read a bounded request body without decoding or exposing submitted data."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("JSON byte limit must be positive")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("JSON body timeout must be positive and finite")
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise HTTPException(415, "Expected JSON")
    body = bytearray()
    try:
        async with asyncio.timeout(timeout_seconds):
            async for chunk in request.stream():
                if len(body) + len(chunk) > max_bytes:
                    raise HTTPException(413, "Request too large")
                body.extend(chunk)
    except TimeoutError:
        raise HTTPException(408, "Request body timeout") from None
    except ClientDisconnect:
        # A truncated body is not valid JSON, but the peer is gone and cannot
        # receive a detailed parse error.  Keep the boundary deterministic and
        # avoid leaking Starlette's transport exception as a 500.
        raise HTTPException(408, "Request body disconnected") from None
    return bytes(body)


async def read_json(
    request: Request,
    *,
    max_bytes: int = MAX_JSON_BODY,
    timeout_seconds: float = JSON_BODY_TIMEOUT_SECONDS,
) -> dict:
    """Read bounded strict JSON; never return decoder details or submitted secrets."""
    return _decode_json_bytes(
        await read_json_bytes(
            request, max_bytes=max_bytes, timeout_seconds=timeout_seconds
        )
    )


def strict_json_route(max_bytes: int) -> type[APIRoute]:
    """Create an APIRoute that enforces the daemon's strict JSON boundary."""

    class StrictJSONRoute(APIRoute):
        # FastAPI accepts duplicate keys and NaN, with no byte or time bound.
        def get_route_handler(self):
            original_handler = super().get_route_handler()

            async def strict_handler(request: Request):
                body = await read_json_bytes(request, max_bytes=max_bytes)
                _decode_json_bytes(body)
                replayed = False
                original_receive = request.receive

                async def replay_receive():
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }
                    return await original_receive()

                return await original_handler(Request(request.scope, replay_receive))

            return strict_handler

    return StrictJSONRoute


async def _http_error(request, exc):
    messages = {
        400: "Bad request",
        408: "Request timeout",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not found",
        405: "Method not allowed",
        409: "Conflict",
        413: "Request too large",
        415: "Expected JSON",
        422: "Invalid request",
    }
    return JSONResponse(
        {"error": messages.get(exc.status_code, "Request failed")},
        exc.status_code,
        headers=exc.headers,
    )


async def _server_error(request, exc):
    return JSONResponse({"error": "Internal server error"}, 500)


async def _validation_error(request, exc) -> JSONResponse:
    return JSONResponse({"error": "Bad request"}, 400)


def create_app(
    settings,
    *,
    token: str,
    daemon_id: str,
    build_id: str,
    services: DaemonServices | None = None,
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
    bootstrap_credentials=None,
    web_assets=None,
) -> FastAPI:
    """No disk access, credentials generation, background work, or listener binding."""
    if services is not None:
        require_services(services)
    credential = credential_bytes(token)
    bootstrap_enabled = bootstrap_credentials is not None
    credential_pool = bootstrap_credentials or BootstrapCredentials(token)
    if bootstrap_enabled and not credential_pool.matches_token(token):
        raise ValueError("Bootstrap credentials must belong to this daemon")
    if not allowed_hosts:
        raise ValueError("At least one explicit allowed host is required")
    allowed_hosts = tuple(allowed_hosts)
    for host in allowed_hosts:
        authority(host)
    if not all(
        isinstance(value, str) and value and len(value) <= 256
        for value in (daemon_id, build_id)
    ):
        raise ValueError("Invalid daemon identity")

    from wavedl_daemon.http.session import session_routes

    routes = session_routes(
        credential_pool,
        services,
        bootstrap_enabled=bootstrap_enabled,
        build_id=build_id,
    )

    if services is not None:
        from wavedl_daemon.http.adoption import adoption_routes
        from wavedl_daemon.http.artifacts import artifact_routes
        from wavedl_daemon.http.catalog import catalog_routes
        from wavedl_daemon.http.events import event_routes
        from wavedl_daemon.http.host import host_routes
        from wavedl_daemon.http.inspection import inspection_routes
        from wavedl_daemon.http.preview import preview_routes
        from wavedl_daemon.http.runs import lifecycle_routes

        routes.extend(adoption_routes(services))
        routes.extend(inspection_routes(services))
        routes.extend(preview_routes(services))
        routes.extend(host_routes(services))
        routes.extend(lifecycle_routes(services.run_service))
        routes.extend(catalog_routes(services.catalog))
        routes.extend(
            event_routes(
                ingestor=services.ingestor,
                event_log=services.event_log,
                broker=services.broker,
                resolve_plan=services.resolve_plan,
            )
        )
        routes.extend(
            artifact_routes(
                services.artifacts,
                services.pool,
                resolve_layout=services.run_service.layout,
            )
        )

    from wavedl_daemon.http.static import static_routes

    routes.extend(static_routes(web_assets))

    @asynccontextmanager
    async def lifespan(app):
        filters = []
        try:
            for name in ("uvicorn.access", "uvicorn.error"):
                logger = logging.getLogger(name)
                filters.append(
                    (
                        logger,
                        install_access_log_redaction(
                            credential_pool,
                            logger,
                            master_token=token,
                        ),
                    )
                )
            if services is not None:
                await services.start()
                identity = services.daemon_id
                if not isinstance(identity, str) or not 1 <= len(identity) <= 256:
                    raise ValueError("Invalid owned daemon identity")
                app.state.daemon_id = identity
            yield
        finally:
            try:
                if services is not None:
                    await services.close()
            finally:
                for logger, redactor in filters:
                    logger.removeFilter(redactor)

    app = FastAPI(
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
        routes=routes,
        lifespan=lifespan,
        middleware=[
            Middleware(
                SecurityBoundary,
                token=credential,
                allowed_hosts=allowed_hosts,
                bootstrap_enabled=bootstrap_enabled,
                bootstrap_credentials=credential_pool,
            )
        ],
        exception_handlers={
            HTTPException: _http_error,
            Exception: _server_error,
            RequestValidationError: _validation_error,
        },
    )
    app.state.settings, app.state.services = settings, services
    app.state.auth_token = credential
    app.state.daemon_id = daemon_id
    return app
