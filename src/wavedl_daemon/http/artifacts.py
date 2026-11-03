# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""List and download a run's ``output/``, with filesystem work off the loop."""

from __future__ import annotations

import re
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.responses import Response, StreamingResponse

from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.http.schemas.artifacts import ArtifactListing
from wavedl_daemon.http.schemas.common import API_ERRORS, NO_STORE, ErrorResponse
from wavedl_daemon.runs.artifacts import (
    ArtifactNotFound,
    UnsafeArtifactPath,
    list_output,
    open_output,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from wavedl_daemon.execution.pool import BoundedThreadPool
    from wavedl_daemon.runs.layout import RunPaths


class _DownloadOwner:
    """Cancellation transfers cleanup to the worker until its I/O really ends."""

    def __init__(self):
        self.lock = threading.Lock()
        self.artifact = None
        self.working = False
        self.abandoned = False

    def run(self, callback):
        with self.lock:
            if self.abandoned:
                raise ValueError("Download closed")
            self.working = True
        try:
            return callback()
        finally:
            with self.lock:
                self.working = False
                artifact = self.artifact if self.abandoned else None
                if artifact is not None:
                    self.artifact = None
            if artifact is not None:
                artifact.close()

    def close(self):
        with self.lock:
            self.abandoned = True
            artifact = self.artifact if not self.working else None
            if artifact is not None:
                self.artifact = None
        if artifact is not None:
            artifact.close()


def _artifact_error(error):
    if isinstance(error, StorageWorkError) and error.__cause__ is not None:
        error = error.__cause__
    if isinstance(error, (KeyError, ArtifactNotFound)):
        return HTTPException(404, "Not found")
    if isinstance(error, UnsafeArtifactPath):
        return HTTPException(400, "Invalid artifact path")
    if isinstance(error, (PermissionError, ValueError)):
        return HTTPException(403, "Forbidden")
    return HTTPException(503, "Artifact storage unavailable")


def artifact_routes(
    pool: BoundedThreadPool, resolve: Callable[[str], Awaitable[RunPaths]]
) -> list[APIRoute]:
    """Mount only inside the bearer/Host/Origin-protected daemon app."""

    async def output(run_id: str) -> Path:
        try:
            canonical = str(UUID(run_id))
        except (ValueError, TypeError, AttributeError):
            raise UnsafeArtifactPath("Invalid run identity") from None
        if canonical != run_id:
            raise UnsafeArtifactPath("Invalid run identity")
        return (await resolve(run_id)).output

    async def listing(run_id: str, response: Response):
        try:
            root = await output(run_id)
            inventory = await pool.submit(lambda token: list_output(root)).wait()
        except (
            BudgetExhausted,
            WorkCancelled,
            KeyError,
            ValueError,
            OSError,
            StorageWorkError,
        ) as error:
            raise _artifact_error(error) from None
        response.headers.update(NO_STORE)
        return {
            "files": [asdict(item) for item in inventory.files],
            "checkpoints": list(inventory.checkpoints),
        }

    async def download(run_id: str, path: str, request: Request):
        owner = _DownloadOwner()

        def open_artifact(root):
            owner.artifact = open_output(root, path)
            return owner.artifact

        try:
            root = await output(run_id)
            artifact = await pool.submit(
                lambda token: owner.run(lambda: open_artifact(root))
            ).wait()
        except (
            BudgetExhausted,
            WorkCancelled,
            KeyError,
            ValueError,
            OSError,
            StorageWorkError,
        ) as error:
            owner.close()
            raise _artifact_error(error) from None
        except BaseException:
            owner.close()
            raise

        filename = re.sub(r"[^A-Za-z0-9._-]", "_", artifact.filename)[:160]
        filename = filename.strip(".") or "artifact"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(artifact.size),
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        }
        if request.method == "HEAD":
            owner.close()
            return Response(media_type="application/octet-stream", headers=headers)

        async def chunks():
            try:
                while chunk := await pool.submit(
                    lambda token: owner.run(artifact.read)
                ).wait():
                    yield chunk
            finally:
                owner.close()

        return _OwnedStreamingResponse(
            chunks(),
            owner=owner,
            media_type="application/octet-stream",
            headers=headers,
        )

    router = APIRouter()
    errors = API_ERRORS | {403: {"model": ErrorResponse}}
    router.add_api_route(
        "/api/runs/{run_id}/artifacts",
        listing,
        operation_id="list_artifacts",
        responses=errors,
        methods=["GET", "HEAD"],
        response_model=ArtifactListing,
    )
    router.add_api_route(
        "/api/runs/{run_id}/artifacts/{path:path}",
        download,
        operation_id="download_artifact",
        responses=errors,
        methods=["GET", "HEAD"],
        response_class=StreamingResponse,
        response_model=None,
    )
    return [route for route in router.routes if isinstance(route, APIRoute)]


class _OwnedStreamingResponse(StreamingResponse):
    def __init__(self, *args, owner, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.owner.close()
