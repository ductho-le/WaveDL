# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Authenticated-app artifact routes with bounded off-loop filesystem work."""

from __future__ import annotations

import re
import threading
from inspect import isawaitable
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote_to_bytes
from uuid import UUID

from fastapi import APIRouter
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.http.schemas.artifacts import ArtifactListing
from wavedl_daemon.http.schemas.common import API_ERRORS, NO_STORE, ErrorResponse
from wavedl_daemon.runs.artifacts import ArtifactNotFound, UnsafeArtifactPath


if TYPE_CHECKING:
    from wavedl_daemon.execution.pool import BoundedThreadPool
    from wavedl_daemon.runs.artifacts import ArtifactStore


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


def _validate_url_path(request):
    """Allow one UTF-8 decoding and RFC path characters, never separators.

    Legal pchars may be literal or percent-encoded: either form resolves to the
    same exact inventory member without any subsequent URL decoding.
    """
    raw = request.scope.get("raw_path")
    if raw is None:
        return
    if re.search(rb"%(?![0-9A-Fa-f]{2})|%2[fF]|%5[cC]", raw):
        raise HTTPException(400, "Invalid artifact path")
    try:
        decoded = unquote_to_bytes(raw).decode("utf-8", errors="strict")
        punctuation = "!$&'()*+,;=:@"
        canonical = quote(decoded, safe="/" + punctuation).encode("ascii")
    except (UnicodeError, ValueError):
        raise HTTPException(400, "Invalid artifact path") from None
    pchars = frozenset(
        (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
            + punctuation
        ).encode()
    )

    def normalize_escape(match):
        value = int(match[0][1:], 16)
        return bytes((value,)) if value in pchars else match[0].upper()

    normalized = re.sub(rb"%[0-9A-Fa-f]{2}", normalize_escape, raw)
    if (
        decoded != request.scope["path"]
        or canonical != normalized
        or any(ord(char) < 32 or ord(char) == 127 for char in decoded)
    ):
        raise HTTPException(400, "Invalid artifact path")


def artifact_routes(
    store: ArtifactStore, pool: BoundedThreadPool, *, resolve_layout=None
) -> list[APIRoute]:
    """Mount only inside the bearer/Host/Origin-protected daemon app."""

    async def resolve(request):
        run_id = request.path_params["run_id"]
        try:
            canonical = str(UUID(run_id))
        except (ValueError, TypeError, AttributeError):
            raise UnsafeArtifactPath("Invalid run identity") from None
        if canonical != run_id:
            raise UnsafeArtifactPath("Invalid run identity")
        if resolve_layout is None:
            return None
        layout = resolve_layout(run_id)
        if isawaitable(layout):
            layout = await layout
        if layout.run_id != run_id:
            raise ValueError("Resolved layout identity mismatch")
        return layout

    async def listing(run_id: str, request: Request, response: Response):
        try:
            layout = await resolve(request)
            records = await pool.submit(
                lambda token: (
                    store.list(request.path_params["run_id"])
                    if layout is None
                    else store.list_layout(layout)
                )
            ).wait()
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
            "generations": [
                {
                    "generation": item.generation,
                    "attempt": item.attempt,
                    "step": item.step,
                    "files": {
                        name: {"size": record.size, "sha256": record.sha256}
                        for name, record in item.files.items()
                    },
                }
                for item in records
            ]
        }

    async def download(run_id: str, generation: str, path: str, request: Request):
        _validate_url_path(request)
        owner = _DownloadOwner()
        layout = None

        def open_artifact():
            arguments = (request.path_params["generation"], request.path_params["path"])
            owner.artifact = (
                store.open(request.path_params["run_id"], *arguments)
                if layout is None
                else store.open_layout(layout, *arguments)
            )
            return owner.artifact

        try:
            layout = await resolve(request)
            artifact = await pool.submit(lambda token: owner.run(open_artifact)).wait()
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

        async def chunks():
            try:
                while chunk := await pool.submit(
                    lambda token: owner.run(lambda: artifact.read(65536))
                ).wait():
                    yield chunk
            finally:
                owner.close()

        filename = re.sub(r"[^A-Za-z0-9._-]", "_", artifact.filename)[:160]
        filename = filename.strip(".") or "artifact"
        response = _OwnedStreamingResponse(
            chunks(),
            owner=owner,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(artifact.size),
                "Content-Security-Policy": "sandbox; default-src 'none'",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )
        return response

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
        "/api/runs/{run_id}/artifacts/{generation}/{path:path}",
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
