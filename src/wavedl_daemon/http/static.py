# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded same-origin static client reads; construction never touches disk."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
from contextlib import ExitStack
from pathlib import Path

import anyio
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.routing import Match, Route

from wavedl.core.safe_files import open_directory


MAX_ASSET_BYTES = 32 * 1024 * 1024
_INLINE_SCRIPT = re.compile(rb"<script\b([^>]*)>(.*?)</script\s*>", re.I | re.S)
_UNREGISTER_SERVICE_WORKER = b"""self.addEventListener("install", event => {
  event.waitUntil(self.skipWaiting());
});
self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    await self.clients.claim();
    await self.registration.unregister();
  })());
});
"""


def _headers(raw: bytes, *, html: bool, immutable: bool) -> dict[str, str]:
    scripts = []
    if html:
        for attributes, content in _INLINE_SCRIPT.findall(raw):
            if not re.search(rb"\bsrc\s*=", attributes, re.I):
                digest = base64.b64encode(hashlib.sha256(content).digest()).decode()
                scripts.append(f"'sha256-{digest}'")
    return {
        "Cache-Control": (
            "no-store"
            if html
            else "public, max-age=31536000, immutable"
            if immutable
            else "no-cache"
        ),
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self' "
            + " ".join(scripts)
            + "; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            "font-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        ),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }


class _ClientRoute(Route):
    """Do not consume unknown API routes, including subsequently added routes."""

    def matches(self, scope):
        path = scope.get("path", "")
        if path == "/api" or path.startswith("/api/"):
            return Match.NONE, {}
        return super().matches(scope)


class StaticClient:
    """Read an optional packaged build through retained no-follow ancestry."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.limiter = anyio.CapacityLimiter(4)

    def _read(self, parts: tuple[str, ...], *, limit: int | None = None) -> bytes:
        with ExitStack() as stack:
            directory = stack.enter_context(open_directory(self.root))
            for part in parts[:-1]:
                directory = stack.enter_context(directory.open_directory(part))
            return directory.read_file(
                parts[-1], limit=MAX_ASSET_BYTES if limit is None else limit
            )

    def _load(self, parts: tuple[str, ...], navigation: bool):
        try:
            raw = self._read(parts)
            name = parts[-1]
        except FileNotFoundError:
            if not navigation:
                raise HTTPException(404, "Not found") from None
            try:
                raw = self._read(("index.html",))
                name = "index.html"
            except FileNotFoundError:
                raise HTTPException(
                    503,
                    "Web client build unavailable",
                    headers={"Cache-Control": "no-store"},
                ) from None
            except (OSError, ValueError):
                raise HTTPException(404, "Not found") from None
        except (OSError, ValueError):
            raise HTTPException(404, "Not found") from None
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        html = media_type == "text/html"
        # Only content-verified digest names are immutable; all other assets
        # remain revalidated across upgrades.
        digest = hashlib.sha256(raw).hexdigest()
        hashes = re.findall(r"(?:^|[.-])([a-f0-9]{12,64})(?=\.)", name)
        immutable = not html and any(digest.startswith(value) for value in hashes)
        return raw, media_type, _headers(raw, html=html, immutable=immutable)

    async def serve(self, request):
        path = request.path_params["path"]
        if path.endswith("/"):
            path = path[:-1]
        parts = tuple(path.split("/")) if path else ("index.html",)
        if any(
            not part
            or part in (".", "..")
            or any(char in part for char in "\\:\0")
            or any(ord(char) < 32 or ord(char) == 127 for char in part)
            for part in parts
        ):
            raise HTTPException(404, "Not found")
        if parts == ("sw.js",):
            response = Response(
                _UNREGISTER_SERVICE_WORKER,
                media_type="application/javascript",
                headers={
                    "Cache-Control": "no-store",
                    "Content-Security-Policy": "default-src 'none'; base-uri 'none'",
                    "Referrer-Policy": "no-referrer",
                    "X-Content-Type-Options": "nosniff",
                },
            )
            if request.method == "HEAD":
                response.body = b""
            return response
        # Missing asset URLs are never the shell. Navigation must explicitly
        # accept HTML, while / is always a shell request.
        navigation = not path or (
            "text/html" in request.headers.get("accept", "") and "." not in parts[-1]
        )
        try:
            raw, media_type, headers = await anyio.to_thread.run_sync(
                self._load, parts, navigation, limiter=self.limiter
            )
        except HTTPException as error:
            if error.status_code != 503:
                raise
            response = Response(
                "Web client build unavailable. Generate and vendor the web client "
                "before building the daemon package.",
                status_code=503,
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-store",
                    "Referrer-Policy": "no-referrer",
                    "X-Content-Type-Options": "nosniff",
                    "Content-Security-Policy": (
                        "default-src 'none'; base-uri 'none'; frame-ancestors 'none'"
                    ),
                },
            )
        else:
            response = Response(raw, media_type=media_type, headers=headers)
        if request.method == "HEAD":
            response.body = b""
        return response


def static_routes(root: Path | None = None) -> list[Route]:
    """Register the final public route without resolving or opening assets."""
    client = StaticClient(
        root if root is not None else Path(__file__).parent.parent / "web_assets"
    )
    return [_ClientRoute("/{path:path}", client.serve, methods=["GET", "HEAD"])]
