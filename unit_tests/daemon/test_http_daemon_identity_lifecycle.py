"""Daemon identity exposure across real hosted app lifecycles."""

from __future__ import annotations

import asyncio
import os
import shutil

import httpx

from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


def test_successive_hosted_app_startups_reuse_and_replace_owner_identity(tmp_path):
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    settings.state_root.mkdir(mode=0o700, parents=True)
    token = "lifecycle-master"

    async def startup_identity(fallback: str) -> str:
        services = HostedServices(settings)
        app = create_app(
            settings,
            token=token,
            daemon_id=fallback,
            build_id="test",
            services=services,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
            ) as client,
        ):
            response = await client.get(
                "/api/health", headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 200
            assert response.json()["daemon_id"] == services.daemon_id
            return response.json()["daemon_id"]

    async def scenario() -> None:
        first = await startup_identity("first-fallback")
        second = await startup_identity("second-fallback")
        assert second == first

        registry = settings.state_root / "registry.db"
        replacement = settings.state_root / "registry-replacement.db"
        shutil.copyfile(registry, replacement)
        assert os.stat(registry).st_ino != os.stat(replacement).st_ino
        os.replace(replacement, registry)

        third = await startup_identity("third-fallback")
        assert third != first

    asyncio.run(scenario())
