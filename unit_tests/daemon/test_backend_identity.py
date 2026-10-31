"""Stable owned registry namespaces survive process restarts, never replacements."""

import asyncio
import os
import shutil

import httpx
import pytest

from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices, ServiceSession
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.settings import Settings


def settings_for(tmp_path, name="one"):
    settings = Settings(tmp_path / name / "state", tmp_path / name / "runs")
    settings.state_root.mkdir(mode=0o700, parents=True)
    return settings


def test_owned_identity_is_stable_across_sessions_but_not_other_roots(tmp_path):
    settings = settings_for(tmp_path)
    with ServiceSession(settings) as session:
        first = session.daemon_id
    with ServiceSession(settings) as session:
        assert session.daemon_id == first
    with ServiceSession(settings_for(tmp_path, "two")) as session:
        assert session.daemon_id != first
    assert len(first) == 64
    assert str(settings.state_root) not in first


def test_registry_replacement_changes_namespace_and_missing_database_is_refused(
    tmp_path,
):
    settings = settings_for(tmp_path)
    with ServiceSession(settings) as session:
        first = session.daemon_id
        path = session.registry.path
    replacement = path.with_name("replacement.db")
    shutil.copyfile(path, replacement)
    assert replacement.stat().st_ino != path.stat().st_ino
    os.replace(replacement, path)
    with ServiceSession(settings) as session:
        assert session.daemon_id != first
    path.rename(path.with_name("removed-registry.db"))
    with pytest.raises(PlanRecoveryRequired), ServiceSession(settings):
        pass


def test_hosted_health_adopts_owner_identity_and_never_exposes_auth_or_paths(tmp_path):
    settings = settings_for(tmp_path)

    async def read(fallback, token):
        services = HostedServices(settings)
        app = create_app(
            settings,
            token=token,
            daemon_id=fallback,
            build_id="test",
            services=services,
        )
        assert app.state.daemon_id == fallback
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://localhost:8420"
            ) as client,
        ):
            assert app.state.daemon_id == services.daemon_id
            unauthorized = await client.get("/api/health")
            assert unauthorized.status_code == 401
            response = await client.get(
                "/api/health", headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 200
            assert token not in response.text
            assert str(settings.state_root) not in response.text
            assert "principal" not in response.text
            return response.json()["daemon_id"]

    async def scenario():
        first = await read("process-one", "secret-auth-one")
        second = await read("process-two", "rotated-auth-two")
        assert first == second

    asyncio.run(scenario())
