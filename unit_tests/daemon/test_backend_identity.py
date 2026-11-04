"""Stable owned database namespaces survive process restarts, never replacements."""

import asyncio
import os
import shutil
from pathlib import Path

import httpx

from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import (
    DATABASE_FILE,
    HostedServices,
    ServiceSession,
)
from wavedl_daemon.storage.database import database_generation, open_database
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
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


def test_database_replacement_or_loss_starts_a_new_namespace(tmp_path):
    settings = settings_for(tmp_path)
    path = settings.state_root / DATABASE_FILE
    with ServiceSession(settings) as session:
        first = session.daemon_id
    replacement = path.with_name("replacement.db")
    shutil.copyfile(path, replacement)
    assert replacement.stat().st_ino != path.stat().st_ino
    os.replace(replacement, path)
    with ServiceSession(settings) as session:
        second = session.daemon_id
    assert second != first
    # A lost database is lost history, never recovered from run directories.
    path.rename(path.with_name("removed.db"))
    with ServiceSession(settings) as session:
        assert session.daemon_id not in {first, second}
        assert session.runs.list() == []


def test_database_loss_with_reused_inode_starts_a_new_namespace(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    path = settings.state_root / DATABASE_FILE
    with ServiceSession(settings) as session:
        first = session.daemon_id
    info = path.stat()
    path.rename(path.with_name("removed.db"))
    original_stat = Path.stat

    def reused_inode(self, *, follow_symlinks=True):
        if self == path:
            return info
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", reused_inode)
    with ServiceSession(settings) as session:
        assert session.daemon_id != first
        assert session.runs.list() == []


def test_existing_schema_one_database_acquires_a_stable_generation(tmp_path):
    settings = settings_for(tmp_path)
    path = settings.state_root / DATABASE_FILE
    with acquire_dispatch_lock(settings.state_root / "dispatch.lock") as lock:
        connection = open_database(path, dispatch_lock=lock)
        try:
            original = database_generation(connection)
            connection.execute("DROP TABLE database_identity")
        finally:
            connection.close()
        connection = open_database(path, dispatch_lock=lock)
        try:
            generation = database_generation(connection)
            assert generation != original
            version = connection.execute("PRAGMA user_version").fetchone()
            assert version is not None
            assert version[0] == 1
        finally:
            connection.close()
        connection = open_database(path, dispatch_lock=lock)
        try:
            assert database_generation(connection) == generation
        finally:
            connection.close()


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
