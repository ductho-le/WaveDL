"""Opt-in Linux tmpfs exhaustion through the assembled hosted lifecycle.

Set WAVEDL_TEST_EMPTY_TMPFS to an empty, dedicated tmpfs mount of at most 8 MiB.
The test never fills a host filesystem and removes only its own temporary files.
"""

import asyncio
import errno
import os
import sqlite3
import sys
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import RunLayout
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings

from .test_hosted_failure_visibility import (
    StartupSupervisor,
    TerminalLocal,
    count_reconciliations,
    until,
)


@pytest.fixture
def isolated_tmpfs():
    """Refuse opt-in paths unless they are an empty bounded Linux tmpfs mount."""
    configured = os.environ.get("WAVEDL_TEST_EMPTY_TMPFS")
    if sys.platform != "linux" or not configured:
        pytest.skip("requires explicit isolated Linux tmpfs mount")
    root = Path(configured).resolve(strict=True)
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    assert any(
        line.split()[4] == str(root) and line.split(" - ")[1].startswith("tmpfs ")
        for line in mounts
    ), "acceptance path must itself be a tmpfs mount"
    capacity = os.statvfs(root)
    assert 0 < capacity.f_blocks * capacity.f_frsize <= 8 * 1024 * 1024
    assert not list(root.iterdir()), "acceptance tmpfs must be empty"
    with tempfile.TemporaryDirectory(prefix="wavedl-acceptance-", dir=root) as child:
        yield Path(child)


def test_real_tmpfs_enospc_preserves_authority_and_recovers(
    isolated_tmpfs, tmp_path, monkeypatch
):
    """Real ENOSPC cannot advance committed events or fabricate completion."""
    settings = Settings(isolated_tmpfs / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        backend_factory=lambda lock: {
            "local": TerminalLocal(
                dispatch_lock=lock,
                owner="tmpfs-acceptance",
                supervisor=StartupSupervisor(),
            )
        },
    )
    app = create_app(
        settings,
        token="tmpfs-token",
        daemon_id="tmpfs-daemon",
        build_id="acceptance",
        services=services,
    )
    filler = isolated_tmpfs / "owned-filler"
    sqlite_full = threading.Event()

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer tmpfs-token"},
            ) as client,
        ):
            await services.host.call(count_reconciliations)
            response = await client.post(
                "/api/runs/plan",
                headers={"Idempotency-Key": "tmpfs-acceptance"},
                json={
                    "backend": "local",
                    "operation": "train",
                    "config": {},
                    "runtime": {
                        "python": sys.executable,
                        "mode": "existing",
                        "device": "cpu",
                        "environment": {},
                    },
                    "topology": {
                        "num_machines": 1,
                        "processes_per_machine": 1,
                        "machine_rank": 0,
                    },
                },
            )
            assert response.status_code == 200, response.text
            preview = response.json()
            assert (
                await services.host.call(
                    lambda session: (
                        session.runs.plan_store.get(
                            preview["plan_id"]
                        ).plan.allocation_mode
                    )
                )
                == "writer"
            )
            layout = RunLayout(settings.run_root / preview["run_id"], preview["run_id"])
            response = await client.post(
                "/api/runs", json={"plan_id": preview["plan_id"]}
            )
            assert response.status_code == 202, response.text

            async def current():
                response = await client.get(f"/api/runs/{layout.run_id}")
                assert response.status_code == 200, response.text
                return response.json()

            async def bound():
                view = await current()
                return view["attempt"] == 1 and view["state"] == "terminal_unknown"

            await until(bound)
            assert not layout.journal(1).exists()
            assert (
                await services.host.call(
                    lambda session: session.registry._connection.execute(
                        "SELECT count(*) FROM initial_bindings WHERE plan_id=?",
                        (preview["plan_id"],),
                    ).fetchone()[0]
                )
                == 1
            )
            with JournalWriter(layout, attempt=1) as writer:
                hello = writer.append(
                    "hello",
                    asdict(
                        HelloPayload(
                            1,
                            1,
                            "acceptance",
                            "3.11",
                            "cpu",
                            ["train"],
                            sys.executable,
                            prefix=sys.prefix,
                            base_prefix=sys.base_prefix,
                        )
                    ),
                )

                async def hello_committed():
                    return await services.host.call(
                        lambda session: session.registry.cursor(layout.run_id, 1) == 0
                    )

                await until(hello_committed)

                def capture_sqlite_full(session):
                    original = session.registry.apply_events

                    def captured(*args):
                        try:
                            return original(*args)
                        except sqlite3.OperationalError as error:
                            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                                sqlite_full.set()
                            raise

                    monkeypatch.setattr(session.registry, "apply_events", captured)

                await services.host.call(capture_sqlite_full)
                bookmark = await services.host.call(
                    lambda session: session.ingestor._bookmarks[(layout.run_id, 1)]
                )
                await services.host.call(
                    lambda session: session.registry._connection.execute(
                        "PRAGMA wal_checkpoint(TRUNCATE)"
                    ).fetchall()
                )
                try:
                    with filler.open("xb", buffering=0) as stream:
                        while True:
                            stream.write(b"x" * 65536)
                except OSError as error:
                    assert error.errno == errno.ENOSPC
                assert os.statvfs(isolated_tmpfs).f_bavail == 0
                middle = writer.append(
                    "log",
                    {
                        "stream": "stdout",
                        "level": "info",
                        "message": "real-enospc:" + "x" * 60000,
                    },
                )
                completed = writer.append("exit", {"status": "completed", "code": 0})

                async def fault_visible():
                    response = await client.get("/api/health")
                    assert response.status_code == 200, response.text
                    return (
                        sqlite_full.is_set()
                        and {
                            "source": preview["plan_id"],
                            "error": "OperationalError",
                        }
                        in response.json()["monitoring_errors"]
                    )

                await until(fault_visible)
                before = await services.host.call(
                    lambda session: session.visibility_laps
                )

                async def three_more_laps():
                    return await services.host.call(
                        lambda session: session.visibility_laps >= before + 3
                    )

                await until(three_more_laps)
                assert (await current())["state"] == "terminal_unknown"
                assert (
                    await services.host.call(
                        lambda session: session.registry.cursor(layout.run_id, 1)
                    )
                    == 0
                )
                assert await services.event_log.read_after(layout.run_id) == [hello]
                assert (
                    await services.host.call(
                        lambda session: session.registry.get(layout.run_id).event_count
                    )
                    == 1
                )
                assert (
                    await services.host.call(
                        lambda session: session.ingestor._bookmarks[(layout.run_id, 1)]
                    )
                    == bookmark
                )
                filler.unlink()

                async def recovered():
                    return (await current())["state"] == "completed"

                await until(recovered)
                assert await services.event_log.read_after(layout.run_id) == [
                    hello,
                    middle,
                    completed,
                ]
                assert (
                    await services.host.call(
                        lambda session: session.registry.cursor(layout.run_id, 1)
                    )
                    == 2
                )
                before = await services.host.call(
                    lambda session: session.visibility_laps
                )
                await until(three_more_laps)
                assert len(await services.event_log.read_after(layout.run_id)) == 3
                response = await client.get("/api/health")
                assert not response.json()["monitoring_errors"]

    try:
        asyncio.run(run())
    finally:
        filler.unlink(missing_ok=True)
