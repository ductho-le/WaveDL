"""Local assembled journal framing failures, not site-filesystem certification.

HTTP, writer startup identity, journal reads, owner pumping and SQLite are real;
the numerical process is controlled and already exited successfully.
"""

import asyncio
import os
import sys
from dataclasses import asdict

import httpx
import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import Envelope, encode
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings

from .test_hosted_failure_visibility import (
    StartupSupervisor,
    TerminalLocal,
    count_reconciliations,
    until,
)


@pytest.mark.parametrize(
    "fault", ["torn-tail", "abandoned-tail", "oversized-line", "oversized-tail"]
)
def test_hosted_journal_fault_never_fabricates_or_skips_completion(tmp_path, fault):
    """A valid prefix survives faults; only a completed torn record can advance."""
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        backend_factory=lambda lock: {
            "local": TerminalLocal(
                dispatch_lock=lock,
                owner="journal-fault-test",
                supervisor=StartupSupervisor(),
            )
        },
    )
    app = create_app(
        settings,
        token="journal-fault-token",
        daemon_id="journal-fault-daemon",
        build_id="test",
        services=services,
    )

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer journal-fault-token"},
            ) as client,
        ):
            await services.host.call(count_reconciliations)
            response = await client.post(
                "/api/runs/plan",
                headers={"Idempotency-Key": "journal-fault"},
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
            planned = response.json()
            layout = RunLayout(settings.run_root / planned["run_id"], planned["run_id"])
            response = await client.post(
                "/api/runs", json={"plan_id": planned["plan_id"]}
            )
            assert response.status_code == 202, response.text

            async def current():
                response = await client.get(f"/api/runs/{layout.run_id}")
                assert response.status_code == 200, response.text
                return response.json()

            async def bound():
                value = await current()
                return value["attempt"] == 1 and value["state"] == "terminal_unknown"

            await until(bound)
            with JournalWriter(layout, attempt=1) as writer:
                hello = writer.append(
                    "hello",
                    asdict(
                        HelloPayload(
                            1,
                            1,
                            "test",
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
            prefix_size = layout.journal(1).stat().st_size
            completed = Envelope(
                layout.run_id,
                1,
                "exit",
                1,
                "2026-10-13T00:00:00Z",
                {"status": "completed", "code": 0},
            )
            encoded = encode(completed)
            split = len(encoded) // 2
            if fault.endswith("tail") and not fault.startswith("oversized"):
                damaged = encoded[:split]
            elif fault == "oversized-line":
                # Valid completion after a prohibited line must not be skipped to.
                damaged = b"x" * (64 * 1024) + b"\n" + encoded
            else:
                damaged = b"x" * (64 * 1024)
            with layout.journal(1).open("ab") as stream:
                stream.write(damaged)
                stream.flush()
                os.fsync(stream.fileno())

            async def laps_without_completion():
                before = await services.host.call(
                    lambda session: session.visibility_laps
                )

                async def three_more_laps():
                    assert (await current())["state"] == "terminal_unknown"
                    return await services.host.call(
                        lambda session: session.visibility_laps >= before + 3
                    )

                await until(three_more_laps)
                assert await services.event_log.read_after(layout.run_id) == [hello]
                assert await services.host.call(
                    lambda session: (
                        session.registry.cursor(layout.run_id, 1),
                        session.registry.get(layout.run_id).event_count,
                    )
                ) == (0, 1)

            await laps_without_completion()
            if fault.startswith("oversized"):
                health = await client.get("/api/health")
                assert health.status_code == 200, health.text
                assert {
                    "source": planned["plan_id"],
                    "error": "ProtocolError",
                } in health.json()["monitoring_errors"]
                return

            bookmark = await services.host.call(
                lambda session: session.ingestor._bookmarks[(layout.run_id, 1)]
            )
            assert bookmark.offset == prefix_size and bookmark.last_seq == 0
            # Exercise reconstruction from the committed byte boundary: no
            # live reader may be required to preserve a partial record.
            await services.host.call(lambda session: session.ingestor._readers.clear())
            await laps_without_completion()
            if fault == "abandoned-tail":
                return

            with layout.journal(1).open("ab") as stream:
                stream.write(encoded[split:])
                stream.flush()
                os.fsync(stream.fileno())

            async def recovered():
                return (await current())["state"] == "completed"

            await until(recovered)
            bookmark = await services.host.call(
                lambda session: session.ingestor._bookmarks[(layout.run_id, 1)]
            )
            assert bookmark.offset == layout.journal(1).stat().st_size
            assert bookmark.last_seq == 1
            assert await services.event_log.read_after(layout.run_id) == [
                hello,
                completed,
            ]
            assert await services.host.call(
                lambda session: (
                    session.registry.cursor(layout.run_id, 1),
                    session.registry.get(layout.run_id).event_count,
                )
            ) == (1, 2)
            response = await client.post(
                "/api/runs", json={"plan_id": planned["plan_id"]}
            )
            assert response.status_code == 202, response.text
            before = await services.host.call(lambda session: session.visibility_laps)

            async def completed_for_three_more_laps():
                assert (await current())["state"] == "completed"
                return await services.host.call(
                    lambda session: session.visibility_laps >= before + 3
                )

            await until(completed_for_three_more_laps)
            assert await services.event_log.read_after(layout.run_id) == [
                hello,
                completed,
            ]
            assert await services.host.call(
                lambda session: (
                    session.registry.cursor(layout.run_id, 1),
                    session.registry.get(layout.run_id).event_count,
                )
            ) == (1, 2)

    asyncio.run(run())
