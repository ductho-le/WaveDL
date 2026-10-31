"""Assembled HTTP/pump regressions for delayed journals and failed commits.

The native backend is controlled, while journaling, owner-thread reconciliation,
SQLite ingestion and authenticated HTTP views are real. This is local visibility
fault injection, not evidence about a particular shared filesystem.
"""

import asyncio
import sqlite3
import sys
import threading
import time
from dataclasses import asdict

import httpx
import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.local_startup import prepare_local_startup
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessResult


class CompletedProcess:
    pid = 12345

    def wait(self):
        return ProcessResult(0, b"", b"")

    def cancel(self, grace_seconds=5):
        pass


class StartupSupervisor:
    """Invoke genuine runtime startup, but do not launch numerical training."""

    def start(self, spec, *, output_limit):
        prepare_local_startup(spec.cwd, spec.environment)
        return CompletedProcess()


class TerminalLocal(LocalBackend):
    def _probe(self, plan):
        # This test isolates storage visibility, not torch capability selection.
        return {
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        }


def count_reconciliations(session):
    original = session.controller.reconcile_once
    session.visibility_laps = 0

    def counted():
        result = original()
        session.visibility_laps += 1
        return result

    session.controller.reconcile_once = counted


async def until(predicate, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("Hosted visibility scenario made no bounded progress")


@pytest.mark.parametrize(
    "failure_point", ["_write_record", "_write_cursor", "query_only", "page_quota"]
)
def test_delayed_completion_and_failed_commit_replay_over_hosted_http(
    tmp_path, monkeypatch, failure_point
):
    """No missing, empty or rolled-back completion may become HTTP success."""
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        backend_factory=lambda lock: {
            "local": TerminalLocal(
                dispatch_lock=lock,
                owner="visibility-test",
                supervisor=StartupSupervisor(),
            )
        },
    )
    app = create_app(
        settings,
        token="visibility-token",
        daemon_id="visibility-daemon",
        build_id="test",
        services=services,
    )
    failed = threading.Event()
    refusing = threading.Event()
    quota = {}

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer visibility-token"},
            ) as client,
        ):
            await services.host.call(count_reconciliations)
            response = await client.post(
                "/api/runs/plan",
                headers={"Idempotency-Key": "visibility"},
                json={
                    "backend": "local",
                    "operation": OperationKind.TRAIN.value,
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
            plan_id = preview["plan_id"]
            layout = RunLayout(settings.run_root / preview["run_id"], preview["run_id"])
            assert (
                await services.host.call(
                    lambda session: (
                        session.runs.plan_store.get(plan_id).plan.allocation_mode
                    )
                )
                == "writer"
            )
            response = await client.post("/api/runs", json={"plan_id": plan_id})
            assert response.status_code == 202, response.text

            async def current():
                response = await client.get(f"/api/runs/{layout.run_id}")
                assert response.status_code == 200, response.text
                return response.json()

            async def terminal_native():
                value = await current()
                return value["attempt"] == 1 and value["state"] == "terminal_unknown"

            await until(terminal_native)
            assert (await current())["state"] == "terminal_unknown"
            assert not layout.journal(1).exists()
            assert await services.host.call(
                lambda session: (
                    session.runs.execution_store.get(plan_id).handle
                    == session.runs.execution_store.get(
                        plan_id
                    ).submission_handle.bind_attempt(1)
                )
            )
            assert (
                await services.host.call(
                    lambda session: session.registry._connection.execute(
                        "SELECT count(*) FROM initial_bindings WHERE plan_id=?",
                        (plan_id,),
                    ).fetchone()[0]
                )
                == 1
            )
            missing_start = await services.host.call(
                lambda session: session.visibility_laps
            )

            async def missing_for_three_laps():
                assert (await current())["state"] == "terminal_unknown"
                return await services.host.call(
                    lambda session: session.visibility_laps >= missing_start + 3
                )

            await until(missing_for_three_laps)
            assert not layout.journal(1).exists()

            with JournalWriter(layout, attempt=1) as writer:
                # Hold a visible but empty ordinary journal across several pump
                # laps, rather than only evaluating the reducer once.
                before = await services.host.call(
                    lambda session: session.visibility_laps
                )

                async def three_more_laps():
                    return await services.host.call(
                        lambda session: session.visibility_laps >= before + 3
                    )

                await until(three_more_laps)
                assert (await current())["state"] == "terminal_unknown"
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
                assert (await current())["state"] == "terminal_unknown"
                assert await services.event_log.read_after(layout.run_id) == [hello]
                committed_bookmark = await services.host.call(
                    lambda session: session.ingestor._bookmarks[(layout.run_id, 1)]
                )

                refusing.set()

                def inject_failure(session):
                    if failure_point == "query_only":
                        # Actual SQLite write refusal, not a mocked exception or
                        # a chmod test that can behave differently as root.
                        session.registry._connection.execute("PRAGMA query_only=ON")
                        return
                    if failure_point == "page_quota":
                        connection = session.registry._connection
                        quota["original"] = connection.execute(
                            "PRAGMA max_page_count"
                        ).fetchone()[0]
                        pages = connection.execute("PRAGMA page_count").fetchone()[0]
                        quota["pages"] = pages
                        assert (
                            connection.execute(
                                f"PRAGMA max_page_count={pages}"
                            ).fetchone()[0]
                            == pages
                        )
                        original = session.registry.apply_events

                        def capture_real_full(*args):
                            try:
                                return original(*args)
                            except sqlite3.OperationalError as error:
                                # Capture SQLite's own error code, not only a
                                # matching string from an injected exception.
                                if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                                    failed.set()
                                raise

                        monkeypatch.setattr(
                            session.registry, "apply_events", capture_real_full
                        )
                        return
                    original = getattr(session.registry, failure_point)

                    def fail_after_write(*args):
                        original(*args)
                        if refusing.is_set():
                            failed.set()
                            raise sqlite3.OperationalError("database or disk is full")

                    monkeypatch.setattr(
                        session.registry, failure_point, fail_after_write
                    )

                await services.host.call(inject_failure)
                middle = (
                    writer.append(
                        "log",
                        {
                            "stream": "stdout",
                            "level": "info",
                            "message": "quota-growth:" + "x" * 60000,
                        },
                    )
                    if failure_point == "page_quota"
                    else writer.append(
                        "metric", {"name": "loss", "value": 0.25, "step": 1}
                    )
                )
                completed = writer.append("exit", {"status": "completed", "code": 0})

                async def failure_seen():
                    health = await client.get("/api/health")
                    assert health.status_code == 200, health.text
                    assert health.headers["Cache-Control"] == "no-store"
                    expected = {"source": plan_id, "error": "OperationalError"}
                    return expected in health.json()["monitoring_errors"] and (
                        failure_point == "query_only" or failed.is_set()
                    )

                await until(failure_seen)
                before = await services.host.call(
                    lambda session: session.visibility_laps
                )
                await until(three_more_laps)
                assert (await current())["state"] == "terminal_unknown"
                assert (
                    await services.host.call(
                        lambda session: session.registry.cursor(layout.run_id, 1)
                    )
                    == 0
                )
                assert (
                    await services.host.call(
                        lambda session: session.registry.get(layout.run_id).event_count
                    )
                    == 1
                )
                assert await services.event_log.read_after(layout.run_id) == [hello]
                if failure_point == "page_quota":
                    assert (
                        await services.host.call(
                            lambda session: session.registry._connection.execute(
                                "PRAGMA page_count"
                            ).fetchone()[0]
                        )
                        == quota["pages"]
                    )
                # The compact reader bookmark must not advance over rolled-back
                # records; replay resumes from the last committed hello.
                assert (
                    await services.host.call(
                        lambda session: session.ingestor._bookmarks[(layout.run_id, 1)]
                    )
                    == committed_bookmark
                )
                refusing.clear()
                if failure_point == "query_only":
                    await services.host.call(
                        lambda session: session.registry._connection.execute(
                            "PRAGMA query_only=OFF"
                        ).close()
                    )
                elif failure_point == "page_quota":
                    await services.host.call(
                        lambda session: session.registry._connection.execute(
                            f"PRAGMA max_page_count={quota['original']}"
                        ).close()
                    )

                async def completed_after_replay():
                    return (await current())["state"] == "completed"

                await until(completed_after_replay)
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
                assert (
                    await services.host.call(
                        lambda session: session.registry.get(layout.run_id).event_count
                    )
                    == 3
                )
                # Further reconciliations do not duplicate a recovered page.
                before = await services.host.call(
                    lambda session: session.visibility_laps
                )
                await until(three_more_laps)
                assert len(await services.event_log.read_after(layout.run_id)) == 3
                assert (await current())["state"] == "completed"
                if failure_point == "page_quota":
                    assert (
                        await services.host.call(
                            lambda session: session.registry._connection.execute(
                                "PRAGMA page_count"
                            ).fetchone()[0]
                        )
                        > quota["pages"]
                    )
                health = await client.get("/api/health")
                assert health.status_code == 200, health.text
                assert not health.json()["monitoring_errors"]

    asyncio.run(run())
