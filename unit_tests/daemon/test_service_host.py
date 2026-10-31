"""Owner affinity and honest bounded admission under cancelled HTTP waits."""

import asyncio
import sqlite3
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError, WorkBudget
from wavedl_daemon.execution.pool import WorkCancelled, WorkDeadlineExceeded
from wavedl_daemon.execution.service_host import AsyncRunService, ServiceHost
from wavedl_daemon.http.app import create_app
from wavedl_daemon.http.runs import lifecycle_routes
from wavedl_daemon.storage.settings import Settings


def test_constructor_inert_sql_owner_and_context_exit(tmp_path):
    calls = []
    main_thread = threading.get_ident()

    @contextmanager
    def factory():
        calls.append(("enter", threading.get_ident()))
        connection = sqlite3.connect(tmp_path / "owned.sqlite")
        connection.execute("CREATE TABLE jobs (key TEXT PRIMARY KEY)")

        class Service:
            def submit(self, key):
                calls.append(("submit", threading.get_ident()))
                connection.execute("INSERT INTO jobs VALUES (?)", (key,))
                connection.commit()
                return key

        try:
            yield SimpleNamespace(runs=Service())
        finally:
            calls.append(("exit", threading.get_ident()))
            connection.close()

    host = ServiceHost(factory)
    assert calls == []
    assert not (tmp_path / "owned.sqlite").exists()

    async def run():
        await host.start()
        assert await AsyncRunService(host).submit("one") == "one"
        await host.close()

    asyncio.run(run())
    assert [name for name, _ in calls] == ["enter", "submit", "exit"]
    assert len({thread for _, thread in calls}) == 1
    assert calls[0][1] != main_thread
    assert not host._thread.is_alive()


def test_sync_slow_work_keeps_main_loop_responsive_and_deadline_retains_capacity():
    gate, entered = threading.Event(), threading.Event()
    jobs = []

    class Service:
        def submit(self, key):
            entered.set()
            gate.wait(3)
            if key not in jobs:
                jobs.append(key)
            return key

    host = ServiceHost(
        Service, budget=WorkBudget(max_workers=1, max_pending=0, timeout_seconds=0.05)
    )

    async def run():
        await host.start()
        beats = []

        async def heartbeat():
            for _ in range(10):
                await asyncio.sleep(0.01)
                beats.append(time.monotonic())

        clock = asyncio.create_task(heartbeat())
        with pytest.raises(WorkDeadlineExceeded):
            await host.call("submit", "one")
        assert entered.is_set()
        assert host.outstanding == 1
        with pytest.raises(BudgetExhausted):
            await host.call("submit", "one")
        await clock
        assert len(beats) == 10
        gate.set()
        while host.outstanding:
            await asyncio.sleep(0.005)
        assert await host.call("submit", "one") == "one"
        assert jobs == ["one"]
        await host.close()

    asyncio.run(run())


def test_client_cancel_capacity_retained_and_queued_cancel_never_executes():
    gate, entered = threading.Event(), threading.Event()
    executed = []

    class Service:
        def mutate(self, key):
            entered.set()
            if key == "active":
                gate.wait(3)
            executed.append(key)
            return key

    host = ServiceHost(
        Service, budget=WorkBudget(max_workers=1, max_pending=1, timeout_seconds=2)
    )

    async def run():
        await host.start()
        active = asyncio.create_task(host.call("mutate", "active"))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        queued = asyncio.create_task(host.call("mutate", "queued"))
        await asyncio.sleep(0.01)
        active.cancel()
        queued.cancel()
        for task in (active, queued):
            with pytest.raises(WorkDeadlineExceeded):
                await task
        assert host.outstanding == 2
        with pytest.raises(BudgetExhausted):
            await host.call("mutate", "third")
        closing = asyncio.create_task(host.close())
        await asyncio.sleep(0.02)
        assert not closing.done()
        gate.set()
        await closing
        assert executed == ["active"]
        assert host.outstanding == 0

    asyncio.run(run())


def test_async_callback_serial_owner_loop_and_callable_access():
    threads, order = [], []

    class Service:
        async def poll_once(self, key):
            threads.append(threading.get_ident())
            order.append((key, "start"))
            await asyncio.sleep(0.02)
            order.append((key, "end"))
            return key

    host = ServiceHost(lambda: SimpleNamespace(ingestor=Service()))

    async def run():
        await host.start()
        assert await asyncio.gather(
            host.call(lambda session: session.ingestor.poll_once("one")),
            host.call(lambda session: session.ingestor.poll_once("two")),
        ) == ["one", "two"]
        await host.close()

    asyncio.run(run())
    assert order == [("one", "start"), ("one", "end"), ("two", "start"), ("two", "end")]
    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()


def test_factory_failure_cleanly_stops_thread():
    def factory():
        raise OSError("private storage path")

    host = ServiceHost(factory)

    async def run():
        with pytest.raises(StorageWorkError, match="Service initialization failed"):
            await host.start()
        await host.close()

    asyncio.run(run())
    assert not host._thread.is_alive()


def test_late_factory_after_start_timeout_exits_context_on_owner():
    gate = threading.Event()
    calls = []

    @contextmanager
    def factory():
        gate.wait(3)
        calls.append(("enter", threading.get_ident()))
        try:
            yield object()
        finally:
            calls.append(("exit", threading.get_ident()))

    host = ServiceHost(factory, budget=WorkBudget(timeout_seconds=0.03))

    async def run():
        with pytest.raises(WorkDeadlineExceeded):
            await host.start()
        gate.set()
        await host.close()

    asyncio.run(run())
    assert calls[0][1] == calls[1][1] != threading.get_ident()
    assert not host._thread.is_alive()


def test_close_caller_cancellation_does_not_interrupt_resource_cleanup():
    gate, entered, exited = threading.Event(), threading.Event(), threading.Event()

    @contextmanager
    def factory():
        class Service:
            def block(self):
                entered.set()
                gate.wait(3)

        try:
            yield Service()
        finally:
            exited.set()

    host = ServiceHost(factory)

    async def run():
        await host.start()
        active = asyncio.create_task(host.call("block"))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        closing = asyncio.create_task(host.close())
        await asyncio.sleep(0.01)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not exited.is_set()
        gate.set()
        with pytest.raises(WorkCancelled):
            await active
        await host.close()
        assert exited.is_set()

    asyncio.run(run())


def test_http_async_adapter_awaits_typed_result_and_maps_deadline_503(tmp_path):
    gate = threading.Event()

    reply = {
        "plan_id": "plan-one",
        "run_id": "one",
        "request_id": "request-one",
        "backend": "local",
        "operation": "train",
        "attempt": None,
        "state": "planned",
        "uncertainty": True,
        "submission_requested": False,
        "cancellation_requested": False,
        "evidence": None,
        "created_at": None,
        "summary": None,
    }

    class Reply:
        def to_dict(self):
            return reply

    class Service:
        def get(self, run_id):
            return Reply()

        def submit(self, plan_id):
            gate.wait(3)
            return Reply()

    host = ServiceHost(
        Service, budget=WorkBudget(max_workers=1, max_pending=0, timeout_seconds=0.03)
    )
    app = create_app(
        Settings(tmp_path / "state", tmp_path / "runs"),
        token="host-test-token",
        daemon_id="test",
        build_id="test",
    )
    app.router.routes.extend(lifecycle_routes(AsyncRunService(host)))

    async def run():
        await host.start()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer host-test-token"},
            ) as client:
                response = await client.get("/api/runs/one")
                assert response.status_code == 200
                assert response.json() == reply
                response = await client.post("/api/runs", json={"plan_id": "one"})
                assert response.status_code == 503
                assert host.outstanding == 1
                assert (await client.get("/api/runs/one")).status_code == 503
        finally:
            gate.set()
            await host.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [True, False])
def test_async_inspection_caller_cancel_or_deadline_reaches_worker_cleanup(
    tmp_path, monkeypatch, cancel
):
    import sys
    from pathlib import Path

    from wavedl_daemon.execution.pool import BoundedThreadPool
    from wavedl_daemon.runs import inspection

    entered, cancellation, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    tokens = []

    def command(spec, budget, *, token):
        tokens.append(token)
        entered.set()
        while not token.cancellation_requested:
            if release.wait(0.005):
                break
        if token.cancellation_requested:
            cancellation.set()
        assert release.wait(2)
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            cancelled=False,
            stdout_truncated=False,
            stderr_truncated=False,
            stdout=b'{"schema_version":1,"format":"npy","variables":[]}',
        )

    monkeypatch.setattr(inspection, "run_command", command)
    pool = BoundedThreadPool(WorkBudget(timeout_seconds=2))

    class Service:
        def __init__(self):
            self.inspector = inspection.DatasetInspector(
                sys.executable, Path.cwd(), pool
            )

        async def inspect(self):
            return await self.inspector.inspect(str(tmp_path / "data.npy"))

        def status(self):
            return "ready"

    host = ServiceHost(
        Service, budget=WorkBudget(timeout_seconds=0.08 if not cancel else 2)
    )

    async def run():
        await host.start()
        task = asyncio.create_task(host.call("inspect"))
        try:
            async with asyncio.timeout(2):
                while not entered.is_set():
                    if task.done():
                        await task  # Preserve the real pre-readiness failure.
                        pytest.fail("Inspection completed without worker admission")
                    await asyncio.sleep(0.005)
            if cancel:
                task.cancel()
            with pytest.raises(WorkDeadlineExceeded):
                await task
            deadline = time.monotonic() + 1
            while not cancellation.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert cancellation.is_set()
            assert tokens[0].cancellation_requested
            assert host.outstanding == 1
            assert pool.outstanding == 1
            release.set()
            async with asyncio.timeout(2):
                while host.outstanding:
                    await asyncio.sleep(0.005)
            assert await host.call("status") == "ready"
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await host.close()

    try:
        asyncio.run(run())
    finally:
        release.set()
        pool.shutdown()


def test_hosted_start_and_close_are_idempotent_and_failure_preserved(tmp_path):
    from wavedl_daemon.runs.runtime_service import HostedServices

    builds = []

    def factory(lock):
        builds.append(lock)
        return {}

    services = HostedServices(
        Settings(tmp_path / "state", tmp_path / "runs"), backend_factory=factory
    )

    async def run():
        await asyncio.gather(services.start(), services.start())
        await services.start()
        assert len(builds) == 1
        await asyncio.gather(services.close(), services.close())
        await services.close()
        assert not services.host._thread.is_alive()
        assert services.pool._closed

    asyncio.run(run())

    def failed_factory(lock):
        raise OSError("original startup failure")

    failed = HostedServices(
        Settings(tmp_path / "bad-state", tmp_path / "bad-runs"),
        backend_factory=failed_factory,
    )

    async def failure():
        with pytest.raises(StorageWorkError) as error:
            await failed.start()
        assert str(error.value.__cause__) == "original startup failure"
        await failed.close()
        await failed.close()
        assert not failed.host._thread.is_alive()
        assert failed.pool._closed

    asyncio.run(failure())


def test_hosted_inspection_cancellation_terminates_real_supervised_child(
    tmp_path, monkeypatch
):
    import os
    import sys
    from pathlib import Path

    from wavedl_daemon.execution.commands import run_command
    from wavedl_daemon.execution.pool import BoundedThreadPool
    from wavedl_daemon.runs import inspection
    from wavedl_daemon.supervision.base import ProcessSpec

    marker = tmp_path / "child-started"
    completed = []
    pool = BoundedThreadPool(WorkBudget(timeout_seconds=5))
    script = (
        "import os,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )

    def command(spec, budget, *, token):
        result = run_command(
            ProcessSpec((sys.executable, "-c", script), Path.cwd()), budget, token=token
        )
        completed.append(result)
        return result

    monkeypatch.setattr(inspection, "run_command", command)

    class Service:
        def __init__(self):
            self.inspector = inspection.DatasetInspector(
                sys.executable, Path.cwd(), pool
            )

        async def inspect(self):
            return await self.inspector.inspect(str(tmp_path / "data.npy"))

    host = ServiceHost(Service, budget=WorkBudget(timeout_seconds=5))

    async def run():
        await host.start()
        task = asyncio.create_task(host.call("inspect"))
        try:
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                if task.done():
                    await task
                    pytest.fail("Inspection completed before real child startup")
                await asyncio.sleep(0.005)
            assert marker.exists()
            pid = int(marker.read_text())
            task.cancel()
            with pytest.raises(WorkDeadlineExceeded):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await host.close()
        assert completed and completed[0].cancelled
        if os.name != "nt":
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert pool.outstanding == 0

    try:
        asyncio.run(run())
    finally:
        pool.shutdown()


def test_cancelled_pool_wait_retrieves_late_failure_without_releasing_capacity():
    from wavedl_daemon.execution.pool import BoundedThreadPool

    entered, release = threading.Event(), threading.Event()
    pool = BoundedThreadPool(WorkBudget(max_workers=1, max_pending=0))

    def work(token):
        entered.set()
        assert release.wait(2)
        raise ValueError("late failure")

    async def run():
        errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        handle = pool.submit(work)
        task = asyncio.create_task(handle.wait())
        while not entered.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.outstanding == 1
        release.set()
        while pool.outstanding:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.01)
        assert errors == []
        with pytest.raises(ValueError, match="late failure"):
            await handle.wait()

    try:
        asyncio.run(run())
    finally:
        release.set()
        pool.shutdown()


def test_opt_in_filesystem_wait_and_cleanup_do_not_block_owner_get_or_cancel():
    from wavedl_daemon.execution.pool import BoundedThreadPool

    entered, release, cleanup = threading.Event(), threading.Event(), threading.Event()
    pool = BoundedThreadPool(WorkBudget(max_workers=1, max_pending=0))
    threads = []

    class Service:
        def __init__(self):
            self.owner = threading.get_ident()
            self.connection = sqlite3.connect(":memory:")
            self.connection.execute("CREATE TABLE requests (name TEXT)")

        async def browse(self):
            assert threading.get_ident() == self.owner

            def work(token):
                assert threading.get_ident() != self.owner
                entered.set()
                assert release.wait(2)
                token.check()

            handle = pool.submit(work)
            try:
                return await handle.wait()
            except asyncio.CancelledError:
                cleanup.set()
                handle.cancel()
                try:
                    await asyncio.shield(handle.wait())
                except Exception:
                    pass
                raise

        def get(self):
            threads.append(threading.get_ident())
            self.connection.execute("SELECT COUNT(*) FROM requests").fetchone()
            return "ready"

        def cancel(self):
            threads.append(threading.get_ident())
            self.connection.execute("INSERT INTO requests VALUES ('cancel')")
            self.connection.commit()
            return "cancelled"

        def close(self):
            self.connection.close()

    host = ServiceHost(Service, budget=WorkBudget(max_workers=1, max_pending=1))

    async def run():
        await host.start()
        browsing = asyncio.create_task(host.call("browse", concurrent_async=True))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        assert await asyncio.wait_for(host.call("get"), 0.5) == "ready"
        assert await asyncio.wait_for(host.call("cancel"), 0.5) == "cancelled"
        browsing.cancel()
        with pytest.raises(WorkDeadlineExceeded):
            await browsing
        while not cleanup.is_set():
            await asyncio.sleep(0.005)
        assert host.outstanding == pool.outstanding == 1
        assert await asyncio.wait_for(host.call("get"), 0.5) == "ready"
        assert await asyncio.wait_for(host.call("cancel"), 0.5) == "cancelled"
        release.set()
        await host.close()
        assert host.outstanding == pool.outstanding == 0
        assert len(set(threads)) == 1
        assert threads[0] != threading.get_ident()

    try:
        asyncio.run(run())
    finally:
        release.set()
        pool.shutdown()
