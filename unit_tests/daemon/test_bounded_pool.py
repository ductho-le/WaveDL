import asyncio
import errno
import threading

import pytest

from wavedl_daemon.execution import (
    BoundedThreadPool,
    BudgetExhausted,
    StorageWorkError,
    WorkBudget,
    WorkCancelled,
    WorkDeadlineExceeded,
)
from wavedl_daemon.execution.pool import CancellationToken


def test_drain_waits_for_control_lane_only(tmp_path):
    from wavedl_daemon.runs.runtime_service import ServiceSession
    from wavedl_daemon.storage.settings import Settings

    control_release = threading.Event()
    shared_release = threading.Event()
    admitted = threading.Event()
    drained = threading.Event()
    errors = []
    pool = BoundedThreadPool(WorkBudget(max_workers=2))
    shared_started = threading.Event()

    def shared(token):
        shared_started.set()
        shared_release.wait(5)

    pool.submit(shared)
    assert shared_started.wait(1)

    def owner():
        try:
            settings = Settings(tmp_path / "state", tmp_path / "runs")
            with ServiceSession(settings, worker_pool=pool) as session:
                started = threading.Event()

                def control(token):
                    started.set()
                    control_release.wait(5)

                session.controller.context.lane.submit(control)
                assert started.wait(1)
                admitted.set()
                session.controller.drain()
                drained.set()
        except BaseException as error:
            errors.append(error)
            admitted.set()

    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert admitted.wait(2)
        assert not errors
        assert not drained.wait(0.05)
        control_release.set()
        assert drained.wait(2)
        assert not shared_release.is_set()
    finally:
        control_release.set()
        shared_release.set()
        thread.join(5)
        pool.shutdown()
    assert not errors


def test_control_lane_admits_while_shared_saturated():
    release = threading.Event()
    with BoundedThreadPool(WorkBudget(max_workers=4, max_pending=0)) as pool:
        try:
            handles = [pool.submit(lambda token: release.wait(2)) for _ in range(3)]
            with pytest.raises(BudgetExhausted):
                pool.submit(lambda token: 0)
            assert pool.control_lane.submit(lambda token: 42).result(1) == 42
        finally:
            release.set()
        for handle in handles:
            handle.result(1)


def test_total_running_never_exceeds_max_workers():
    release = threading.Event()
    saturated = threading.Event()
    exceeded = threading.Event()
    lock = threading.Lock()
    running = [0]
    peak = [0]

    def blocked(token):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            if running[0] == 4:
                saturated.set()
            if running[0] > 4:
                exceeded.set()
        release.wait(2)
        with lock:
            running[0] -= 1

    with BoundedThreadPool(WorkBudget(max_workers=4, max_pending=2)) as pool:
        handles = []
        try:
            for lane in (pool, pool.control_lane):
                while True:
                    try:
                        handles.append(lane.submit(blocked))
                    except BudgetExhausted:
                        break
            assert saturated.wait(1)
            assert pool.outstanding == 11
            assert pool.control_lane.outstanding == 6
            assert not exceeded.wait(0.05)
        finally:
            release.set()
        for handle in handles:
            handle.result(2)
        assert peak[0] <= 4


def test_control_lane_borrows_idle_capacity():
    barrier = threading.Barrier(4, timeout=1)
    with BoundedThreadPool(WorkBudget(max_workers=4, max_pending=0)) as pool:
        handles = [
            pool.control_lane.submit(lambda token: barrier.wait()) for _ in range(4)
        ]
        assert sorted(handle.result(2) for handle in handles) == [0, 1, 2, 3]


@pytest.mark.parametrize("control_holders", [False, True])
def test_cancelled_token_does_not_wait_for_permit(control_holders):
    release = threading.Event()
    started = threading.Barrier(3, timeout=1)
    called = threading.Event()
    with BoundedThreadPool(
        WorkBudget(max_workers=2, max_pending=1, timeout_seconds=0.2)
    ) as pool:

        def blocked(token):
            started.wait()
            release.wait(2)

        # The waiting lane needs an executor thread free to reach the permit.
        holder_lane = pool.control_lane if control_holders else pool
        waiting_lane = pool if control_holders else pool.control_lane
        holders = [holder_lane.submit(blocked), pool.control_lane.submit(blocked)]
        try:
            started.wait()
            waiting = waiting_lane.submit(lambda token: called.set())
            with pytest.raises((WorkDeadlineExceeded, WorkCancelled)):
                waiting.result(1)
            assert not called.is_set()
            assert not any(handle.done() for handle in holders)
            assert pool.control_lane.outstanding == (2 if control_holders else 1)
            assert pool.outstanding == 2
        finally:
            release.set()
        for handle in holders:
            with pytest.raises((WorkDeadlineExceeded, WorkCancelled)):
                handle.result(1)


def test_single_worker_budget_shares_one_executor():
    release = threading.Event()
    started = threading.Event()
    order = []
    with BoundedThreadPool(WorkBudget(max_workers=1)) as pool:

        def first(token):
            started.set()
            release.wait(2)
            order.append("shared")

        shared = pool.submit(first)
        assert started.wait(1)
        control = pool.control_lane.submit(lambda token: order.append("control"))
        try:
            assert not control.done()
        finally:
            release.set()
        shared.result(1)
        control.result(1)
        assert order == ["shared", "control"]


def test_control_lane_shutdown_leaves_shared_lane_open():
    with BoundedThreadPool(WorkBudget()) as pool:
        pool.control_lane.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            pool.control_lane.submit(lambda token: 0)
        assert pool.submit(lambda token: 42).result(1) == 42


@pytest.mark.parametrize("timeout,expected", [(None, True), (0.25, False)])
def test_token_wait_rechecks_early_native_timeout(monkeypatch, timeout, expected):
    clock = [0.0]
    waits = []
    monkeypatch.setattr("wavedl_daemon.execution.pool.time.monotonic", lambda: clock[0])
    token = CancellationToken(1.0)

    class EarlyTimeout:
        def is_set(self):
            return False

        def wait(self, remaining):
            waits.append(remaining)
            clock[0] += remaining / 2 if len(waits) == 1 else remaining
            return False

    token._event = EarlyTimeout()
    assert token.wait(timeout) is expected
    assert len(waits) == 2
    assert clock[0] == (1.0 if timeout is None else timeout)


def test_exhaustion_and_cancel_does_not_release_running_slot():
    started = threading.Event()
    release = threading.Event()
    with BoundedThreadPool(WorkBudget(max_workers=1, max_pending=1)) as pool:

        def blocked(token):
            started.set()
            release.wait(2)

        first = pool.submit(blocked)
        assert started.wait(1)
        second = pool.submit(lambda token: 42)
        try:
            first.cancel()
            second.cancel()
            with pytest.raises(BudgetExhausted):
                pool.submit(lambda token: 0)
            assert pool.outstanding == 2
            assert not first.done()
        finally:
            release.set()
        with pytest.raises(WorkCancelled):
            first.result(2)
        with pytest.raises(WorkCancelled):
            second.result(2)
        assert pool.outstanding == 0
        assert pool.submit(lambda token: 7).result(1) == 7


def test_deadline_observed_by_callback():
    observed = threading.Event()
    with BoundedThreadPool(WorkBudget(timeout_seconds=0.04)) as pool:

        def cooperate(token):
            token.wait()
            observed.set()
            token.check()

        handle = pool.submit(cooperate)
        with pytest.raises(WorkDeadlineExceeded):
            handle.result(1)
        assert observed.is_set()
        assert pool.outstanding == 0


def test_wait_timeout_is_not_cancellation():
    release = threading.Event()
    with BoundedThreadPool(WorkBudget(max_workers=1, max_pending=0)) as pool:
        handle = pool.submit(lambda token: release.wait(2))
        try:
            with pytest.raises(TimeoutError):
                handle.result(0.01)
            assert not handle.token.cancelled
            assert pool.outstanding == 1
        finally:
            release.set()
        assert handle.result(1)


def test_task_exceptions_and_storage_failure_are_explicit():
    with BoundedThreadPool(WorkBudget(max_workers=1)) as pool:

        def fail(token):
            raise ValueError("invalid")

        with pytest.raises(ValueError, match="invalid"):
            pool.submit(fail).result(1)

        def disk_full(token):
            raise OSError(errno.ENOSPC, "disk full")

        with pytest.raises(StorageWorkError) as captured:
            pool.submit(disk_full).result(1)
        assert captured.value.__cause__.errno == errno.ENOSPC
        assert pool.outstanding == 0


def test_async_handle_does_not_block_event_loop():
    async def exercise():
        with BoundedThreadPool(WorkBudget()) as pool:
            handle = pool.submit(lambda token: 42)
            assert await handle.wait() == 42

    asyncio.run(exercise())


def test_shutdown_refuses_new_admission():
    pool = BoundedThreadPool(WorkBudget())
    pool.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit(lambda token: 0)


def test_noncooperative_deadline_keeps_slot_until_callback_stops():
    release = threading.Event()
    with BoundedThreadPool(
        WorkBudget(max_workers=1, max_pending=0, timeout_seconds=0.02)
    ) as pool:
        handle = pool.submit(lambda token: release.wait(2))
        try:
            assert handle.token.wait(1)
            assert not handle.done()
            with pytest.raises(BudgetExhausted):
                pool.submit(lambda token: 0)
        finally:
            release.set()
        with pytest.raises(WorkDeadlineExceeded):
            handle.result(1)


def test_async_cancel_requests_cooperative_stop():
    started = threading.Event()
    stopped = threading.Event()

    def cooperate(token):
        started.set()
        token.wait()
        stopped.set()
        token.check()

    async def exercise():
        with BoundedThreadPool(WorkBudget()) as pool:
            handle = pool.submit(cooperate)
            assert started.wait(1)
            waiter = asyncio.create_task(handle.wait())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert stopped.wait(1)
            with pytest.raises(WorkCancelled):
                handle.result(1)

    asyncio.run(exercise())
