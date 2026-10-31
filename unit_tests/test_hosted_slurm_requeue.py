"""Assembled HTTP + native-shaped proof + actual CPU scheduler-wrapper restart.

Only scheduler command responses are fake. The daemon, backend, rendered batch
script, training workers, journals, checkpoints, SQL activation and SSE are real.
This is not acceptance against an actual Slurm installation or shared NFS.
"""

import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import decode
from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl.core.submission_outcomes import submission_outcome_name
from wavedl_daemon.backends.contracts import ObservationStatus, OutcomeStatus
from wavedl_daemon.backends.slurm import SlurmBackend
from wavedl_daemon.backends.slurm_plan import SlurmSite
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessResult, ProcessSpec


class NativeScheduler:
    """Thread-safe shell-free responses with exact owner/request/native identity."""

    def __init__(self, owner):
        self.owner = owner
        self._lock = threading.Lock()
        self._state = ("RUNNING", 0, 0)
        self.request_id = None
        self.script = None
        self.calls = []

    def state(self, status, restart, exit_code=0):
        with self._lock:
            self._state = status, restart, exit_code

    def run(self, argv):
        assert isinstance(argv, tuple) and argv[0] in {
            "sbatch",
            "scontrol",
            "squeue",
            "sacct",
            "scancel",
        }
        with self._lock:
            self.calls.append(argv)
            status, restart, exit_code = self._state
            if argv[0] == "sbatch":
                assert self.request_id is None, "Requeue must not submit a new job"
                self.request_id = next(
                    value.split("=", 1)[1]
                    for value in argv
                    if value.startswith("--comment=")
                )
                self.script = Path(argv[-1])
                return ProcessResult(0, b"12345;test-cluster\n", b"")
            if argv[0] == "scontrol" and "config" in argv:
                return ProcessResult(
                    0,
                    b"MinJobAge = 300\nAccountingStorageType = accounting_storage/none\n",
                    b"",
                )
            if argv[0] == "sacct":
                return ProcessResult(1, b"", b"Accounting storage is disabled")
            if argv[0] == "scancel":
                raise AssertionError("Acceptance did not request cancellation")
            if self.request_id is None:
                return ProcessResult(0, b"", b"")
            name = f"wavedl-{self.request_id}"
            if argv[0] == "squeue":
                output = f"12345|{self.owner}|{name}|{status}|None|{self.request_id}\n"
            else:
                output = (
                    f"JobId=12345 UserId={self.owner}({os.getuid()}) "
                    f"JobName={name} Comment={self.request_id} Cluster=test-cluster "
                    f"JobState={status} Restarts={restart} ExitCode={exit_code}:0 Reason=None\n"
                )
            return ProcessResult(0, output.encode(), b"")


async def wait_view(client, run_id, *, attempt, state, timeout=30):
    deadline = time.monotonic() + timeout
    response = None
    while time.monotonic() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        value = response.json()
        if value["attempt"] == attempt and value["state"] == state:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"Hosted state never became {attempt}:{state}: {response.text}"
    )


def training_payload(environment):
    return {
        "backend": "slurm",
        "operation": "train",
        "topology": {"num_machines": 1, "processes_per_machine": 1, "machine_rank": 0},
        "config": {
            "model": "cnn",
            "epochs": 1,
            "fresh": True,
            "data_path": str(environment.data),
            "precision": "no",
            "workers": 0,
            "batch_size": 8,
        },
        "runtime": {
            "python": sys.executable,
            "mode": "selected",
            "device": "cpu",
            "environment": {
                "PYTHONPATH": str(environment.source),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MPLBACKEND": "Agg",
                "PATH": os.defpath,
            },
        },
    }


def speed_fleet(session):
    fleet = session.controller._fleets["slurm"]
    fleet.policy = replace(
        fleet.policy, interval=0.02, backoff_initial=0.02, backoff_max=0.02
    )


async def replay_sse(app, run_id):
    """Read the real public stream, then disconnect after both durable exits."""
    disconnected = asyncio.Event()
    seen, starts = [], []
    initial = True

    async def receive():
        nonlocal initial
        if initial:
            initial = False
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if disconnected.is_set():
            return
        if message["type"] == "http.response.start":
            starts.append(message)
        if message["type"] == "http.response.body":
            for line in message.get("body", b"").splitlines():
                if line.startswith(b"data: "):
                    seen.append(json.loads(line[6:]))
            if any("error" in item for item in seen) or {
                item["attempt"] for item in seen if item.get("type") == "exit"
            } == {
                1,
                2,
            }:
                disconnected.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/api/runs/{run_id}/events",
        "raw_path": f"/api/runs/{run_id}/events".encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"localhost:8420"),
            (b"authorization", b"Bearer hosted-requeue-token"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("localhost", 8420),
    }
    await asyncio.wait_for(app(scope, receive, send), 5)
    assert starts[0]["status"] == 200
    assert not any("error" in item for item in seen), seen
    assert (b"content-type", b"text/event-stream; charset=utf-8") in starts[0][
        "headers"
    ]
    return seen


@pytest.fixture
def hosted_execution(tmp_path, request):
    if os.name != "posix":
        pytest.skip("Scheduler binding requires POSIX")
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    import pwd

    owner = pwd.getpwuid(os.getuid()).pw_name
    scheduler = NativeScheduler(owner)
    source = Path(__file__).resolve().parents[1] / "src"
    data = tmp_path / "data.npz"
    generator = np.random.default_rng(19)
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    settings = Settings(tmp_path / "state", tmp_path / "runs")

    def backend_factory(lock):
        return {
            "slurm": SlurmBackend(
                SlurmSite(
                    partition="cpu",
                    cluster="test-cluster",
                    requeue=True,
                    stage_out_command=getattr(request, "param", None),
                ),
                scheduler,
                owner=owner,
                dispatch_lock=lock,
            )
        }

    services = HostedServices(settings, backend_factory=backend_factory)
    app = create_app(
        settings,
        token="hosted-requeue-token",
        daemon_id="daemon",
        build_id="build",
        services=services,
    )

    def execute(script, restart):
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(
                ("WAVEDL_", "SLURM_", "RANK", "LOCAL_RANK", "WORLD_SIZE")
            )
        }
        environment.update(
            SLURM_JOB_ID="12345",
            SLURM_RESTART_COUNT=str(restart),
            SLURM_CLUSTER_NAME="test-cluster",
            SLURM_JOB_USER=owner,
        )
        return run_command(
            ProcessSpec(("/bin/bash", str(script)), tmp_path, environment),
            WorkBudget(timeout_seconds=90, max_output_bytes=1024 * 1024),
        )

    return SimpleNamespace(
        scheduler=scheduler,
        source=source,
        data=data,
        settings=settings,
        services=services,
        app=app,
        execute=execute,
    )


def test_hosted_http_actual_cpu_requeue_preserves_both_execution_generations(
    hosted_execution,
):
    env = hosted_execution
    scheduler, settings, services, app, execute = (
        env.scheduler,
        env.settings,
        env.services,
        env.app,
        env.execute,
    )

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer hosted-requeue-token"},
            ) as client,
        ):
            await services.host.call(speed_fleet)
            response = await client.post(
                "/api/runs/plan",
                headers={"Idempotency-Key": "hosted-requeue"},
                json=training_payload(env),
            )
            assert response.status_code == 200, response.text
            preview = response.json()
            plan_id, run_id = preview["plan_id"], preview["run_id"]
            response = await client.post("/api/runs", json={"plan_id": plan_id})
            assert response.status_code == 202, response.text
            await wait_view(client, run_id, attempt=None, state="running")
            assert scheduler.script is not None
            layout = RunLayout(settings.run_root / run_id, run_id)
            manifest_before = layout.manifest.read_bytes()
            first = await asyncio.to_thread(execute, scheduler.script, 0)
            assert first.returncode == 0 and not first.timed_out, (
                first.stdout + first.stderr
            ).decode(errors="replace")
            await wait_view(client, run_id, attempt=1, state="running")
            scheduler.state("COMPLETED", 0)
            first_view = await wait_view(client, run_id, attempt=1, state="completed")
            assert first_view["evidence"]["batch"]["status"] == "success"
            assert first_view["evidence"]["stage_out"]["status"] == "success"
            original = await services.host.call(
                lambda session: session.runs.execution_store.get(plan_id)
            )
            hello1 = await services.host.call(
                lambda session: session.runs.provenance.get(plan_id, 1)
            )
            assert original.observation.status == ObservationStatus.TERMINAL
            assert original.observation.batch.status == OutcomeStatus.SUCCESS

            collection1 = await services.host.call(
                lambda session: session.runs.collections.get(plan_id, 1)
            )
            assert collection1.result.runtime.status == OutcomeStatus.SUCCESS
            assert collection1.result.batch.status == OutcomeStatus.SUCCESS
            assert collection1.result.stage_out.status == OutcomeStatus.SUCCESS
            initial_receipt_path = (
                layout.run_dir
                / "submissions"
                / preview["request_id"]
                / submission_outcome_name(
                    SchedulerIdentity(
                        run_id,
                        preview["request_id"],
                        "test-cluster",
                        "12345",
                        scheduler.owner,
                        0,
                    )
                )
            )
            initial_receipt = initial_receipt_path.read_bytes()
            assert hello1.event.attempt == 1
            scheduler.state("RUNNING", 1)
            await services.host.call(
                lambda session: session.controller._restart_due.__setitem__(plan_id, 0)
            )
            second_task = asyncio.create_task(
                asyncio.to_thread(execute, scheduler.script, 1)
            )
            await wait_view(client, run_id, attempt=2, state="running")
            second = await second_task
            assert second.returncode == 0 and not second.timed_out, (
                second.stdout + second.stderr
            ).decode(errors="replace")
            scheduler.state("COMPLETED", 1)
            final = await wait_view(client, run_id, attempt=2, state="completed")
            assert final["uncertainty"] is False
            assert final["evidence"]["batch"]["status"] == "success"
            assert final["evidence"]["stage_out"]["status"] == "success"
            history, hello2 = await services.host.call(
                lambda session: (
                    session.runs.execution_store.history(plan_id),
                    session.runs.provenance.get(plan_id, 2),
                )
            )
            assert len(history) == 1 and history[0].attempt == 1
            assert history[0].handle == original.handle
            assert history[0].observation == original.observation
            assert initial_receipt_path.read_bytes() == initial_receipt
            assert hello2.event.attempt == 2 and hello2.event.seq == 0
            public_history = await client.get(f"/api/runs/{run_id}/attempts")
            assert public_history.status_code == 200, public_history.text
            histories = public_history.json()
            assert histories["total"] == 2
            assert [item["attempt"] for item in histories["attempts"]] == [1, 2]
            assert [item["current"] for item in histories["attempts"]] == [False, True]
            for attempt, entry in enumerate(histories["attempts"], 1):
                assert entry["handle"]["attempt"] == attempt
                assert entry["observation"]["status"] == "terminal"
                assert entry["runtime"] == asdict(
                    (hello1 if attempt == 1 else hello2).hello
                )
                assert all(
                    entry["collection"][key]["status"] == "success"
                    for key in ("runtime", "batch", "stage_out")
                )
            assert (
                await services.host.call(
                    lambda session: session.runs.collections.get(plan_id, 1)
                )
            ) == collection1
            assert histories["attempts"][1]["scheduler_binding"]["restart_count"] == 1
            for offset, expected in ((0, [1]), (1, [2]), (2, [])):
                page = await client.get(
                    f"/api/runs/{run_id}/attempts?limit=1&offset={offset}"
                )
                assert page.status_code == 200, page.text
                assert page.json()["total"] == 2
                assert [item["attempt"] for item in page.json()["attempts"]] == expected
            assert (
                await services.host.call(
                    lambda session: session.runs.provenance.get(plan_id, 1)
                )
            ).event == hello1.event
            assert layout.manifest.read_bytes() == manifest_before
            assert sum(call[0] == "sbatch" for call in scheduler.calls) == 1
            durable = await services.event_log.read_after(run_id, limit=256)
            assert {event.attempt for event in durable if event.type == "hello"} == {
                1,
                2,
            }
            assert {event.attempt for event in durable if event.type == "exit"} == {
                1,
                2,
            }
            assert [(event.attempt, event.seq) for event in durable] == sorted(
                (event.attempt, event.seq) for event in durable
            )
            streamed = await replay_sse(app, run_id)
            assert {
                event["attempt"] for event in streamed if event["type"] == "hello"
            } == {1, 2}
            assert {
                event["attempt"] for event in streamed if event["type"] == "exit"
            } == {1, 2}
            for attempt in (1, 2):
                events = [
                    decode(line)
                    for line in layout.journal(attempt).read_bytes().splitlines()
                ]
                assert events[0].seq == 0 and events[0].type == "hello"
                assert events[0].attempt == attempt
                assert events[-1].payload["status"] == "completed"
                receipt_name = submission_outcome_name(
                    SchedulerIdentity(
                        run_id,
                        preview["request_id"],
                        "test-cluster",
                        "12345",
                        scheduler.owner,
                        attempt - 1,
                    )
                )
                receipt = json.loads(
                    (
                        layout.run_dir
                        / "submissions"
                        / preview["request_id"]
                        / receipt_name
                    ).read_bytes()
                )
                assert (
                    receipt["attempt"],
                    receipt["restart_count"],
                    receipt["request_id"],
                ) == (attempt, attempt - 1, preview["request_id"])
                assert all(
                    receipt[key]["status"] == "success"
                    for key in ("runtime", "batch", "stage_out")
                )
            generations = published_generations(layout)
            assert {generation.attempt for generation in generations} == {1, 2}
            assert all(
                (generation.path / "COMPLETE.json").is_file()
                for generation in generations
            )
            assert (
                await client.get(f"/api/runs/{run_id}/artifacts")
            ).status_code == 200

    asyncio.run(run())


@pytest.mark.parametrize("hosted_execution", ["exit 7"], indirect=True)
def test_hosted_stage_out_failure_preserves_successful_actual_runtime(hosted_execution):
    env = hosted_execution

    async def run():
        async with (
            env.app.router.lifespan_context(env.app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=env.app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer hosted-requeue-token"},
            ) as client,
        ):
            await env.services.host.call(speed_fleet)
            planned = await client.post(
                "/api/runs/plan",
                json=training_payload(env),
                headers={"Idempotency-Key": "stage-failure"},
            )
            assert planned.status_code == 200, planned.text
            preview = planned.json()
            response = await client.post(
                "/api/runs", json={"plan_id": preview["plan_id"]}
            )
            assert response.status_code == 202, response.text
            await wait_view(client, preview["run_id"], attempt=None, state="running")
            result = await asyncio.to_thread(env.execute, env.scheduler.script, 0)
            assert result.returncode == 7 and not result.timed_out, (
                result.stdout + result.stderr
            ).decode(errors="replace")
            await wait_view(client, preview["run_id"], attempt=1, state="running")
            env.scheduler.state("FAILED", 0, 7)
            await wait_view(client, preview["run_id"], attempt=1, state="failed")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                history = await client.get(f"/api/runs/{preview['run_id']}/attempts")
                assert history.status_code == 200, history.text
                collected = history.json()["attempts"][0]["collection"]
                if collected is not None:
                    break
                await asyncio.sleep(0.02)
            assert collected is not None, history.text
            assert collected["runtime"]["status"] == "success"
            assert collected["stage_out"]["status"] == "failed"
            assert collected["stage_out"]["exit_code"] == 7
            view = (await client.get(f"/api/runs/{preview['run_id']}")).json()
            assert view["state"] == "failed"
            assert view["evidence"]["runtime"]["outcome"]["status"] == "success"
            assert view["evidence"]["stage_out"]["status"] == "failed"
            rows = await env.services.event_log.read_after(preview["run_id"], limit=256)
            assert rows[-1].type == "exit" and rows[-1].payload["status"] == "completed"
            assert sum(call[0] == "sbatch" for call in env.scheduler.calls) == 1

    asyncio.run(run())
