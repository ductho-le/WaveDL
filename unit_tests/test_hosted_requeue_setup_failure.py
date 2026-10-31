"""Real HTTP/Bash/CPU proof for an unbound restart after a completed runtime.

Only native scheduler command responses are simulated. This requires a POSIX
shell and optional ML dependencies; it is not acceptance against real Slurm/NFS.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from unit_tests.test_hosted_slurm_requeue import (
    NativeScheduler,
    speed_fleet,
    training_payload,
    wait_view,
)
from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import decode
from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl.core.submission_outcomes import submission_outcome_name
from wavedl_daemon.backends.contracts import OutcomeStatus
from wavedl_daemon.backends.slurm import SlurmBackend
from wavedl_daemon.backends.slurm_plan import SlurmSite
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessSpec


@pytest.fixture
def setup_failure_execution(tmp_path, request):
    if os.name != "posix":
        pytest.skip("Rendered scheduler wrapper execution requires POSIX")
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    import pwd

    owner = pwd.getpwuid(os.getuid()).pw_name
    scheduler = NativeScheduler(owner)
    source = Path(__file__).resolve().parents[1] / "src"
    data = tmp_path / "data.npz"
    generator = np.random.default_rng(29)
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    stage_exit = request.param

    def backend_factory(lock):
        return {
            "slurm": SlurmBackend(
                SlurmSite(
                    partition="cpu",
                    cluster="test-cluster",
                    requeue=True,
                    setup_commands=(
                        'if [ "$SLURM_RESTART_COUNT" -eq 0 ]; then :; '
                        "else (exit 9); fi",
                    ),
                    stage_out_command=(
                        'if [ "$SLURM_RESTART_COUNT" -eq 0 ]; then :; '
                        f"else exit {stage_exit}; fi"
                    ),
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
        stage_exit=stage_exit,
    )


@pytest.mark.parametrize("setup_failure_execution", [0, 7], indirect=True)
def test_completed_runtime_then_unbound_requeue_setup_failure(setup_failure_execution):
    env = setup_failure_execution

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
                headers={"Idempotency-Key": "bound-requeue-setup-failure"},
                json=training_payload(env),
            )
            assert planned.status_code == 200, planned.text
            preview = planned.json()
            plan_id, run_id = preview["plan_id"], preview["run_id"]
            submitted = await client.post("/api/runs", json={"plan_id": plan_id})
            assert submitted.status_code == 202, submitted.text
            await wait_view(client, run_id, attempt=None, state="running")
            assert env.scheduler.script is not None
            layout = RunLayout(env.settings.run_root / run_id, run_id)
            manifest_before = layout.manifest.read_bytes()
            first = await asyncio.to_thread(env.execute, env.scheduler.script, 0)
            assert first.returncode == 0 and not first.timed_out, (
                first.stdout + first.stderr
            ).decode(errors="replace")
            await wait_view(client, run_id, attempt=1, state="running")
            env.scheduler.state("COMPLETED", 0)
            await wait_view(client, run_id, attempt=1, state="completed")

            async def first_collection():
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    value = await env.services.host.call(
                        lambda session: session.runs.collections.get(plan_id, 1)
                    )
                    if value is not None:
                        return value
                    await asyncio.sleep(0.02)
                raise AssertionError("Completed actual runtime was not collected")

            positive_collection = await first_collection()
            assert all(
                getattr(positive_collection.result, name).status
                == OutcomeStatus.SUCCESS
                for name in ("runtime", "batch", "stage_out")
            )
            original, hello1 = await env.services.host.call(
                lambda session: (
                    session.runs.execution_store.get(plan_id),
                    session.runs.provenance.get(plan_id, 1),
                )
            )
            assert json.loads(original.handle.raw_json)["restart_count"] == 0
            journal_before = layout.journal(1).read_bytes()
            events_before = await env.services.event_log.read_after(run_id, limit=256)
            assert any(event.type == "metric" for event in events_before)
            decoded = [decode(line) for line in journal_before.splitlines()]
            assert decoded[0].type == "hello"
            assert decoded[-1].type == "exit"
            assert decoded[-1].payload["status"] == "completed"
            assert any(event.type == "metric" for event in decoded)
            generations = published_generations(layout)
            assert {generation.attempt for generation in generations} == {1}
            generation_files = {
                path: path.read_bytes()
                for generation in generations
                for path in generation.path.rglob("*")
                if path.is_file()
            }
            assert any(path.name == "COMPLETE.json" for path in generation_files)
            native0 = SchedulerIdentity(
                run_id,
                preview["request_id"],
                "test-cluster",
                "12345",
                env.scheduler.owner,
                0,
            )
            receipt_dir = layout.run_dir / "submissions" / preview["request_id"]
            receipt0_path = receipt_dir / submission_outcome_name(native0)
            receipt0 = receipt0_path.read_bytes()

            # No old-runtime replay error or restart hint is introduced here.
            # A normal native Restarts=1 row must advance the independent head.
            env.scheduler.state("RUNNING", 1)
            await env.services.host.call(
                lambda session: session.controller._restart_due.__setitem__(plan_id, 0)
            )
            failed = await asyncio.to_thread(env.execute, env.scheduler.script, 1)
            assert failed.returncode == 9 and not failed.timed_out, (
                failed.stdout + failed.stderr
            ).decode(errors="replace")
            native1 = SchedulerIdentity(
                run_id,
                preview["request_id"],
                "test-cluster",
                "12345",
                env.scheduler.owner,
                1,
            )
            receipt1 = json.loads(
                (receipt_dir / submission_outcome_name(native1)).read_bytes()
            )
            assert receipt1["attempt"] is None
            assert receipt1["restart_count"] == 1
            assert receipt1["runtime"]["status"] == "unknown"
            assert receipt1["batch"]["exit_code"] == 9
            assert receipt1["stage_out"]["exit_code"] == env.stage_exit
            env.scheduler.state("FAILED", 1, 9)
            view = await wait_view(client, run_id, attempt=1, state="failed")
            assert view["uncertainty"] is False
            assert view["evidence"]["confidence"] == "confirmed"
            assert view["evidence"]["runtime"]["hello_seen"] is False
            assert view["evidence"]["runtime"]["outcome"] is None

            deadline = time.monotonic() + 10
            native_collection = None
            while time.monotonic() < deadline:
                native_collection = await env.services.host.call(
                    lambda session: session.runs.collections.get_submission_generation(
                        plan_id, native1
                    )
                )
                if native_collection is not None:
                    break
                await asyncio.sleep(0.02)
            assert native_collection is not None
            assert native_collection.identity == native1
            assert native_collection.result.runtime.status == OutcomeStatus.UNKNOWN
            assert native_collection.result.runtime.exit_code is None
            assert native_collection.result.batch.status == OutcomeStatus.FAILED
            assert native_collection.result.batch.exit_code == 9
            assert native_collection.result.stage_out.exit_code == env.stage_exit
            assert native_collection.result.stage_out.status == (
                OutcomeStatus.FAILED if env.stage_exit else OutcomeStatus.SUCCESS
            )
            final_view = (await client.get(f"/api/runs/{run_id}")).json()
            assert final_view["state"] == "failed"
            assert final_view["uncertainty"] is False
            assert final_view["evidence"]["runtime"]["hello_seen"] is False
            assert final_view["evidence"]["batch"]["exit_code"] == 9
            assert final_view["evidence"]["stage_out"]["exit_code"] == env.stage_exit
            (
                current,
                head,
                provenance,
                old_collection,
                history,
            ) = await env.services.host.call(
                lambda session: (
                    session.runs.execution_store.get(plan_id),
                    session.runs.collections.get_native_head(plan_id),
                    session.runs.provenance.get(plan_id, 1),
                    session.runs.collections.get(plan_id, 1),
                    session.runs.execution_store.history(plan_id),
                )
            )
            assert head == native1
            assert current.attempt == 1 and current.handle == original.handle
            assert current.observation == original.observation
            assert old_collection == positive_collection
            assert provenance == hello1
            assert history == []
            assert (
                await env.services.host.call(
                    lambda session: session.runs.provenance.get(plan_id, 2)
                )
            ) is None
            assert not layout.journal(2).exists()
            assert not layout.attempt_dir(2).exists()
            assert layout.manifest.read_bytes() == manifest_before
            assert layout.journal(1).read_bytes() == journal_before
            assert receipt0_path.read_bytes() == receipt0
            assert all(
                path.read_bytes() == data for path, data in generation_files.items()
            )
            assert published_generations(layout) == generations
            assert (
                await env.services.event_log.read_after(run_id, limit=256)
            ) == events_before
            assert sum(call[0] == "sbatch" for call in env.scheduler.calls) == 1

            page = await client.get(f"/api/runs/{run_id}/attempts")
            assert page.status_code == 200, page.text
            attempts = page.json()
            assert attempts["total"] == 1
            entry = attempts["attempts"][0]
            assert entry["attempt"] == 1 and entry["current"] is False
            assert entry["last_runtime"] is True and entry["native_pending"] is True
            assert entry["runtime"] == asdict(hello1.hello)
            assert entry["collection"] == asdict(positive_collection.result)
            assert (
                entry["submission_collection"] == native_collection.proposal.to_dict()
            )
            assert entry["submission_collection"]["identity"]["restart_count"] == 1
            assert entry["handle"]["attempt"] == 1
            assert (
                await client.get(f"/api/runs/{run_id}/artifacts")
            ).status_code == 200

    asyncio.run(run())
