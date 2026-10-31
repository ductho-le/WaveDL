"""Real CPU execution of the generated Slurm wrapper, without a real scheduler."""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from wavedl.core.generations import published_generations
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.protocol import decode
from wavedl.core.training_cli import training_defaults
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite
from wavedl_daemon.backends.slurm_submission import SlurmSubmission
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.storage.ownership import DispatchLock
from wavedl_daemon.supervision.base import ProcessResult, ProcessSpec


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.name != "posix", reason="Scheduler startup requires private POSIX storage"
    ),
]


class FakeSchedulerRunner:
    """Record one shell-free submission; never execute any scheduler program."""

    def __init__(self):
        self.calls = []

    def run(self, argv):
        assert isinstance(argv, tuple) and argv[0] == "sbatch"
        self.calls.append(argv)
        return ProcessResult(0, b"12345;test-cluster\n", b"")


@pytest.fixture
def submitted(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    import pwd

    owner = pwd.getpwuid(os.getuid()).pw_name
    source_root = Path(__file__).resolve().parents[1] / "src"
    rng = np.random.default_rng(17)
    data = tmp_path / "data.npz"
    np.savez(
        data,
        X=rng.normal(size=(20, 32)).astype("float32"),
        y=rng.normal(size=(20, 1)).astype("float32"),
    )
    layout = RunLayout(tmp_path / "run", str(uuid4()))
    config = training_defaults() | {
        "model": "cnn",
        "data_path": str(data),
        "epochs": 1,
        "batch_size": 8,
        "workers": 0,
        "precision": "no",
        "fresh": True,
    }
    # Utility-only list_models is not an execution-plan option; the worker
    # supplies its unchanged approved parser default itself.
    config.pop("list_models")
    runtime = RuntimeSelection(
        sys.executable,
        "selected",
        "cpu",
        {
            "PYTHONPATH": str(source_root),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "PATH": os.defpath,
        },
    )
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config=config,
        runtime=runtime,
        submission_cwd=tmp_path,
    )
    runner = FakeSchedulerRunner()
    request = str(uuid4())

    def submit(*, setup_commands=(), allocation_mode="reserved"):
        plan = SlurmPlanner(
            SlurmSite(
                partition="cpu",
                cluster="test-cluster",
                requeue=True,
                setup_commands=setup_commands,
            )
        ).plan(manifest, allocation_mode=allocation_mode)
        with DispatchLock(tmp_path / "dispatch.lock") as lock:
            handle = SlurmSubmission(runner, owner=owner, dispatch_lock=lock).submit(
                plan,
                request_id=request,
                attempt=None if allocation_mode == "writer" else 1,
            )
        script = Path(runner.calls[0][-1])
        return handle, script

    def execute(script, restart):
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(
                ("WAVEDL_", "SLURM_", "RANK", "LOCAL_RANK", "WORLD_SIZE")
            )
        }
        environment.update(
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_RESTART_COUNT": str(restart),
                "SLURM_CLUSTER_NAME": "test-cluster",
                "SLURM_JOB_USER": owner,
            }
        )
        return run_command(
            ProcessSpec(("/bin/bash", str(script)), tmp_path, environment),
            WorkBudget(timeout_seconds=90, max_output_bytes=1024 * 1024),
        )

    return layout, runner, request, owner, submit, execute


def journal(layout, attempt):
    return [decode(line) for line in layout.journal(attempt).read_bytes().splitlines()]


def test_real_cpu_writer_wrapper_allocates_initial_and_restart_only_at_runtime(
    submitted,
):
    layout, runner, request, _, submit, execute = submitted
    handle, script = submit(allocation_mode="writer")
    assert handle.job_id == "12345;test-cluster"
    assert not layout.attempts.exists() or list(layout.attempts.iterdir()) == []
    approved = layout.manifest.read_bytes()
    for restart, attempt in ((0, 1), (1, 2)):
        result = execute(script, restart)
        assert result.returncode == 0, (result.stdout + result.stderr).decode(
            errors="replace"
        )
        assert not result.timed_out
        events = journal(layout, attempt)
        assert events[0].type == "hello"
        assert events[-1].type == "exit"
        assert events[-1].payload["status"] == "completed"
        assert {event.attempt for event in events} == {attempt}
        identity = json.loads(
            (layout.attempt_dir(attempt) / "runtime_identity.json").read_bytes()
        )
        assert identity["scheduler_binding"]["restart_count"] == restart
        assert identity["scheduler_binding"]["request_id"] == request
        assert (
            layout.attempt_dir(attempt) / "runtime_request.json"
        ).read_bytes() == approved
    assert len(runner.calls) == 1
    assert layout.manifest.read_bytes() == approved
    assert not (layout.run_dir / "scheduler_attempts").exists()
    assert (layout.run_dir / "submissions" / request / "launchers").is_dir()


def test_writer_wrapper_pre_runtime_setup_failure_never_allocates(submitted):
    layout, runner, request, _, submit, execute = submitted
    _, script = submit(allocation_mode="writer", setup_commands=("false",))
    result = execute(script, 0)
    assert result.returncode != 0 and not result.timed_out
    assert not layout.attempts.exists() or list(layout.attempts.iterdir()) == []
    assert not list(layout.run_dir.rglob("events.jsonl"))
    assert not list(layout.run_dir.rglob("runtime_identity.json"))
    receipts = list(
        (layout.run_dir / "submissions" / request).glob("slurm-outcome-*.json")
    )
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_bytes())["attempt"] is None
    assert len(runner.calls) == 1


def test_real_cpu_generated_wrapper_restart_binds_new_attempt_without_resubmission(
    submitted,
):
    layout, runner, request, owner, submit, execute = submitted
    handle, script = submit()
    manifest_before = layout.manifest.read_bytes()
    initial = execute(script, 0)
    assert initial.returncode == 0, (initial.stdout + initial.stderr).decode(
        errors="replace"
    )
    assert not initial.timed_out
    restarted = execute(script, 1)
    assert restarted.returncode == 0, (restarted.stdout + restarted.stderr).decode(
        errors="replace"
    )
    assert not restarted.timed_out
    assert len(runner.calls) == 1
    assert handle.job_id == "12345;test-cluster"
    assert layout.manifest.read_bytes() == manifest_before
    assert sorted(path.name for path in layout.attempts.iterdir()) == ["1", "2"]
    for attempt, restart in ((1, 0), (2, 1)):
        events = journal(layout, attempt)
        assert events[0].seq == 0 and events[0].type == "hello"
        assert events[0].attempt == attempt and events[0].payload["device"] == "cpu"
        assert events[-1].type == "exit" and events[-1].payload["status"] == "completed"
        identity = json.loads(
            (layout.attempt_dir(attempt) / "runtime_identity.json").read_bytes()
        )
        assert identity["scheduler_binding"] == {
            "run_id": layout.run_id,
            "request_id": request,
            "cluster": "test-cluster",
            "job_id": "12345",
            "owner": owner,
            "restart_count": restart,
            "attempt": attempt,
        }
        filename = (
            "slurm-outcome.json" if restart == 0 else "slurm-outcome-restart-1.json"
        )
        outcome = json.loads((layout.attempt_dir(attempt) / filename).read_bytes())
        assert (
            outcome["attempt"],
            outcome["request_id"],
            outcome["restart_count"],
        ) == (attempt, request, restart)
        assert (
            outcome["runtime"]["exit_code"]
            == outcome["batch"]["exit_code"]
            == outcome["stage_out"]["exit_code"]
            == 0
        )
    assert not (layout.attempt_dir(1) / "slurm-outcome-restart-1.json").exists()
    generations = published_generations(layout)
    assert {item.attempt for item in generations} == {1, 2}
    assert all(
        (item.path / "COMPLETE.json").is_file() and "training_history.csv" in item.files
        for item in generations
    )


@pytest.mark.parametrize("failure", ["setup", "binding"])
def test_restart_setup_or_binding_failure_never_publishes_outcome_in_old_attempt(
    submitted, failure
):
    layout, runner, _request, owner, submit, execute = submitted
    setup = ('test "${SLURM_RESTART_COUNT:-0}" -eq 0',) if failure == "setup" else ()
    _handle, script = submit(setup_commands=setup)
    initial = execute(script, 0)
    assert initial.returncode == 0, (initial.stdout + initial.stderr).decode(
        errors="replace"
    )
    old_outcome = (layout.attempt_dir(1) / "slurm-outcome.json").read_bytes()
    old_journal = layout.journal(1).read_bytes()
    if failure == "binding":
        submission = layout.attempt_dir(1) / "submission.json"
        data = json.loads(submission.read_text())
        data["owner"] = owner + "-wrong"
        submission.write_text(json.dumps(data))
    restarted = execute(script, 1)
    assert restarted.returncode != 0 and not restarted.timed_out
    assert len(runner.calls) == 1
    assert (layout.attempt_dir(1) / "slurm-outcome.json").read_bytes() == old_outcome
    assert layout.journal(1).read_bytes() == old_journal
    assert not (layout.attempt_dir(1) / "slurm-outcome-restart-1.json").exists()
    assert not layout.attempt_dir(2).exists()
