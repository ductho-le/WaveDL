"""Scheduler adapters are shell-free, byte-bounded and deadline supervised."""

import json
import sys

import pytest

from wavedl_daemon.backends.scheduler import SchedulerCommands
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget


def test_scheduler_runs_selected_program_without_shell(tmp_path):
    runner = SchedulerCommands(cwd=tmp_path, executables={"squeue": sys.executable})
    result = runner.run(
        ("squeue", "-c", "import sys;print(sys.argv[1])", "$(touch unsafe)")
    )
    assert result.returncode == 0
    assert result.stdout.strip() == b"$(touch unsafe)"
    assert not (tmp_path / "unsafe").exists()


def test_scheduler_timeout_and_output_truncation_are_visible(tmp_path):
    runner = SchedulerCommands(
        cwd=tmp_path,
        executables={"squeue": sys.executable},
        budget=WorkBudget(max_output_bytes=64, timeout_seconds=0.1),
    )
    result = runner.run(
        ("squeue", "-u", "-c", "import time; print('x'*1000); time.sleep(5)")
    )
    assert result.timed_out
    assert result.stdout_truncated
    assert len(result.stdout) <= 32


def test_scheduler_rejects_unbounded_or_unknown_commands(tmp_path):
    runner = SchedulerCommands(cwd=tmp_path, budget=WorkBudget(max_command_bytes=8))
    with pytest.raises(BudgetExhausted):
        runner.run(("squeue", "too long"))
    with pytest.raises(ValueError):
        runner.run(("bash", "-c", "anything"))


def test_scheduler_options_cannot_override_approved_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("SBATCH_JOB_NAME", "ambient-override")
    monkeypatch.setenv("SLURM_EXPORT_ENV", "ambient-leak")
    runner = SchedulerCommands(
        cwd=tmp_path,
        executables={"sbatch": sys.executable},
        environment={
            "SBATCH_JOB_NAME": "foreign",
            "SBATCH_ARRAY_INX": "1-1000",
            "SCANCEL_BATCH": "1",
            "SQUEUE_FORMAT": "invalid",
            "SLURM_CLUSTERS": "foreign",
            "SLURM_EXPORT_ENV": "ALL",
            "SLURM_HINT": "multithread",
            "SLURM_CONF": "/etc/site/slurm.conf",
            "SLURM_JWT": "submission-auth-only",
            "TZ": "Asia/Ho_Chi_Minh",
        },
    )
    result = runner.run(
        ("sbatch", "-c", "import os,json; print(json.dumps(dict(os.environ)))")
    )
    environment = json.loads(result.stdout)
    assert not any(
        key.startswith(("SBATCH_", "SCANCEL_", "SQUEUE_")) for key in environment
    )
    assert "SLURM_CLUSTERS" not in environment
    assert "SLURM_EXPORT_ENV" not in environment
    assert "SLURM_HINT" not in environment
    assert environment["SLURM_CONF"] == "/etc/site/slurm.conf"
    assert environment["SLURM_JWT"] == "submission-auth-only"
    assert environment["TZ"] == "UTC"
