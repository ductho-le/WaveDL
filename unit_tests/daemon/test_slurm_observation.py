"""Scheduler-shaped fake responses; no real cluster commands."""

from uuid import uuid4

import pytest

from wavedl_daemon.backends.slurm import SlurmBackend
from wavedl_daemon.backends.slurm_plan import SlurmSite
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.model import (
    ObservationFailed,
    Phase,
    RunId,
    SlurmHandle,
    SlurmObservation,
)
from wavedl_daemon.supervision.base import ProcessResult


class Runner:
    """Answers each scheduler command with its configured output."""

    def __init__(self, **responses):
        self.responses = responses
        self.calls = []

    def run(self, argv):
        self.calls.append(argv)
        result = self.responses.get(argv[0], "")
        if isinstance(result, Exception):
            raise result
        if isinstance(result, ProcessResult):
            return result
        return ProcessResult(0, result.encode(), b"")


def run(job="42"):
    return RunId(str(uuid4())), SlurmHandle(job, "alpha", "alice")


def account(target, state="COMPLETED", code="0:0", restart=0, suffix="", **changes):
    run_id, handle = target
    fields = {
        "job": handle.job_id + suffix,
        "cluster": "alpha",
        "owner": "alice",
        "name": f"wavedl-{run_id}",
        "state": state,
        "code": code,
        "restart": str(restart),
        "submit": "2026-10-01T10:00:00",
        "comment": run_id,
    }
    fields.update(changes)
    return "|".join(fields.values()) + "\n"


def controller(target, state="RUNNING", restart=0, code="0:0", **changes):
    run_id, handle = target
    fields = {
        "JobId": handle.job_id,
        "UserId": "alice(1000)",
        "JobName": f"wavedl-{run_id}",
        "Comment": run_id,
        "JobState": state,
        "Restarts": str(restart),
        "ExitCode": code,
    }
    fields.update(changes)
    return " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def queue(target, state="PENDING", reason="None", comment=None):
    run_id, handle = target
    return f"{handle.job_id}|alice|wavedl-{run_id}|{state}|{reason}|{comment or run_id}\n"


def observe(runner, *targets):
    backend = SlurmBackend(SlurmSite("cpu", cluster="alpha"), runner, owner="alice")
    result = backend.observe_many(targets)
    return result[targets[0][0]] if len(targets) == 1 else result


@pytest.mark.parametrize(
    ("state", "phase"),
    [
        ("RUNNING", Phase.RUNNING),
        ("PENDING", Phase.QUEUED),
        ("REQUEUED", Phase.QUEUED),
        ("SPECIAL_EXIT", Phase.HELD),
        ("COMPLETED", None),
        ("FAILED", None),
    ],
)
def test_accounting_state_maps_to_a_phase_or_terminal(state, phase):
    target = run()
    observation = observe(Runner(sacct=account(target, state)), target)
    assert observation.phase is phase
    assert observation.terminal is (phase is None)
    assert observation.native_state == state


def test_root_row_not_batch_extern_or_step():
    target = run()
    rows = account(target, "RUNNING") + "".join(
        account(target, "FAILED", "9:0", suffix=suffix)
        for suffix in (".batch", ".extern", ".0")
    )
    observation = observe(Runner(sacct=rows), target)
    assert observation.phase is Phase.RUNNING and observation.exit_code is None


def test_exit_and_signal_are_preserved_separately():
    target = run()
    observation = observe(Runner(sacct=account(target, "FAILED", "0:15")), target)
    assert (observation.exit_code, observation.signal) == (0, 15)


def test_a_requeue_reports_the_restart_count_and_the_ended_execution():
    target = run()
    observation = observe(
        Runner(
            scontrol=controller(target, state="PENDING", restart=1),
            sacct=account(target, "NODE_FAIL", "0:0", restart=0),
        ),
        target,
    )
    assert observation == SlurmObservation(
        1, Phase.QUEUED, False, "PENDING", None, None, None, {0: "NODE_FAIL"}
    )


def test_current_accounting_execution_ignores_old_restart_rows():
    target = run()
    rows = account(target, "FAILED", "9:0", restart=0) + account(
        target, "COMPLETED", restart=1
    )
    observation = observe(Runner(sacct=rows), target)
    assert observation.terminal and observation.restart_count == 1
    assert observation.exit_code == 0 and observation.earlier == {0: "FAILED"}


def test_ambiguous_accounting_executions_rejected():
    target = run()
    rows = account(target) + account(target)
    assert isinstance(observe(Runner(sacct=rows), target), ObservationFailed)


def test_accounting_ahead_of_the_controller_is_rejected():
    target = run()
    runner = Runner(scontrol=controller(target), sacct=account(target, restart=1))
    assert isinstance(observe(runner, target), ObservationFailed)


def test_current_controller_running_overrides_accounting_terminal():
    target = run()
    runner = Runner(scontrol=controller(target), sacct=account(target))
    assert observe(runner, target).phase is Phase.RUNNING


@pytest.mark.parametrize(
    "change",
    [{"owner": "bob"}, {"cluster": "other"}, {"comment": "foreign"}, {"name": "foreign"}],
)
def test_native_identity_mismatch_rejected(change):
    target = run()
    observation = observe(Runner(sacct=account(target, **change)), target)
    assert isinstance(observation, ObservationFailed)


def test_another_owners_job_is_never_queried():
    run_id, handle = run()
    runner = Runner()
    observation = observe(runner, (run_id, SlurmHandle(handle.job_id, "alpha", "bob")))
    assert isinstance(observation, ObservationFailed)
    assert runner.calls == []
    with pytest.raises(ValueError):
        SlurmBackend(SlurmSite("cpu", cluster="alpha"), runner, owner="--allusers")


@pytest.mark.parametrize(
    "failure",
    [
        ProcessResult(1, b"", b"Access denied"),
        ProcessResult(0, b"", b"", timed_out=True),
        ProcessResult(0, b"", b"", stdout_truncated=True),
        BudgetExhausted("too many jobs"),
        RuntimeError("scheduler timeout"),
    ],
)
def test_query_failure_is_not_absence(failure):
    target = run()
    assert isinstance(observe(Runner(squeue=failure), target), ObservationFailed)


def test_accounting_outage_is_not_equivalent_to_disabled_accounting():
    target = run()
    outage = Runner(sacct=ProcessResult(1, b"", b"Connection refused"))
    assert isinstance(observe(outage, target), ObservationFailed)
    disabled = Runner(sacct=ProcessResult(1, b"", b"Accounting storage is disabled"))
    assert observe(disabled, target).terminal


def test_a_purged_job_is_terminal_with_an_unknown_state():
    target = run()
    for runner in (Runner(), Runner(scontrol="No jobs in the system\n")):
        observation = observe(runner, target)
        assert observation.terminal and observation.native_state is None


def test_a_queued_job_needs_the_controller_for_its_restart_count():
    target = run()
    failed = observe(
        Runner(
            squeue=queue(target),
            scontrol=ProcessResult(1, b"", b"controller down"),
            sacct=account(target, "PENDING"),
        ),
        target,
    )
    assert isinstance(failed, ObservationFailed) and "controller down" in failed.detail


@pytest.mark.parametrize(
    "accounting",
    ["", "42||||||||\n", "garbage", "42|alpha|alice|name| |0:0|0|time|comment\n"],
)
def test_malformed_and_missing_states_fail_closed(accounting):
    target = run()
    if not accounting:
        accounting = account(target, state=" ")
    assert isinstance(observe(Runner(sacct=accounting), target), ObservationFailed)


def test_queries_are_batched_per_cluster():
    first, second = run(), run("43")
    runner = Runner(sacct=account(first) + account(second))
    result = observe(runner, first, second)
    assert set(result) == {first[0], second[0]}
    assert all(observation.terminal for observation in result.values())
    assert [call[0] for call in runner.calls] == ["squeue", "scontrol", "sacct"]
    assert "42,43" in runner.calls[2]


def test_live_identity_can_verify_accounting_without_stored_comments():
    target = run()
    runner = Runner(
        scontrol=controller(target, state="COMPLETED"), sacct=account(target, comment="")
    )
    assert observe(runner, target).terminal
    alone = Runner(sacct=account(target, comment=""))
    assert isinstance(observe(alone, target), ObservationFailed)


def test_held_reason_and_queue_comment_identity():
    target = run()
    held = Runner(
        squeue=queue(target, reason="JobHeldUser"),
        scontrol=controller(target, state="PENDING", Reason="JobHeldUser"),
    )
    assert observe(held, target).phase is Phase.HELD
    foreign = Runner(
        squeue=queue(target, comment="foreign"),
        scontrol=controller(target, state="PENDING"),
    )
    assert isinstance(observe(foreign, target), ObservationFailed)


@pytest.mark.parametrize(
    "changes",
    [
        {"JobState": ""},
        {"Restarts": "invalid"},
        {"Comment": "foreign"},
        {"ExitCode": "invalid", "JobState": "COMPLETED"},
        {"RestartCnt": "2"},
    ],
)
def test_controller_malformed_identity_and_fields_fail_closed(changes):
    target = run()
    observation = observe(Runner(scontrol=controller(target, **changes)), target)
    assert isinstance(observation, ObservationFailed)


def test_a_bad_job_does_not_fail_its_peer():
    first, second = run(), run("43")
    runner = Runner(scontrol=controller(first, Comment="foreign") + controller(second))
    result = observe(runner, first, second)
    assert isinstance(result[first[0]], ObservationFailed)
    assert result[second[0]].phase is Phase.RUNNING


def test_controller_truncation_does_not_hide_valid_accounting():
    target = run()
    runner = Runner(
        scontrol=ProcessResult(0, b"partial", b"", stdout_truncated=True),
        sacct=account(target),
    )
    assert observe(runner, target).terminal
    runner.responses["sacct"] = ""
    assert isinstance(observe(runner, target), ObservationFailed)


@pytest.mark.parametrize("first_value", ["foreign", ""])
@pytest.mark.parametrize("key", ["JobId", "UserId", "Comment", "Restarts", "JobState"])
def test_duplicate_controller_fields_never_last_value_wins(key, first_value):
    target = run()
    output = f"{key}={first_value} " + controller(target, restart=1)
    observation = observe(Runner(scontrol=output), target)
    assert isinstance(observation, ObservationFailed)
    assert "Duplicate native" in observation.detail


@pytest.mark.parametrize("key", ["Comment", "JobName"])
def test_quoted_native_identity_does_not_gain_authority(key):
    target = run()
    correct = target[0] if key == "Comment" else f"wavedl-{target[0]}"
    output = controller(target, restart=1, **{key: f'"{correct}"'})
    assert isinstance(observe(Runner(scontrol=output), target), ObservationFailed)


def test_native_field_name_suffix_does_not_gain_identity_authority():
    target = run()
    output = controller(target, restart=1).replace("JobId=", "Other-JobId=")
    observation = observe(Runner(scontrol=output), target)
    assert isinstance(observation, ObservationFailed)
    assert "Malformed native" in observation.detail
