"""Full native writer heads survive unsupported lifecycle states, not bad proof."""

import json
from dataclasses import asdict
from uuid import uuid4

import pytest

from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl_daemon.backends.contracts import ObservationStatus, SubmissionHandle
from wavedl_daemon.backends.slurm_observation import (
    RestartBindingUnavailable,
    SlurmObservation,
)
from wavedl_daemon.supervision.base import ProcessResult


class Runner:
    def __init__(self, **responses):
        self.responses = responses
        self.calls = []

    def run(self, argv):
        self.calls.append(argv)
        result = self.responses.get(argv[0], "")
        if isinstance(result, ProcessResult):
            return result
        return ProcessResult(0, result.encode(), b"")


def submission(tmp_path, job="42"):
    request = str(uuid4())
    return SubmissionHandle(
        "slurm",
        str(uuid4()),
        request,
        "alice",
        str(tmp_path),
        job_id=f"{job};alpha",
        raw_json=json.dumps(
            {
                "cluster": "alpha",
                "job_name": f"wavedl-{request}",
                "comment": request,
                "restart_count": 0,
                "operation": "train",
                "allocation_mode": "writer",
            }
        ),
    )


def control(handle, **changes):
    fields = {
        "JobId": handle.job_id.split(";")[0],
        "Cluster": "alpha",
        "UserId": "alice(1000)",
        "JobName": f"wavedl-{handle.request_id}",
        "Comment": handle.request_id,
        "JobState": "FUTURE_NATIVE_STATE",
        "Restarts": "3",
        "ExitCode": "9:15",
    } | changes
    return " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def accounting(handle, **changes):
    fields = {
        "job": handle.job_id.split(";")[0],
        "cluster": "alpha",
        "owner": "alice",
        "name": f"wavedl-{handle.request_id}",
        "state": "FUTURE_NATIVE_STATE",
        "exit": "9:15",
        "restart": "3",
        "submit": "2026-10-05T10:00:00",
        "comment": handle.request_id,
    } | changes
    return "|".join(fields.values()) + "\n"


def queue(handle, state="RUNNING"):
    return (
        f"{handle.job_id.split(';')[0]}|alice|wavedl-{handle.request_id}"
        f"|{state}|None|{handle.request_id}\n"
    )


def identity(handle, restart=3):
    return SchedulerIdentity(
        handle.run_id,
        handle.request_id,
        "alpha",
        handle.job_id.split(";")[0],
        "alice",
        restart,
    )


@pytest.mark.parametrize("source", ["scontrol", "sacct", "both"])
def test_unknown_native_state_retains_full_verified_head(tmp_path, source):
    handle = submission(tmp_path)
    runner = Runner(
        scontrol=control(handle) if source in {"scontrol", "both"} else "",
        sacct=accounting(handle) if source in {"sacct", "both"} else "",
    )
    observer = SlurmObservation(runner, owner="alice")
    result = observer.observe_submission(handle)
    assert result.status == ObservationStatus.ERROR
    assert result.native_state == "FUTURE_NATIVE_STATE"
    assert result.native_identity == identity(handle)
    assert result.to_dict()["native_identity"] == asdict(identity(handle))
    assert result.exit_code is result.signal is result.batch is result.stage_out is None
    assert not hasattr(result, "attempt")
    assert observer.collect_submission(handle) is None


def test_error_after_identity_proof_retains_head_without_exit(tmp_path):
    handle = submission(tmp_path)
    observer = SlurmObservation(
        Runner(scontrol=control(handle, JobState="COMPLETED", ExitCode="broken")),
        owner="alice",
    )
    result = observer.observe_submission(handle)
    assert result.status == ObservationStatus.ERROR
    assert result.native_identity == identity(handle)
    assert result.exit_code is result.signal is result.batch is None
    assert "Malformed scheduler exit status" in result.detail


def test_newer_unknown_native_state_retains_verified_restart_discovery(tmp_path):
    receipt = submission(tmp_path)
    active = receipt.bind_attempt(1)
    observer = SlurmObservation(Runner(scontrol=control(receipt)), owner="alice")
    with pytest.raises(RestartBindingUnavailable) as caught:
        observer.discover_restart(active)
    assert caught.value.native_restart_count == 3
    # Malformed/foreign identity cannot become a durable native head hint.
    observer = SlurmObservation(
        Runner(scontrol=control(receipt, Comment="foreign")), owner="alice"
    )
    with pytest.raises(ValueError) as caught:
        observer.discover_restart(active)
    assert not isinstance(caught.value, RestartBindingUnavailable)


@pytest.mark.parametrize(
    "changes",
    [
        {"Restarts": "broken"},
        {"Restarts": "-1"},
        {"Restarts": str(2**63)},
        {"Restarts": ""},
        {"Cluster": "foreign"},
        {"UserId": "bob(1000)"},
        {"JobName": "foreign"},
        {"Comment": "foreign"},
        {"JobState": ""},
        {"JobId": "43"},
    ],
)
def test_invalid_controller_proof_never_exposes_head(tmp_path, changes):
    handle = submission(tmp_path)
    result = SlurmObservation(
        Runner(scontrol=control(handle, **changes)), owner="alice"
    ).observe_submission(handle)
    assert result.native_identity is None
    assert result.status in {ObservationStatus.ERROR, ObservationStatus.ABSENT}


@pytest.mark.parametrize(
    "changes",
    [
        {"restart": "broken"},
        {"restart": "-1"},
        {"restart": str(2**63)},
        {"cluster": "foreign"},
        {"owner": "bob"},
        {"name": "foreign"},
        {"comment": "foreign"},
        {"comment": ""},
        {"state": ""},
    ],
)
def test_invalid_accounting_proof_never_exposes_head(tmp_path, changes):
    handle = submission(tmp_path)
    result = SlurmObservation(
        Runner(sacct=accounting(handle, **changes)), owner="alice"
    ).observe_submission(handle)
    assert result.status == ObservationStatus.ERROR
    assert result.native_identity is None


@pytest.mark.parametrize(
    "rows",
    [
        lambda h: accounting(h, restart="4"),
        lambda h: accounting(h, owner="bob"),
        lambda h: accounting(h) + accounting(h),
    ],
)
def test_accounting_conflict_prevents_controller_head(tmp_path, rows):
    handle = submission(tmp_path)
    result = SlurmObservation(
        Runner(scontrol=control(handle), sacct=rows(handle)), owner="alice"
    ).observe_submission(handle)
    assert result.status == ObservationStatus.ERROR
    assert result.native_identity is None


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETED", "FUTURE_NATIVE_STATE"])
def test_queue_only_never_invents_restart_from_saved_ack(tmp_path, state):
    handle = submission(tmp_path)
    result = SlurmObservation(
        Runner(squeue=queue(handle, state)), owner="alice"
    ).observe_submission(handle)
    assert result.native_identity is None


@pytest.mark.parametrize(
    "responses",
    [
        {},
        {"scontrol": "No jobs in the system\n"},
        {"scontrol": ProcessResult(1, b"", b"Access denied")},
        {"sacct": ProcessResult(1, b"", b"Accounting unavailable")},
        {"squeue": ProcessResult(0, b"partial", b"", stdout_truncated=True)},
        {"scontrol": "malformed"},
        {"sacct": "malformed"},
    ],
)
def test_absent_or_failed_native_rows_never_expose_head(tmp_path, responses):
    result = SlurmObservation(Runner(**responses), owner="alice").observe_submission(
        submission(tmp_path)
    )
    assert result.native_identity is None


def test_unknown_native_state_does_not_activate_initial_writer(tmp_path):
    handle = submission(tmp_path)
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    with pytest.raises(ValueError, match="state unavailable"):
        observer.discover_initial(handle)
    assert not (tmp_path / "attempts").exists()


def test_mixed_batch_keeps_heads_per_job_with_exact_three_query_policy(tmp_path):
    unknown, running, malformed, absent, queued = (
        submission(tmp_path, str(job)) for job in range(42, 47)
    )
    runner = Runner(
        squeue=queue(queued),
        scontrol=control(unknown)
        + control(running, JobState="RUNNING", Restarts="1")
        + control(malformed, Comment="foreign"),
    )
    result = SlurmObservation(runner, owner="alice").observe_many(
        (unknown, running, malformed, absent, queued)
    )
    assert result[unknown.job_id].status == ObservationStatus.ERROR
    assert result[unknown.job_id].native_identity == identity(unknown)
    assert result[running.job_id].status == ObservationStatus.RUNNING
    assert result[running.job_id].native_identity == identity(running, 1)
    assert result[malformed.job_id].status == ObservationStatus.ERROR
    assert result[absent.job_id].status == ObservationStatus.ABSENT
    assert result[queued.job_id].status == ObservationStatus.RUNNING
    for handle in (malformed, absent, queued):
        assert result[handle.job_id].native_identity is None
    assert runner.calls == [
        (
            "squeue",
            "--noheader",
            "--local",
            "--clusters",
            "alpha",
            "--jobs",
            "42,43,44,45,46",
            "--states=all",
            "--format=%i|%u|%j|%T|%r|%k",
        ),
        ("scontrol", "--clusters", "alpha", "--oneliner", "show", "jobs"),
        (
            "sacct",
            "--noheader",
            "--parsable2",
            "--duplicates",
            "--clusters",
            "alpha",
            "--jobs",
            "42,43,44,45,46",
            "--format=JobIDRaw,Cluster,User,JobName%256,State%64,ExitCode,Restarts,Submit,Comment%256",
        ),
    ]
