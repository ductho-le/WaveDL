"""Writer Slurm submission contracts with fake native scheduler commands only."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.scheduler_attempt import SchedulerIdentity, coordinate_attempt
from wavedl.core.submission_outcomes import write_submission_outcome_receipt
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    OutcomeStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.slurm import SlurmBackend
from wavedl_daemon.backends.slurm_observation import SlurmObservation
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite, bind_submission
from wavedl_daemon.backends.slurm_submission import (
    CancellationError,
    SchedulerUnavailable,
    SlurmSubmission,
    SubmissionUnknown,
)
from wavedl_daemon.storage.ownership import DispatchLock
from wavedl_daemon.supervision.base import ProcessResult


pwd = pytest.importorskip(
    "pwd", reason="Slurm writer submission requires POSIX identity"
)


class Runner:
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


@pytest.fixture
def submitted(tmp_path):
    layout = RunLayout.create(tmp_path, str(uuid4()))
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 1, "model": "mlp", "data_path": "train.npz"},
        runtime=RuntimeSelection("/usr/bin/python3", "selected", "cpu", {}),
        submission_cwd=tmp_path,
    )
    plan = SlurmPlanner(SlurmSite(partition="cpu", cluster="alpha", requeue=True)).plan(
        manifest, allocation_mode="writer"
    )
    runner = Runner(sbatch="42;alpha\n")
    with DispatchLock(tmp_path / "dispatch.lock") as lock:
        backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
        handle = backend.submit(plan, request_id=str(uuid4()), attempt=None)
        yield layout, plan, handle, runner, backend


def control(handle, **changes):
    fields = {
        "JobId": "42",
        "UserId": "alice(1000)",
        "JobName": f"wavedl-{handle.request_id}",
        "Comment": handle.request_id,
        "JobState": "RUNNING",
        "Restarts": "0",
        "ExitCode": "0:0",
    } | changes
    return " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def identity(handle, restart=0):
    return SchedulerIdentity(
        handle.run_id, handle.request_id, "alpha", "42", "alice", restart
    )


def test_writer_submission_has_no_attempt_and_immutable_namespace(submitted):
    layout, plan, handle, runner, backend = submitted
    assert isinstance(handle, SubmissionHandle)
    assert "attempt" not in handle.to_dict()
    assert list(layout.attempts.iterdir()) == []
    namespace = layout.run_dir / "submissions" / handle.request_id
    assert Path(runner.calls[0][-1]) == namespace / "submission.sh"
    assert (namespace / "plan.json").read_text() == plan.to_json()
    script = (namespace / "submission.sh").read_text()
    assert "unset WAVEDL_ATTEMPT" in script
    assert "WAVEDL_ATTEMPT_ALLOCATION=writer" in script
    assert hashlib.sha256(plan.to_json().encode()).hexdigest() in script
    assert "@@WAVEDL_" not in script
    before = {
        path.name: path.read_bytes() for path in namespace.iterdir() if path.is_file()
    }
    with pytest.raises(SubmissionUnknown, match="reserved"):
        backend.submit(plan, request_id=handle.request_id, attempt=None)
    assert len(runner.calls) == 1
    assert {
        path.name: path.read_bytes() for path in namespace.iterdir() if path.is_file()
    } == before


def test_mode_remains_explicit_and_reserved_default(submitted):
    _, plan, handle, _, _ = submitted
    site = SlurmSite(partition="cpu", cluster="alpha")
    backend = SlurmBackend(site, Runner(), owner="alice")
    assert backend.plan(plan.manifest).allocation_mode == "reserved"
    writer = backend.plan(plan.manifest, allocation_mode="writer")
    assert writer.allocation_mode == "writer"
    with pytest.raises(ValueError, match="reserve"):
        bind_submission(writer, handle.request_id, 1)


def test_discover_initial_needs_native_proof_and_writer_binding(submitted):
    layout, _, handle, _, _ = submitted
    runner = Runner(scontrol=control(handle))
    observer = SlurmObservation(runner, owner="alice")
    assert observer.discover_initial(handle) is None
    assert list(layout.attempts.iterdir()) == []
    binding = coordinate_attempt(layout.run_dir, identity(handle), initial_attempt=None)
    assert observer.discover_initial(handle) == binding
    assert list(layout.attempts.iterdir()) == [layout.attempt_dir(binding.attempt)]
    runner.responses["scontrol"] = control(handle, Comment=str(uuid4()))
    with pytest.raises(ValueError, match="identity"):
        observer.discover_initial(handle)


@pytest.mark.parametrize("alias", ["0", "1", "bad", ""])
def test_initial_discovery_rejects_conflicting_restart_aliases(submitted, alias):
    layout, _, handle, _, _ = submitted
    binding = coordinate_attempt(layout.run_dir, identity(handle), initial_attempt=None)
    observer = SlurmObservation(
        Runner(scontrol=control(handle, RestartCnt=alias)), owner="alice"
    )
    if alias == "0":
        assert observer.discover_initial(handle) == binding
        assert observer.observe(handle).native_identity.restart_count == 0
    else:
        with pytest.raises(ValueError, match="restart"):
            observer.discover_initial(handle)
        assert observer.observe(handle).status == ObservationStatus.ERROR


@pytest.mark.parametrize(
    "changed",
    [
        {"Restarts": "bad"},
        {"Cluster": "other"},
        {"UserId": "bob(1000)"},
        {"JobId": "43"},
    ],
)
def test_native_identity_mismatch_never_discovers_attempt(submitted, changed):
    layout, _, handle, _, _ = submitted
    coordinate_attempt(layout.run_dir, identity(handle), initial_attempt=None)
    observer = SlurmObservation(
        Runner(scontrol=control(handle, **changed)), owner="alice"
    )
    with pytest.raises(ValueError):
        observer.discover_initial(handle)


def test_attemptless_observation_and_cancel_use_native_identity(submitted):
    _, _, handle, runner, backend = submitted
    observer = SlurmObservation(
        Runner(scontrol=control(handle, JobState="PENDING")), owner="alice"
    )
    observation = observer.observe(handle)
    assert isinstance(observation, SubmissionObservation)
    assert observation.status == ObservationStatus.QUEUED
    assert not hasattr(observation, "attempt")
    runner.responses["squeue"] = (
        f"42|alice|wavedl-{handle.request_id}|{handle.request_id}|alpha\n"
    )
    backend.request_cancel(handle)
    assert runner.calls[-1][0] == "scancel"
    runner.responses["squeue"] = "42|bob|wrong|wrong|alpha\n"
    with pytest.raises(CancellationError):
        backend.request_cancel(handle)
    assert runner.calls[-1][0] == "squeue"


def test_writer_restart_reads_new_writer_claim_without_reserved_attempt(submitted):
    layout, _, handle, _, _ = submitted
    initial = coordinate_attempt(layout.run_dir, identity(handle), initial_attempt=None)
    active = handle.bind_attempt(initial.attempt)
    restarted = coordinate_attempt(
        layout.run_dir, identity(handle, 1), initial_attempt=None
    )
    observer = SlurmObservation(
        Runner(scontrol=control(handle, Restarts="1")), owner="alice"
    )
    binding, proposed = observer.discover_restart(active)
    assert binding == restarted
    assert proposed.attempt == restarted.attempt
    assert json.loads(proposed.raw_json)["restart_count"] == 1
    assert observer.discover_restart(proposed) is None


@pytest.mark.parametrize("bound", [False, True])
def test_writer_collection_keeps_command_outcomes_independent(submitted, bound):
    layout, plan, handle, _, _ = submitted
    attempt = None
    if bound:
        binding = coordinate_attempt(
            layout.run_dir, identity(handle), initial_attempt=None
        )
        attempt = binding.attempt
        handle = handle.bind_attempt(attempt)
    write_submission_outcome_receipt(
        layout.run_dir,
        request_id=handle.request_id,
        owner="alice",
        plan_sha256=hashlib.sha256(plan.to_json().encode()).hexdigest(),
        job_id="42",
        cluster="alpha",
        restart_count=0,
        operation="train",
        runtime_exit=0 if bound else 125,
        batch_exit=0 if bound else 9,
        stage_out_exit=7,
        attempt=attempt,
    )
    observer = SlurmObservation(
        Runner(scontrol=control(handle, JobState="FAILED", ExitCode="7:0")),
        owner="alice",
    )
    collected = observer.collect(handle)
    assert collected.runtime.status == (
        OutcomeStatus.SUCCESS if bound else OutcomeStatus.UNKNOWN
    )
    assert collected.batch.exit_code == (0 if bound else 9)
    assert collected.stage_out.exit_code == 7
    if not bound:
        assert list(layout.attempts.iterdir()) == []


def test_writer_recovery_reads_submission_namespace(submitted):
    _, plan, handle, runner, backend = submitted
    runner.responses["squeue"] = (
        f"42|alice|wavedl-{handle.request_id}|{handle.request_id}|alpha\n"
    )
    recovered = backend.find_by_request_id(
        plan, request_id=handle.request_id, attempt=None
    )
    assert isinstance(recovered, SubmissionHandle)
    assert recovered.job_id == handle.job_id
    with pytest.raises(SchedulerUnavailable):
        backend.find_by_request_id(
            replace(plan, warnings=("changed",)),
            request_id=handle.request_id,
            attempt=None,
        )


def test_actual_setup_failure_publishes_receipt_without_allocating(tmp_path):
    """The generated batch wrapper handles failure before the binder is called."""
    layout = RunLayout(tmp_path / "run", str(uuid4()))
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 1, "model": "mlp", "data_path": "train.npz"},
        runtime=RuntimeSelection(sys.executable, "selected", "cpu", {}),
        submission_cwd=tmp_path,
    )
    site = SlurmSite(partition="cpu", cluster="alpha", setup_commands=("false",))
    plan = SlurmPlanner(site).plan(manifest, allocation_mode="writer")
    runner = Runner(sbatch="42;alpha\n")
    owner = pwd.getpwuid(os.getuid()).pw_name
    with DispatchLock(tmp_path / "dispatch.lock") as lock:
        handle = SlurmSubmission(runner, owner=owner, dispatch_lock=lock).submit(
            plan, request_id=str(uuid4()), attempt=None
        )
    source = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        ("/bin/bash", runner.calls[0][-1]),
        env=os.environ
        | {"PYTHONPATH": str(source), "SLURM_JOB_ID": "42", "SLURM_RESTART_COUNT": "0"},
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1, result.stderr.decode()
    assert not layout.attempts.exists()
    assert not list(layout.run_dir.rglob("events.jsonl"))
    namespace = layout.run_dir / "submissions" / handle.request_id
    receipts = list(namespace.glob("slurm-outcome-*.json"))
    assert len(receipts) == 1
    outcome = json.loads(receipts[0].read_bytes())
    assert outcome["attempt"] is None
    assert outcome["runtime"]["status"] == "unknown"
    assert outcome["batch"]["exit_code"] == 1
    assert outcome["stage_out"]["exit_code"] == 0


def test_unavailable_query_and_ambiguous_native_record_are_typed_errors(submitted):
    _, _, handle, _, _ = submitted
    observer = SlurmObservation(Runner(scontrol=OSError("unavailable")), owner="alice")
    assert observer.observe(handle).status == ObservationStatus.ERROR
    observer.runner.responses["scontrol"] = control(handle) * 2
    assert observer.observe(handle).status == ObservationStatus.ERROR


def test_changed_plan_file_is_not_attempt_adoption_authority(submitted):
    layout, _, handle, _, _ = submitted
    coordinate_attempt(layout.run_dir, identity(handle), initial_attempt=None)
    (layout.run_dir / "submissions" / handle.request_id / "plan.json").write_text("{}")
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    assert observer.discover_initial(handle) is None


def test_retained_accounting_proves_completed_initial_writer(submitted):
    layout, _, handle, _, _ = submitted
    binding = coordinate_attempt(layout.run_dir, identity(handle), initial_attempt=None)
    account = f"42|alpha|alice|wavedl-{handle.request_id}|COMPLETED|0:0|0|2026-10-01T10:00:00|{handle.request_id}\n"
    runner = Runner(sacct=account)
    observer = SlurmObservation(runner, owner="alice")
    assert observer.discover_initial(handle) == binding
    observation = observer.observe(handle)
    assert isinstance(observation, SubmissionObservation)
    assert observation.status == ObservationStatus.TERMINAL
    runner.responses["sacct"] = account.replace(handle.request_id, "(null)")
    with pytest.raises(ValueError, match="identity unavailable"):
        observer.discover_initial(handle)


def test_unbound_cluster_batch_keeps_scheduler_calls_bounded(submitted):
    _, _, first, _, _ = submitted
    request = str(uuid4())
    second = replace(
        first,
        request_id=request,
        job_id="43;alpha",
        raw_json=json.dumps(
            json.loads(first.raw_json)
            | {"job_name": f"wavedl-{request}", "comment": request}
        ),
    )
    runner = Runner(scontrol=control(first) + control(second, JobId="43"))
    observations = SlurmObservation(runner, owner="alice").observe_many((first, second))
    assert all(
        isinstance(item, SubmissionObservation) for item in observations.values()
    )
    assert [call[0] for call in runner.calls] == ["squeue", "scontrol", "sacct"]


def test_unbound_ack_never_attributes_newer_setup_failure_to_generation_zero(submitted):
    """A restart-2 receipt cannot be collected under the original restart-0 ACK."""
    layout, plan, handle, _, _ = submitted
    write_submission_outcome_receipt(
        layout.run_dir,
        request_id=handle.request_id,
        owner="alice",
        plan_sha256=hashlib.sha256(plan.to_json().encode()).hexdigest(),
        job_id="42",
        cluster="alpha",
        restart_count=2,
        operation="train",
        runtime_exit=125,
        batch_exit=9,
        stage_out_exit=0,
    )
    observer = SlurmObservation(
        Runner(
            scontrol=control(handle, Restarts="2", JobState="FAILED", ExitCode="9:0")
        ),
        owner="alice",
    )
    collected = observer.collect(handle)
    assert collected.runtime.status == OutcomeStatus.UNKNOWN
    assert collected.stage_out.status == OutcomeStatus.UNKNOWN
    assert collected.batch.exit_code == 9
    assert json.loads(handle.raw_json)["restart_count"] == 0
    assert list(layout.attempts.iterdir()) == []
