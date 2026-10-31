"""Real immutable receipts and fake native commands, without runtime allocation."""

import hashlib
import json
import os
from dataclasses import replace
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl.core.submission_outcomes import write_submission_outcome_receipt
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    OutcomeStatus,
    Plan,
    Topology,
    UnsupportedPlan,
)
from wavedl_daemon.backends.slurm import SlurmBackend
from wavedl_daemon.backends.slurm_observation import SlurmObservation
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite
from wavedl_daemon.backends.slurm_submission import SlurmSubmission
from wavedl_daemon.storage.ownership import DispatchLock
from wavedl_daemon.supervision.base import ProcessResult


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
def submission(tmp_path):
    def create(*, requeue=True):
        layout = RunLayout.create(tmp_path, str(uuid4()))
        manifest = build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={"epochs": 1, "model": "mlp", "data_path": "train.npz"},
            runtime=RuntimeSelection("/usr/bin/python3", "selected", "cpu", {}),
            submission_cwd=tmp_path,
        )
        site = SlurmSite(partition="cpu", cluster="alpha", requeue=requeue)
        planner = SlurmPlanner(site)
        if os.name == "nt":
            # These tests persist real receipts but execute only fake scheduler
            # commands. Native Windows paths must remain refused by the real
            # shell/directive planner; a typed inert plan isolates collection.
            with pytest.raises(UnsupportedPlan, match="Slurm directive paths"):
                planner.plan(manifest, allocation_mode="writer")
            topology = Topology()
            plan = Plan(
                target="slurm",
                manifest_json=json.dumps(manifest.to_dict()),
                script=(
                    "#!/bin/sh\n"
                    "# Fake scheduler only: @@WAVEDL_REQUEST_ID@@ "
                    "@@WAVEDL_JOB_NAME@@ @@WAVEDL_PLAN_SHA256@@\n"
                ),
                topology=topology,
                resources=tuple(
                    sorted(
                        {
                            "partition": site.partition,
                            "account": site.account,
                            "cluster": site.cluster,
                            "nodes": topology.num_machines,
                            "ntasks": topology.num_machines,
                            "ntasks_per_node": 1,
                            "processes_per_machine": topology.processes_per_machine,
                            "num_processes": topology.num_machines
                            * topology.processes_per_machine,
                            "cpus_per_task": site.cpus_per_task,
                            "memory_per_node": site.memory,
                            "gpus_per_node": site.gpus_per_node,
                            "time": site.time,
                            "constraint": site.constraint,
                            "rendezvous_port": site.rendezvous_port,
                            "requeue": site.requeue,
                        }.items()
                    )
                ),
                environment=tuple(sorted(manifest.runtime.environment.items())),
                allocation_mode="writer",
            )
            assert plan.manifest == manifest
        else:
            plan = planner.plan(manifest, allocation_mode="writer")
        with DispatchLock(tmp_path / "dispatch.lock") as lock:
            handle = SlurmSubmission(
                Runner(sbatch="42;alpha\n"), owner="alice", dispatch_lock=lock
            ).submit(plan, request_id=str(uuid4()), attempt=None)
        return layout, plan, handle, site

    return create


def control(handle, **changes):
    fields = {
        "JobId": "42",
        "UserId": "alice(1000)",
        "JobName": f"wavedl-{handle.request_id}",
        "Comment": handle.request_id,
        "JobState": "FAILED",
        "Restarts": "2",
        "ExitCode": "9:0",
        "Cluster": "alpha",
    } | changes
    return " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def receipt(layout, plan, handle, **changes):
    values = {
        "request_id": handle.request_id,
        "owner": "alice",
        "plan_sha256": hashlib.sha256(plan.to_json().encode()).hexdigest(),
        "job_id": "42",
        "cluster": "alpha",
        "restart_count": 2,
        "operation": "train",
        "runtime_exit": 125,
        "batch_exit": 9,
        "stage_out_exit": 0,
    } | changes
    return write_submission_outcome_receipt(layout.run_dir, **values)


@pytest.mark.parametrize("accounting", [False, True])
def test_ack_zero_collects_actual_current_receipt_without_runtime(
    submission, accounting
):
    layout, plan, handle, site = submission()
    receipt(layout, plan, handle)
    responses = (
        {
            "sacct": f"42|alpha|alice|wavedl-{handle.request_id}|FAILED|9:0|2|2026-10-01T10:00:00|{handle.request_id}\n"
        }
        if accounting
        else {"scontrol": control(handle)}
    )
    runner = Runner(**responses)
    backend = SlurmBackend(site, runner, owner="alice")
    before = sorted(
        str(path.relative_to(layout.run_dir)) for path in layout.run_dir.rglob("*")
    )
    proposal = backend.collect_submission(handle)
    assert proposal is not None
    assert proposal.identity.restart_count == 2
    assert proposal.observation.native_identity == proposal.identity
    assert proposal.plan_sha256 == hashlib.sha256(plan.to_json().encode()).hexdigest()
    assert (
        proposal.manifest_sha256
        == hashlib.sha256((layout.run_dir / "manifest.json").read_bytes()).hexdigest()
    )
    assert proposal.result.runtime.status == OutcomeStatus.UNKNOWN
    assert proposal.result.runtime.exit_code is None
    assert proposal.result.batch.exit_code == 9
    assert proposal.result.stage_out.status == OutcomeStatus.SUCCESS
    assert json.loads(handle.raw_json)["restart_count"] == 0
    assert list(layout.attempts.iterdir()) == []
    assert not list(layout.run_dir.rglob("events.jsonl"))
    assert (
        sorted(
            str(path.relative_to(layout.run_dir)) for path in layout.run_dir.rglob("*")
        )
        == before
    )
    legacy = backend.collect(handle)
    assert legacy.runtime.status == OutcomeStatus.UNKNOWN
    assert legacy.stage_out.status == OutcomeStatus.UNKNOWN


@pytest.mark.parametrize(
    "changes",
    [
        {"Cluster": "foreign"},
        {"Comment": str(uuid4())},
        {"Restarts": "garbled"},
        {"Restarts": "-1"},
        {"JobState": "RUNNING"},
        {"JobState": "FUTURE_UNKNOWN"},
    ],
)
def test_native_guard_never_returns_collection(submission, changes):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle)
    observer = SlurmObservation(
        Runner(scontrol=control(handle, **changes)), owner="alice"
    )
    assert observer.collect_submission(handle) is None
    if changes.get("JobState") == "FUTURE_UNKNOWN":
        # Unknown lifecycle does not erase the independently verified native
        # restart identity, but still cannot certify terminal collection.
        observation = observer.observe_submission(handle)
        assert observation.status == ObservationStatus.ERROR
        job, cluster = handle.job_id.split(";")
        assert observation.native_identity == SchedulerIdentity(
            handle.run_id, handle.request_id, cluster, job, handle.owner, 2
        )
    elif changes.get("JobState") != "RUNNING":
        assert observer.observe_submission(handle).native_identity is None


def test_unapproved_restart_receipt_is_not_authority(submission):
    layout, plan, handle, _ = submission(requeue=False)
    receipt(layout, plan, handle)
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    assert observer.collect_submission(handle) is None


@pytest.mark.parametrize("malformed", [False, True])
def test_missing_or_malformed_exact_receipt_returns_no_proposal(submission, malformed):
    layout, plan, handle, _ = submission()
    if malformed:
        path = receipt(layout, plan, handle)
        path.write_text('{"schema_version":2,"schema_version":2}')
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    assert observer.collect_submission(handle) is None


def test_accounting_failure_and_unproven_queue_generation_are_unavailable(submission):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle)
    runner = Runner(
        squeue=f"42|alice|wavedl-{handle.request_id}|FAILED||{handle.request_id}\n",
        scontrol=ProcessResult(1, b"", b"not available"),
        sacct=ProcessResult(1, b"", b"not available"),
    )
    observer = SlurmObservation(runner, owner="alice")
    assert observer.collect_submission(handle) is None
    assert observer.observe_submission(handle).native_identity is None


@pytest.mark.parametrize("member", ["plan.json", "request.json", "manifest.json"])
def test_plan_and_manifest_tampering_block_collection(submission, member):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle)
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    path = (
        layout.run_dir / "manifest.json"
        if member == "manifest.json"
        else layout.run_dir / "submissions" / handle.request_id / member
    )
    path.write_text("{}")
    assert observer.collect_submission(handle) is None
    assert list(layout.attempts.iterdir()) == []


@pytest.mark.parametrize("mutation", ["cluster", "request", "ambiguous", "restart"])
def test_retained_accounting_identity_guards(submission, mutation):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle)
    row = f"42|alpha|alice|wavedl-{handle.request_id}|FAILED|9:0|2|2026-10-01T10:00:00|{handle.request_id}\n"
    if mutation == "cluster":
        row = row.replace("|alpha|", "|foreign|")
    elif mutation == "request":
        row = row.replace(handle.request_id, str(uuid4()))
    elif mutation == "ambiguous":
        row *= 2
    else:
        row = row.replace("|2|", "|garbled|")
    observer = SlurmObservation(Runner(sacct=row), owner="alice")
    assert observer.collect_submission(handle) is None
    assert observer.observe_submission(handle).native_identity is None


def test_unknown_batch_stage_out_remain_independent_receipt_evidence(submission):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle, batch_exit=125, stage_out_exit=125)
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    proposal = observer.collect_submission(handle)
    assert proposal is not None
    assert proposal.result.batch.status == OutcomeStatus.UNKNOWN
    assert proposal.result.stage_out.status == OutcomeStatus.UNKNOWN
    assert proposal.observation.batch.exit_code == 9


def test_saved_native_generation_cannot_regress(submission):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle)
    current = replace(
        handle,
        raw_json=json.dumps(json.loads(handle.raw_json) | {"restart_count": 3}),
    )
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    assert observer.collect_submission(current) is None


def test_consistent_foreign_cluster_cannot_override_approved_plan(submission):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle, cluster="foreign")
    foreign = replace(
        handle,
        job_id="42;foreign",
        raw_json=json.dumps(json.loads(handle.raw_json) | {"cluster": "foreign"}),
    )
    observer = SlurmObservation(
        Runner(scontrol=control(foreign, Cluster="foreign")), owner="alice"
    )
    assert observer.observe_submission(foreign).native_identity.cluster == "foreign"
    assert observer.collect_submission(foreign) is None


def test_receipt_generation_cannot_follow_diagnostic_text(submission, monkeypatch):
    layout, plan, handle, _ = submission()
    receipt(layout, plan, handle)
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    verified = replace(observer.observe_submission(handle), detail="restart=999999")
    monkeypatch.setattr(observer, "observe_submission", lambda _: verified)
    proposal = observer.collect_submission(handle)
    assert proposal is not None
    assert proposal.identity.restart_count == 2


def test_observe_many_attaches_only_verified_full_native_identity(submission):
    _, _, handle, _ = submission()
    observer = SlurmObservation(Runner(scontrol=control(handle)), owner="alice")
    observation = observer.observe_many((handle,))[handle.job_id]
    assert observation.status == ObservationStatus.TERMINAL
    assert observation.native_identity.run_id == handle.run_id
    assert observation.native_identity.request_id == handle.request_id
    assert observation.native_identity.cluster == "alpha"
    assert observation.native_identity.job_id == "42"
    assert observation.native_identity.owner == "alice"
    assert observation.native_identity.restart_count == 2
