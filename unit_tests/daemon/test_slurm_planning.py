"""Pure Slurm plans, and their batch script run by real Bash with fake Slurm."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import replace
from uuid import uuid4

import pytest

from wavedl_daemon.backends.contracts import (
    Plan,
    Runtime,
    StorageMapping,
    Topology,
    UnsupportedPlan,
)
from wavedl_daemon.backends.slurm_exit import parse_exit_record
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite, job_name
from wavedl_daemon.events.metrics_tail import read_tail
from wavedl_daemon.runs.model import ExitRecord, Metric, Operation
from wavedl_daemon.runs.service import build_spec


posix = pytest.mark.skipif(os.name != "posix", reason="Real Bash runs the batch script")


def spec(tmp_path, *, device="cpu", operation=Operation.TRAIN, python=None, **config):
    defaults = (
        {"model": "cnn", "data_path": "data.npz"}
        if operation is Operation.TRAIN
        else {"checkpoint": "best", "data_path": "data.npz"}
    )
    return build_spec(
        tmp_path / "owned" / str(uuid4()),
        operation=operation,
        resolved_config=defaults | config,
        runtime=Runtime(
            python or str(tmp_path / "venv" / "bin" / "python"), "selected", device, {}
        ),
        submission_cwd=tmp_path,
    )


def planner(**kwargs):
    return SlurmPlanner(
        SlurmSite(partition="compute", cluster="test-cluster", **kwargs)
    )


def test_plan_is_pure_and_round_trips(tmp_path):
    request = spec(tmp_path)
    plan = planner().plan(request)
    assert not (tmp_path / "owned").exists()
    assert plan == Plan.from_json(plan.to_json())
    assert plan.argv == ()
    assert f"#SBATCH --job-name=wavedl-{request.run_id}" in plan.script
    assert f"#SBATCH --comment={request.run_id}" in plan.script
    assert "/logs/slurm-%j.out" in plan.script
    assert "--no-requeue" in plan.script
    assert "#SBATCH --export=NIL" in plan.script
    assert "export SLURM_EXPORT_ENV=ALL" in plan.script
    assert f"--output_dir {request.working_directory}/output" in plan.script
    assert '--metrics_jsonl "$WAVEDL_ATTEMPT/metrics.jsonl"' in plan.script
    assert "wavedl_daemon" not in plan.script and "scheduler-bind" not in plan.script


def test_test_runs_save_their_evaluation_in_the_attempt(tmp_path):
    plan = planner().plan(spec(tmp_path, operation=Operation.TEST, device="auto"))
    assert "-m wavedl.test" in plan.script
    assert '--metrics_json "$WAVEDL_ATTEMPT/evaluation.json"' in plan.script
    assert "--metrics_jsonl" not in plan.script


@pytest.mark.parametrize(
    "identity",
    ["prefix", "A" * 32, "11111111111141118111111111111111", "x\n#SBATCH --nodes=999"],
)
def test_noncanonical_run_id_is_refused(identity):
    with pytest.raises(ValueError):
        job_name(identity)


@pytest.mark.parametrize(
    "changes",
    [
        {"partition": "gpu\n#SBATCH --nodes=99"},
        {"account": "bad account"},
        {"constraint": "gpu#bad"},
        {"memory": "0G"},
        {"time": ""},
        {"cpus_per_task": True},
        {"gpus_per_node": -1},
        {"rendezvous_port": 65536},
        {"setup_commands": ("true\nfalse",)},
    ],
)
def test_site_rejects_unexpressible_or_injectable_settings(changes):
    with pytest.raises(UnsupportedPlan):
        SlurmSite(**({"partition": "cpu", "cluster": "site"} | changes))


def test_explicit_cluster_required(tmp_path):
    with pytest.raises(UnsupportedPlan, match="cluster"):
        SlurmPlanner(SlurmSite(partition="cpu")).plan(spec(tmp_path))


def test_topology_exact_total_and_one_launcher_per_machine(tmp_path):
    plan = planner(gpus_per_node=4).plan(
        spec(tmp_path, device="cuda"), topology=Topology(3, 4)
    )
    assert dict(plan.resources)["num_processes"] == 12
    assert "--num_gpus 12" in plan.script
    assert "--num_machines 3" in plan.script
    assert '--machine_rank "$SLURM_PROCID"' in plan.script
    assert "srun --nodes=3 --ntasks=3 --ntasks-per-node=1" in plan.script
    assert "#SBATCH --gpus-per-node=4" in plan.script
    assert "--main_process_port 29500" in plan.script
    assert "--cpu " not in plan.script


def test_cross_validation_distributed_shape_is_refused_at_preview(tmp_path):
    with pytest.raises(UnsupportedPlan, match="Cross-validation"):
        planner(gpus_per_node=2).plan(
            spec(tmp_path, device="cuda", cv=2), topology=Topology(1, 2)
        )


@pytest.mark.parametrize("topology", [Topology(2), Topology(1, 2), Topology(2, 2, 1)])
def test_cpu_unexpressible_topology_refused(tmp_path, topology):
    with pytest.raises(UnsupportedPlan):
        planner().plan(spec(tmp_path), topology=topology)


def test_unsupported_config_modes_and_gpu_settings_refused(tmp_path):
    request = spec(tmp_path)
    for changed in (
        replace(request, resolved_config={"model": "cnn"}),
        replace(request, runtime=replace(request.runtime, mode="bundled")),
        replace(request, runtime=replace(request.runtime, device="auto")),
        replace(request, runtime=replace(request.runtime, device="cuda")),
        replace(
            request,
            runtime=replace(request.runtime, environment={"WAVEDL_ATTEMPT": "3"}),
        ),
        replace(
            request,
            runtime=replace(request.runtime, environment={"SLURM_RESTART_COUNT": "0"}),
        ),
    ):
        with pytest.raises(UnsupportedPlan):
            planner().plan(changed)
    with pytest.raises(UnsupportedPlan):
        planner(gpus_per_node=2).plan(spec(tmp_path, device="cuda"))


def test_mapped_execution_paths_remain_separate_from_host_paths(tmp_path):
    request = build_spec(
        tmp_path / "owned" / str(uuid4()),
        operation=Operation.TRAIN,
        resolved_config={"model": "cnn", "data_path": "data.npz"},
        runtime=Runtime("venv/bin/python", "selected", "cpu", {}),
        submission_cwd=tmp_path,
        storage_map=(StorageMapping(str(tmp_path), "/compute/shared"),),
    )
    plan = planner().plan(request)
    run = f"/compute/shared/owned/{request.run_id}"
    assert "--data_path=/compute/shared/data.npz" in plan.script
    assert f"#SBATCH --output={run}/logs/slurm-%j.out" in plan.script
    assert f"export WAVEDL_ATTEMPT={run}/attempts/" in plan.script
    assert f"--output_dir {run}/output" in plan.script
    assert str(tmp_path) not in plan.script


def test_unsafe_directive_paths_are_refused(tmp_path):
    for suffix in ("percent%", "quote'", 'quote"', "slash\\", "newline\n"):
        with pytest.raises(UnsupportedPlan):
            planner().plan(spec(tmp_path / suffix))


TRAINER = """#!/bin/bash
if [ "${SLURM_JWT+x}" = x ]; then exit 91; fi
if [ "$1" = "-c" ]; then exit "${PROBE_EXIT:-0}"; fi
printf '%s\\n' "$@" >> "$CAPTURE"
if [ -n "${INTS-}" ]; then
    # A launcher: stop on SIGINT after an emergency save, logging every SIGINT.
    trap 'printf "INT %s\\n" "$SLURM_NODEID" >> "$INTS"; interrupted=1' INT
    interrupted=0
    printf 'ready %s\\n' "$SLURM_NODEID" >> "$INTS"
    until [ "$interrupted" = 1 ]; do sleep 0.02; done
    sleep 0.3
    exit 130
fi
printf '%s\\n' "${SPECIAL-}" "${N-}" "${A-}" > "$CAPTURE.env"
while [ "$#" -gt 0 ]; do
    if [ "$1" = --metrics_jsonl ]; then printf '{"epoch": 1, "val_loss": 0.5}\\n' > "$2"; fi
    shift
done
exit "${TRAINING_EXIT:-0}"
"""
FAKE_SLURM = {
    "scontrol": "#!/bin/bash\nprintf 'host0\\nhost1\\n'",
    # One task per machine, each with its own rank, started as slurmstepd does:
    # not with the SIGINT a backgrounded srun ignores. Exits with the highest code.
    "srun": f"""#!{sys.executable}
import os, signal, subprocess, sys
if "SLURM_JWT" in os.environ:
    sys.exit(92)
args = sys.argv[1:]
with open(os.environ["SRUN_LOG"], "a") as log:
    log.write(" ".join(args) + "\\n")
while args[0].startswith("--"):
    args.pop(0)
def task(rank):
    env = os.environ | {{"SLURM_PROCID": str(rank), "SLURM_NODEID": str(rank)}}
    return subprocess.Popen(
        args, env=env, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL)
    )
codes = []
if os.environ.get("FAKE_CONCURRENT"):
    tasks = [task(rank) for rank in range(int(os.environ.get("FAKE_NODES", "1")))]
    codes = [process.wait() for process in tasks]
else:
    for rank in range(int(os.environ.get("FAKE_NODES", "1"))):
        codes.append(task(rank).wait())
        if codes[-1]:
            break
sys.exit(max(128 - code if code < 0 else code for code in codes))
""",
}


def batch(tmp_path, plan, *, restart=None, start=False, **environment):
    """Run the plan's script as Slurm would on a node: fake srun, real Bash.

    ``start`` returns the running process instead of its result.
    """
    paths = plan.spec.paths
    for directory in (paths.run_dir, paths.output, paths.attempt(1).parent, paths.logs):
        directory.mkdir(parents=True, exist_ok=True)
    paths.submission_script.write_text(plan.script)
    bin_dir = tmp_path / "bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        for name, body in FAKE_SLURM.items():
            (bin_dir / name).write_text(body)
            (bin_dir / name).chmod(0o700)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "42",
        "SLURM_JOB_NODELIST": "host[0-1]",
        "SLURM_JWT": "scheduler-auth-secret",
        "CAPTURE": str(tmp_path / "argv"),
        "SRUN_LOG": str(tmp_path / "srun.log"),
        **environment,
    }
    env.pop("SLURM_RESTART_COUNT", None)
    if restart is not None:
        env["SLURM_RESTART_COUNT"] = str(restart)
    if start:
        return subprocess.Popen(["bash", str(paths.submission_script)], env=env)
    return subprocess.run(
        ["bash", str(paths.submission_script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def stub(tmp_path):
    trainer = tmp_path / "trainer venv's" / "python"
    trainer.parent.mkdir(exist_ok=True)
    trainer.write_text(TRAINER)
    trainer.chmod(0o700)
    return str(trainer)


def exit_record(plan, attempt):
    return parse_exit_record(plan.spec.paths.exit_record(attempt).read_bytes())


@posix
def test_the_script_claims_its_attempt_from_the_restart_count(tmp_path):
    plan = planner(requeue=True).plan(spec(tmp_path, python=stub(tmp_path)))
    paths = plan.spec.paths
    assert subprocess.run(["bash", "-n"], input=plan.script, text=True).returncode == 0

    first = batch(tmp_path, plan)
    assert first.returncode == 0, first.stderr
    assert exit_record(plan, 1) == ExitRecord(0, 0, 0)
    rows = read_tail(paths.metrics(1), start=0).rows
    assert [type(row) for row in rows] == [Metric] and rows[0].epoch == 1
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[argv.index("--output_dir") + 1] == str(paths.output)
    assert argv[argv.index("--metrics_jsonl") + 1] == str(paths.metrics(1))

    requeued = batch(tmp_path, plan, restart=1)
    assert requeued.returncode == 0, requeued.stderr
    assert exit_record(plan, 2) == ExitRecord(0, 0, 0)
    assert paths.metrics(2).is_file()
    assert sorted(path.name for path in paths.attempt(1).parent.iterdir()) == ["1", "2"]

    claimed = paths.exit_record(1).read_bytes()
    again = batch(tmp_path, plan, restart=0)
    assert again.returncode == 125
    assert paths.exit_record(1).read_bytes() == claimed


@posix
def test_a_restart_is_refused_when_the_plan_disables_requeue(tmp_path):
    plan = planner().plan(spec(tmp_path, python=stub(tmp_path)))
    result = batch(tmp_path, plan, restart=1)
    assert result.returncode == 125
    assert not plan.spec.paths.attempt(2).exists()
    assert not (tmp_path / "argv").exists()


@posix
@pytest.mark.parametrize(
    "training,stage,expected", [(0, 0, 0), (7, 0, 7), (0, 9, 9), (7, 9, 7)]
)
def test_exit_codes_are_recorded_independently_and_quoted_values_stay_inert(
    tmp_path, training, stage, expected
):
    special = "quoted ' $$ $? $(touch ENV_INJECTED)"
    request = spec(
        tmp_path,
        python=stub(tmp_path),
        data_path="value ' $(touch INJECTED) $HOME.npz",
    )
    request = replace(
        request,
        runtime=replace(
            request.runtime, environment={"SPECIAL": special, "N": "mine", "A": "mine"}
        ),
    )
    plan = planner(stage_out_command=f"exit {stage}").plan(request)
    result = batch(tmp_path, plan, TRAINING_EXIT=str(training))
    assert result.returncode == expected, result.stderr
    assert exit_record(plan, 1) == ExitRecord(0, training, stage)
    # The script's own variables never clobber the runtime environment.
    assert (tmp_path / "argv.env").read_text().splitlines() == [special, "mine", "mine"]
    assert "$(touch INJECTED)" in (tmp_path / "argv").read_text()
    assert not (plan.spec.paths.run_dir / "INJECTED").exists()
    assert not (plan.spec.paths.run_dir / "ENV_INJECTED").exists()


@posix
def test_setup_failure_never_launches_training(tmp_path):
    plan = planner(setup_commands=("false || (exit 6)", "touch SHOULD_NOT_RUN")).plan(
        spec(tmp_path, python=stub(tmp_path))
    )
    result = batch(tmp_path, plan)
    assert result.returncode == 6, result.stderr
    assert exit_record(plan, 1) == ExitRecord(6, 125, 0)
    assert not (tmp_path / "argv").exists()
    assert not (plan.spec.paths.run_dir / "SHOULD_NOT_RUN").exists()


@posix
def test_cuda_probe_failure_prevents_training(tmp_path):
    plan = planner(gpus_per_node=1).plan(
        spec(tmp_path, device="cuda", python=stub(tmp_path))
    )
    assert "torch.cuda.device_count() != 1" in plan.script
    result = batch(tmp_path, plan, PROBE_EXIT="23")
    assert result.returncode == 23, result.stderr
    assert exit_record(plan, 1) == ExitRecord(0, 23, 0)
    assert not (tmp_path / "argv").exists()


@posix
def test_multinode_ranks_expand_on_each_node_without_the_scheduler_token(tmp_path):
    plan = planner(gpus_per_node=2).plan(
        spec(tmp_path, device="cuda", python=stub(tmp_path)), topology=Topology(2, 2)
    )
    result = batch(tmp_path, plan, FAKE_NODES="2")
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "argv").read_text()
    assert "--machine_rank\n0\n" in argv and "--machine_rank\n1\n" in argv
    assert argv.count("--num_gpus\n4\n") == 2
    assert argv.count("--main_process_ip\nhost0\n") == 2
    assert argv.count(f"--metrics_jsonl\n{plan.spec.paths.metrics(1)}\n") == 2
    assert exit_record(plan, 1) == ExitRecord(0, 0, 0)


def wait_for(predicate, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("The batch script made no progress")
        time.sleep(0.02)


@posix
def test_a_cancel_sends_each_launcher_exactly_one_sigint(tmp_path):
    plan = planner(gpus_per_node=1).plan(
        spec(tmp_path, device="cuda", python=stub(tmp_path)), topology=Topology(2, 1)
    )
    assert "--kill-on-bad-exit=0 --wait=120" in plan.script
    paths, ints = plan.spec.paths, tmp_path / "ints"
    job = batch(
        tmp_path, plan, start=True, FAKE_NODES="2", FAKE_CONCURRENT="1", INTS=str(ints)
    )
    try:
        wait_for(lambda: ints.exists() and ints.read_text().count("ready") == 2)
        pids = {
            node: int(paths.attempt(1).joinpath(f"launcher.{node}").read_text())
            for node in (0, 1)
        }
        assert len(set(pids.values())) == 2
        # scancel --batch --signal=USR1, delivered twice as a retried cancel would.
        job.send_signal(signal.SIGUSR1)
        wait_for(lambda: "INT" in ints.read_text())
        job.send_signal(signal.SIGUSR1)
        assert job.wait(timeout=20) == 130
    finally:
        job.kill()
    assert sorted(
        line for line in ints.read_text().splitlines() if line.startswith("INT")
    ) == [
        "INT 0",
        "INT 1",
    ]
    assert exit_record(plan, 1) == ExitRecord(0, 130, 0)
    overlap = [
        line
        for line in (tmp_path / "srun.log").read_text().splitlines()
        if "--overlap" in line
    ]
    assert overlap == [
        "--overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c "
        'kill -INT "$(cat -- "$WAVEDL_ATTEMPT/launcher.$SLURM_NODEID")"'
    ]


@posix
def test_a_cancel_before_training_starts_never_launches_it(tmp_path):
    started = tmp_path / "setup started"
    plan = planner(
        setup_commands=(f"touch {shlex.quote(str(started))} && sleep 0.5",)
    ).plan(spec(tmp_path, python=stub(tmp_path)))
    job = batch(tmp_path, plan, start=True)
    try:
        wait_for(started.exists)
        job.send_signal(signal.SIGUSR1)
        assert job.wait(timeout=20) == 125
    finally:
        job.kill()
    assert exit_record(plan, 1) == ExitRecord(0, 125, 0)
    assert not (tmp_path / "argv").exists()
