"""Real Bash probes for pure Slurm rendering; never run a scheduler."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, StorageMapping, build_manifest
from wavedl_daemon.backends.contracts import Plan, Topology, UnsupportedPlan
from wavedl_daemon.backends.slurm_plan import (
    SlurmPlanner,
    SlurmSite,
    bind_submission,
    job_name,
)


def manifest(tmp_path, *, device="cpu", operation=OperationKind.TRAIN, **config):
    return build_manifest(
        RunLayout(tmp_path / "owned", str(uuid4())),
        operation=operation,
        resolved_config={"model": "cnn", "data_path": "data.npz"} | config,
        runtime=RuntimeSelection(
            str(tmp_path / "venv" / "bin" / "python"), "selected", device, {}
        ),
        submission_cwd=tmp_path,
    )


def planner(**kwargs):
    return SlurmPlanner(
        SlurmSite(partition="compute", cluster="test-cluster", **kwargs)
    )


def native_plan_refused(request, selected_planner):
    """Windows is not a Slurm execution host; successful rendering is POSIX-only."""
    if os.name != "nt":
        return False
    with pytest.raises(UnsupportedPlan, match="Slurm directive paths"):
        selected_planner.plan(request)
    assert not Path(request.run_directory).exists()
    return True


def test_plan_is_immutable_pure_and_preserves_lexical_interpreter(tmp_path):
    request = manifest(tmp_path)
    if native_plan_refused(request, planner()):
        return
    plan = planner().plan(request)
    assert not (tmp_path / "owned").exists()
    assert request.runtime.python in plan.script
    assert "-m wavedl.launcher" in plan.script
    assert "#SBATCH --output=" in plan.script
    assert "/logs/slurm-%j.out" in plan.script
    assert plan == Plan.from_json(plan.to_json())
    assert "wavedl_daemon" not in plan.script
    assert "--no-requeue" in plan.script
    assert "#SBATCH --export=NIL" in plan.script
    assert "export SLURM_EXPORT_ENV=ALL" in plan.script


@pytest.mark.skipif(os.name != "posix", reason="Real Bash environment probe")
@pytest.mark.parametrize("requeue", [False, True])
@pytest.mark.parametrize("multinode", [False, True])
def test_scheduler_authentication_jwt_never_reaches_setup_binding_or_rank(
    tmp_path, requeue, multinode
):
    # NIL still propagates SLURM_*; official JWT examples explicitly unset it
    # before srun: https://slurm.schedmd.com/jwt.html
    selected = tmp_path / "python-stub"
    selected.write_text(
        "#!/bin/bash\n"
        'if [ "${SLURM_JWT+x}" = x ]; then exit 91; fi\n'
        'printf "%s\\n" "$@" >> "$CAPTURE"\n'
        'if [ "$1" = "-c" ]; then printf "alice\\n"; fi\n'
        'if [ "$2" = "wavedl.launch_runtime" ]; then printf "1\\n"; fi\n'
        "exit 0\n"
    )
    selected.chmod(0o700)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "scontrol": "printf 'host0\\nhost1\\n'",
        "srun": 'test "${SLURM_JWT+x}" != x || exit 92; while [[ "$1" == --* ]]; do shift; done; SLURM_PROCID=0 "$@"',
    }.items():
        executable = bin_dir / name
        executable.write_text("#!/bin/bash\n" + body + "\n")
        executable.chmod(0o700)
    request = manifest(tmp_path, device="cuda" if multinode else "cpu")
    request = replace(request, runtime=replace(request.runtime, python=str(selected)))
    Path(request.working_directory).mkdir(mode=0o700)
    plan = planner(
        requeue=requeue,
        gpus_per_node=1 if multinode else 0,
        setup_commands=('test "${SLURM_JWT+x}" != x',),
    ).plan(request, topology=Topology(2, 1) if multinode else Topology())
    script = tmp_path / "submission.sh"
    script.write_text(bind_submission(plan, str(uuid4()), 1))
    result = subprocess.run(
        ["bash", str(script)],
        env=os.environ
        | {
            "SLURM_JWT": "scheduler-auth-secret-do-not-propagate",
            "SLURM_JOB_ID": "42",
            "SLURM_JOB_NODELIST": "host[0-1]",
            "PATH": str(bin_dir) + ":" + os.environ["PATH"],
            "CAPTURE": str(tmp_path / "calls"),
        },
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text()
    assert "wavedl.launcher" in calls
    assert "wavedl.core.outcomes" in calls
    assert ("scheduler-bind" in calls) == requeue
    assert "scheduler-auth-secret" not in calls


def test_full_uuid_identity_binding_does_not_rerender(tmp_path):
    request = manifest(tmp_path)
    if native_plan_refused(request, planner()):
        return
    plan = planner().plan(request)
    first = "11111111-1111-4111-8111-111111111111"
    second = "11111111-2222-4222-8222-222222222222"
    assert job_name(first) != job_name(second)
    bound = bind_submission(plan, first, 7)
    assert f"#SBATCH --job-name=wavedl-{first}" in bound
    assert f"#SBATCH --comment={first}" in bound
    assert "WAVEDL_ATTEMPT=7" in bound
    assert "@@WAVEDL_" in plan.script
    assert "@@WAVEDL_" not in bound
    assert "training_exit=$?" in bound
    assert "${SLURM_RESTART_COUNT:-0}" in bound
    assert plan.resources == Plan.from_json(plan.to_json()).resources


@pytest.mark.parametrize(
    "identity",
    ["prefix", "A" * 32, "11111111111141118111111111111111", "x\n#SBATCH --nodes=999"],
)
def test_noncanonical_uuid_refused(identity):
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
        SlurmPlanner(SlurmSite(partition="cpu")).plan(manifest(tmp_path))


def test_topology_exact_total_and_one_launcher_per_machine(tmp_path):
    if native_plan_refused(manifest(tmp_path), planner()):
        return
    plan = planner(gpus_per_node=4).plan(
        manifest(tmp_path, device="cuda"), topology=Topology(3, 4)
    )
    assert dict(plan.resources)["num_processes"] == 12
    assert "--num_gpus 12" in plan.script
    assert "--num_machines 3" in plan.script
    assert '--machine_rank "$SLURM_PROCID"' in plan.script
    assert "srun --nodes=3 --ntasks=3 --ntasks-per-node=1" in plan.script
    assert "#SBATCH --gpus-per-node=4" in plan.script
    assert "--main_process_port 29500" in plan.script
    assert "${WAVEDL_MAIN_HOST%%$" in plan.script


def test_cross_validation_distributed_shape_is_refused_at_preview(tmp_path):
    with pytest.raises(UnsupportedPlan, match="Cross-validation"):
        planner(gpus_per_node=2).plan(
            manifest(tmp_path, device="cuda", cv=2), topology=Topology(1, 2)
        )


@pytest.mark.parametrize("topology", [Topology(2), Topology(1, 2), Topology(2, 2, 1)])
def test_cpu_unexpressible_topology_refused(tmp_path, topology):
    with pytest.raises(UnsupportedPlan):
        planner().plan(manifest(tmp_path), topology=topology)


def test_unsupported_config_modes_and_gpu_settings_refused(tmp_path):
    request = manifest(tmp_path)
    for changed in (
        replace(request, resolved_config={"output_dir": request.working_directory}),
        replace(request, runtime=replace(request.runtime, mode="bundled")),
        replace(request, runtime=replace(request.runtime, device="auto")),
        replace(request, runtime=replace(request.runtime, device="cuda")),
        replace(
            request,
            runtime=replace(request.runtime, environment={"WAVEDL_ATTEMPT": "3"}),
        ),
    ):
        with pytest.raises(UnsupportedPlan):
            planner().plan(changed)
    with pytest.raises(UnsupportedPlan):
        planner(gpus_per_node=2).plan(manifest(tmp_path, device="cuda"))
    with pytest.raises(UnsupportedPlan):
        planner().plan(request, arguments=("--output_dir=/elsewhere",))


def test_mapped_execution_paths_remain_separate_from_host_paths(tmp_path):
    execution = (
        str(tmp_path / "compute/shared") if os.name == "nt" else "/compute/shared"
    )
    request = build_manifest(
        RunLayout(tmp_path / "owned", str(uuid4())),
        operation=OperationKind.TRAIN,
        resolved_config={"model": "cnn", "data_path": "data.npz"},
        runtime=RuntimeSelection("venv/bin/python", "selected", "cpu", {}),
        submission_cwd=tmp_path,
        storage_map=(StorageMapping(str(tmp_path), execution),),
    )
    if native_plan_refused(request, planner()):
        return
    plan = planner().plan(request)
    assert plan.run_dir == str(tmp_path / "owned")
    assert plan.working_directory == "/compute/shared/owned"
    assert "--data_path /compute/shared/data.npz" in plan.script
    assert "#SBATCH --output=/compute/shared/owned/logs/slurm-%j.out" in plan.script
    assert str(tmp_path) not in plan.script


@pytest.mark.skipif(os.name != "posix", reason="Bash execution probes are POSIX-only")
@pytest.mark.parametrize(
    "training,stage,expected", [(0, 0, 0), (7, 0, 7), (0, 9, 9), (7, 9, 7)]
)
def test_real_shell_preserves_independent_exits_and_quoted_values(
    tmp_path, training, stage, expected
):
    selected = tmp_path / "chosen venv's" / "python"
    selected.parent.mkdir()
    selected.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "wavedl.core.outcomes" ]; then\n'
        f'  exec {sys.executable} "$@"\nfi\n'
        'printf "%s\\n" "$@" > "$CAPTURE"\n'
        'printf "%s\\n" "$SPECIAL" > "$ENV_CAPTURE"\n'
        f"exit {training}\n"
    )
    selected.chmod(0o700)
    request = manifest(tmp_path, data_path="value ' $(touch INJECTED) $HOME.npz")
    special = "quoted ' $$ $? $(touch ENV_INJECTED)"
    request = replace(
        request,
        runtime=replace(
            request.runtime, python=str(selected), environment={"SPECIAL": special}
        ),
    )
    plan = planner(stage_out_command=f"exit {stage}").plan(request)
    attempt = tmp_path / "owned" / "attempts" / "1"
    attempt.mkdir(parents=True)
    script = tmp_path / "submission.sh"
    script.write_text(bind_submission(plan, str(uuid4()), 1))
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    environment = os.environ | {
        "SLURM_JOB_ID": "42",
        "SLURM_CLUSTER_NAME": "test-cluster",
        "CAPTURE": str(tmp_path / "argv"),
        "ENV_CAPTURE": str(tmp_path / "env"),
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
    }
    result = subprocess.run(
        ["bash", str(script)], env=environment, capture_output=True, check=False
    )
    assert result.returncode == expected, result.stderr
    receipt = json.loads((attempt / "slurm-outcome.json").read_text())
    assert receipt["runtime"]["exit_code"] == training
    assert receipt["stage_out"]["exit_code"] == stage
    assert receipt["batch"]["exit_code"] == 0
    assert (tmp_path / "env").read_text().strip() == special
    assert "$(touch INJECTED)" in (tmp_path / "argv").read_text()
    assert not (tmp_path / "owned" / "INJECTED").exists()
    assert not (tmp_path / "owned" / "ENV_INJECTED").exists()


@pytest.mark.skipif(os.name != "posix", reason="Bash execution probes are POSIX-only")
def test_multinode_bash_expands_rank_only_on_each_node(tmp_path):
    request = manifest(tmp_path, device="cuda")
    selected = tmp_path / "python-stub"
    selected.write_text(
        '#!/bin/bash\nif [ "$2" = "wavedl.core.outcomes" ]; then exit 0; fi\nprintf "%s\\n" "$@" >> "$CAPTURE"\n'
    )
    selected.chmod(0o700)
    request = replace(request, runtime=replace(request.runtime, python=str(selected)))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    control = bin_dir / "scontrol"
    control.write_text("#!/bin/bash\nprintf 'host0\\nhost1\\n'\n")
    control.chmod(0o700)
    srun = bin_dir / "srun"
    srun.write_text(
        '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nfor rank in 0 1; do SLURM_PROCID="$rank" "$@" || exit "$?"; done\n'
    )
    srun.chmod(0o700)
    Path(request.working_directory).mkdir()
    script = tmp_path / "submission.sh"
    script.write_text(
        bind_submission(
            planner(gpus_per_node=2).plan(request, topology=Topology(2, 2)),
            str(uuid4()),
            1,
        )
    )
    result = subprocess.run(
        ["bash", str(script)],
        env=os.environ
        | {
            "PATH": str(bin_dir) + ":" + os.environ["PATH"],
            "SLURM_JOB_ID": "42",
            "SLURM_JOB_NODELIST": "host[0-1]",
            "CAPTURE": str(tmp_path / "argv"),
        },
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "argv").read_text()
    assert "--machine_rank\n0\n" in argv
    assert "--machine_rank\n1\n" in argv
    assert argv.count("--num_gpus\n4\n") == 2
    assert argv.count("--main_process_ip\nhost0\n") == 2
    assert argv.count("Selected CUDA runtime is unavailable") == 2
    assert argv.count("torch.cuda.synchronize(index)") == 2


@pytest.mark.skipif(os.name != "posix", reason="Bash execution probes are POSIX-only")
def test_cuda_probe_failure_prevents_launcher_cpu_fallback(tmp_path):
    request = manifest(tmp_path, device="cuda")
    selected = tmp_path / "python-stub"
    selected.write_text(
        '#!/bin/bash\nif [ "$1" = "-c" ]; then exit 23; fi\n'
        'if [ "$2" = "wavedl.core.outcomes" ]; then\n'
        f'  exec {sys.executable} "$@"\nfi\n'
        'touch "$CAPTURE"\nexit 0\n'
    )
    selected.chmod(0o700)
    request = replace(request, runtime=replace(request.runtime, python=str(selected)))
    attempt = tmp_path / "owned" / "attempts" / "1"
    attempt.mkdir(parents=True)
    plan = planner(gpus_per_node=1).plan(request)
    assert "torch.cuda.device_count() != 1" in plan.script
    assert "torch.empty(1," in plan.script
    assert "assert torch" not in plan.script
    script = tmp_path / "submission.sh"
    script.write_text(bind_submission(plan, str(uuid4()), 1))
    result = subprocess.run(
        ["bash", str(script)],
        env=os.environ
        | {
            "SLURM_JOB_ID": "42",
            "CAPTURE": str(tmp_path / "capture"),
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
        capture_output=True,
        check=False,
    )
    assert result.returncode == 23, result.stderr
    assert not (tmp_path / "capture").exists()
    receipt = json.loads((attempt / "slurm-outcome.json").read_text())
    assert receipt["runtime"]["status"] == "failed"
    assert receipt["runtime"]["exit_code"] == 23


@pytest.mark.skipif(os.name != "posix", reason="Bash execution probes are POSIX-only")
@pytest.mark.parametrize("restart,setup_exit", [(0, 6), (1, 0)])
def test_setup_failure_and_scheduler_restart_never_launch_training(
    tmp_path, restart, setup_exit
):
    request = manifest(tmp_path)
    selected = tmp_path / "python-stub"
    selected.write_text(
        '#!/bin/bash\nif [ "$2" = "wavedl.core.outcomes" ]; then\n'
        f'  exec {sys.executable} "$@"\nfi\n'
        'touch "$CAPTURE"\nexit 0\n'
    )
    selected.chmod(0o700)
    request = replace(request, runtime=replace(request.runtime, python=str(selected)))
    attempt = tmp_path / "owned" / "attempts" / "1"
    attempt.mkdir(parents=True)
    original = attempt / "slurm-outcome.json"
    if restart:
        original.write_text("prior execution receipt")
    script = tmp_path / "submission.sh"
    script.write_text(
        bind_submission(
            planner(
                setup_commands=(f"false || (exit {setup_exit})", "touch SHOULD_NOT_RUN")
            ).plan(request),
            str(uuid4()),
            1,
        )
    )
    result = subprocess.run(
        ["bash", str(script)],
        env=os.environ
        | {
            "SLURM_JOB_ID": "42",
            "SLURM_RESTART_COUNT": str(restart),
            "CAPTURE": str(tmp_path / "capture"),
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
        capture_output=True,
        check=False,
    )
    assert result.returncode == (125 if restart else setup_exit), result.stderr
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / "owned" / "SHOULD_NOT_RUN").exists()
    path = attempt / (
        f"slurm-outcome-restart-{restart}.json" if restart else "slurm-outcome.json"
    )
    receipt = json.loads(path.read_text())
    assert receipt["runtime"]["status"] == "unknown"
    assert receipt["restart_count"] == restart
    if restart:
        assert original.read_text() == "prior execution receipt"
        assert receipt["stage_out"]["status"] == "unknown"


def test_unsafe_directive_paths_and_identity_markers_are_refused(tmp_path):
    for suffix in ("percent%", "quote'", 'quote"', "slash\\", "newline\n"):
        with pytest.raises(UnsupportedPlan):
            planner().plan(manifest(tmp_path / suffix))
    if native_plan_refused(manifest(tmp_path), planner()):
        return
    with pytest.raises(UnsupportedPlan, match="marker"):
        planner().plan(manifest(tmp_path, run_name="@@WAVEDL_ATTEMPT@@"))
