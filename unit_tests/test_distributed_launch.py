"""Callable distributed launch routing and native POSIX rank containment."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from wavedl import distributed_launch, launcher
from wavedl_daemon.supervision.base import ProcessSpec
from wavedl_daemon.supervision.posix import POSIXSupervisor


@pytest.mark.parametrize("cpu,count", [(True, 2), (False, 2), (False, 1)])
def test_launcher_selects_contained_multi_rank_routes(monkeypatch, cpu, count):
    # This fake child tests command routing, not native PID/PGID ownership.
    monkeypatch.setenv("WAVEDL_SUPERVISED", "1")
    args = SimpleNamespace(
        cpu=cpu,
        num_gpus=count,
        num_machines=2,
        machine_rank=1,
        mixed_precision="bf16",
        dynamo_backend="inductor",
        main_process_ip="10.0.0.1",
        main_process_port=29999,
    )
    commands = []
    monkeypatch.setattr(launcher, "handle_fast_path_args", lambda: None)
    monkeypatch.setattr(launcher, "parse_args", lambda: (args, ["--epochs", "3"]))
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(launcher, "print_summary", lambda *a: None)
    import wavedl.launch_runtime

    monkeypatch.setattr(
        wavedl.launch_runtime,
        "prepare_runtime",
        lambda train, *a, **kw: (train, {"CALLER_SETTING": "preserved"}),
    )

    def spawn(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return SimpleNamespace(pid=123, wait=lambda **kw: 0)

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    assert launcher.main() == 0
    cmd, options = commands[0]
    assert options["env"]["CALLER_SETTING"] == "preserved"
    assert cmd[-2:] == ["--epochs", "3"]
    if cpu:
        assert cmd[2] == "torch.distributed.run"
        assert "--run-path" in cmd and "--start-method=spawn" in cmd
        assert str(Path(launcher.__file__).with_name("train.py").resolve()) in cmd
        assert options["env"]["ACCELERATE_USE_CPU"] == "true"
    else:
        expected = (
            "wavedl.distributed_launch" if count > 1 else "accelerate.commands.launch"
        )
        assert cmd[2] == expected
        assert "--num_machines=2" in cmd and "--machine_rank=1" in cmd
        assert "--mixed_precision=bf16" in cmd and "--dynamo_backend=inductor" in cmd
        assert "--main_process_ip=10.0.0.1" in cmd
        assert "--main_process_port=29999" in cmd
        assert cmd[-4:-2] == ["-m", "wavedl.train"]


def test_actual_accelerate_configuration_retains_callable_route(monkeypatch, tmp_path):
    """Use installed parser/filter/config, without pretending a GPU launch ran."""
    launch = pytest.importorskip("accelerate.commands.launch")
    run = pytest.importorskip("torch.distributed.run")
    config = tmp_path / "accelerate.yaml"
    config.write_text(
        "compute_environment: LOCAL_MACHINE\n"
        "distributed_type: MULTI_GPU\n"
        "mixed_precision: 'no'\n"
        "num_processes: 2\n"
        "num_machines: 1\n"
        "machine_rank: 0\n"
        "gpu_ids: '0,1'\n"
        "use_cpu: false\n"
        "dynamo_config:\n"
        "  dynamo_backend: INDUCTOR\n"
    )
    captured = []
    monkeypatch.setattr(launch, "check_cuda_p2p_ib_support", lambda: True)

    def inspect_run(args):
        configuration, entrypoint, worker_args = run.config_from_args(args)
        captured.append((configuration, entrypoint, worker_args, dict(os.environ)))

    monkeypatch.setattr(run, "run", inspect_run)
    distributed_launch.main(
        [
            "--config_file",
            str(config),
            "--multi_gpu",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
            "--main_process_port=0",
            "--mixed_precision=no",
            "--dynamo_backend=inductor",
            "-m",
            "wavedl.train",
            "--epochs",
            "3",
        ]
    )
    configuration, entrypoint, worker_args, environment = captured[0]
    assert entrypoint is run.run_script_path
    assert configuration.start_method == "spawn"
    assert configuration.nproc_per_node == 2
    assert worker_args == [
        str(Path(distributed_launch.__file__).with_name("train.py").resolve()),
        "--epochs",
        "3",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert environment["ACCELERATE_MIXED_PRECISION"] == "no"
    assert environment["ACCELERATE_DYNAMO_BACKEND"] == "INDUCTOR"


@pytest.mark.parametrize(
    "args",
    [
        ["-m", "wavedl.train"],
        ["--multi_gpu", "-m", "another.module"],
        ["--multi_gpu", "--no_python", "-m", "wavedl.train"],
        ["--multi_gpu", "--use_deepspeed", "-m", "wavedl.train"],
    ],
)
def test_bridge_rejects_uncontained_routes(args):
    pytest.importorskip("accelerate")
    with pytest.raises(ValueError, match="contained distributed launch"):
        distributed_launch.main(args)


@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_native_two_rank_callable_workers_cancel_inside_owner_group(tmp_path):
    pytest.importorskip("torch.distributed.run")
    worker = tmp_path / "marker_worker.py"
    worker.write_text(
        "import json, os, signal, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "rank = os.environ['LOCAL_RANK']\n"
        "Path(sys.argv[1], rank + '.json').write_text(json.dumps(\n"
        "    {'pid': os.getpid(), 'pgid': os.getpgrp(), 'sid': os.getsid(0)}))\n"
        "print('rank ' + rank + ' ready', flush=True)\n"
        "while True: time.sleep(.05)\n"
    )
    process = POSIXSupervisor().start(
        ProcessSpec(
            (
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "--local_addr=127.0.0.1",
                "--start-method=spawn",
                "--run-path",
                str(worker),
                str(tmp_path),
            ),
            tmp_path,
            {"OMP_NUM_THREADS": "1"},
        )
    )
    markers = []
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            files = [tmp_path / "0.json", tmp_path / "1.json"]
            if all(file.exists() for file in files):
                markers = [json.loads(file.read_text()) for file in files]
                break
            time.sleep(0.02)
        assert len(markers) == 2, "both native ranks must reach their marker"
        assert {item["pgid"] for item in markers} == {process.pid}
        assert {item["sid"] for item in markers} == {process.pid}
        started = time.monotonic()
        process.cancel(grace_seconds=0.2)
        result = process.wait(10)
        assert time.monotonic() - started < 10
        assert result.cancelled and not result.timed_out
        assert b"rank 0 ready" in result.stdout and b"rank 1 ready" in result.stdout
        assert process.wait() is result
        for marker in markers:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(marker["pid"])],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            assert not state or state.startswith("Z"), f"live rank remains: {marker}"
    finally:
        # Only this supervisor's reserved process-group identity is cleaned up.
        process.cancel(grace_seconds=0)
        process.wait(10)
