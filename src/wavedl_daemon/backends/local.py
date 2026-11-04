"""
Local backend: pure planning and bounded ownership of live processes.
Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

from pathlib import Path

from wavedl_daemon.runs import training_options
from wavedl_daemon.runs.model import Operation

from .contracts import Plan, RunSpec, Topology, UnsupportedPlan
from .local_execution import LocalExecution, reserved_local_environment


def launch_argv(spec: RunSpec, topology: Topology) -> list[str]:
    """The trainer command for ``spec``, with execution-side paths."""
    config = spec.resolved_config
    if spec.operation in {Operation.TEST, Operation.PLOT} and any(
        not config.get(name) for name in ("checkpoint", "data_path")
    ):
        raise UnsupportedPlan("Test and plot require checkpoint and data_path")
    try:
        options = training_options.cli_args(spec.operation, config)
    except ValueError as error:
        raise UnsupportedPlan(str(error)) from error
    runtime = spec.runtime
    argv = [runtime.python, "-u", "-m"]
    if spec.operation == Operation.TRAIN:
        # Explicit topology prevents launcher hardware auto-detection from
        # silently multiplying the request. --num_gpus is accelerate's total
        # process count across machines.
        argv += [
            "wavedl.launcher",
            "--num_machines",
            str(topology.num_machines),
            "--machine_rank",
            str(topology.machine_rank),
            "--num_gpus",
            str(topology.num_machines * topology.processes_per_machine),
        ]
        if runtime.device == "cpu":
            argv.append("--cpu")
    else:
        argv += ["wavedl.test"]
    argv += options
    if spec.operation == Operation.PLOT and "--plot" not in argv:
        argv.append("--plot")
    argv += ["--output_dir", spec.execution_path(spec.paths.output)]
    return argv


class LocalBackend(LocalExecution):
    """Plan without side effects; submit only under live dispatch ownership."""

    name = "local"

    def plan(self, spec: RunSpec, *, topology: Topology = Topology()) -> Plan:
        if not isinstance(spec, RunSpec) or not isinstance(topology, Topology):
            raise UnsupportedPlan("A resolved RunSpec and Topology are required")
        if topology.num_machines != 1 or topology.machine_rank != 0:
            raise UnsupportedPlan("Local supports one machine only")
        if spec.operation == Operation.TRAIN:
            folds = spec.resolved_config.get("cv", 0)
            if type(folds) is not int or folds < 0:
                raise UnsupportedPlan(
                    "Cross-validation fold count must be a nonnegative integer"
                )
            if folds and topology.processes_per_machine != 1:
                raise UnsupportedPlan("Cross-validation requires one training process")
            try:
                training_options.check_cv(spec.resolved_config)
            except ValueError as error:
                raise UnsupportedPlan(str(error)) from error
        runtime = spec.runtime
        if any(reserved_local_environment(key) for key in runtime.environment):
            raise UnsupportedPlan(
                "Local environment cannot override execution identity"
            )
        if (
            runtime.mode not in {"existing", "selected"}
            or runtime.container is not None
        ):
            raise UnsupportedPlan("Local does not implement bundled/container runtimes")
        if spec.operation == Operation.TRAIN and runtime.device == "mps":
            raise UnsupportedPlan("Explicit MPS training selection is unsupported")
        if any(
            Path(item.host_root) != Path(item.execution_root)
            for item in spec.storage_map
        ):
            raise UnsupportedPlan("Local cannot execute remapped storage paths")
        if spec.run_directory != spec.working_directory:
            raise UnsupportedPlan(
                "Local working directory must be the owned run directory"
            )
        if spec.operation != Operation.TRAIN and topology.processes_per_machine != 1:
            raise UnsupportedPlan("Only training supports multiprocess Local plans")
        if topology.processes_per_machine > 1 and runtime.device not in {"cpu", "cuda"}:
            raise UnsupportedPlan("Multiprocess Local training requires CPU or CUDA")
        if runtime.device == "cpu" and topology.processes_per_machine > 64:
            raise UnsupportedPlan("Local CPU training supports at most 64 processes")
        if runtime.device == "cuda":
            visible = [
                value.strip()
                for value in runtime.environment.get("CUDA_VISIBLE_DEVICES", "").split(
                    ","
                )
            ]
            if (
                runtime.device != "cuda"
                or any(not value or value == "-1" for value in visible)
                or len(set(visible)) < topology.processes_per_machine
                or len(visible) != len(set(visible))
            ):
                raise UnsupportedPlan(
                    "CUDA Local training requires explicit distinct visible CUDA devices"
                )
        return Plan(
            target=self.name,
            spec=spec,
            argv=tuple(launch_argv(spec, topology)),
            topology=topology,
            environment=tuple(sorted(runtime.environment.items())),
            warnings=(
                "Preview only: submission verifies the selected runtime/device before execution.",
                *(
                    (
                        "Submission must verify the selected CUDA allocation; CPU fallback is not permitted.",
                    )
                    if runtime.device == "cuda"
                    else ()
                ),
            ),
        )
