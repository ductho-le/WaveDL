"""
Pure Slurm planning: site validation and the frozen batch script.

The script claims its own attempt: attempt N is ``1 + SLURM_RESTART_COUNT`` and
``mkdir`` makes the claim exclusive.

A cancel sends SIGINT exactly once to each node's launcher, never step-wide,
because a second copy would abort the emergency checkpoint. The daemon signals
the batch shell alone (``scancel --batch``, USR1) and each launcher by the PID
its task wrote to the attempt directory.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from string import Template
from uuid import UUID

from wavedl_daemon.runs.layout import RESULT_OPTIONS
from wavedl_daemon.runs.model import Operation

from .contracts import Plan, RunSpec, Topology, UnsupportedPlan
from .local import launch_argv
from .slurm_exit import render_exit_record


_TOKEN = re.compile(r"[A-Za-z0-9_.:+,=/\-]+\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_RESERVED_ENV = (
    "WAVEDL_",
    "SLURM_",
    "SRUN_",
    "SBATCH_",
    "SCANCEL_",
    "SQUEUE_",
    "SACCT_",
    "SCONTROL_",
)


def job_name(run_id: str) -> str:
    """The scheduler name of a run; accepts canonical UUIDs only."""
    try:
        canonical = str(UUID(run_id))
    except (TypeError, ValueError, AttributeError):
        canonical = None
    if canonical != run_id:
        raise ValueError("run_id must be a canonical UUID")
    return f"wavedl-{canonical}"


def _text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(char in value for char in ("\0", "\r", "\n"))
    ):
        raise UnsupportedPlan(f"Invalid {name}")


@dataclass(frozen=True)
class SlurmSite:
    """Explicit site resources and trusted administrator shell hooks.

    Hooks are shell programs, not user-provided training arguments. Container
    and bundled runtimes are deliberately refused until their contracts exist.
    Memory is per node; CPUs are per launcher task (one task per machine).
    """

    partition: str
    account: str | None = None
    time: str = "01:00:00"
    cpus_per_task: int = 1
    memory: str = "4G"
    gpus_per_node: int = 0
    constraint: str | None = None
    cluster: str | None = None
    rendezvous_port: int = 29500
    setup_commands: tuple[str, ...] = ()
    stage_out_command: str | None = None
    requeue: bool = False

    def __post_init__(self):
        if type(self.requeue) is not bool:
            raise UnsupportedPlan("requeue must be explicitly boolean")
        for name in ("partition", "account", "time", "memory", "constraint", "cluster"):
            value = getattr(self, name)
            if value is None and name in {"account", "constraint", "cluster"}:
                continue
            _text(value, name)
            if not _TOKEN.fullmatch(value):
                raise UnsupportedPlan(f"Unsupported scheduler token: {name}")
        if not re.fullmatch(r"(?:\d+-)?\d+:[0-5]\d:[0-5]\d", self.time):
            raise UnsupportedPlan("time must be [days-]hours:minutes:seconds")
        if not re.fullmatch(r"[1-9]\d*[KMGT]?", self.memory):
            raise UnsupportedPlan("memory must be a positive Slurm memory quantity")
        if type(self.cpus_per_task) is not int or self.cpus_per_task < 1:
            raise UnsupportedPlan("cpus_per_task must be positive")
        if type(self.gpus_per_node) is not int or self.gpus_per_node < 0:
            raise UnsupportedPlan("gpus_per_node must be nonnegative")
        if (
            type(self.rendezvous_port) is not int
            or not 1 <= self.rendezvous_port <= 65535
        ):
            raise UnsupportedPlan("Invalid rendezvous port")
        if not isinstance(self.setup_commands, tuple):
            raise UnsupportedPlan("setup_commands must be an immutable tuple")
        for command in self.setup_commands:
            _text(command, "trusted setup command")
        if self.stage_out_command is not None:
            _text(self.stage_out_command, "trusted stage-out command")


# srun --wait: how long the other launchers may outlive the first one to exit,
# while rank 0 finishes its emergency save (its launcher allows 30 s) or its
# final outputs. A larger bound only delays ending a crashed run.
_STEP_WAIT_SECONDS = 120
# Bounds the srun that delivers the interrupt, well inside the daemon's 45 s
# cancel grace.
_INTERRUPT_TIMEOUT_SECONDS = 20

# Runtime shell dollars are escaped for string.Template; inserted quoted values
# are not recursively interpreted by Template. $? remains the real shell status.
# WAVEDL_INTERRUPT is 0, 1 once a cancel arrives, and 2 once it is delivered.
# A trap ends wait early, so wait repeats until no trap ran; Bash returns an
# already reaped job's status again.
_BATCH = Template("""#!/bin/bash
$directives
set -uo pipefail
WAVEDL_INTERRUPT=0
WAVEDL_WOKEN=0
WAVEDL_STEP=
wavedl_interrupt() {
    WAVEDL_WOKEN=1
    if [ "$$WAVEDL_INTERRUPT" -eq 2 ]; then return 0; fi
    WAVEDL_INTERRUPT=1
    if [ -z "$$WAVEDL_STEP" ]; then return 0; fi
    WAVEDL_INTERRUPT=2
    $interrupt
}
trap wavedl_interrupt USR1
# NIL still exports SLURM_*; JWT is for submission authentication, not ranks.
# https://slurm.schedmd.com/sbatch.html https://slurm.schedmd.com/jwt.html
unset SLURM_JWT
export SLURM_EXPORT_ENV=ALL
$environment
WAVEDL_ATTEMPT_NUMBER=$$((1 + $${SLURM_RESTART_COUNT:-0}))
$restart_guard
export WAVEDL_ATTEMPT=$attempts/"$$WAVEDL_ATTEMPT_NUMBER"
mkdir -- "$$WAVEDL_ATTEMPT" || exit 125
WAVEDL_BATCH_EXIT=0
WAVEDL_TRAINING_EXIT=125
WAVEDL_STAGE_OUT_EXIT=125
cd -- $working_directory || WAVEDL_BATCH_EXIT=$$?
if [ "$$WAVEDL_BATCH_EXIT" -eq 0 ]; then
    $setup
    WAVEDL_BATCH_EXIT=$$?
fi
if [ "$$WAVEDL_BATCH_EXIT" -eq 0 ]; then
    $rendezvous
    WAVEDL_BATCH_EXIT=$$?
fi
if [ "$$WAVEDL_BATCH_EXIT" -eq 0 ] && [ "$$WAVEDL_INTERRUPT" -eq 0 ]; then
    $launch &
    WAVEDL_STEP=$$!
    if [ "$$WAVEDL_INTERRUPT" -eq 1 ]; then wavedl_interrupt; fi
    WAVEDL_WOKEN=1
    while [ "$$WAVEDL_WOKEN" -eq 1 ]; do
        WAVEDL_WOKEN=0
        wait "$$WAVEDL_STEP"
        WAVEDL_TRAINING_EXIT=$$?
    done
fi
$stage_out
WAVEDL_STAGE_OUT_EXIT=$$?
$exit_record
if [ "$$WAVEDL_BATCH_EXIT" -ne 0 ]; then exit "$$WAVEDL_BATCH_EXIT"; fi
if [ "$$WAVEDL_TRAINING_EXIT" -ne 0 ]; then exit "$$WAVEDL_TRAINING_EXIT"; fi
exit "$$WAVEDL_STAGE_OUT_EXIT"
""")


def _cuda_probe(python: str, processes: int) -> str:
    """Probe each task's own Slurm-owned devices with the selected interpreter.

    Import and count alone cannot prove the allocated devices accept work.
    """
    code = (
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError('Selected CUDA runtime is unavailable')\n"
        f"if torch.cuda.device_count() != {processes}:\n"
        "    raise RuntimeError('CUDA visibility differs from planned allocation')\n"
        f"for index in range({processes}):\n"
        "    tensor = torch.empty(1, device=f'cuda:{index}')\n"
        "    torch.cuda.synchronize(index)\n"
        "    del tensor\n"
    )
    return shlex.join([python, "-c", code]) + " && "


class SlurmPlanner:
    """Render a single allocation with exactly one launcher task per machine."""

    name = "slurm"

    def __init__(self, site: SlurmSite):
        if not isinstance(site, SlurmSite):
            raise UnsupportedPlan("Explicit SlurmSite configuration is required")
        self.site = site

    def plan(self, spec: RunSpec, *, topology: Topology = Topology()) -> Plan:
        if not isinstance(spec, RunSpec) or not isinstance(topology, Topology):
            raise UnsupportedPlan("A resolved RunSpec and Topology are required")
        site = self.site
        if not site.cluster or not re.fullmatch(r"[A-Za-z0-9_.-]+", site.cluster):
            raise UnsupportedPlan("An explicit single Slurm cluster is required")
        try:
            name = job_name(spec.run_id)
        except ValueError as error:
            raise UnsupportedPlan(str(error)) from None
        if topology.machine_rank != 0:
            raise UnsupportedPlan("The allocation plan must start at machine_rank 0")
        training = spec.operation is Operation.TRAIN
        if training and spec.resolved_config.get("cv", 0) and topology != Topology():
            raise UnsupportedPlan(
                "Cross-validation requires one machine and one process"
            )
        runtime = spec.runtime
        if (
            runtime.mode not in {"selected", "existing"}
            or runtime.container is not None
        ):
            raise UnsupportedPlan("Slurm does not implement bundled/container runtimes")
        if training and runtime.device not in {"cpu", "cuda"}:
            raise UnsupportedPlan(
                "Slurm training requires explicit CPU or CUDA selection"
            )
        if training and any(
            not spec.resolved_config.get(name) for name in ("model", "data_path")
        ):
            raise UnsupportedPlan("Training requires model and data_path")
        if runtime.device == "cpu" and topology != Topology():
            raise UnsupportedPlan("CPU launcher supports one machine and one process")
        if not training and topology != Topology():
            raise UnsupportedPlan("Only training supports distributed topology")
        if runtime.device == "cuda":
            if site.gpus_per_node != topology.processes_per_machine:
                raise UnsupportedPlan("GPU allocation must equal processes_per_machine")
            if "CUDA_VISIBLE_DEVICES" in runtime.environment:
                raise UnsupportedPlan(
                    "CUDA visibility must be owned by Slurm allocation"
                )
        elif site.gpus_per_node != 0:
            raise UnsupportedPlan("GPU allocation requires explicit CUDA training")
        working = spec.working_directory
        for path in (working, runtime.python):
            _text(path, "execution path")
            if not PurePosixPath(path).is_absolute():
                raise UnsupportedPlan(
                    "Slurm execution paths must be absolute POSIX paths"
                )
        if "%" in working:
            raise UnsupportedPlan(
                "Slurm filename patterns cannot express percent in run paths"
            )
        if any(char in working for char in ("'", '"', "\\")):
            raise UnsupportedPlan(
                "Slurm directive paths cannot express shell quote concatenation"
            )
        for key, value in runtime.environment.items():
            if not _ENV.fullmatch(key) or key.startswith(_RESERVED_ENV):
                raise UnsupportedPlan(
                    "Environment cannot override runtime/scheduler identity"
                )
            if any(char in value for char in ("\0", "\r", "\n")):
                raise UnsupportedPlan("Invalid environment value")
        execution = PurePosixPath(working)
        return Plan(
            target=self.name,
            spec=spec,
            script=_BATCH.substitute(
                directives="\n".join(self._directives(name, spec, topology)),
                environment="\n".join(
                    f"export {key}={shlex.quote(value)}"
                    for key, value in sorted(runtime.environment.items())
                ),
                restart_guard=""
                if site.requeue
                else '[ "$WAVEDL_ATTEMPT_NUMBER" -eq 1 ] || exit 125',
                attempts=shlex.quote(str(execution / "attempts")),
                working_directory=shlex.quote(working),
                setup=" && ".join(
                    f"{{ {command}; }}" for command in site.setup_commands
                )
                or ":",
                rendezvous=(
                    'WAVEDL_MAIN_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST") && '
                    "WAVEDL_MAIN_HOST=${WAVEDL_MAIN_HOST%%$'\\n'*} && "
                    'test -n "$WAVEDL_MAIN_HOST" && export WAVEDL_MAIN_HOST'
                    if topology.num_machines > 1
                    else ":"
                ),
                launch=self._launch(spec, topology),
                interrupt=self._interrupt(topology),
                stage_out=f"( {site.stage_out_command} )"
                if site.stage_out_command
                else ":",
                exit_record=render_exit_record("WAVEDL_ATTEMPT").rstrip("\n"),
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
            environment=tuple(sorted(runtime.environment.items())),
            warnings=(
                "Submission creates the run, log and attempt directories on shared storage.",
                "A Slurm requeue starts a new attempt that resumes from the run's output."
                if site.requeue
                else "Scheduler requeue is disabled; a restarted job refuses to run.",
                "Site shell setup and stage-out hooks are trusted administrator configuration.",
                "Site must support --export=NIL; runtime environment and setup hooks define execution environment explicitly.",
            ),
        )

    def _directives(self, name: str, spec: RunSpec, topology: Topology) -> list[str]:
        site = self.site
        logs = PurePosixPath(spec.working_directory) / "logs"
        directives = [
            f"#SBATCH --job-name={name}",
            f"#SBATCH --comment={spec.run_id}",
            f"#SBATCH --partition={site.partition}",
            f"#SBATCH --nodes={topology.num_machines}",
            f"#SBATCH --ntasks={topology.num_machines}",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={site.cpus_per_task}",
            f"#SBATCH --mem={site.memory}",
            f"#SBATCH --time={site.time}",
            "#SBATCH --requeue" if site.requeue else "#SBATCH --no-requeue",
            "#SBATCH --export=NIL",
            f"#SBATCH --chdir={shlex.quote(spec.working_directory)}",
            f"#SBATCH --output={shlex.quote(str(logs / 'slurm-%j.out'))}",
            f"#SBATCH --error={shlex.quote(str(logs / 'slurm-%j.err'))}",
        ]
        for field, option in (
            ("account", "account"),
            ("constraint", "constraint"),
            ("cluster", "clusters"),
        ):
            value = getattr(site, field)
            if value is not None:
                directives.append(f"#SBATCH --{option}={value}")
        if site.gpus_per_node:
            directives.append(f"#SBATCH --gpus-per-node={site.gpus_per_node}")
        return directives

    def _launch(self, spec: RunSpec, topology: Topology) -> str:
        """``srun`` of the probe and the trainer, one launcher task per machine."""
        # Paths in the spec are execution-side; render them as the run sees them.
        command = launch_argv(spec, topology)
        machines = topology.num_machines
        if machines > 1:
            rank = command.index("--machine_rank") + 1
            launch = (
                f'{shlex.join(command[:rank])} "$SLURM_PROCID" '
                f"{shlex.join(command[rank + 1 :])} "
                f'--main_process_ip "$WAVEDL_MAIN_HOST" '
                f"--main_process_port {self.site.rendezvous_port}"
            )
        else:
            launch = shlex.join(command)
        if spec.operation in RESULT_OPTIONS:
            option, file = RESULT_OPTIONS[spec.operation]
            launch += f' {option} "$WAVEDL_ATTEMPT/{file}"'
        probe = (
            _cuda_probe(spec.runtime.python, topology.processes_per_machine)
            if spec.runtime.device == "cuda"
            else ""
        )
        # Rank 0 may still be saving when another node's launcher exits 130.
        return (
            f"srun --nodes={machines} --ntasks={machines} --ntasks-per-node=1 "
            f"--cpus-per-task={self.site.cpus_per_task} --kill-on-bad-exit=0 "
            f"--wait={_STEP_WAIT_SECONDS} "
            "bash -c "
            + shlex.quote(
                probe
                + 'printf \'%s\\n\' "$$" > "$WAVEDL_ATTEMPT/launcher.$SLURM_NODEID" && exec '
                + launch
            )
        )

    @staticmethod
    def _interrupt(topology: Topology) -> str:
        """SIGINT each node's launcher once, from a step beside training."""
        machines = topology.num_machines
        kill = 'kill -INT "$(cat -- "$WAVEDL_ATTEMPT/launcher.$SLURM_NODEID")"'
        return (
            f"timeout {_INTERRUPT_TIMEOUT_SECONDS} srun --overlap --nodes={machines} "
            f"--ntasks={machines} --ntasks-per-node=1 bash -c {shlex.quote(kill)}"
        )
