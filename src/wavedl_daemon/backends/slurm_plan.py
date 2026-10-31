# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Pure, identity-bindable Slurm plans; no scheduler or filesystem side effects.

The preview is immutable. Submission may substitute only the three declared
identity slots, never re-render resources, runtime choices or user arguments.
Scheduler directives are literal Slurm tokens, not shell expansions.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from string import Template
from uuid import UUID

from wavedl.core.layout import OperationKind
from wavedl.core.manifest import Manifest
from wavedl.core.scheduler_attempt import validate_scheduler_view

from .contracts import Plan, Topology, UnsupportedPlan, _json
from .local import LocalBackend


_REQUEST = "@@WAVEDL_REQUEST_ID@@"
_ATTEMPT = "@@WAVEDL_ATTEMPT@@"
_JOB = "@@WAVEDL_JOB_NAME@@"
_PLAN = "@@WAVEDL_PLAN_SHA256@@"
_MARKERS = (_REQUEST, _ATTEMPT, _JOB, _PLAN)
_TOKEN = re.compile(r"[A-Za-z0-9_.:+,=/\-]+\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def job_name(request_id: str) -> str:
    """Canonical collision-free scheduler name; accepts canonical UUIDs only."""
    if not isinstance(request_id, str):
        raise ValueError("request_id must be a canonical UUID")
    try:
        canonical = str(UUID(request_id))
    except ValueError as exc:
        raise ValueError("request_id must be a canonical UUID") from exc
    if canonical != request_id:
        raise ValueError("request_id must be a canonical UUID")
    return f"wavedl-{canonical}"


def _text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(char in value for char in ("\0", "\r", "\n", *_MARKERS))
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


# Runtime shell dollars are escaped for string.Template; inserted quoted values
# are not recursively interpreted by Template. $? remains the real shell status.
_BATCH = Template("""#!/bin/bash
$directives
set -uo pipefail
# NIL still exports SLURM_*; JWT is for submission authentication, not ranks.
# https://slurm.schedmd.com/sbatch.html https://slurm.schedmd.com/jwt.html
unset SLURM_JWT
export WAVEDL_RUN_ID=$run_id
export WAVEDL_REQUEST_ID=$request_id
export WAVEDL_ATTEMPT=$attempt
export WAVEDL_RUN_DIR=$working_directory
export WAVEDL_SLURM_RESTART_COUNT="$${SLURM_RESTART_COUNT:-0}"
export SLURM_EXPORT_ENV=ALL
$environment
training_exit=125
batch_exit=0
stage_out_exit=125
$binding_environment
if $restart_refused; then
    # Never silently reuse a journal/attempt after a scheduler requeue (T19).
    batch_exit=125
else
    cd -- $working_directory || batch_exit=$$?
    if [ "$$batch_exit" -eq 0 ]; then
        $setup
        batch_exit=$$?
    fi
    if [ "$$batch_exit" -eq 0 ]; then
        $bind_attempt
        batch_exit=$$?
    fi
    if [ "$$batch_exit" -eq 0 ]; then
        $rendezvous
        batch_exit=$$?
    fi
    if [ "$$batch_exit" -eq 0 ]; then
        $launch
        training_exit=$$?
    fi
    $stage_out
    stage_out_exit=$$?
fi
if $receipt_refused; then exit "$$batch_exit"; fi
$python -m wavedl.core.outcomes $receipt_directory --run-id $run_id --request-id $request_id --attempt $receipt_attempt --job-id "$$SLURM_JOB_ID" --cluster $cluster --restart-count "$${SLURM_RESTART_COUNT:-0}" --operation $operation --runtime-exit "$$training_exit" --batch-exit "$$batch_exit" --stage-out-exit "$$stage_out_exit"
receipt_exit=$$?
if [ "$$receipt_exit" -ne 0 ]; then exit "$$receipt_exit"; fi
if [ "$$batch_exit" -ne 0 ]; then exit "$$batch_exit"; fi
if [ "$$training_exit" -ne 0 ]; then exit "$$training_exit"; fi
exit "$$stage_out_exit"
""")


class SlurmPlanner:
    """Render a single allocation with exactly one launcher per machine."""

    name = "slurm"

    def __init__(self, site: SlurmSite):
        if not isinstance(site, SlurmSite):
            raise UnsupportedPlan("Explicit SlurmSite configuration is required")
        self.site = site

    def plan(
        self,
        manifest: Manifest,
        *,
        topology: Topology = Topology(),
        arguments: tuple[str, ...] = (),
        allocation_mode: str = "reserved",
    ) -> Plan:
        if allocation_mode not in {"reserved", "writer"}:
            raise UnsupportedPlan("Invalid attempt allocation mode")
        if not isinstance(manifest, Manifest) or not isinstance(topology, Topology):
            raise UnsupportedPlan("A resolved Manifest and Topology are required")
        if not self.site.cluster or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", self.site.cluster
        ):
            raise UnsupportedPlan("An explicit single Slurm cluster is required")
        if topology.machine_rank != 0:
            raise UnsupportedPlan("The allocation plan must start at machine_rank 0")
        if (
            manifest.operation == OperationKind.TRAIN
            and manifest.resolved_config.get("cv", 0)
            and topology != Topology()
        ):
            raise UnsupportedPlan(
                "Cross-validation requires one machine and one process"
            )
        runtime = manifest.runtime
        if self.site.requeue or allocation_mode == "writer":
            try:
                validate_scheduler_view(manifest, Path(manifest.run_directory))
            except ValueError:
                raise UnsupportedPlan(
                    "Scheduler requeue requires exact storage views"
                ) from None
        if (
            runtime.mode not in {"selected", "existing"}
            or runtime.container is not None
        ):
            raise UnsupportedPlan("Slurm does not implement bundled/container runtimes")
        if (
            runtime.device not in {"cpu", "cuda"}
            and manifest.operation == OperationKind.TRAIN
        ):
            raise UnsupportedPlan(
                "Slurm training requires explicit CPU or CUDA selection"
            )
        if manifest.operation == OperationKind.TRAIN and any(
            not manifest.resolved_config.get(name) for name in ("model", "data_path")
        ):
            raise UnsupportedPlan("Training requires model and data_path")
        if runtime.device == "cpu" and topology != Topology():
            raise UnsupportedPlan("CPU launcher supports one machine and one process")
        if manifest.operation != OperationKind.TRAIN and topology != Topology():
            raise UnsupportedPlan("Only training supports distributed topology")
        if runtime.device == "cuda":
            if self.site.gpus_per_node != topology.processes_per_machine:
                raise UnsupportedPlan("GPU allocation must equal processes_per_machine")
            if "CUDA_VISIBLE_DEVICES" in runtime.environment:
                raise UnsupportedPlan(
                    "CUDA visibility must be owned by Slurm allocation"
                )
        elif self.site.gpus_per_node != 0:
            raise UnsupportedPlan("GPU allocation requires explicit CUDA training")
        for path in (manifest.working_directory, runtime.python):
            _text(path, "execution path")
        mapped_run = Path(manifest.run_directory)
        for mapping in sorted(
            manifest.storage_map, key=lambda item: len(item.host_root), reverse=True
        ):
            try:
                relative = mapped_run.relative_to(mapping.host_root)
            except ValueError:
                continue
            mapped_run = Path(mapping.execution_root) / relative
            break
        if mapped_run != Path(manifest.working_directory):
            raise UnsupportedPlan(
                "Run host/execution paths require an exact storage mapping"
            )
        if "%" in manifest.working_directory:
            raise UnsupportedPlan(
                "Slurm filename patterns cannot express percent in run paths"
            )
        if any(char in manifest.working_directory for char in ("'", '"', "\\")):
            raise UnsupportedPlan(
                "Slurm directive paths cannot express shell quote concatenation"
            )
        for key, value in runtime.environment.items():
            if not _ENV.fullmatch(key) or key.startswith(
                (
                    "WAVEDL_",
                    "SLURM_",
                    "SRUN_",
                    "SBATCH_",
                    "SCANCEL_",
                    "SQUEUE_",
                    "SACCT_",
                    "SCONTROL_",
                )
            ):
                raise UnsupportedPlan(
                    "Environment cannot override runtime/scheduler identity"
                )
            if any(char in value for char in ("\0", "\r", "\n", *_MARKERS)):
                raise UnsupportedPlan("Invalid environment value")

        # Reuse the established CLI option contract without pretending Local can
        # execute remapped storage or multi-node allocations. All paths already
        # belong to the execution side of the resolved manifest.
        render_manifest = replace(
            manifest,
            run_directory=manifest.working_directory,
            storage_map=(),
            runtime=replace(runtime, device="cpu")
            if manifest.operation == OperationKind.TRAIN
            else runtime,
        )
        command = list(LocalBackend().plan(render_manifest, arguments=arguments).argv)
        if manifest.operation == OperationKind.TRAIN:
            command[command.index("--num_machines") + 1] = str(topology.num_machines)
            # launcher --num_gpus is accelerate's TOTAL --num_processes.
            command[command.index("--num_gpus") + 1] = str(
                topology.num_machines * topology.processes_per_machine
            )
            if runtime.device == "cuda":
                command.remove("--cpu")
        for argument in command:
            if any(marker in argument for marker in _MARKERS):
                raise UnsupportedPlan(
                    "Reserved submission identity marker in arguments"
                )
        launch = shlex.join(command)
        probe = ""
        if runtime.device == "cuda":
            # Probe inside EACH scheduler task, with the selected interpreter
            # and that node's Slurm-owned visibility. Import/count alone cannot
            # prove the allocated devices can actually accept work.
            probe_code = (
                "import torch\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('Selected CUDA runtime is unavailable')\n"
                f"if torch.cuda.device_count() != {topology.processes_per_machine}:\n"
                "    raise RuntimeError('CUDA visibility differs from planned allocation')\n"
                f"for index in range({topology.processes_per_machine}):\n"
                "    tensor = torch.empty(1, device=f'cuda:{index}')\n"
                "    torch.cuda.synchronize(index)\n"
                "    del tensor\n"
            )
            probe = shlex.join([runtime.python, "-c", probe_code]) + " && "
        rendezvous = ":"
        if topology.num_machines > 1:
            command[command.index("--machine_rank") + 1] = "@@RANK@@"
            launch = shlex.join(command).replace("@@RANK@@", '"$SLURM_PROCID"')
            launch += f' --main_process_ip "$WAVEDL_MAIN_HOST" --main_process_port {self.site.rendezvous_port}'
            rendezvous = (
                'WAVEDL_MAIN_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST") && '
                "WAVEDL_MAIN_HOST=${WAVEDL_MAIN_HOST%%$'\\n'*} && "
                'test -n "$WAVEDL_MAIN_HOST" && export WAVEDL_MAIN_HOST'
            )
            launch = (
                f"srun --nodes={topology.num_machines} --ntasks={topology.num_machines} "
                f"--ntasks-per-node=1 --cpus-per-task={self.site.cpus_per_task} "
                "--kill-on-bad-exit=1 bash -c " + shlex.quote(probe + "exec " + launch)
            )
        else:
            launch = probe + launch
        directives = [
            f"#SBATCH --job-name={_JOB}",
            f"#SBATCH --comment={_REQUEST}",
            f"#SBATCH --partition={self.site.partition}",
            f"#SBATCH --nodes={topology.num_machines}",
            f"#SBATCH --ntasks={topology.num_machines}",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={self.site.cpus_per_task}",
            f"#SBATCH --mem={self.site.memory}",
            f"#SBATCH --time={self.site.time}",
            "#SBATCH --requeue" if self.site.requeue else "#SBATCH --no-requeue",
            "#SBATCH --export=NIL",
            f"#SBATCH --chdir={shlex.quote(manifest.working_directory)}",
            f"#SBATCH --output={shlex.quote(str(Path(manifest.working_directory) / 'logs' / 'slurm-%j.out'))}",
            f"#SBATCH --error={shlex.quote(str(Path(manifest.working_directory) / 'logs' / 'slurm-%j.err'))}",
        ]
        for name, option in (
            ("account", "account"),
            ("constraint", "constraint"),
            ("cluster", "clusters"),
        ):
            value = getattr(self.site, name)
            if value is not None:
                directives.append(f"#SBATCH --{option}={value}")
        if self.site.gpus_per_node:
            directives.append(f"#SBATCH --gpus-per-node={self.site.gpus_per_node}")
        script = _BATCH.substitute(
            directives="\n".join(directives),
            run_id=shlex.quote(manifest.run_id),
            request_id=_REQUEST,
            attempt=_ATTEMPT,
            operation=manifest.operation.value,
            working_directory=shlex.quote(manifest.working_directory),
            python=shlex.quote(runtime.python),
            environment="\n".join(
                f"export {key}={shlex.quote(value)}"
                for key, value in sorted(runtime.environment.items())
            ),
            setup=" && ".join(
                f"{{ {command}; }}" for command in self.site.setup_commands
            )
            or ":",
            rendezvous=rendezvous,
            launch=launch,
            stage_out=f"( {self.site.stage_out_command} )"
            if self.site.stage_out_command
            else ":",
            receipt_directory=shlex.quote(
                str(Path(manifest.working_directory) / "attempts" / _ATTEMPT)
            ),
            cluster=shlex.quote(self.site.cluster),
            binding_environment=(
                "export WAVEDL_SCHEDULER_BINDING=1\n"
                f"export WAVEDL_SCHEDULER_INITIAL_ATTEMPT={_ATTEMPT}\n"
                f"export WAVEDL_SCHEDULER_CLUSTER={shlex.quote(self.site.cluster)}\n"
                "WAVEDL_BINDING_READY=0\n"
                "export WAVEDL_SCHEDULER_OWNER=''"
                if self.site.requeue
                else ""
            ),
            restart_refused="false"
            if self.site.requeue
            else '[ "$WAVEDL_SLURM_RESTART_COUNT" != "0" ]',
            bind_attempt=(
                f"WAVEDL_SCHEDULER_OWNER=$({shlex.quote(runtime.python)} -c "
                + shlex.quote("import os,pwd; print(pwd.getpwuid(os.getuid()).pw_name)")
                + ") && export WAVEDL_SCHEDULER_OWNER && "
                f"WAVEDL_ATTEMPT=$({shlex.quote(runtime.python)} -m wavedl.launch_runtime scheduler-bind "
                f"{shlex.quote(manifest.working_directory)}) && "
                "export WAVEDL_ATTEMPT && WAVEDL_BINDING_READY=1"
                if self.site.requeue
                else ":"
            ),
            receipt_refused='[ "$WAVEDL_BINDING_READY" != "1" ]'
            if self.site.requeue
            else "false",
            receipt_attempt='"$WAVEDL_ATTEMPT"' if self.site.requeue else _ATTEMPT,
        )
        if self.site.requeue:
            # Bound dynamic receipt path must use the coordinator's reservation,
            # never the original submission attempt after a restart.
            original = shlex.quote(
                str(Path(manifest.working_directory) / "attempts" / _ATTEMPT)
            )
            script = script.replace(
                original,
                shlex.quote(str(Path(manifest.working_directory) / "attempts"))
                + '/"$WAVEDL_ATTEMPT"',
            )
        if allocation_mode == "writer":
            # Preserve approved legacy bytes exactly; the new mode explicitly
            # removes preallocated identity and binds every initial writer.
            script = script.replace(
                f"export WAVEDL_ATTEMPT={_ATTEMPT}\n", "unset WAVEDL_ATTEMPT\n"
            )
            binding_environment = (
                "export WAVEDL_SCHEDULER_BINDING=1\n"
                "export WAVEDL_ATTEMPT_ALLOCATION=writer\n"
                f"export WAVEDL_PLAN_SHA256={_PLAN}\n"
                "unset WAVEDL_SCHEDULER_INITIAL_ATTEMPT\n"
                f"export WAVEDL_SCHEDULER_CLUSTER={shlex.quote(self.site.cluster)}\n"
                "WAVEDL_BINDING_READY=0\n"
                f"WAVEDL_SCHEDULER_OWNER=$({shlex.quote(runtime.python)} -c "
                + shlex.quote("import os,pwd; print(pwd.getpwuid(os.getuid()).pw_name)")
                + ") || exit 125\nexport WAVEDL_SCHEDULER_OWNER"
            )
            if self.site.requeue:
                start = script.index("export WAVEDL_SCHEDULER_BINDING=1")
                end = script.index("\nif ", start)
                script = script[:start] + binding_environment + script[end:]
            else:
                script = script.replace(
                    "\nif ", "\n" + binding_environment + "\nif ", 1
                )
                bind = (
                    f"WAVEDL_SCHEDULER_OWNER=$({shlex.quote(runtime.python)} -c "
                    + shlex.quote(
                        "import os,pwd; print(pwd.getpwuid(os.getuid()).pw_name)"
                    )
                    + ") && export WAVEDL_SCHEDULER_OWNER && "
                    f"WAVEDL_ATTEMPT=$({shlex.quote(runtime.python)} -m wavedl.launch_runtime scheduler-bind "
                    f"{shlex.quote(manifest.working_directory)}) && "
                    "export WAVEDL_ATTEMPT && WAVEDL_BINDING_READY=1"
                )
                # The three sequential guarded stages are setup/bind/rendezvous.
                marker = '    if [ "$batch_exit" -eq 0 ]; then\n        :\n        batch_exit=$?\n    fi'
                script = script.replace(
                    marker, marker.replace("        :", "        " + bind), 1
                )
            receipt_start = script.index("if ", script.index("    stage_out_exit=$?"))
            receipt_end = script.index("\nreceipt_exit=$?", receipt_start)
            receipt = (
                'receipt_attempt=()\nif [ "$WAVEDL_BINDING_READY" = "1" ]; then receipt_attempt=(--attempt "$WAVEDL_ATTEMPT"); fi\n'
                f"{shlex.quote(runtime.python)} -m wavedl.core.submission_outcomes "
                f"{shlex.quote(manifest.working_directory)} --run-id {shlex.quote(manifest.run_id)} --request-id {_REQUEST} "
                '--owner "$WAVEDL_SCHEDULER_OWNER" ${receipt_attempt[@]+"${receipt_attempt[@]}"} '
                f'--plan-sha256 {_PLAN} --job-id "$SLURM_JOB_ID" '
                f"--cluster {shlex.quote(self.site.cluster)} "
                '--restart-count "${SLURM_RESTART_COUNT:-0}" '
                f'--operation {manifest.operation.value} --runtime-exit "$training_exit" '
                '--batch-exit "$batch_exit" --stage-out-exit "$stage_out_exit"'
            )
            script = script[:receipt_start] + receipt + script[receipt_end:]
        return Plan(
            target=self.name,
            manifest_json=_json(manifest.to_dict()),
            script=script,
            topology=topology,
            resources=tuple(
                sorted(
                    {
                        "partition": self.site.partition,
                        "account": self.site.account,
                        "cluster": self.site.cluster,
                        "nodes": topology.num_machines,
                        "ntasks": topology.num_machines,
                        "ntasks_per_node": 1,
                        "processes_per_machine": topology.processes_per_machine,
                        "num_processes": topology.num_machines
                        * topology.processes_per_machine,
                        "cpus_per_task": self.site.cpus_per_task,
                        "memory_per_node": self.site.memory,
                        "gpus_per_node": self.site.gpus_per_node,
                        "time": self.site.time,
                        "constraint": self.site.constraint,
                        "rendezvous_port": self.site.rendezvous_port,
                        "requeue": self.site.requeue,
                    }.items()
                )
            ),
            environment=tuple(sorted(runtime.environment.items())),
            warnings=(
                "Submission must precreate host run/log/attempt directories on shared mapped storage.",
                "Scheduler requeue uses a writer startup binding and native-verified durable activation."
                if self.site.requeue
                else "Scheduler requeue is disabled; nonzero restart_count refuses execution.",
                "Site shell setup and stage-out hooks are trusted administrator configuration.",
                "Site must support --export=NIL; runtime environment and setup hooks define execution environment explicitly.",
            ),
            allocation_mode=allocation_mode,
        )


def bind_submission(plan: Plan, request_id: str, attempt: int | None) -> str:
    """Bind durable identity only; never regenerate the approved preview."""
    name = job_name(request_id)
    writer = isinstance(plan, Plan) and plan.allocation_mode == "writer"
    if writer and attempt is not None:
        raise ValueError("Writer submission must not reserve an attempt")
    if not writer and (type(attempt) is not int or attempt < 1):
        raise ValueError("attempt must be a positive integer")
    if not isinstance(plan, Plan) or plan.target != "slurm" or not plan.script:
        raise UnsupportedPlan("A Slurm script plan is required")
    script = plan.script
    required = (_REQUEST, _JOB, _PLAN) if writer else (_REQUEST, _ATTEMPT, _JOB)
    if not all(marker in script for marker in required):
        raise UnsupportedPlan("Missing reserved submission identity slots")
    for marker, value in (
        (_REQUEST, request_id),
        (_ATTEMPT, str(attempt)),
        (_JOB, name),
        (_PLAN, hashlib.sha256(plan.to_json().encode()).hexdigest()),
    ):
        script = script.replace(marker, value)
    return script
