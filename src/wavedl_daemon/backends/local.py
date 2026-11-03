# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Pure Local planning with bounded live process ownership for execution."""

from __future__ import annotations

from pathlib import Path

from wavedl.core.inference_cli import validate_import_modules
from wavedl.core.training_cli import validate_cv_options
from wavedl_daemon.runs.model import Operation

from .contracts import Plan, RunSpec, Topology, UnsupportedPlan
from .local_execution import LocalExecution, reserved_local_environment


_TRAIN_VALUES = frozenset(
    [
        "model",
        "batch_size",
        "lr",
        "epochs",
        "patience",
        "weight_decay",
        "grad_clip",
        "grad_accum_steps",
        "loss",
        "huber_delta",
        "loss_weights",
        "optimizer",
        "momentum",
        "betas",
        "scheduler",
        "scheduler_patience",
        "min_lr",
        "scheduler_factor",
        "warmup_epochs",
        "step_size",
        "milestones",
        "data_path",
        "workers",
        "seed",
        "cache_validate",
        "cv",
        "cv_bins",
        "resume",
        "save_every",
        "precision",
        "constraint",
        "constraint_file",
        "constraint_weight",
        "constraint_reduction",
        "project_name",
        "run_name",
        "import_modules",
    ]
)
_TRAIN_FLAGS = frozenset(
    [
        "pretrained",
        "nesterov",
        "deterministic",
        "single_channel",
        "cv_stratify",
        "fresh",
        "compile",
        "wandb",
        "wandb_watch",
    ]
)
_TEST_VALUES = frozenset(
    [
        "checkpoint",
        "data_path",
        "model",
        "format",
        "input_key",
        "output_key",
        "param_names",
        "input_channels",
        "batch_size",
        "workers",
        "plot_format",
        "export",
        "export_path",
        "export_opset",
        "out_size",
        "import_modules",
    ]
)
_TEST_FLAGS = frozenset(["save_predictions", "plot", "verbose", "no_denorm"])
# Recorded CLI configurations carry these, but the daemon never runs them:
# pickled datasets execute code, and selection_metric is an HPO-only switch.
_DEFAULT_ONLY = {
    Operation.TRAIN: {"allow_pickle": False, "selection_metric": "loss"},
    Operation.TEST: {"allow_pickle": False},
    Operation.PLOT: {"allow_pickle": False},
}


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
                validate_cv_options(spec.resolved_config)
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
        if (
            spec.operation != Operation.TRAIN
            and topology.processes_per_machine != 1
        ):
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
        values = (
            _TRAIN_VALUES if spec.operation == Operation.TRAIN else _TEST_VALUES
        )
        flags = (
            _TRAIN_FLAGS if spec.operation == Operation.TRAIN else _TEST_FLAGS
        )
        config = spec.resolved_config
        if "import_modules" in config:
            try:
                validate_import_modules(config["import_modules"])
            except ValueError as error:
                raise UnsupportedPlan(str(error)) from error
        if spec.operation in {Operation.TEST, Operation.PLOT} and any(
            not config.get(name) for name in ("checkpoint", "data_path")
        ):
            raise UnsupportedPlan("Test and plot require checkpoint and data_path")
        argv = [runtime.python, "-u", "-m"]
        if spec.operation == Operation.TRAIN:
            # Explicit topology prevents launcher hardware auto-detection from
            # silently multiplying this single-machine request.
            argv += [
                "wavedl.launcher",
                "--num_machines",
                "1",
                "--machine_rank",
                "0",
                "--num_gpus",
                str(topology.processes_per_machine),
            ]
            if runtime.device == "cpu":
                argv.append("--cpu")
            if config.get("precision") is not None:
                argv += ["--mixed_precision", str(config["precision"])]
        else:
            argv += ["wavedl.test"]
        for name, value in sorted(config.items()):
            if name in {"output_dir", "config"} or value is None:
                continue
            default_only = _DEFAULT_ONLY.get(spec.operation, {})
            if name in default_only:
                if value != default_only[name] or type(value) is not type(
                    default_only[name]
                ):
                    raise UnsupportedPlan(f"Unsupported resolved option: {name}")
                continue
            if name in flags:
                if type(value) is not bool:
                    raise UnsupportedPlan(f"{name} must be boolean")
                if value:
                    argv.append(f"--{name}")
            elif name in values:
                multiple = (
                    {"constraint", "constraint_weight", "import_modules"}
                    if spec.operation == Operation.TRAIN
                    else {"param_names", "plot_format", "import_modules"}
                )
                if isinstance(value, list) and name not in multiple:
                    raise UnsupportedPlan(f"Option does not accept a list: {name}")
                if value == [] and name in {"constraint", "import_modules"}:
                    continue
                items = value if isinstance(value, list) else [value]
                if not items or not all(
                    type(item) in {str, int, float} for item in items
                ):
                    raise UnsupportedPlan(f"Cannot render resolved option: {name}")
                if any(
                    isinstance(item, str) and ("\0" in item or item.startswith("--"))
                    for item in items
                ):
                    raise UnsupportedPlan(f"Unsafe option value: {name}")
                option = "import" if name == "import_modules" else name
                argv += [f"--{option}", *(str(item) for item in items)]
            else:
                raise UnsupportedPlan(f"Unsupported resolved option: {name}")
        if spec.operation == Operation.PLOT and "--plot" not in argv:
            argv.append("--plot")
        argv += ["--output_dir", spec.execution_path(spec.paths.output)]
        return Plan(
            target=self.name,
            spec=spec,
            argv=tuple(argv),
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
