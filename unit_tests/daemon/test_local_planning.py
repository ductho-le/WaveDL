"""Local plans refuse requests they cannot faithfully execute."""

from dataclasses import replace
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, StorageMapping, build_manifest
from wavedl_daemon.backends.contracts import Topology, UnsupportedPlan
from wavedl_daemon.backends.local import LocalBackend


def manifest(tmp_path, operation=OperationKind.TRAIN, **config):
    return build_manifest(
        RunLayout(tmp_path / "owned", str(uuid4())),
        operation=operation,
        resolved_config=config,
        runtime=RuntimeSelection(
            "venv/bin/python",
            "selected",
            "cpu" if operation == OperationKind.TRAIN else "auto",
            {},
        ),
        submission_cwd=tmp_path,
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("resume", "private-checkpoint"),
        ("save_every", 1),
        ("fresh", True),
        ("compile", True),
        ("wandb", True),
        ("wandb_watch", True),
        ("project_name", "private-project"),
        ("run_name", "private-run"),
    ],
)
def test_direct_local_cv_plan_refuses_unsupported_options(tmp_path, option, value):
    with pytest.raises(UnsupportedPlan, match=f"--{option}") as error:
        LocalBackend().plan(manifest(tmp_path, cv=2, **{option: value}))
    assert "private" not in str(error.value)
    assert not (tmp_path / "owned").exists()


def test_selected_venv_is_preserved_and_plan_has_no_side_effects(tmp_path):
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").symlink_to("/usr/bin/python3")
    request = manifest(tmp_path, data_path="train data.npz", epochs=2)
    plan = LocalBackend().plan(request)
    assert plan.argv[0] == str(tmp_path / "venv" / "bin" / "python")
    assert plan.argv[-2:] == ("--output_dir", request.working_directory)
    assert str(tmp_path / "train data.npz") in plan.argv
    assert not (tmp_path / "owned").exists()
    assert callable(LocalBackend().submit)
    assert "--cpu" in plan.argv


def test_writer_allocation_is_explicit_and_preserves_reserved_contract(tmp_path):
    request = manifest(tmp_path, epochs=1)
    backend = LocalBackend()
    default = backend.plan(request)
    reserved = backend.plan(request, allocation_mode="reserved")
    writer = backend.plan(request, allocation_mode="writer")
    assert default == reserved
    assert default.to_json() == reserved.to_json()
    assert "allocation_mode" not in reserved.to_dict()
    assert writer.to_dict() == {**reserved.to_dict(), "allocation_mode": "writer"}
    assert writer == replace(reserved, allocation_mode="writer")
    assert not (tmp_path / "owned").exists()


@pytest.mark.parametrize("mode", ["invalid", "", None, True, 0, [], {}])
def test_invalid_allocation_mode_refused_before_manifest_validation(mode):
    with pytest.raises(UnsupportedPlan, match="allocation mode"):
        LocalBackend().plan(None, allocation_mode=mode)


@pytest.mark.parametrize(
    "arguments",
    [("--output_dir", "/other"), ("--output_dir=/other",), ("--output-dir=/other",)],
)
def test_caller_output_override_refused(tmp_path, arguments):
    with pytest.raises(UnsupportedPlan, match="owned"):
        LocalBackend().plan(manifest(tmp_path), arguments=arguments)


def test_multiple_machines_refused(tmp_path):
    with pytest.raises(UnsupportedPlan, match="one machine"):
        LocalBackend().plan(manifest(tmp_path), topology=Topology(2, 1, 0))


def test_cross_validation_distributed_shape_is_refused_at_preview(tmp_path):
    with pytest.raises(UnsupportedPlan, match="Cross-validation"):
        LocalBackend().plan(manifest(tmp_path, cv=2), topology=Topology(1, 2))
    with pytest.raises(UnsupportedPlan, match="fold count"):
        LocalBackend().plan(manifest(tmp_path, cv="2"))


@pytest.mark.parametrize(
    "mode,container", [("container", "image"), ("bundled", None), ("selected", "image")]
)
def test_unsupported_runtime_refused(tmp_path, mode, container):
    request = manifest(tmp_path)
    request = replace(
        request, runtime=replace(request.runtime, mode=mode, container=container)
    )
    with pytest.raises(UnsupportedPlan, match="runtimes"):
        LocalBackend().plan(request)


def test_storage_remapping_refused(tmp_path):
    request = manifest(tmp_path)
    request = replace(
        request,
        storage_map=(StorageMapping(str(tmp_path), str(tmp_path / "compute")),),
    )
    with pytest.raises(UnsupportedPlan, match="remapped"):
        LocalBackend().plan(request)


@pytest.mark.parametrize("operation", [OperationKind.TEST, OperationKind.PLOT])
def test_operation_has_explicit_entrypoint_and_required_inputs(tmp_path, operation):
    with pytest.raises(UnsupportedPlan, match="checkpoint"):
        LocalBackend().plan(manifest(tmp_path, operation))
    plan = LocalBackend().plan(
        manifest(tmp_path, operation, checkpoint="checkpoint", data_path="data.npz")
    )
    assert plan.operation == operation
    assert plan.argv[3] == "wavedl.test"
    assert ("--plot" in plan.argv) == (operation == OperationKind.PLOT)


def test_unrenderable_config_is_not_silently_dropped(tmp_path):
    with pytest.raises(UnsupportedPlan, match="Unsupported resolved"):
        LocalBackend().plan(manifest(tmp_path, arbitrary={"nested": 1}))


def test_scalar_cli_option_rejects_list_instead_of_rendering_invalid_command(tmp_path):
    with pytest.raises(UnsupportedPlan, match="does not accept a list"):
        LocalBackend().plan(manifest(tmp_path, epochs=[1, 2]))
    plan = LocalBackend().plan(manifest(tmp_path, constraint=[], import_modules=[]))
    assert "--constraint" not in plan.argv


def test_precision_is_the_same_for_launcher_and_training(tmp_path):
    plan = LocalBackend().plan(manifest(tmp_path, precision="no"))
    assert plan.argv[plan.argv.index("--mixed_precision") + 1] == "no"
    assert plan.argv[plan.argv.index("--precision") + 1] == "no"


@pytest.mark.parametrize(
    "operation,device",
    [(OperationKind.TRAIN, "mps")],
)
def test_unexpressible_device_selection_refused(tmp_path, operation, device):
    request = manifest(
        tmp_path, operation, checkpoint="checkpoint", data_path="data.npz"
    )
    request = replace(request, runtime=replace(request.runtime, device=device))
    with pytest.raises(UnsupportedPlan, match="selection is unsupported"):
        LocalBackend().plan(request)


@pytest.mark.parametrize("operation", [OperationKind.TEST, OperationKind.PLOT])
def test_inference_device_selection_matches_approved_runtime(tmp_path, operation):
    request = manifest(
        tmp_path, operation, checkpoint="checkpoint", data_path="data.npz", device="cpu"
    )
    request = replace(request, runtime=replace(request.runtime, device="cpu"))
    plan = LocalBackend().plan(request)
    assert plan.argv[plan.argv.index("--device") + 1] == "cpu"


@pytest.mark.parametrize("visible", [None, "", "-1", "0,0", "0, 0"])
def test_single_process_cuda_cannot_silently_use_default_or_disabled_device(
    tmp_path, visible
):
    request = manifest(tmp_path)
    environment = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
    request = replace(
        request,
        runtime=replace(request.runtime, device="cuda", environment=environment),
    )
    with pytest.raises(UnsupportedPlan, match="visible CUDA"):
        LocalBackend().plan(request)


def test_cuda_topology_is_explicit_not_hardware_autodetected(tmp_path):
    request = manifest(tmp_path)
    cpu_plan = LocalBackend().plan(request, topology=Topology(1, 2))
    assert "--cpu" in cpu_plan.argv
    assert cpu_plan.argv[cpu_plan.argv.index("--num_gpus") + 1] == "2"
    request = replace(request, runtime=replace(request.runtime, device="cuda"))
    with pytest.raises(UnsupportedPlan, match="CUDA"):
        LocalBackend().plan(request, topology=Topology(1, 2))
    request = replace(
        request,
        runtime=replace(
            request.runtime, device="cuda", environment={"CUDA_VISIBLE_DEVICES": "0,1"}
        ),
    )
    plan = LocalBackend().plan(request, topology=Topology(1, 2))
    assert plan.argv[plan.argv.index("--num_gpus") + 1] == "2"
    assert dict(plan.environment)["CUDA_VISIBLE_DEVICES"] == "0,1"
