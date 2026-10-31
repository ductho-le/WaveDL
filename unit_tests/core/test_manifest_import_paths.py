"""Explicit registration paths are resolved before execution changes cwd."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, StorageMapping, build_manifest


RUN_ID = "00000000-0000-4000-8000-000000000001"


def manifest(tmp_path, config, *, operation=OperationKind.TEST, mappings=()):
    return build_manifest(
        RunLayout(tmp_path / "runs" / RUN_ID, RUN_ID),
        operation=operation,
        resolved_config=config,
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
        storage_map=mappings,
    )


@pytest.mark.parametrize("operation", list(OperationKind))
def test_import_files_resolve_against_submission_directory(
    tmp_path, monkeypatch, operation
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    absolute = str(tmp_path / "absolute model.py")
    config = {
        "import_modules": ["custom.module", "models/../custom model.py", absolute]
    }
    original = deepcopy(config)

    result = manifest(tmp_path, config, operation=operation)

    assert result.resolved_config["import_modules"] == [
        "custom.module",
        str(tmp_path / "custom model.py"),
        absolute,
    ]
    assert config == original
    assert result.resolved_config["import_modules"] is not config["import_modules"]


def test_import_files_follow_most_specific_map_once(tmp_path):
    specific = tmp_path / "models"
    execution = tmp_path / "execution"
    mappings = (
        StorageMapping(str(tmp_path), str(execution)),
        StorageMapping(str(specific), str(execution / "special")),
    )
    config = {
        "import_modules": [
            "models/custom.py",
            str(specific / "absolute.py"),
            "other.py",
            "custom.module",
        ]
    }

    result = manifest(tmp_path, config, mappings=mappings)

    assert result.resolved_config["import_modules"] == [
        str(execution / "special" / "custom.py"),
        str(execution / "special" / "absolute.py"),
        str(execution / "other.py"),
        "custom.module",
    ]
    # Bootstrap compares parsed execution-side argv without the host map.
    actual = build_manifest(
        RunLayout(Path(result.working_directory), RUN_ID),
        operation=result.operation,
        resolved_config=result.resolved_config,
        runtime=result.runtime,
        submission_cwd=Path(result.working_directory),
        path_fields=result.path_fields,
    )
    assert actual.resolved_config == result.resolved_config


def test_import_host_symlink_resolves_but_execution_root_stays_lexical(tmp_path):
    actual = tmp_path / "host"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    execution = tmp_path / "execution"
    execution.symlink_to(unrelated, target_is_directory=True)

    result = manifest(
        tmp_path,
        {"import_modules": [str(alias / "model.py")]},
        mappings=(StorageMapping(str(alias), str(execution)),),
    )

    assert result.resolved_config["import_modules"] == [str(execution / "model.py")]
    assert result.storage_map == (StorageMapping(str(actual), str(execution)),)


def test_derived_plot_registration_host_paths_are_mapped_exactly_once(tmp_path):
    """A second approved operation must reuse host inputs, not mapped argv."""
    host = tmp_path / "host"
    execution = tmp_path / "execution"
    mappings = (
        StorageMapping(str(host), str(execution)),
        StorageMapping(str(execution), str(tmp_path / "double-mapped")),
    )
    config = {"import_modules": ["registered.module", str(host / "custom model.py")]}
    training = manifest(
        tmp_path, config, operation=OperationKind.TRAIN, mappings=mappings
    )
    plot = manifest(tmp_path, config, operation=OperationKind.PLOT, mappings=mappings)
    assert training.resolved_config["import_modules"] == [
        "registered.module",
        str(execution / "custom model.py"),
    ]
    assert (
        plot.resolved_config["import_modules"]
        == (training.resolved_config["import_modules"])
    )
    remapped = manifest(
        tmp_path,
        training.resolved_config,
        operation=OperationKind.PLOT,
        mappings=mappings,
    )
    assert (
        remapped.resolved_config["import_modules"]
        != (training.resolved_config["import_modules"])
    ), "Execution-side imports are not safe submission-side inputs"


@pytest.mark.parametrize("config", [{}, {"import_modules": []}])
def test_missing_and_empty_imports_keep_legacy_config(tmp_path, config):
    result = manifest(tmp_path, config)
    assert ("import_modules" in result.resolved_config) == ("import_modules" in config)
    if "import_modules" in config:
        assert result.resolved_config["import_modules"] == []


@pytest.mark.parametrize(
    "modules",
    [
        None,
        "custom.py",
        ("custom.py",),
        {"custom.py": True},
        True,
        [None],
        [1],
        [""],
        ["--device"],
        ["-x.py"],
        ["bad\0name"],
        ["x" * 8193],
        ["custom"] * 129,
    ],
)
def test_malformed_import_lists_are_rejected(tmp_path, modules):
    with pytest.raises(ValueError):
        manifest(tmp_path, {"import_modules": modules})
