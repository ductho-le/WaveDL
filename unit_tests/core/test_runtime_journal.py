import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl import journal as runtime_journal
from wavedl.core import journal_writer
from wavedl.core.events import HelloPayload, MetricPayload, StatePayload
from wavedl.core.manifest import RuntimeSelection, StorageMapping, write_manifest
from wavedl.core.protocol import ProtocolError, decode
from wavedl.journal import RunJournal, bootstrap_run


def runtime():
    return RuntimeSelection(sys.executable, "selected", "cpu", {"TEST": "yes"})


def hello():
    return HelloPayload(1, 1, "test-runtime", "3.11", "cpu", ["train"], sys.executable)


def boot(tmp_path, **kwargs):
    return bootstrap_run(
        tmp_path,
        resolved_config={"data_path": "original.npz", "epochs": 1},
        runtime=runtime(),
        environment={},
        **kwargs,
    )


def events(bootstrap):
    return [
        decode(line)
        for line in bootstrap.layout.journal(bootstrap.attempt)
        .read_bytes()
        .splitlines()
    ]


def test_standalone_preserves_legacy_output_and_reserves_resume(tmp_path):
    legacy = tmp_path / "best_model.pth"
    legacy.write_bytes(b"legacy")
    first = boot(tmp_path)
    manifest_bytes = first.layout.manifest.read_bytes()
    assert first.manifest.runtime == runtime()
    assert first.manifest.resolved_config["data_path"].endswith("original.npz")
    with RunJournal(first, hello=hello()) as journal:
        journal.emit("state", StatePayload("running"))
    second = boot(tmp_path)
    assert second.attempt == 2
    assert first.layout.run_id == second.layout.run_id
    assert second.layout.manifest.read_bytes() == manifest_bytes
    assert legacy.read_bytes() == b"legacy"
    assert len(events(first)) == 2


def test_injected_identity_not_regenerated_or_reallocated(tmp_path):
    first = boot(tmp_path)
    supplied = bootstrap_run(
        tmp_path,
        resolved_config=first.request_manifest.resolved_config,
        runtime=runtime(),
        environment=first.environment,
    )
    assert supplied == first
    assert sorted(path.name for path in first.layout.attempts.iterdir()) == ["1"]
    with RunJournal(supplied, hello=hello()):
        pass
    with pytest.raises(FileExistsError):
        RunJournal(supplied, hello=hello())


@pytest.mark.parametrize(
    "key", ["WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT"]
)
def test_injected_identity_mismatch_rejected(tmp_path, key):
    first = boot(tmp_path)
    env = first.environment | {key: "2" if key == "WAVEDL_ATTEMPT" else str(uuid4())}
    with pytest.raises(ValueError):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=env,
        )
    assert not first.layout.journal(1).exists()


def test_rank_only_diagnostics_and_no_rank_bootstrap_writes(tmp_path):
    with pytest.raises(ValueError, match="Distributed"):
        boot(tmp_path / "absent", is_main_process=False, rank=1)
    assert not (tmp_path / "absent").exists()
    first = boot(tmp_path)
    follower = bootstrap_run(
        tmp_path,
        resolved_config=first.request_manifest.resolved_config,
        runtime=runtime(),
        environment=first.environment,
        is_main_process=False,
        rank=1,
    )
    with RunJournal(follower, hello=hello(), is_main_process=False, rank=1) as journal:
        assert journal.emit("state", StatePayload("running")) is None
        journal.log("rank diagnostic")
    assert not first.layout.journal(1).exists()
    assert (
        first.layout.attempt_dir(1) / "rank/1.log"
    ).read_text() == "rank diagnostic\n"


def test_hello_mandatory_first_validated_and_not_repeated(tmp_path):
    first = boot(tmp_path)
    with pytest.raises(ProtocolError):
        RunJournal(first, hello=replace(hello(), protocol_min=2, protocol_max=2))
    assert not first.layout.journal(1).exists()
    with RunJournal(first, hello=hello()) as journal:
        with pytest.raises(ProtocolError):
            journal.emit("hello", hello())
        with pytest.raises(ProtocolError):
            journal.emit("metric", {"name": "loss", "value": "bad", "step": 0})
        journal.emit("metric", MetricPayload("loss", float("inf"), 0))
    assert [(event.seq, event.type) for event in events(first)] == [
        (0, "hello"),
        (1, "metric"),
    ]
    assert events(first)[1].payload["value"] is None


def test_fsync_failure_is_visible_and_attempt_not_reopened(tmp_path, monkeypatch):
    first = boot(tmp_path)

    def fail(fd):
        raise OSError("storage failed")

    monkeypatch.setattr(journal_writer.os, "fsync", fail)
    with pytest.raises(OSError, match="storage failed"):
        RunJournal(first, hello=hello())
    with pytest.raises(FileExistsError):
        RunJournal(first, hello=hello())


def test_requeue_requires_new_attempt(tmp_path):
    first = boot(tmp_path)
    with pytest.raises(ValueError, match="requeue"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment | {"SLURM_RESTART_COUNT": "1"},
        )
    second = bootstrap_run(
        tmp_path,
        resolved_config=first.request_manifest.resolved_config,
        runtime=runtime(),
        environment={"SLURM_RESTART_COUNT": "1"},
    )
    assert second.attempt == 2


def test_manifest_execution_side_mapping_accepted(tmp_path):
    first = boot(tmp_path / "execution")
    manifest = replace(
        first.manifest,
        run_directory=str(tmp_path / "host"),
        storage_map=(
            StorageMapping(str(tmp_path / "host"), str(tmp_path / "execution")),
        ),
    )
    # Replace the fixture only; production manifests are never overwritten.
    first.layout.manifest.unlink()
    write_manifest(first.layout, manifest)
    result = bootstrap_run(
        tmp_path / "execution",
        resolved_config=first.request_manifest.resolved_config,
        runtime=runtime(),
        environment=first.environment,
    )
    assert result.manifest.run_directory == str(tmp_path / "host")


def test_missing_follower_identity_never_written(tmp_path):
    first = boot(tmp_path)
    path = first.layout.attempt_dir(1) / "runtime_identity.json"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment,
            is_main_process=False,
            rank=1,
        )
    assert not path.exists()


def test_manifest_and_attempt_identity_are_real_json(tmp_path):
    first = boot(tmp_path)
    manifest = json.loads(first.layout.manifest.read_bytes())
    assert manifest["runtime"]["python"] == sys.executable
    identity = json.loads(
        (first.layout.attempt_dir(1) / "runtime_identity.json").read_bytes()
    )
    assert identity["request_id"] == first.request_id


def test_configured_output_mismatch_refused_without_writes(tmp_path):
    with pytest.raises(ValueError, match="output/runtime"):
        bootstrap_run(
            tmp_path,
            resolved_config={"output_dir": str(tmp_path / "different")},
            runtime=runtime(),
            environment={},
        )
    assert not (tmp_path / "manifest.json").exists()


def test_append_storage_failure_is_visible_and_poisoned(tmp_path, monkeypatch):
    first = boot(tmp_path)
    journal = RunJournal(first, hello=hello())

    def fail(fd):
        raise OSError("storage failed")

    monkeypatch.setattr(journal_writer.os, "fsync", fail)
    with pytest.raises(OSError, match="storage failed"):
        journal.emit("state", StatePayload("running"))
    with pytest.raises(RuntimeError, match="unavailable"):
        journal.emit("state", StatePayload("running"))
    journal.close()


def test_stdlib_only_runtime_import():
    # -S bypasses editable-install .pth files, so expose only the checkout source.
    source_root = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import wavedl.journal; "
            "assert not any(name in sys.modules for name in "
            "('torch', 'numpy', 'PySide6', 'wavedl_daemon'))",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"PYTHONPATH": str(source_root)},
    )
    assert result.returncode == 0, result.stderr


def test_standalone_changed_request_has_immutable_attempt_snapshot(tmp_path):
    first = boot(tmp_path)
    original = first.layout.manifest.read_bytes()
    second = bootstrap_run(
        tmp_path,
        resolved_config={"data_path": "new.npz", "epochs": 2},
        runtime=runtime(),
        environment={},
    )
    assert second.attempt == 2
    assert second.layout.manifest.read_bytes() == original
    assert second.request_manifest.resolved_config["epochs"] == 2
    assert second.request_manifest.resolved_config["data_path"].endswith("new.npz")
    request = json.loads(
        (second.layout.attempt_dir(2) / "runtime_request.json").read_bytes()
    )
    assert request["resolved_config"] == second.request_manifest.resolved_config


def test_main_process_requires_real_boolean(tmp_path):
    with pytest.raises(ValueError, match="rank zero"):
        boot(tmp_path, is_main_process=1)
    first = boot(tmp_path)
    with pytest.raises(ValueError, match="rank zero"):
        RunJournal(first, hello=hello(), is_main_process=1)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
@pytest.mark.parametrize("metadata", ["manifest.json", "runtime_identity.json"])
def test_fifo_metadata_refused_without_blocking(tmp_path, metadata):
    first = boot(tmp_path)
    path = (
        first.layout.manifest
        if metadata == "manifest.json"
        else first.layout.attempt_dir(1) / metadata
    )
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="bounded regular"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment,
        )


def test_oversized_runtime_identity_refused(tmp_path):
    first = boot(tmp_path)
    (first.layout.attempt_dir(1) / "runtime_identity.json").write_bytes(b"x" * 4097)
    with pytest.raises(ValueError, match="bounded regular"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment,
        )


@pytest.mark.parametrize(
    "change", [{"epochs": 99}, {"data_path": "changed.npz"}, {"epochs": True}]
)
def test_injected_actual_config_difference_refused_before_writes(tmp_path, change):
    first = boot(tmp_path)
    identity_path = first.layout.attempt_dir(1) / "runtime_identity.json"
    identity_path.unlink()
    with pytest.raises(ValueError, match="config differs"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config | change,
            runtime=runtime(),
            environment=first.environment,
        )
    assert not identity_path.exists()
    assert not first.layout.journal(1).exists()


def test_injected_missing_approved_config_key_refused(tmp_path):
    first = boot(tmp_path)
    actual = dict(first.request_manifest.resolved_config)
    del actual["epochs"]
    with pytest.raises(ValueError, match=r"config differs.*epochs"):
        bootstrap_run(
            tmp_path,
            resolved_config=actual,
            runtime=runtime(),
            environment=first.environment,
        )


def test_injected_extra_cli_defaults_allowed(tmp_path):
    first = boot(tmp_path)
    result = bootstrap_run(
        tmp_path,
        resolved_config=first.request_manifest.resolved_config | {"extra_default": 10},
        runtime=runtime(),
        environment=first.environment,
        approved_defaults={"extra_default": 10},
    )
    assert result == first


@pytest.mark.parametrize(
    "extension,defaults",
    [
        ({"model": "Dangerous"}, None),
        ({"import_modules": ["dangerous.plugin"]}, None),
        ({"extra_default": 10}, None),
        ({"extra_default": 10}, {"extra_default": 9}),
        ({"extra_default": True}, {"extra_default": 1}),
    ],
)
def test_injected_unapproved_semantic_extension_refused(tmp_path, extension, defaults):
    first = boot(tmp_path)
    identity_path = first.layout.attempt_dir(1) / "runtime_identity.json"
    identity_path.unlink()
    with pytest.raises(ValueError, match="Unapproved runtime config extension"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config | extension,
            runtime=runtime(),
            environment=first.environment,
            approved_defaults=defaults,
        )
    assert not identity_path.exists()


def test_injected_actual_path_is_canonicalized(tmp_path):
    first = boot(tmp_path)
    expected = Path(first.request_manifest.resolved_config["data_path"])
    actual = first.request_manifest.resolved_config | {
        "data_path": str(expected.parent / "absent" / ".." / expected.name)
    }
    result = bootstrap_run(
        tmp_path,
        resolved_config=actual,
        runtime=runtime(),
        environment=first.environment,
    )
    assert result == first


@pytest.mark.parametrize(
    "change", [{"python": "/different/python"}, {"environment": {}}, {"device": "auto"}]
)
def test_injected_selected_runtime_difference_refused(tmp_path, change):
    first = boot(tmp_path)
    with pytest.raises(ValueError, match="Runtime selection"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=replace(runtime(), **change),
            environment=first.environment,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow flag contract")
def test_metadata_no_follow_missing_fails_closed(tmp_path, monkeypatch):
    first = boot(tmp_path)
    monkeypatch.delattr(runtime_journal.os, "O_NOFOLLOW", raising=False)
    with pytest.raises(OSError, match="no-follow/nonblocking"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment,
        )


def test_symlink_runtime_identity_refused(tmp_path):
    first = boot(tmp_path)
    identity = first.layout.attempt_dir(1) / "runtime_identity.json"
    target = tmp_path / "identity-copy.json"
    target.write_bytes(identity.read_bytes())
    identity.unlink()
    identity.symlink_to(target)
    expected = ValueError if os.name == "nt" else OSError
    with pytest.raises(expected, match="Reparse" if os.name == "nt" else None):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment,
        )


def test_dangling_attempt_request_symlink_does_not_fall_back_to_root(tmp_path):
    first = boot(tmp_path)
    request = first.layout.attempt_dir(1) / "runtime_request.json"
    request.unlink()
    request.symlink_to(tmp_path / "absent-request.json")
    expected = ValueError if os.name == "nt" else OSError
    with pytest.raises(expected, match="Reparse" if os.name == "nt" else None):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment,
        )


@pytest.mark.parametrize(
    "change", [{"attempt": True}, {"restart_count": False}, {"extra": 0}]
)
def test_runtime_identity_exact_field_types_and_keys(tmp_path, change):
    first = boot(tmp_path)
    path = first.layout.attempt_dir(1) / "runtime_identity.json"
    receipt = json.loads(path.read_bytes()) | change
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Attempt identity mismatch"):
        bootstrap_run(
            tmp_path,
            resolved_config=first.request_manifest.resolved_config,
            runtime=runtime(),
            environment=first.environment,
        )
