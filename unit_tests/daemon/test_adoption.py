import json
import os
import sys

import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.manifest import RuntimeSelection
from wavedl.journal import RunJournal, bootstrap_run
from wavedl_daemon.runs.adoption import verify_standalone_run


@pytest.fixture
def standalone(tmp_path):
    output = tmp_path / "standalone"
    output.mkdir(mode=0o700)
    bootstrap = bootstrap_run(
        output,
        resolved_config={"epochs": 1},
        runtime=RuntimeSelection(sys.executable, "selected", "cpu", {}),
        environment={},
    )
    with RunJournal(
        bootstrap,
        hello=HelloPayload(1, 1, "1.0", "3.11", "cpu", ["train"], sys.executable),
    ):
        pass
    return bootstrap


def test_actual_standalone_bootstrap_and_journal_are_adoptable(standalone):
    with verify_standalone_run(standalone.layout.run_dir) as verified:
        assert verified.manifest == standalone.manifest
        assert verified.request_manifest == standalone.request_manifest
        assert verified.request_id == standalone.request_id
        assert verified.hello.type == "hello"
        assert verified.attempt == standalone.attempt
        assert not hasattr(verified, "pid")
        assert not hasattr(verified, "backend_handle")


@pytest.mark.parametrize(
    "name",
    ["manifest.json", "runtime_request.json", "runtime_identity.json", "events.jsonl"],
)
def test_missing_actual_evidence_is_refused(standalone, name):
    parent = (
        standalone.layout.run_dir
        if name == "manifest.json"
        else standalone.layout.attempt_dir(1)
    )
    (parent / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_standalone_run(standalone.layout.run_dir)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode ownership policy")
@pytest.mark.parametrize(
    "member",
    [".", "attempts", "attempts/1", "manifest.json", "attempts/1/events.jsonl"],
)
def test_group_writable_run_and_evidence_are_refused(standalone, member):
    path = standalone.layout.run_dir / member
    path.chmod(path.stat().st_mode | 0o020)
    with pytest.raises((ValueError, PermissionError)):
        verify_standalone_run(standalone.layout.run_dir)


def test_receipt_identity_mismatch_refused(standalone):
    path = standalone.layout.attempt_dir(1) / "runtime_identity.json"
    identity = json.loads(path.read_bytes())
    identity["attempt"] = 2
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError):
        verify_standalone_run(standalone.layout.run_dir)


def test_first_hello_cannot_be_reconstructed_from_manifest(standalone):
    standalone.layout.journal(1).write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        verify_standalone_run(standalone.layout.run_dir)


def test_retained_guard_rejects_metadata_replacement(standalone):
    with verify_standalone_run(standalone.layout.run_dir) as verified:
        path = standalone.layout.manifest
        old = path.read_bytes()
        if os.name == "nt":
            with pytest.raises(PermissionError) as denied:
                path.unlink()
            assert denied.value.winerror == 32
            assert path.read_bytes() == old
            verified.guard()
            return
        path.unlink()
        path.write_bytes(old)
        with pytest.raises((ValueError, OSError)):
            verified.guard()
