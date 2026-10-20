"""Launcher regressions for the real distributed smoke check."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_local_check_configures_rendezvous_before_launch(
    monkeypatch, tmp_path, platform
):
    helpers = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/check_distributed.py")
    )
    monkeypatch.setattr(helpers["sys"], "platform", platform)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.setattr(
        helpers["sys"],
        "argv",
        ["check_distributed", "--work-dir", str(tmp_path / "work")],
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(helpers["subprocess"], "run", run)
    assert helpers["main"]() == 7
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "--rdzv-backend=static" in command
    assert "--master-addr=127.0.0.1" in command
    assert "--standalone" not in command
    assert "--nproc_per_node=2" in command
    port = next(
        arg.split("=", 1)[1] for arg in command if arg.startswith("--master-port=")
    )
    assert 0 < int(port) <= 65535
    assert kwargs["env"]["ACCELERATE_USE_CPU"] == "true"
    assert kwargs["env"].get("GLOO_SOCKET_IFNAME") == (
        "lo0" if platform == "darwin" else None
    )
