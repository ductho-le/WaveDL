"""
Unit Tests for UTF-8 CLI Output
===============================

WaveDL CLIs print Unicode (emoji status markers, ``R²``). A Windows console
(cp1252) or a pipe under a non-UTF-8 locale would otherwise raise
``UnicodeEncodeError`` and abort training or inference.

**Tested Components**:
    - utils/stdio.py: configure_utf8_stdio, utf8_stdio_environment
    - Entry points: train.main, test.main, hpo.main, launcher.main

Author: Ductho Le (ductho.le@outlook.com)
"""

import io
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wavedl.utils.stdio import configure_utf8_stdio, utf8_stdio_environment


class TestConfigureUtf8Stdio:
    """Tests for in-process stream reconfiguration."""

    def test_legacy_streams_write_utf8(self, monkeypatch):
        """Unicode output must not raise on a Windows cp1252 console."""
        out_bytes, err_bytes = io.BytesIO(), io.BytesIO()
        stdout = io.TextIOWrapper(out_bytes, encoding="cp1252")
        stderr = io.TextIOWrapper(err_bytes, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)

        configure_utf8_stdio()
        print("🚀 R² ✔", flush=True)
        print("⚠️ warning", file=sys.stderr, flush=True)

        assert out_bytes.getvalue().decode("utf-8") == "🚀 R² ✔\n"
        assert err_bytes.getvalue().decode("utf-8") == "⚠️ warning\n"
        assert stdout.errors == stderr.errors == "backslashreplace"

    def test_ascii_child_process_prints_unicode(self):
        """A real interpreter with an ASCII-only stdout prints UTF-8."""
        script = (
            "from wavedl.utils.stdio import configure_utf8_stdio\n"
            "configure_utf8_stdio()\n"
            "print('🚀 R² ✔')\n"
        )
        env = dict(os.environ, PYTHONIOENCODING="ascii:strict")
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            timeout=60,
            check=True,
        )
        assert result.stdout.decode("utf-8").strip() == "🚀 R² ✔"


class TestUtf8StdioEnvironment:
    """Tests for the child-process environment helper."""

    def test_returns_copy_with_utf8_io_encoding(self):
        """The input mapping is not mutated; the copy forces UTF-8."""
        environment = {"PATH": "/bin", "PYTHONIOENCODING": "cp1252:strict"}
        child = utf8_stdio_environment(environment)

        assert environment["PYTHONIOENCODING"] == "cp1252:strict"
        assert child == {"PATH": "/bin", "PYTHONIOENCODING": "utf-8:backslashreplace"}


class TestEntryPointsConfigureStdio:
    """Each CLI entry point configures UTF-8 output before parsing arguments."""

    @pytest.mark.parametrize(
        ("module_name", "prog"),
        [
            ("wavedl.train", "wavedl-train"),
            ("wavedl.launcher", "wavedl-train"),
        ],
    )
    def test_main_configures_stdio_before_parsing(
        self, monkeypatch, capsys, module_name, prog
    ):
        """--help exits during parsing, after stdio was configured."""
        import importlib

        module = importlib.import_module(module_name)
        calls = []
        monkeypatch.setattr(module, "configure_utf8_stdio", lambda: calls.append(1))
        monkeypatch.setattr(sys, "argv", [prog, "--help"])

        with pytest.raises(SystemExit) as excinfo:
            module.main()

        assert excinfo.value.code == 0
        assert calls == [1]

    def test_launcher_passes_utf8_environment_to_training(self, monkeypatch):
        """The accelerate/training child inherits a UTF-8 PYTHONIOENCODING."""
        from wavedl import launcher

        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(launcher, "configure_utf8_stdio", lambda: None)
        monkeypatch.setattr(launcher, "setup_environment", lambda: None)
        monkeypatch.setattr(launcher, "print_summary", lambda *args: None)
        monkeypatch.setattr(launcher.subprocess, "run", fake_run)
        monkeypatch.setattr(
            sys,
            "argv",
            ["wavedl-train", "--num_gpus", "1", "--model", "cnn", "--data_path", "x"],
        )

        assert launcher.main() == 0
        assert captured["env"]["PYTHONIOENCODING"] == "utf-8:backslashreplace"
        assert captured["env"]["PATH"] == os.environ["PATH"]
