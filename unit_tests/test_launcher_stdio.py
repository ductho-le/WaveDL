"""Training CLI logs remain UTF-8 when Windows redirects legacy-codepage streams."""

import ast
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl import launcher


@pytest.mark.parametrize("module", ["train", "test"])
def test_direct_runtime_main_configures_stdio_before_argument_parsing(module):
    source = (
        Path(launcher.__file__).with_name(f"{module}.py").read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    first = main.body[0]
    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name)
    assert first.value.func.id == "configure_utf8_stdio"


@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "utf-8"])
def test_cli_streams_use_utf8_without_losing_unicode(monkeypatch, encoding):
    stdout_bytes = io.BytesIO()
    stderr_bytes = io.BytesIO()
    stdout = io.TextIOWrapper(stdout_bytes, encoding=encoding)
    stderr = io.TextIOWrapper(stderr_bytes, encoding=encoding)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    launcher._configure_cli_stdio()
    print("💻 café 波", flush=True)
    print("⚠️ diagnostic", file=sys.stderr, flush=True)
    assert stdout_bytes.getvalue().decode("utf-8").splitlines() == ["💻 café 波"]
    assert stderr_bytes.getvalue().decode("utf-8").splitlines() == ["⚠️ diagnostic"]
    assert stdout.errors == stderr.errors == "backslashreplace"


def test_cli_preserves_custom_streams_and_configures_before_fast_path(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    calls = []
    monkeypatch.setattr(launcher, "_configure_cli_stdio", lambda: calls.append("stdio"))
    monkeypatch.setattr(
        launcher, "handle_fast_path_args", lambda: calls.append("fast-path") or 0
    )
    assert launcher.main() == 0
    assert calls == ["stdio", "fast-path"]
    monkeypatch.undo()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    launcher._configure_cli_stdio()
    print("💻")
    assert stream.getvalue() == "💻\n"


def test_redirected_legacy_cli_and_real_python_child_share_utf8():
    environment = dict(os.environ) | {"PYTHONIOENCODING": "cp1252:strict"}
    before = dict(environment)
    child_environment = launcher._training_stdio_environment(environment)
    assert environment == before
    assert child_environment["PYTHONIOENCODING"] == "utf-8:backslashreplace"
    script = """
import os, subprocess, sys
from wavedl import launcher
assert sys.stdout.encoding == 'cp1252'
assert sys.stdout.errors == 'strict'
launcher._configure_cli_stdio()
print('💻 café 波', flush=True)
print('⚠️ parent', file=sys.stderr, flush=True)
result = subprocess.run(
    [sys.executable, '-c', "import sys; print('✅ child café 波'); print('⚠️ child', file=sys.stderr)"],
    env=launcher._training_stdio_environment(dict(os.environ)),
    timeout=15,
)
raise SystemExit(result.returncode)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.decode("utf-8").splitlines() == [
        "💻 café 波",
        "✅ child café 波",
    ]
    assert result.stderr.decode("utf-8").splitlines() == ["⚠️ parent", "⚠️ child"]
