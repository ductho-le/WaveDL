"""The supported product ships the daemon/browser app, not the retired Qt client."""

import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_web_validation_keeps_only_daemon_client_logic():
    web = ROOT / "src/wavedl_daemon/web"
    scripts = json.loads((web / "package.json").read_text())["scripts"]
    assert "typecheck" in scripts
    assert set(scripts).isdisjoint({"test:browser", "test:generated"})
    tests = {
        path.name.removesuffix(".test.ts")
        for path in (web / "test/workstation").glob("*.test.ts")
    }
    assert tests == {
        "daemon-api",
        "catalog-api",
        "catalog-locate",
        "event-stream",
        "artifact-client",
        "artifact-download",
        "connection-session",
        "session-credential",
        "session-resume",
        "auth-routing",
        "run-monitor",
        "run-monitor-publication",
        "autoplot-controller",
        "adoption",
        "result-operations",
    }
    assert not list((web / "test/generated").glob("*.test.mjs"))


def test_distribution_declares_only_supported_packages_and_entrypoints():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert set(config["tool"]["setuptools"]["packages"]["find"]["include"]) == {
        "wavedl",
        "wavedl.*",
        "wavedl_daemon",
        "wavedl_daemon.*",
    }
    assert "gui" not in config["project"]["optional-dependencies"]
    scripts = config["project"]["scripts"]
    assert "wavedl-gui" not in scripts
    assert "wavedl-daemon" not in scripts
    assert "daemon" not in config["project"]["optional-dependencies"]
    dependencies = config["project"]["dependencies"]
    for name in ("fastapi", "pydantic", "starlette", "uvicorn", "h11"):
        assert any(item.startswith(name + ">=") for item in dependencies)
    assert scripts["wavedl"] == "wavedl.client:main"


def test_supported_runtime_has_no_desktop_imports():
    forbidden = {"wavedl_gui", "PySide6", "qfluentwidgets", "pyqtgraph"}
    for package in ("wavedl", "wavedl_daemon"):
        for path in (ROOT / "src" / package).rglob("*.py"):
            if ROOT / "src" / "wavedl_daemon" / "web" in path.parents:
                continue  # Workstation source and node_modules are not Python.
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not forbidden.intersection(
                    name.split(".")[0] for name in names
                ), path


def test_browser_class_utility_is_present_and_not_ignored(tmp_path):
    """A clean checkout must include the shared shadcn class utility."""
    utility = "src/wavedl_daemon/web/app/lib/utils.ts"
    assert (ROOT / utility).is_file()
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    subprocess.run(
        ["git", "init", "--quiet", str(tmp_path)],
        capture_output=True,
        check=True,
    )
    ignored = ["lib/package.py", "src/wavedl_daemon/web/app/lib/package.py"]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=tmp_path,
        input="\n".join([utility, *ignored]) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ignored


def test_model_catalog_loads_without_desktop_or_ml_dependencies(tmp_path):
    script = """
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])
blocked = {"wavedl_gui", "PySide6", "qfluentwidgets", "pyqtgraph", "numpy", "torch"}

class NoOptionalStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split(".")[0] not in blocked, fullname

sys.meta_path.insert(0, NoOptionalStack())
from wavedl_daemon.runs.catalog import CatalogService

service = CatalogService({})
assert service._metadata_path.parent.name == "runs"
catalog = service.models()
assert catalog["source"] == "bundled_metadata"
assert catalog["runtime_verified"] is False
assert len(catalog["models"]) == 72
assert blocked.isdisjoint(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(ROOT / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
