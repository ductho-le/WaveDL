"""The daemon must not import the trainer package `wavedl` at runtime.

Every import that still exists is listed in ``ALLOWED`` with the exact modules
it names. A new import fails, and so does an entry that no longer matches the
source, so each cleanup must shrink this list until it is empty.
"""

import ast
from pathlib import Path


DAEMON = Path(__file__).resolve().parents[2] / "src" / "wavedl_daemon"

ALLOWED: dict[str, set[str]] = {
    "backends/local.py": {
        "wavedl.core.inference_cli",
        "wavedl.core.training_cli",
    },
    "cli.py": {
        "wavedl",
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
    },
    "client.py": {
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
    },
    "credentials.py": {"wavedl.core.safe_files"},
    "http/static.py": {"wavedl.core.safe_files"},
    "runs/catalog.py": {"wavedl.core.safe_files"},
    "runs/dataset_headers.py": {"wavedl.core.safe_files"},
    "runs/dataset_preview.py": {
        "wavedl.core.data_keys",
        "wavedl.core.safe_files",
    },
    "runs/preview_authority.py": {"wavedl.core.safe_files"},
    "runs/runtime_service.py": {"wavedl.core.safe_files"},
    "runs/service.py": {
        "wavedl.core.inference_cli",
        "wavedl.core.training_cli",
    },
    "serve_state.py": {
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
    },
}


def _trainer_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        found.update(
            name for name in names if name == "wavedl" or name.startswith("wavedl.")
        )
    return found


def test_daemon_imports_of_the_trainer_match_the_allowlist():
    web = DAEMON / "web"
    actual = {}
    for path in sorted(DAEMON.rglob("*.py")):
        if web in path.parents:
            continue
        if imports := _trainer_imports(path):
            actual[path.relative_to(DAEMON).as_posix()] = imports
    new = {
        name: sorted(imports - ALLOWED.get(name, set()))
        for name, imports in actual.items()
        if imports - ALLOWED.get(name, set())
    }
    stale = {
        name: sorted(imports - actual.get(name, set()))
        for name, imports in ALLOWED.items()
        if imports - actual.get(name, set())
    }
    assert not new, f"new trainer imports in the daemon: {new}"
    assert not stale, f"remove stale allowlist entries: {stale}"
