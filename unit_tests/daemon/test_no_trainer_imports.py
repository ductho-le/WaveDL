"""The daemon never imports the trainer package ``wavedl``; it only launches it."""

import ast
from pathlib import Path


DAEMON = Path(__file__).resolve().parents[2] / "src" / "wavedl_daemon"


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add(node.module or "")
    return found


def test_the_daemon_does_not_import_the_trainer():
    web = DAEMON / "web"
    offenders = {
        path.relative_to(DAEMON).as_posix(): sorted(trainer)
        for path in sorted(DAEMON.rglob("*.py"))
        if web not in path.parents
        and (
            trainer := {
                name
                for name in _imports(path)
                if name == "wavedl" or name.startswith("wavedl.")
            }
        )
    }
    assert not offenders
