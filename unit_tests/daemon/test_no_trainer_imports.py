"""The daemon must not import the trainer package `wavedl` at runtime.

Every import that still exists is listed in ``ALLOWED`` with the exact modules
it names. A new import fails, and so does an entry that no longer matches the
source, so each cleanup must shrink this list until it is empty.
"""

import ast
from pathlib import Path


DAEMON = Path(__file__).resolve().parents[2] / "src" / "wavedl_daemon"

ALLOWED: dict[str, set[str]] = {
    "backends/contracts.py": {
        "wavedl.core.layout",
        "wavedl.core.manifest",
        "wavedl.core.runtime_identity",
        "wavedl.core.scheduler_attempt",
    },
    "backends/local.py": {
        "wavedl.core.inference_cli",
        "wavedl.core.layout",
        "wavedl.core.manifest",
        "wavedl.core.training_cli",
    },
    "backends/local_execution.py": {
        "wavedl.core.layout",
        "wavedl.core.local_attempt",
        "wavedl.core.persistence",
        "wavedl.core.runtime_identity",
        "wavedl.core.submission",
    },
    "backends/slurm.py": {
        "wavedl.core.manifest",
        "wavedl.core.scheduler_attempt",
    },
    "backends/slurm_observation.py": {
        "wavedl.core.layout",
        "wavedl.core.outcomes",
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
        "wavedl.core.scheduler_attempt",
        "wavedl.core.submission",
        "wavedl.core.submission_outcomes",
    },
    "backends/slurm_plan.py": {
        "wavedl.core.layout",
        "wavedl.core.manifest",
        "wavedl.core.scheduler_attempt",
    },
    "backends/slurm_submission.py": {
        "wavedl.core.layout",
        "wavedl.core.manifest",
        "wavedl.core.persistence",
        "wavedl.core.safe_files",
        "wavedl.core.submission",
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
    "events/ingestion.py": {
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
    },
    "events/journal_reader.py": {
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
    },
    "events/streaming.py": {"wavedl.core.protocol"},
    "execution/service_host.py": {
        "wavedl.core.checkpoint_source",
        "wavedl.core.layout",
    },
    "http/static.py": {"wavedl.core.safe_files"},
    "runs/artifacts.py": {
        "wavedl.core.generations",
        "wavedl.core.layout",
        "wavedl.core.safe_files",
    },
    "runs/catalog.py": {"wavedl.core.safe_files"},
    "runs/control/derivation.py": {"wavedl.core.protocol"},
    "runs/control/dispatch.py": {
        "wavedl.core.layout",
        "wavedl.core.manifest",
    },
    "runs/control/handshake.py": {
        "wavedl.core.layout",
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
    },
    "runs/control/records.py": {"wavedl.core.safe_files"},
    "runs/control/restarts.py": {"wavedl.core.scheduler_attempt"},
    "runs/controller.py": {"wavedl.core.protocol"},
    "runs/dataset_headers.py": {"wavedl.core.safe_files"},
    "runs/dataset_preview.py": {
        "wavedl.core.data_keys",
        "wavedl.core.safe_files",
    },
    "runs/negotiation.py": {
        "wavedl.core.events",
        "wavedl.core.protocol",
    },
    "runs/preview_authority.py": {"wavedl.core.safe_files"},
    "runs/reconciliation.py": {"wavedl.core.scheduler_attempt"},
    "runs/reducer.py": {
        "wavedl.core.events",
        "wavedl.core.layout",
        "wavedl.core.protocol",
    },
    "runs/runtime_service.py": {
        "wavedl.core.layout",
        "wavedl.core.safe_files",
    },
    "runs/service.py": {
        "wavedl.core.checkpoint_source",
        "wavedl.core.inference_cli",
        "wavedl.core.layout",
        "wavedl.core.manifest",
        "wavedl.core.training_cli",
    },
    "serve_state.py": {
        "wavedl.core.protocol",
        "wavedl.core.safe_files",
    },
    "storage/collections.py": {
        "wavedl.core.layout",
        "wavedl.core.protocol",
    },
    "storage/event_log.py": {"wavedl.core.protocol"},
    "storage/executions.py": {
        "wavedl.core.local_attempt",
        "wavedl.core.scheduler_attempt",
    },
    "storage/metric_history.py": {
        "wavedl.core.events",
        "wavedl.core.protocol",
    },
    "storage/metric_projection.py": {
        "wavedl.core.events",
        "wavedl.core.protocol",
    },
    "storage/native_heads.py": {"wavedl.core.scheduler_attempt"},
    "storage/native_lifecycle.py": {"wavedl.core.scheduler_attempt"},
    "storage/registry.py": {
        "wavedl.core.generations",
        "wavedl.core.layout",
        "wavedl.core.manifest",
        "wavedl.core.persistence",
        "wavedl.core.protocol",
    },
    "storage/runtime_provenance.py": {
        "wavedl.core.events",
        "wavedl.core.protocol",
        "wavedl.core.runtime_identity",
    },
    "storage/schema.py": {"wavedl.core.persistence"},
    "storage/submission_generations.py": {"wavedl.core.scheduler_attempt"},
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
