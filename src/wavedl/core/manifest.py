"""Resolved execution plans, shared by standalone runtimes and the daemon."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from wavedl.core.inference_cli import validate_import_modules
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.persistence import write_once


PATH_FIELDS = (
    "data_path",
    "config",
    "resume",
    "checkpoint",
    "constraint_file",
    "export_path",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _bad_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


@dataclass(frozen=True)
class RuntimeSelection:
    python: str
    mode: str
    device: str
    environment: dict[str, str]
    container: str | None = None

    def __post_init__(self):
        if self.mode not in {"existing", "selected", "bundled", "container"}:
            raise ValueError("Invalid runtime mode")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("Invalid device")
        if not isinstance(self.python, str) or not self.python:
            raise ValueError("Runtime interpreter is required")
        if not isinstance(self.environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise ValueError("Runtime environment must contain strings")
        if self.container is not None and not isinstance(self.container, str):
            raise ValueError("Invalid container")


@dataclass(frozen=True)
class StorageMapping:
    host_root: str
    execution_root: str

    def __post_init__(self):
        if (
            not Path(self.host_root).is_absolute()
            or not Path(self.execution_root).is_absolute()
        ):
            raise ValueError("Storage mapping roots must be absolute")


@dataclass(frozen=True)
class Manifest:
    run_id: str
    operation: OperationKind
    resolved_config: dict[str, object]
    runtime: RuntimeSelection
    submission_directory: str
    working_directory: str
    run_directory: str
    storage_map: tuple[StorageMapping, ...]
    path_fields: tuple[str, ...]
    schema_version: int = 1
    attempt_index: str = "attempts"
    checkpoint_source: dict | None = None

    def to_dict(self) -> dict:
        """Preserve legacy canonical bytes when no explicit source is approved."""
        result = asdict(self)
        if self.checkpoint_source is None:
            result.pop("checkpoint_source")
        return result

    def __post_init__(self):
        RunLayout(Path(self.run_directory), self.run_id)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported manifest schema")
        if self.attempt_index != "attempts":
            raise ValueError("Invalid attempt index")
        if self.checkpoint_source is not None:
            from wavedl.core.checkpoint_source import validate_checkpoint_source

            validate_checkpoint_source(self.checkpoint_source)
            if self.operation == OperationKind.TRAIN:
                raise ValueError("Training cannot use an inference source pin")
        if not isinstance(self.operation, OperationKind):
            raise ValueError("Invalid operation")
        for path in (
            self.submission_directory,
            self.working_directory,
            self.run_directory,
        ):
            if not Path(path).is_absolute():
                raise ValueError("Manifest directories must be absolute")
        if not Path(self.runtime.python).is_absolute():
            raise ValueError("Runtime interpreter must be resolved")
        if not isinstance(self.resolved_config, dict):
            raise ValueError("resolved_config must be an object")
        for name in self.path_fields:
            value = self.resolved_config.get(name)
            if value is not None and (
                not isinstance(value, str) or not Path(value).is_absolute()
            ):
                raise ValueError(f"Unresolved path: {name}")
        if self.resolved_config.get("output_dir") != self.working_directory:
            raise ValueError("Output directory must belong to this run")
        _json_bytes(asdict(self))


def build_manifest(
    layout: RunLayout,
    *,
    operation: OperationKind,
    resolved_config: dict[str, object],
    runtime: RuntimeSelection,
    submission_cwd: Path,
    storage_map: tuple[StorageMapping, ...] = (),
    path_fields: tuple[str, ...] = PATH_FIELDS,
    checkpoint_source: dict | None = None,
) -> Manifest:
    """Snapshot already merged defaults/YAML/CLI values; resolve before changing cwd."""
    base = submission_cwd.resolve()
    mappings = tuple(
        sorted(
            (
                StorageMapping(str(Path(item.host_root).resolve()), item.execution_root)
                for item in storage_map
            ),
            key=lambda item: len(item.host_root),
            reverse=True,
        )
    )
    if len({item.host_root for item in mappings}) != len(mappings):
        raise ValueError("Duplicate storage mapping")

    def _mapped(host: Path) -> str:
        """Rewrite a host path into its execution-side location."""
        for mapping in mappings:
            try:
                # Persist exactly the approved root used for this rewrite.
                # Re-resolving here could change authority after a link swap.
                relative = host.relative_to(Path(mapping.host_root))
            except ValueError:
                continue
            return str(Path(mapping.execution_root) / relative)
        return str(host)

    def resolve(value: str) -> str:
        """Data paths: absolute, symlinks dereferenced, then mapped."""
        return _mapped((base / Path(value).expanduser()).resolve())

    def resolve_interpreter(value: str) -> str:
        """The interpreter is made absolute and mapped, but never dereferenced.

        A virtualenv's ``bin/python`` is a symlink to the base interpreter, so
        ``Path.resolve()`` rewrites ``<venv>/bin/python`` to something like
        ``/usr/.../python3.11``. Running that path yields a different
        ``sys.prefix`` and therefore different ``site-packages``: the planned
        environment -- torch, wavedl, every pinned dependency -- silently
        disappears, and the manifest records a runtime that was never selected.
        Lexical normalisation keeps the link intact.
        """
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        return _mapped(Path(os.path.normpath(path)))

    # File registrations have the same submission-side path semantics as data.
    # Named modules remain interpreter imports, not filesystem paths. Validate
    # before JSON copying so tuples cannot silently become approved lists.
    imports = (
        validate_import_modules(resolved_config["import_modules"])
        if "import_modules" in resolved_config
        else None
    )
    config = json.loads(_json_bytes(resolved_config))
    if imports is not None:
        config["import_modules"] = validate_import_modules(
            [resolve(item) if item.endswith(".py") else item for item in imports]
        )
    for name in path_fields:
        if config.get(name) is not None:
            if not isinstance(config[name], str) or not config[name]:
                raise ValueError(f"Invalid path: {name}")
            config[name] = resolve(config[name])
    working_directory = resolve(str(layout.run_dir))
    config["output_dir"] = working_directory
    runtime = RuntimeSelection(
        python=resolve_interpreter(runtime.python),
        mode=runtime.mode,
        device=runtime.device,
        environment=dict(runtime.environment),
        container=runtime.container,
    )
    return Manifest(
        run_id=layout.run_id,
        operation=operation,
        resolved_config=config,
        runtime=runtime,
        submission_directory=str(base),
        working_directory=working_directory,
        run_directory=str(layout.run_dir),
        storage_map=mappings,
        path_fields=tuple(path_fields),
        checkpoint_source=checkpoint_source,
    )


def describes_layout(manifest: Manifest, layout: RunLayout) -> bool:
    """Whether the manifest describes this layout, on EITHER side of a mapping.

    Under a storage map the same run has two paths: ``run_directory`` is where
    the daemon sees it, ``working_directory`` is where the compute node does.
    Code that runs on the execution side -- checkpoint publication above all --
    holds a layout rooted at the execution path, so an equality test against
    ``run_directory`` alone rejects every mapped run. Both sides are normalised
    because a mapping root and a layout root can spell the same directory
    differently (``/var`` vs ``/private/var``, ``/home`` vs an autofs target).
    """
    if manifest.run_id != layout.run_id:
        return False
    sides = {
        Path(manifest.run_directory).resolve(),
        Path(manifest.working_directory).resolve(),
    }
    return layout.run_dir.resolve() in sides


def write_manifest(layout: RunLayout, manifest: Manifest) -> None:
    if not describes_layout(manifest, layout):
        raise ValueError("Manifest/layout identity mismatch")
    Manifest(
        **asdict(manifest)
        | {
            "operation": manifest.operation,
            "runtime": manifest.runtime,
            "storage_map": manifest.storage_map,
        }
    )
    write_once(layout.manifest, _json_bytes(manifest.to_dict()))


def read_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_bytes(), parse_constant=_bad_constant)
    return Manifest(
        **(
            data
            | {
                "operation": OperationKind(data["operation"]),
                "runtime": RuntimeSelection(**data["runtime"]),
                "storage_map": tuple(
                    StorageMapping(**item) for item in data["storage_map"]
                ),
                "path_fields": tuple(data["path_fields"]),
            }
        )
    )
