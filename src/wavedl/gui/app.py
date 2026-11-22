#!/usr/bin/env python3
"""
WaveDL Lightweight Web GUI
==========================

A standalone, zero-dependency local web GUI for building WaveDL YAML
configurations, launching/monitoring `wavedl-train` runs, and testing or
exporting trained models with `wavedl-test`.

Usage:
    wavedl-gui
    wavedl-gui --port 7860 --no-browser
"""

from __future__ import annotations

import argparse
import ast
import codecs
import contextlib
import datetime
import hmac
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


try:
    import yaml as _pyyaml

    class _GuiSafeLoader(_pyyaml.SafeLoader):
        """YAML SafeLoader supporting YAML 1.2 scientific notation (e.g., 1e-4)."""

    _GuiSafeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            r"""^(?:
             [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
except ImportError:
    _pyyaml = None
    _GuiSafeLoader = None  # type: ignore[assignment,misc]


GUI_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path.cwd()
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.yaml"
RESTRICT_WORKSPACE = False
SESSION_TOKEN = secrets.token_urlsafe(32)

# Mirror wavedl.utils.config._KNOWN_CONFIG_SECTIONS without importing ML packages.
_KNOWN_CONFIG_SECTIONS: tuple[str, ...] = (
    "cross_validation",
    "hyperparameters",
    "checkpointing",
    "constraints",
    "performance",
    "optimizer",
    "scheduler",
    "training",
    "logging",
    "output",
    "model",
    "wandb",
    "loss",
    "data",
    "cv",
)

_CANONICAL_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "model_kwargs",
        "pretrained",
        "import_modules",
        "batch_size",
        "lr",
        "epochs",
        "patience",
        "weight_decay",
        "grad_clip",
        "grad_accum_steps",
        "loss",
        "selection_metric",
        "huber_delta",
        "loss_weights",
        "optimizer",
        "momentum",
        "nesterov",
        "betas",
        "scheduler",
        "scheduler_factor",
        "scheduler_patience",
        "min_lr",
        "warmup_epochs",
        "step_size",
        "milestones",
        "cv",
        "cv_stratify",
        "cv_bins",
        "data_path",
        "val_data_path",
        "output_dir",
        "cache_dir",
        "resume",
        "save_every",
        "fresh",
        "workers",
        "seed",
        "deterministic",
        "cache_validate",
        "single_channel",
        "input_channels",
        "input_transform",
        "input_range",
        "target_scaling",
        "target_range",
        "allow_pickle",
        "precision",
        "compile",
        "no_tf32",
        "wandb",
        "wandb_watch",
        "project_name",
        "run_name",
        "constraint",
        "constraint_file",
        "constraint_weight",
        "constraint_reduction",
    }
)

_SECTION_ALIASES: dict[str, str] = {
    "model_name": "model",
    "loss_type": "loss",
    "loss_name": "loss",
    "optimizer_type": "optimizer",
    "optimizer_name": "optimizer",
    "scheduler_type": "scheduler",
    "scheduler_name": "scheduler",
    "mixed_precision": "precision",
    "import": "import_modules",
}

_LOSS_ALIASES: dict[str, str] = {
    "l1": "mae",
    "logcosh": "log_cosh",
}


def _strip_known_section_prefix(key: str) -> str | None:
    """Strip a recognized nested YAML section prefix from ``key`` if present."""
    for section in _KNOWN_CONFIG_SECTIONS:
        prefix = f"{section}_"
        if key.startswith(prefix) and len(key) > len(prefix):
            return key[len(prefix) :]
    return None


def _is_preserved_mapping_key(key: str) -> bool:
    """Return True for keys whose YAML mapping value must stay nested."""
    return "model_kwargs" in (key, _strip_known_section_prefix(key))


def _normalize_yaml_bool_choices(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize YAML 1.1 boolean ``no`` (parsed as ``False``) for precision keys."""
    normalized = dict(config)
    for key, value in list(normalized.items()):
        stripped = _strip_known_section_prefix(key) or key
        if stripped in {"precision", "mixed_precision"} and value is False:
            normalized[key] = "no"
    return normalized


def _strip_yaml_comment(line: str) -> str:
    """Strip inline YAML comments (`# ...`) outside of quoted strings."""
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_double:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
        elif in_single:
            if ch == "'":
                if i + 1 < n and line[i + 1] == "'":
                    i += 1  # YAML escaped single quote ('')
                else:
                    in_single = False
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "#" and (i == 0 or line[i - 1].isspace()):
                return line[:i].rstrip()
        i += 1
    return line.rstrip()


def _split_yaml_flow_items(inner: str) -> list[str]:
    """Split a YAML flow sequence body on top-level commas, respecting quotes/brackets."""
    parts: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    depth = 0
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if in_double:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
        elif in_single:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and inner[i + 1] == "'":
                    buf.append("'")
                    i += 1
                else:
                    in_single = False
        else:
            if ch == '"':
                in_double = True
                buf.append(ch)
            elif ch == "'":
                in_single = True
                buf.append(ch)
            elif ch in "[{":
                depth += 1
                buf.append(ch)
            elif ch in "]}":
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail or parts:
        parts.append(tail)
    return parts


def _parse_yaml_scalar(raw: str) -> Any:
    """Parse a scalar or inline list value from YAML text using stdlib only."""
    val = raw.strip()
    if val == "" or val.lower() in ("null", "~"):
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.startswith('"') and val.endswith('"') and len(val) >= 2:
        try:
            parsed_str = json.loads(val)
            if isinstance(parsed_str, str):
                return parsed_str
        except Exception:
            pass
        return val[1:-1]
    if val.startswith("'") and val.endswith("'") and len(val) >= 2:
        return val[1:-1].replace("''", "'")
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [_parse_yaml_scalar(part) for part in _split_yaml_flow_items(inner)]
    if val.startswith("{") and val.endswith("}"):
        inner = val[1:-1].strip()
        if not inner:
            return {}
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        mapping: dict[str, Any] = {}
        for part in _split_yaml_flow_items(inner):
            if ":" not in part:
                raise ValueError(f"Invalid YAML flow mapping entry: {part}")
            k_part, v_part = part.split(":", 1)
            mapping[str(_parse_yaml_scalar(k_part))] = _parse_yaml_scalar(v_part)
        return mapping
    try:
        return int(val)
    except ValueError:
        pass
    try:
        parsed_float = float(val)
        return parsed_float if math.isfinite(parsed_float) else None
    except ValueError:
        pass
    return val


def _sanitize_yaml_value(value: Any) -> Any:
    """Convert non-finite floats to None so JSON serialization remains valid."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_sanitize_yaml_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_yaml_value(v) for k, v in value.items()}
    return value


def _flatten_yaml_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested YAML mappings with underscore-joined keys."""
    flat: dict[str, Any] = {}
    for raw_key, value in mapping.items():
        key = f"{prefix}_{raw_key}" if prefix else str(raw_key)
        if isinstance(value, dict) and not _is_preserved_mapping_key(key):
            flat.update(_flatten_yaml_mapping(value, key))
        else:
            flat[key] = _sanitize_yaml_value(value)
    return flat


def parse_yaml_text(text: str) -> dict[str, Any]:
    """Parse YAML text into a flattened dictionary (PyYAML if available, else stdlib)."""
    if _pyyaml is not None and _GuiSafeLoader is not None:
        loaded = _pyyaml.load(text, Loader=_GuiSafeLoader)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML root must be a key-value mapping.")
        return _normalize_yaml_bool_choices(_flatten_yaml_mapping(loaded))

    result: dict[str, Any] = {}
    current_list_target: tuple[dict[str, Any], str] | None = None
    prefix_stack: list[tuple[int, str, str]] = []
    for raw_line in text.splitlines():
        cleaned = _strip_yaml_comment(raw_line)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        stripped = cleaned.strip()
        if stripped.startswith("- ") or stripped == "-":
            if current_list_target is not None:
                container, list_key = current_list_target
                item_raw = stripped[2:] if stripped.startswith("- ") else ""
                existing = container.get(list_key)
                if isinstance(existing, dict) and not existing:
                    container[list_key] = []
                container.setdefault(list_key, []).append(_parse_yaml_scalar(item_raw))
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML line: {raw_line}")
        while prefix_stack and indent <= prefix_stack[-1][0]:
            prefix_stack.pop()
        key_part, val_part = stripped.split(":", 1)
        key = key_part.strip()
        val_str = val_part.strip()
        full_key = f"{prefix_stack[-1][1]}_{key}" if prefix_stack else key
        mapping_idx = next(
            (
                idx
                for idx, (_, pkey, _) in enumerate(prefix_stack)
                if _is_preserved_mapping_key(pkey)
            ),
            None,
        )
        if mapping_idx is not None:
            root_key = prefix_stack[mapping_idx][1]
            container = result.setdefault(root_key, {})
            if not isinstance(container, dict):
                container = {}
                result[root_key] = container
            for _, _, sub_key in prefix_stack[mapping_idx + 1 :]:
                nested = container.get(sub_key)
                if not isinstance(nested, dict):
                    nested = {}
                    container[sub_key] = nested
                container = nested
            if val_str == "":
                container.setdefault(key, {})
                prefix_stack.append((indent, full_key, key))
                current_list_target = (container, key)
            else:
                current_list_target = None
                container[key] = _parse_yaml_scalar(val_str)
        elif val_str == "":
            prefix_stack.append((indent, full_key, key))
            current_list_target = (result, full_key)
        else:
            current_list_target = None
            result[full_key] = _parse_yaml_scalar(val_str)
    return _normalize_yaml_bool_choices(result)


# PyYAML follows YAML 1.1, where bare yes/no/on/off load as booleans.
YAML_RESERVED_WORDS = {
    "true",
    "false",
    "yes",
    "no",
    "on",
    "off",
    "y",
    "n",
    "null",
    "none",
    "nan",
    ".nan",
    "inf",
    ".inf",
    "+.inf",
    "-.inf",
}


def _is_safe_unquoted_yaml_str(s: str) -> bool:
    """Return True if ``s`` round-trips as an unquoted string in YAML 1.1."""
    if not s or s.lower() in YAML_RESERVED_WORDS:
        return False
    if not all(c.isalnum() or c in "_./-" for c in s):
        return False
    return s[0].isalpha() or s.startswith(("./", "../", "/"))


def _dump_yaml_scalar(val: Any) -> str:
    """Serialize a single scalar value to YAML 1.1-safe syntax."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return "null"
        return str(val)
    s = str(val)
    if _is_safe_unquoted_yaml_str(s):
        return s
    return json.dumps(s)


def _dump_yaml_kv(key: str, val: Any, indent: int = 0) -> str:
    """Serialize a single key-value pair to clean YAML."""
    pad = " " * indent
    if isinstance(val, dict):
        if not val:
            return f"{pad}{key}: {{}}"
        sub_lines = [
            _dump_yaml_kv(str(k), v, indent=indent + 2) for k, v in val.items()
        ]
        return f"{pad}{key}:\n" + "\n".join(sub_lines)
    if isinstance(val, list):
        items = ", ".join(_dump_yaml_scalar(x) for x in val)
        return f"{pad}{key}: [{items}]"
    return f"{pad}{key}: {_dump_yaml_scalar(val)}"


def _sanitize_json_value(value: Any) -> Any:
    """Normalize YAML-loaded values (dates, non-finite floats) for strict JSON."""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    return value


def _canonicalize_config_key(key: str) -> str:
    """Map known nested section keys and aliases to WaveDL's flat schema."""
    if key in _CANONICAL_CONFIG_KEYS:
        return key
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    stripped = _strip_known_section_prefix(key)
    if stripped is not None:
        if stripped in _CANONICAL_CONFIG_KEYS:
            return stripped
        if stripped in _SECTION_ALIASES:
            return _SECTION_ALIASES[stripped]
    return key


def _flatten_config(
    config: dict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """Flatten nested YAML dictionaries and canonicalize WaveDL section aliases."""
    raw_items: dict[str, Any] = {}
    for key, value in config.items():
        if key == "_metadata" or not isinstance(key, str):
            continue
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict) and not _is_preserved_mapping_key(new_key):
            raw_items.update(_flatten_config(value, new_key, sep))
        else:
            raw_items[new_key] = _sanitize_json_value(value)

    if parent_key:
        return raw_items

    normalized = _normalize_yaml_bool_choices(raw_items)
    canonical: dict[str, Any] = {}
    for key, value in normalized.items():
        canon_key = _canonicalize_config_key(key)
        if canon_key == "precision" and value is False:
            value = "no"
        elif canon_key == "loss" and isinstance(value, str):
            value = _LOSS_ALIASES.get(value.lower(), value)
        # Prefer explicit top-level canonical keys over legacy aliases if both exist
        if canon_key not in canonical or key == canon_key:
            canonical[canon_key] = value
    return canonical


def format_config_yaml(cfg: dict[str, Any]) -> str:
    """Format a configuration dictionary into a clean, sectioned YAML string."""
    sections: list[tuple[str, list[str]]] = [
        ("MODEL", ["model", "model_kwargs", "pretrained", "import_modules"]),
        (
            "TRAINING HYPERPARAMETERS",
            [
                "batch_size",
                "lr",
                "epochs",
                "patience",
                "weight_decay",
                "grad_clip",
                "grad_accum_steps",
            ],
        ),
        (
            "LOSS FUNCTION",
            ["loss", "huber_delta", "loss_weights", "selection_metric"],
        ),
        ("OPTIMIZER", ["optimizer", "momentum", "nesterov", "betas"]),
        (
            "LEARNING RATE SCHEDULER",
            [
                "scheduler",
                "scheduler_factor",
                "scheduler_patience",
                "min_lr",
                "warmup_epochs",
                "step_size",
                "milestones",
            ],
        ),
        ("CROSS-VALIDATION", ["cv", "cv_stratify", "cv_bins"]),
        (
            "DATA & OUTPUT",
            [
                "data_path",
                "val_data_path",
                "output_dir",
                "cache_dir",
                "resume",
                "save_every",
                "fresh",
                "workers",
                "seed",
                "deterministic",
                "cache_validate",
                "single_channel",
                "input_channels",
                "input_transform",
                "input_range",
                "target_scaling",
                "target_range",
                "allow_pickle",
            ],
        ),
        ("PERFORMANCE OPTIMIZATION", ["precision", "compile", "no_tf32"]),
        (
            "LOGGING & MONITORING",
            ["wandb", "wandb_watch", "project_name", "run_name"],
        ),
        (
            "PHYSICAL CONSTRAINTS",
            [
                "constraint",
                "constraint_file",
                "constraint_weight",
                "constraint_reduction",
            ],
        ),
    ]

    lines = [
        "# WaveDL Configuration",
        "# Generated by WaveDL GUI",
        "# Usage: wavedl-train --config <path/to/config.yaml>",
        "",
    ]

    used_keys: set[str] = set()
    for title, keys in sections:
        section_lines = []
        for k in keys:
            if k not in cfg:
                continue
            val = cfg[k]
            if val is None or val == "" or val == [] or val == {}:
                continue
            used_keys.add(k)
            section_lines.append(_dump_yaml_kv(k, val))
        if section_lines:
            lines.append(f"# {'=' * 75}")
            lines.append(f"# {title}")
            lines.append(f"# {'=' * 75}")
            lines.extend(section_lines)
            lines.append("")

    extra_lines = []
    for k, val in cfg.items():
        if (
            k in used_keys
            or k.startswith("_")
            or val is None
            or val == ""
            or val == []
            or val == {}
        ):
            continue
        extra_lines.append(_dump_yaml_kv(k, val))
    if extra_lines:
        lines.append(f"# {'=' * 75}")
        lines.append("# ADDITIONAL OPTIONS")
        lines.append(f"# {'=' * 75}")
        lines.extend(extra_lines)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _resolve_wavedl_command(
    script: str = "wavedl-train", module: str = "wavedl.launcher"
) -> list[str]:
    """Prefer the active Python environment's CLI, then PATH, then module launch."""
    executable = f"{script}.exe" if os.name == "nt" else script
    candidate = Path(sysconfig.get_path("scripts")) / executable
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return [str(candidate)]
    executable_on_path = shutil.which(script)
    if executable_on_path:
        return [executable_on_path]
    return [sys.executable, "-m", module]


TEST_MODE_ARGS = {
    "test": ["--plot", "--save_predictions"],
    "export": ["--export", "onnx"],
}


MAX_LOG_LINES = 5000
MAX_JSON_PAYLOAD_BYTES = 2 * 1024 * 1024


class _PayloadTooLargeError(ValueError):
    """Raised when an HTTP POST body exceeds ``MAX_JSON_PAYLOAD_BYTES``."""


STATIC_ASSETS = {
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def _read_gui_file(filename: str) -> bytes:
    """Read a GUI file from filesystem or ``importlib.resources`` fallback."""
    gui_file = GUI_DIR / filename
    if gui_file.is_file():
        return gui_file.read_bytes()
    from importlib.resources import files

    return files("wavedl.gui").joinpath(filename).read_bytes()


def _read_index_html() -> bytes:
    """Read ``index.html`` and inject the session token."""
    raw_bytes = _read_gui_file("index.html")
    token_meta = (
        f'<meta name="wavedl-token" content="{SESSION_TOKEN}" />\n</head>'.encode()
    )
    return raw_bytes.replace(b"</head>", token_meta, 1)


class TrainingManager:
    """Manages a single background WaveDL subprocess (train, test, or export) and captures logs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self.process: subprocess.Popen[bytes] | None = None
        self._stream_thread: threading.Thread | None = None
        self._job_id: int = 0
        self._logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._log_base_offset: int = 0
        self._active_progress: str = ""
        self.cmd: list[str] = []
        self.exit_code: int | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None

    @property
    def logs(self) -> list[str]:
        return list(self._logs)

    def _append_log_unlocked(self, line: str) -> None:
        if len(self._logs) == self._logs.maxlen:
            self._log_base_offset += 1
        self._logs.append(line)

    def is_running(self) -> bool:
        with self._lock:
            return self._active()

    def _active(self) -> bool:
        # exit_code is set by the stream thread only after all output is captured,
        # so the UI keeps polling until the final log lines are available.
        return self.process is not None and self.exit_code is None

    def start(
        self,
        config_path: Path,
        num_gpus: int | None = None,
        yaml_text: str | None = None,
    ) -> dict[str, Any]:
        with self._start_lock:
            if self.is_running():
                raise RuntimeError("Another WaveDL job is already running.")
            if yaml_text is not None:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml_text, encoding="utf-8")
            cmd = _resolve_wavedl_command()
            if num_gpus is not None and num_gpus > 0:
                cmd.extend(["--num_gpus", str(num_gpus)])
            cmd.extend(["--config", str(config_path)])
            return self.run(cmd)

    def run(self, cmd: list[str]) -> dict[str, Any]:
        with self._start_lock, self._lock:
            if self._active():
                raise RuntimeError("Another WaveDL job is already running.")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            popen_kwargs: dict[str, Any] = {
                "cwd": str(WORKSPACE_ROOT),
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "bufsize": 0,
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            else:
                popen_kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )

            # Publish the new job only after launch succeeds. A failed Popen must
            # preserve the previous completed/idle state so the user can retry.
            proc = subprocess.Popen(cmd, **popen_kwargs)
            self.process = proc
            self._job_id += 1
            self._logs.clear()
            self._log_base_offset = 0
            self._active_progress = ""
            self._append_log_unlocked(f"$ {' '.join(cmd)}")
            self._append_log_unlocked("")
            self.cmd = cmd
            self.exit_code = None
            self.start_time = time.time()
            self.end_time = None
            thread = threading.Thread(
                target=self._stream_output, args=(proc,), daemon=True
            )
            self._stream_thread = thread

        thread.start()
        return self.status(0)

    def _stream_output(self, proc: subprocess.Popen[bytes]) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        try:
            if proc.stdout is not None:
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    pending += decoder.decode(chunk, final=False)
                    while "\n" in pending:
                        raw_line, pending = pending.split("\n", 1)
                        # Collapse carriage-return (\r) progress updates on a single line
                        if "\r" in raw_line:
                            parts = [p for p in raw_line.split("\r") if p]
                            clean = parts[-1] if parts else ""
                        else:
                            clean = raw_line
                        with self._lock:
                            self._active_progress = ""
                            self._append_log_unlocked(clean)
                    if "\r" in pending:
                        segments = pending.split("\r")
                        pending = segments[-1]
                        latest = next((s for s in reversed(segments) if s.strip()), "")
                        with self._lock:
                            self._active_progress = latest
                tail = pending + decoder.decode(b"", final=True)
                if tail:
                    parts = [p for p in tail.split("\r") if p]
                    clean_tail = parts[-1] if parts else ""
                    if clean_tail:
                        with self._lock:
                            self._active_progress = ""
                            self._append_log_unlocked(clean_tail)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            rc = proc.wait()
            with self._lock:
                self._active_progress = ""
                self.exit_code = rc
                self.end_time = time.time()
                self._append_log_unlocked("")
                self._append_log_unlocked(f"[Process exited with code {rc}]")

    def stop(
        self,
        *,
        sigint_timeout: float = 30.0,
        sigterm_timeout: float = 5.0,
    ) -> dict[str, Any]:
        with self._stop_lock:
            with self._lock:
                proc = self.process
                stream_thread = self._stream_thread
                running = proc is not None and proc.poll() is None
                if running:
                    self._append_log_unlocked(
                        "\n[Sending interrupt signal to running job (up to 30s for graceful checkpoint save)...]"
                    )

            if not running or proc is None:
                return self.status(0)

            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                else:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                proc.terminate()

            deadline = time.monotonic() + max(0.1, sigint_timeout)
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                except Exception:
                    pass
                term_deadline = time.monotonic() + max(0.1, sigterm_timeout)
                while time.monotonic() < term_deadline:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                else:
                    try:
                        if os.name == "posix":
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        else:
                            with contextlib.suppress(Exception):
                                subprocess.run(
                                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    timeout=5,
                                    check=False,
                                )
                            proc.kill()
                    except Exception:
                        pass

            if stream_thread is not None:
                stream_thread.join(timeout=2.0)
            return self.status(0)

    def status(self, offset: int = 0) -> dict[str, Any]:
        with self._lock:
            running = self._active()
            pid = self.process.pid if self.process is not None else None
            total_seen = self._log_base_offset + len(self._logs)
            clamped = max(self._log_base_offset, min(offset, total_seen))
            rel_start = clamped - self._log_base_offset
            new_logs = list(self._logs)[rel_start:]
            elapsed = None
            if self.start_time is not None:
                end = self.end_time if self.end_time is not None else time.time()
                elapsed = round(end - self.start_time, 1)
            return {
                "job_id": self._job_id,
                "running": running,
                "pid": pid,
                "exit_code": self.exit_code,
                "cmd": " ".join(self.cmd),
                "logs": new_logs,
                "progress": self._active_progress,
                "next_offset": total_seen,
                "elapsed": elapsed,
            }


TRAINER = TrainingManager()


def _resolve_user_path(raw_path: str) -> Path:
    if "\x00" in raw_path:
        raise ValueError("Path must not contain NUL bytes.")
    p = Path(os.path.expanduser(raw_path.strip() or "."))
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    resolved = p.resolve()
    if RESTRICT_WORKSPACE:
        root_resolved = WORKSPACE_ROOT.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"Path escapes workspace root ({root_resolved}): {raw_path}"
            ) from exc
    return resolved


def _validate_yaml_config_path(raw_path: str) -> Path:
    target = _resolve_user_path(raw_path)
    if target.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Config file must use a .yaml or .yml extension.")
    return target


def _parse_int_field(
    raw: Any, name: str, *, min_value: int | None = None, max_value: int | None = None
) -> int:
    """Parse an integer JSON field while explicitly rejecting booleans."""
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{name} must be an integer, got {raw!r}.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}.")
    return value


def _validate_cli_token(value: str, name: str) -> str:
    """Reject empty, option-like (leading '-'), or control-character CLI tokens."""
    token = value.strip()
    if not token:
        raise ValueError(f"{name} entries must be non-empty.")
    if token.startswith("-"):
        raise ValueError(f"{name} entry must not start with '-': {token!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in token):
        raise ValueError(f"{name} entry contains invalid control characters.")
    return token


_MODEL_DISCOVERY_LOCK = threading.Lock()
_CACHED_REGISTERED_MODELS: list[dict[str, Any]] | None = None
_MODEL_DISCOVERY_ATTEMPTED = False


def _discover_registered_models() -> list[dict[str, Any]] | None:
    """Lazily query ``wavedl.models.registry`` if available in this environment."""
    global _CACHED_REGISTERED_MODELS, _MODEL_DISCOVERY_ATTEMPTED
    if _MODEL_DISCOVERY_ATTEMPTED:
        return _CACHED_REGISTERED_MODELS
    with _MODEL_DISCOVERY_LOCK:
        if _MODEL_DISCOVERY_ATTEMPTED:
            return _CACHED_REGISTERED_MODELS
        try:
            from wavedl.models import get_model, list_models

            discovered: list[dict[str, Any]] = []
            for name in sorted(list_models()):
                cls = get_model(name)
                supports_spatial = bool(getattr(cls, "supports_spatial_output", False))
                requires_spatial = bool(getattr(cls, "requires_spatial_output", False))
                supported_dims = getattr(cls, "supported_dims", None)
                dims: list[str] = []
                if supported_dims:
                    dims = [f"{int(d)}D" for d in supported_dims]
                if supports_spatial or requires_spatial:
                    if "2D" not in dims:
                        dims.append("2D")
                    dims.append("field")
                discovered.append(
                    {
                        "name": name,
                        "dims": dims or ["2D"],
                        "supports_spatial_output": supports_spatial,
                        "requires_spatial_output": requires_spatial,
                    }
                )
            _CACHED_REGISTERED_MODELS = discovered
        except Exception:
            _CACHED_REGISTERED_MODELS = None
        finally:
            _MODEL_DISCOVERY_ATTEMPTED = True
        return _CACHED_REGISTERED_MODELS


_VALID_PLOT_FORMATS = frozenset({"png", "pdf", "svg", "eps"})


class WaveDLGUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the WaveDL GUI."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_security_headers(self, *, is_html: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if is_html:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'",
            )

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers(is_html=False)
        self.end_headers()
        self.wfile.write(body)

    def _allowed_hosts(self) -> tuple[bool, set[str]]:
        bound = str(self.server.server_address[0]).strip("[]").lower()
        wildcard = bound in ("0.0.0.0", "::", "")
        allowed = {"127.0.0.1", "localhost", "::1"}
        if not wildcard:
            allowed.add(bound)
        return wildcard, allowed

    def _validate_request(
        self, *, require_json: bool, require_token: bool = False
    ) -> bool:
        """Protect local HTTP server against DNS rebinding, CSRF, and unauthenticated API access."""
        wildcard, allowed_hosts = self._allowed_hosts()
        loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        server_port = int(self.server.server_address[1])
        host_header = self.headers.get("Host")
        parsed_host: str | None = None
        host_port: int | None = None
        if host_header:
            try:
                host_parts = urllib.parse.urlsplit(f"//{host_header}")
                parsed_host = host_parts.hostname
                host_port = host_parts.port
            except ValueError:
                self._send_json(
                    {"error": "Malformed Host header."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return False
        if host_header and not wildcard:
            if parsed_host is None or parsed_host.lower() not in allowed_hosts:
                self._send_json(
                    {"error": "Forbidden Host header."},
                    status=HTTPStatus.FORBIDDEN,
                )
                return False

        if require_json:
            origin = self.headers.get("Origin")
            if origin:
                try:
                    origin_parts = urllib.parse.urlsplit(origin)
                    origin_host = origin_parts.hostname
                    origin_port = origin_parts.port
                except ValueError:
                    self._send_json(
                        {"error": "Cross-origin POST request blocked."},
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return False
                if origin_host is None:
                    self._send_json(
                        {"error": "Cross-origin POST request blocked."},
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return False
                origin_lc = origin_host.lower()
                if not wildcard and origin_lc not in allowed_hosts:
                    self._send_json(
                        {"error": "Cross-origin POST request blocked."},
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return False
                if origin_port is None:
                    origin_port = 443 if origin_parts.scheme == "https" else 80
                expected_port = host_port if host_port is not None else server_port
                if origin_port != expected_port:
                    self._send_json(
                        {"error": "Cross-origin POST request blocked."},
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return False
                if parsed_host is not None:
                    host_lc = parsed_host.lower()
                    both_loopback = (
                        origin_lc in loopback_hosts and host_lc in loopback_hosts
                    )
                    if not both_loopback and origin_lc != host_lc:
                        self._send_json(
                            {"error": "Cross-origin POST request blocked."},
                            status=HTTPStatus.FORBIDDEN,
                        )
                        return False

            fetch_site = self.headers.get("Sec-Fetch-Site", "").strip().lower()
            if fetch_site and fetch_site not in ("same-origin", "none"):
                self._send_json(
                    {"error": "Cross-site POST request blocked."},
                    status=HTTPStatus.FORBIDDEN,
                )
                return False

            ctype = (
                self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            )
            if ctype != "application/json":
                self._send_json(
                    {"error": "Content-Type must be application/json."},
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return False

        # Unconditionally require per-session token on all POST routes and protected API GET routes
        if require_json or require_token:
            req_token = self.headers.get("X-WaveDL-Token", "")
            if not hmac.compare_digest(req_token, SESSION_TOKEN):
                self._send_json(
                    {"error": "Invalid or missing session token."},
                    status=HTTPStatus.FORBIDDEN,
                )
                return False

        return True

    def _read_json(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        length = int(raw_len)
        if length < 0:
            raise ValueError("Negative Content-Length header.")
        if length > MAX_JSON_PAYLOAD_BYTES:
            raise _PayloadTooLargeError(
                f"Payload exceeds maximum allowed size ({MAX_JSON_PAYLOAD_BYTES} bytes)."
            )
        raw = self.rfile.read(length) if length > 0 else b"{}"
        loaded = json.loads(raw.decode("utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("JSON body must be an object.")
        return loaded

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        is_api = route.startswith("/api/")
        if not self._validate_request(require_json=False, require_token=is_api):
            return

        if route in ("/", "/index.html") or route in STATIC_ASSETS:
            if route in STATIC_ASSETS:
                filename, content_type = STATIC_ASSETS[route]
                content = _read_gui_file(filename)
            else:
                content_type = "text/html; charset=utf-8"
                content = _read_index_html()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers(is_html=route not in STATIC_ASSETS)
            self.end_headers()
            self.wfile.write(content)
            return

        if route == "/api/check_path":
            raw_path = query.get("path", [""])[0].strip()
            if not raw_path:
                self._send_json({"exists": False, "is_file": False, "is_dir": False})
                return
            try:
                target = _resolve_user_path(raw_path)
                exists = target.exists()
                is_file = target.is_file()
                is_dir = target.is_dir()
                size_mb = None
                if is_file:
                    try:
                        size_mb = round(target.stat().st_size / (1024 * 1024), 2)
                    except OSError:
                        size_mb = None
            except (ValueError, OSError) as exc:
                self._send_json(
                    {"error": f"Invalid path: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json(
                {
                    "exists": exists,
                    "is_file": is_file,
                    "is_dir": is_dir,
                    "resolved_path": str(target),
                    "size_mb": size_mb,
                }
            )
            return

        if route == "/api/info":
            rel_default = os.path.relpath(DEFAULT_CONFIG_PATH, WORKSPACE_ROOT)
            self._send_json(
                {
                    "workspace_root": str(WORKSPACE_ROOT),
                    "default_config": rel_default,
                    "has_default_config": DEFAULT_CONFIG_PATH.is_file(),
                    "restrict_workspace": RESTRICT_WORKSPACE,
                }
            )
            return

        if route == "/api/models":
            self._send_json({"models": _discover_registered_models()})
            return

        if route == "/api/config":
            raw_path = query.get("path", ["config.yaml"])[0]
            try:
                target = _validate_yaml_config_path(raw_path)
            except (ValueError, OSError) as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if not target.is_file():
                self._send_json(
                    {"error": f"Config file not found: {raw_path}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            try:
                if target.stat().st_size > MAX_JSON_PAYLOAD_BYTES:
                    self._send_json(
                        {
                            "error": f"Config file exceeds maximum size ({MAX_JSON_PAYLOAD_BYTES} bytes)."
                        },
                        status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    )
                    return
                raw_text = target.read_text(encoding="utf-8")
                loaded = parse_yaml_text(raw_text)
                flat = _flatten_config(loaded)
                self._send_json(
                    {
                        "path": raw_path,
                        "resolved_path": str(target),
                        "config": flat,
                        "raw_yaml": raw_text,
                    }
                )
            except Exception as exc:
                self._send_json(
                    {"error": f"Failed to load YAML: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return

        if route == "/api/browse":
            raw_dir = query.get("dir", [str(WORKSPACE_ROOT)])[0]
            try:
                target_dir = _resolve_user_path(raw_dir)
                if target_dir.is_file():
                    target_dir = target_dir.parent
                if not target_dir.is_dir():
                    target_dir = WORKSPACE_ROOT.resolve()
            except (ValueError, OSError) as exc:
                self._send_json(
                    {"error": f"Invalid directory path: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            entries: list[dict[str, Any]] = []
            try:
                raw_items = list(target_dir.iterdir())
            except OSError:
                raw_items = []

            classified: list[tuple[bool, str, Path]] = []
            for item in raw_items:
                if item.name.startswith("."):
                    continue
                try:
                    is_dir = item.is_dir()
                except OSError:
                    continue
                classified.append((is_dir, item.name.lower(), item))

            classified.sort(key=lambda entry: (not entry[0], entry[1]))
            for is_dir, _, item in classified:
                try:
                    rel = os.path.relpath(item, WORKSPACE_ROOT)
                    display_path = str(item) if rel.startswith("..") else rel
                except ValueError:
                    display_path = str(item)
                entries.append(
                    {
                        "name": item.name,
                        "is_dir": is_dir,
                        "path": display_path,
                        "abs_path": str(item),
                    }
                )

            parent: str | None = None
            if target_dir.parent != target_dir:
                if not RESTRICT_WORKSPACE:
                    parent = str(target_dir.parent)
                else:
                    try:
                        target_dir.parent.relative_to(WORKSPACE_ROOT.resolve())
                        parent = str(target_dir.parent)
                    except ValueError:
                        parent = None

            self._send_json(
                {
                    "current_dir": str(target_dir),
                    "parent_dir": parent,
                    "workspace_root": str(WORKSPACE_ROOT),
                    "entries": entries,
                }
            )
            return

        if route == "/api/train/status":
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                self._send_json(
                    {"error": "Invalid offset query parameter."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json(TRAINER.status(offset))
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._validate_request(require_json=True):
            return
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        try:
            payload = self._read_json()
        except _PayloadTooLargeError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return
        except Exception as exc:
            self._send_json(
                {"error": f"Invalid JSON body: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if route == "/api/config/parse":
            raw_yaml = str(payload.get("raw_yaml") or "")
            try:
                loaded = parse_yaml_text(raw_yaml)
                self._send_json({"config": _flatten_config(loaded)})
            except Exception as exc:
                self._send_json(
                    {"error": f"Invalid YAML: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return

        if route == "/api/config":
            raw_path = str(payload.get("path") or "config.yaml").strip()
            cfg = payload.get("config")
            raw_yaml = payload.get("raw_yaml")

            try:
                target = _validate_yaml_config_path(raw_path)
                if isinstance(raw_yaml, str) and raw_yaml.strip():
                    parse_yaml_text(raw_yaml)
                    yaml_text = raw_yaml
                elif isinstance(cfg, dict):
                    yaml_text = format_config_yaml(cfg)
                else:
                    raise ValueError("Either 'config' or 'raw_yaml' must be provided.")

                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(yaml_text, encoding="utf-8")
                self._send_json(
                    {
                        "ok": True,
                        "path": raw_path,
                        "resolved_path": str(target),
                        "yaml": yaml_text,
                    }
                )
            except Exception as exc:
                self._send_json(
                    {"error": f"Failed to save config: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return

        if route == "/api/train/start":
            raw_path = str(payload.get("path") or "config.yaml").strip()
            cfg = payload.get("config")
            raw_yaml = payload.get("raw_yaml")
            num_gpus = payload.get("num_gpus")

            try:
                target = _validate_yaml_config_path(raw_path)
                if TRAINER.is_running():
                    raise RuntimeError("Another WaveDL job is already running.")

                parsed_gpus: int | None = None
                if num_gpus is not None and str(num_gpus).strip() != "":
                    gpu_val = _parse_int_field(num_gpus, "num_gpus", min_value=0)
                    if gpu_val > 0:
                        parsed_gpus = gpu_val

                if "raw_yaml" in payload and raw_yaml is not None:
                    if not isinstance(raw_yaml, str) or not raw_yaml.strip():
                        raise ValueError("raw_yaml must be a non-empty YAML document.")
                    parse_yaml_text(raw_yaml)
                    yaml_text = raw_yaml
                elif isinstance(cfg, dict):
                    yaml_text = format_config_yaml(cfg)
                else:
                    yaml_text = None

                status = TRAINER.start(
                    config_path=target,
                    num_gpus=parsed_gpus,
                    yaml_text=yaml_text,
                )
                self._send_json(status)
            except Exception as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return

        if route == "/api/test/start":
            mode = payload.get("mode")
            fields = {
                "checkpoint": (
                    "Checkpoint folder",
                    "Train a model first, or browse to an existing best_checkpoint.",
                ),
                "data_path": ("Test data file", "Check the path or use Browse."),
            }
            try:
                if mode not in TEST_MODE_ARGS:
                    raise ValueError(f"Unknown mode: {mode!r}")
                cmd = _resolve_wavedl_command("wavedl-test", "wavedl.test")
                for key, (label, hint) in fields.items():
                    raw = str(payload.get(key) or "").strip()
                    if not raw:
                        raise ValueError(f"{label} is required.")
                    path = _resolve_user_path(raw)
                    if not path.exists():
                        raise ValueError(f"{label} not found: {raw}. {hint}")
                    cmd += [f"--{key}", str(path)]
                output_dir = str(payload.get("output_dir") or "").strip()
                cmd += [
                    "--output_dir",
                    str(_resolve_user_path(output_dir or "test_results")),
                ]
                import_modules = payload.get("import_modules")
                if isinstance(import_modules, list):
                    clean_mods = [
                        _validate_cli_token(str(m), "import_modules")
                        for m in import_modules
                        if str(m).strip()
                    ]
                    if clean_mods:
                        cmd += ["--import", *clean_mods]
                if payload.get("allow_pickle") is True:
                    cmd.append("--allow_pickle")
                if payload.get("no_tf32") is True:
                    cmd.append("--no_tf32")
                if payload.get("no_denorm") is True:
                    cmd.append("--no_denorm")
                raw_ic = payload.get("input_channels")
                if raw_ic is not None and str(raw_ic).strip() != "":
                    ic = _parse_int_field(raw_ic, "input_channels", min_value=1)
                    cmd += ["--input_channels", str(ic)]
                raw_bs = payload.get("batch_size")
                if raw_bs is not None and str(raw_bs).strip() != "":
                    bs = _parse_int_field(raw_bs, "batch_size", min_value=1)
                    cmd += ["--batch_size", str(bs)]
                raw_opset = payload.get("export_opset")
                if raw_opset is not None and str(raw_opset).strip() != "":
                    opset = _parse_int_field(
                        raw_opset, "export_opset", min_value=11, max_value=20
                    )
                    cmd += ["--export_opset", str(opset)]
                plot_format = str(payload.get("plot_format") or "").strip().lower()
                if plot_format:
                    if plot_format not in _VALID_PLOT_FORMATS:
                        raise ValueError(
                            f"plot_format must be one of {sorted(_VALID_PLOT_FORMATS)}."
                        )
                    cmd += ["--plot_format", plot_format]
                for opt_key in ("input_key", "output_key"):
                    opt_val = str(payload.get(opt_key) or "").strip()
                    if opt_val:
                        cmd += [
                            f"--{opt_key}",
                            _validate_cli_token(opt_val, opt_key),
                        ]
                param_names = payload.get("param_names")
                if isinstance(param_names, list):
                    clean_params = [
                        _validate_cli_token(str(p), "param_names")
                        for p in param_names
                        if str(p).strip()
                    ]
                    if clean_params:
                        cmd += ["--param_names", *clean_params]
                self._send_json(TRAINER.run(cmd + TEST_MODE_ARGS[mode]))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if route == "/api/train/stop":
            self._send_json(TRAINER.stop())
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def _socket_family_for_host(host: str) -> socket.AddressFamily:
    clean_host = host.strip("[]")
    if ":" in clean_host:
        return socket.AF_INET6
    try:
        infos = socket.getaddrinfo(clean_host, None, type=socket.SOCK_STREAM)
        if infos:
            return infos[0][0]
    except OSError:
        pass
    return socket.AF_INET


def _find_free_port(host: str, preferred_port: int) -> int:
    clean_host = host.strip("[]")
    family = _socket_family_for_host(clean_host)
    for port in range(preferred_port, preferred_port + 50):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((clean_host, port))
                return port
            except OSError:
                continue
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((clean_host, 0))
        return int(sock.getsockname()[1])


def main() -> int:
    global WORKSPACE_ROOT, DEFAULT_CONFIG_PATH, RESTRICT_WORKSPACE

    parser = argparse.ArgumentParser(
        description="WaveDL Lightweight Web GUI (Config Builder, Training, Test & Export)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Preferred port to bind (default: 7860)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web browser on launch",
    )
    parser.add_argument(
        "--restrict-workspace",
        action="store_true",
        help="Restrict file browsing and YAML reads/writes to the launch workspace directory",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "Explicitly acknowledge binding to a non-loopback --host interface "
            "(exposes YAML editing and job launching without network authentication; "
            "prefer SSH port forwarding with default --host 127.0.0.1)"
        ),
    )
    args = parser.parse_args()

    WORKSPACE_ROOT = Path.cwd()
    DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.yaml"
    RESTRICT_WORKSPACE = bool(args.restrict_workspace)
    bind_host = args.host.strip("[]")
    is_loopback = bind_host.lower() in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and not args.allow_remote:
        parser.error(
            f"Refusing to bind non-loopback host {args.host!r} without --allow-remote. "
            "WaveDL Studio's session token protects against cross-site requests only; "
            "any client that can reach a non-loopback listener can read/write YAML "
            "configs and launch jobs. Prefer SSH port forwarding with default "
            "--host 127.0.0.1, or pass --allow-remote to acknowledge."
        )
    family = _socket_family_for_host(bind_host)
    port = _find_free_port(bind_host, args.port)

    class _BoundThreadingHTTPServer(ThreadingHTTPServer):
        address_family = family

    server = _BoundThreadingHTTPServer((bind_host, port), WaveDLGUIHandler)
    if bind_host == "0.0.0.0":
        browser_host = "127.0.0.1"
    elif bind_host == "::":
        browser_host = "[::1]"
    elif ":" in bind_host:
        browser_host = f"[{bind_host}]"
    else:
        browser_host = bind_host
    url = f"http://{browser_host}:{server.server_port}"
    print("=" * 60)
    print(f"  WaveDL GUI running at: {url}")
    print(f"  Workspace root:        {WORKSPACE_ROOT}")
    if RESTRICT_WORKSPACE:
        print("  Workspace confinement: ENABLED (--restrict-workspace)")
    if bind_host.lower() not in ("127.0.0.1", "localhost", "::1"):
        print(
            "  WARNING: Bound to a non-loopback interface. Anyone who can reach\n"
            "           this port can read/write YAML configs and launch jobs.\n"
            "           Prefer SSH port forwarding with default --host 127.0.0.1."
        )
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    if not args.no_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down WaveDL GUI...")
    finally:
        if TRAINER.is_running():
            TRAINER.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
