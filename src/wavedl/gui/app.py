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
import json
import os
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
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


try:
    import yaml as _pyyaml
except ImportError:
    _pyyaml = None


GUI_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path.cwd()
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.yaml"


def _strip_yaml_comment(line: str) -> str:
    """Strip inline YAML comments (`# ...`) outside of quotes."""
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1].isspace():
                return line[:i].rstrip()
    return line.rstrip()


def _parse_yaml_scalar(raw: str) -> Any:
    """Parse a scalar or inline list value from YAML text using stdlib only."""
    val = raw.strip()
    if val == "" or val.lower() in ("null", "~", "none"):
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
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
        return [_parse_yaml_scalar(part) for part in inner.split(",")]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def parse_yaml_text(text: str) -> dict[str, Any]:
    """Parse YAML text into a dictionary (uses PyYAML if available, else stdlib fallback)."""
    if _pyyaml is not None:
        loaded = _pyyaml.safe_load(text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML root must be a key-value mapping.")
        return loaded

    result: dict[str, Any] = {}
    current_list_key: str | None = None
    current_prefix = ""
    for raw_line in text.splitlines():
        cleaned = _strip_yaml_comment(raw_line)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        stripped = cleaned.strip()
        if stripped.startswith("- "):
            if current_list_key is not None:
                result.setdefault(current_list_key, []).append(
                    _parse_yaml_scalar(stripped[2:])
                )
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML line: {raw_line}")
        key_part, val_part = stripped.split(":", 1)
        key = key_part.strip()
        val_str = val_part.strip()
        if indent == 0:
            current_prefix = ""
        full_key = f"{current_prefix}_{key}" if current_prefix and indent > 0 else key
        if val_str == "":
            current_prefix = key
            current_list_key = full_key
        else:
            current_list_key = None
            result[full_key] = _parse_yaml_scalar(val_str)
    return result


def _dump_yaml_kv(key: str, val: Any) -> str:
    """Serialize a single key-value pair to clean YAML."""
    if isinstance(val, bool):
        return f"{key}: {'true' if val else 'false'}"
    if val is None:
        return f"{key}: null"
    if isinstance(val, (int, float)):
        return f"{key}: {val}"
    if isinstance(val, list):
        items = ", ".join(json.dumps(x) if isinstance(x, str) else str(x) for x in val)
        return f"{key}: [{items}]"
    s = str(val)
    if (
        s != ""
        and all(c.isalnum() or c in "_./-" for c in s)
        and s.lower() not in ("true", "false", "null", "none")
    ):
        return f"{key}: {s}"
    return f"{key}: {json.dumps(s)}"


def _flatten_config(
    config: dict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """Flatten nested YAML dictionaries to match WaveDL's flat config schema."""
    items: dict[str, Any] = {}
    for key, value in config.items():
        if key == "_metadata" or not isinstance(key, str):
            continue
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(_flatten_config(value, new_key, sep))
        else:
            items[new_key] = value
    return items


def format_config_yaml(cfg: dict[str, Any]) -> str:
    """Format a configuration dictionary into a clean, sectioned YAML string."""
    sections: list[tuple[str, list[str]]] = [
        ("MODEL", ["model", "pretrained", "import_modules"]),
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
                "output_dir",
                "resume",
                "save_every",
                "fresh",
                "workers",
                "seed",
                "deterministic",
                "cache_validate",
                "single_channel",
                "allow_pickle",
            ],
        ),
        ("PERFORMANCE OPTIMIZATION", ["precision", "compile"]),
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
            if val is None or val == "" or val == []:
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
        if k in used_keys or k.startswith("_") or val is None or val == "":
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


class TrainingManager:
    """Manages a single background WaveDL subprocess (train, test, or export) and captures logs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.logs: list[str] = []
        self.cmd: list[str] = []
        self.exit_code: int | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._active()

    def _active(self) -> bool:
        # exit_code is set by the stream thread only after all output is captured,
        # so the UI keeps polling until the final log lines are available.
        return self.process is not None and self.exit_code is None

    def start(self, config_path: Path, num_gpus: int | None = None) -> dict[str, Any]:
        cmd = _resolve_wavedl_command()
        if num_gpus is not None and num_gpus > 0:
            cmd.extend(["--num_gpus", str(num_gpus)])
        cmd.extend(["--config", str(config_path)])
        return self.run(cmd)

    def run(self, cmd: list[str]) -> dict[str, Any]:
        with self._lock:
            if self._active():
                raise RuntimeError("Another WaveDL job is already running.")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self.logs = [f"$ {' '.join(cmd)}", ""]
            self.cmd = cmd
            self.exit_code = None
            self.start_time = time.time()
            self.end_time = None

            popen_kwargs: dict[str, Any] = {
                "cwd": str(WORKSPACE_ROOT),
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            else:
                popen_kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )

            self.process = subprocess.Popen(cmd, **popen_kwargs)
            proc = self.process

        thread = threading.Thread(target=self._stream_output, args=(proc,), daemon=True)
        thread.start()
        return self.status(0)

    def _stream_output(self, proc: subprocess.Popen[str]) -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    clean = line.rstrip("\r\n")
                    with self._lock:
                        self.logs.append(clean)
                        if len(self.logs) > 5000:
                            self.logs = self.logs[-4000:]
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            rc = proc.wait()
            with self._lock:
                self.exit_code = rc
                self.end_time = time.time()
                self.logs.append("")
                self.logs.append(f"[Process exited with code {rc}]")

    def stop(self) -> dict[str, Any]:
        with self._lock:
            proc = self.process
            running = proc is not None and proc.poll() is None
            if running:
                self.logs.append("\n[Sending interrupt signal to running job...]")

        if not running:
            return self.status(0)

        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            else:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            proc.terminate()

        for _ in range(30):
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

        return self.status(0)

    def status(self, offset: int = 0) -> dict[str, Any]:
        with self._lock:
            running = self._active()
            pid = self.process.pid if self.process is not None else None
            offset = max(0, min(offset, len(self.logs)))
            new_logs = self.logs[offset:]
            elapsed = None
            if self.start_time is not None:
                end = self.end_time if self.end_time is not None else time.time()
                elapsed = round(end - self.start_time, 1)
            return {
                "running": running,
                "pid": pid,
                "exit_code": self.exit_code,
                "cmd": " ".join(self.cmd),
                "logs": new_logs,
                "next_offset": len(self.logs),
                "elapsed": elapsed,
            }


TRAINER = TrainingManager()


def _resolve_user_path(raw_path: str) -> Path:
    p = Path(os.path.expanduser(raw_path.strip() or "."))
    if not p.is_absolute():
        p = (WORKSPACE_ROOT / p).resolve()
    return p


class WaveDLGUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the WaveDL GUI."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            html_file = GUI_DIR / "index.html"
            content = html_file.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return

        if route == "/api/check_path":
            raw_path = query.get("path", [""])[0].strip()
            if not raw_path:
                self._send_json({"exists": False, "is_file": False, "is_dir": False})
                return
            target = _resolve_user_path(raw_path)
            size_mb = None
            if target.is_file():
                try:
                    size_mb = round(target.stat().st_size / (1024 * 1024), 2)
                except OSError:
                    size_mb = None
            self._send_json(
                {
                    "exists": target.exists(),
                    "is_file": target.is_file(),
                    "is_dir": target.is_dir(),
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
                }
            )
            return

        if route == "/api/config":
            raw_path = query.get("path", ["config.yaml"])[0]
            target = _resolve_user_path(raw_path)
            if not target.is_file():
                self._send_json(
                    {"error": f"Config file not found: {raw_path}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            try:
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
            target_dir = _resolve_user_path(raw_dir)
            if target_dir.is_file():
                target_dir = target_dir.parent
            if not target_dir.is_dir():
                target_dir = WORKSPACE_ROOT

            entries: list[dict[str, Any]] = []
            try:
                for item in sorted(
                    target_dir.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                ):
                    if item.name.startswith("."):
                        continue
                    try:
                        rel = os.path.relpath(item, WORKSPACE_ROOT)
                        if rel.startswith(".."):
                            display_path = str(item)
                        else:
                            display_path = rel
                    except ValueError:
                        display_path = str(item)
                    entries.append(
                        {
                            "name": item.name,
                            "is_dir": item.is_dir(),
                            "path": display_path,
                            "abs_path": str(item),
                        }
                    )
            except PermissionError:
                pass

            parent = str(target_dir.parent) if target_dir.parent != target_dir else None
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
            offset = int(query.get("offset", ["0"])[0])
            self._send_json(TRAINER.status(offset))
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        try:
            payload = self._read_json()
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
            target = _resolve_user_path(raw_path)

            try:
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
            target = _resolve_user_path(raw_path)

            try:
                if isinstance(raw_yaml, str) and raw_yaml.strip():
                    parse_yaml_text(raw_yaml)
                    yaml_text = raw_yaml
                elif isinstance(cfg, dict):
                    yaml_text = format_config_yaml(cfg)
                else:
                    yaml_text = None

                if yaml_text is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(yaml_text, encoding="utf-8")

                status = TRAINER.start(
                    config_path=target,
                    num_gpus=int(num_gpus) if num_gpus else None,
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
                self._send_json(TRAINER.run(cmd + TEST_MODE_ARGS[mode]))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if route == "/api/train/stop":
            self._send_json(TRAINER.stop())
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def _find_free_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main() -> int:
    global WORKSPACE_ROOT, DEFAULT_CONFIG_PATH

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
    args = parser.parse_args()

    WORKSPACE_ROOT = Path.cwd()
    DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.yaml"
    port = _find_free_port(args.host, args.port)

    server = ThreadingHTTPServer((args.host, port), WaveDLGUIHandler)
    url = f"http://{args.host}:{server.server_port}"
    print("=" * 60)
    print(f"  WaveDL GUI running at: {url}")
    print(f"  Workspace root:        {WORKSPACE_ROOT}")
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
