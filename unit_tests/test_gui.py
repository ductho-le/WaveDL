"""Exercise the standalone GUI's workspace and subprocess boundary without ML imports.

Can also run with stdlib only: python -m unittest discover -s unit_tests -p test_gui.py
"""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


APP_PATH = Path(__file__).resolve().parents[1] / "src/wavedl/gui/app.py"
spec = importlib.util.spec_from_file_location("standalone_gui", APP_PATH)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class TestGUIWorkspace(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="wavedl-gui-")
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name).resolve()
        self.trainer = app.TrainingManager()
        for name, value in (
            ("WORKSPACE_ROOT", self.root),
            ("DEFAULT_CONFIG_PATH", self.root / "config.yaml"),
            ("TRAINER", self.trainer),
        ):
            patcher = patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), app.WaveDLGUIHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.trainer.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, path, payload=None, extra_headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}{path}",
            data=data,
            headers=headers,
        )
        with urlopen(request, timeout=5) as response:
            return response.read()

    def test_html_browse_and_default_config_use_workspace(self):
        (self.root / "dataset.npz").touch()
        html = self.request("/").decode()
        self.assertIn('value="config.yaml"', html)
        info = json.loads(self.request("/api/info"))
        self.assertEqual(info["workspace_root"], str(self.root))
        self.assertEqual(info["default_config"], "config.yaml")
        self.assertFalse(info["has_default_config"])
        listing = json.loads(self.request("/api/browse"))
        self.assertEqual(listing["current_dir"], str(self.root))
        self.assertEqual(listing["entries"][0]["path"], "dataset.npz")
        result = json.loads(
            self.request(
                "/api/config",
                {"config": {"model": "cnn", "epochs": 2, "precision": "no"}},
            )
        )
        self.assertEqual(result["resolved_path"], str(self.root / "config.yaml"))
        self.assertIn("model: cnn", (self.root / "config.yaml").read_text())
        loaded = json.loads(self.request("/api/config"))
        self.assertEqual(
            loaded["config"], {"model": "cnn", "epochs": 2, "precision": "no"}
        )

    def test_training_inherits_workspace_and_environment(self):
        # A real child process records its launch context in place of costly training.
        child = (
            "import json, os, pathlib, sys; "
            "pathlib.Path('launch.json').write_text(json.dumps({"
            "'cwd': os.getcwd(), 'args': sys.argv[1:], "
            "'pythonpath': os.environ.get('PYTHONPATH')})); "
            "print('training completed')"
        )
        with (
            patch.object(
                app,
                "_resolve_wavedl_command",
                return_value=[sys.executable, "-c", child],
            ),
            patch.dict(os.environ, {"PYTHONPATH": "user-supplied-path"}),
        ):
            self.request(
                "/api/train/start",
                {"config": {"model": "cnn"}, "num_gpus": 2},
            )
        process = self.trainer.process
        try:
            self.assertEqual(process.wait(timeout=10), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        context = json.loads((self.root / "launch.json").read_text())
        self.assertEqual(Path(context["cwd"]), self.root)
        self.assertEqual(context["pythonpath"], "user-supplied-path")
        self.assertEqual(
            context["args"],
            ["--num_gpus", "2", "--config", str(self.root / "config.yaml")],
        )
        self.assertTrue((self.root / "config.yaml").is_file())

    def test_export_launches_wavedl_test_and_rejects_missing_checkpoint(self):
        (self.root / "run" / "best_checkpoint").mkdir(parents=True)
        (self.root / "test.npz").touch()
        child = (
            "import json, pathlib, sys; "
            "pathlib.Path('args.json').write_text(json.dumps(sys.argv[1:]))"
        )
        payload = {"mode": "export", "checkpoint": "missing", "data_path": "test.npz"}
        with self.assertRaises(HTTPError) as err:
            self.request("/api/test/start", payload)
        self.assertEqual(err.exception.code, 400)
        self.assertIn("Checkpoint folder not found", err.exception.read().decode())
        payload["checkpoint"] = "run/best_checkpoint"
        with patch.object(
            app, "_resolve_wavedl_command", return_value=[sys.executable, "-c", child]
        ) as resolve:
            self.request("/api/test/start", payload)
        resolve.assert_called_once_with("wavedl-test", "wavedl.test")
        self.assertEqual(self.trainer.process.wait(timeout=10), 0)
        self.assertEqual(
            json.loads((self.root / "args.json").read_text()),
            [
                "--checkpoint",
                str(self.root / "run" / "best_checkpoint"),
                "--data_path",
                str(self.root / "test.npz"),
                "--output_dir",
                str(self.root / "test_results"),
                "--export",
                "onnx",
            ],
        )

    def test_failed_launch_preserves_state_and_allows_retry(self):
        finished = threading.Event()
        stream_output = self.trainer._stream_output

        def capture_output(process):
            try:
                stream_output(process)
            finally:
                finished.set()

        with patch.object(self.trainer, "_stream_output", side_effect=capture_output):
            # Cover both a failed first launch and failure after a completed job.
            for previous_job in (False, True):
                with self.subTest(previous_job=previous_job):
                    before = self.trainer.status()
                    with (
                        patch.object(
                            app,
                            "_resolve_wavedl_command",
                            return_value=[str(self.root / "missing-executable")],
                        ),
                        self.assertRaises(HTTPError) as error,
                    ):
                        self.request("/api/train/start", {"config": {"model": "cnn"}})
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
                    status = json.loads(self.request("/api/train/status"))
                    self.assertFalse(status["running"])
                    for key in ("pid", "exit_code", "cmd", "logs", "next_offset"):
                        self.assertEqual(status[key], before[key])
                    self.assertFalse(
                        json.loads(self.request("/api/train/stop", {}))["running"]
                    )

                    finished.clear()
                    with patch.object(
                        app,
                        "_resolve_wavedl_command",
                        return_value=[sys.executable, "-c", "print('retry completed')"],
                    ):
                        self.request("/api/train/start", {"config": {"model": "cnn"}})
                    process = self.trainer.process
                    try:
                        self.assertTrue(finished.wait(timeout=10))
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=5)
                            finished.wait(timeout=5)
                    status = json.loads(self.request("/api/train/status"))
                    self.assertFalse(status["running"])
                    self.assertEqual(status["exit_code"], 0)
                    self.assertIn("retry completed", status["logs"])
                    self.assertEqual(status["logs"][-1], "[Process exited with code 0]")

    def test_stop_when_idle_returns_without_deadlocking(self):
        status = json.loads(self.request("/api/train/stop", {}))
        self.assertFalse(status["running"])

    def test_active_environment_command_precedes_path(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        executable = scripts / (
            "wavedl-train.exe" if os.name == "nt" else "wavedl-train"
        )
        executable.touch(mode=0o755)
        with (
            patch.object(app.sysconfig, "get_path", return_value=str(scripts)),
            patch.object(
                app.shutil, "which", return_value="other/wavedl-train"
            ) as which,
        ):
            self.assertEqual(app._resolve_wavedl_command(), [str(executable)])
            which.assert_not_called()
        executable.unlink()
        with (
            patch.object(app.sysconfig, "get_path", return_value=str(scripts)),
            patch.object(app.shutil, "which", return_value=None),
        ):
            self.assertEqual(
                app._resolve_wavedl_command(), [sys.executable, "-m", "wavedl.launcher"]
            )

    def test_browser_csrf_and_origin_enforcement(self):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        # Browser-originated POST without X-WaveDL-Token must be rejected with 403
        with self.assertRaises(HTTPError) as err:
            self.request(
                "/api/config",
                {"config": {"model": "cnn"}},
                extra_headers={"Origin": origin},
            )
        self.assertEqual(err.exception.code, 403)
        err.exception.close()

        # Browser-originated POST with valid X-WaveDL-Token succeeds
        res = json.loads(
            self.request(
                "/api/config",
                {"config": {"model": "cnn"}},
                extra_headers={
                    "Origin": origin,
                    "X-WaveDL-Token": app.SESSION_TOKEN,
                },
            )
        )
        self.assertTrue(res["ok"])

        # Cross-origin POST must be rejected even with token
        with self.assertRaises(HTTPError) as cross_err:
            self.request(
                "/api/config",
                {"config": {"model": "cnn"}},
                extra_headers={
                    "Origin": "http://evil.example:9999",
                    "X-WaveDL-Token": app.SESSION_TOKEN,
                },
            )
        self.assertEqual(cross_err.exception.code, 403)
        cross_err.exception.close()

    def test_config_rejects_non_yaml_extension(self):
        with self.assertRaises(HTTPError) as err:
            self.request("/api/config?path=notes.txt")
        self.assertEqual(err.exception.code, 400)
        err.exception.close()

    def test_stdlib_yaml_scientific_nested_and_dump_scalars(self):
        sample_yaml = (
            "model: cnn\n"
            "lr: 1e-4\n"
            "nested:\n"
            "  alpha: 2.5e-3\n"
            "  enabled: true\n"
            "items:\n"
            "  - true\n"
            "  - 10\n"
        )
        with patch.object(app, "_pyyaml", None):
            parsed = app.parse_yaml_text(sample_yaml)
            self.assertAlmostEqual(parsed["lr"], 1e-4)
            self.assertAlmostEqual(parsed["nested_alpha"], 2.5e-3)
            self.assertIs(parsed["nested_enabled"], True)
            self.assertEqual(parsed["items"], [True, 10])

        if app._pyyaml is not None:
            self.assertEqual(app.parse_yaml_text(sample_yaml), parsed)

        dumped = app._dump_yaml_kv("flags", [True, None, "yes"])
        self.assertEqual(dumped, 'flags: [true, null, "yes"]')


if __name__ == "__main__":
    unittest.main()
