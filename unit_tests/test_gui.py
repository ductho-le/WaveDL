"""Exercise the standalone GUI's workspace and subprocess boundary without ML imports.

Can also run with stdlib only: python -m unittest discover -s unit_tests -p test_gui.py
"""

import importlib.util
import json
import os
import re
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

    def request(self, path, payload=None, extra_headers=None, *, include_token=True):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if include_token and path.startswith("/api/"):
            headers["X-WaveDL-Token"] = app.SESSION_TOKEN
        if extra_headers:
            headers.update(extra_headers)
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}{path}",
            data=data,
            headers=headers,
        )
        with urlopen(request, timeout=5) as response:
            return response.read()

    def test_static_assets(self):
        html = self.request("/").decode()
        self.assertIn('<link rel="stylesheet" href="/style.css">', html)
        self.assertIn('<script src="/app.js"></script>', html)
        self.assertNotRegex(html, r"(?i)<style\b")
        self.assertEqual(len(re.findall(r"(?i)<script\b", html)), 2)
        inline_scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
        self.assertEqual(len(inline_scripts), 1)
        self.assertIn('localStorage.getItem("wavedl-theme")', inline_scripts[0])

        for path, content_type in (
            ("/style.css", "text/css; charset=utf-8"),
            ("/app.js", "text/javascript; charset=utf-8"),
        ):
            with (
                self.subTest(path=path),
                urlopen(
                    f"http://127.0.0.1:{self.server.server_port}{path}", timeout=5
                ) as response,
            ):
                content = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], content_type)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertEqual(int(response.headers["Content-Length"]), len(content))
                self.assertTrue(content.strip())
        with self.assertRaises(HTTPError) as raised:
            self.request("/secret.py")
        self.assertEqual(raised.exception.code, 404)

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
                include_token=False,
            )
        self.assertEqual(err.exception.code, 403)
        err.exception.close()

        # Non-browser POST without Origin/Sec-Fetch-Site and without token must ALSO be 403
        with self.assertRaises(HTTPError) as no_origin_err:
            self.request(
                "/api/config",
                {"config": {"model": "cnn"}},
                include_token=False,
            )
        self.assertEqual(no_origin_err.exception.code, 403)
        no_origin_err.exception.close()

        # Protected API GET without token must be rejected with 403
        for api_path in (
            "/api/browse",
            "/api/config",
            "/api/check_path?path=config.yaml",
        ):
            with self.assertRaises(HTTPError) as get_err:
                self.request(api_path, include_token=False)
            self.assertEqual(get_err.exception.code, 403)
            get_err.exception.close()

        # Cross-port Origin on loopback must be rejected even with token
        wrong_port = self.server.server_port + 1
        with self.assertRaises(HTTPError) as port_err:
            self.request(
                "/api/config",
                {"config": {"model": "cnn"}},
                extra_headers={
                    "Origin": f"http://127.0.0.1:{wrong_port}",
                    "X-WaveDL-Token": app.SESSION_TOKEN,
                },
            )
        self.assertEqual(port_err.exception.code, 403)
        port_err.exception.close()

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

        # HTML response includes security headers
        req = Request(f"http://127.0.0.1:{self.server.server_port}/")
        with urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
            self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
            csp = resp.headers.get("Content-Security-Policy", "")
            policy = {
                parts[0]: parts[1:]
                for directive in csp.split(";")
                if (parts := directive.split())
            }
            self.assertEqual(policy["frame-ancestors"], ["'none'"])
            for directive in ("script-src", "style-src"):
                self.assertIn("'self'", policy[directive])
                self.assertIn("'unsafe-inline'", policy[directive])

    def test_workspace_confinement_and_input_validation(self):
        # NUL bytes return 400 JSON rather than unhandled 500
        for bad_path in ("/api/check_path?path=foo%00bar", "/api/browse?dir=foo%00bar"):
            with self.assertRaises(HTTPError) as nul_err:
                self.request(bad_path)
            self.assertEqual(nul_err.exception.code, 400)
            nul_err.exception.close()

        # Restrict-workspace blocks escaping paths
        with patch.object(app, "RESTRICT_WORKSPACE", True):
            with self.assertRaises(HTTPError) as esc_err:
                self.request("/api/config?path=../escape.yaml")
            self.assertEqual(esc_err.exception.code, 400)
            esc_err.exception.close()
            with self.assertRaises(HTTPError) as browse_err:
                self.request("/api/browse?dir=..")
            self.assertEqual(browse_err.exception.code, 400)
            browse_err.exception.close()

        # Boolean integers and option-like import_modules are rejected
        with self.assertRaises(HTTPError) as bool_gpu_err:
            self.request(
                "/api/train/start",
                {"config": {"model": "cnn"}, "num_gpus": True},
            )
        self.assertEqual(bool_gpu_err.exception.code, 400)
        bool_gpu_err.exception.close()

        (self.root / "ckpt").mkdir()
        (self.root / "test.npz").touch()
        with self.assertRaises(HTTPError) as bool_ic_err:
            self.request(
                "/api/test/start",
                {
                    "mode": "test",
                    "checkpoint": "ckpt",
                    "data_path": "test.npz",
                    "input_channels": True,
                },
            )
        self.assertEqual(bool_ic_err.exception.code, 400)
        bool_ic_err.exception.close()

        with self.assertRaises(HTTPError) as dash_mod_err:
            self.request(
                "/api/test/start",
                {
                    "mode": "test",
                    "checkpoint": "ckpt",
                    "data_path": "test.npz",
                    "import_modules": ["--malicious"],
                },
            )
        self.assertEqual(dash_mod_err.exception.code, 400)
        dash_mod_err.exception.close()

    def test_test_start_forwards_optional_flags(self):
        (self.root / "run" / "best_checkpoint").mkdir(parents=True)
        (self.root / "test.npz").touch()
        child = (
            "import json, pathlib, sys; "
            "pathlib.Path('test_args.json').write_text(json.dumps(sys.argv[1:]))"
        )
        payload = {
            "mode": "export",
            "checkpoint": "run/best_checkpoint",
            "data_path": "test.npz",
            "output_dir": "out",
            "input_channels": 5,
            "batch_size": 16,
            "export_opset": 18,
            "plot_format": "pdf",
            "input_key": "X",
            "output_key": "Y",
            "no_tf32": True,
            "no_denorm": True,
            "allow_pickle": True,
            "import_modules": ["my_mod"],
        }
        with patch.object(
            app, "_resolve_wavedl_command", return_value=[sys.executable, "-c", child]
        ):
            self.request("/api/test/start", payload)
        self.assertEqual(self.trainer.process.wait(timeout=10), 0)
        argv = json.loads((self.root / "test_args.json").read_text())
        for expected in (
            "--input_channels",
            "5",
            "--batch_size",
            "16",
            "--export_opset",
            "18",
            "--plot_format",
            "pdf",
            "--input_key",
            "X",
            "--output_key",
            "Y",
            "--no_tf32",
            "--no_denorm",
            "--allow_pickle",
            "--import",
            "my_mod",
            "--export",
            "onnx",
        ):
            self.assertIn(expected, argv)

    def test_carriage_return_progress_collapses_into_single_line(self):
        finished = threading.Event()
        stream_output = self.trainer._stream_output

        def capture_output(process):
            try:
                stream_output(process)
            finally:
                finished.set()

        child = (
            "import sys; "
            "sys.stdout.write('step 1\\rstep 2\\rstep 3\\rstep 100%\\n'); "
            "sys.stdout.write('done\\n'); "
            "sys.stdout.flush()"
        )
        with (
            patch.object(self.trainer, "_stream_output", side_effect=capture_output),
            patch.object(
                app,
                "_resolve_wavedl_command",
                return_value=[sys.executable, "-c", child],
            ),
        ):
            self.request("/api/train/start", {"config": {"model": "cnn"}})
            self.assertTrue(finished.wait(timeout=10))
        status = json.loads(self.request("/api/train/status"))
        self.assertIn("step 100%", status["logs"])
        self.assertIn("done", status["logs"])
        self.assertNotIn("step 1", status["logs"])
        self.assertNotIn("step 2", status["logs"])

    def test_yaml_canonicalization_and_precision_no(self):
        raw = (
            "model: cnn\n"
            "precision: no\n"
            "loss: l1\n"
            "optimizer:\n"
            "  lr: 0.0005\n"
            "training:\n"
            "  batch_size: 32\n"
        )
        parsed = app.parse_yaml_text(raw)
        self.assertEqual(parsed["precision"], "no")
        canonical = app._flatten_config(parsed)
        self.assertEqual(canonical["precision"], "no")
        self.assertEqual(canonical["loss"], "mae")
        self.assertAlmostEqual(canonical["lr"], 0.0005)
        self.assertEqual(canonical["batch_size"], 32)
        self.assertNotIn("optimizer_lr", canonical)
        self.assertNotIn("training_batch_size", canonical)

        # Also check /api/config/parse round-trip
        res = json.loads(self.request("/api/config/parse", {"raw_yaml": raw}))
        self.assertEqual(res["config"]["precision"], "no")
        self.assertEqual(res["config"]["loss"], "mae")
        self.assertAlmostEqual(res["config"]["lr"], 0.0005)

    def test_config_rejects_non_yaml_extension(self):
        with self.assertRaises(HTTPError) as err:
            self.request("/api/config?path=notes.txt")
        self.assertEqual(err.exception.code, 400)
        err.exception.close()

    def test_stdlib_yaml_scientific_nested_and_dump_scalars(self):
        sample_yaml = (
            "model: cnn\n"
            "lr: 1e-4\n"
            "precision: no\n"
            'run_name: "foo \\" # not a comment"\n'
            'constraint: [true, "max(y0, y1) < 10"]\n'
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
            self.assertEqual(parsed["precision"], "no")
            self.assertEqual(parsed["run_name"], 'foo " # not a comment')
            self.assertEqual(parsed["constraint"], [True, "max(y0, y1) < 10"])
            self.assertAlmostEqual(parsed["nested_alpha"], 2.5e-3)
            self.assertIs(parsed["nested_enabled"], True)
            self.assertEqual(parsed["items"], [True, 10])

        if app._pyyaml is not None:
            self.assertEqual(app.parse_yaml_text(sample_yaml), parsed)

        dumped = app._dump_yaml_kv("flags", [True, None, "yes"])
        self.assertEqual(dumped, 'flags: [true, null, "yes"]')

    def test_model_catalog_and_preprocessing_parity(self):
        html = self.request("/").decode()
        script = self.request("/app.js").decode()
        for expected in ('"inversionnet"', '"inversionnet_resize"'):
            self.assertIn(expected, script)
        for expected in (
            "val_data_path",
            "input_transform",
            "target_scaling",
            "no_tf32",
            "cache_dir",
        ):
            self.assertIn(expected, html)

        # Verify exact parity with live model registry when ML stack is importable
        try:
            from wavedl.models import list_models
        except Exception:
            return

        catalog_names = set(re.findall(r'\{\s*name:\s*"([^"]+)"\s*,\s*cat:', script))
        self.assertEqual(catalog_names, set(list_models()))

    def test_yaml_bare_dot_and_none_scalar_handling(self):
        raw = "output_dir: .\ninput_transform: none\n"
        with patch.object(app, "_pyyaml", None):
            parsed_stdlib = app.parse_yaml_text(raw)
        self.assertEqual(parsed_stdlib["output_dir"], ".")
        self.assertEqual(parsed_stdlib["input_transform"], "none")
        if app._pyyaml is not None:
            parsed_pyyaml = app.parse_yaml_text(raw)
            self.assertEqual(parsed_pyyaml["output_dir"], ".")
            self.assertEqual(parsed_pyyaml["input_transform"], "none")

    def test_malformed_host_and_origin_headers_return_4xx_json(self):
        import http.client

        # Malformed Host port must return 400 JSON rather than 500
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        try:
            conn.request(
                "GET",
                "/api/info",
                headers={
                    "Host": "127.0.0.1:notaport",
                    "X-WaveDL-Token": app.SESSION_TOKEN,
                },
            )
            resp = conn.getresponse()
            body = json.loads(resp.read().decode())
            self.assertEqual(resp.status, 400)
            self.assertIn("Host", body["error"])
        finally:
            conn.close()

        # Malformed Origin port on POST must return 403 JSON rather than 500
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        try:
            conn.request(
                "POST",
                "/api/config",
                body=json.dumps({"config": {"model": "cnn"}}),
                headers={
                    "Content-Type": "application/json",
                    "Origin": "http://127.0.0.1:notaport",
                    "X-WaveDL-Token": app.SESSION_TOKEN,
                },
            )
            resp = conn.getresponse()
            body = json.loads(resp.read().decode())
            self.assertEqual(resp.status, 403)
            self.assertIn("error", body)
        finally:
            conn.close()

    def test_train_start_rejects_empty_raw_yaml(self):
        (self.root / "config.yaml").write_text("model: cnn\n", encoding="utf-8")
        with self.assertRaises(HTTPError) as err:
            self.request(
                "/api/train/start",
                {"path": "config.yaml", "raw_yaml": "   \n  "},
            )
        self.assertEqual(err.exception.code, 400)
        body = json.loads(err.exception.read().decode())
        self.assertIn("raw_yaml", body["error"])
        err.exception.close()

    def test_model_rendering_avoids_inline_dom_xss(self):
        for path in ("/", "/app.js"):
            with self.subTest(path=path):
                content = self.request(path).decode()
                self.assertNotIn('onclick="chooseModel(', content)
                self.assertNotIn('onclick="setModelCat(', content)
                self.assertNotIn(
                    'document.getElementById("modelCatPills").innerHTML', content
                )
                self.assertNotIn("grid.innerHTML =", content)

    def test_model_kwargs_yaml_roundtrip(self):
        raw_block = (
            "model: inversionnet_resize\n"
            "model_kwargs:\n"
            "  base_channels: 16\n"
            "  bottleneck_grid: 2\n"
            "  upsample_mode: nearest\n"
        )
        raw_nested = (
            "model: inversionnet\n"
            "training:\n"
            "  model_kwargs:\n"
            "    base_channels: 8\n"
            "    norm: none\n"
        )
        raw_flow = 'model: cnn\nmodel_kwargs: {"base_channels": 12}\n'

        expected_block = {
            "base_channels": 16,
            "bottleneck_grid": 2,
            "upsample_mode": "nearest",
        }
        for label, use_pyyaml in (("pyyaml", True), ("stdlib", False)):
            with self.subTest(parser=label):
                ctx = (
                    patch.object(app, "_pyyaml", None)
                    if not use_pyyaml
                    else patch.object(app, "_pyyaml", app._pyyaml)
                )
                with ctx:
                    parsed = app._flatten_config(app.parse_yaml_text(raw_block))
                    self.assertEqual(parsed["model"], "inversionnet_resize")
                    self.assertEqual(parsed["model_kwargs"], expected_block)
                    self.assertNotIn("model_kwargs_base_channels", parsed)

                    nested = app._flatten_config(app.parse_yaml_text(raw_nested))
                    self.assertEqual(
                        nested["model_kwargs"], {"base_channels": 8, "norm": "none"}
                    )

                    flow = app._flatten_config(app.parse_yaml_text(raw_flow))
                    self.assertEqual(flow["model_kwargs"], {"base_channels": 12})

        # API parse, save (structured config), and reload round-trip
        parsed_api = json.loads(
            self.request("/api/config/parse", {"raw_yaml": raw_block})
        )
        self.assertEqual(parsed_api["config"]["model_kwargs"], expected_block)

        saved = json.loads(
            self.request(
                "/api/config",
                {
                    "config": {
                        "model": "inversionnet_resize",
                        "model_kwargs": expected_block,
                    }
                },
            )
        )
        self.assertIn("model_kwargs:\n  base_channels: 16", saved["yaml"])
        reloaded = json.loads(self.request("/api/config"))
        self.assertEqual(reloaded["config"]["model_kwargs"], expected_block)

        # Client JS must preserve nested mappings instead of stringifying them
        script = self.request("/app.js").decode()
        self.assertIn('"model_kwargs"', script)
        self.assertIn("fmtEntry(", script)


if __name__ == "__main__":
    unittest.main()
