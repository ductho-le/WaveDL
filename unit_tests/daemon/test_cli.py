"""Real loopback CLI startup, cleanup and additive client behavior."""

import asyncio
import contextlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx
import pytest
import uvicorn

from wavedl.client import ClientError, DaemonClient
from wavedl.core.credentials import read_credential
from wavedl_daemon import cli
from wavedl_daemon.cli import _site, build_parser, serve
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.ownership import acquire_dispatch_lock

from .test_hosted_app import Backend


@pytest.fixture
def daemon_entrypoint(tmp_path):
    """Exercise the declared callable in a fresh dependency-free interpreter."""
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    entrypoint = project["project"]["scripts"]["wavedl"]
    assert entrypoint == "wavedl.client:main"
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "WAVEDL_DAEMON_TOKEN"):
        environment.pop(key, None)
    environment["XDG_STATE_HOME"] = str(tmp_path / "default-state")

    def invoke(*args, module=False):
        script = (
            "import importlib, importlib.abc, runpy, sys; "
            "sys.path.insert(0, sys.argv.pop(1)); "
            "target = sys.argv.pop(1); "
            "sys.argv[0] = 'wavedl'\n"
            "blocked = {'torch', 'numpy', 'PySide6', 'qfluentwidgets', "
            "'uvicorn', 'starlette'}\n"
            "class NoOptionalDependencies(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        assert fullname.split('.')[0] not in blocked, fullname\n"
            "sys.meta_path.insert(0, NoOptionalDependencies())\n"
            "try:\n"
            + (
                "    runpy.run_module(target, run_name='__main__')\n"
                if module
                else "    name, function = target.split(':'); "
                "raise SystemExit(getattr(importlib.import_module(name), function)())\n"
            )
            + "finally:\n"
            "    assert blocked.isdisjoint(sys.modules)\n"
        )
        return subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                script,
                str(root / "src"),
                "wavedl_daemon.cli" if module else entrypoint,
                *(args if module else ("serve", *args)),
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    return invoke


@pytest.mark.parametrize("module", [False, True], ids=["declared-script", "module"])
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ("--help",),
            (
                "usage: wavedl serve",
                "--credential-file",
                "--state-root",
                "--run-root",
                "--port",
            ),
        ),
    ],
)
def test_daemon_help_is_dependency_free_and_has_no_state_side_effects(
    daemon_entrypoint, tmp_path, module, args, expected
):
    result = daemon_entrypoint(*args, module=module)
    assert result.returncode == 0, result.stderr
    assert all(text in result.stdout for text in expected)
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_daemon_inspect_is_not_a_supported_command(daemon_entrypoint, tmp_path):
    """Do not advertise an inspect command that the package does not implement."""
    result = daemon_entrypoint("inspect")
    assert result.returncode == 2
    assert "unrecognized arguments: inspect" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", [None, b"", b"private secret\r\n", b"a" * 4098])
def test_daemon_credential_failure_exits_redacted_before_startup(
    daemon_entrypoint, tmp_path, content
):
    credential = tmp_path / "private-secret-credential"
    if content is not None:
        credential.write_bytes(content)
        credential.chmod(0o600)
    result = daemon_entrypoint(
        "--port",
        "0",
        "--state-root",
        str(tmp_path / "state"),
        "--credential-file",
        str(credential),
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "Daemon unavailable: check listener, private credentials and storage.\n"
    )
    assert "private-secret" not in result.stderr
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "default-state").exists()


def options(tmp_path, *extra):
    return build_parser().parse_args(
        ["--state-root", str(tmp_path / "state"), "--port", "0", *extra]
    )


def test_wavedl_help_lists_serve():
    result = subprocess.run(
        [sys.executable, "-m", "wavedl.client", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "Start WaveDL and open it in your browser" in result.stdout


@pytest.mark.parametrize(
    ("kind", "message", "code"),
    [
        ("running-url", "WaveDL is already running at http://127.0.0.1:8420/", 1),
        ("running", "WaveDL is already running.", 1),
        (
            "ports",
            "Ports 8420 and 8421 are both in use. Close the program using them or choose another port with --port.",
            1,
        ),
        ("port", "Port 9000 is in use. Choose another port with --port.", 1),
        (
            "permission",
            "WaveDL cannot use its state folder. Check that it belongs to you and is private, or choose another with --state-root.",
            1,
        ),
        (
            "other",
            "Daemon unavailable: check listener, private credentials and storage.",
            1,
        ),
        ("interrupt", "", 130),
    ],
)
def test_main_messages(monkeypatch, capsys, kind, message, code):
    async def fail(options, **kwargs):
        errors = {
            "running-url": cli.AlreadyRunning("http://127.0.0.1:8420/"),
            "running": cli.AlreadyRunning(None),
            "ports": cli.PortUnavailable((8420, 8421)),
            "port": cli.PortUnavailable((9000,)),
            "permission": PermissionError("private"),
            "other": ValueError("private"),
            "interrupt": KeyboardInterrupt(),
        }
        raise errors[kind]

    monkeypatch.setattr(cli, "serve", fail)
    assert (
        cli.main(["--port", "9000"] if kind == "port" else [], open_browser=None)
        == code
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (message + "\n" if message else "")


def test_busy_port_refuses_before_storage_or_factory(tmp_path):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        settings = options(tmp_path, "--port", str(occupied.getsockname()[1]))
        with pytest.raises(OSError):
            asyncio.run(
                serve(
                    settings,
                    services_factory=lambda *args, **kw: pytest.fail("factory called"),
                )
            )
    assert not settings.state_root.exists()


def test_explicit_slurm_site_is_bounded_private_and_validated(tmp_path):
    path = tmp_path / "site.json"
    path.write_text(
        json.dumps(
            {"partition": "cpu", "cluster": "fixture", "setup_commands": ["true"]}
        )
    )
    site = _site(path)
    assert site.cluster == "fixture" and site.setup_commands == ("true",)
    path.write_text(json.dumps({"partition": "cpu", "unknown": True}))
    with pytest.raises(ValueError):
        _site(path)
    path.write_bytes(b" " * 65537)
    with pytest.raises(ValueError):
        _site(path)


@pytest.mark.parametrize("port", [-1, 65536])
def test_invalid_port_no_side_effects(tmp_path, port):
    with pytest.raises(ValueError):
        asyncio.run(serve(options(tmp_path, "--port", str(port))))
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("content", [b"", b"bad token", b"secret\r\n", b"a" * 4098])
def test_bad_credential_is_bounded_and_redacted(tmp_path, content):
    path = tmp_path / "credential"
    path.write_bytes(content)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="private, owned and bounded"):
        read_credential(path)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode confidentiality")
def test_credential_file_private_and_no_follow(tmp_path):
    path = tmp_path / "credential"
    path.write_text("private-token\n")
    path.chmod(0o600)
    assert read_credential(path) == "private-token"
    path.chmod(0o644)
    with pytest.raises(ValueError):
        read_credential(path)
    path.chmod(0o600)
    linked = tmp_path / "link"
    linked.symlink_to(path)
    with pytest.raises(ValueError):
        read_credential(linked)


@pytest.mark.parametrize("credential_mode", ["ephemeral", "private-file"])
def test_actual_serve_bootstrap_client_idempotency_and_shutdown(
    tmp_path, credential_mode
):
    output = io.StringIO()
    retained = {}
    extra = []
    if credential_mode == "private-file":
        credential = tmp_path / "daemon-token"
        credential.write_bytes(b"private-startup-token\n")
        credential.chmod(0o600)
        assert read_credential(credential) == "private-startup-token"
        extra = ["--credential-file", str(credential)]

    class ControlledServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            retained["port"] = sockets[0].getsockname()[1]
            retained["server"] = self

    def factory(settings, **kwargs):
        backend = Backend()
        services = HostedServices(
            settings, backend_factory=lambda lock: {"local": backend}, **kwargs
        )
        retained.update(backend=backend, services=services)
        return services

    async def exercise():
        task = asyncio.create_task(
            serve(
                options(tmp_path, *extra),
                output=output,
                services_factory=factory,
                server_factory=ControlledServer,
            )
        )
        try:
            deadline = time.monotonic() + 3
            while (
                "server" not in retained
                and not task.done()
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.01)
            assert "server" in retained
            base = f"http://127.0.0.1:{retained['port']}"
            app = retained["server"].config.loaded_app
            # Test injection omits ReadyServer's printing but shares its actual
            # app and single-use bootstrap pool through the captured config.
            token = app.state.auth_token.decode()
            if credential_mode == "private-file":
                assert token == read_credential(credential)
            client = DaemonClient(base, token)
            health = await client.request_async("GET", "/api/health")
            assert health["protocol_min"] == 1
            payload = {
                "backend": "local",
                "operation": "train",
                "config": {"epochs": 1},
                "runtime": {
                    "python": sys.executable,
                    "mode": "existing",
                    "device": "cpu",
                    "environment": {},
                },
                "topology": {
                    "num_machines": 1,
                    "processes_per_machine": 1,
                    "machine_rank": 0,
                },
            }
            first = await client.request_async(
                "POST", "/api/runs/plan", payload, idempotency_key="retry-me"
            )
            second = await client.request_async(
                "POST", "/api/runs/plan", payload, idempotency_key="retry-me"
            )
            assert first == second
            await client.request_async(
                "POST", "/api/runs", {"plan_id": first["plan_id"]}
            )
            await client.request_async(
                "POST", "/api/runs", {"plan_id": first["plan_id"]}
            )
            deadline = time.monotonic() + 3
            while not retained["backend"].submissions and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert retained["backend"].submissions == 1
            async with httpx.AsyncClient() as http:
                response = await http.get(base + "/api/health")
                assert response.status_code == 401
            assert token not in output.getvalue()
        finally:
            if "server" in retained:
                retained["server"].should_exit = True
            await asyncio.wait_for(task, 5)

    asyncio.run(exercise())
    assert not retained["services"].host._thread.is_alive()
    with acquire_dispatch_lock(tmp_path / "state" / "dispatch.lock"):
        pass


def test_default_serve_prints_only_bootstrap_after_ready(tmp_path, monkeypatch):
    output = io.StringIO()
    original = uvicorn.Server.serve

    async def stop_after_ready(server, sockets=None):
        async def stop():
            while not server.started and not server.should_exit:
                await asyncio.sleep(0.01)
            server.should_exit = True

        task = asyncio.create_task(stop())
        try:
            await original(server, sockets=sockets)
        finally:
            await task

    monkeypatch.setattr(uvicorn.Server, "serve", stop_after_ready)
    asyncio.run(serve(options(tmp_path), output=output))
    assert re.search(
        r"http://127\.0\.0\.1:\d+/\?bootstrap=[A-Za-z0-9_-]{43}", output.getvalue()
    )
    assert "?token=" not in output.getvalue()
    link = output.getvalue().splitlines()[0].removeprefix("WaveDL is running at ")
    assert output.getvalue().splitlines() == [
        f"WaveDL is running at {link}",
        "Open that link in your browser if it did not open automatically. Press Ctrl+C to stop.",
    ]
    assert not (tmp_path / "state" / "serve.json").exists()


@pytest.mark.parametrize("failure", ["exception", "cancel"])
def test_server_failure_after_startup_drains_owned_resources(tmp_path, failure):
    retained = {}

    class BrokenServer(uvicorn.Server):
        async def main_loop(self):
            if failure == "cancel":
                raise asyncio.CancelledError
            raise OSError("Broken listener")

    def factory(settings, **kwargs):
        services = HostedServices(settings, **kwargs)
        retained["services"] = services
        return services

    with pytest.raises(asyncio.CancelledError if failure == "cancel" else OSError):
        asyncio.run(
            serve(
                options(tmp_path), services_factory=factory, server_factory=BrokenServer
            )
        )
    assert not retained["services"].host._thread.is_alive()
    assert not (tmp_path / "state" / "serve.json").exists()
    with acquire_dispatch_lock(tmp_path / "state" / "dispatch.lock"):
        pass


def test_second_serve_reports_already_running(tmp_path):
    settings = options(tmp_path)
    with (
        acquire_dispatch_lock(settings.state_root / "dispatch.lock"),
        pytest.raises(cli.AlreadyRunning),
    ):
        asyncio.run(
            serve(
                settings,
                services_factory=lambda *a, **kw: pytest.fail("factory called"),
            )
        )


@pytest.mark.parametrize("failure", ["exception", "cancel", "interrupt-before-ready"])
def test_run_file_removed_after_shutdown(tmp_path, monkeypatch, failure):
    from wavedl_daemon.serve_state import read_run_url

    async def broken_loop(server):
        assert read_run_url(tmp_path / "state") is not None
        if failure == "cancel":
            raise asyncio.CancelledError
        raise OSError("Broken listener")

    async def interrupted_startup(server, sockets=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn.Server, "main_loop", broken_loop)
    if failure == "interrupt-before-ready":
        monkeypatch.setattr(uvicorn.Server, "startup", interrupted_startup)
    error = {
        "exception": OSError,
        "cancel": asyncio.CancelledError,
        "interrupt-before-ready": KeyboardInterrupt,
    }[failure]
    with pytest.raises(error):
        asyncio.run(serve(options(tmp_path), output=io.StringIO()))
    assert not (tmp_path / "state" / "serve.json").exists()
    with acquire_dispatch_lock(tmp_path / "state" / "dispatch.lock"):
        pass


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX termination signal")
def test_sigterm_removes_run_file(tmp_path):
    root = Path(__file__).resolve().parents[2]
    state = tmp_path / "state"
    environment = dict(os.environ, BROWSER="true", PYTHONPATH=str(root / "src"))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "wavedl_daemon.cli",
            "--state-root",
            str(state),
            "--port",
            "0",
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert "WaveDL is running at" in process.stdout.readline()
        deadline = time.monotonic() + 10
        while not (state / "serve.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        process.terminate()
        process.wait(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
    assert not (state / "serve.json").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_busy_ports_raise_port_unavailable(tmp_path, monkeypatch, explicit):
    with socket.socket() as first, socket.socket() as second:
        for listener in (first, second):
            listener.bind(("127.0.0.1", 0))
            listener.listen()
        ports = (first.getsockname()[1], second.getsockname()[1])
        monkeypatch.setattr(cli, "DEFAULT_PORTS", ports)
        settings = options(tmp_path)
        settings.port = ports[0] if explicit else None
        with pytest.raises(cli.PortUnavailable) as error:
            asyncio.run(serve(settings))
        assert error.value.ports == (ports[:1] if explicit else ports)
        assert not settings.state_root.exists()


@pytest.mark.parametrize("browser", ["record", "raise", "hang"])
def test_launch_link_reusable_browser_and_port_fallback(tmp_path, monkeypatch, browser):
    import threading
    from urllib.parse import parse_qs, urlsplit

    from wavedl_daemon.serve_state import read_run_url

    output = io.StringIO()
    called = threading.Event()
    release = threading.Event()
    opened = []
    original = uvicorn.Server.main_loop

    def opener(url):
        opened.append(url)
        called.set()
        if browser == "raise":
            raise RuntimeError("private browser failure")
        if browser == "hang":
            release.wait(10)

    async def exercise(server):
        link = output.getvalue().splitlines()[0].removeprefix("WaveDL is running at ")
        assert read_run_url(tmp_path / "state") == link.split("?")[0]
        code = parse_qs(urlsplit(link).query)["bootstrap"][0]
        async with httpx.AsyncClient() as client:
            for _ in range(2):
                response = await client.post(
                    link.split("?")[0] + "api/bootstrap", json={"code": code}
                )
                assert response.status_code == 200
        assert called.wait(1)
        assert opened == [link]
        server.should_exit = True
        await original(server)

    monkeypatch.setattr(uvicorn.Server, "main_loop", exercise)
    with socket.socket() as occupied, socket.socket() as free:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        free.bind(("127.0.0.1", 0))
        ports = (occupied.getsockname()[1], free.getsockname()[1])
        free.close()
        monkeypatch.setattr(cli, "DEFAULT_PORTS", ports)
        settings = options(tmp_path)
        settings.port = None
        try:
            asyncio.run(serve(settings, output=output, open_browser=opener))
        finally:
            release.set()
    assert f":{ports[1]}/" in output.getvalue()
    assert not (tmp_path / "state" / "serve.json").exists()


@pytest.mark.parametrize(
    "path", [None, 1, "/api/bad path", "/api/bad\tpath", "/api/é", "/api/" + "x" * 8193]
)
def test_client_refuses_malformed_request_target(path):
    with pytest.raises(ClientError):
        DaemonClient("http://127.0.0.1", "token").request("GET", path)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "http://127.0.0.1/?token=secret",
        "https://u:p@example.com",
        "https://example.com/api",
        "http://127.1",
    ],
)
def test_client_refuses_unsafe_origin(url):
    with pytest.raises(ValueError):
        DaemonClient(url, "private-token")


@pytest.mark.parametrize(
    "reply",
    [
        b"HTTP/1.1 302 Found\r\nLocation: http://example.com\r\nContent-Length: 2\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1048577\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
    ],
)
def test_client_refuses_redirect_and_ambiguous_framing(reply):
    async def exercise():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(reply)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            with pytest.raises(ClientError):
                await DaemonClient(
                    f"http://127.0.0.1:{port}", "private-token"
                ).request_async("GET", "/api/health")

    asyncio.run(exercise())


def test_client_absolute_slow_body_deadline_closes_socket():
    async def exercise():
        stopped = asyncio.Event()

        async def handler(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 128\r\n\r\n"
                )
                await writer.drain()
                for _ in range(100):
                    writer.write(b" ")
                    await writer.drain()
                    await asyncio.sleep(0.01)
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                writer.close()
                with contextlib.suppress(ConnectionError):
                    await writer.wait_closed()
                stopped.set()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            started = time.monotonic()
            with pytest.raises(ClientError):
                await DaemonClient(
                    f"http://127.0.0.1:{port}", "private-token", timeout=0.05
                ).request_async("GET", "/api/health")
            assert time.monotonic() - started < 0.4
            await asyncio.wait_for(stopped.wait(), 1)

    asyncio.run(exercise())
