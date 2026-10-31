"""T39 acceptance uses real wheel console scripts outside the checkout."""

import asyncio
import configparser
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import httpx
import pytest
from packaging.requirements import Requirement

from wavedl_daemon.supervision.base import ProcessSpec
from wavedl_daemon.supervision.windows import WindowsProcess, WindowsSupervisor, _Win32


class _InstalledWindowsAPI(_Win32):
    """Retain native cleanup witnesses without changing supervision policy."""

    def __init__(self):
        super().__init__()
        self.forced_kills = 0
        self.observed_empty = False
        self.reserved_process = None

    def assign(self, job, process):
        # Reserve the leader's native identity before assignment/resume. The
        # supervisor may finish and close its own handle before shutdown starts.
        current = self.winapi.GetCurrentProcess()
        self.reserved_process = self.winapi.DuplicateHandle(
            current,
            process,
            current,
            0,
            False,
            2,  # DUPLICATE_SAME_ACCESS
        )
        super().assign(job, process)

    def close_reservation(self):
        if self.reserved_process is not None:
            handle = self.reserved_process
            self.reserved_process = None
            self.close(handle)

    def terminate_job(self, job):
        self.forced_kills += 1
        super().terminate_job(job)

    def active(self, job):
        count = super().active(job)
        self.observed_empty |= count == 0
        return count


def _start_windows_installed(argv, cwd, environment, *, api=None):
    return WindowsSupervisor(_api=api).start(
        ProcessSpec(tuple(argv), Path(cwd), environment, inherit_environment=False),
        output_limit=8 * 1024 * 1024,
    )


def _require_windows_result(result, argv, timeout):
    if result.timed_out:
        raise subprocess.TimeoutExpired(argv, timeout)
    assert not result.cancelled
    assert not result.stdout_truncated and not result.stderr_truncated
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout.decode()


def _pip_install_phase(output):
    """Report the last observed marker, not an inferred reason for a timeout."""
    for line in reversed(output.splitlines()):
        for phase, prefixes in (
            ("complete", (b"Successfully installed ",)),
            ("uninstall-complete", (b"Successfully uninstalled ",)),
            ("uninstall", (b"Attempting uninstall:", b"Uninstalling ")),
            ("unpack", (b"Installing collected packages:",)),
            ("resolve", (b"Processing ", b"Collecting ")),
        ):
            if line.strip().startswith(prefixes):
                return phase
    return "unknown"


def _pip_timeout_diagnostic(result):
    print(
        "Installed dependency timeout: "
        f"last_progress={_pip_install_phase(result.stdout)}, "
        f"timed_out={bool(result.timed_out)}, cancelled={bool(result.cancelled)}, "
        f"stdout_truncated={bool(result.stdout_truncated)}, "
        f"stderr_truncated={bool(result.stderr_truncated)}"
    )


class _WindowsInstalledDaemon:
    """One suspended-assigned Job and its existing bounded output drain."""

    def __init__(self, argv, cwd, environment):
        self.api = _InstalledWindowsAPI()
        # Do not attach to, detach from, or allocate the runner's console. A
        # missing console cannot establish targeted graceful shutdown evidence.
        if not self.api.kernel.GetConsoleCP():
            raise OSError("Native graceful daemon acceptance requires a console")
        try:
            self.process = _start_windows_installed(
                argv, cwd, environment, api=self.api
            )
        except BaseException:
            self.api.close_reservation()
            raise
        assert isinstance(self.process, WindowsProcess)
        self.position = 0

    async def readline(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            # Read only the owned production drain's captured bytes. A second
            # pipe reader would race and steal the bootstrap line from it.
            data = self.process._stdout.data
            assert isinstance(data, bytearray)
            snapshot = bytes(data)
            end = snapshot.find(b"\n", self.position)
            if end >= 0:
                line = snapshot[self.position : end + 1]
                self.position = end + 1
                return line
            await asyncio.sleep(0.01)
        raise TimeoutError("Native installed daemon did not emit readiness")

    async def shutdown(self, record_property=None):
        try:
            assert self.api.reserved_process is not None
            if not self.api.leader_running(self.api.reserved_process):
                raise AssertionError("Owned native daemon exited before CTRL_BREAK")
            # A nonzero process-group ID targets only this owned daemon and
            # descendants. Unlike production best-effort cancellation, errors
            # 6/87 are acceptance failures, not graceful-shutdown evidence.
            if not self.api.kernel.GenerateConsoleCtrlEvent(1, self.process.pid):
                import ctypes

                error = ctypes.get_last_error()
                raise OSError(f"Targeted native CTRL_BREAK failed: winerror={error}")
            result = await asyncio.to_thread(self.process.wait, 15)
            print(f"Native installed daemon exit code: {result.returncode}")
            print(
                "Native installed daemon cleanup: "
                f"job_empty={self.api.observed_empty}, "
                f"forced_kills={self.api.forced_kills}, "
                f"timed_out={result.timed_out}, cancelled={result.cancelled}"
            )
            if record_property is not None:
                record_property("native_windows_daemon_exit_code", result.returncode)
            # Native Uvicorn SIGBREAK re-raise was observed as WinCRT exit 3.
            # Keep the raw status, but never accept arbitrary crashes merely
            # because the shutdown log was emitted before the failure.
            assert type(result.returncode) is int and result.returncode in {0, 3}
            assert not result.timed_out and not result.cancelled
            assert not result.stdout_truncated and not result.stderr_truncated
            assert self.api.observed_empty and self.api.forced_kills == 0
            assert b"Application shutdown complete" in result.stderr
            return result.stdout[self.position :], result.stderr
        finally:
            # Failure still owns/reaps the entire Job, but never converts that
            # forceful cleanup into passing graceful-shutdown evidence.
            try:
                self.process.cancel(grace_seconds=0)
                await asyncio.to_thread(self.process.wait)
            finally:
                self.api.close_reservation()


def checked(argv, *, cwd, environment, timeout=None, pip_diagnostic=False):
    if timeout is None:
        timeout = int(os.environ.get("WAVEDL_ACCEPTANCE_TIMEOUT", "60"))
    if timeout < 1 or timeout > 300:
        raise ValueError("WAVEDL_ACCEPTANCE_TIMEOUT must be between 1 and 300 seconds")
    if os.name == "nt":
        process = _start_windows_installed(argv, cwd, environment)
        try:
            result = process.wait(timeout=timeout)
            token = environment.get("WAVEDL_DAEMON_TOKEN")
            if token and token.encode() in result.stdout + result.stderr:
                raise AssertionError("Installed client leaked authentication material")
            if pip_diagnostic and result.timed_out:
                _pip_timeout_diagnostic(result)
            return _require_windows_result(result, argv, timeout)
        finally:
            process.cancel(grace_seconds=0)
            process.wait()
    # A training launcher owns descendants; timeout cleanup must reap its entire
    # process group rather than leaving accelerate/the runtime behind.
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate(timeout=min(15, max(10, timeout // 4)))
            raise
        assert process.returncode == 0, stderr
        return stdout


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    return build_installed(tmp_path_factory)


def build_installed(tmp_path_factory, *, wheelhouse=None):
    """Build once in staging; optionally resolve dependencies in a clean venv."""

    root = Path(__file__).resolve().parents[2]
    outside = tmp_path_factory.mktemp("installed-cli")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("WAVEDL_DAEMON_TOKEN", None)
    environment.pop("WAVEDL_DAEMON_URL", None)
    environment.pop("PYTHONHOME", None)
    if wheelhouse is not None:
        environment["PYTHONNOUSERSITE"] = "1"
    for key in tuple(environment):
        if key.startswith("PIP_"):
            environment.pop(key)
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    wheels = outside / "wheels"
    wheels.mkdir()
    # Setuptools reuses build/lib and does not remove deleted source modules.
    # Build in an isolated staging tree, preserving the user's checkout/build
    # output and ensuring the wheel reflects only the current source inventory.
    acceptance_wheel = os.environ.get("WAVEDL_ACCEPTANCE_WHEEL")
    acceptance_sdist = os.environ.get("WAVEDL_ACCEPTANCE_SDIST")
    assert not (acceptance_wheel and acceptance_sdist), (
        "WAVEDL_ACCEPTANCE_WHEEL and WAVEDL_ACCEPTANCE_SDIST are mutually exclusive"
    )
    if acceptance_wheel:
        # Release acceptance must exercise the exact artifact to be published,
        # never silently rebuild a different wheel from the editable checkout.
        source = Path(acceptance_wheel).resolve()
        assert source.is_file() and source.suffix == ".whl", source
        shutil.copy2(source, wheels / source.name)
    elif acceptance_sdist:
        # A configured source distribution is an exact release input. Build its
        # wheel in the outside staging directory; never silently fall back to
        # the checkout when the path is missing or malformed.
        source = Path(acceptance_sdist).resolve()
        assert source.is_file() and source.name.endswith(".tar.gz"), source
        assert wheelhouse is not None, (
            "Exact sdist acceptance requires an offline wheelhouse"
        )
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_copy = outside / source.name
        shutil.copy2(source, source_copy)
        expected = sdist_inventory(source_copy)
        build_venv = outside / "build-venv"
        checked(
            [sys.executable, "-m", "venv", str(build_venv)],
            cwd=outside,
            environment=environment,
        )
        build_python = build_venv / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        # Explicit build tools, isolated from ambient setuptools/pip settings.
        # The supplied wheelhouse must contain these as well as runtime wheels.
        checked(
            [
                str(build_python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--only-binary=:all:",
                "--find-links",
                str(wheelhouse),
                "setuptools>=83.0.0,<85",
                "wheel",
            ],
            cwd=outside,
            environment=environment,
            timeout=180,
        )
        checked(
            [
                str(build_python),
                "-m",
                "pip",
                "wheel",
                str(source_copy),
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                str(wheels),
            ],
            cwd=outside,
            environment=environment,
            timeout=180,
        )
        built = list(wheels.glob("*.whl"))
        assert len(built) == 1, built
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
        assert hashlib.sha256(source_copy.read_bytes()).hexdigest() == source_digest
        assert wheel_inventory(built[0]) == expected, (
            "Sdist rebuild changed the version or complete web asset inventory"
        )
    else:
        staging = outside / "source"
        staging.mkdir()
        generated = shutil.ignore_patterns("__pycache__", "*.egg-info", "AppData")

        def ignore(directory, names):
            # Mirror the sdist: the workstation source and node_modules never ship.
            ignored = generated(directory, names)
            if Path(directory) == root / "src" / "wavedl_daemon":
                ignored.add("web")
            return ignored

        shutil.copytree(root / "src", staging / "src", ignore=ignore)
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
            source = root / name
            if source.exists():
                shutil.copy2(source, staging / name)
        checked(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(staging),
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                str(wheels),
            ],
            cwd=outside,
            environment=environment,
        )
    venv = outside / "venv"
    checked(
        [sys.executable, "-m", "venv"]
        + (["--system-site-packages"] if wheelhouse is None else [])
        + [str(venv)],
        cwd=outside,
        environment=environment,
    )
    binary = venv / ("Scripts" if os.name == "nt" else "bin")
    python = binary / ("python.exe" if os.name == "nt" else "python")
    (wheel,) = wheels.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {
        "wavedl/core/client_transport.py",
        "wavedl/distributed_launch.py",
        "wavedl/launcher.py",
        "wavedl/runtime.py",
        "wavedl/train.py",
        "wavedl_daemon/runs/catalog.py",
        "wavedl_daemon/runs/model_metadata.json",
    } <= names
    assert_no_qt_wheel(wheel)
    checked(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            *(
                ["--ignore-installed", "--no-deps", str(wheel)]
                if wheelhouse is None
                else [
                    "--only-binary=:all:",
                    "--find-links",
                    str(wheelhouse),
                    str(wheel),
                ]
            ),
        ],
        cwd=outside,
        environment=environment,
        # Full offline dependency installation has its own finite budget; do
        # not increase runtime commands or cheap inherited-site installation.
        timeout=180 if wheelhouse is None else 300,
        pip_diagnostic=wheelhouse is not None,
    )
    return outside, venv, binary, python, environment


def sdist_inventory(path):
    """Read the exact release tar's version and complete packaged asset bytes."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names)), "Duplicate sdist members"
        roots = {name.split("/")[0] for name in names}
        assert len(roots) == 1, "Sdist must have one source root"
        root = roots.pop()
        assert not any("wavedl_gui" in Path(name).parts for name in names), (
            "The sdist must not ship the retired Qt application"
        )
        assert f"{root}/src/wavedl_daemon/runs/model_metadata.json" in names
        metadata = archive.getmember(f"{root}/PKG-INFO")
        assert metadata.isfile(), "Sdist metadata must be an ordinary file"
        package_metadata = Parser().parsestr(
            archive.extractfile(metadata).read().decode()
        )
        assert_no_qt_metadata(package_metadata)
        version = package_metadata["Version"]
        config = archive.getmember(f"{root}/pyproject.toml")
        assert config.isfile(), "Sdist project configuration must be an ordinary file"
        project = tomllib.loads(archive.extractfile(config).read().decode())["project"]
        assert "wavedl-gui" not in project["scripts"]
        prefix = f"{root}/src/wavedl_daemon/web_assets/"
        assets = {}
        for member in members:
            if member.name.startswith(prefix) and not member.isdir():
                assert member.isfile(), "Sdist assets must be ordinary files"
                assets[member.name.removeprefix(prefix)] = archive.extractfile(
                    member
                ).read()
        assert version and {"index.html", ".wavedl-assets.json"} <= assets.keys()
        return version, assets


def wheel_inventory(path):
    """Compare a rebuilt wheel to its source tar, including hidden manifests."""
    assert_no_qt_wheel(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate wheel members"
        (metadata,) = (name for name in names if name.endswith(".dist-info/METADATA"))
        version = Parser().parsestr(archive.read(metadata).decode())["Version"]
        prefix = "wavedl_daemon/web_assets/"
        assets = {
            name.removeprefix(prefix): archive.read(name)
            for name in names
            if name.startswith(prefix) and not name.endswith("/")
        }
        return version, assets


def assert_no_qt_metadata(metadata):
    assert "gui" not in metadata.get_all("Provides-Extra", [])
    for value in metadata.get_all("Requires-Dist", []):
        name = Requirement(value).name.lower().replace("_", "-")
        assert not name.startswith(("pyside", "pyqt", "qfluent", "shiboken")), value


def assert_no_qt_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert not any("wavedl_gui" in Path(name).parts for name in names), (
            "The wheel must not ship the retired Qt application"
        )
        (metadata,) = (name for name in names if name.endswith(".dist-info/METADATA"))
        assert_no_qt_metadata(Parser().parsestr(archive.read(metadata).decode()))
        (entrypoints,) = (
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        config = configparser.ConfigParser()
        config.read_string(archive.read(entrypoints).decode())
        scripts = config["console_scripts"]
        assert "wavedl-gui" not in scripts
        assert not any("wavedl_gui" in target for target in scripts.values())


@pytest.fixture(scope="module")
def isolated_resolved(tmp_path_factory):
    """Install the wheel and its declared dependencies from a local wheelhouse.

    The normal installed acceptance intentionally remains cheap and offline.  This
    opt-in gate is the dependency-resolution proof: it never consults an index and
    skips only when no wheelhouse/cache was configured. Incomplete explicit inputs
    fail acceptance. Set ``WAVEDL_ACCEPTANCE_WHEELHOUSE`` in CI to the populated
    wheel cache (or place wheels in ``wheelhouse/``).
    """

    cache = os.environ.get("WAVEDL_ACCEPTANCE_PIP_CACHE")
    wheelhouse = os.environ.get("WAVEDL_ACCEPTANCE_WHEELHOUSE")
    if cache and not wheelhouse:
        wheelhouse = tmp_path_factory.mktemp("acceptance-cached-wheels")
        export_cached_wheels(Path(cache), Path(wheelhouse))
    candidates = (
        [Path(wheelhouse)]
        if wheelhouse
        else [
            Path("wheelhouse"),
            Path(".wheelhouse"),
            Path("dist/wheels"),
        ]
    )
    wheel_dir = next(
        (
            path.resolve()
            for path in candidates
            if path.is_dir() and any(path.glob("*.whl"))
        ),
        None,
    )
    if wheel_dir is None:
        if wheelhouse or cache:
            pytest.fail(f"configured acceptance wheelhouse has no wheels: {wheelhouse}")
        pytest.skip(
            "isolated dependency acceptance requires a local wheelhouse; "
            "set WAVEDL_ACCEPTANCE_WHEELHOUSE or WAVEDL_ACCEPTANCE_PIP_CACHE "
            "(no network is used)"
        )
    return build_installed(tmp_path_factory, wheelhouse=wheel_dir)


def export_cached_wheels(cache, destination):
    """Expose actual pip wheel ZIPs, not reconstructed installed distributions.

    HTTP cache bodies are opaque filenames. Wheel metadata supplies the original
    distribution/version and compatibility tags; no archive bytes are modified.
    Cache layout is deliberately optional, never an assumption of normal tests.
    """

    assert cache.is_dir(), f"Pip cache does not exist: {cache}"
    candidates = list(cache.rglob("*.whl")) + list(cache.rglob("*.body"))
    for source in candidates:
        if not zipfile.is_zipfile(source):
            continue
        with zipfile.ZipFile(source) as archive:
            headers = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
            if len(headers) != 1:
                continue
            header = headers[0]
            stem = header.split("/")[0].removesuffix(".dist-info")
            metadata = Parser().parsestr(archive.read(header).decode())
            for tag in metadata.get_all("Tag", []):
                filename = f"{stem}-{tag}.whl"
                assert re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename), filename
                target = destination / filename
                if not target.exists():
                    # Hard links preserve cache bytes and avoid duplicating the
                    # large ML stack when temporary storage is on the same volume.
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)


def test_installed_entrypoints_and_lightweight_modules(installed):
    outside, venv, binary, python, environment = installed
    for name, args in (
        ("wavedl", ["run", "--help"]),
        ("wavedl", ["serve", "--help"]),
        ("wavedl-train", ["--help"]),
    ):
        script = binary / (name + (".exe" if os.name == "nt" else ""))
        assert script.exists()
        text = checked([str(script), *args], cwd=outside, environment=environment)
        assert "usage:" in text.lower()
    site = checked(
        [
            str(python),
            "-c",
            "import wavedl,pathlib; print(pathlib.Path(wavedl.__file__).parent.parent)",
        ],
        cwd=outside,
        environment=environment,
    ).strip()
    assert Path(site).is_relative_to(venv)
    checked(
        [
            str(python),
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import wavedl.client,wavedl.operations,wavedl.distributed_launch,wavedl_daemon.cli,wavedl.core.scheduler_attempt,wavedl.core.client_transport; "
            "assert not any(n in sys.modules for n in ('torch','numpy','PySide6','qfluentwidgets','starlette','uvicorn'))",
            site,
        ],
        cwd=outside,
        environment=environment,
    )
    checked(
        [
            str(python),
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import importlib.util; "
            "assert importlib.util.find_spec('wavedl_gui') is None; "
            "from wavedl_daemon.runs.catalog import CatalogService; "
            "assert CatalogService({}).models()['models']; "
            "assert not any(n.startswith(('torch','numpy','PySide6','qfluentwidgets','wavedl_gui')) for n in sys.modules)",
            site,
        ],
        cwd=outside,
        environment=environment,
    )


def test_isolated_resolved_wheelhouse_console_scripts_and_assets(
    isolated_resolved, record_property
):
    """A local wheelhouse proves resolver installation without checkout leakage."""

    outside, _venv, binary, python, environment = isolated_resolved
    assert not environment.get("PYTHONPATH")
    assert "include-system-site-packages = false" in (_venv / "pyvenv.cfg").read_text()
    checked([str(python), "-m", "pip", "check"], cwd=outside, environment=environment)
    for name, args in (
        ("wavedl", ["run", "--help"]),
        ("wavedl", ["serve", "--help"]),
        ("wavedl-train", ["--help"]),
    ):
        script = binary / (name + (".exe" if os.name == "nt" else ""))
        assert script.is_file()
        assert (
            "usage:"
            in checked(
                [str(script), *args], cwd=outside, environment=environment
            ).lower()
        )
    probe = checked(
        [
            str(python),
            "-c",
            "import pathlib,sys,wavedl,wavedl_daemon; "
            "root=pathlib.Path(wavedl.__file__).parents[1]; "
            "assert root.is_relative_to(pathlib.Path(sys.prefix)); "
            "assert (pathlib.Path(wavedl_daemon.__file__).parent/'web_assets/index.html').is_file(); "
            "assert not any(pathlib.Path(p).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()) for p in sys.path if p)",
            str(Path(__file__).resolve().parents[2]),
        ],
        cwd=outside,
        environment=environment,
    )
    assert not probe.strip()
    test_installed_entrypoints_and_lightweight_modules(isolated_resolved)
    # Calling the CPU exercise directly intentionally does not consult the
    # host's torch availability: this runtime must have resolved its own stack.
    test_installed_standalone_training_requires_no_daemon(isolated_resolved)
    test_installed_serve_bootstrap_plan_client_and_shutdown(
        isolated_resolved, record_property
    )


def test_installed_serve_bootstrap_plan_client_and_shutdown(installed, record_property):
    outside, _venv, binary, python, environment = installed
    state = outside / "state"
    (wheel,) = (outside / "wheels").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        prefix = "wavedl_daemon/web_assets/"
        assets = {
            name.removeprefix(prefix): archive.read(name)
            for name in archive.namelist()
            if name.startswith(prefix)
            and not name.endswith("/")
            and not name.removeprefix(prefix).startswith(".")
        }

    async def exercise():
        argv = [
            str(binary / ("wavedl.exe" if os.name == "nt" else "wavedl")),
            "serve",
            "--port",
            "0",
            "--state-root",
            str(state),
        ]
        native = (
            _WindowsInstalledDaemon(argv, outside, environment)
            if os.name == "nt"
            else None
        )
        process = (
            None
            if native is not None
            else await asyncio.create_subprocess_exec(
                *argv,
                cwd=outside,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        try:
            line = (
                await native.readline()
                if native is not None
                else await asyncio.wait_for(process.stdout.readline(), 15)
            ).decode()
            match = re.search(
                r"WaveDL is running at (http://127\.0\.0\.1:\d+)/\?bootstrap=([A-Za-z0-9_-]{43})",
                line,
            )
            assert match, line
            base, code = match.groups()
            async with httpx.AsyncClient(base_url=base) as client:
                assert "index.html" in assets
                assert any(name.startswith("_nuxt/") for name in assets)
                for name, expected in assets.items():
                    if name == "sw.js":
                        continue
                    response = await client.get("/" + name)
                    assert response.status_code == 200, (name, response.text)
                    assert response.content == expected, name
                worker = await client.get("/sw.js")
                assert worker.status_code == 200
                assert worker.headers["cache-control"] == "no-store"
                assert b"registration.unregister" in worker.content
                assert b'addEventListener("fetch"' not in worker.content
                assert b"navigate(" not in worker.content
                assert b"caches" not in worker.content
                root = await client.get("/")
                assert root.content == assets["index.html"]
                assert root.headers["cache-control"] == "no-store"
                assert (await client.get("/api/health")).status_code == 401
                response = await client.post("/api/bootstrap", json={"code": code})
                assert response.status_code == 200, response.text
                token = response.json()["access_token"]
                assert token not in line
                assert (
                    await client.post("/api/bootstrap", json={"code": code})
                ).status_code == 200
                headers = {"Authorization": f"Bearer {token}"}
                authenticated_health = await client.get("/api/health", headers=headers)
                assert authenticated_health.status_code == 200, (
                    authenticated_health.text
                )
                assert authenticated_health.json()["protocol_min"] == 1
                models = await client.get("/api/models", headers=headers)
                assert models.status_code == 200, models.text
                assert (
                    models.json()["models"]
                    and models.json()["runtime_verified"] is False
                )
                request = outside / "request.json"
                request.write_text(
                    json.dumps(
                        {
                            "backend": "local",
                            "operation": "train",
                            "config": {"epochs": 1},
                            "runtime": {
                                "python": str(python),
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
                    )
                )
                responses = []
                for _ in range(2):
                    client_argv = [
                        str(binary / ("wavedl.exe" if os.name == "nt" else "wavedl")),
                        "run",
                        "--daemon",
                        base,
                        "--request",
                        str(request),
                        "--idempotency-key",
                        "installed-retry",
                        "--plan-only",
                    ]
                    if native is not None:
                        stdout = await asyncio.to_thread(
                            checked,
                            client_argv,
                            cwd=outside,
                            environment=environment | {"WAVEDL_DAEMON_TOKEN": token},
                            timeout=10,
                        )
                        assert token not in stdout
                        responses.append(json.loads(stdout))
                        continue
                    client_process = await asyncio.create_subprocess_exec(
                        *client_argv,
                        cwd=outside,
                        env=environment | {"WAVEDL_DAEMON_TOKEN": token},
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            client_process.communicate(), 10
                        )
                    finally:
                        if client_process.returncode is None:
                            client_process.kill()
                            await asyncio.wait_for(client_process.communicate(), 5)
                    assert client_process.returncode == 0, stderr.decode()
                    assert token.encode() not in stdout + stderr
                    responses.append(json.loads(stdout))
                assert responses[0] == responses[1]
                assert not (state / "runs" / responses[0]["run_id"]).exists()
        finally:
            if native is not None:
                await native.shutdown(record_property)
            else:
                await shutdown_posix(process)

    async def shutdown_posix(process):
        if process.returncode is None:
            process.terminate()
        try:
            _stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.communicate(), 5)
            raise
        # Uvicorn restores/re-raises SIGTERM after lifespan cleanup. The
        # signal exit code is not evidence of an abandoned service.
        assert process.returncode in {0, -signal.SIGTERM}, stderr.decode()
        assert b"Application shutdown complete" in stderr

    asyncio.run(exercise())
    checked(
        [
            str(python),
            "-c",
            "import pathlib,sqlite3,sys; "
            "from wavedl_daemon.storage.ownership import acquire_dispatch_lock; "
            "root=pathlib.Path(sys.argv[1]); "
            "lock=acquire_dispatch_lock(root/'dispatch.lock'); "
            "lock.__enter__(); "
            "db=sqlite3.connect(root/'registry.db',timeout=1); "
            "assert db.execute('PRAGMA integrity_check').fetchone()==('ok',); "
            "db.execute('BEGIN IMMEDIATE'); db.rollback(); db.close(); "
            "lock.__exit__(None,None,None)",
            str(state),
        ],
        cwd=outside,
        environment=environment,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="Actual installed CPU training needs ML dependencies",
)
def test_installed_standalone_training_requires_no_daemon(installed):
    outside, _venv, binary, python, environment = installed
    dataset = outside / "cpu-data.npz"
    checked(
        [
            str(python),
            "-c",
            "import numpy as np,sys; r=np.random.default_rng(19); "
            "np.savez(sys.argv[1],X=r.normal(size=(20,32)).astype('float32'),y=r.normal(size=(20,1)).astype('float32'))",
            str(dataset),
        ],
        cwd=outside,
        environment=environment,
    )
    output = outside / "standalone-cpu"
    checked(
        [
            str(binary / ("wavedl-train.exe" if os.name == "nt" else "wavedl-train")),
            "--cpu",
            "--num_gpus",
            "1",
            "--mixed_precision",
            "no",
            "--precision",
            "no",
            "--model",
            "cnn",
            "--data_path",
            str(dataset),
            "--epochs",
            "1",
            "--batch_size",
            "8",
            "--workers",
            "0",
            "--output_dir",
            str(output),
        ],
        cwd=outside,
        environment=environment | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    rows = [
        json.loads(line)
        for line in (output / "attempts/1/events.jsonl").read_text().splitlines()
    ]
    assert rows[0]["type"] == "hello" and rows[-1]["type"] == "exit"
    assert rows[-1]["payload"]["code"] == 0
    assert list((output / "checkpoints").glob("gen-*/COMPLETE.json"))
