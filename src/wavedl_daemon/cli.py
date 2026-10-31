# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Explicit daemon startup; no listener, threads or storage at import time."""

from __future__ import annotations

import argparse
import asyncio
import errno
import ipaddress
import os
import secrets
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from uuid import uuid4

from wavedl import __version__
from wavedl.core.credentials import read_credential
from wavedl.core.live_push import LivePushConfig
from wavedl.core.protocol import strict_loads
from wavedl.core.safe_files import open_directory
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.backends.scheduler import SchedulerCommands
from wavedl_daemon.backends.slurm import SlurmBackend, SlurmSite
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.serve_state import (
    AlreadyRunning,
    check_not_running,
    remove_run_url,
    write_run_url,
)
from wavedl_daemon.storage.settings import Settings, default_state_dir


DEFAULT_PORTS = (8420, 8421)


class PortUnavailable(OSError):
    """None of the requested listener ports could be bound."""

    def __init__(self, ports: tuple[int, ...]):
        super().__init__("Requested ports are in use")
        self.ports = ports


def build_parser():
    parser = argparse.ArgumentParser(prog="wavedl serve")
    serve = parser
    serve.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--state-root", type=Path, default=default_state_dir())
    serve.add_argument("--run-root", type=Path)
    serve.add_argument("--credential-file", type=Path)
    serve.add_argument(
        "--slurm-site",
        type=Path,
        help="Explicit trusted SlurmSite JSON; requires shared run root",
    )
    serve.add_argument(
        "--browse-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Explicit additional filesystem browse authority",
    )
    serve.add_argument("--max-workers", type=int, default=4)
    serve.add_argument("--max-pending", type=int, default=8)
    serve.add_argument("--work-timeout", type=float, default=30)
    return parser


def _site(path):
    if path is None:
        return None
    path = Path(path).expanduser().absolute()
    with (
        open_directory(path.parent) as directory,
        directory.open_file(path.name) as stream,
    ):
        stream.verify_private_owner()
        if stream.stat().size > 65536:
            raise ValueError("Slurm site exceeds byte bound")
        value = strict_loads(stream.read(65537))
        stream.guard()
    if type(value) is not dict:
        raise ValueError("Slurm site must be an object")
    setup = value.get("setup_commands", [])
    if type(setup) is not list or len(setup) > 32:
        raise ValueError("Invalid trusted setup commands")
    value["setup_commands"] = tuple(setup)
    try:
        return SlurmSite(**value)
    except TypeError:
        raise ValueError("Invalid Slurm site fields") from None


async def serve(
    options,
    *,
    output=None,
    services_factory=None,
    server_factory=None,
    open_browser=None,
):
    """Pre-bind before storage mutation; lifespan owns actual draining/cleanup.

    Only loopback binding is supported. Remote clients use SSH forwarding or an
    explicitly managed HTTPS tunnel; forwarded headers are never trusted.
    """
    output = output or sys.stdout
    if (
        options.host not in {"127.0.0.1", "::1"}
        or not ipaddress.ip_address(options.host).is_loopback
    ):
        raise ValueError("Daemon must bind a loopback address")
    if options.port is not None and (
        type(options.port) is not int or not 0 <= options.port <= 65535
    ):
        raise ValueError("Port must be between 0 and 65535")
    budget = WorkBudget(
        max_workers=options.max_workers,
        max_pending=options.max_pending,
        timeout_seconds=options.work_timeout,
    )
    site = _site(options.slurm_site)
    if site is not None and os.name != "posix":
        raise ValueError("Slurm execution requires a POSIX scheduler host")
    settings = Settings(
        options.state_root,
        options.run_root or options.state_root / "runs",
        shared_execution=site is not None,
    )
    browse_roots = {"runs": settings.run_root}
    for item in options.browse_root:
        name, separator, path = item.partition("=")
        if not separator or name in browse_roots or not Path(path).is_absolute():
            raise ValueError("Browse authority must be a unique name and absolute path")
        browse_roots[name] = Path(path)
    check_not_running(settings.state_root)
    token = (
        read_credential(options.credential_file)
        if options.credential_file
        else secrets.token_urlsafe(32)
    )
    import uvicorn

    from wavedl_daemon.http.app import create_app
    from wavedl_daemon.http.bootstrap import BootstrapCredentials
    from wavedl_daemon.runs.runtime_service import HostedServices

    bootstrap = BootstrapCredentials(token)
    family = socket.AF_INET6 if ":" in options.host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        # Do not use SO_REUSEPORT: one listener must have one dispatch owner.
        ports = DEFAULT_PORTS if options.port is None else (options.port,)
        for candidate in ports:
            try:
                listener.bind((options.host, candidate))
                break
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
        else:
            raise PortUnavailable(ports)
        listener.listen(128)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        authority = (
            f"[{options.host}]:{port}"
            if family == socket.AF_INET6
            else f"{options.host}:{port}"
        )
        if services_factory is not None:
            services = services_factory(settings, budget=budget)
        else:
            live_push = LivePushConfig(f"http://{authority}", token)

            def backends(lock):
                result = {
                    "local": LocalBackend(dispatch_lock=lock, live_push=live_push)
                }
                if site is not None:
                    import pwd

                    result["slurm"] = SlurmBackend(
                        site,
                        SchedulerCommands(cwd=settings.state_root, budget=budget),
                        owner=pwd.getpwuid(os.getuid()).pw_name,
                        dispatch_lock=lock,
                    )
                return result

            services = HostedServices(
                settings,
                budget=budget,
                backend_factory=backends,
                browse_roots=browse_roots,
            )
        app = create_app(
            settings,
            token=token,
            daemon_id=str(uuid4()),
            build_id=__version__,
            services=services,
            allowed_hosts=(authority, f"localhost:{port}"),
            bootstrap_credentials=bootstrap,
        )

        class ReadyServer(uvicorn.Server):
            async def shutdown(self, sockets=None):
                # Uvicorn re-raises a captured SIGTERM after serve() returns,
                # killing the process before the outer finally can run.
                try:
                    remove_run_url(settings.state_root)
                except OSError:
                    pass
                await super().shutdown(sockets=sockets)

            async def startup(self, sockets=None):
                await super().startup(sockets=sockets)
                if self.started:
                    code = bootstrap.mint_launch()
                    link = f"http://{authority}/?bootstrap={code}"
                    print(
                        f"WaveDL is running at {link}",
                        file=output,
                        flush=True,
                    )
                    print(
                        "Open that link in your browser if it did not open automatically. "
                        "Press Ctrl+C to stop.",
                        file=output,
                        flush=True,
                    )
                    try:
                        write_run_url(settings.state_root, f"http://{authority}/")
                    except OSError:
                        # Only "already running" messages read it; never fatal.
                        pass
                    if open_browser is not None:

                        def launch_browser():
                            try:
                                open_browser(link)
                            except Exception:
                                pass

                        threading.Thread(target=launch_browser, daemon=True).start()

        config = uvicorn.Config(
            app,
            host=options.host,
            port=port,
            lifespan="on",
            proxy_headers=False,
            access_log=False,
            ws="none",
            server_header=False,
            limit_concurrency=256,
            timeout_keep_alive=5,
            timeout_graceful_shutdown=10,
            h11_max_incomplete_event_size=16384,
        )
        server = (server_factory or ReadyServer)(config)

        async def close_owned_resources():
            try:
                lifespan = getattr(server, "lifespan", None)
                if (
                    lifespan is not None
                    and lifespan.startup_event.is_set()
                    and not lifespan.shutdown_event.is_set()
                ):
                    if hasattr(server, "servers"):
                        await server.shutdown(sockets=[listener])
                    else:
                        await lifespan.shutdown()
            finally:
                await services.close()

        try:
            await server.serve(sockets=[listener])
        finally:
            try:
                remove_run_url(settings.state_root)
            except OSError:
                pass
            # Uvicorn's mainloop/startup failures do not always run shutdown.
            # Explicit ownership also covers caller cancellation and broken
            # startup output. HostedServices.close is idempotent and draining.
            cleanup = asyncio.create_task(close_owned_resources())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
        if not server.started:
            raise RuntimeError("Daemon startup failed")


def main(argv=None, *, open_browser=webbrowser.open):
    options = build_parser().parse_args(argv)
    try:
        asyncio.run(serve(options, open_browser=open_browser))
    except KeyboardInterrupt:
        return 130
    except AlreadyRunning as error:
        print(
            f"WaveDL is already running at {error.url}"
            if error.url
            else "WaveDL is already running.",
            file=sys.stderr,
        )
        return 1
    except PortUnavailable as error:
        print(
            "Ports 8420 and 8421 are both in use. Close the program using them "
            "or choose another port with --port."
            if options.port is None
            else f"Port {error.ports[0]} is in use. Choose another port with --port.",
            file=sys.stderr,
        )
        return 1
    except PermissionError:
        print(
            "WaveDL cannot use its state folder. Check that it belongs to you "
            "and is private, or choose another with --state-root.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, RuntimeError, ImportError):
        print(
            "Daemon unavailable: check listener, private credentials and storage.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
