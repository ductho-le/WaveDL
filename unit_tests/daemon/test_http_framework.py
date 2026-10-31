"""The FastAPI shell preserves the daemon's HTTP boundary and client routing."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict
from starlette.requests import Request  # noqa: TC002 - FastAPI resolves it at runtime

from wavedl_daemon.http import app as http_app
from wavedl_daemon.http.app import create_app
from wavedl_daemon.storage.settings import Settings


TOKEN = "private-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class ProbeBody(BaseModel):
    model_config = ConfigDict(strict=True)

    n: int


@pytest.fixture
def app(tmp_path):
    settings = Settings(state_root=tmp_path / "state", run_root=tmp_path / "runs")
    return create_app(
        settings,
        token=TOKEN,
        daemon_id="daemon-test",
        build_id="build-test",
        web_assets=None,
    )


def request(app, method="GET", path="/api/health", *, content=None):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:8420"
        ) as client:
            return await client.request(
                method,
                path,
                headers=AUTH | {"Content-Type": "application/json"},
                content=content,
            )

    return asyncio.run(run())


def test_create_app_is_fastapi(app):
    assert isinstance(app, FastAPI)


def test_create_app_requires_full_services_protocol(fake_services):
    # Everything except preview conforms, so this cannot pass on another member.
    fake_services.preview = None
    with pytest.raises(TypeError, match="preview"):
        create_app(
            None,
            token=TOKEN,
            daemon_id="daemon-test",
            build_id="build-test",
            services=fake_services,
        )


def test_session_routes_are_api_routes():
    from wavedl_daemon.http.bootstrap import BootstrapCredentials

    app = create_app(
        None,
        token=TOKEN,
        daemon_id="daemon-test",
        build_id="build-test",
        bootstrap_credentials=BootstrapCredentials(TOKEN),
    )
    routes = [route for route in app.routes if route.path.startswith("/api")]
    assert len(routes) == 4
    assert all(isinstance(route, APIRoute) for route in routes)


def hosted_app(fake_services):
    return create_app(
        None,
        token=TOKEN,
        daemon_id="daemon-test",
        build_id="build-test",
        services=fake_services,
    )


def test_dataset_routes_are_api_routes(fake_services):
    paths = {
        "/api/runs/adopt",
        "/api/models",
        "/api/browse",
        "/api/browse/resolve",
        "/api/datasets/inspect",
        "/api/datasets/preview",
        "/api/host",
    }
    routes = [
        route for route in hosted_app(fake_services).routes if route.path in paths
    ]
    assert len(routes) == len(paths)
    assert all(isinstance(route, APIRoute) for route in routes)


def test_browse_rejects_repeated_query_key(fake_services):
    response = request(
        hosted_app(fake_services), path="/api/browse?root=data&root=data"
    )
    assert response.status_code == 400
    fake_services.catalog.browse.assert_not_called()


def test_models_rejects_any_query(fake_services):
    response = request(hosted_app(fake_services), path="/api/models?private=secret")
    assert response.status_code == 400
    fake_services.catalog.models.assert_not_called()


@pytest.mark.parametrize("query", ["limit=1.0", "offset=0.0"])
def test_browse_preserves_integer_pagination_syntax(fake_services, query):
    response = request(hosted_app(fake_services), path=f"/api/browse?root=data&{query}")
    assert response.status_code == 400
    fake_services.catalog.browse.assert_not_called()


def test_inspect_cancelled_on_disconnect(fake_services):
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def inspect_dataset(root, path):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        fake_services.inspect = inspect_dataset
        app = hosted_app(fake_services)
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {
                    "type": "http.request",
                    "more_body": False,
                    "body": b'{"root":"data","path":"data.npy"}',
                }
            await started.wait()
            return {"type": "http.disconnect"}

        sent = []

        async def send(message):
            sent.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "scheme": "http",
                "method": "POST",
                "path": "/api/datasets/inspect",
                "query_string": b"",
                "headers": [
                    (b"host", b"localhost:8420"),
                    (b"authorization", f"Bearer {TOKEN}".encode()),
                    (b"content-type", b"application/json"),
                ],
            },
            receive,
            send,
        )
        assert cancelled.is_set()
        assert sent[0]["status"] == 503

    asyncio.run(scenario())


def test_artifact_routes_are_api_routes(fake_services):
    routes = [
        route
        for route in hosted_app(fake_services).routes
        if "/artifacts" in route.path
    ]
    assert len(routes) == 2
    assert all(isinstance(route, APIRoute) for route in routes)


def test_event_routes_are_api_routes(fake_services):
    routes = [
        route
        for route in hosted_app(fake_services).routes
        if route.path.endswith("/events")
    ]
    assert len(routes) == 2
    assert all(isinstance(route, APIRoute) for route in routes)


@pytest.mark.parametrize("malformed", [None, [], "private-secret", 42, True, {}])
def test_event_malformed_envelope_is_rejected_by_ingestor(fake_services, malformed):
    run_id = "11111111-1111-1111-1111-111111111111"
    response = request(
        hosted_app(fake_services),
        method="POST",
        path=f"/api/runs/{run_id}/events",
        content=json.dumps({"attempt": 1, "events": [malformed]}),
    )
    assert response.status_code == 409
    assert response.content == b'{"error":"Conflict"}'
    fake_services.ingestor.reject_runtime_events.assert_called_once()
    args = fake_services.ingestor.reject_runtime_events.call_args.args
    assert args[:3] == (run_id, run_id, 1)
    fake_services.ingestor.push.assert_not_called()


def test_event_push_ignores_extra_top_level_key(fake_services):
    from wavedl.core.protocol import Envelope, encode

    run_id = "11111111-1111-1111-1111-111111111111"
    event = Envelope(
        run_id,
        1,
        "log",
        0,
        "2026-10-31T00:00:00Z",
        {"message": "message", "level": "info", "stream": "stdout"},
    )
    fake_services.ingestor.push.return_value = 0
    response = request(
        hosted_app(fake_services),
        method="POST",
        path=f"/api/runs/{run_id}/events",
        content=json.dumps(
            {"attempt": 1, "events": [json.loads(encode(event))], "extra": True}
        ),
    )
    assert response.status_code == 200
    assert response.content == b'{"attempt":1,"seq":0}'
    assert "cache-control" not in response.headers
    fake_services.ingestor.push.assert_called_once_with(run_id, run_id, 1, [event])
    fake_services.ingestor.reject_runtime_events.assert_not_called()


@pytest.mark.parametrize("content", [b"not-json", b" " * (1048576 + 1)])
def test_event_push_checks_run_identity_before_reading_body(fake_services, content):
    response = request(
        hosted_app(fake_services),
        method="POST",
        path="/api/runs/not-a-uuid/events",
        content=content,
    )
    assert response.status_code == 400
    fake_services.ingestor.reject_runtime_events.assert_not_called()
    fake_services.ingestor.push.assert_not_called()


def test_every_api_route_is_fastapi(fake_services):
    routes = [
        route
        for route in hosted_app(fake_services).routes
        if route.path.startswith("/api")
    ]
    assert (
        len(routes) == 20
    )  # Bootstrap is separately covered with credentials enabled.
    assert all(isinstance(route, APIRoute) for route in routes)


@pytest.mark.parametrize(
    ("path", "limit"),
    [
        ("/api/bootstrap", 4096),
        ("/api/runs/plan", 65536),
        ("/api/runs", 4096),
        ("/api/runs/adopt", 8192),
        ("/api/browse/resolve", 8192),
        ("/api/datasets/inspect", 16384),
        ("/api/datasets/preview", 65536),
        ("/api/runs/11111111-1111-1111-1111-111111111111/events", 1048576),
    ],
)
@pytest.mark.parametrize("case", ["duplicate", "nan", "oversize"])
def test_strict_json_on_every_body_route(fake_services, path, limit, case):
    from wavedl_daemon.http.bootstrap import BootstrapCredentials

    app = create_app(
        None,
        token=TOKEN,
        daemon_id="daemon-test",
        build_id="build-test",
        services=fake_services,
        bootstrap_credentials=BootstrapCredentials(TOKEN),
    )
    # Bodyless POST /session and /runs/{id}/cancel must not enter this gate.
    body = {
        "duplicate": b'{"attempt":1,"attempt":2}',
        "nan": b'{"attempt":NaN}',
        "oversize": b"{" + b" " * limit,
    }[case]
    response = request(app, method="POST", path=path, content=body)
    assert response.status_code == (413 if case == "oversize" else 400)


@pytest.mark.parametrize("group", ["inspect", "preview"])
def test_dataset_refusal_is_bad_request(fake_services, group):
    from wavedl.core.dataset_headers import InspectionRefused
    from wavedl_daemon.runs.preview_service import PreviewRefused

    async def inspect_dataset(root, path):
        raise InspectionRefused("private-secret")

    async def preview_dataset(body):
        raise PreviewRefused("private-secret")

    fake_services.inspect = inspect_dataset
    fake_services.preview = preview_dataset
    body = {"root": "data", "path": "data.npz"}
    if group == "preview":
        body["view"] = "all"
    response = request(
        hosted_app(fake_services),
        method="POST",
        path=f"/api/datasets/{group}",
        content=json.dumps(body),
    )
    assert response.status_code == 400
    assert response.content == b'{"error":"Bad request"}'


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
def test_no_public_openapi_or_docs(app, path):
    response = request(app, path=path)
    if response.headers.get("content-type", "").startswith("application/json"):
        assert "openapi" not in response.json()
    assert "swagger-ui" not in response.text.lower()
    assert "redoc" not in response.text.lower()


def test_request_validation_error_is_bad_request(app):
    async def probe(body: ProbeBody):
        return {"n": body.n}

    app.router.routes.append(APIRoute("/api/probe", probe, methods=["POST"]))
    response = request(app, method="POST", path="/api/probe", content='{"n":"x"}')
    assert response.status_code == 400
    assert response.json() == {"error": "Bad request"}


def test_api_route_wins_over_client_catch_all(app):
    response = request(app)
    assert response.status_code == 200
    assert response.json() == {
        "daemon_id": "daemon-test",
        "build_id": "build-test",
        "protocol_min": 1,
        "protocol_max": 1,
    }
    assert response.headers["cache-control"] == "no-store"


def strict_probe(app, endpoint):
    router = APIRouter()
    router.add_api_route(
        "/api/probe",
        endpoint,
        methods=["POST"],
        route_class_override=http_app.strict_json_route(32),
    )
    app.router.routes.extend(router.routes)


@pytest.mark.parametrize(
    ("body", "content_type", "status"),
    [
        (b'{"n":1,"n":2}', "application/json", 400),
        (b'{"n":NaN}', "application/json", 400),
        (b'{"n":Infinity}', "application/json", 400),
        (b'{"n":1e999}', "application/json", 400),
        (b"[1]", "application/json", 400),
        (b'{"n":"\xff"}', "application/json", 400),
        (b'{"private-secret":', "application/json", 400),
        (b'{"n":"1"}', "application/json", 400),
        (b'{"n":1}' + b" " * 26, "application/json", 413),
        (b'{"n":1}', "text/plain", 415),
        (b'{"n":1}', "application/json", 200),
    ],
)
def test_strict_route_json_body(app, body, content_type, status):
    received = []

    async def probe(body: ProbeBody):
        received.append(body.n)
        return {"n": body.n}

    strict_probe(app, probe)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8420"
        ) as client:
            return await client.post(
                "/api/probe",
                headers=AUTH | {"Content-Type": content_type},
                content=body,
            )

    response = asyncio.run(run())
    assert response.status_code == status
    if status == 200:
        assert response.json() == {"n": 1}
        assert received == [1]  # The endpoint only consumes the Pydantic model.
    else:
        assert received == []
        assert response.json() == {
            "error": {
                400: "Bad request",
                413: "Request too large",
                415: "Expected JSON",
            }[status]
        }


async def call_probe(app, receive, *, content_type=b"application/json"):
    messages = []

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/probe",
            "query_string": b"",
            "headers": [
                (b"host", b"localhost:8420"),
                (b"authorization", f"Bearer {TOKEN}".encode()),
                (b"content-type", content_type),
            ],
        },
        receive,
        send,
    )
    return messages


@pytest.mark.parametrize(
    ("mode", "status", "error"),
    [
        ("content-type", 415, "Expected JSON"),
        ("oversize", 413, "Request too large"),
        ("timeout", 408, "Request timeout"),
        ("disconnect", 408, "Request timeout"),
    ],
)
def test_strict_route_body_timeout_is_408(app, monkeypatch, mode, status, error):
    original = http_app.read_json_bytes

    async def short_read(incoming, *, max_bytes):
        return await original(incoming, max_bytes=max_bytes, timeout_seconds=0.001)

    monkeypatch.setattr(http_app, "read_json_bytes", short_read)

    async def probe(body: ProbeBody):
        pytest.fail("Invalid body reached endpoint")

    strict_probe(app, probe)
    calls = []

    async def receive():
        calls.append(True)
        if mode == "content-type":
            pytest.fail("Content type must be checked before receiving")
        if len(calls) == 1:
            return {
                "type": "http.request",
                "body": b"x" * 33 if mode == "oversize" else b'{"private-secret":',
                "more_body": True,
            }
        if mode == "disconnect":
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    messages = asyncio.run(
        call_probe(
            app,
            receive,
            content_type=b"text/plain"
            if mode == "content-type"
            else b"application/json",
        )
    )
    assert messages[0]["status"] == status
    assert messages[1]["body"] == ('{"error":"' + error + '"}').encode()
    assert len(calls) == {"content-type": 0, "oversize": 1}.get(mode, 2)


def test_strict_route_replays_body_once_then_delegates_disconnect(app):
    received = []

    async def probe(incoming: Request):
        received.append(await incoming.receive())
        received.append(await incoming.receive())
        return {"ok": True}

    strict_probe(app, probe)
    messages = iter(
        [
            {"type": "http.request", "body": b'{"n":', "more_body": True},
            {"type": "http.request", "body": b"1}", "more_body": False},
            {"type": "http.disconnect"},
        ]
    )

    async def receive():
        return next(messages)

    response = asyncio.run(call_probe(app, receive))
    assert response[0]["status"] == 200
    assert received == [
        {"type": "http.request", "body": b'{"n":1}', "more_body": False},
        {"type": "http.disconnect"},
    ]
