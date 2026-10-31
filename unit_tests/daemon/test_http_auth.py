"""The ASGI boundary never authenticates or emits ambient cookie credentials."""

import asyncio

import httpx
import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from wavedl_daemon.http.app import create_app, read_json
from wavedl_daemon.http.bootstrap import BootstrapCredentials
from wavedl_daemon.storage.settings import Settings


TOKEN = "private-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app(tmp_path):
    settings = Settings(state_root=tmp_path / "state", run_root=tmp_path / "runs")
    return create_app(
        settings, token=TOKEN, daemon_id="daemon-test", build_id="build-test"
    )


def request(app, method="GET", path="/api/health", *, headers=None, content=None):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:8420"
        ) as client:
            return await client.request(method, path, headers=headers, content=content)

    response = asyncio.run(run())
    assert "set-cookie" not in response.headers
    return response


def test_health_is_authenticated_minimal_identity_without_state_creation(app):
    assert request(app).status_code == 401
    response = request(app, headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {
        "daemon_id": "daemon-test",
        "build_id": "build-test",
        "protocol_min": 1,
        "protocol_max": 1,
    }
    assert TOKEN not in response.text
    assert not app.state.settings.state_root.exists()
    assert not app.state.settings.run_root.exists()


def test_master_can_rotate_session_when_bootstrap_exchange_is_disabled(app):
    response = request(app, method="POST", path="/api/session", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "Bearer"
    assert body["access_token"] != TOKEN
    assert body["daemon_id"] == "daemon-test"
    assert isinstance(body["expires_at"], int)
    assert (
        request(
            app,
            method="GET",
            path="/api/health",
            headers={"Authorization": f"Bearer {body['access_token']}"},
        ).status_code
        == 200
    )


@pytest.mark.parametrize(
    ("method", "path", "headers"),
    [
        ("GET", "/api/runs", {"Origin": "http://localhost:8421"}),
        (
            "POST",
            "/api/runs/12345678-1234-4234-8234-123456789abc/cancel",
            {"Origin": "http://localhost:8421", "Cookie": f"token={TOKEN}"},
        ),
        (
            "OPTIONS",
            "/api/runs/12345678-1234-4234-8234-123456789abc/cancel",
            {
                "Origin": "http://localhost:8421",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        ),
    ],
)
def test_foreign_origin_is_rejected_before_auth_and_cors(method, path, headers, app):
    response = request(app, method, path, headers=headers)
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


def test_expected_backend_identity_matches_and_header_remains_optional(app):
    response = request(app, headers=AUTH | {"X-WaveDL-Daemon-ID": "daemon-test"})
    assert response.status_code == 200
    assert response.json()["daemon_id"] == "daemon-test"
    assert request(app, headers=AUTH).status_code == 200


@pytest.mark.parametrize("identity", ["another-backend", "", "daemon-test "])
def test_mismatched_backend_identity_is_rejected_before_mutating_endpoint(
    app, identity
):
    effects = []

    async def mutate(request):
        effects.append("mutated")
        return JSONResponse({"ok": True})

    app.router.routes.append(Route("/api/mutate", mutate, methods=["POST"]))
    response = request(
        app,
        method="POST",
        path="/api/mutate",
        headers=AUTH | {"X-WaveDL-Daemon-ID": identity},
    )
    assert response.status_code == 409
    assert response.headers["x-wavedl-error-code"] == "daemon_identity_changed"
    assert effects == []
    assert TOKEN not in response.text and "daemon-test" not in response.text


def test_mismatched_daemon_identity_is_distinct_from_missing_auth(app):
    response = request(
        app,
        path="/api/health",
        headers={"X-WaveDL-Daemon-ID": "previous-daemon"},
    )
    assert response.status_code == 409
    assert response.json() == {"error": "Daemon identity changed"}
    assert response.headers["x-wavedl-error-code"] == "daemon_identity_changed"


def test_domain_conflict_does_not_claim_daemon_identity_change(app):
    async def conflict(request):
        raise HTTPException(409, "Plan conflict")

    app.router.routes.append(Route("/api/domain-conflict", conflict, methods=["POST"]))
    response = request(
        app,
        method="POST",
        path="/api/domain-conflict",
        headers=AUTH,
    )
    assert response.status_code == 409
    assert "x-wavedl-error-code" not in response.headers


@pytest.mark.parametrize(
    "identities", [("daemon-test", "daemon-test"), ("daemon-test", "foreign-backend")]
)
def test_duplicate_backend_identity_headers_are_rejected_before_mutation(
    app, identities
):
    effects = []

    async def mutate(request):
        effects.append("mutated")
        return JSONResponse({"ok": True})

    app.router.routes.append(Route("/api/mutate", mutate, methods=["POST"]))
    response = request(
        app,
        method="POST",
        path="/api/mutate",
        headers=[
            ("Authorization", f"Bearer {TOKEN}"),
            ("X-WaveDL-Daemon-ID", identities[0]),
            ("x-wavedl-daemon-id", identities[1]),
        ],
    )
    assert response.status_code == 409
    assert effects == []


def test_backend_identity_uses_current_owned_app_identity_not_constructor_fallback(app):
    app.state.daemon_id = "owned-registry-id"
    assert (
        request(app, headers=AUTH | {"X-WaveDL-Daemon-ID": "daemon-test"}).status_code
        == 409
    )
    assert (
        request(
            app, headers=AUTH | {"X-WaveDL-Daemon-ID": "owned-registry-id"}
        ).status_code
        == 200
    )
    assert (
        request(app, headers={"X-WaveDL-Daemon-ID": "owned-registry-id"}).status_code
        == 401
    )


@pytest.mark.parametrize("identity", [None, "daemon-test"])
def test_matching_or_legacy_identity_allows_authenticated_mutation(app, identity):
    effects = []

    async def mutate(request):
        effects.append("mutated")
        return JSONResponse({"ok": True})

    app.router.routes.append(Route("/api/mutate", mutate, methods=["POST"]))
    headers = AUTH if identity is None else AUTH | {"X-WaveDL-Daemon-ID": identity}
    response = request(app, method="POST", path="/api/mutate", headers=headers)
    assert response.status_code == 200
    assert effects == ["mutated"]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "Basic private-test-token",
        "Bearer",
        "Bearer  private-test-token",
        "Bearer private-test-token ",
        "Bearer private-test-token, Bearer private-test-token",
        "Bearer wrong",
        "Bearer\tprivate-test-token",
        "Bearer privé",
    ],
)
def test_malformed_or_wrong_authorization_is_rejected(app, value):
    response = request(app, headers=[(b"authorization", value.encode("utf-8"))])
    assert response.status_code == 401
    assert "daemon-test" not in response.text
    assert "build-test" not in response.text
    assert TOKEN not in response.text


def test_mixed_case_bearer_scheme_and_header_are_accepted(app):
    assert request(app, headers={"aUtHoRiZaTiOn": f"bEaReR {TOKEN}"}).status_code == 200


def test_cookie_and_query_credentials_never_authenticate(app):
    assert (
        request(
            app,
            path=f"/api/health?token={TOKEN}",
            headers={"Cookie": f"token={TOKEN}; session={TOKEN}"},
        ).status_code
        == 401
    )


def test_session_auth_does_not_grant_runtime_event_ingest(tmp_path):
    credentials = BootstrapCredentials(TOKEN)
    app = create_app(
        None,
        token=TOKEN,
        daemon_id="daemon-test",
        build_id="build-test",
        bootstrap_credentials=credentials,
    )

    async def ingest(request):
        return JSONResponse({"ok": True})

    app.router.routes.append(
        Route(
            "/api/runs/12345678-1234-4234-8234-123456789abc/events",
            ingest,
            methods=["POST"],
        )
    )
    app.router.routes.append(
        Route(
            "/api/runs/12345678-1234-4234-8234-123456789abc/events/",
            ingest,
            methods=["POST"],
        )
    )
    session = credentials.mint_session()
    path = "/api/runs/12345678-1234-4234-8234-123456789abc/events"
    response = request(
        app,
        method="POST",
        path=path,
        headers={"Authorization": f"Bearer {session.token}"},
    )
    assert response.status_code == 403
    assert request(app, method="POST", path=path, headers=AUTH).status_code == 200
    trailing = path + "/"
    assert (
        request(
            app,
            method="POST",
            path=trailing,
            headers={"Authorization": f"Bearer {session.token}"},
        ).status_code
        == 403
    )
    assert request(app, method="POST", path=trailing, headers=AUTH).status_code == 200


def test_multiple_authorization_headers_are_rejected(app):
    assert (
        request(
            app,
            headers=[
                ("authorization", f"Bearer {TOKEN}"),
                ("authorization", f"Bearer {TOKEN}"),
            ],
        ).status_code
        == 401
    )


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "localhost.evil",
        "localhost@evil",
        "localhost:",
        "localhost:0",
        "localhost:99999",
        "localhost:08420",
        "localhost.",
        "localhost,evil",
        "localhost\x00",
        "localhöst",
        "::1",
        "[::1]junk",
        "localhost:8420:9",
    ],
)
def test_disallowed_and_malformed_hosts_cover_health_and_mutations(app, host):
    headers = [
        (b"host", host.encode("utf-8")),
        (b"authorization", f"Bearer {TOKEN}".encode()),
    ]
    assert request(app, headers=headers).status_code == 400
    assert (
        request(app, method="POST", path="/api/runs", headers=headers).status_code
        == 400
    )


def test_duplicate_host_is_rejected(app):
    assert (
        request(
            app,
            headers=[("Host", "localhost"), ("Host", "evil.example"), *AUTH.items()],
        ).status_code
        == 400
    )


def test_missing_host_is_rejected_using_raw_asgi(app):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "GET",
        "path": "/api/health",
        "raw_path": b"/api/health",
        "query_string": b"",
        "headers": [(b"authorization", f"Bearer {TOKEN}".encode())],
        "server": ("localhost", 8420),
        "client": ("127.0.0.1", 1234),
    }
    asyncio.run(app(scope, receive, send))
    assert messages[0]["status"] == 400


@pytest.mark.parametrize(
    "host", ["localhost:8420", "127.0.0.1:8420", "[::1]:8420", "LOCALHOST:8420"]
)
def test_explicit_loopback_hosts_are_accepted(app, host):
    assert request(app, headers=AUTH | {"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "http://evil.example",
        "http://localhost:8421",
        "https://localhost:8420",
        "http://localhost:8420/",
        "http://localhost:8420?foo",
        "http://user@localhost:8420",
        " http://localhost:8420",
        "http://localhost:8420\t",
    ],
)
def test_foreign_or_malformed_origin_is_rejected(app, origin):
    assert request(app, headers=AUTH | {"Origin": origin}).status_code == 403


def test_same_origin_is_accepted_and_duplicate_origin_rejected(app):
    assert (
        request(app, headers=AUTH | {"Origin": "http://localhost:8420"}).status_code
        == 200
    )
    assert (
        request(
            app,
            headers=[
                *AUTH.items(),
                ("origin", "http://localhost:8420"),
                ("origin", "http://localhost:8420"),
            ],
        ).status_code
        == 403
    )


def test_origin_default_port_is_normalized(app):
    assert (
        request(
            app, headers=AUTH | {"Host": "localhost", "Origin": "http://localhost:80"}
        ).status_code
        == 200
    )


def test_port_bound_explicit_allowlist(tmp_path):
    app = create_app(
        None,
        token=TOKEN,
        daemon_id="daemon",
        build_id="build",
        allowed_hosts=("example.test:8443",),
    )
    assert request(app, headers=AUTH | {"Host": "example.test:8443"}).status_code == 200
    assert request(app, headers=AUTH | {"Host": "example.test:8444"}).status_code == 400


def test_api_not_found_and_errors_do_not_bypass_security(app):
    assert request(app, path="/api/missing").status_code == 401
    assert request(app, path="/api/missing", headers=AUTH).status_code == 404
    assert (
        request(app, path="/missing", headers={"Host": "evil.example"}).status_code
        == 400
    )
    assert request(app, method="POST", headers=AUTH).status_code == 405


def test_cookie_headers_removed_even_from_future_routes(app):
    async def cookie(request):
        response = JSONResponse({"ok": True})
        response.set_cookie("forbidden", "value")
        return response

    app.router.routes.append(Route("/api/cookie", cookie))
    assert request(app, path="/api/cookie", headers=AUTH).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b'{"token": NaN}',
        b'{"token": Infinity}',
        b'{"a": 1e999}',
        b'{"a":1,"a":2}',
        b"[]",
        b'{"a":"\xff"}',
        b'{"token":"secret"}garbage',
    ],
)
def test_strict_json_errors_are_generic(app, body):
    async def parse(request):
        return JSONResponse(await read_json(request))

    app.router.routes.append(Route("/api/json", parse, methods=["POST"]))
    response = request(
        app,
        method="POST",
        path="/api/json",
        headers=AUTH | {"Content-Type": "application/json"},
        content=body,
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Bad request"}


def test_json_body_is_bounded_and_valid_json_accepted(app):
    async def parse(request):
        return JSONResponse(await read_json(request, max_bytes=32))

    app.router.routes.append(Route("/api/json", parse, methods=["POST"]))
    assert (
        request(
            app, method="POST", path="/api/json", headers=AUTH, content=b"{}"
        ).status_code
        == 415
    )
    headers = AUTH | {"Content-Type": "application/json"}
    assert request(
        app, method="POST", path="/api/json", headers=headers, content=b'{"a": 1}'
    ).json() == {"a": 1}
    assert (
        request(
            app,
            method="POST",
            path="/api/json",
            headers=headers,
            content=b'{"a":"' + b"x" * 40 + b'"}',
        ).status_code
        == 413
    )


def test_json_body_read_deadline_rejects_stalled_receive():
    async def receive():
        await asyncio.sleep(10)
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/json",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    async def parse():
        return await read_json(request, timeout_seconds=0.001)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(parse())
    assert caught.value.status_code == 408
    assert str(caught.value) == "408: Request body timeout"


def test_json_body_read_disconnect_is_sanitized():
    messages = iter(
        (
            {"type": "http.request", "body": b'{"partial":', "more_body": True},
            {"type": "http.disconnect"},
        )
    )

    async def receive():
        return next(messages)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/json",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(read_json(request))
    assert caught.value.status_code == 408
    assert str(caught.value) == "408: Request body disconnected"


def test_partial_json_deadline_is_sanitized_and_does_not_invoke_mutation(app):
    effects = []
    cancelled = []

    async def mutate(incoming):
        await read_json(incoming, timeout_seconds=0.01)
        effects.append("mutated")
        return JSONResponse({"ok": True})

    async def body():
        try:
            yield b'{"private-body-secret":"unfinished'
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    app.router.routes.append(Route("/api/slow-json", mutate, methods=["POST"]))

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8420"
        ) as client:
            return await asyncio.wait_for(
                client.post(
                    "/api/slow-json",
                    headers=AUTH | {"Content-Type": "application/json"},
                    content=body(),
                ),
                timeout=2,
            )

    response = asyncio.run(run())
    assert response.status_code == 408
    assert response.json() == {"error": "Request timeout"}
    assert cancelled == [True]
    assert effects == []
    assert "private-body-secret" not in response.text and TOKEN not in response.text


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_invalid_json_body_deadline_refuses_before_receiving(timeout):
    async def receive():
        pytest.fail("Invalid deadline reached request receive")

    incoming = Request(
        {"type": "http", "headers": [(b"content-type", b"application/json")]},
        receive,
    )
    with pytest.raises(ValueError, match="positive and finite"):
        asyncio.run(read_json(incoming, timeout_seconds=timeout))


def test_internal_error_contains_no_details(app):
    async def failure(request):
        raise RuntimeError(f"private run id and {TOKEN}")

    app.router.routes.append(Route("/api/fail", failure))
    response = request(app, path="/api/fail", headers=AUTH)
    assert response.status_code == 500
    assert response.json() == {"error": "Internal server error"}


@pytest.mark.parametrize("token", ["", "with space", "nonascii-é", "line\nbreak"])
def test_invalid_factory_credential_is_rejected(token):
    with pytest.raises(ValueError, match="credential"):
        create_app(None, token=token, daemon_id="daemon", build_id="build")
