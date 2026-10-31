"""The only unauthenticated API exception consumes a same-origin capability."""

import asyncio

import httpx
import pytest

from wavedl_daemon.http.app import create_app
from wavedl_daemon.http.bootstrap import BootstrapCredentials


def request(app, path, *, code=None, headers=None, method="POST"):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8420"
        ) as client:
            return await client.request(
                method, path, json={"code": code}, headers=headers
            )

    response = asyncio.run(run())
    assert "set-cookie" not in response.headers
    return response


def test_single_use_exchange_does_not_open_health_or_cookie_auth():
    credentials = BootstrapCredentials("test-access-token")
    app = create_app(
        None,
        token="test-access-token",
        daemon_id="daemon",
        build_id="build",
        bootstrap_credentials=credentials,
    )
    code = credentials.mint()
    assert request(app, "/api/health", method="GET").status_code == 401
    result = request(
        app, "/api/bootstrap", code=code, headers={"Origin": "http://localhost:8420"}
    )
    assert result.status_code == 200
    body = result.json()
    assert body["token_type"] == "Bearer"
    assert body["daemon_id"] == "daemon"
    assert isinstance(body["expires_at"], int)
    assert body["access_token"] != "test-access-token"
    session = body["access_token"]
    assert result.headers["cache-control"] == "no-store"
    assert request(app, "/api/bootstrap", code=code).status_code == 401
    assert (
        request(
            app,
            "/api/health",
            method="GET",
            headers={"Cookie": "token=test-access-token"},
        ).status_code
        == 401
    )
    assert (
        request(
            app,
            "/api/health",
            method="GET",
            headers={"Authorization": "Bearer test-access-token"},
        ).status_code
        == 200
    )
    rotated = request(
        app, "/api/session", headers={"Authorization": f"Bearer {session}"}
    )
    assert rotated.status_code == 200
    assert rotated.json()["access_token"] != session
    assert (
        request(
            app, "/api/health", headers={"Authorization": f"Bearer {session}"}
        ).status_code
        == 401
    )
    assert (
        request(
            app,
            "/api/session",
            headers={"Authorization": f"Bearer {session}"},
        ).status_code
        == 401
    )
    assert (
        request(
            app,
            "/api/session",
            method="DELETE",
            headers={"Authorization": f"Bearer {rotated.json()['access_token']}"},
        ).status_code
        == 204
    )
    assert (
        request(
            app,
            "/api/health",
            headers={"Authorization": f"Bearer {rotated.json()['access_token']}"},
        ).status_code
        == 401
    )
    assert (
        request(
            app,
            "/api/session",
            method="DELETE",
            headers={"Authorization": f"Bearer {rotated.json()['access_token']}"},
        ).status_code
        == 401
    )
    master_rotated = request(
        app, "/api/session", headers={"Authorization": "Bearer test-access-token"}
    )
    assert master_rotated.status_code == 200
    assert master_rotated.json()["access_token"] != "test-access-token"


@pytest.mark.parametrize(
    "headers",
    [{"Host": "evil.example"}, {"Origin": "http://evil.example"}, {"Origin": "null"}],
)
def test_host_origin_rejection_does_not_consume_nonce(headers):
    credentials = BootstrapCredentials("test-access-token")
    app = create_app(
        None,
        token="test-access-token",
        daemon_id="daemon",
        build_id="build",
        bootstrap_credentials=credentials,
    )
    code = credentials.mint()
    assert request(app, "/api/bootstrap", code=code, headers=headers).status_code in {
        400,
        403,
    }
    assert request(app, "/api/bootstrap", code=code).status_code == 200


def test_bootstrap_disabled_and_factory_identity_mismatch_fail_closed():
    app = create_app(
        None, token="test-access-token", daemon_id="daemon", build_id="build"
    )
    assert request(app, "/api/bootstrap", code="anything").status_code == 401
    assert request(app, "/api/bootstrap/other", code="anything").status_code == 401
    with pytest.raises(ValueError, match="belong"):
        create_app(
            None,
            token="other-token",
            daemon_id="daemon",
            build_id="build",
            bootstrap_credentials=BootstrapCredentials("test-access-token"),
        )
