"""The optional packaged shell cannot shadow API authority or escape its root."""

import asyncio
import hashlib

import httpx
import pytest

from wavedl_daemon.http.app import create_app
from wavedl_daemon.storage.settings import Settings


def make_app(tmp_path, root):
    return create_app(
        Settings(state_root=tmp_path / "state", run_root=tmp_path / "runs"),
        token="static-test-token",
        daemon_id="daemon-static",
        build_id="build-static",
        web_assets=root,
    )


def request(app, path="/", *, method="GET", headers=None):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            return await client.request(method, path, headers=headers)

    return asyncio.run(run())


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "index.html").write_text(
        '<html><script type="application/json">{"hello":1}</script></html>'
    )
    (root / "sw.js").write_text("worker")
    (root / "manifest.webmanifest").write_text("{}")
    return root


def test_public_shell_and_deep_link_head_have_security_headers(tmp_path, assets):
    app = make_app(tmp_path, assets)
    shell = request(app)
    deep = request(app, "/runs/run123", headers={"Accept": "text/html"})
    assert shell.status_code == deep.status_code == 200
    assert shell.content == deep.content
    for path in (
        "/experiments/00000000-0000-4000-8000-000000000001",
        "/results/00000000-0000-4000-8000-000000000001",
    ):
        routed = request(app, path, headers={"Accept": "text/html"})
        assert routed.status_code == 200
        assert routed.content == shell.content
    assert (
        request(app, "/runs/", headers={"Accept": "text/html"}).content == shell.content
    )
    assert shell.headers["cache-control"] == "no-store"
    assert "'sha256-" in shell.headers["content-security-policy"]
    assert "unsafe-eval" not in shell.headers["content-security-policy"]
    assert shell.headers["referrer-policy"] == "no-referrer"
    head = request(app, "/", method="HEAD")
    assert head.content == b""
    assert head.headers["content-length"] == str(len(shell.content))
    assert not app.state.settings.state_root.exists()


def test_api_unknown_never_returns_shell(tmp_path, assets):
    app = make_app(tmp_path, assets)
    for path in ("/api", "/api/unknown", "/api/runs/unknown/path"):
        assert request(app, path).status_code == 401
        response = request(
            app,
            path,
            headers={
                "Authorization": "Bearer static-test-token",
                "Accept": "text/html",
            },
        )
        assert response.status_code == 404
        assert response.headers["content-type"] == "application/json"
    assert request(app, "/", method="POST").status_code == 405


def test_missing_build_is_explicit_and_app_construction_is_inert(tmp_path):
    root = tmp_path / "missing"
    app = make_app(tmp_path, root)
    assert not root.exists()
    assert request(app).status_code == 503
    assert "Generate and vendor" in request(app).text
    assert request(app, "/missing.js").status_code == 404
    assert not root.exists()


def test_only_content_verified_hashes_are_immutable(tmp_path, assets):
    raw = b"export const version = 1;"
    name = "bundle." + hashlib.sha256(raw).hexdigest()[:16] + ".js"
    (assets / name).write_bytes(raw)
    (assets / "bundle.aaaaaaaaaaaa.js").write_bytes(raw)
    app = make_app(tmp_path, assets)
    assert "immutable" in request(app, "/" + name).headers["cache-control"]
    for path in ("/manifest.webmanifest", "/bundle.aaaaaaaaaaaa.js"):
        assert request(app, path).headers["cache-control"] == "no-cache"
    assert (
        request(app, "/missing.js", headers={"Accept": "text/html"}).status_code == 404
    )
    assert request(app, "/missing").status_code == 404


def test_nuxt_chunk_without_inventory_is_served_and_revalidated(tmp_path, assets):
    (assets / "_nuxt").mkdir()
    name = "entry.BOzjEQym.css"
    raw = b".workstation { display: block; }"
    (assets / "_nuxt" / name).write_bytes(raw)
    assert not (assets / ".wavedl-assets.json").exists()

    response = request(make_app(tmp_path, assets), f"/_nuxt/{name}")

    assert response.status_code == 200
    assert response.content == raw
    assert response.headers["cache-control"] == "no-cache"


def test_stale_service_worker_is_replaced_with_unregister_script(tmp_path, assets):
    response = request(make_app(tmp_path, assets), "/sw.js")
    assert response.status_code == 200
    assert response.content != b"worker"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"].startswith("application/javascript")
    assert b"skipWaiting" in response.content
    assert b"clients.claim" in response.content
    assert b"registration.unregister" in response.content
    assert b'addEventListener("fetch"' not in response.content
    assert b"navigate(" not in response.content
    assert b"caches" not in response.content
    head = request(make_app(tmp_path, assets), "/sw.js", method="HEAD")
    assert head.status_code == 200 and not head.content
    assert head.headers["content-length"] == str(len(response.content))


@pytest.mark.parametrize("path", ["/%2e%2e/secret", "/bad%5cname", "/bad:name"])
def test_invalid_members_are_not_shell_navigation(tmp_path, assets, path):
    response = request(
        make_app(tmp_path, assets), path, headers={"Accept": "text/html"}
    )
    assert response.status_code == 404


def test_symlink_leaf_and_parent_fail_closed(tmp_path, assets):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.js").write_text("secret")
    try:
        (assets / "link.js").symlink_to(outside / "secret.js")
        (assets / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable")
    app = make_app(tmp_path, assets)
    for path in ("/link.js", "/linked/secret.js", "/linked/navigation"):
        assert request(app, path, headers={"Accept": "text/html"}).status_code == 404


def test_root_swap_to_symlink_fails_closed(tmp_path, assets):
    app = make_app(tmp_path, assets)
    moved = tmp_path / "moved"
    assets.rename(moved)
    try:
        assets.symlink_to(moved, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable")
    assert request(app).status_code == 404


def test_fallback_never_follows_replaced_index(tmp_path, assets):
    outside = tmp_path / "outside.html"
    outside.write_text("not the build")
    (assets / "index.html").unlink()
    try:
        (assets / "index.html").symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable")
    response = request(
        make_app(tmp_path, assets), "/runs/id", headers={"Accept": "text/html"}
    )
    assert response.status_code == 404


def test_assets_are_bounded_and_unsafe_shell_is_not_navigation_fallback(
    tmp_path, assets, monkeypatch
):
    monkeypatch.setattr("wavedl_daemon.http.static.MAX_ASSET_BYTES", 4)
    (assets / "large.js").write_bytes(b"more than four bytes")
    app = make_app(tmp_path, assets)
    assert request(app).status_code == 404
    assert request(app, "/sw.js").status_code == 200
    assert request(app, "/large.js").status_code == 404
    assert request(app, "/runs/id", headers={"Accept": "text/html"}).status_code == 404
