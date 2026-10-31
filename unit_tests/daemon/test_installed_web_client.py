"""Real vendored Nuxt assets must survive a wheel install outside the checkout."""

import hashlib
import json
from pathlib import Path

import pytest

from .test_installed_cli import checked, installed


@pytest.fixture(scope="module")
def installed_web(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    assets = root / "src/wavedl_daemon/web_assets"
    if not (assets / "index.html").is_file():
        pytest.skip(
            "Generate and vendor the real web build before installed acceptance"
        )
    inventory = {
        file.relative_to(assets).as_posix(): {
            "size": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        for file in assets.rglob("*")
        if file.is_file()
    }
    assert any(name.startswith("_nuxt/") for name in inventory)
    instance = installed.__wrapped__(tmp_path_factory)
    return instance, inventory


_EXERCISE = r"""
import asyncio, base64, hashlib, json, re, sys
from pathlib import Path
import httpx
import wavedl_daemon
from wavedl_daemon.http.app import create_app

package = Path(wavedl_daemon.__file__).parent
assert package.is_relative_to(Path(sys.prefix)), package
assets = package / 'web_assets'
inventory = json.loads(Path(sys.argv[1]).read_text())
packaged = {file.relative_to(assets).as_posix() for file in assets.rglob('*') if file.is_file()}
assert packaged == set(inventory), ('Wheel inventory mismatch', packaged ^ set(inventory))
for name, expected in inventory.items():
    raw = (assets / name).read_bytes()
    assert len(raw) == expected['size'], name
    assert hashlib.sha256(raw).hexdigest() == expected['sha256'], name

async def exercise():
    app = create_app(None, token='installed-static', daemon_id='installed-daemon',
                     build_id='installed-build')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url='http://localhost') as client:
        root = await client.get('/')
        assert root.status_code == 200, root.text
        assert root.content == (assets / 'index.html').read_bytes()
        assert root.headers['cache-control'] == 'no-store'
        assert root.headers['content-type'].startswith('text/html')
        assert root.headers['referrer-policy'] == 'no-referrer'
        policy = root.headers['content-security-policy']
        assert "script-src 'self'" in policy
        assert "style-src 'self' 'unsafe-inline'" in policy
        assert 'unsafe-eval' not in policy
        for attributes, body in re.findall(rb'<script\b([^>]*)>(.*?)</script\s*>',
                                           root.content, re.I | re.S):
            if not re.search(rb'\bsrc\s*=', attributes, re.I):
                digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
                assert "'sha256-" + digest + "'" in policy
        deep = await client.get('/runs/installed-run', headers={'Accept':'text/html'})
        assert deep.status_code == 200 and deep.content == root.content
        assert deep.headers['cache-control'] == 'no-store'
        head = await client.head('/runs/installed-run', headers={'Accept':'text/html'})
        assert head.status_code == 200 and not head.content
        assert head.headers['content-length'] == str(len(root.content))
        assert (await client.get('/api/unknown')).status_code == 401
        missing = await client.get('/api/unknown', headers={
            'Authorization':'Bearer installed-static', 'Accept':'text/html'})
        assert missing.status_code == 404
        assert missing.headers['content-type'] == 'application/json'
        assert (await client.post('/runs/installed-run')).status_code == 405
        assert (await client.get('/_nuxt/missing.js', headers={'Accept':'text/html'})).status_code == 404
        worker = await client.get('/sw.js')
        assert worker.status_code == 200
        assert worker.headers['cache-control'] == 'no-store'
        assert b'registration.unregister' in worker.content
        assert b'addEventListener("fetch"' not in worker.content
        assert b'navigate(' not in worker.content
        assert b'caches' not in worker.content
        for name in inventory:
            if name == 'sw.js':
                continue
            response = await client.get('/' + name)
            assert response.status_code == 200, (name, response.status_code)
            assert hashlib.sha256(response.content).hexdigest() == inventory[name]['sha256'], name
            assert response.headers['x-content-type-options'] == 'nosniff'
            if name.endswith('.js'):
                assert 'javascript' in response.headers['content-type'], name
            elif name.endswith('.css'):
                assert response.headers['content-type'].startswith('text/css'), name
    print(json.dumps({'packaged_files':len(inventory),
                      'service_worker_retirement':'verified',
                      'static_and_api':'verified'}))
asyncio.run(exercise())
"""


def test_installed_real_web_inventory_shell_csp_and_api_separation(installed_web):
    (outside, _venv, _binary, python, environment), inventory = installed_web
    expected = outside / "web-inventory.json"
    expected.write_text(json.dumps(inventory))
    output = checked(
        [str(python), "-c", _EXERCISE, str(expected)],
        cwd=outside,
        environment=environment,
    )
    evidence = json.loads(output)
    assert evidence["packaged_files"] == len(inventory)
    assert evidence["static_and_api"] == "verified"
    assert evidence["service_worker_retirement"] == "verified"
