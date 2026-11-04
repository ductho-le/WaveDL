# Development

## Python

`uv sync` installs the repository into `.venv` in editable mode, so `.venv/bin/wavedl` runs the working tree. Re-run it after pulling if `wavedl` fails with `ModuleNotFoundError`: the launcher is generated from `[project.scripts]` and goes stale when an entry point moves.

## Web UI with hot reload

The daemon serves the built UI from `src/wavedl_daemon/web_assets`, so frontend edits are invisible to `wavedl serve` until rebuilt. For day-to-day UI work, run the daemon and the Nuxt dev server side by side:

```sh
# terminal 1: the daemon, on a fixed port
.venv/bin/wavedl serve --port 8420

# terminal 2: the UI, with hot reload
cd src/wavedl_daemon/web
pnpm dev
```

The daemon prints a one-time link such as `http://127.0.0.1:8420/?bootstrap=…`. Open it with the origin changed to the dev server, `http://localhost:3000/?bootstrap=…`, within 120 seconds. The link is single-use; restart the daemon for a new one, or turn on "Stay signed in" to survive dev-server reloads.

`pnpm dev` proxies `/api/*` to `http://127.0.0.1:8420` (see `devProxy` in `web/nuxt.config.ts`). Set `WAVEDL_DAEMON_URL` to point it elsewhere, for example `WAVEDL_DAEMON_URL=http://127.0.0.1:8421 pnpm dev`.

Things that will trip you up:

- The daemon rejects any request whose `Origin` is not its own address. The proxy rewrites `Origin` and `Host` to the daemon's so requests from the dev server pass. Do not "fix" a 403 Forbidden Origin by loosening the daemon's check; it is what stops other sites from driving the daemon through your browser.
- If the daemon is not running, API calls through the proxy hang rather than fail.
- After stopping the daemon, wait a few seconds before restarting it on the same port, or it exits with "Port … is in use".
- The dev server does not send the daemon's Content-Security-Policy. Check the built UI before shipping anything that loads scripts, styles, or fonts differently.

## Shipping UI changes

To make `wavedl serve` serve your changes on its own, rebuild and vendor the UI, then restart the daemon (it reads the asset manifest at startup):

```sh
cd src/wavedl_daemon/web && pnpm generate && cd -
.venv/bin/python tools/vendor_web_assets.py
```

This rewrites the committed files in `src/wavedl_daemon/web_assets` and leaves a backup in `src/wavedl_daemon/web/.vendor-backups/`. Commit the vendored assets only when you mean to ship that build.

## Checks

```sh
cd src/wavedl_daemon/web && pnpm typecheck
```

This must stay at zero errors; CI blocks on it.
