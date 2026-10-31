## Run the packaged browser workstation

The checkout already includes the generated browser assets. Install the daemon
from the repository root and start it; Node is not required to run this bundle:

```sh
python -m pip install -e .
wavedl serve --credential-file /absolute/path/to/daemon-token --browse-root datasets=/absolute/path/to/datasets
```

Use an owner-only token file for the recommended restart-stable master
credential; see the root [README](../../../README.md) for the creation command. The
ephemeral default is still supported, but a daemon restart requires a new
bootstrap link.

If you change browser source, regenerate and vendor its assets using the
development commands below before testing those changes through the daemon.

Open the single-use bootstrap URL printed by the daemon. The bootstrap code is
stripped from the URL synchronously before requests begin and is exchanged for
a session token; the master credential never enters the URL or browser storage.
Select data and a model, enter the absolute Python interpreter on the daemon
host, review the persisted plan, and approve it. The selected Python must
contain the WaveDL ML dependencies; daemon health alone does not certify that
environment. Keep the daemon bound to loopback; use SSH forwarding for remote
hosts, not unauthenticated LAN exposure.

Closing/detaching stops browser follow-up work without cancelling running jobs.
Use the explicit cancellation action for that. Browser downloads use your
browser's location/permission settings, not the daemon's run directory.

## Development and verification

Node 24 is used in CI (Node 22.19+ is supported). Install pnpm 12.11.2, pinned by
`packageManager` in `package.json`. Dependencies are locked in `pnpm-lock.yaml`;
`pnpm-workspace.yaml` preserves dependency overrides.

```sh
cd src/wavedl_daemon/web
pnpm install --frozen-lockfile
pnpm run dev
# Run Nuxt prepare/typecheck and generate sequentially, not concurrently.
pnpm run lint
pnpm run format:check
pnpm run typecheck
pnpm test
pnpm run generate
cd ../../../
python tools/vendor_web_assets.py
```
