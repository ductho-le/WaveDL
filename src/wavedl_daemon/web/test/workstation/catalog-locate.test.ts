import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it, vi } from 'vitest';
import { parseTrainingConfig } from '../../app/lib/training-import';
import { ConnectionSession } from './session-fixture';
import { CatalogApi } from '../../app/lib/catalog-api';
import { locateRequestSchema } from '../../shared/schemas/workstation-controls';

const health = { daemon_id: 'daemon-one', build_id: '1.8.1', protocol_min: 1, protocol_max: 1 };

const bases = {
  data: '/srv/data',
  scratch: '/scratch/team/',
  windows: 'D:\\Research',
};

function json(body: FixtureResponse, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function resolveBody(init: RequestInit | undefined): { root: string; path: string } | null {
  const body = JSON.parse(String(init?.body ?? 'null'));

  return locateRequestSchema.safeParse(body).data ?? null;
}

async function setup(
  options: {
    roots?: string[];
    onResolve?: (root: string, path: string) => Response | Promise<Response> | null;
  } = {},
) {
  const fetcher = vi.fn<typeof fetch>(async (url, init) => {
    const { pathname } = new URL(String(url));

    if (pathname === '/api/health') return json(health);

    if (pathname === '/api/browse')
      return json({ roots: (options.roots ?? Object.keys(bases)).map((id) => ({ id })) });

    if (pathname === '/api/browse/resolve') {
      const body = resolveBody(init);

      if (!body) return json({ detail: 'bad request' }, 400);
      const custom = options.onResolve?.(body.root, body.path);

      if (custom) return custom;
      const base = Object.entries(bases).find(([root]) => root === body.root)?.[1] ?? '/missing';
      const separator = base.includes('\\') ? '\\' : '/';

      return json({
        root: body.root,
        relative_path: body.path,
        path: body.path
          ? `${base.replace(/[\\/]$/, '')}${separator}${body.path.replaceAll('/', separator)}`
          : base,
        kind: body.path ? 'file' : 'directory',
      });
    }

    return json({ detail: 'not found' }, 404);
  });

  const session = new ConnectionSession({
    origin: 'http://localhost:8765',
    buildId: '1.8.1',
    fetcher,
  });

  expect(await session.connect('token')).toBe(true);

  const resolved = () =>
    fetcher.mock.calls.flatMap(([url, init]) =>
      String(url).endsWith('/api/browse/resolve') ? [resolveBody(init)] : [],
    );

  return { api: new CatalogApi(session), fetcher, resolved };
}

describe('locating imported dataset paths', () => {
  it('maps a path under a configured root onto a scoped selection', async () => {
    const { api, resolved } = await setup();
    const selected = await api.locate('/scratch/team/runs/2026/train.npz');
    expect(selected).toMatchObject({
      root: 'scratch',
      relativePath: 'runs/2026/train.npz',
      path: '/scratch/team/runs/2026/train.npz',
      kind: 'file',
      daemonId: 'daemon-one',
    });

    if (!selected) throw new Error('missing selection');
    api.assertSelection(selected);
    expect(resolved()).toEqual([
      { root: 'data', path: '' },
      { root: 'scratch', path: '' },
      { root: 'scratch', path: 'runs/2026/train.npz' },
    ]);
  });

  it('maps Windows host paths with backslash separators', async () => {
    const { api } = await setup();
    expect(await api.locate('D:\\Research\\waves\\train.npz')).toMatchObject({
      root: 'windows',
      relativePath: 'waves/train.npz',
    });
  });

  it.each([
    '/home/me/train.npz',
    '/srv/database/train.npz',
    '/srv/data',
    '/srv/data/',
    'E:\\Research\\waves\\train.npz',
  ])('returns null for %s, which no root contains', async (path) => {
    const { api } = await setup();
    expect(await api.locate(path)).toBeNull();
  });

  it.each([
    'train.npz',
    '',
    './train.npz',
    '/srv/data/../private/key.npz',
    'file:///srv/data/a.npz',
  ])('returns null for non-absolute or non-canonical %s without any request', async (path) => {
    const { api, fetcher } = await setup();
    const calls = fetcher.mock.calls.length;
    expect(await api.locate(path)).toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(calls);
  });

  it('returns null when there are no roots', async () => {
    const { api, resolved } = await setup({ roots: [] });
    expect(await api.locate('/srv/data/train.npz')).toBeNull();
    expect(resolved()).toEqual([]);
  });

  it('returns null when the daemon cannot resolve the mapped path', async () => {
    const { api, resolved } = await setup({
      onResolve: (_root, path) => (path ? json({ detail: 'missing' }, 404) : null),
    });

    expect(await api.locate('/srv/data/missing.npz')).toBeNull();
    expect(resolved().at(-1)).toEqual({ root: 'data', path: 'missing.npz' });
  });

  it('rethrows when the caller aborts during the final resolution', async () => {
    const controller = new AbortController();

    const { api } = await setup({
      onResolve: (_root, path) => {
        if (!path) return null;
        controller.abort();

        return Promise.reject(new DOMException('The operation was aborted.', 'AbortError'));
      },
    });

    await expect(api.locate('/srv/data/train.npz', controller.signal)).rejects.toThrow(
      'discard this result',
    );
  });
});

describe('imported draft boundary parity', () => {
  it('normalizes CLI lists and preserves individually valid fields beside invalid YAML numbers', () => {
    const imported = parseTrainingConfig(
      'epochs: 4\nlr: .inf\nloss_weights: [1, 2]\nbetas: [0.9, 0.99]\nmodel: "custom.Model"\ndata_path: " /data/train.npz "\nunknown_option: .nan\n',
    );

    expect(imported.config.epochs).toBe(4);
    expect(imported.config.loss_weights).toBe('1,2');
    expect(imported.config.betas).toBe('0.9,0.99');
    expect(imported.model).toBe('custom.Model');
    expect(imported.dataPath).toBe(' /data/train.npz ');
    expect(imported.skipped).toEqual(['lr', 'unknown_option']);
  });

  it('keeps the precision alias precedence and rejects non-object documents and duplicate keys', () => {
    expect(parseTrainingConfig('mixed_precision: bf16\n').config.precision).toBe('bf16');
    expect(parseTrainingConfig('mixed_precision: bf16\nprecision: no\n').config.precision).toBe(
      'no',
    );

    for (const source of ['null', '[]', 'epochs: 1\nepochs: 2\n']) {
      expect(() => parseTrainingConfig(source)).toThrow();
    }
  });
});
