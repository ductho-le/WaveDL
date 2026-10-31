import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it, vi } from 'vitest';
import { ConnectionSession } from './session-fixture';
import { CatalogApi, relativePath } from '../../app/lib/catalog-api';
import { parseDatasetInspection } from '../../app/lib/dataset-inspection';
import { previewValuesSchema } from '../../shared/schemas/dataset-preview';

describe('dataset boundary parity', () => {
  const variable = { name: 'X', dtype: 'float32', shape: [2, 3], sample_count: 2, declared_bytes: 24 };

  const header = { schema_version: 1, format: 'npz', variables: [variable, { ...variable, name: 'Y', shape: [2], declared_bytes: 8 }] };

  it('keeps exact metadata bounds, selected extension, sample counts and unique variable names', () => {
    expect(parseDatasetInspection(header, { relativePath: 'train.npz' })).toMatchObject({ numSamples: 2, inputShape: [3], outputShape: [1] });

    for (const value of [
      { ...header, extra: true },
      { ...header, variables: [variable, { ...variable, name: 'Y', sample_count: null }] },
      { ...header, variables: [variable, variable] },
      { ...header, variables: [variable, { ...variable, name: 'Y', shape: [2.5] }] },
    ]) {
      expect(() => parseDatasetInspection(value, { relativePath: 'train.npz' })).toThrow();
    }

    expect(() => parseDatasetInspection(header, { relativePath: 'train.mat' })).toThrow();
  });

  it('retains finite nullable nested preview values without coercion', () => {
    expect(previewValuesSchema.parse([[null, -1e100], [0, 1e100]])).toEqual([[null, -1e100], [0, 1e100]]);

    for (const value of [[[true]], [['1']], [[Infinity]], [[NaN]], {}]) {
      expect(previewValuesSchema.safeParse(value).success).toBe(false);
    }
  });
});

const health = { daemon_id: 'daemon-one', build_id: '1.8.1', protocol_min: 1, protocol_max: 1 };

const catalogue = {
  source: 'bundled_metadata',
  runtime_verified: false,
  categories: { CNNs: { short_description: 'Convolution', models: ['cnn'] } },
  models: [
    {
      id: 'cnn',
      display_name: 'CNN',
      architecture_family: 'CNN',
      author_institution: 'Example',
      year_published: 1989,
      params_m: 1.6,
      size_mb: 6,
      short_description: 'Simple convolutional baseline',
      description: 'Description',
      best_for: 'Baseline',
      pros: ['Fast'],
      cons: ['Limited capacity'],
      compute_tier: 'Edge',
      tags: ['baseline'],
      supported_dims: [1, 2, 3],
      is_pretrained: false,
    },
  ],
};

describe('catalog record schema parity', () => {
  it.each([
    { roots: [{ id: 'data', extra: true }] },
    { roots: [{ id: 'data' }, { id: 'data' }] },
    { roots: [{ id: '' }] },
    { roots: [], extra: true },
  ])('rejects exact-key or duplicate root violations %j', async (body) => {
    const { api } = await setup(body);

    await expect(api.roots()).rejects.toMatchObject({ clientMessageKey: 'daemonResponseInvalid' });
  });

  it.each([
    { year_published: 1799 }, { year_published: 3001 }, { year_published: 2000.5 },
    { architecture_family: 'Other' }, { compute_tier: 'Huge' },
    { tags: ['same', 'same'] }, { tags: [' '] }, { tags: ['x\0y'] },
    { pros: Array.from({ length: 33 }, (_, index) => String(index)) },
    { supported_dims: [] }, { supported_dims: [1, 1] }, { supported_dims: [4] },
    { display_name: '' }, { params_m: -1 }, { is_pretrained: 1 }, { extra: true },
  ])('rejects invalid model fields %j', async (patch) => {
    const { api } = await setup({ ...catalogue, models: [{ ...catalogue.models[0], ...patch }] });

    await expect(api.models()).rejects.toMatchObject({ clientMessageKey: 'daemonResponseInvalid' });
  });

  it('accepts category additions, bound endpoints and empty lists while freezing mapped values', async () => {
    const { api } = await setup({ ...catalogue,
      categories: { '': { short_description: '', models: ['cnn'], additive: true } },
      models: [{ ...catalogue.models[0], year_published: 1800, params_m: 1_000_000, size_mb: 0, pros: [], cons: [], tags: [], supported_dims: [3] }],
    });
    const models = await api.models();

    expect(models[0]).toMatchObject({ yearPublished: 1800, paramsM: 1_000_000, supportedDims: [3], category: '' });
    expect(Object.isFrozen(models[0])).toBe(true);
    expect(Object.isFrozen(models[0]?.tags)).toBe(true);
  });

  it.each([
    { categories: { x: { short_description: '', models: ['cnn', 'cnn'] } } },
    { categories: { x: { short_description: '', models: [] } } },
    { categories: { x: { short_description: '', models: ['cnn'] }, y: { short_description: '', models: ['cnn'] } } },
    { models: [catalogue.models[0], catalogue.models[0]] },
  ])('retains cross-record membership and duplicate checks %j', async (patch) => {
    const { api } = await setup({ ...catalogue, ...patch });

    await expect(api.models()).rejects.toMatchObject({ clientMessageKey: 'daemonResponseInvalid' });
  });

  it('keeps relative-path UTF8 and component-count limits', () => {
    expect(relativePath('é'.repeat(2048))).toBe(true);
    expect(relativePath('é'.repeat(2049))).toBe(false);
    expect(relativePath(Array(128).fill('a').join('/'))).toBe(true);
    expect(relativePath(Array(129).fill('a').join('/'))).toBe(false);
    expect(relativePath('')).toBe(true);
  });
});

function json(body: FixtureResponse) {
  return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
}

async function setup(body: FixtureResponse, origin = 'http://localhost:8765') {
  let observedHealth = health;

  const fetcher = vi.fn<typeof fetch>(async (url) =>
    json(String(url).endsWith('/api/health') ? observedHealth : body),
  );

  const session = new ConnectionSession({ origin, buildId: '1.8.1', fetcher });
  expect(await session.connect('token')).toBe(true);

  return {
    session,
    api: new CatalogApi(session),
    fetcher,
    skew: () => {
      observedHealth = { ...health, daemon_id: 'different-daemon' };
    },
  };
}

describe('scoped daemon catalogues', () => {
  it('scopes explicit custom model declarations without inventing metadata certification', async () => {
    const { api, session, fetcher } = await setup({});
    const selected = await api.selectCustomModel('custom.Wave-Net_2');
    expect(selected).toEqual({
      model: 'custom.Wave-Net_2',
      daemonId: 'daemon-one',
      source: 'custom',
      runtimeVerified: false,
    });
    expect(Object.isFrozen(selected)).toBe(true);
    api.assertSelection(selected);
    expect(fetcher.mock.calls.every(([url]) => String(url).endsWith('/api/health'))).toBe(true);
    expect(() => api.assertSelection({ ...selected })).toThrow('Select again');
    const other = await setup({});
    expect(() => other.api.assertSelection(selected)).toThrow('Select again');
    await session.reconnect('token');
    expect(() => api.assertSelection(selected)).toThrow('Select again');
  });
  it.each([
    '',
    '../wave',
    '/tmp/model.py',
    'a/b',
    'a\\b',
    'wave()',
    'a;exec',
    ' a',
    'x\0y',
    'a'.repeat(129),
  ])('rejects invalid custom registry identifier %s before network work', async (id) => {
    const { api, fetcher } = await setup({});
    const calls = fetcher.mock.calls.length;
    await expect(api.selectCustomModel(id)).rejects.toMatchObject({ code: 'request' });
    expect(fetcher).toHaveBeenCalledTimes(calls);
  });
  it('verifies daemon identity before custom model authoring', async () => {
    const { api, session, skew } = await setup({});
    skew();
    await expect(api.selectCustomModel('wave')).rejects.toMatchObject({ code: 'identity' });
    expect(session.snapshot().status).toBe('skew');
  });
  it.each([
    { root: 'wrong', relative_path: 'a.npz', path: '/srv/a.npz', kind: 'file' },
    { root: 'data', relative_path: 'wrong.npz', path: '/srv/a.npz', kind: 'file' },
    { root: 'data', relative_path: 'a.npz', path: '/srv/../private/a.npz', kind: 'file' },
    { root: 'data', relative_path: 'a.npz', path: 'file:///srv/a.npz', kind: 'file' },
    { root: 'data', relative_path: 'a.npz', path: '/srv/a.npz', kind: 'symlink' },
  ])('rejects malformed selection resolution', async (body) => {
    const { api } = await setup(body);
    await expect(api.resolve('data', 'a.npz')).rejects.toMatchObject({ code: 'response' });
  });
  it('uses the current forwarded origin, with authenticated health checks', async () => {
    const { api, fetcher } = await setup({ roots: [{ id: 'datasets' }] }, 'http://127.0.0.1:19238');
    expect(await api.roots()).toEqual(['datasets']);
    expect(
      fetcher.mock.calls.every(([url]) => String(url).startsWith('http://127.0.0.1:19238/')),
    ).toBe(true);
    expect(fetcher.mock.calls.at(-1)?.[1]?.headers).toBeDefined();
  });
  it.each(['../x', '/tmp/a', 'a//b', 'a/./b', 'a\\b', 'C:/data', 'x\0y'])(
    'rejects noncanonical path %s without any request',
    async (path) => {
      const { api, fetcher } = await setup({});
      const calls = fetcher.mock.calls.length;
      expect(relativePath(path)).toBe(false);
      await expect(api.browse('data', path)).rejects.toMatchObject({ code: 'request' });
      expect(fetcher).toHaveBeenCalledTimes(calls);
    },
  );
  it('checks page identity and strict pagination', async () => {
    const body = {
      root: 'data',
      path: 'nested',
      entries: [{ name: 'wave.npz', kind: 'file' }],
      total: 1,
      offset: 0,
      limit: 100,
    };

    const { api } = await setup(body);
    expect((await api.browse('data', 'nested')).entries[0]?.name).toBe('wave.npz');
    const wrong = await setup({ ...body, path: '' });
    await expect(wrong.api.browse('data', 'nested')).rejects.toMatchObject({ code: 'response' });
    expect(wrong.session.api).toBeNull();
  });
  it.each([
    { roots: [{ id: '../escape' }] },
    { roots: [{ id: 'data' }, { id: 'data' }] },
    { roots: Array.from({ length: 33 }, (_, id) => ({ id: String(id) })) },
  ])('rejects malformed or oversized roots', async (body) => {
    const { api } = await setup(body);
    await expect(api.roots()).rejects.toMatchObject({ code: 'response' });
  });
  it.each([
    { entries: [{ name: '../escape', kind: 'file' }], total: 1 },
    { entries: [{ name: 'link', kind: 'symlink' }], total: 1 },
    { entries: [], total: 10001 },
    {
      entries: Array.from({ length: 101 }, (_, n) => ({ name: String(n), kind: 'file' })),
      total: 101,
    },
  ])('rejects malformed or oversized inventories', async (fields) => {
    const { api } = await setup({ root: 'data', path: '', limit: 100, offset: 0, ...fields });
    await expect(api.browse('data')).rejects.toMatchObject({ code: 'response' });
  });
  it.each([
    '/srv/datasets/original name.npz',
    'D:\\Research\\original name.npz',
    '\\\\server\\share\\original name.npz',
  ])('preserves daemon path %s without browser normalization', async (path) => {
    const { api, session } = await setup({
      root: 'data',
      relative_path: 'original name.npz',
      path,
      kind: 'file',
    });

    const selected = await api.resolve('data', 'original name.npz');
    expect(selected.path).toBe(path);
    expect(Object.isFrozen(selected)).toBe(true);
    new CatalogApi(session).assertSelection(selected);
    expect(() => api.assertSelection({ ...selected })).toThrow('Select again');
    session.detach();
    expect(() => api.assertSelection(selected)).toThrow('Select again');
  });
  it('rejects reused selections after same-daemon reconnection or in another session', async () => {
    const fixture = { root: 'data', relative_path: 'a.npz', path: '/srv/a.npz', kind: 'file' };
    const first = await setup(fixture);
    const second = await setup(fixture);
    const selected = await first.api.resolve('data', 'a.npz');
    expect(() => second.api.assertSelection(selected)).toThrow('Select again');
    await first.session.reconnect('token');
    expect(() => first.api.assertSelection(selected)).toThrow('Select again');
  });
  it('fails closed on forwarded-port daemon replacement before reading a catalogue', async () => {
    const { api, session, skew, fetcher } = await setup({ roots: [] });
    skew();
    await expect(api.roots()).rejects.toMatchObject({ code: 'identity' });
    expect(session.snapshot().status).toBe('skew');
    expect(fetcher.mock.calls.some(([url]) => String(url).endsWith('/api/browse'))).toBe(false);
  });
  it('brands model selections from current installed metadata, not supplied guesses', async () => {
    const { api, session } = await setup(catalogue);
    const models = await api.models();
    const model = models[0];

    if (!model) throw new Error('missing fixture');
    const selected = api.selectModel(model, models);
    expect(selected.model).toBe('cnn');
    api.assertSelection(selected);
    expect(() => api.selectModel({ ...model }, models)).toThrow('Reload');
    await session.reconnect('token');
    expect(() => api.selectModel(model, models)).toThrow('Reload');
  });
  it('retains the daemon structured fields on the frozen catalogue object', async () => {
    const { api } = await setup(catalogue);
    const models = await api.models();
    const model = models[0];
    expect(model).toMatchObject({
      architectureFamily: 'CNN',
      authorInstitution: 'Example',
      yearPublished: 1989,
      paramsM: 1.6,
      sizeMb: 6,
      shortDescription: 'Simple convolutional baseline',
      pros: ['Fast'],
      cons: ['Limited capacity'],
      computeTier: 'Edge',
      tags: ['baseline'],
    });
    expect(Object.isFrozen(models)).toBe(true);
    expect(Object.isFrozen(model?.pros)).toBe(true);
    expect(Object.isFrozen(model?.tags)).toBe(true);
  });
  it.each([
    { ...catalogue, runtime_verified: true },
    { ...catalogue, models: [{ ...catalogue.models[0], supported_dims: [1, 4] }] },
    { ...catalogue, models: Array.from({ length: 1001 }, () => catalogue.models[0]) },
  ])('rejects malformed model capability claims', async (body) => {
    const { api } = await setup(body);
    await expect(api.models()).rejects.toMatchObject({ code: 'response' });
  });
  it.each([
    { ...catalogue, models: [{ ...catalogue.models[0], display_name: 42 }] },
    { ...catalogue, models: [{ ...catalogue.models[0], description: '\0' }] },
    {
      ...catalogue,
      models: [{ ...catalogue.models[0], id: 'missing_category' }],
    },
    {
      ...catalogue,
      categories: {
        CNNs: { short_description: 'Convolution', models: ['cnn', 'cnn'] },
      },
    },
    {
      ...catalogue,
      models: [catalogue.models[0], { ...catalogue.models[0] }],
    },
  ])('rejects malformed model rows without exposing partial metadata', async (body) => {
    const { api } = await setup(body);
    await expect(api.models()).rejects.toMatchObject({ code: 'response' });
  });
  it('discards asynchronous work after scope changes even when transport ignores abort', async () => {
    let finish: ((response: Response) => void) | undefined;

    const session = new ConnectionSession({
      origin: 'http://localhost:8765',
      buildId: '1.8.1',
      fetcher: async (url) =>
        String(url).endsWith('/api/health')
          ? json(health)
          : new Promise((resolve) => {
              finish = resolve;
            }),
    });

    await session.connect('token');
    const work = new CatalogApi(session).roots();
    await vi.waitFor(() => expect(finish).toBeDefined());
    session.detach();
    finish?.(json({ roots: [{ id: 'old' }] }));
    await expect(work).rejects.toMatchObject({ code: 'stale' });
  });
});
