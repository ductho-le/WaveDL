import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it } from 'vitest';
import {
  ArtifactClient,
  BLOB_BUDGET,
  parseInventory,
  pngPreviewBlob,
} from '../../app/lib/artifact-client';
import { ConnectionSession } from './session-fixture';
import type { Json } from '../../app/lib/daemon-api';

const run = '22222222-2222-4222-8222-222222222222';
const health = { daemon_id: 'daemon-one', build_id: '1.8.1', protocol_min: 1, protocol_max: 1 };
const modified = '2026-11-03T10:00:00.123456+00:00';

function json(body: FixtureResponse) {
  return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
}

async function setup(
  bytes = new TextEncoder().encode('published'),
  size = bytes.length,
  path = 'best_checkpoint/model weights.bin',
  listed = size,
) {
  const calls: Array<{ path: string; init?: RequestInit }> = [];

  const inventory = {
    files: [{ path, size: listed, modified }],
    checkpoints: ['best_checkpoint'],
  };

  const session = new ConnectionSession({
    origin: 'http://localhost:8765',
    buildId: '1.8.1',
    fetcher: async (url, init) => {
      const path = new URL(String(url)).pathname;
      calls.push({ path, init });

      if (path === '/api/health') return json(health);

      if (path.endsWith('/artifacts')) return json(inventory);

      return new Response(bytes, {
        headers: { 'content-type': 'application/octet-stream', 'content-length': String(size) },
      });
    },
  });

  await session.connect('secret-token');

  return { session, calls, client: new ArtifactClient(session) };
}

describe('published artifact authority', () => {
  it('decodes nested output files and checkpoint names bound to the run', () => {
    const inventory = parseInventory(
      {
        files: [
          { path: 'best_checkpoint/model.safetensors', size: Number.MAX_SAFE_INTEGER, modified },
          { path: 'fold_1/model.pth', size: 0, modified: '2026-11-03T10:00:00Z' },
        ],
        checkpoints: ['best_checkpoint', 'epoch_12_checkpoint', 'interrupted_checkpoint'],
      },
      run,
    );

    expect(inventory.files).toEqual([
      { runId: run, path: 'best_checkpoint/model.safetensors', size: Number.MAX_SAFE_INTEGER, modified },
      { runId: run, path: 'fold_1/model.pth', size: 0, modified: '2026-11-03T10:00:00Z' },
    ]);
    expect(inventory.checkpoints).toEqual([
      { runId: run, name: 'best_checkpoint' },
      { runId: run, name: 'epoch_12_checkpoint' },
      { runId: run, name: 'interrupted_checkpoint' },
    ]);
    expect(() => parseInventory({ files: [], checkpoints: [] }, 'not-a-run')).toThrow();
  });

  it('rejects key drift, unsafe sizes, unzoned times, duplicates and unknown checkpoint names', () => {
    const file = { path: 'training_history.csv', size: 1, modified };

    const malformed: Json[] = [
      { files: [file], checkpoints: [], extra: true },
      { files: [file] },
      { files: [{ ...file, extra: true }], checkpoints: [] },
      { files: [{ ...file, size: 1e100 }], checkpoints: [] },
      { files: [{ ...file, size: -1 }], checkpoints: [] },
      { files: [{ ...file, size: 0.5 }], checkpoints: [] },
      { files: [{ ...file, modified: '2026-11-03T10:00:00' }], checkpoints: [] },
      { files: [{ ...file, modified: '2026-11-03' }], checkpoints: [] },
      { files: [file, file], checkpoints: [] },
      { files: [], checkpoints: ['best_checkpoint', 'best_checkpoint'] },
      { files: [], checkpoints: ['gen-000001'] },
      { files: [], checkpoints: ['epoch_x_checkpoint'] },
      { files: [], checkpoints: ['fold_1/best_checkpoint'] },
    ];

    for (const value of malformed) expect(() => parseInventory(value, run)).toThrow();
  });

  it('bounds inventory and rejects unsafe paths', () => {
    for (const path of ['../weights', 'x%2Fy', 'x:y', 'x\\y', '/x', 'x//y', 'x/./y', 'x\u0000y'])
      expect(() =>
        parseInventory({ files: [{ path, size: 2, modified }], checkpoints: [] }, run),
      ).toThrow();

    const files = Array.from({ length: 16385 }, (_, index) => ({
      path: `file-${index}.bin`,
      size: 0,
      modified,
    }));

    expect(() => parseInventory({ files, checkpoints: [] }, run)).toThrow();
    expect(parseInventory({ files: files.slice(1), checkpoints: [] }, run).files).toHaveLength(16384);
  });

  it('limits raster preview dimensions before handing bytes to the decoder', async () => {
    const bytes = new Uint8Array(33);
    bytes.set([137, 80, 78, 71, 13, 10, 26, 10]);
    const view = new DataView(bytes.buffer);
    view.setUint32(8, 13);
    bytes.set([73, 72, 68, 82], 12);
    view.setUint32(16, 10);
    view.setUint32(20, 10);
    expect((await pngPreviewBlob(new Blob([bytes]))).type).toBe('image/png');
    view.setUint32(16, 9000);
    await expect(pngPreviewBlob(new Blob([bytes]))).rejects.toThrow('pixel');
    await expect(pngPreviewBlob(new Blob(['<svg/>']))).rejects.toThrow('PNG');
  });
  it('discards a late streamed response when the owning session detaches', async () => {
    let stream: ReadableStreamDefaultController<Uint8Array> | undefined;

    const session = new ConnectionSession({
      origin: 'http://localhost:8765',
      buildId: '1.8.1',
      fetcher: async (input) => {
        const path = new URL(String(input)).pathname;

        if (path === '/api/health') return json(health);

        if (path.endsWith('/artifacts'))
          return json({ files: [{ path: 'weights.bin', size: 2, modified }], checkpoints: [] });

        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              stream = controller;
            },
          }),
          { headers: { 'content-type': 'application/octet-stream', 'content-length': '2' } },
        );
      },
    });

    await session.connect('token');
    const client = new ArtifactClient(session);
    const artifact = (await client.list(run)).files[0];

    if (!artifact) throw new Error('Missing artifact');
    const pending = client.verifiedBlob(artifact);
    const failure = expect(pending).rejects.toThrow();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(stream).toBeDefined();
    session.detach();
    await failure;
  });
  it('downloads exact-size bytes by encoded output path and keeps credentials out of URLs', async () => {
    const { client, calls } = await setup();
    const artifact = (await client.list(run)).files[0];

    if (!artifact) throw new Error('Missing artifact');
    expect(await (await client.verifiedBlob(artifact)).text()).toBe('published');
    const download = calls.at(-1);
    expect(download?.path).toBe(`/api/runs/${run}/artifacts/best_checkpoint/model%20weights.bin`);
    expect(download?.init?.redirect).toBe('error');
    expect(download?.init?.credentials).toBe('omit');
    expect(new Headers(download?.init?.headers).get('Authorization')).toBe(
      'Bearer session000000000000000000000000000000000001',
    );
    expect(calls.every((call) => !call.path.includes('secret-token'))).toBe(true);
  });
  it('rejects truncated and oversized bodies without publishing', async () => {
    for (const { bytes, size } of [
      { bytes: new Uint8Array([1, 2]), size: 3 },
      { bytes: new Uint8Array([1, 2]), size: 1 },
    ]) {
      const { client } = await setup(bytes, size);
      const artifact = (await client.list(run)).files[0];

      if (!artifact) throw new Error('Missing artifact');
      await expect(client.verifiedBlob(artifact)).rejects.toThrow();
    }
  });
  it('downloads a file a live run rewrote since the listing, bounded by its declared length', async () => {
    const grown = await setup(new TextEncoder().encode('published'), 9, 'training_history.csv', 4);
    const artifact = (await grown.client.list(run)).files[0];

    if (!artifact) throw new Error('Missing artifact');
    expect(await (await grown.client.verifiedBlob(artifact)).text()).toBe('published');

    const oversized = await setup(new Uint8Array([1]), BLOB_BUDGET + 1, 'training_history.csv', 1);
    const listed = (await oversized.client.list(run)).files[0];

    if (!listed) throw new Error('Missing artifact');
    await expect(oversized.client.verifiedBlob(listed)).rejects.toThrow(
      'This artifact exceeds the 64 MiB browser memory budget. Use the CLI to export it.',
    );
    expect(oversized.session.api).not.toBeNull();
  });
  it('refuses oversized allocation before issuing download and invalidates detached grants', async () => {
    const { client, calls, session } = await setup(new Uint8Array([1]), BLOB_BUDGET + 1);
    const inventory = await client.list(run);
    const artifact = inventory.files[0];
    const checkpoint = inventory.checkpoints[0];

    if (!artifact || !checkpoint) throw new Error('Missing inventory');
    await expect(client.verifiedBlob(artifact)).rejects.toThrow(
      'This artifact exceeds the 64 MiB browser memory budget. Use the CLI to export it.',
    );
    expect(calls.some((call) => call.path.endsWith('weights.bin'))).toBe(false);
    expect(() => client.assertSelection(checkpoint)).not.toThrow();
    expect(() => client.assertSelection({ ...checkpoint })).toThrow();
    session.detach();
    expect(() => client.assertSelection(artifact)).toThrow();
    expect(() => client.assertSelection(checkpoint)).toThrow();
  });
});
