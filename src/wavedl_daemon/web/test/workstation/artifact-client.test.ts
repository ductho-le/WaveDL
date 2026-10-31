import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it } from 'vitest';
import {
  ArtifactClient,
  BLOB_BUDGET,
  parseInventory,
  parseSelectionMetadata,
  pngPreviewBlob,
} from '../../app/lib/artifact-client';
import { ConnectionSession } from './session-fixture';

const run = '22222222-2222-4222-8222-222222222222';
const health = { daemon_id: 'daemon-one', build_id: '1.8.1', protocol_min: 1, protocol_max: 1 };

function json(body: FixtureResponse) {
  return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
}

async function setup(
  bytes = new TextEncoder().encode('published'),
  size = bytes.length,
  wrongHash = false,
) {
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');

  const calls: Array<{ path: string; init?: RequestInit }> = [];

  const inventory = {
    generations: [
      {
        generation: 'gen-000001',
        attempt: 1,
        step: 3,
        files: { 'weights.bin': { size, sha256: wrongHash ? '0'.repeat(64) : hash } },
      },
    ],
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
  it('preserves strict inventory keys and safe-integer bounds during decoding', () => {
    const generation = {
      generation: 'gen-000001',
      attempt: 1,
      step: 0,
      files: { 'weights.bin': { size: Number.MAX_SAFE_INTEGER, sha256: 'a'.repeat(64) } },
    };

    expect(parseInventory({ generations: [generation] }, run)[0]?.size).toBe(Number.MAX_SAFE_INTEGER);

    for (const value of [
      { generations: [generation], extra: true },
      { generations: [{ ...generation, extra: true }] },
      { generations: [{ ...generation, attempt: 0 }] },
      { generations: [{ ...generation, step: 0.5 }] },
      { generations: [generation, generation] },
      { generations: [{ ...generation, files: { 'weights.bin': { size: 1e100, sha256: 'a'.repeat(64) } } }] },
      { generations: [{ ...generation, files: { '../weights.bin': { size: 0, sha256: 'a'.repeat(64) } } }] },
    ]) {
      expect(() => parseInventory(value, run)).toThrow();
    }
  });

  it('keeps absent checkpoint coordinates omitted and rejects explicit nulls', () => {
    const metadata = parseSelectionMetadata({ schema_version: 1, kind: 'best' });

    expect(Object.keys(metadata)).toEqual(['schema_version', 'kind']);
    expect(Object.isFrozen(metadata)).toBe(true);
    expect(() => parseSelectionMetadata({ schema_version: 1, kind: 'best', source_seq: null })).toThrow();
    expect(() => parseSelectionMetadata({ schema_version: 1, kind: 'best', extra: 0 })).toThrow();
    expect(parseSelectionMetadata({ schema_version: 1, kind: 'cross_validation_fold' })).toEqual({ schema_version: 1, kind: 'cross_validation_fold' });
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
    const bytes = new Uint8Array([1, 2]);

    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('');

    let stream: ReadableStreamDefaultController<Uint8Array> | undefined;

    const session = new ConnectionSession({
      origin: 'http://localhost:8765',
      buildId: '1.8.1',
      fetcher: async (input) => {
        const path = new URL(String(input)).pathname;

        if (path === '/api/health') return json(health);

        if (path.endsWith('/artifacts'))
          return json({
            generations: [
              {
                generation: 'gen-000001',
                attempt: 1,
                step: 0,
                files: { 'weights.bin': { size: 2, sha256: hash } },
              },
            ],
          });

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
    const artifact = (await client.list(run))[0];

    if (!artifact) throw new Error('Missing artifact');
    const pending = client.verifiedBlob(artifact);
    const failure = expect(pending).rejects.toThrow();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(stream).toBeDefined();
    session.detach();
    await failure;
  });
  it('bounds inventory and rejects paths, ambiguous generation and unsafe numeric records', () => {
    for (const path of ['../weights', 'x%2Fy', 'x:y', 'x\\y', '/x', 'x\u0000y'])
      expect(() =>
        parseInventory(
          {
            generations: [
              {
                generation: 'gen-000001',
                attempt: 1,
                step: 0,
                files: { [path]: { size: 2, sha256: 'a'.repeat(64) } },
              },
            ],
          },
          run,
        ),
      ).toThrow();
    expect(() => parseInventory({ generations: [], unknown: true }, run)).toThrow();
    expect(() =>
      parseInventory(
        { generations: [{ generation: 'gen-000001', attempt: 0, step: 0, files: {} }] },
        run,
      ),
    ).toThrow();
  });
  it('only publishes exact size/hash bytes and keeps credentials out of URLs', async () => {
    const { client, calls } = await setup();
    const artifact = (await client.list(run))[0];

    if (!artifact) throw new Error('Missing artifact');
    expect(await (await client.verifiedBlob(artifact)).text()).toBe('published');
    const download = calls.find((call) => call.path.endsWith('weights.bin'));
    expect(download?.init?.redirect).toBe('error');
    expect(download?.init?.credentials).toBe('omit');
    expect(new Headers(download?.init?.headers).get('Authorization')).toBe(
      'Bearer session000000000000000000000000000000000001',
    );
    expect(calls.every((call) => !call.path.includes('secret-token'))).toBe(true);
  });
  it('rejects checksum mismatch and truncation without publishing', async () => {
    for (const kind of ['checksum', 'truncated']) {
      const { client } = await setup(
        new Uint8Array([1, 2]),
        kind === 'truncated' ? 3 : 2,
        kind === 'checksum',
      );

      const artifact = (await client.list(run))[0];

      if (!artifact) throw new Error('Missing artifact');
      await expect(client.verifiedBlob(artifact)).rejects.toThrow();
    }
  });
  it('refuses oversized allocation before issuing download and invalidates detached grants', async () => {
    const { client, calls, session } = await setup(new Uint8Array([1]), BLOB_BUDGET + 1);
    const artifact = (await client.list(run))[0];

    if (!artifact) throw new Error('Missing artifact');
    await expect(client.verifiedBlob(artifact)).rejects.toThrow(
      'This artifact exceeds the 64 MiB browser memory budget. Use the CLI to export it.',
    );
    expect(calls.some((call) => call.path.endsWith('weights.bin'))).toBe(false);
    session.detach();
    expect(() => client.assertSelection(artifact)).toThrow();
  });
  it('rejects forged and invalid CV metadata', () => {
    expect(() => parseSelectionMetadata({ schema_version: 1, kind: 'best', fold: 1 })).toThrow();
    expect(() =>
      parseSelectionMetadata({ schema_version: 1, kind: 'best', source_seq: true }),
    ).toThrow();
    expect(
      parseSelectionMetadata({ schema_version: 1, kind: 'cross_validation_fold', fold: 2 }),
    ).toEqual({ schema_version: 1, kind: 'cross_validation_fold', fold: 2 });
  });
});
