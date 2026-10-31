import { describe, expect, it } from 'vitest';
import type { Artifact } from '../../app/lib/artifact-client';
import { artifactDownloadName } from '../../app/lib/artifact-download';

const artifact: Artifact = {
  runId: '22222222-2222-4222-8222-222222222222',
  generation: 'gen-000001',
  attempt: 1,
  step: 0,
  path: 'results/metrics.csv',
  size: 5,
  sha256: 'a'.repeat(64),
};

describe('artifact download provenance', () => {
  it.each(['../unsafe', '', 'a'.repeat(63), 'A'.repeat(64)])(
    'rejects unverified digest data in a filename label: %s',
    async (sha256) => {
      await expect(artifactDownloadName({ ...artifact, sha256 })).rejects.toThrow('provenance');
    },
  );
  it.each(['metrics.csv', 'model.onnx'])(
    'keeps full run and generation identity for %s',
    async (path) => {
      const names = await Promise.all([
        artifactDownloadName({ ...artifact, path }),
        artifactDownloadName({ ...artifact, path, generation: 'gen-000002' }),
        artifactDownloadName({ ...artifact, path, runId: '33333333-3333-4333-8333-333333333333' }),
      ]);

      expect(new Set(names).size).toBe(3);
      expect(names[0]).toBe(`${artifact.runId}_${artifact.generation}_${path}`);
      expect(names.every((name) => name.endsWith(path))).toBe(true);
      expect(await artifactDownloadName({ ...artifact, path })).toBe(names[0]);
    },
  );
  it('sanitizes native separators, controls, bidi markers, Windows characters and reserved names', async () => {
    for (const path of [
      'dir/CON.csv',
      'dir/NUL',
      'dir/a\\b<>:"|?*\u0000\u007f\u202e.csv',
      'dir/.. ',
      'dir/ＦＵＬＬ.csv',
    ]) {
      const name = await artifactDownloadName({ ...artifact, path });
      expect(name).not.toMatch(/[\p{Cc}\p{Cf}<>:"/\\|?*]/u);
      expect(name).not.toMatch(/[. ]$/u);
      expect(name.startsWith(`${artifact.runId}_${artifact.generation}_`)).toBe(true);
    }

    expect(
      (await artifactDownloadName({ ...artifact, path: 'dir/CON.csv' })).endsWith('_CON.csv'),
    ).toBe(true);
    expect(
      (await artifactDownloadName({ ...artifact, path: 'dir/.. ' })).endsWith('_artifact'),
    ).toBe(true);
  });
  it('bounds multibyte and API-sized names without splitting Unicode or losing ordinary extensions', async () => {
    for (const path of [`${'測🌊'.repeat(1000)}.onnx`, `${'a'.repeat(4096)}.csv`]) {
      const name = await artifactDownloadName({
        ...artifact,
        path,
        generation: `gen-${'9'.repeat(60)}`,
      });

      expect(new TextEncoder().encode(name).byteLength).toBeLessThanOrEqual(255);
      expect(name).toContain('.');
      expect(name.endsWith(path.endsWith('.onnx') ? '.onnx' : '.csv')).toBe(true);
      expect(name).not.toContain('\ufffd');
    }
  });
  it('fails closed rather than silently truncating unrepresentable generation provenance', async () => {
    await expect(
      artifactDownloadName({ ...artifact, generation: `gen-${'9'.repeat(61)}` }),
    ).rejects.toThrow('provenance');
  });
  it('disambiguates sanitized aliases within one publication', async () => {
    const aliases = [
      { ...artifact, path: 'left/a?.csv', sha256: 'a'.repeat(64) },
      { ...artifact, path: 'right/a*.csv', sha256: 'b'.repeat(64) },
    ];

    const names = await Promise.all(aliases.map((item) => artifactDownloadName(item, aliases)));
    expect(new Set(names).size).toBe(2);
    expect(names.every((name) => name.endsWith('.csv'))).toBe(true);
  });
  it.each([
    ['left/metrics.csv', 'right/metrics.csv'],
    ['metrics.csv', 'METRICS.csv'],
    ['ＦＵＬＬ.csv', 'FULL.csv'],
    ['CON.csv', '_CON.csv'],
  ])('separates %s from %s even when their verified bytes are identical', async (left, right) => {
    const aliases = [left, right].map((path) => ({ ...artifact, path }));
    const names = await Promise.all(aliases.map((item) => artifactDownloadName(item, aliases)));
    expect(new Set(names.map((name) => name.toLowerCase())).size).toBe(2);

    for (const [index, item] of aliases.entries())
      expect(await artifactDownloadName(item, [...aliases].reverse())).toBe(names[index]);
  });
  it('retains unique nested labels and ignores aliases owned by another publication', async () => {
    const other = { ...artifact, path: 'elsewhere/metrics.csv', generation: 'gen-000002' };
    expect(await artifactDownloadName(artifact, [artifact, other])).toBe(
      `${artifact.runId}_${artifact.generation}_metrics.csv`,
    );
  });
  it('disambiguates long paths with a shared truncated prefix and identical bytes', async () => {
    const paths = [`${'a'.repeat(4090)}1.csv`, `${'a'.repeat(4090)}2.csv`];

    const names = await Promise.all(
      paths.map((path) => artifactDownloadName({ ...artifact, path })),
    );

    expect(new Set(names).size).toBe(2);
    expect(names.every((name) => new TextEncoder().encode(name).byteLength <= 255)).toBe(true);
  });
  it('disambiguates a short sibling deliberately matching a truncated final label', async () => {
    const long = { ...artifact, path: `${'a'.repeat(300)}.csv` };
    const prefix = `${artifact.runId}_${artifact.generation}_`;
    const short = { ...artifact, path: (await artifactDownloadName(long)).slice(prefix.length) };
    const inventory = [long, short];
    const names = await Promise.all(inventory.map((item) => artifactDownloadName(item, inventory)));
    expect(new Set(names).size).toBe(2);
    expect(names.every((name) => new TextEncoder().encode(name).byteLength <= 255)).toBe(true);
  });
  it('refuses deliberately chained generated aliases instead of publishing an ambiguous label', async () => {
    const prefix = `${artifact.runId}_${artifact.generation}_`;
    const long = { ...artifact, path: `${'a'.repeat(300)}.csv` };
    const first = { ...artifact, path: (await artifactDownloadName(long)).slice(prefix.length) };

    const second = {
      ...artifact,
      path: (await artifactDownloadName(first, [long, first])).slice(prefix.length),
    };

    await expect(artifactDownloadName(first, [long, first, second])).rejects.toThrow('distinct');
    await expect(artifactDownloadName(second, [long, first, second])).rejects.toThrow('distinct');
  });
});
