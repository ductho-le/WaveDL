import { jsonObjectSchema } from '../../shared/schemas/json';
import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it } from 'vitest';
import { inferenceDefaults, resultRequestSchema } from '../../shared/schemas/result-request';
import { ResultOperations, checkpointOptions } from '../../app/lib/result-operations';
import { ArtifactClient } from '../../app/lib/artifact-client';
import { CatalogApi } from '../../app/lib/catalog-api';
import { ConnectionSession } from './session-fixture';
import { createResultOperationDefaults } from '../../app/lib/result-operation-defaults';
import { parseCvSummary } from '../../app/lib/cv-summary';

describe('CV summary decoding parity', () => {
  const config = {
    model: '', folds: 2, stratify: false, stratify_bins: 1, batch_size: 1,
    lr: 1e100, epochs: 1, patience: 0, loss: '', optimizer: '', scheduler: '',
  };

  const fold = {
    fold: 1, best_val_loss: null, best_epoch: 0, r2: -1e100,
    mae_normalized: 0, mae_physical: null, epochs_trained: 1,
    r2_target_0: null, mae_target_0: 1e100,
  };

  const summary = {
    config, timestamp: '2026-10-31', folds: 2, r2_mean: -1e100, r2_std: null,
    mae_mean: 0, mae_std: 0, val_loss_mean: null, val_loss_std: 0,
    fold_results: [fold, { ...fold, fold: 2 }],
  };

  it('accepts empty producer labels, finite metrics beyond safe integers and nullable metrics', () => {
    const parsed = parseCvSummary(summary);

    expect(parsed.r2_mean).toBe(-1e100);
    expect(parsed.fold_results[0]?.targets).toEqual([{ index: 0, r2: null, mae: 1e100 }]);
    expect(Object.hasOwn(parsed, 'config')).toBe(false);
  });

  it('rejects key drift, duplicate folds, missing target pairs and mismatched target counts', () => {
    for (const value of [
      { ...summary, extra: true },
      { ...summary, config: { ...config, extra: true } },
      { ...summary, config: { ...config, folds: 3 } },
      { ...summary, fold_results: [fold, fold] },
      { ...summary, fold_results: [fold, { ...fold, fold: 2, best_epoch: 2 }] },
      { ...summary, fold_results: [fold, { ...fold, fold: 2, mae_target_0: -1 }] },
      { ...summary, fold_results: [fold, { ...fold, fold: 2, r2_target_1: 0, mae_target_1: 0 }] },
      { ...summary, fold_results: [fold, { ...fold, fold: 2, r2_target_01: 0, mae_target_01: 0 }] },
    ]) {
      expect(() => parseCvSummary(value)).toThrow();
    }

    const { mae_target_0: omitted, ...unpaired } = fold;

    expect(omitted).toBe(1e100);
    expect(() => parseCvSummary({ ...summary, fold_results: [fold, { ...unpaired, fold: 2 }] })).toThrow();
  });
});

const sourceRun = '11111111-1111-4111-8111-111111111111';

const ids = {
  run_id: '22222222-2222-4222-8222-222222222222',
  plan_id: '33333333-3333-4333-8333-333333333333',
  request_id: '44444444-4444-4444-8444-444444444444',
};

function json(body: FixtureResponse) {
  return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
}

async function setup(kind = 'best', fold?: number) {
  const classification = { schema_version: 1, kind };

  if (fold !== undefined) Object.assign(classification, { fold });

  const bytes = new TextEncoder().encode(
    JSON.stringify(classification),
  );

  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');

  const keys: Array<string | null> = [];
  const bodies: string[] = [];

  const session = new ConnectionSession({
    origin: 'http://localhost:8765',
    buildId: '1.8.1',
    fetcher: async (input, init) => {
      const path = new URL(String(input)).pathname;

      if (path === '/api/health')
        return json({ daemon_id: 'd', build_id: '1.8.1', protocol_min: 1, protocol_max: 1 });

      if (path === '/api/browse/resolve')
        return json({
          root: 'data',
          relative_path: 'test.npz',
          kind: 'file',
          path: '/data/test.npz',
        });

      if (path.endsWith('/artifacts'))
        return json({
          generations: [
            {
              generation: 'gen-000001',
              attempt: 1,
              step: 5,
              files: { 'checkpoint_selection.json': { size: bytes.length, sha256: hash } },
            },
          ],
        });

      if (path.endsWith('checkpoint_selection.json'))
        return new Response(bytes, {
          headers: {
            'content-type': 'application/octet-stream',
            'content-length': String(bytes.length),
          },
        });

      if (path === '/api/runs/plan') {
        keys.push(new Headers(init?.headers).get('Idempotency-Key'));
        bodies.push(String(init?.body));

        return json({ ...ids, preview: {} });
      }

      return json({
        ...ids,
        backend: 'local',
        operation: 'test',
        attempt: null,
        state: 'submitted',
        uncertainty: false,
        submission_requested: true,
        cancellation_requested: false,
        evidence: null,
      });
    },
  });

  await session.connect('token');
  const data = await new CatalogApi(session).resolve('data', 'test.npz');
  const metadata = (await new ArtifactClient(session).list(sourceRun))[0];

  if (!metadata) throw new Error('Missing metadata');

  const draft = resultRequestSchema.parse({
    backend: 'local',
    operation: 'test',
    config: { ...inferenceDefaults, data_path: data.path },
    checkpoint_source: { run_id: sourceRun, generation: metadata.generation },
    runtime: { python: '/usr/bin/python3', mode: 'existing', device: 'cpu', environment: {} },
    topology: { num_machines: 1, processes_per_machine: 1, machine_rank: 0 },
  });

  return { session, data, metadata, draft, keys, bodies, workflow: new ResultOperations(session) };
}

describe('explicit published inference workflow', () => {
  it('keeps nested draft defaults isolated across instances', () => {
    const first = createResultOperationDefaults();
    const second = createResultOperationDefaults();
    first.config.import_modules.push('first-registration');
    first.runtime.environment.FIRST = 'private';
    first.topology.machine_rank = 1;
    expect(second.config.import_modules).toEqual([]);
    expect(second.runtime.environment).toEqual({});
    expect(second.topology.machine_rank).toBe(0);
  });
  it('mount defaults carry a valid local CPU backend for planning', () => {
    const resultOperationDefaults = createResultOperationDefaults();

    const draft = resultRequestSchema.safeParse({
      ...resultOperationDefaults,
      config: { ...resultOperationDefaults.config, data_path: '/data/test.npz' },
      checkpoint_source: { run_id: sourceRun, generation: 'gen-000001' },
      runtime: { ...resultOperationDefaults.runtime, python: '/usr/bin/python3' },
    });

    expect(draft.success).toBe(true);

    if (draft.success) expect(draft.data.backend).toBe('local');
  });
  it('plans exact source and freezes approval, reusing unchanged retry key', async () => {
    const { workflow, draft, data, metadata, keys, bodies } = await setup();
    await workflow.prepare(draft, data, metadata);
    await workflow.preview();
    await workflow.prepare(draft, data, metadata);
    await workflow.preview();
    expect(keys[0]).toBe(keys[1]);
    expect(resultRequestSchema.parse(JSON.parse(bodies[0] ?? '{}')).checkpoint_source).toEqual({
      run_id: sourceRun,
      generation: 'gen-000001',
    });
    expect(jsonObjectSchema.parse(resultRequestSchema.parse(JSON.parse(bodies[0] ?? '{}')).config).checkpoint).toBeUndefined();
    expect((await workflow.submit()).run_id).toBe(ids.run_id);
    workflow.invalidate();
    await expect(workflow.submit()).rejects.toThrow();
  });
  it('never selects CV summary or anonymous folds automatically', async () => {
    for (const kind of ['cross_validation_summary', 'cross_validation_fold']) {
      const { workflow, draft, data, metadata } = await setup(kind);
      await expect(workflow.prepare(draft, data, metadata)).rejects.toThrow('CV');
    }

    const { workflow, draft, data, metadata } = await setup('cross_validation_fold', 2);
    await workflow.prepare(draft, data, metadata);
    await workflow.preview();
    expect(workflow.approved).not.toBeNull();
  });
  it('rejects unrelated dataset/source and stale connection grants', async () => {
    const { workflow, draft, data, metadata, session } = await setup();
    await expect(
      workflow.prepare(
        { ...draft, checkpoint_source: { ...draft.checkpoint_source, generation: 'gen-000002' } },
        data,
        metadata,
      ),
    ).rejects.toThrow('exact');
    await workflow.prepare(draft, data, metadata);
    await workflow.preview();
    session.detach();
    expect(workflow.approved).toBeNull();
    await expect(workflow.prepare(draft, data, metadata)).rejects.toThrow();
  });
  it('offers verified typed source labels and disables summary/anonymous CV folds', async () => {
    for (const [kind, fold, selectable] of [
      ['best', undefined, true],
      ['periodic', undefined, true],
      ['interrupted', undefined, true],
      ['cross_validation_summary', undefined, false],
      ['cross_validation_fold', undefined, false],
      ['cross_validation_fold', 2, true],
    ] as const) {
      const { session, metadata } = await setup(kind, fold);
      const options = await checkpointOptions(session, [metadata]);
      expect(options[0]?.classification.kind).toBe(kind);
      expect(options[0]?.selectable).toBe(selectable);
      expect(options[0]?.label).toContain(kind.replaceAll('_', ' '));
      session.detach();
      await expect(checkpointOptions(session, [metadata])).rejects.toThrow();
    }
  });
  it('rejects prepared selections after reconnect before sending a preview request', async () => {
    const { workflow, draft, data, metadata, session, bodies } = await setup();
    await workflow.prepare(draft, data, metadata);
    expect(await session.reconnect('replacement-token')).toBe(true);

    await expect(workflow.preview()).rejects.toThrow('current connection');
    expect(bodies).toHaveLength(0);
    expect(workflow.approved).toBeNull();

    const currentData = await new CatalogApi(session).resolve('data', 'test.npz');
    const currentMetadata = (await new ArtifactClient(session).list(sourceRun))[0];

    if (!currentMetadata) throw new Error('Missing current metadata');
    await workflow.prepare(draft, currentData, currentMetadata);
    await workflow.preview();
    expect(bodies).toHaveLength(1);
    expect(workflow.approved).not.toBeNull();
    session.detach();
  });
  it('plans diagnostic plots using an explicitly classified CV fold', async () => {
    const { workflow, draft, data, metadata, bodies } = await setup('cross_validation_fold', 3);
    await workflow.prepare({ ...draft, operation: 'plot' }, data, metadata);
    await workflow.preview();
    expect(resultRequestSchema.parse(JSON.parse(bodies[0] ?? '{}')).operation).toBe('plot');
    expect(resultRequestSchema.parse(JSON.parse(bodies[0] ?? '{}')).checkpoint_source).toEqual({
      run_id: sourceRun,
      generation: metadata.generation,
    });
  });
});
