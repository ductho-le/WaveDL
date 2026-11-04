import { fixturePlanRequestSchema, fixtureTrainingManifestSchema } from './fixture-schemas';
import { resultRequestSchema } from '../../shared/schemas/result-request';
import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it, vi } from 'vitest';
import { translateClientDiagnostic } from '../../app/i18n/client-messages';
import { AutoPlotController, automaticPlotRequest } from '../../app/lib/autoplot-controller';
import { ConnectionSession } from './session-fixture';
import { trainingDefaults, trainingRequestSchema } from '../../shared/schemas/training-request';
import { mappedStoragePath, type StorageMapping } from '../../shared/schemas/storage-mapping';
import type { PlanView, RunView } from '../../app/lib/daemon-api';
import { parseJson } from '../../app/lib/daemon-api';
import { approvedTrainingManifestSchema } from '../../shared/schemas/autoplot';

const train: RunView = {
  run_id: '22222222-2222-4222-8222-222222222222',
  plan_id: '11111111-1111-4111-8111-111111111111',
  request_id: '33333333-3333-4333-8333-333333333333',
  backend: 'local',
  operation: 'train',
  attempt: 1,
  state: 'completed',
  uncertainty: false,
  submission_requested: true,
  cancellation_requested: false,
  evidence: null,
};

const plot: RunView = {
  ...train,
  run_id: '44444444-4444-4444-8444-444444444444',
  plan_id: '55555555-5555-4555-8555-555555555555',
  request_id: '66666666-6666-4666-8666-666666666666',
  operation: 'plot',
  state: 'submitted',
  attempt: null,
};

function approved(cv = 0): PlanView {
  return {
    run_id: train.run_id,
    plan_id: train.plan_id,
    request_id: train.request_id,
    preview: {
      target: 'local',
      topology: { num_machines: 1, processes_per_machine: 1, machine_rank: 0 },
      spec: {
        schema_version: 1,
        run_id: train.run_id,
        operation: 'train',
        resolved_config: {
          ...trainingDefaults,
          model: 'cnn',
          data_path: '/approved/train.npz',
          output_dir: '/isolated/run',
          cv,
        },
        runtime: {
          python: '/approved/python',
          mode: 'existing',
          device: 'cpu',
          environment: {},
          container: null,
        },
      },
    },
  };
}

describe('approved training manifest schema parity', () => {
  it.each([
    { operation: 'plot' },
    { resolved_config: [] },
    { runtime: { container: 'image' } },
    { runtime: {} },
  ])('rejects unsupported manifest records %j with the existing diagnostic', (patch) => {
    const plan = approved();
    const parsed = approvedTrainingManifestSchema.parse(plan.preview.spec ?? null);
    const manifest = { ...parsed, ...patch };
    const changed = { ...plan, preview: { ...plan.preview, spec: manifest } };

    try {
      automaticPlotRequest(train, changed);
      throw new Error('Expected manifest rejection');
    } catch (cause) {
      expect(cause).toMatchObject({ clientMessageKey: 'unsupportedTrainingManifest' });
    }
  });

  it('keeps additive manifest fields but rejects unknown resolved config keys', () => {
    const plan = approved();
    const manifest = approvedTrainingManifestSchema.parse(plan.preview.spec ?? null);

    const additive = {
      ...plan,
      preview: { ...plan.preview, spec: { ...manifest, additive: true } },
    };

    expect(automaticPlotRequest(train, additive).config.model).toBe('cnn');

    const unknown = {
      ...plan,
      preview: {
        ...plan.preview,
        spec: {
          ...manifest,
          resolved_config: { ...manifest.resolved_config, unknown_option: true },
        },
      },
    };

    try {
      automaticPlotRequest(train, unknown);
      throw new Error('Expected config rejection');
    } catch (cause) {
      expect(cause).toMatchObject({ clientMessageKey: 'unsupportedTrainingConfig' });
    }
  });
});

function json(body: FixtureResponse) {
  return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
}

async function setup(
  options: {
    deadlineMs?: number;
    checkpoints?: string[];
  } = {},
) {
  const keys: string[] = [],
    bodies: string[] = [],
    routes: string[] = [];

  let lost = false,
    plotSubmitted = false,
    losePlan = false,
    rejectPlotPolicy = false;

  let current = train;
  let inventoryFault: 'authentication' | 'identity' | 'protocol' | null = null;
  let inventoryGate = Promise.resolve();
  let planGate = Promise.resolve();
  let trainingPlan = approved();
  let alteredPlot: 'data' | 'runtime' | 'map' | 'checkpoint' | 'topology' | 'imports' | null = null;

  const session = new ConnectionSession({
    origin: 'http://localhost:8765',
    buildId: '1.8.1',
    fetcher: async (input, init) => {
      const path = new URL(String(input)).pathname;
      routes.push(`${init?.method ?? 'GET'} ${path}`);

      if (path === '/api/health')
        return json({
          daemon_id: inventoryFault === 'identity' ? 'other' : 'daemon',
          build_id: '1.8.1',
          protocol_min: inventoryFault === 'protocol' ? 2 : 1,
          protocol_max: inventoryFault === 'protocol' ? 2 : 1,
        });

      if (path === `/api/runs/${train.run_id}`) return json(current);

      if (path.endsWith('/artifacts')) {
        await inventoryGate;

        if (inventoryFault === 'authentication') return new Response('', { status: 401 });

        return json({
          files: [
            {
              path: 'best_checkpoint/model.safetensors',
              size: 25,
              modified: '2026-11-03T10:00:00+00:00',
            },
            { path: 'training_history.csv', size: 10, modified: '2026-11-03T10:00:01+00:00' },
          ],
          checkpoints: options.checkpoints ?? ['best_checkpoint', 'epoch_5_checkpoint'],
        });
      }

      if (path === '/api/runs/plan') {
        const request = fixturePlanRequestSchema.parse(parseJson(String(init?.body)));

        if (request.operation === 'train') return json(trainingPlan);
        keys.push(new Headers(init?.headers).get('Idempotency-Key') ?? '');
        bodies.push(String(init?.body));

        if (losePlan) return new Response('Upstream acknowledgment lost', { status: 503 });

        if (rejectPlotPolicy)
          return new Response('GPU allocation must equal processes_per_machine', { status: 422 });
        await planGate;
        const source = fixtureTrainingManifestSchema.parse(trainingPlan.preview.spec);

        const manifest = {
          schema_version: 1,
          run_id: plot.run_id,
          operation: 'plot',
          resolved_config: {
            ...request.config,
            data_path: source.resolved_config.data_path,
            import_modules: source.resolved_config.import_modules,
          },
          runtime: source.runtime,
          storage_map: source.storage_map ?? [],
          checkpoint_source: {
            ...request.checkpoint_source,
            path: `/isolated/run/output/${request.checkpoint_source.checkpoint}`,
          },
        };

        if (alteredPlot === 'data') manifest.resolved_config.data_path = '/changed/train.npz';

        if (alteredPlot === 'runtime') manifest.runtime.python = '/changed/python';

        if (alteredPlot === 'map')
          manifest.storage_map = [{ host_root: '/approved', execution_root: '/changed' }];

        if (alteredPlot === 'checkpoint')
          manifest.checkpoint_source.checkpoint = 'epoch_5_checkpoint';

        if (alteredPlot === 'imports')
          manifest.resolved_config.import_modules = ['unapproved.module'];

        return json({
          run_id: plot.run_id,
          plan_id: plot.plan_id,
          request_id: plot.request_id,
          preview: {
            target: current.backend,
            topology: alteredPlot === 'topology' ? trainingPlan.preview.topology : request.topology,
            spec: manifest,
          },
        });
      }

      if (path === '/api/runs') {
        plotSubmitted = true;

        if (lost) return new Response('Upstream acknowledgment lost', { status: 503 });

        return json({ ...plot, backend: current.backend });
      }

      if (path === `/api/runs/${plot.run_id}`)
        return json({ ...plot, backend: current.backend, submission_requested: plotSubmitted });
      throw new Error(`Unexpected ${path}`);
    },
  });

  await session.connect('token');
  const controller = new AutoPlotController(session, { pollMs: 60000, ...options });

  return {
    session,
    controller,
    keys,
    bodies,
    routes,
    alterPlot: (value: typeof alteredPlot) => {
      alteredPlot = value;
    },
    approve: async (
      cv = 0,
      backend: 'local' | 'slurm' = 'local',
      mappings?: StorageMapping[],
      sourceRun = train,
      processes = 1,
      imports: string[] = [],
      model = 'cnn',
    ) => {
      trainingPlan = approved(cv);
      trainingPlan.preview.topology = {
        num_machines: 1,
        processes_per_machine: processes,
        machine_rank: 0,
      };
      trainingPlan.run_id = sourceRun.run_id;
      trainingPlan.plan_id = sourceRun.plan_id;
      trainingPlan.request_id = sourceRun.request_id;
      const sourceManifest = fixtureTrainingManifestSchema.parse(trainingPlan.preview.spec);
      sourceManifest.run_id = sourceRun.run_id;
      sourceManifest.resolved_config.import_modules = imports;
      sourceManifest.resolved_config.model = model;
      trainingPlan.preview.spec = sourceManifest;
      trainingPlan.preview.target = backend;

      const request = trainingRequestSchema.parse({
        backend,
        operation: 'train',
        config: {
          ...trainingDefaults,
          model,
          data_path: '/approved/train.npz',
          import_modules: imports,
          cv,
        },
        runtime: { python: '/approved/python', mode: 'existing', device: 'cpu', environment: {} },
        topology: trainingPlan.preview.topology,
      });

      if (mappings !== undefined) request.storage_map = mappings;

      if (mappings) {
        const manifest = fixtureTrainingManifestSchema.parse(trainingPlan.preview.spec);
        manifest.storage_map = mappings;
        manifest.resolved_config.data_path = '/execution/train.npz';
        manifest.runtime.python = '/execution/python';
        manifest.resolved_config.import_modules = imports.map((item) =>
          item.endsWith('.py') ? mappedStoragePath(item, mappings) : item,
        );
        trainingPlan.preview.spec = manifest;
      }

      const result = await session.perform((api, signal) =>
        api.plan(request, 'training-key', signal),
      );

      routes.length = 0;

      return result;
    },
    holdArtifacts: () => {
      let release = () => {};

      inventoryGate = new Promise<void>((resolve) => {
        release = resolve;
      });

      return () => release();
    },
    holdPlotPlan: () => {
      let release = () => {};

      planGate = new Promise<void>((resolve) => {
        release = resolve;
      });

      return () => release();
    },
    faultInventory: (value: typeof inventoryFault) => {
      inventoryFault = value;
    },
    loseSubmit: () => {
      lost = true;
    },
    losePlan: (value: boolean) => {
      losePlan = value;
    },
    rejectPlotPolicy: () => {
      rejectPlotPolicy = true;
    },
    current: (value: RunView) => {
      current = value;
    },
  };
}

describe('actual captured automatic plot coordinator', () => {
  it('isolates captured imports and run identity and reconstructs the same durable plot key', async () => {
    const keys: string[] = [];

    for (let reconstruction = 0; reconstruction < 2; reconstruction++) {
      const owner = await setup();

      try {
        const imports = ['custom_models.registration', 'custom_losses'];
        const plan = await owner.approve(0, 'local', undefined, train, 1, imports, 'custom_model');
        const captured = structuredClone(train);
        owner.controller.capture(captured, plan);
        imports.push('later_unapproved_module');
        captured.backend = 'slurm';
        captured.run_id = plot.run_id;
        owner.losePlan(true);
        await owner.controller.observeFresh(train);
        owner.losePlan(false);
        await owner.controller.retry();
        expect(owner.bodies).toHaveLength(2);
        expect(owner.bodies[1]).toBe(owner.bodies[0]);
        expect(owner.keys[1]).toBe(owner.keys[0]);
        const request = resultRequestSchema.parse(parseJson(owner.bodies[0] ?? ''));
        expect(request.config.model).toBe('custom_model');
        expect(request.config.import_modules).toEqual([
          'custom_models.registration',
          'custom_losses',
        ]);
        expect(request.config.data_path).toBe('/approved/train.npz');
        expect(request.runtime.python).toBe('/approved/python');
        expect(request.operation).toBe('plot');
        keys.push(owner.keys[0] ?? '');
        await owner.controller.observeFresh(train);
        expect(owner.bodies).toHaveLength(2);
      } finally {
        owner.controller.dispose();
        owner.session.detach();
      }
    }

    expect(keys[0]).toBe(`web-autoplot-v1:${train.run_id}:${train.request_id}:best_checkpoint`);
    expect(keys[1]).toBe(keys[0]);
  });
  it.each([
    { uncertainty: true },
    { state: 'running' as const },
    { cancellation_requested: true },
    { run_id: plot.run_id },
    { request_id: plot.request_id },
    { plan_id: plot.plan_id },
    { attempt: 2 },
  ])('does not plan from ineligible fresh evidence %j', async (update) => {
    const owner = await setup();

    try {
      owner.controller.capture(train, await owner.approve());
      await owner.controller.observeFresh({ ...train, ...update });
      expect(owner.bodies).toHaveLength(0);
      expect(owner.routes).not.toContain('POST /api/runs');
    } finally {
      owner.controller.dispose();
      owner.session.detach();
    }
  });
  it('does not submit or rewrite resource authority when the daemon cannot express the plot plan', async () => {
    const owner = await setup();

    try {
      const plan = await owner.approve(0, 'local', undefined, train, 4);
      owner.rejectPlotPolicy();
      owner.controller.capture(train, plan);
      await owner.controller.observeFresh(train);
      expect(owner.controller.state.value.status).toBe('error');
      expect(owner.routes.includes('POST /api/runs')).toBe(false);
      await owner.controller.retry();
      expect(owner.controller.state.value.status).toBe('error');
      expect(owner.routes.includes('POST /api/runs')).toBe(false);
      expect(owner.bodies).toHaveLength(2);
      expect(owner.bodies[1]).toBe(owner.bodies[0]);
      expect(owner.keys[1]).toBe(owner.keys[0]);
    } finally {
      owner.controller.dispose();
      owner.session.detach();
    }
  });
  it.each([false, true])(
    'attests the derived singleprocess plot topology after distributed training (tampered=%s)',
    async (tampered) => {
      const owner = await setup();

      try {
        const plan = await owner.approve(0, 'local', undefined, train, 4);

        if (tampered) owner.alterPlot('topology');
        owner.controller.capture(train, plan);
        await owner.controller.observeFresh(train);
        const request = resultRequestSchema.parse(parseJson(owner.bodies[0] ?? ''));
        expect(request.topology).toEqual({
          num_machines: 1,
          processes_per_machine: 1,
          machine_rank: 0,
        });
        expect(request.runtime).toEqual({
          python: '/approved/python',
          mode: 'existing',
          device: 'cpu',
          environment: {},
        });
        expect(owner.controller.state.value.status).toBe(tampered ? 'unavailable' : 'submitted');
        expect(owner.routes.includes('POST /api/runs')).toBe(!tampered);
      } finally {
        owner.controller.dispose();
        owner.session.detach();
      }
    },
  );
  it.each(['capture', 'detach'])(
    'does not let an old pending plot plan overwrite a newer %s scope',
    async (change) => {
      const owner = await setup();
      const plan = await owner.approve();
      const release = owner.holdPlotPlan();

      try {
        owner.controller.capture(train, plan);
        const work = owner.controller.observeFresh(train);
        await vi.waitFor(() => expect(owner.bodies).toHaveLength(1));

        if (change === 'capture') {
          const next: RunView = {
            ...train,
            run_id: '77777777-7777-4777-8777-777777777777',
            plan_id: '88888888-8888-4888-8888-888888888888',
            request_id: '99999999-9999-4999-8999-999999999999',
          };

          owner.controller.capture(next, await owner.approve(0, 'local', undefined, next));
          expect(owner.controller.state.value.status).toBe('armed');
        } else owner.session.detach();
        release();
        await work;
        expect(owner.controller.state.value.status).toBe(
          change === 'capture' ? 'armed' : 'stopped',
        );
        expect(owner.routes).not.toContain('POST /api/runs');
      } finally {
        release();
        owner.controller.dispose();
        owner.session.detach();
      }
    },
  );
  it.each(['replace', 'clear'])(
    'disarms revoked source approval before inventory or planning: %s',
    async (change) => {
      const owner = await setup();
      const plan = await owner.approve();
      owner.controller.capture(train, plan);
      const api = owner.session.api;

      if (!api) throw new Error('Missing connected API');

      if (change === 'replace') api.setToken('replacement-token');
      else api.clearToken();
      await owner.controller.observeFresh(train);
      expect(owner.controller.state.value.status).toBe('stopped');
      expect(owner.routes).toEqual([]);
      api.setToken('token');
      await owner.controller.retry();
      owner.controller.capture(train, plan);
      await owner.controller.observeFresh(train);
      expect(owner.routes).toEqual([]);
      owner.controller.dispose();
      owner.session.detach();
    },
  );

  const alterations: Array<'data' | 'runtime' | 'map' | 'checkpoint' | 'imports'> = [
    'data',
    'runtime',
    'map',
    'checkpoint',
    'imports',
  ];

  it.each(alterations)(
    'does not submit plot preview that changes approved %s authority',
    async (alteration) => {
      const owner = await setup();

      const plan = await owner.approve(0, 'local', [
        { host_root: '/approved', execution_root: '/execution' },
      ]);

      owner.alterPlot(alteration);
      owner.controller.capture(train, plan);
      await owner.controller.observeFresh(train);
      expect(owner.controller.state.value.status).toBe('unavailable');
      expect(owner.bodies).toHaveLength(1);
      expect(owner.routes).not.toContain('POST /api/runs');
      await owner.controller.retry();
      expect(owner.bodies).toHaveLength(1);
      owner.controller.dispose();
      owner.session.detach();
    },
  );
  it('retains mapped Slurm source scope and disarms on requeue before retry', async () => {
    const owner = await setup();
    const source: RunView = { ...train, backend: 'slurm' };

    owner.current(source);

    const plan = await owner.approve(0, 'slurm', [
      { host_root: '/approved', execution_root: '/execution' },
    ]);

    owner.losePlan(true);
    owner.controller.capture(source, plan);
    await owner.controller.observeFresh(source);
    expect(owner.controller.state.value.status).toBe('error');
    owner.current({ ...source, attempt: 2 });
    owner.losePlan(false);
    await owner.controller.retry();
    expect(owner.controller.state.value.status).toBe('stopped');
    expect(owner.bodies).toHaveLength(1);
    const request = resultRequestSchema.parse(parseJson(owner.bodies[0] ?? '{}'));
    expect(request.backend).toBe('slurm');
    expect(request.storage_map).toEqual([{ host_root: '/approved', execution_root: '/execution' }]);
    expect(request.checkpoint_source.run_id).toBe(source.run_id);
    owner.controller.dispose();
    owner.session.detach();
  });
  it('uses exact host paths and map with a stable best pin/body on mapped retries', async () => {
    const owner = await setup();

    const mappings = [
      { host_root: '/approved', execution_root: '/execution' },
      { host_root: '/another-host', execution_root: '/execution' },
      { host_root: '/execution', execution_root: '/wrong-double-map' },
    ];

    const plan = await owner.approve(0, 'local', mappings);
    owner.losePlan(true);
    owner.controller.capture(train, plan);
    await owner.controller.observeFresh(train);
    expect(owner.controller.state.value.status).toBe('error');
    owner.losePlan(false);
    await owner.controller.retry();
    expect(owner.controller.state.value.status).toBe('submitted');
    expect(owner.bodies[0]).toBe(owner.bodies[1]);
    expect(owner.keys[0]).toBe(owner.keys[1]);
    const request = resultRequestSchema.parse(parseJson(owner.bodies[0] ?? '{}'));
    expect(request.storage_map).toEqual(mappings);
    expect(request.config.data_path).toBe('/approved/train.npz');
    expect(request.runtime.python).toBe('/approved/python');
    expect(request.checkpoint_source.checkpoint).toBe('best_checkpoint');
    owner.controller.dispose();
    owner.session.detach();
  });
  it('attests normalized registration imports after mapping host inputs into a derived plot plan', async () => {
    const owner = await setup();

    try {
      const mappings = [
        { host_root: '/approved', execution_root: '/execution' },
        { host_root: '/execution', execution_root: '/wrong-double-map' },
      ];

      const imports = ['registered.module', '/approved/custom model.py'];
      const plan = await owner.approve(0, 'local', mappings, train, 1, imports);
      owner.controller.capture(train, plan);
      await owner.controller.observeFresh(train);
      expect(owner.controller.state.value.status).toBe('submitted');
      expect(owner.routes.filter((route) => route === 'POST /api/runs')).toHaveLength(1);
      expect(
        resultRequestSchema.parse(parseJson(owner.bodies[0] ?? '{}')).config.import_modules,
      ).toEqual(imports);
      await owner.controller.observeFresh(train);
      expect(owner.bodies).toHaveLength(1);
    } finally {
      owner.controller.dispose();
      owner.session.detach();
    }
  });
  it('does not arm manufactured, foreign or revoked plan approvals', async () => {
    const foreign = await setup();
    const foreignPlan = await foreign.approve();

    for (const plan of [approved(), { ...foreignPlan }, foreignPlan]) {
      const owner = await setup();
      owner.controller.capture(train, plan);
      expect(owner.controller.state.value.status).toBe('unavailable');
      expect(owner.bodies).toHaveLength(0);
      owner.controller.dispose();
      owner.session.detach();
    }

    foreign.session.detach();
    await foreign.session.connect('token');
    foreign.controller.capture(train, foreignPlan);
    expect(foreign.controller.state.value.status).toBe('unavailable');
    foreign.controller.dispose();
    foreign.session.detach();
  });
  it('does not re-arm disabled work when the same training status is acknowledged again', async () => {
    const owner = await setup();
    owner.controller.capture(train, await owner.approve());
    expect(owner.controller.state.value.status).toBe('armed');
    owner.controller.stop();
    owner.controller.capture(train, await owner.approve());
    await owner.controller.observeFresh(train);
    await owner.controller.retry();
    expect(owner.controller.state.value.status).toBe('stopped');
    expect(owner.routes.some((route) => route.startsWith('POST'))).toBe(false);
    owner.controller.dispose();
    owner.session.detach();
  });
  it('disarms when the observed runtime attempt regresses', async () => {
    const owner = await setup();
    owner.controller.capture({ ...train, state: 'running', attempt: 2 }, await owner.approve());
    await owner.controller.observeFresh({ ...train, state: 'running', attempt: 1 });
    expect(owner.controller.state.value.status).toBe('stopped');
    expect(owner.routes.some((route) => route.startsWith('POST'))).toBe(false);
    owner.controller.dispose();
    owner.session.detach();
  });
  it('aborts late inventory on detach and bounds an otherwise stalled artifact request', async () => {
    const owner = await setup();
    const release = owner.holdArtifacts();
    owner.controller.capture(train, await owner.approve());
    const work = owner.controller.observeFresh(train);
    await vi.waitFor(() =>
      expect(owner.routes.some((route) => route.endsWith('/artifacts'))).toBe(true),
    );
    owner.session.detach();
    release();
    await work;
    expect(owner.controller.state.value.status).toBe('stopped');
    expect(owner.routes.some((route) => route.startsWith('POST'))).toBe(false);
    owner.controller.dispose();
    const deadline = await setup({ deadlineMs: 5 });
    const unblock = deadline.holdArtifacts();
    deadline.controller.capture(train, await deadline.approve());
    await deadline.controller.observeFresh(train);
    expect(deadline.controller.state.value.status).toBe('error');
    expect(deadline.routes.some((route) => route.startsWith('POST'))).toBe(false);
    unblock();
    deadline.controller.dispose();
    deadline.session.detach();
  });
  it('plans from the best checkpoint, never a later periodic or interrupted one', async () => {
    const { controller, session, bodies, approve } = await setup({
      checkpoints: ['best_checkpoint', 'epoch_9_checkpoint', 'interrupted_checkpoint'],
    });

    controller.capture(train, await approve());
    await controller.observeFresh(train);
    expect(controller.state.value.status).toBe('submitted');
    expect(controller.state.value.plotRunId).toBe(plot.run_id);
    const request = resultRequestSchema.parse(parseJson(bodies[0] ?? '{}'));
    expect(request.checkpoint_source).toEqual({
      run_id: train.run_id,
      checkpoint: 'best_checkpoint',
    });
    expect(request.config.data_path).toBe('/approved/train.npz');
    expect(request.runtime.python).toBe('/approved/python');
    await controller.observeFresh(train);
    expect(bodies).toHaveLength(1);
    controller.dispose();
    session.detach();
  });
  it.each([{ checkpoints: [] }, { checkpoints: ['epoch_5_checkpoint', 'interrupted_checkpoint'] }])(
    'refuses a run without a best checkpoint %j without planning, submitting or cancelling any plot',
    async ({ checkpoints }) => {
      const owner = await setup({ checkpoints });

      try {
        owner.controller.capture(train, await owner.approve());
        await owner.controller.observeFresh(train);
        expect(owner.controller.state.value.status).toBe('unavailable');
        expect(translateClientDiagnostic('en', owner.controller.state.value.message)).toContain(
          'incomplete',
        );
        expect(owner.session.snapshot().status).toBe('connected');
        await owner.controller.retry();
        await owner.controller.observeFresh(train);
        expect(owner.bodies).toHaveLength(0);
        expect(owner.routes.some((route) => route.startsWith('POST'))).toBe(false);
      } finally {
        owner.controller.dispose();
        owner.session.detach();
      }
    },
  );

  const faults: Array<'authentication' | 'identity' | 'protocol'> = [
    'authentication',
    'identity',
    'protocol',
  ];

  it.each(faults)('revokes automatic plotting after an inventory %s fault', async (fault) => {
    const owner = await setup();

    try {
      owner.controller.capture(train, await owner.approve());
      owner.faultInventory(fault);
      await owner.controller.observeFresh(train);
      expect(owner.controller.state.value.status).toBe('stopped');
      expect(owner.session.api).toBeNull();
      expect(owner.bodies).toHaveLength(0);
      expect(owner.routes).not.toContain('POST /api/runs');
    } finally {
      owner.controller.dispose();
      owner.session.detach();
    }
  });
  it('recovers lost submit acknowledgment by GET without a second submission', async () => {
    const { controller, session, routes, loseSubmit, approve } = await setup();
    loseSubmit();
    controller.capture(train, await approve());
    await controller.observeFresh(train);
    expect(controller.state.value.status).toBe('error');
    await controller.retry();
    expect(controller.state.value.status).toBe('submitted');
    expect(routes.filter((route) => route === 'POST /api/runs')).toHaveLength(1);
    expect(routes).toContain(`GET /api/runs/${plot.run_id}`);
    controller.dispose();
    session.detach();
  });
  it('explicit lost-plan retry keeps exact body/key and rejects requeued source before retry', async () => {
    const { controller, session, keys, bodies, losePlan, approve } = await setup();
    losePlan(true);
    controller.capture(train, await approve());
    await controller.observeFresh(train);
    expect(controller.state.value.status).toBe('error');
    losePlan(false);
    await controller.retry();
    expect(keys).toHaveLength(2);
    expect(keys[0]).toBe(keys[1]);
    expect(bodies[0]).toBe(bodies[1]);
    controller.dispose();
    session.detach();
    const next = await setup();
    next.loseSubmit();
    next.controller.capture(train, await next.approve());
    await next.controller.observeFresh(train);
    next.current({ ...train, attempt: 2 });
    await next.controller.retry();
    expect(next.controller.state.value.status).toBe('stopped');
    expect(next.routes.filter((route) => route === 'POST /api/runs')).toHaveLength(1);
    next.controller.dispose();
    next.session.detach();
  });
  it('CV, pending, foreign and detached scopes never create plans or cancel server work', async () => {
    const { controller, session, routes, approve } = await setup();
    controller.capture(train, await approve(3));
    await controller.observeFresh(train);
    expect(controller.state.value.status).toBe('stopped');
    controller.capture(train, await approve());
    await controller.observeFresh({ ...train, state: 'pending' });
    await controller.observeFresh({ ...train, run_id: plot.run_id });
    session.detach();
    await controller.observeFresh(train);
    expect(routes.filter((route) => route.startsWith('POST'))).toHaveLength(0);
    controller.dispose();
  });
  it('refuses unsupported normalized manifest instead of falling back to draft defaults', () => {
    const plan = approved();
    expect(automaticPlotRequest(train, plan).config.data_path).toBe('/approved/train.npz');
    expect(() => automaticPlotRequest(train, { ...plan, preview: {} })).toThrow();
    const manifest = fixtureTrainingManifestSchema.parse(plan.preview.spec);
    manifest.runtime.container = 'image';
    expect(() =>
      automaticPlotRequest(train, {
        ...plan,
        preview: { ...plan.preview, spec: manifest },
      }),
    ).toThrow();
  });
  it('refuses mapped execution-path inputs and lexical/symlink mismatches, retaining venv python spelling', () => {
    const plan = approved();
    const manifest = fixtureTrainingManifestSchema.parse(plan.preview.spec);

    const map = [
      { host_root: '/approved', execution_root: '/execution' },
      { host_root: '/execution', execution_root: '/double-mapped' },
    ];

    manifest.storage_map = map;
    manifest.resolved_config.data_path = '/execution/train.npz';
    manifest.runtime.python = '/execution/venv/bin/python';
    plan.preview.spec = manifest;

    const original = trainingRequestSchema.parse({
      backend: 'local',
      operation: 'train',
      storage_map: map,
      config: { ...trainingDefaults, data_path: '/approved/train.npz', model: 'different-draft' },
      runtime: {
        python: '/approved/venv/bin/python',
        mode: 'existing',
        device: 'cpu',
        environment: {},
      },
      topology: plan.preview.topology,
    });

    const request = automaticPlotRequest(train, plan, original);
    expect(request.config.data_path).toBe('/approved/train.npz');
    expect(request.config.model).toBe('cnn');
    expect(request.runtime.python).toBe('/approved/venv/bin/python');
    expect(() => automaticPlotRequest(train, plan)).toThrow();
    expect(() =>
      automaticPlotRequest(train, plan, {
        ...original,
        config: { ...original.config, data_path: '/execution/train.npz' },
      }),
    ).toThrow();
    expect(() =>
      automaticPlotRequest(train, plan, {
        ...original,
        runtime: { ...original.runtime, python: '/usr/bin/python' },
      }),
    ).toThrow();
    manifest.resolved_config.data_path = '/real-symlink-target/train.npz';
    plan.preview.spec = manifest;
    expect(() => automaticPlotRequest(train, plan, original)).toThrow();
  });
  it('restores approved host registration paths for exactly one forward mapping', () => {
    const plan = approved();
    const manifest = fixtureTrainingManifestSchema.parse(plan.preview.spec);

    const mappings = [
      { host_root: '/approved', execution_root: '/execution' },
      { host_root: '/execution', execution_root: '/double-mapped' },
    ];

    manifest.storage_map = mappings;
    manifest.resolved_config.data_path = '/execution/train.npz';
    manifest.runtime.python = '/execution/python';
    manifest.resolved_config.import_modules = ['registered.module', '/execution/custom model.py'];
    plan.preview.spec = manifest;

    const original = trainingRequestSchema.parse({
      backend: 'local',
      operation: 'train',
      storage_map: mappings,
      config: {
        ...trainingDefaults,
        data_path: '/approved/train.npz',
        import_modules: ['registered.module', '/approved/custom model.py'],
      },
      runtime: { python: '/approved/python', mode: 'existing', device: 'cpu', environment: {} },
      topology: plan.preview.topology,
    });

    const request = automaticPlotRequest(train, plan, original);
    expect(request.config.import_modules).toEqual(original.config.import_modules);
    expect(fixtureTrainingManifestSchema.parse(plan.preview.spec)).toEqual(manifest);

    for (const imports of [
      ['registered.module', '/execution/custom model.py'],
      ['registered.module', 'custom model.py'],
      ['different.module', '/approved/custom model.py'],
      ['/approved/custom model.py', 'registered.module'],
      ['registered.module'],
      ['registered.module', '/approved/../custom model.py'],
    ]) {
      expect(() =>
        automaticPlotRequest(train, plan, {
          ...original,
          config: { ...original.config, import_modules: imports },
        }),
      ).toThrow();
    }
  });
});
