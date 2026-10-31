import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it, vi } from 'vitest';
import { AdoptionWorkflow } from '../../app/lib/adoption';
import { CatalogApi } from '../../app/lib/catalog-api';
import { ConnectionSession } from './session-fixture';

const adopted = {
  plan_id: '11111111-1111-4111-8111-111111111111',
  run_id: '22222222-2222-4222-8222-222222222222',
  request_id: '33333333-3333-4333-8333-333333333333',
  operation: 'train',
  attempt: 2,
  backend: 'local',
  state: 'supervision_lost',
  uncertainty: true,
  submission_requested: true,
  cancellation_requested: false,
  backend_status: 'unknown',
  adopted: true,
};

function json(body: FixtureResponse, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

async function setup(reply: () => Promise<Response> = async () => json(adopted)) {
  const calls: { path: string; body: string; key: string | null }[] = [];
  let daemonId = 'daemon';

  const session = new ConnectionSession({
    origin: 'http://localhost:8765',
    buildId: '1.8.1',
    fetcher: async (url, init) => {
      const path = new URL(String(url)).pathname;

      if (path === '/api/health')
        return json({ daemon_id: daemonId, build_id: '1.8.1', protocol_min: 1, protocol_max: 1 });

      if (path === '/api/browse/resolve')
        return json({
          root: 'runs',
          relative_path: 'standalone',
          kind: 'directory',
          path: '/runs/standalone',
        });
      calls.push({
        path,
        body: String(init?.body),
        key: new Headers(init?.headers).get('Idempotency-Key'),
      });

      return reply();
    },
  });

  await session.connect('token');
  const directory = await new CatalogApi(session).resolve('runs', 'standalone');
  const workflow = new AdoptionWorkflow(session, () => 'fixed-key');

  return {
    session,
    directory,
    workflow,
    calls,
    changeDaemon: () => {
      daemonId = 'other';
    },
  };
}

describe('explicit standalone run adoption', () => {
  it('registers only on explicit confirmation and validates exact uncertainty contract', async () => {
    const { workflow, directory, calls } = await setup();
    workflow.prepare(directory, 2);
    expect(calls).toHaveLength(0);
    expect(Object.isFrozen(workflow.request)).toBe(true);
    expect(Object.isFrozen(workflow.source)).toBe(true);
    const result = await workflow.adopt();
    expect(result).toEqual(adopted);
    expect(Object.isFrozen(result)).toBe(true);
    expect(calls).toEqual([
      {
        path: '/api/runs/adopt',
        body: JSON.stringify({ attempt: 2, path: '/runs/standalone' }),
        key: 'fixed-key',
      },
    ]);
    expect(await workflow.adopt()).toBe(result);
    expect(calls).toHaveLength(1);
  });
  it('rejects fake, foreign and stale grants without POST', async () => {
    const a = await setup();
    const b = await setup();
    expect(() => a.workflow.prepare({ ...a.directory }, 2)).toThrow();
    expect(() => a.workflow.prepare(b.directory, 2)).toThrow();
    expect(() => a.workflow.prepare(a.directory, 0)).toThrow();
    expect(() => a.workflow.prepare(a.directory, 1.5)).toThrow();
    a.workflow.prepare(a.directory, 2);
    await a.session.reconnect('token');
    await expect(a.workflow.adopt()).rejects.toThrow();
    expect(() => a.workflow.prepare(a.directory, 2)).toThrow();
    expect(a.calls).toHaveLength(0);
  });
  it('reconfirms ownership after reconnect without rotating the registration key', async () => {
    const { workflow, directory, session, calls } = await setup();
    workflow.prepare(directory, 2);
    await workflow.adopt();
    await session.reconnect('token');
    expect(workflow.result).toBeNull();
    const selected = await new CatalogApi(session).resolve('runs', 'standalone');
    workflow.prepare(selected, 2);
    await workflow.adopt();
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual(calls[1]);
  });
  it('retains frozen request and key after lost ack and requires fresh browse revalidation', async () => {
    let fail = true;

    const { workflow, directory, session, calls } = await setup(async () => {
      if (fail) throw new Error('lost acknowledgement');

      return json(adopted);
    });

    workflow.prepare(directory, 2);
    const frozen = workflow.request;
    await expect(workflow.adopt()).rejects.toThrow('unknown');
    expect(workflow.status).toBe('acknowledgement_unknown');
    fail = false;
    await session.reconnect('token');
    await expect(workflow.adopt()).rejects.toThrow();
    const selected = await new CatalogApi(session).resolve('runs', 'standalone');
    expect(() => workflow.prepare(selected, 3)).toThrow('same');
    workflow.prepare(selected, 2);
    expect(workflow.request).toBe(frozen);
    expect((await workflow.adopt()).run_id).toBe(adopted.run_id);
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual(calls[1]);
  });
  it('never moves an unknown request to another daemon', async () => {
    const { workflow, directory, session, changeDaemon, calls } = await setup(async () => {
      throw new Error('lost');
    });

    workflow.prepare(directory, 2);
    await expect(workflow.adopt()).rejects.toThrow();
    session.detach();
    changeDaemon();
    await session.connect('token');
    const next = await new CatalogApi(session).resolve('runs', 'standalone');
    expect(() => workflow.prepare(next, 2)).toThrow('same');
    expect(calls).toHaveLength(1);
  });
  it.each([400, 403, 409, 503])('classifies public HTTP %s safely', async (status) => {
    const { workflow, directory, session } = await setup(async () =>
      json({ detail: 'private filesystem path' }, status),
    );

    workflow.prepare(directory, 2);
    await expect(workflow.adopt()).rejects.toMatchObject({ status });
    expect(workflow.status).toBe(status < 500 ? 'rejected' : 'acknowledgement_unknown');
    expect(workflow.result).toBeNull();

    if (status < 500) expect(session.snapshot().status).toBe('connected');
  });
  it.each([
    { run_id: '../foreign' },
    { request_id: 'INVALID' },
    { plan_id: null },
    { attempt: 3 },
    { state: 'completed' },
    { backend: 'slurm' },
    { uncertainty: false },
    { adopted: false },
    { cancellation_requested: true },
    { extra_manifest: {} },
    { backend_status: 'running' },
  ])('does not fabricate a valid acknowledgement from %j', async (invalid) => {
    const { workflow, directory } = await setup(async () => json({ ...adopted, ...invalid }));
    workflow.prepare(directory, 2);
    await expect(workflow.adopt()).rejects.toThrow('unknown');
    expect(workflow.result).toBeNull();
  });
  it('prevents repeat POST while waiting and does not cancel a process when detached', async () => {
    let finish: ((response: Response) => void) | undefined;

    const { workflow, directory, session, calls } = await setup(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );

    workflow.prepare(directory, 2);
    const pending = workflow.adopt();
    const settled = pending.catch(() => null);
    await vi.waitFor(() => expect(calls).toHaveLength(1));
    await expect(workflow.adopt()).rejects.toThrow('pending');
    expect(() => workflow.prepare(directory, 2)).toThrow('Wait');
    session.detach();
    expect(await settled).toBeNull();
    finish?.(json(adopted));
    expect(calls.every((call) => call.path === '/api/runs/adopt')).toBe(true);
    expect(calls).toHaveLength(1);
    expect(workflow.result).toBeNull();
  });
});
