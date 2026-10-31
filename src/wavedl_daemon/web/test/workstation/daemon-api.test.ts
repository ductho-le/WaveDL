import type { FixtureResponse } from './fixture-schemas';
import { describe, expect, it, vi } from 'vitest';
import { ConnectionSession } from '../../app/lib/connection-session';
import { DaemonApi, DaemonApiError, parseJson, stripBootstrapUrl } from '../../app/lib/daemon-api';
import { translateClientDiagnostic } from '../../app/i18n/client-messages';
import { jsonSchema, jsonObjectSchema } from '../../shared/schemas/json';
import { planSchema, runPendingEvidenceSchema } from '../../app/lib/daemon-api-schemas';

describe('bounded JSON schema parity', () => {
  it('retains additive own members and rejects invalid numbers without narrowing finite metrics', () => {
    const value = parseJson('{"__proto__":{"metric":1e100},"nested":[null,true,""]}');
    const parsed = jsonSchema.parse(value);

    expect(parsed).toEqual(value);
    expect(Object.hasOwn(parsed, '__proto__')).toBe(true);
    expect(jsonSchema.safeParse({ metric: Infinity }).success).toBe(false);
    expect(jsonSchema.safeParse({ metric: NaN }).success).toBe(false);
    expect(jsonSchema.safeParse({ metric: undefined }).success).toBe(false);
    expect(jsonObjectSchema.safeParse({ nested: { value: undefined } }).success).toBe(false);
    expect(jsonObjectSchema.safeParse({ nested: { value: Infinity } }).success).toBe(false);
    expect(jsonObjectSchema.safeParse(parseJson('{"__proto__":{"value":null}}')).success).toBe(true);
  });
});

const token = 'a'.repeat(64);
const origin = 'http://127.0.0.1:8765';
const health = { daemon_id: 'daemon-a', build_id: 'build-a', protocol_min: 1, protocol_max: 1 };

const grant = (daemonId = health.daemon_id, suffix = 'g') => ({
  access_token: `session-${suffix.repeat(35)}`,
  token_type: 'Bearer',
  expires_at: Date.now() + 86_400_000,
  daemon_id: daemonId,
});

const json = (body: FixtureResponse, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const matrixId = '11111111-1111-1111-1111-111111111111';

const matrixRun = {
  plan_id: matrixId,
  run_id: matrixId,
  request_id: matrixId,
  backend: 'local',
  operation: 'train',
  attempt: 1,
  state: 'running',
  uncertainty: false,
  submission_requested: true,
  cancellation_requested: false,
  evidence: null,
};

const matrixPlan = { plan_id: matrixId, run_id: matrixId, request_id: matrixId, preview: {} };

describe('plan preview boundary', () => {
  it('keeps legacy missing manifests and additive preview fields but requires present manifests to be strings', () => {
    expect(planSchema.safeParse(matrixPlan).success).toBe(true);
    expect(planSchema.parse({ ...matrixPlan, preview: { manifest_json: '', additive: 1 } }).preview).toEqual({ manifest_json: '', additive: 1 });

    for (const manifest_json of [null, false, 1, [], {}]) {
      expect(planSchema.safeParse({ ...matrixPlan, preview: { manifest_json } }).success).toBe(false);
    }
  });
});

describe('run pending evidence boundary', () => {
  it('accepts absent and boolean flags with additive JSON but rejects malformed flags', () => {
    expect(runPendingEvidenceSchema.parse({ additive: [1, null] })).toEqual({ additive: [1, null] });
    expect(runPendingEvidenceSchema.parse({ native_pending: false }).native_pending).toBe(false);

    for (const native_pending of ['yes', null, 0, [], {}]) {
      expect(runPendingEvidenceSchema.safeParse({ native_pending }).success).toBe(false);
    }
  });
});

const negativeResponses = [
  ...[
    null,
    [],
    {},
    { ...health, daemon_id: ' ' },
    { ...health, protocol_min: 1.5 },
    { ...health, protocol_max: Number.MAX_SAFE_INTEGER + 1 },
  ].map((body) => ({
    method: 'health',
    body,
    code: 'response',
    key: 'invalidDaemonHealthResponse',
  })),
  ...[
    { ...health, protocol_min: 0 },
    { ...health, protocol_min: 2 },
    { ...health, protocol_max: 0 },
  ].map((body) => ({ method: 'health', body, code: 'protocol', key: 'daemonProtocolMismatch' })),
  ...[null, [], {}, { runs: null }].map((body) => ({
    method: 'list',
    body,
    code: 'response',
    key: 'invalidRunList',
  })),
  ...Object.keys(matrixRun).map((field) => ({
    method: 'run',
    body: { ...matrixRun, [field]: [] },
    code: 'response',
    key: 'invalidRunResponse',
  })),
  ...Object.keys(matrixRun).map((field) => ({
    method: 'run',
    body: { ...matrixRun, [field]: undefined },
    code: 'response',
    key: 'invalidRunResponse',
  })),
  ...[
    { attempt: 0 },
    { attempt: 1.5 },
    { attempt: Number.MAX_SAFE_INTEGER + 1 },
    { backend: '' },
    { operation: 'infer' },
    { state: 'finished' },
    { run_id: matrixId.toUpperCase().replace('1', 'A') },
  ].map((update) => ({
    method: 'run',
    body: { ...matrixRun, ...update },
    code: 'response',
    key: 'invalidRunResponse',
  })),
  { method: 'list', body: { runs: [{}] }, code: 'response', key: 'invalidRunResponse' },
  ...Object.keys(matrixPlan).map((field) => ({
    method: 'plan',
    body: { ...matrixPlan, [field]: [] },
    code: 'response',
    key: 'invalidPlanResponse',
  })),
  ...['bootstrap', 'session'].flatMap((method) =>
    Object.keys(grant()).map((field) => ({
      method,
      body: { ...grant(), [field]: null },
      code: 'response',
      key: method === 'bootstrap' ? 'invalidBootstrapResponse' : 'invalidSessionResponse',
    })),
  ),
  ...[
    ['native_pending', 'true', 'native_pending'],
    ['current_runtime_attempt', 0, 'current_runtime_attempt'],
    ['last_runtime_attempt', 0, 'last_runtime_attempt'],
    ['confidence', 'certain', 'confidence'],
    ['reason', false, 'reason'],
    ['runtime', null, 'runtime'],
    ['runtime', { hello_seen: 1 }, 'hello_seen'],
    ['runtime', { outcome: { status: 'completed' } }, 'status'],
    ['batch', { signal: 0 }, 'signal'],
    ['backend', { status: 'success' }, 'status'],
    ['stage_out', [], 'stage_out'],
    ['native_identity', {}, 'native_identity'],
  ].map(([field, value, diagnostic]) => ({
    method: 'run',
    body: { ...matrixRun, evidence: { [String(field)]: value } },
    code: 'response',
    key: 'invalidRunEvidence',
    params: { field: diagnostic },
  })),
  ...[
    ['runtime', { conflicting_exits: 1 }, 'conflicting_exits'],
    ['runtime', { state: 1 }, 'state'],
    ['runtime', { state_seq: -2 }, 'state_seq'],
    ['runtime', { exit_seq: -2 }, 'exit_seq'],
    ['runtime', { outcome: [] }, 'outcome'],
    ...['batch', 'stage_out', 'runtime'].flatMap((name) =>
      [
        ['status', 'completed'],
        ['exit_code', 1.5],
        ['signal', 0],
        ['detail', false],
      ].map(([field, value]) => [
        name,
        name === 'runtime' ? { outcome: { [String(field)]: value } } : { [String(field)]: value },
        field,
      ]),
    ),
    ...['backend', 'last_observation'].flatMap((name) =>
      [
        ['status', 'success'],
        ['native_state', 1],
        ['detail', false],
        ['attempt', 0],
        ['native_identity', {}],
        ['exit_code', 1.5],
        ['signal', 0],
        ['batch', []],
        ['stage_out', { signal: 0 }],
      ].map(([field, value]) => [
        name,
        { [String(field)]: value },
        field === 'stage_out' ? 'signal' : field,
      ]),
    ),
    ...['run_id', 'request_id', 'cluster', 'owner', 'job_id', 'restart_count'].map((field) => [
      'native_identity',
      {
        run_id: matrixId,
        request_id: matrixId,
        cluster: 'site',
        owner: 'user',
        job_id: '1',
        restart_count: 0,
        [field]: [],
      },
      'native_identity',
    ]),
    [
      'native_identity',
      {
        run_id: matrixId,
        request_id: matrixId,
        cluster: 'site',
        owner: 'user',
        job_id: 'invalid',
        restart_count: 0,
      },
      'native_identity',
    ],
  ].map(([field, value, diagnostic]) => ({
    method: 'run',
    body: { ...matrixRun, evidence: { [String(field)]: value } },
    code: 'response',
    key: 'invalidRunEvidence',
    params: { field: diagnostic },
  })),
  {
    method: 'run',
    body: { ...matrixRun, evidence: { native_pending: true, current_runtime_attempt: 1 } },
    code: 'response',
    key: 'pendingRuntimeConflict',
  },
];

describe('malformed response diagnostic compatibility matrix', () => {
  it.each(negativeResponses)('$method $key $body', async ({ method, body, code, key, ...rest }) => {
    const api = new DaemonApi(origin, {}, async (url) =>
      json(method === 'health' || !String(url).endsWith('/health') ? body : health),
    );

    api.setToken(token);

    const operation =
      method === 'health'
        ? api.health()
        : method === 'list'
          ? api.listRuns()
          : method === 'plan'
            ? api.plan({}, 'matrix')
            : method === 'bootstrap'
              ? api.exchangeBootstrapCode('code')
              : method === 'session'
                ? api.rotateSession()
                : api.getRun(matrixId);

    const expected = {
      code,
      clientMessageKey: key,
    };

    if ('params' in rest) Object.assign(expected, { clientMessageParams: rest.params });
    await expect(operation).rejects.toMatchObject(expected);
  });
});

describe('daemon transport', () => {
  it.each(['/api/datasets/inspect', '/api/datasets/preview'])(
    'keeps a dataset refusal on %s a sanitized request error without retiring the session',
    async (path) => {
      const api = new DaemonApi(origin, {}, async (url) =>
        json(
          String(url).endsWith('/health') ? health : { error: 'Bad request' },
          String(url).endsWith('/health') ? 200 : 400,
        ),
      );

      api.setToken(token);
      const selection = { root: 'data', path: 'data.npz' };
      const body = path.endsWith('/preview') ? { ...selection, view: 'all' } : selection;
      await expect(api.mutate(path, body)).rejects.toMatchObject({
        code: 'http',
        status: 400,
        clientMessageKey: 'daemonRequestFailed',
        daemonIdentityChanged: false,
      });
      await expect(api.health()).resolves.toEqual(health);
    },
  );

  it('builds API errors from catalog keys while preserving classification metadata', () => {
    const failure = DaemonApiError.fromKey('http', 'daemonRequestFailed', {
      status: 503,
      params: { status: 503 },
    });

    expect(failure).toBeInstanceOf(DaemonApiError);
    expect(failure).toMatchObject({
      code: 'http',
      message: 'Daemon request failed (503)',
      status: 503,
      buildMismatch: false,
      daemonIdentityChanged: false,
      clientMessageKey: 'daemonRequestFailed',
      clientMessageParams: { status: 503 },
    });

    const mismatch = DaemonApiError.fromKey('identity', 'daemonPinsChanged', {
      buildMismatch: true,
    });

    expect(mismatch).toMatchObject({ code: 'identity', buildMismatch: true });
  });

  it('does not expose a rejected credential on any own or inherited error property', async () => {
    const api = new DaemonApi(origin, {}, async () => json({}, 401));
    api.setToken(token);
    let failure: unknown;

    try {
      await api.health();
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(DaemonApiError);

    if (!(failure instanceof DaemonApiError)) throw new Error('Missing authentication failure');
    expect(failure.status).toBe(401);
    expect('credentialToken' in failure).toBe(false);
    let prototype: object | null = failure;

    while (prototype !== null) {
      for (const key of Reflect.ownKeys(prototype)) {
        const descriptor = Object.getOwnPropertyDescriptor(prototype, key);

        expect(descriptor?.value).not.toBe(token);

        if (descriptor?.get) expect(descriptor.get.call(failure)).not.toBe(token);
      }

      prototype = Object.getPrototypeOf(prototype);
    }

    expect(JSON.stringify(failure)).not.toContain(token);
  });

  it.each([
    { native_pending: 'true' },
    { native_pending: true, current_runtime_attempt: 1 },
    { last_runtime_attempt: 9007199254740992 },
    { current_runtime_attempt: 0 },
    { confidence: 'certain' },
    { reason: false },
    { runtime: [] },
    { runtime: { hello_seen: 'true' } },
    { runtime: { state_seq: 9007199254740992 } },
    { runtime: { outcome: { status: 'completed' } } },
    { backend: { status: 'success' } },
    { stage_out: { signal: 0 } },
    {
      native_identity: {
        run_id: '11111111-1111-1111-1111-111111111111',
        request_id: '11111111-1111-1111-1111-111111111111',
        cluster: 'site',
        owner: 'user',
        job_id: '9223372036854775808',
        restart_count: 1,
      },
    },
    {
      native_identity: {
        run_id: '22222222-2222-2222-2222-222222222222',
        request_id: '11111111-1111-1111-1111-111111111111',
        cluster: 'site',
        owner: 'user',
        job_id: '1',
        restart_count: 1,
      },
    },
  ])('rejects malformed known native/runtime evidence %j', async (evidence) => {
    const run = '11111111-1111-1111-1111-111111111111';

    const api = new DaemonApi(origin, {}, async (url) =>
      json(
        String(url).endsWith('/health')
          ? health
          : {
              plan_id: run,
              run_id: run,
              request_id: run,
              backend: 'slurm',
              operation: 'train',
              attempt: 1,
              state: 'queued',
              uncertainty: true,
              submission_requested: true,
              cancellation_requested: false,
              evidence,
            },
      ),
    );

    api.setToken(token);
    await expect(api.getRun(run)).rejects.toMatchObject({ code: 'response' });
  });
  it('preserves valid native pending history and additive evidence', async () => {
    const run = '11111111-1111-1111-1111-111111111111';

    const evidence = {
      native_pending: true,
      current_runtime_attempt: null,
      last_runtime_attempt: 1,
      confidence: 'unconfirmed',
      reason: 'requeued',
      future: { metric: 1e100 },
      native_identity: {
        run_id: run,
        request_id: run,
        cluster: 'site',
        owner: 'user',
        job_id: '9223372036854775807',
        restart_count: 2,
      },
    };

    const api = new DaemonApi(origin, {}, async (url) =>
      json(
        String(url).endsWith('/health')
          ? health
          : {
              plan_id: run,
              run_id: run,
              request_id: run,
              backend: 'slurm',
              operation: 'train',
              attempt: 1,
              state: 'queued',
              uncertainty: true,
              submission_requested: true,
              cancellation_requested: false,
              evidence,
            },
      ),
    );

    api.setToken(token);
    expect((await api.getRun(run)).evidence).toEqual(evidence);
  });
  it('does not let stale unauthorized responses revoke a replacement credential', async () => {
    let respond: ((response: Response) => void) | undefined;

    const api = new DaemonApi(
      origin,
      {},
      async () =>
        new Promise<Response>((resolve) => {
          respond = resolve;
        }),
    );

    api.setToken(token);
    const pending = expect(api.health()).rejects.toThrow('Connection credential changed');
    api.setToken('replacement');
    respond?.(json({}, 401));
    await pending;
    expect(api.authenticated).toBe(true);
  });
  it('drains in-flight requests before rotating and accepts their finished responses', async () => {
    let readStarted = false;
    let finishRead: ((response: Response) => void) | undefined;
    let rotationStarted = false;
    const replacement = grant(health.daemon_id, 'r');

    const api = new DaemonApi(origin, {}, async (url) => {
      const path = new URL(String(url)).pathname;

      if (path === '/api/health') return json(health);

      if (path === '/api/models') {
        readStarted = true;

        return new Promise<Response>((resolve) => {
          finishRead = resolve;
        });
      }

      if (path === '/api/session') {
        rotationStarted = true;

        return json(replacement);
      }

      throw new Error('Unexpected API route');
    });

    api.setToken(token);
    const reading = api.read('/api/models');
    await vi.waitFor(() => expect(readStarted).toBe(true));
    const rotating = api.rotateSession();
    await Promise.resolve();
    expect(rotationStarted).toBe(false);
    finishRead?.(json({ models: [] }));
    await expect(reading).resolves.toEqual({ models: [] });
    await expect(rotating).resolves.toEqual(replacement);
    expect(rotationStarted).toBe(true);
    expect(api.authenticated).toBe(true);
  });
  it('does not restore credentials from a revoked bootstrap body', async () => {
    let controller: ReadableStreamDefaultController<Uint8Array> | undefined;

    const body = new ReadableStream<Uint8Array>({
      start(value) {
        controller = value;
      },
    });

    const api = new DaemonApi(
      origin,
      {},
      async () => new Response(body, { headers: { 'Content-Type': 'application/json' } }),
    );

    const pending = expect(
      api.bootstrap(`${origin}/?bootstrap=once`, () => undefined),
    ).rejects.toThrow('Connection credential changed');

    // Let the response reader begin before revoking its connection scope.
    await Promise.resolve();
    await Promise.resolve();
    api.clearToken();
    controller?.enqueue(
      new TextEncoder().encode(JSON.stringify({ access_token: token, token_type: 'Bearer' })),
    );
    controller?.close();
    await pending;
    expect(api.authenticated).toBe(false);
  });
  it('bounds stalled JSON bodies and releases the reader', async () => {
    vi.useFakeTimers();
    let cancelled = false;

    try {
      const body = new ReadableStream<Uint8Array>({
        cancel() {
          cancelled = true;
        },
      });

      const api = new DaemonApi(
        origin,
        {},
        async () => new Response(body, { headers: { 'Content-Type': 'application/json' } }),
      );

      api.setToken(token);
      const pending = expect(api.health()).rejects.toThrow('Daemon response timed out');
      await vi.advanceTimersByTimeAsync(30001);
      await pending;
      expect(cancelled).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
  it('bounds stalled response headers without replacing caller cancellation', async () => {
    vi.useFakeTimers();

    try {
      const api = new DaemonApi(
        origin,
        {},
        async (_, init) =>
          new Promise<Response>((_, reject) => {
            init?.signal?.addEventListener('abort', () => reject(init.signal?.reason), {
              once: true,
            });
          }),
      );

      api.setToken(token);
      const pending = expect(api.health()).rejects.toThrow('Daemon request timed out');
      await vi.advanceTimersByTimeAsync(15001);
      await pending;
    } finally {
      vi.useRealTimers();
    }
  });
  it.each(['{"daemon_id":"a","daemon_id":"b"}', '{"nested":{"key":1,"k\\u0065y":2}}'])(
    'rejects ambiguous JSON members: %s',
    (text) => {
      expect(() => parseJson(text)).toThrow('Duplicate JSON member');
    },
  );
  it('allows repeated names in distinct objects and punctuation inside strings', () => {
    expect(parseJson('{"items":[{"key":"{,\\\"key\\\":1}"},{"key":2}]}')).toEqual({
      items: [{ key: '{,"key":1}' }, { key: 2 }],
    });
  });
  it('bounds request bytes and header values before making any request', async () => {
    const fetcher = vi.fn<typeof fetch>();
    const api = new DaemonApi(origin, {}, fetcher);
    api.setToken(token);
    await expect(api.mutate('/api/runs/plan', { text: '🌊'.repeat(300000) })).rejects.toThrow(
      'Request exceeds byte limit',
    );
    await expect(
      api.mutate('/api/runs/plan', {}, { idempotencyKey: 'bad\u0000key' }),
    ).rejects.toThrow('Invalid idempotency key');
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('strips bootstrap and pasted secrets before exchange and keeps bearer only in memory', async () => {
    const order: string[] = [];

    const fetcher = vi.fn<typeof fetch>(async (url, init) => {
      order.push('fetch');
      expect(init).toMatchObject({
        credentials: 'omit',
        redirect: 'error',
        cache: 'no-store',
        referrerPolicy: 'no-referrer',
      });

      if (String(url).endsWith('/bootstrap')) {
        expect(new Headers(init?.headers).has('Authorization')).toBe(false);
        expect(init?.body).toBe('{"code":"once"}');

        return json(grant());
      }

      if (String(url).endsWith('/session')) return json(grant());
      expect(new Headers(init?.headers).get('Authorization')).toBe(`Bearer ${token}`);

      return json(health);
    });

    const api = new DaemonApi(origin, {}, fetcher);
    await api.bootstrap(`${origin}/?bootstrap=once&token=secret&tab=runs#secret`, (url) => {
      order.push('strip');
      expect(url).toBe(`${origin}/?tab=runs`);
    });
    expect(order).toEqual(['strip', 'fetch']);
    expect(api.authenticated).toBe(true);
    expect(JSON.stringify(api)).not.toContain(token);
    expect(new DaemonApi(origin).authenticated).toBe(false);
    api.clearToken();
    expect(api.authenticated).toBe(false);
  });
  it('rejects duplicate bootstrap codes after removing them', async () => {
    const fetcher = vi.fn<typeof fetch>();
    const replace = vi.fn();
    await expect(
      new DaemonApi(origin, {}, fetcher).bootstrap(`${origin}/?bootstrap=a&bootstrap=b`, replace),
    ).rejects.toMatchObject({ code: 'authentication' });
    expect(replace).toHaveBeenCalledWith(`${origin}/`);
    expect(fetcher).not.toHaveBeenCalled();
    expect(stripBootstrapUrl(`${origin}/?ACCESS_TOKEN=x&ok=y`).url).toBe(`${origin}/?ok=y`);
  });
  it('re-prompts on 401 and forgets the credential', async () => {
    const api = new DaemonApi(
      origin,
      {},
      vi.fn<typeof fetch>(async () => json({}, 401)),
    );

    api.setToken(token);
    await expect(api.health()).rejects.toMatchObject({ code: 'authentication', status: 401 });
    expect(api.authenticated).toBe(false);
  });
  it('keeps a verified session after a domain conflict without retrying mutation', async () => {
    const fetcher = vi.fn<typeof fetch>(async (url) => {
      if (String(url).endsWith('/api/health')) return json(health);

      if (String(url).endsWith('/api/session')) return json(grant());

      return json({ error: 'Conflict' }, 409);
    });

    const session = new ConnectionSession({ origin, buildId: 'build-a', fetcher });
    expect(await session.connect(token)).toBe(true);
    const revision = session.snapshot().revision;
    const api = session.api;
    await expect(
      session.perform((current, signal) =>
        current.mutate('/api/runs/plan', {}, { signal, idempotencyKey: 'same-draft' }),
      ),
    ).rejects.toMatchObject({ code: 'request' });
    expect(session.snapshot()).toMatchObject({ status: 'connected', revision, health });
    expect(session.api).toBe(api);
    expect(api?.authenticated).toBe(true);
    expect(
      fetcher.mock.calls.filter(([url]) => String(url).endsWith('/api/runs/plan')),
    ).toHaveLength(1);
    session.detach();
  });
  it('retires authority when a domain route labels a daemon identity fence', async () => {
    const fetcher = vi.fn<typeof fetch>(async (url) => {
      if (String(url).endsWith('/api/health')) return json(health);

      if (String(url).endsWith('/api/session')) return json(grant());

      return new Response(JSON.stringify({ detail: 'ignored' }), {
        status: 409,
        headers: {
          'content-type': 'application/json',
          'X-WaveDL-Error-Code': 'daemon_identity_changed',
        },
      });
    });

    const owner = new ConnectionSession({ origin, buildId: 'build-a', fetcher });
    expect(await owner.connect(token)).toBe(true);
    await expect(
      owner.perform((api, signal) => api.read('/api/models', signal)),
    ).rejects.toMatchObject({ code: 'identity' });
    expect(owner.snapshot().status).toBe('skew');
    const message = owner.snapshot().message;

    if (message === null) throw new Error('Expected a coded connection notice');
    expect(translateClientDiagnostic('en', message)).toBe(
      'WaveDL was restarted or replaced. Sign in again.',
    );
    expect(owner.api).toBeNull();
    owner.dispose();
  });
  it('revokes the session when health conflicts with pinned daemon identity', async () => {
    let replaced = false;

    const fetcher = vi.fn<typeof fetch>(async (url) => {
      if (String(url).endsWith('/api/session')) return json(grant());

      return replaced ? json({ error: 'Conflict' }, 409) : json(health);
    });

    const session = new ConnectionSession({ origin, buildId: 'build-a', fetcher });
    expect(await session.connect(token)).toBe(true);
    const api = session.api;
    replaced = true;
    await expect(
      session.perform((current, signal) => current.health(signal)),
    ).rejects.toMatchObject({
      code: 'identity',
    });
    expect(session.snapshot().status).toBe('skew');
    expect(session.api).toBeNull();
    expect(api?.authenticated).toBe(false);
    session.detach();
  });
  it.each([
    [{ ...health, daemon_id: 'other' }, 'identity'],
    [{ ...health, build_id: 'other' }, 'identity'],
    [{ ...health, protocol_min: 2, protocol_max: 2 }, 'protocol'],
  ])('blocks mutation after daemon/build/protocol skew', async (body, code) => {
    const fetcher = vi.fn<typeof fetch>(async () => json(body));
    const api = new DaemonApi(origin, { daemonId: 'daemon-a', buildId: 'build-a' }, fetcher);
    api.setToken(token);
    await expect(api.mutate('/api/runs', { plan_id: 'p' })).rejects.toMatchObject({ code });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('checks health before mutation and fences its daemon identity', async () => {
    const fetcher = vi.fn<typeof fetch>(async (url, init) => {
      if (String(url).endsWith('/health')) return json(health);
      expect(new Headers(init?.headers).get('X-WaveDL-Daemon-ID')).toBe('daemon-a');
      expect(new Headers(init?.headers).get('Idempotency-Key')).toBe('stable-request');

      return json({ plan_id: 'p' });
    });

    const api = new DaemonApi(origin, {}, fetcher);
    api.setToken(token);
    await api.mutate('/api/runs/plan', {}, { idempotencyKey: 'stable-request' });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
  it('rejects cross-origin and encoded path escapes', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => json(health));
    const api = new DaemonApi(origin, {}, fetcher);
    api.setToken(token);
    await expect(api.read('//evil.test/api/runs')).rejects.toBeInstanceOf(DaemonApiError);
    await expect(api.read('/api/../secret')).rejects.toBeInstanceOf(DaemonApiError);
    await expect(api.read('/api/runs?token=secret')).rejects.toBeInstanceOf(DaemonApiError);
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('accepts valid large finite metrics without narrowing arbitrary JSON numbers', async () => {
    const api = new DaemonApi(
      origin,
      {},
      vi.fn<typeof fetch>(async (url) =>
        json(String(url).endsWith('/health') ? health : { metric: 1e100 }),
      ),
    );

    api.setToken(token);
    await expect(api.read('/api/runs')).resolves.toEqual({ metric: 1e100 });
  });
  it('runtime validates lifecycle state and run ownership', async () => {
    const run = '11111111-1111-1111-1111-111111111111';

    const view = {
      plan_id: run,
      run_id: run,
      request_id: run,
      backend: 'local',
      operation: 'train',
      attempt: null,
      state: 'submission_unknown',
      uncertainty: true,
      submission_requested: true,
      cancellation_requested: false,
      evidence: null,
    };

    let response = view;

    const api = new DaemonApi(
      origin,
      {},
      vi.fn<typeof fetch>(async (url) => json(String(url).endsWith('/health') ? health : response)),
    );

    api.setToken('short');
    await expect(api.getRun(run)).resolves.toEqual(view);
    response = { ...view, run_id: '22222222-2222-2222-2222-222222222222' };
    await expect(api.getRun(run)).rejects.toMatchObject({ code: 'response' });
    response = { ...view, state: 'success' };
    await expect(api.getRun(run)).rejects.toMatchObject({ code: 'response' });
  });
  it('rejects unsafe attempt coordinates while retaining pending native null attempts', async () => {
    const run = '11111111-1111-1111-1111-111111111111';

    const body = {
      plan_id: run,
      run_id: run,
      request_id: run,
      backend: 'slurm',
      operation: 'test',
      attempt: 9007199254740992,
      state: 'queued',
      uncertainty: true,
      submission_requested: true,
      cancellation_requested: false,
      evidence: {},
    };

    const api = new DaemonApi(
      origin,
      {},
      vi.fn<typeof fetch>(async (url) => json(String(url).endsWith('/health') ? health : body)),
    );

    api.setToken(token);
    await expect(api.getRun(run)).rejects.toMatchObject({ code: 'response' });
  });
  it('rejects browser origin mismatch without sending a credential', () => {
    vi.stubGlobal('location', { origin: 'http://localhost:8765' });

    try {
      expect(() => new DaemonApi(origin)).toThrow('current browser origin');
    } finally {
      vi.unstubAllGlobals();
    }
  });
  it('rejects control-bearing health identities', async () => {
    const api = new DaemonApi(
      origin,
      {},
      vi.fn<typeof fetch>(async () => json({ ...health, daemon_id: 'bad\r\nid' })),
    );

    api.setToken(token);
    await expect(api.health()).rejects.toMatchObject({ code: 'response' });
  });
  it('submits only immutable plans approved on this connection and permits retry', async () => {
    const run = '11111111-1111-1111-1111-111111111111';
    const planned = { plan_id: run, run_id: run, request_id: run, preview: { argv: ['train'] } };

    const submitted = {
      plan_id: run,
      run_id: run,
      request_id: run,
      backend: 'local',
      operation: 'train',
      attempt: null,
      state: 'pending',
      uncertainty: true,
      submission_requested: true,
      cancellation_requested: false,
      evidence: null,
    };

    const api = new DaemonApi(origin, {}, async (url) =>
      json(
        String(url).endsWith('/health')
          ? health
          : String(url).endsWith('/plan')
            ? planned
            : submitted,
      ),
    );

    api.setToken(token);
    const plan = await api.plan({}, 'stable-key');
    expect(api.approvedRequest(plan)).toEqual({});
    expect(() => api.approvedRequest({ ...plan })).toThrow();
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.preview.argv)).toBe(true);
    await expect(api.submit(plan)).resolves.toEqual(submitted);
    await expect(api.submit(plan)).resolves.toEqual(submitted);
    api.clearToken();
    api.setToken(token);
    expect(() => api.approvedRequest(plan)).toThrow();
    await expect(api.submit(plan)).rejects.toMatchObject({ code: 'identity' });
    await expect(api.submit({ ...plan })).rejects.toMatchObject({ code: 'identity' });
    const other = new DaemonApi(origin);
    other.setToken(token);
    expect(() => other.approvedRequest(plan)).toThrow();
    await expect(other.submit(plan)).rejects.toMatchObject({ code: 'identity' });
  });
  it('copies the exact request before health and POST waits and returns isolated approval copies', async () => {
    const run = '11111111-1111-1111-1111-111111111111';
    let releaseHealth = () => {};
    let releasePost = () => {};

    const healthGate = new Promise<void>((resolve) => {
      releaseHealth = resolve;
    });

    const postGate = new Promise<void>((resolve) => {
      releasePost = resolve;
    });

    const bodies: string[] = [];

    const api = new DaemonApi(origin, {}, async (url, init) => {
      if (String(url).endsWith('/health')) {
        await healthGate;

        return json(health);
      }

      bodies.push(String(init?.body));
      await postGate;

      return json({ plan_id: run, run_id: run, request_id: run, preview: {} });
    });

    api.setToken(token);

    const request = {
      config: { data_path: '/host/train.npz' },
      runtime: { python: '/host/venv/bin/python' },
      storage_map: [{ host_root: '/host', execution_root: '/exec' }],
    };

    const original = structuredClone(request);
    const work = api.plan(request, 'stable-key');
    request.config.data_path = '/mutated';
    request.storage_map[0].execution_root = '/changed';
    releaseHealth();
    await vi.waitFor(() => expect(bodies).toHaveLength(1));
    request.runtime.python = '/changed/python';
    releasePost();
    const plan = await work;
    expect(JSON.parse(bodies[0] ?? '{}')).toEqual(original);
    expect(api.approvedRequest(plan)).toEqual(original);
    const copy = api.approvedRequest(plan);
    copy.config = null;
    expect(api.approvedRequest(plan)).toEqual(original);
  });
});
