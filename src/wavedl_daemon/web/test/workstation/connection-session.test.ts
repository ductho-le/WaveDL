import type { FixtureResponse } from './fixture-schemas';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConnectionSession, ConnectionSessionError } from '../../app/lib/connection-session';
import { DaemonApi, DaemonApiError } from '../../app/lib/daemon-api';
import { healthSchema } from '../../app/lib/daemon-api-schemas';

const origin = 'http://localhost:8765';
const health = { daemon_id: 'daemon-one', build_id: '1.8.1', protocol_min: 1, protocol_max: 1 };

function json(body: FixtureResponse, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function session(fetcher: typeof fetch, online = true) {
  let daemonId = health.daemon_id;

  const wrapped: typeof fetch = async (input, init) => {
    const url = String(input);

    if (url.endsWith('/api/session') && init?.method === 'POST')
      return json({
        access_token: `session-${'x'.repeat(35)}`,
        token_type: 'Bearer',
        expires_at: Date.now() + 86_400_000,
        daemon_id: daemonId,
      });
    const response = await fetcher(input, init);

    if (url.endsWith('/api/health') && response.ok) {
      try {
        const parsed = healthSchema
          .pick({ daemon_id: true })
          .safeParse(await response.clone().json());

        if (parsed.success) daemonId = parsed.data.daemon_id;
      } catch {
        // The session under test owns malformed-response assertions.
      }
    }

    return response;
  };

  return new ConnectionSession({
    origin,
    buildId: '1.8.1',
    fetcher: wrapped,
    online,
    timeoutMs: 1000,
  });
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('browser connection authority', () => {
  it('builds session errors from catalog keys and preserves the safe status', () => {
    const failure = ConnectionSessionError.fromKey('identity', 'daemonReplaced', {
      status: 409,
    });

    expect(failure).toBeInstanceOf(ConnectionSessionError);
    expect(failure).toMatchObject({
      code: 'identity',
      message: 'WaveDL was restarted or replaced. Sign in again.',
      status: 409,
      clientMessageKey: 'daemonReplaced',
    });
  });

  it.each([400, 500])(
    'preserves only the safe HTTP %s status through session error sanitization',
    async (status) => {
      const owner = session(async (input) =>
        String(input).endsWith('/api/health')
          ? json(health)
          : json({ detail: 'secret /private/path credential' }, status),
      );

      expect(await owner.connect('credential')).toBe(true);

      try {
        await owner.perform((api, signal) => api.read('/api/models', signal));
        throw new Error('Expected HTTP failure');
      } catch (error) {
        expect(error).toBeInstanceOf(ConnectionSessionError);

        if (!(error instanceof ConnectionSessionError)) throw error;
        expect(error.code).toBe('request');
        expect(error.status).toBe(status);
        expect(error.message).toBe(
          'The daemon request failed. Verify connectivity and the operation outcome before retrying.',
        );
        expect(String(error)).not.toContain('/private/path');
        expect(String(error)).not.toContain('secret');
      }

      expect(owner.snapshot().status).toBe('connected');
      owner.detach();
    },
  );

  it.each([
    new DaemonApiError('http', 'secret', 399),
    new DaemonApiError('http', 'secret', 600),
    new DaemonApiError('http', 'secret', 400.5),
    new DaemonApiError('response', 'secret', 400),
    new DaemonApiError('authentication', 'secret', 401),
  ])('does not attach HTTP status to invalid or non-HTTP errors', async (failure) => {
    const owner = session(async () => json(health));
    expect(await owner.connect('credential')).toBe(true);

    try {
      await owner.perform(async () => {
        throw failure;
      });
      throw new Error('Expected failure');
    } catch (error) {
      if (!(error instanceof ConnectionSessionError)) throw error;
      expect(error.status).toBeUndefined();
      expect(error.message).not.toContain('secret');
    }

    owner.detach();
  });
  it('binds the default browser fetch receiver for bootstrap and bearer connect', async () => {
    const receiverSensitive = vi.fn(function (this: typeof globalThis, input: RequestInfo | URL) {
      expect(this).toBe(globalThis);

      const body = String(input).endsWith('/api/bootstrap')
        ? {
            token_type: 'Bearer',
            access_token: `session-${'b'.repeat(35)}`,
            expires_at: Date.now() + 86_400_000,
            daemon_id: health.daemon_id,
          }
        : String(input).endsWith('/api/session')
          ? {
              token_type: 'Bearer',
              access_token: `session-${'c'.repeat(35)}`,
              expires_at: Date.now() + 86_400_000,
              daemon_id: health.daemon_id,
            }
          : health;

      return Promise.resolve(json(body));
    });

    vi.stubGlobal('fetch', receiverSensitive);
    const bootstrapped = new ConnectionSession({ origin, buildId: '1.8.1' });
    expect(await bootstrapped.bootstrap(`${origin}/?bootstrap=secret`, () => undefined)).toBe(true);
    const manual = new ConnectionSession({ origin, buildId: '1.8.1' });
    expect(await manual.connect('credential')).toBe(true);
    const direct = new DaemonApi(origin, { buildId: '1.8.1' });
    direct.setToken('credential');
    expect(await direct.health()).toEqual(health);
    expect(receiverSensitive).toHaveBeenCalledTimes(5);
    bootstrapped.detach();
    manual.detach();
  });
  it('removes every bootstrap URL secret before the first request and verifies build', async () => {
    let clean = '';

    const fetcher = vi.fn<typeof fetch>(async (input, init) => {
      expect(clean).toBe(`${origin}/?view=runs`);
      expect(init?.credentials).toBe('omit');
      expect(init?.referrerPolicy).toBe('no-referrer');

      return String(input).endsWith('/api/bootstrap')
        ? json({
            token_type: 'Bearer',
            access_token: `session-${'d'.repeat(35)}`,
            expires_at: Date.now() + 86_400_000,
            daemon_id: health.daemon_id,
          })
        : json(health);
    });

    const owner = session(fetcher);
    expect(
      await owner.bootstrap(`${origin}/?bootstrap=secret&token=secret&view=runs#private`, (url) => {
        clean = url;
      }),
    ).toBe(true);
    expect(owner.snapshot().health).toEqual(health);
    expect(JSON.stringify(owner.snapshot())).not.toContain('private-token');
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
  it('strips bootstrap secrets while offline, without making any request', async () => {
    const fetcher = vi.fn<typeof fetch>();
    let clean = '';
    const owner = session(fetcher, false);
    expect(
      await owner.bootstrap(`${origin}/?bootstrap=secret`, (url) => {
        clean = url;
      }),
    ).toBe(false);
    expect(clean).toBe(`${origin}/`);
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('rejects ambiguous bootstrap and safely handles malformed input', async () => {
    const fetcher = vi.fn<typeof fetch>();
    const owner = session(fetcher);
    let clean = '';
    expect(
      await owner.bootstrap(`${origin}/?bootstrap=one&bootstrap=two`, (url) => {
        clean = url;
      }),
    ).toBe(false);
    expect(clean).toBe(`${origin}/`);
    expect(owner.snapshot().status).toBe('authentication');
    expect(await owner.bootstrap('secret malformed URL', () => undefined)).toBe(false);
    expect(JSON.stringify(owner.snapshot())).not.toContain('secret malformed');
    expect(fetcher).not.toHaveBeenCalled();
  });
  it.each([
    ['build', { ...health, build_id: '2.0.0' }, 'skew'],
    ['protocol', { ...health, protocol_min: 2, protocol_max: 2 }, 'skew'],
    ['malformed', { daemon_id: 'private-response' }, 'error'],
  ])('fails closed on %s mismatch', async (_name, body, status) => {
    const owner = session(async () => json(body));
    expect(await owner.connect('credential')).toBe(false);
    expect(owner.snapshot().status).toBe(status);
    expect(owner.api).toBeNull();
  });
  it('marks only an actual build-id mismatch as eligible for the reload prompt', async () => {
    const buildMismatch = session(async () => json({ ...health, build_id: '2.0.0' }));
    expect(await buildMismatch.connect('credential')).toBe(false);
    expect(buildMismatch.snapshot()).toMatchObject({ status: 'skew', buildMismatch: true });
    // A same-daemon build change is fixed by updating the client, not by signing in again.
    expect(buildMismatch.snapshot().message).toMatchObject({
      code: 'skew',
      clientMessageKey: 'daemonScopeChanged',
    });

    const protocolMismatch = session(async () =>
      json({ ...health, protocol_min: 2, protocol_max: 2 }),
    );

    expect(await protocolMismatch.connect('credential')).toBe(false);
    expect(protocolMismatch.snapshot()).toMatchObject({ status: 'skew', buildMismatch: false });
  });
  it('retains identity pins across reauthentication and refuses port reuse until detach', async () => {
    let current = health;
    const owner = session(async () => json(current));
    expect(await owner.connect('credential')).toBe(true);
    current = { ...health, daemon_id: 'daemon-two' };
    expect(await owner.reconnect('replacement')).toBe(false);
    expect(owner.snapshot().status).toBe('skew');
    owner.detach();
    expect(await owner.connect('replacement')).toBe(true);
    expect(owner.snapshot().health?.daemon_id).toBe('daemon-two');
  });
  it('401 revokes simultaneous active work and prompts for a new credential', async () => {
    let unauthorized = false;
    const owner = session(async () => (unauthorized ? json({}, 401) : json(health)));
    await owner.connect('credential');
    let activeSignal: AbortSignal | undefined;

    const active = owner.perform(
      (_api, signal) => {
        activeSignal = signal;

        return new Promise<string>(() => undefined);
      },
      { timeoutMs: 0 },
    );

    const rejection = expect(active).rejects.toMatchObject({ code: 'stale' });
    unauthorized = true;
    await expect(owner.perform((api, signal) => api.health(signal))).rejects.toMatchObject({
      code: 'authentication',
    });
    await rejection;
    expect(activeSignal?.aborted).toBe(true);
    expect(owner.api).toBeNull();
    expect(owner.snapshot().status).toBe('authentication');
  });
  it('identity or response failure revokes all work, not just the failing request', async () => {
    const owner = session(async () => json(health));
    await owner.connect('credential');
    const work = owner.perform(() => new Promise<void>(() => undefined), { timeoutMs: 0 });
    const rejection = expect(work).rejects.toMatchObject({ code: 'stale' });
    await expect(
      owner.perform(async () => {
        throw new DaemonApiError('identity', 'private-native-secret');
      }),
    ).rejects.toMatchObject({ code: 'identity' });
    await rejection;
    expect(owner.snapshot().status).toBe('skew');
    expect(owner.snapshot().message).not.toContain('private-native-secret');
  });
  it('detaches without submitting, cancelling, or exposing stale results', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => json(health));
    const owner = session(fetcher);
    await owner.connect('credential');
    let resolve: ((value: string) => void) | undefined;

    const work = owner.perform(
      () =>
        new Promise<string>((done) => {
          resolve = done;
        }),
    );

    const rejected = expect(work).rejects.toBeInstanceOf(ConnectionSessionError);
    owner.detach();
    resolve?.('stale');
    await rejected;
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(owner.snapshot().status).toBe('disconnected');
  });
  it('ignores a stale connect completing after detach or a newer connect', async () => {
    let resolve: ((response: Response) => void) | undefined;
    let first = true;

    const owner = session(async () => {
      if (first) {
        first = false;

        return new Promise<Response>((done) => {
          resolve = done;
        });
      }

      return json(health);
    });

    const stale = owner.connect('old-token');
    expect(await owner.connect('new-token')).toBe(true);
    resolve?.(json({ ...health, daemon_id: 'stale-daemon' }));
    expect(await stale).toBe(false);
    expect(owner.snapshot().health).toEqual(health);
  });
  it('offline aborts work and online requires an explicit verified reconnect', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => json(health));
    const owner = session(fetcher);
    await owner.connect('credential');
    owner.setOnline(false);
    await expect(owner.perform(async () => 'unsafe')).rejects.toMatchObject({
      code: 'unavailable',
    });
    owner.setOnline(true);
    expect(owner.snapshot().status).toBe('disconnected');
    expect(owner.api).toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('connect deadline aborts the underlying bootstrap request', async () => {
    vi.useFakeTimers();
    let signal: AbortSignal | null | undefined;

    const owner = session(async (_input, init) => {
      signal = init?.signal;

      return new Promise<Response>(() => undefined);
    });

    const pending = owner.bootstrap(`${origin}/?bootstrap=secret`, () => undefined);
    await vi.advanceTimersByTimeAsync(1001);
    expect(await pending).toBe(false);
    expect(signal?.aborted).toBe(true);
    expect(owner.snapshot().status).toBe('error');
  });
  it('redacts generic transport failure details and enforces operation deadlines', async () => {
    const owner = session(async () => {
      throw new Error('private URL/token leaked by transport');
    });

    expect(await owner.connect('credential')).toBe(false);
    expect(owner.snapshot().message).not.toContain('private');
    const live = session(async () => json(health));
    await live.connect('credential');
    vi.useFakeTimers();
    const work = live.perform(() => new Promise<string>(() => undefined), { timeoutMs: 5 });
    const rejection = expect(work).rejects.toMatchObject({ code: 'request' });
    await vi.advanceTimersByTimeAsync(6);
    await rejection;
    expect(live.snapshot().status).toBe('error');
  });
  it('external abort does not revoke the verified connection', async () => {
    const owner = session(async () => json(health));
    await owner.connect('credential');
    const external = new AbortController();

    const work = owner.perform(() => new Promise<void>(() => undefined), {
      signal: external.signal,
      timeoutMs: 0,
    });

    const rejection = expect(work).rejects.toMatchObject({ code: 'stale' });
    external.abort();
    await rejection;
    expect(owner.snapshot().status).toBe('connected');
  });
  it('refresh cannot resurrect a detached session between async continuations', async () => {
    const owner = session(async () => json(health));
    await owner.connect('credential');
    vi.spyOn(owner, 'perform').mockImplementation(async () => {
      queueMicrotask(() => owner.detach());

      return health;
    });
    expect(await owner.refresh()).toBe(false);
    expect(owner.snapshot().status).toBe('disconnected');
  });
  it('raw stream failure reports revoke work while a normal HTTP rejection does not', async () => {
    const owner = session(async () => json(health));
    await owner.connect('credential');
    owner.reportFailure(new DaemonApiError('http', 'invalid request', 422));
    expect(owner.snapshot().status).toBe('connected');
    owner.reportFailure(new DaemonApiError('protocol', 'mismatch'));
    expect(owner.api).toBeNull();
    expect(owner.snapshot().status).toBe('skew');
  });
  it('captures boot identity options rather than allowing external retargeting', async () => {
    const options = {
      origin,
      buildId: '1.8.1',
      fetcher: async () => json({ ...health, build_id: '2.0.0' }),
    };

    const owner = new ConnectionSession(options);
    options.buildId = '2.0.0';
    expect(await owner.connect('credential')).toBe(false);
    expect(owner.snapshot().status).toBe('skew');
  });
  it('a retired monitor cannot revoke a newer verified connection', async () => {
    const owner = session(async () => json(health));
    await owner.connect('first-token');
    const oldRevision = owner.snapshot().revision;
    owner.detach();
    await owner.connect('second-token');
    owner.reportFailure(
      new DaemonApiError('authentication', 'retired stream rejected'),
      oldRevision,
    );
    expect(owner.snapshot().status).toBe('connected');
    expect(owner.api?.authenticated).toBe(true);
    owner.reportFailure(
      new DaemonApiError('authentication', 'current stream rejected'),
      owner.snapshot().revision,
    );
    expect(owner.snapshot().status).toBe('authentication');
    expect(owner.api).toBeNull();
  });
  it('synchronous detach on a connecting notification prevents any request', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => json(health));
    const owner = session(fetcher);

    const unsubscribe = owner.subscribe((state) => {
      if (state.status === 'connecting') owner.detach();
    });

    expect(await owner.connect('credential')).toBe(false);
    expect(fetcher).not.toHaveBeenCalled();
    unsubscribe();
  });
});
