// @vitest-environment happy-dom
import type { FixtureResponse } from './fixture-schemas';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConnectionSession } from '../../app/lib/connection-session';
import { translateClientDiagnostic } from '../../app/i18n/client-messages';
import { credentialSchema } from '../../shared/schemas/session-credential';
import {
  loadSessionCredential,
  SESSION_CREDENTIAL_KEY,
  type SessionCredential,
} from '../../app/lib/session-credential';

const origin = window.location.origin;
const health = { daemon_id: 'daemon-one', build_id: 'build-a', protocol_min: 1, protocol_max: 1 };

const grant = (token: string, expiresAt = Date.now() + 12 * 60 * 60 * 1000) => ({
  access_token: token,
  token_type: 'Bearer',
  expires_at: expiresAt,
  daemon_id: health.daemon_id,
});

const credential: SessionCredential = {
  version: 1,
  daemonId: health.daemon_id,
  origin,
  token: `session-${'a'.repeat(35)}`,
  expiresAt: Date.now() + 12 * 60 * 60 * 1000,
};

const localValues = new Map<string, string>();

const localStorageMock = {
  getItem: (key: string) => localValues.get(key) ?? null,
  setItem: (key: string, value: string) => localValues.set(key, value),
  removeItem: (key: string) => localValues.delete(key),
  clear: () => localValues.clear(),
};

Object.defineProperty(window, 'localStorage', { configurable: true, value: localStorageMock });

const json = (body: FixtureResponse, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

function session(fetcher: typeof fetch) {
  return new ConnectionSession({ origin, buildId: 'build-a', fetcher, timeoutMs: 1000 });
}

beforeEach(() => {
  window.sessionStorage.clear();
  localStorageMock.clear();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  window.sessionStorage.clear();
  localStorageMock.clear();
});

describe('resumable browser session', () => {
  it('exchanges the master before persisting and stores only a session token', async () => {
    const master = 'master-only-secret';
    const sessionToken = `session-${'b'.repeat(35)}`;
    const calls: Array<{ path: string; authorization: string | null }> = [];

    const owner = session(async (input, init) => {
      const url = new URL(String(input));
      calls.push({
        path: url.pathname,
        authorization: new Headers(init?.headers).get('Authorization'),
      });

      if (url.pathname === '/api/health') return json(health);

      if (url.pathname === '/api/session') return json(grant(sessionToken));
      throw new Error('Unexpected request');
    });

    expect(await owner.connect(master)).toBe(true);
    expect(calls.map(({ path }) => path)).toEqual(['/api/health', '/api/session']);
    expect(calls.every(({ authorization }) => authorization === `Bearer ${master}`)).toBe(true);

    const persisted = credentialSchema.parse(JSON.parse(
      window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY) ?? 'null',
    ));

    expect(persisted.token).toBe(sessionToken);
    expect(JSON.stringify(persisted)).not.toContain(master);
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();
    owner.dispose();
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).not.toBeNull();
  });

  it('moves the current envelope when keep-signed-in changes', async () => {
    const sessionToken = `session-${'l'.repeat(35)}`;

    const owner = session(async (input) => {
      const url = new URL(String(input));

      return url.pathname === '/api/health' ? json(health) : json(grant(sessionToken));
    });

    expect(await owner.connect('master-only-secret')).toBe(true);
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).not.toBeNull();
    expect(owner.setKeepSignedIn(true)).toBe(true);
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();
    expect(credentialSchema.parse(JSON.parse(window.localStorage.getItem(SESSION_CREDENTIAL_KEY) ?? 'null')).token).toBe(
      sessionToken,
    );
    expect(owner.setKeepSignedIn(false)).toBe(true);
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();
    expect(credentialSchema.parse(JSON.parse(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY) ?? 'null')).token).toBe(
      sessionToken,
    );
    owner.dispose();
  });

  it('replaces this tab’s stale session copy when bootstrap saves to local storage', async () => {
    const stale = { ...credential, token: `session-${'p'.repeat(35)}` };
    const fresh = `session-${'q'.repeat(35)}`;
    window.sessionStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(stale));

    const owner = session(async (input) => {
      const url = new URL(String(input));

      if (url.pathname === '/api/bootstrap') return json(grant(fresh));

      if (url.pathname === '/api/health') return json(health);
      throw new Error('Unexpected request');
    });

    expect(await owner.bootstrapCode('bootstrap-code', false, true)).toBe(true);
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();
    expect(credentialSchema.parse(JSON.parse(window.localStorage.getItem(SESSION_CREDENTIAL_KEY) ?? 'null')).token).toBe(
      fresh,
    );
    expect(loadSessionCredential({}, origin)?.credential.token).toBe(fresh);
    owner.dispose();
  });

  it('resumes with health only and preserves the stored token and grant expiry', async () => {
    window.sessionStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(credential));
    window.localStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(credential));

    const calls: Array<{ path: string; authorization: string | null; daemonId: string | null }> =
      [];

    const owner = session(async (input, init) => {
      const url = new URL(String(input));
      const headers = new Headers(init?.headers);
      calls.push({
        path: url.pathname,
        authorization: headers.get('Authorization'),
        daemonId: headers.get('X-WaveDL-Daemon-ID'),
      });

      if (url.pathname === '/api/health') return json(health);
      throw new Error('Unexpected request');
    });

    expect(await owner.resume(credential)).toBe(true);
    expect(calls.map(({ path }) => path)).toEqual(['/api/health']);
    expect(calls.every(({ authorization }) => authorization === `Bearer ${credential.token}`)).toBe(
      true,
    );
    expect(calls.every(({ daemonId }) => daemonId === credential.daemonId)).toBe(true);
    expect(JSON.parse(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY) ?? 'null')).toEqual(
      credential,
    );
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(credential));
    owner.dispose();
  });

  it('lets a duplicated tab resume without invalidating the original tab', async () => {
    window.sessionStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(credential));
    const requests: string[] = [];

    const fetcher: typeof fetch = async (input) => {
      const path = new URL(String(input)).pathname;
      requests.push(path);

      if (path === '/api/health') return json(health);

      if (path === '/api/models') return json({ models: [] });
      throw new Error('Unexpected request');
    };

    const original = session(fetcher);
    const duplicate = session(fetcher);
    expect(await original.resume(credential)).toBe(true);
    expect(await duplicate.resume(credential)).toBe(true);
    await expect(
      original.perform((api, signal) => api.read('/api/models', signal)),
    ).resolves.toEqual({
      models: [],
    });
    expect(requests).not.toContain('/api/session');
    expect(original.sessionCredential?.token).toBe(credential.token);
    expect(duplicate.sessionCredential?.token).toBe(credential.token);
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(credential));
    original.dispose();
    duplicate.dispose();
  });

  it('uses the daemon response to retire a rejected session token', async () => {
    window.sessionStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(credential));
    const expired = session(async () => json({ detail: 'ignored' }, 401));
    expect(await expired.resume(credential)).toBe(false);
    expect(expired.snapshot().status).toBe('authentication');
    const expiredMessage = expired.snapshot().message;

    if (expiredMessage === null) throw new Error('Expected a coded authentication notice');
    expect(translateClientDiagnostic('en', expiredMessage)).toBe(
      'Your session ended. Sign in again.',
    );
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();

    const master = 'master-only-secret';

    const active = session(async (input) => {
      const url = new URL(String(input));

      if (url.pathname === '/api/health') return json(health);

      if (url.pathname === '/api/session') return json(grant(`session-${'d'.repeat(35)}`));

      return json({}, 401);
    });

    expect(await active.connect(master)).toBe(true);
    await expect(
      active.perform((api, signal) => api.read('/api/private', signal)),
    ).rejects.toMatchObject({ code: 'authentication' });
    expect(active.snapshot().status).toBe('authentication');
    const activeMessage = active.snapshot().message;

    if (activeMessage === null) throw new Error('Expected a coded authentication notice');
    expect(translateClientDiagnostic('en', activeMessage)).toBe(
      'Your session ended. Sign in again.',
    );
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();
  });

  it('treats an identity 409 separately from 401 and clears the envelope', async () => {
    window.localStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(credential));
    const owner = session(async () => json({ detail: 'ignored' }, 409));
    expect(await owner.resume(credential, true)).toBe(false);
    expect(owner.snapshot().status).toBe('skew');
    const message = owner.snapshot().message;

    if (message === null) throw new Error('Expected a coded identity notice');
    expect(translateClientDiagnostic('en', message)).toBe(
      'WaveDL was restarted or replaced. Sign in again.',
    );
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();
  });

  it('retains the envelope after transient resume failures and does not trust local expiry', async () => {
    const staleExpiry = { ...credential, expiresAt: 1 };
    window.sessionStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(staleExpiry));
    const loaded = loadSessionCredential({}, origin);
    expect(loaded?.credential).toEqual(staleExpiry);
    let requests = 0;

    const owner = session(async () => {
      requests += 1;

      return json({ detail: 'temporarily unavailable' }, 503);
    });

    expect(await owner.resume(staleExpiry)).toBe(false);
    expect(requests).toBe(1);
    expect(owner.snapshot().status).toBe('error');
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(staleExpiry));
  });

  it('retains a resumable envelope through network and offline retry states', async () => {
    window.sessionStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(credential));

    const networkFailure = session(async () => {
      throw new TypeError('offline transport');
    });

    expect(await networkFailure.resume(credential)).toBe(false);
    expect(networkFailure.snapshot().status).toBe('error');
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(credential));

    const offline = new ConnectionSession({
      origin,
      buildId: 'build-a',
      online: false,
      fetcher: async () => {
        throw new Error('Offline resume must not fetch');
      },
    });

    expect(await offline.resume(credential)).toBe(false);
    expect(offline.snapshot().status).toBe('offline');
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(credential));
  });

  it('retains the envelope when resume times out', async () => {
    vi.useFakeTimers();
    window.sessionStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(credential));

    const owner = new ConnectionSession({
      origin,
      buildId: 'build-a',
      timeoutMs: 1000,
      fetcher: async () => new Promise<Response>(() => undefined),
    });

    const pending = owner.resume(credential);
    await vi.advanceTimersByTimeAsync(1001);
    expect(await pending).toBe(false);
    expect(owner.snapshot().status).toBe('error');
    expect(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(credential));
    owner.dispose();
  });

  it('does not delete a newer shared local token when a stale tab receives 401', async () => {
    const oldCredential = { ...credential, token: `session-${'j'.repeat(35)}` };
    const newCredential = { ...credential, token: `session-${'k'.repeat(35)}` };
    window.localStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(oldCredential));

    const owner = session(async (input) => {
      if (new URL(String(input)).pathname === '/api/health') return json(health);

      if (new URL(String(input)).pathname === '/api/session')
        return json(grant(oldCredential.token));

      return json({}, 401);
    });

    expect(await owner.connect('master-old', true)).toBe(true);

    const otherTab = session(async (input) =>
      json(
        new URL(String(input)).pathname === '/api/health'
          ? health
          : { ...grant(newCredential.token), expires_at: newCredential.expiresAt },
      ),
    );

    expect(await otherTab.connect('master-new', true)).toBe(true);
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(newCredential));
    await expect(
      owner.perform((api, signal) => api.read('/api/private', signal)),
    ).rejects.toMatchObject({ code: 'authentication' });
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(newCredential));
    expect(otherTab.snapshot().status).toBe('connected');
    otherTab.dispose();
  });

  it('does not delete a newer shared local token when an old identity gets 409', async () => {
    const oldCredential = { ...credential, token: `session-${'n'.repeat(35)}` };
    const newCredential = { ...credential, token: `session-${'o'.repeat(35)}` };
    window.localStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(oldCredential));

    const owner = session(async () => {
      window.localStorage.setItem(SESSION_CREDENTIAL_KEY, JSON.stringify(newCredential));

      return json({ detail: 'ignored' }, 409);
    });

    expect(await owner.resume(oldCredential, true)).toBe(false);
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBe(JSON.stringify(newCredential));
  });

  it('retires authority immediately on disconnect and revokes only the captured token', async () => {
    const master = 'master-only-secret';
    const sessionToken = `session-${'e'.repeat(35)}`;
    const calls: Array<{ path: string; method: string; authorization: string | null }> = [];

    const owner = session(async (input, init) => {
      const url = new URL(String(input));
      const method = init?.method ?? 'GET';
      const authorization = new Headers(init?.headers).get('Authorization');
      calls.push({ path: url.pathname, method, authorization });

      if (url.pathname === '/api/health') return json(health);

      if (url.pathname === '/api/session' && method === 'DELETE')
        return new Response(null, { status: 204 });

      if (url.pathname === '/api/session') return json(grant(sessionToken));
      throw new Error('Unexpected request');
    });

    expect(await owner.connect(master, true)).toBe(true);
    await owner.disconnect();
    expect(owner.snapshot().status).toBe('disconnected');
    expect(window.localStorage.getItem(SESSION_CREDENTIAL_KEY)).toBeNull();
    expect(calls.filter(({ method }) => method === 'DELETE')).toEqual([
      { path: '/api/session', method: 'DELETE', authorization: `Bearer ${sessionToken}` },
    ]);
  });

  it('does not schedule automatic token rotation after a successful sign-in', async () => {
    vi.useFakeTimers();
    const first = `session-${'f'.repeat(35)}`;
    const second = `session-${'g'.repeat(35)}`;
    let rotations = 0;

    const owner = session(async (input) => {
      const url = new URL(String(input));

      if (url.pathname === '/api/health') return json(health);
      rotations += 1;

      return json(grant(rotations === 1 ? first : second, Date.now() + 10_000));
    });

    expect(await owner.connect('master-only-secret')).toBe(true);
    await vi.advanceTimersByTimeAsync(24 * 60 * 60 * 1000);
    expect(rotations).toBe(1);
    expect(
      JSON.parse(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY) ?? 'null'),
    ).toMatchObject({ token: first, daemonId: health.daemon_id, expiresAt: expect.any(Number) });
    expect(
      credentialSchema.parse(JSON.parse(window.sessionStorage.getItem(SESSION_CREDENTIAL_KEY) ?? 'null')).token,
    ).not.toBe(second);
    owner.dispose();
  });

  it('does not let a late disconnect response retire a newer connection', async () => {
    const oldToken = `session-${'h'.repeat(35)}`;
    const newToken = `session-${'i'.repeat(35)}`;
    let delayedDelete: ((response: Response) => void) | undefined;
    let masterNumber = 0;

    const owner = session(async (input, init) => {
      const url = new URL(String(input));

      if (url.pathname === '/api/health') return json(health);

      if (init?.method === 'DELETE')
        return new Promise<Response>((resolve) => {
          delayedDelete = resolve;
        });
      masterNumber += 1;

      return json(grant(masterNumber === 1 ? oldToken : newToken));
    });

    expect(await owner.connect('master-one')).toBe(true);
    const disconnecting = owner.disconnect();
    expect(owner.snapshot().status).toBe('disconnected');
    expect(await owner.connect('master-two')).toBe(true);
    delayedDelete?.(new Response(null, { status: 204 }));
    await disconnecting;
    expect(owner.snapshot().status).toBe('connected');
    expect(owner.sessionCredential?.token).toBe(newToken);
    owner.dispose();
  });

  it('continues in memory and removes same-token stale storage when persistence fails', async () => {
    const sessionToken = `session-${'m'.repeat(35)}`;
    const storageDescriptor = Object.getOwnPropertyDescriptor(window, 'sessionStorage');

    const values = new Map<string, string>([
      [SESSION_CREDENTIAL_KEY, JSON.stringify({ ...credential, token: sessionToken })],
    ]);

    let failedWrites = 0;

    const failingStorage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: () => {
        failedWrites += 1;
        throw new Error('storage unavailable');
      },
      removeItem: (key: string) => values.delete(key),
      clear: () => values.clear(),
    };

    Object.defineProperty(window, 'sessionStorage', {
      configurable: true,
      value: failingStorage,
    });

    try {
      const owner = session(async (input) => {
        const url = new URL(String(input));

        return url.pathname === '/api/health' ? json(health) : json(grant(sessionToken));
      });

      expect(await owner.connect('master-only-secret')).toBe(true);
      expect(owner.api).not.toBeNull();
      expect(failedWrites).toBe(1);
      expect(values.get(SESSION_CREDENTIAL_KEY)).toBeUndefined();
      owner.dispose();
    } finally {
      if (storageDescriptor) Object.defineProperty(window, 'sessionStorage', storageDescriptor);
    }
  });
});
