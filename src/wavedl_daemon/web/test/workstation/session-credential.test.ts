import { describe, expect, it } from 'vitest';
import {
  clearSessionCredential,
  loadSessionCredential,
  saveSessionCredential,
  SESSION_CREDENTIAL_KEY,
  type SessionCredentialStorage,
} from '../../app/lib/session-credential';

function storage(): SessionCredentialStorage & { values: Map<string, string> } {
  const values = new Map<string, string>();

  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

const credential = {
  version: 1 as const,
  daemonId: 'daemon-one',
  origin: 'http://127.0.0.1:8765',
  token: `session-${'x'.repeat(35)}`,
  expiresAt: 1_800_000_000_000,
};

describe('session credential envelope', () => {
  it('round-trips only the bounded session envelope and defaults to session storage', () => {
    const session = storage();
    const local = storage();
    expect(saveSessionCredential(credential, 'session', { session, local })).toBe(true);
    expect(session.values.has(SESSION_CREDENTIAL_KEY)).toBe(true);
    expect(local.values.size).toBe(0);
    expect(loadSessionCredential({ session, local })).toEqual({
      credential,
      storage: 'session',
    });
  });
  it('writes only the explicitly selected persistence area', () => {
    const session = storage();
    const local = storage();
    session.values.set(SESSION_CREDENTIAL_KEY, 'another stored envelope');
    expect(saveSessionCredential(credential, 'local', { session, local })).toBe(true);
    expect(session.values.get(SESSION_CREDENTIAL_KEY)).toBe('another stored envelope');
    expect(loadSessionCredential({ session, local })?.storage).toBe('local');
  });
  it('keeps server-authoritative expiry and rejects malformed, oversized and foreign-origin envelopes', () => {
    const session = storage();
    const local = storage();
    const expired = { ...credential, expiresAt: 1 };
    session.values.set(SESSION_CREDENTIAL_KEY, JSON.stringify(expired));
    expect(loadSessionCredential({ session, local })?.credential).toEqual(expired);
    session.values.set(SESSION_CREDENTIAL_KEY, JSON.stringify({ ...credential, token: 'master' }));
    expect(loadSessionCredential({ session, local })).toBeNull();
    session.values.set(
      SESSION_CREDENTIAL_KEY,
      JSON.stringify({ ...credential, origin: 'https://other.example' }),
    );
    expect(loadSessionCredential({ session, local }, credential.origin)).toBeNull();
    session.values.set(SESSION_CREDENTIAL_KEY, 'x'.repeat(17 * 1024));
    expect(loadSessionCredential({ session, local })).toBeNull();
  });
  it('clears both browser storage areas', () => {
    const session = storage();
    const local = storage();
    session.values.set(SESSION_CREDENTIAL_KEY, 'secret');
    local.values.set(SESSION_CREDENTIAL_KEY, 'secret');
    clearSessionCredential({ session, local });
    expect(session.values.size).toBe(0);
    expect(local.values.size).toBe(0);
  });
});
