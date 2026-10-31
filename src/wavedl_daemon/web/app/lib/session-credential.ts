import { clientError } from '../i18n/client-messages';
import { parseJson, type Json } from './daemon-api';
import { credentialSchema, type SessionCredential } from '../../shared/schemas/session-credential';

export const SESSION_CREDENTIAL_KEY = 'wavedl.session.v1';
export const SESSION_CREDENTIAL_LIMIT = 16 * 1024;

export type { SessionCredential } from '../../shared/schemas/session-credential';

export type SessionCredentialStorageArea = 'session' | 'local';

export interface SessionCredentialStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface StoredSessionCredential {
  credential: SessionCredential;
  storage: SessionCredentialStorageArea;
}

function validOrigin(value: string): boolean {
  try {
    const url = new URL(value);

    return (
      /^https?:$/.test(url.protocol) &&
      !url.username &&
      !url.password &&
      url.origin === value &&
      url.pathname === '/' &&
      !url.search &&
      !url.hash
    );
  } catch {
    return false;
  }
}

function parseCredential(value: Json | SessionCredential): SessionCredential | null {
  try {
    const credential = credentialSchema.parse(value);

    if (!validOrigin(credential.origin)) return null;

    return credential;
  } catch {
    return null;
  }
}

function browserStorage(area: SessionCredentialStorageArea): SessionCredentialStorage | null {
  try {
    if (typeof window === 'undefined') return null;

    return area === 'local' ? window.localStorage : window.sessionStorage;
  } catch {
    return null;
  }
}

function boundedText(value: string): string {
  if (new TextEncoder().encode(value).byteLength > SESSION_CREDENTIAL_LIMIT)
    throw clientError('sessionCredentialTooLarge', {});

  return value;
}

export function parseStoredSessionCredential(raw: string | null): SessionCredential | null {
  try {
    if (raw === null) return null;
    const bounded = boundedText(raw);

    return parseCredential(parseJson(bounded));
  } catch {
    return null;
  }
}

export function loadSessionCredential(
  storages: {
    session?: SessionCredentialStorage | null;
    local?: SessionCredentialStorage | null;
  } = {},
  expectedOrigin?: string,
): StoredSessionCredential | null {
  try {
    const session = storages.session === undefined ? browserStorage('session') : storages.session;
    const local = storages.local === undefined ? browserStorage('local') : storages.local;
    const sessionRaw = session?.getItem(SESSION_CREDENTIAL_KEY) ?? null;
    const sessionCredential = parseStoredSessionCredential(sessionRaw);

    if (sessionCredential && (!expectedOrigin || sessionCredential.origin === expectedOrigin)) {
      return { credential: sessionCredential, storage: 'session' };
    }

    if (sessionRaw !== null) session?.removeItem(SESSION_CREDENTIAL_KEY);

    const localRaw = local?.getItem(SESSION_CREDENTIAL_KEY) ?? null;
    const localCredential = parseStoredSessionCredential(localRaw);

    if (localCredential && (!expectedOrigin || localCredential.origin === expectedOrigin))
      return { credential: localCredential, storage: 'local' };

    if (localRaw !== null) local?.removeItem(SESSION_CREDENTIAL_KEY);

    return null;
  } catch {
    return null;
  }
}

export function saveSessionCredential(
  value: SessionCredential,
  storageArea: SessionCredentialStorageArea = 'session',
  storages: {
    session?: SessionCredentialStorage | null;
    local?: SessionCredentialStorage | null;
  } = {},
): boolean {
  try {
    const credential = parseCredential(value);

    if (!credential) return false;
    const serialized = boundedText(JSON.stringify(credential));
    const session = storages.session === undefined ? browserStorage('session') : storages.session;
    const local = storages.local === undefined ? browserStorage('local') : storages.local;
    const target = storageArea === 'local' ? local : session;

    if (!target) return false;
    target.setItem(SESSION_CREDENTIAL_KEY, serialized);

    return true;
  } catch {
    return false;
  }
}

function parseCredentialForMatch(raw: string | null): SessionCredential | null {
  try {
    return raw === null ? null : parseCredential(parseJson(boundedText(raw)));
  } catch {
    return null;
  }
}

/** Remove only envelopes that still contain the rejected session token. */
export function removeMatchingSessionCredential(
  token: string,
  storages: {
    session?: SessionCredentialStorage | null;
    local?: SessionCredentialStorage | null;
  } = {},
): void {
  for (const area of ['session', 'local'] as const) {
    removeMatchingSessionCredentialInArea(token, area, storages);
  }
}

export function removeMatchingSessionCredentialInArea(
  token: string,
  area: SessionCredentialStorageArea,
  storages: {
    session?: SessionCredentialStorage | null;
    local?: SessionCredentialStorage | null;
  } = {},
): void {
  try {
    const storage = storages[area] === undefined ? browserStorage(area) : storages[area];

    if (!storage) return;
    const raw = storage.getItem(SESSION_CREDENTIAL_KEY);

    if (parseCredentialForMatch(raw)?.token === token) storage.removeItem(SESSION_CREDENTIAL_KEY);
  } catch {
    // Storage may be disabled; memory authority is still retired by the caller.
  }
}

/** Remove one persistence area, used by the explicit opt-out control. */
export function clearSessionCredentialArea(
  area: SessionCredentialStorageArea,
  storages: {
    session?: SessionCredentialStorage | null;
    local?: SessionCredentialStorage | null;
  } = {},
): void {
  try {
    const storage = storages[area] === undefined ? browserStorage(area) : storages[area];
    storage?.removeItem(SESSION_CREDENTIAL_KEY);
  } catch {
    // Storage may be disabled.
  }
}

export function clearSessionCredential(
  storages: {
    session?: SessionCredentialStorage | null;
    local?: SessionCredentialStorage | null;
  } = {},
): void {
  for (const area of ['session', 'local'] as const) {
    try {
      const storage = storages[area] === undefined ? browserStorage(area) : storages[area];
      storage?.removeItem(SESSION_CREDENTIAL_KEY);
    } catch {
      // Storage may be disabled or full; authority is also cleared from memory.
    }
  }
}
