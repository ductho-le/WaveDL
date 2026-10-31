import {
  clientError,
  type ClientErrorKey,
  type ClientErrorParams,
  type ClientMessage,
} from '../i18n/client-messages';
import {
  DaemonApi,
  DaemonApiError,
  stripBootstrapUrl,
  type Health,
  type SessionGrant,
} from './daemon-api';
import {
  clearSessionCredentialArea,
  removeMatchingSessionCredential,
  saveSessionCredential,
  type SessionCredential,
  type SessionCredentialStorageArea,
} from './session-credential';

export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'authentication'
  | 'skew'
  | 'offline'
  | 'error';

export interface ConnectionState {
  status: ConnectionStatus;
  health: Health | null;
  message: ClientMessage | Error | null;
  revision: number;
  buildMismatch: boolean;
}

export interface ConnectionOptions {
  origin: string;
  buildId: string;
  fetcher?: typeof fetch;
  timeoutMs?: number;
  online?: boolean;
}

export interface WorkOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export class ConnectionSessionError extends Error {
  constructor(
    public readonly code:
      | 'unavailable'
      | 'stale'
      | 'authentication'
      | 'identity'
      | 'skew'
      | 'response'
      | 'request',
    message: string,
    public readonly status?: number,
    public readonly clientMessageKey?: ClientErrorKey,
    public readonly clientMessageParams?: ClientErrorParams,
  ) {
    super(message);
    this.name = 'ConnectionSessionError';
  }

  static fromKey(
    code: ConnectionSessionError['code'],
    key: ClientErrorKey,
    options: { status?: number; params?: ClientErrorParams } = {},
  ): ConnectionSessionError {
    const { status, params } = options;

    return new ConnectionSessionError(code, clientError(key, params).message, status, key, params);
  }
}

/** One browser-memory authority scope. Detaching never cancels server work. */
export class ConnectionSession {
  #state: ConnectionState = Object.freeze({
    status: 'disconnected',
    health: null,
    message: null,
    revision: 0,
    buildMismatch: false,
  });
  #listeners = new Set<(state: ConnectionState) => void>();
  #api: DaemonApi | null = null;
  #scope: AbortController | null = null;
  #daemonId: string | undefined;
  #grant: SessionGrant | null = null;
  #storageArea: SessionCredentialStorageArea = 'session';
  #online: boolean;
  readonly #fetcher: typeof fetch;
  readonly #timeoutMs: number;
  readonly #origin: string;
  readonly #buildId: string;

  constructor(options: ConnectionOptions) {
    this.#origin = options.origin;
    this.#buildId = options.buildId;
    this.#fetcher = options.fetcher ?? globalThis.fetch.bind(globalThis);
    this.#timeoutMs = options.timeoutMs ?? 15_000;

    if (!Number.isSafeInteger(this.#timeoutMs) || this.#timeoutMs < 1)
      throw clientError('invalidConnectionDeadline', {});
    this.#online = options.online ?? true;

    if (!this.#online) this.#state = Object.freeze({ ...this.#state, status: 'offline' });
  }
  get api(): DaemonApi | null {
    return this.#state.status === 'connected' ? this.#api : null;
  }
  snapshot(): ConnectionState {
    return this.#state;
  }
  subscribe(listener: (state: ConnectionState) => void): () => void {
    this.#listeners.add(listener);
    listener(this.#state);

    return () => this.#listeners.delete(listener);
  }
  #publish(
    status: ConnectionStatus,
    message: ClientMessage | Error | null = null,
    health: Health | null = null,
    buildMismatch = false,
  ): void {
    this.#state = Object.freeze({
      status,
      message,
      health,
      revision: this.#state.revision,
      buildMismatch,
    });

    for (const listener of this.#listeners) listener(this.#state);
  }
  #revoke(): void {
    this.#state = Object.freeze({ ...this.#state, revision: this.#state.revision + 1 });
    this.#scope?.abort();
    this.#scope = null;
    this.#api?.clearToken();
    this.#api = null;
    this.#grant = null;
  }
  detach(): void {
    const token = this.#grant?.access_token;

    if (token) removeMatchingSessionCredential(token);
    this.#revoke();
    this.#daemonId = undefined;
    this.#publish(this.#online ? 'disconnected' : 'offline');
  }
  /** Release only in-memory authority when the UI owner is disposed or reloaded. */
  dispose(): void {
    this.#revoke();
  }
  setKeepSignedIn(keepSignedIn: boolean): boolean {
    const credential = this.sessionCredential;
    const destination: SessionCredentialStorageArea = keepSignedIn ? 'local' : 'session';

    if (!keepSignedIn) clearSessionCredentialArea('local');

    if (!credential) {
      this.#storageArea = destination;

      return false;
    }

    if (!saveSessionCredential(credential, destination)) {
      removeMatchingSessionCredential(credential.token);

      return false;
    }

    if (destination === 'local' && this.#storageArea === 'session')
      clearSessionCredentialArea('session');
    this.#storageArea = destination;

    return true;
  }
  setOnline(online: boolean): void {
    if (online === this.#online) return;
    this.#online = online;
    this.#revoke();
    this.#publish(
      online ? 'disconnected' : 'offline',
      online ? clientError('reconnectRequired') : clientError('connectionOffline'),
    );
  }
  async connect(token: string, keepSignedIn = false): Promise<boolean> {
    return this.#connect(
      async (api, signal) => {
        api.setToken(token);
        const health = await api.health(signal);
        const grant = await api.rotateSession(signal);

        if (grant.access_token === token) {
          throw DaemonApiError.fromKey('response', 'masterCredentialReturned');
        }

        api.setRotatedToken(grant.access_token);
        this.#assertGrantIdentity(grant, health);

        return { health, grant };
      },
      keepSignedIn,
      token,
    );
  }
  async reconnect(token: string, keepSignedIn = false): Promise<boolean> {
    return this.connect(token, keepSignedIn);
  }
  get sessionCredential(): SessionCredential | null {
    if (!this.#grant || !this.#daemonId) return null;

    return {
      version: 1,
      daemonId: this.#daemonId,
      origin: this.#origin,
      token: this.#grant.access_token,
      expiresAt: this.#grant.expires_at,
    };
  }
  async resume(credential: SessionCredential, keepSignedIn = false): Promise<boolean> {
    if (credential.origin !== this.#origin) {
      this.#fail(DaemonApiError.fromKey('identity', 'daemonIdentityChanged'));

      return false;
    }

    this.#daemonId = credential.daemonId;

    return this.#connect(
      async (api, signal) => {
        this.#daemonId = credential.daemonId;
        api.setToken(credential.token);
        const health = await api.health(signal);

        const grant: SessionGrant = {
          access_token: credential.token,
          token_type: 'Bearer',
          expires_at: credential.expiresAt,
          daemon_id: health.daemon_id,
        };

        return { health, grant };
      },
      keepSignedIn,
      credential.token,
      false,
    );
  }
  /** Raw stream consumers report fatal transport/contract errors to the owner. */
  reportFailure(error?: Error, expectedRevision?: number): void {
    if (expectedRevision !== undefined && expectedRevision !== this.#state.revision) return;

    if (error instanceof DaemonApiError && error.code === 'http') return;

    if (error instanceof ConnectionSessionError && error.code === 'stale') return;
    const rejectedToken = this.#grant?.access_token;
    this.#fail(error, rejectedToken);
  }
  async bootstrap(input: string, replaceUrl: (url: string) => void): Promise<boolean> {
    // Strip credentials even when offline, before starting the connection work.
    try {
      replaceUrl(stripBootstrapUrl(input).url);
    } catch {
      this.#fail();

      return false;
    }

    return this.#connect(async (api, signal) => {
      const grant = await api.bootstrap(input, () => undefined);

      if (!grant) return null;
      const health = await api.health(signal);

      return { health, grant };
    }, false);
  }
  async bootstrapCode(
    code: string | null,
    invalid = false,
    keepSignedIn = false,
  ): Promise<boolean> {
    if (invalid || !code) {
      this.#fail(DaemonApiError.fromKey('authentication', 'credentialRequired'));

      return false;
    }

    return this.#connect(async (api, signal) => {
      const grant = await api.exchangeBootstrapCode(code, signal);
      const health = await api.health(signal);
      this.#assertGrantIdentity(grant, health);

      return { health, grant };
    }, keepSignedIn);
  }
  async disconnect(): Promise<void> {
    const grant = this.#grant;
    const daemonId = this.#daemonId;

    const revoker =
      grant && daemonId
        ? new DaemonApi(this.#origin, { daemonId, buildId: this.#buildId }, this.#fetcher)
        : null;

    revoker?.setToken(grant?.access_token ?? '');
    this.detach();

    if (!revoker) return;
    const standalone = new AbortController();
    await this.#bounded(
      (signal) => revoker.revokeSession(signal),
      standalone.signal,
      Math.min(this.#timeoutMs, 5000),
    )
      .catch(() => undefined)
      .finally(() => revoker.clearToken());
  }
  async refresh(): Promise<boolean> {
    const revision = this.#state.revision;

    try {
      const health = await this.perform((api, signal) => api.health(signal));

      if (revision !== this.#state.revision || !this.api) return false;
      this.#publish('connected', null, Object.freeze(health));

      return true;
    } catch {
      return false;
    }
  }
  async #connect(
    handshake: (
      api: DaemonApi,
      signal: AbortSignal,
    ) => Promise<{ health: Health; grant: SessionGrant } | null>,
    keepSignedIn = false,
    retirementToken?: string,
    persistCredential = true,
  ): Promise<boolean> {
    this.#revoke();

    if (!this.#online) {
      this.#publish('offline', clientError('connectionOffline'));

      return false;
    }

    const revision = this.#state.revision;
    const scope = new AbortController();
    this.#scope = scope;
    this.#publish('connecting');

    if (revision !== this.#state.revision || scope.signal.aborted) return false;
    let candidate: DaemonApi | null = null;
    let presentedToken: string | undefined;

    try {
      let handshakeSignal = scope.signal;

      const scopedFetch: typeof fetch = (input, init) => {
        const authorization = new Headers(init?.headers).get('Authorization');
        presentedToken = authorization?.startsWith('Bearer ')
          ? authorization.slice('Bearer '.length)
          : undefined;

        return this.#fetcher(input, {
          ...init,
          signal: AbortSignal.any([
            scope.signal,
            handshakeSignal,
            ...(init?.signal ? [init.signal] : []),
          ]),
        });
      };

      const api = new DaemonApi(
        this.#origin,
        { daemonId: this.#daemonId, buildId: this.#buildId },
        scopedFetch,
      );

      candidate = api;

      const result = await this.#bounded(
        (signal) => {
          handshakeSignal = signal;

          return handshake(api, signal).finally(() => {
            // An injected/slow transport may finish even after abort. Do not
            // let a late bootstrap retain a credential on its retired client.
            if (signal.aborted || revision !== this.#state.revision) api.clearToken();
          });
        },
        scope.signal,
        this.#timeoutMs,
      );

      if (revision !== this.#state.revision || scope.signal.aborted) {
        candidate.clearToken();

        return false;
      }

      if (!result) {
        candidate.clearToken();
        this.#scope = null;
        this.#publish('authentication', clientError('credentialRequired'));

        return false;
      }

      this.#assertGrantIdentity(result.grant, result.health);
      this.#daemonId = result.health.daemon_id;
      this.#grant = result.grant;
      this.#storageArea = keepSignedIn ? 'local' : 'session';
      this.#api = candidate;
      const credential = this.sessionCredential;

      if (persistCredential && credential) {
        if (!saveSessionCredential(credential, this.#storageArea))
          removeMatchingSessionCredential(credential.token);
        else if (this.#storageArea === 'local') clearSessionCredentialArea('session');
      }

      this.#publish('connected', null, Object.freeze(result.health));

      return revision === this.#state.revision && this.api === candidate;
    } catch (error) {
      const tokenRejected =
        error instanceof DaemonApiError &&
        (error.status === 401 || (error.status === 409 && error.daemonIdentityChanged));

      const rejectedToken = tokenRejected ? (presentedToken ?? retirementToken) : undefined;
      candidate?.clearToken();

      if (revision === this.#state.revision) {
        this.#fail(error instanceof Error ? error : undefined, rejectedToken);
      }

      return false;
    }
  }
  #assertGrantIdentity(grant: SessionGrant, health: Health): void {
    if (grant.daemon_id !== health.daemon_id)
      throw DaemonApiError.fromKey('identity', 'daemonIdentityChanged', {
        status: 409,
        daemonIdentityChanged: true,
      });
  }
  async perform<T>(
    work: (api: DaemonApi, signal: AbortSignal) => Promise<T>,
    options: WorkOptions = {},
  ): Promise<T> {
    const api = this.api;
    const scope = this.#scope;

    if (!this.#online || !api || !scope)
      throw ConnectionSessionError.fromKey('unavailable', 'connectionRequired');
    const revision = this.#state.revision;
    const signal = options.signal ? AbortSignal.any([scope.signal, options.signal]) : scope.signal;

    try {
      const result = await this.#bounded(
        (requestSignal) => work(api, requestSignal),
        signal,
        options.timeoutMs ?? this.#timeoutMs,
      );

      if (revision !== this.#state.revision || signal.aborted)
        throw ConnectionSessionError.fromKey('stale', 'connectionScopeChanged');

      return result;
    } catch (error) {
      if (revision !== this.#state.revision || signal.aborted)
        throw ConnectionSessionError.fromKey('stale', 'connectionScopeChanged');
      this.reportFailure(error instanceof Error ? error : undefined);
      throw this.#safeError(error instanceof Error ? error : undefined);
    }
  }
  async #bounded<T>(
    work: (signal: AbortSignal) => Promise<T>,
    scope: AbortSignal,
    timeoutMs: number,
  ): Promise<T> {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 0)
      throw ConnectionSessionError.fromKey('request', 'invalidRequestDeadline');
    const controller = new AbortController();
    const signal = AbortSignal.any([scope, controller.signal]);
    let timer: ReturnType<typeof setTimeout> | undefined;
    let rejectAbort: (() => void) | undefined;

    const aborted = new Promise<never>((_resolve, reject) => {
      rejectAbort = () =>
        reject(ConnectionSessionError.fromKey('request', 'requestOutcomeUnknown'));
      signal.addEventListener('abort', rejectAbort, { once: true });

      if (signal.aborted) rejectAbort();
    });

    if (timeoutMs > 0) timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      if (signal.aborted) return await aborted;

      return await Promise.race([work(signal), aborted]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);

      if (rejectAbort) signal.removeEventListener('abort', rejectAbort);
    }
  }
  #safeError(error?: Error): ConnectionSessionError {
    if (error instanceof ConnectionSessionError) return error;

    if (error instanceof DaemonApiError) {
      if (error.code === 'authentication')
        return ConnectionSessionError.fromKey('authentication', 'yourSessionEndedSignInAgain');

      if (error.code === 'identity')
        return error.daemonIdentityChanged
          ? ConnectionSessionError.fromKey('identity', 'daemonReplaced', { status: error.status })
          : error.buildMismatch
            ? // Same daemon, different build: updating the client fixes it, signing in does not.
              ConnectionSessionError.fromKey('skew', 'daemonScopeChanged')
            : ConnectionSessionError.fromKey('identity', 'daemonIdentityUnverified', {
                status: error.status,
              });

      if (error.code === 'protocol')
        return ConnectionSessionError.fromKey('skew', 'daemonScopeChanged');

      if (error.code === 'response')
        return ConnectionSessionError.fromKey('response', 'daemonResponseInvalid');
    }

    return ConnectionSessionError.fromKey('request', 'daemonUnavailable', {
      status:
        error instanceof DaemonApiError &&
        error.code === 'http' &&
        error.status !== undefined &&
        Number.isInteger(error.status) &&
        error.status >= 400 &&
        error.status <= 599
          ? error.status
          : undefined,
    });
  }
  #fail(error?: Error, rejectedToken?: string): void {
    const safe = this.#safeError(error);

    if (
      rejectedToken &&
      error instanceof DaemonApiError &&
      (error.status === 401 || (error.status === 409 && error.daemonIdentityChanged))
    )
      removeMatchingSessionCredential(rejectedToken);
    this.#revoke();
    this.#publish(
      safe.code === 'authentication'
        ? 'authentication'
        : safe.code === 'identity' || safe.code === 'skew'
          ? 'skew'
          : 'error',
      safe,
      null,
      error instanceof DaemonApiError && error.buildMismatch,
    );
  }
}
