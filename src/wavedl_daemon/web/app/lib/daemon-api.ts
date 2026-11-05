/** Browser-only, same-origin daemon transport. Browser authority is session-scoped. */
import { z } from 'zod';
import { jsonObjectSchema, jsonScalarSchema, jsonContainerSchema } from '../../shared/schemas/json';
import {
  RUN_STATES,
  healthSchema,
  planSchema,
  runListSchema,
  runViewSchema,
  planPreviewSchema,
  sessionGrantSchema,
  isUuid,
} from './daemon-api-schemas';
import { runEvidenceSchema } from '../../shared/schemas/run-evidence';
import { clientError, type ClientErrorKey, type ClientErrorParams } from '../i18n/client-messages';

import type { Json, JsonObject } from '../../shared/schemas/json';

export type { Json, JsonObject } from '../../shared/schemas/json';

export interface Health {
  daemon_id: string;
  build_id: string;
  protocol_min: number;
  protocol_max: number;
}

export interface SessionGrant {
  access_token: string;
  token_type: 'Bearer';
  expires_at: number;
  daemon_id: string;
}

export interface IdentityPins {
  daemonId?: string;
  buildId?: string;
  protocolVersion?: number;
}

export { RUN_STATES };

export type RunState = (typeof RUN_STATES)[number];

export interface RunProgress {
  epoch: number;
  totalEpochs: number;
  fraction: number;
}

export interface RunView {
  plan_id: string;
  run_id: string;
  request_id: string;
  backend: string;
  operation: 'train' | 'test' | 'plot';
  attempt: number | null;
  state: RunState;
  uncertainty: boolean;
  submission_requested: boolean;
  cancellation_requested: boolean;
  evidence: z.output<typeof runEvidenceSchema> | null;
  created_at?: string | null;
  summary?: RunSummary | null;
  progress?: RunProgress | null;
}

/** The approved inputs that name a run; never live state. */
export interface RunSummary {
  name: string | null;
  model: string | null;
  data_path: string | null;
  epochs: number | null;
  cv: number | null;
  source_run_id: string | null;
}

export interface PlanView {
  plan_id: string;
  run_id: string;
  request_id: string;
  preview: z.output<typeof planPreviewSchema>;
}

export type ApiErrorCode = 'authentication' | 'identity' | 'protocol' | 'response' | 'http';

export interface DaemonApiErrorOptions {
  status?: number;
  buildMismatch?: boolean;
  daemonIdentityChanged?: boolean;
  params?: ClientErrorParams;
}

export class DaemonApiError extends Error {
  constructor(
    public readonly code: ApiErrorCode,
    message: string,
    public readonly status?: number,
    public readonly buildMismatch = false,
    public readonly daemonIdentityChanged = false,
    public readonly clientMessageKey?: ClientErrorKey,
    public readonly clientMessageParams?: ClientErrorParams,
  ) {
    super(message);
    this.name = 'DaemonApiError';
  }

  static fromKey(
    code: ApiErrorCode,
    key: ClientErrorKey,
    options: DaemonApiErrorOptions = {},
  ): DaemonApiError {
    const { status, buildMismatch = false, daemonIdentityChanged = false, params } = options;

    return new DaemonApiError(
      code,
      clientError(key, params).message,
      status,
      buildMismatch,
      daemonIdentityChanged,
      key,
      params,
    );
  }
}

export function object(value: unknown): value is JsonObject {
  return jsonObjectSchema.safeParse(value).success;
}

export { isUuid } from './daemon-api-schemas';

function freezeJson(item: Json): void {
  if (item instanceof Object) {
    for (const child of Object.values(item)) freezeJson(child);
    Object.freeze(item);
  }
}

function runEvidence(value: JsonObject): z.output<typeof runEvidenceSchema> {
  const result = runEvidenceSchema.safeParse(value);

  if (!result.success) {
    throw DaemonApiError.fromKey('response', 'invalidRunEvidence', {
      params: { field: String(result.error.issues[0]?.path[0]) },
    });
  }

  return result.data;
}

function runView(value: Json, expectedRunId?: string): RunView {
  const result = runViewSchema.safeParse(value);

  if (!result.success || (expectedRunId !== undefined && result.data.run_id !== expectedRunId))
    throw DaemonApiError.fromKey('response', 'invalidRunResponse');
  const run = result.data;

  return {
    ...run,
    evidence: run.evidence === null ? null : runEvidence(run.evidence),
    progress: run.progress
      ? {
          epoch: run.progress.epoch,
          totalEpochs: run.progress.total_epochs,
          fraction: Math.min(1, run.progress.epoch / run.progress.total_epochs),
        }
      : null,
  };
}

/** Bound arbitrary JSON, while allowing the server's full finite metric domain. */
export function validateJson<T>(
  value: T,
  depth = 0,
  budget = { nodes: 0 },
): asserts value is T & Json {
  if (depth > 32 || ++budget.nodes > 65536)
    throw DaemonApiError.fromKey('response', 'jsonTooComplex');

  if (jsonScalarSchema.safeParse(value).success) return;

  if (Array.isArray(value)) {
    for (const child of value) validateJson(child, depth + 1, budget);

    return;
  }

  const container = jsonContainerSchema.safeParse(value);

  if (container.success) {
    for (const child of Object.values(container.data)) validateJson(child, depth + 1, budget);

    return;
  }

  throw DaemonApiError.fromKey('response', 'invalidJsonValue');
}

export function parseJson(text: string): Json {
  if (new TextEncoder().encode(text).byteLength > 2 * 1024 * 1024) {
    throw DaemonApiError.fromKey('response', 'jsonTooLarge');
  }

  let value: unknown;

  try {
    value = JSON.parse(text);
  } catch {
    throw DaemonApiError.fromKey('response', 'malformedJsonResponse');
  }

  validateJson(value);
  // JSON.parse silently keeps the last duplicate member. Reject ambiguity before
  // trusting identity fields, including escaped spellings of the same key.
  const stack: Array<{ keys: Set<string> | null; key: boolean }> = [];

  for (const match of text.matchAll(
    /"(?:[^"\\]|\\.)*"|[{}[\],:]|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g,
  )) {
    const token = match[0];
    const current = stack.at(-1);

    if (token === '{' || token === '[')
      stack.push({ keys: token === '{' ? new Set() : null, key: token === '{' });
    else if (token === '}' || token === ']') stack.pop();
    else if (token === ',' && current?.keys) current.key = true;
    else if (token === ':' && current) current.key = false;
    else if (token.startsWith('"') && current?.keys && current.key) {
      const key: string = JSON.parse(token);

      if (current.keys.has(key)) throw DaemonApiError.fromKey('response', 'duplicateJsonMember');
      current.keys.add(key);
    }
  }

  return value;
}

const secretNames = new Set([
  'bootstrap',
  'bootstrap_code',
  'token',
  'access_token',
  'authorization',
  'code',
]);

export function stripBootstrapUrl(input: string) {
  const url = new URL(input);
  const codes = url.searchParams.getAll('bootstrap');

  for (const key of [...url.searchParams.keys()])
    if (secretNames.has(key.toLowerCase())) url.searchParams.delete(key);
  // Fragments are never needed by the workstation; they may carry pasted secrets.
  url.hash = '';

  return {
    url: url.href,
    code: codes.length === 1 ? (codes[0] ?? null) : null,
    invalid: codes.length > 1,
  };
}

function sessionGrant(value: Json): SessionGrant | null {
  const result = sessionGrantSchema.safeParse(value);

  return result.success && sessionToken(result.data.access_token) ? result.data : null;
}

function sessionToken(value: string): boolean {
  return /^[A-Za-z0-9_-]{43}$/.test(value);
}

async function boundedJson(response: Response, limit = 2 * 1024 * 1024): Promise<Json> {
  if (response.headers.get('content-type')?.split(';', 1)[0]?.trim() !== 'application/json') {
    throw DaemonApiError.fromKey('response', 'expectedAJsonResponse');
  }

  if (!response.body) throw DaemonApiError.fromKey('response', 'missingResponseBody');
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let length = 0;
  let text = '';
  let deadline: ReturnType<typeof setTimeout> | undefined;

  const expired = new Promise<never>((_, reject) => {
    deadline = setTimeout(
      () => reject(DaemonApiError.fromKey('http', 'daemonResponseTimedOut')),
      30000,
    );
  });

  try {
    for (;;) {
      const { value, done } = await Promise.race([reader.read(), expired]);

      if (done) break;
      length += value.byteLength;

      if (length > limit) throw DaemonApiError.fromKey('response', 'responseExceedsByteLimit');
      text += decoder.decode(value, { stream: true });
    }

    text += decoder.decode();

    return parseJson(text);
  } catch (error) {
    if (error instanceof DaemonApiError) throw error;
    throw DaemonApiError.fromKey('response', 'invalidUtf8Response');
  } finally {
    clearTimeout(deadline);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export class DaemonApi {
  #token = '';
  #credentialRevision = 0;
  #scopeRevision = 0;
  #activeRequests = 0;
  #rotation: Promise<void> | null = null;
  #idleWaiters = new Set<() => void>();
  #responseRevisions = new WeakMap<Response, number>();
  #daemonId: string | undefined;
  #buildId: string | undefined;
  #approvedPlans = new WeakMap<
    PlanView,
    { daemonId: string; buildId: string; revision: number; request: JsonObject }
  >();
  readonly origin: string;
  readonly protocolVersion: number;
  constructor(
    origin: string,
    pins: IdentityPins = {},
    private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis),
  ) {
    const url = new URL(origin);

    if (
      !/^https?:$/.test(url.protocol) ||
      url.username ||
      url.password ||
      url.pathname !== '/' ||
      url.search ||
      url.hash
    ) {
      throw DaemonApiError.fromKey('identity', 'expectedAnHttpOrigin');
    }

    this.origin = url.origin;

    if (typeof globalThis.location !== 'undefined' && globalThis.location.origin !== this.origin) {
      throw DaemonApiError.fromKey('identity', 'apiOriginMismatch');
    }

    this.#daemonId = pins.daemonId;
    this.#buildId = pins.buildId;
    this.protocolVersion = pins.protocolVersion ?? 1;

    if (
      !Number.isSafeInteger(this.protocolVersion) ||
      this.protocolVersion < 1 ||
      [pins.daemonId, pins.buildId].some(
        (pin) => pin !== undefined && !/^[\x21-\x7e]{1,256}$/.test(pin),
      )
    ) {
      throw DaemonApiError.fromKey('identity', 'invalidDaemonIdentityPins');
    }
  }
  get daemonId(): string | undefined {
    return this.#daemonId;
  }
  get buildId(): string | undefined {
    return this.#buildId;
  }
  get authenticated(): boolean {
    return this.#token.length > 0;
  }
  setToken(token: string): void {
    if (!/^[A-Za-z0-9._~+/=-]{1,4096}$/.test(token))
      throw DaemonApiError.fromKey('authentication', 'invalidBearerCredential');

    if (token !== this.#token) {
      this.#approvedPlans = new WeakMap();
      this.#credentialRevision++;
      this.#scopeRevision++;
    }

    this.#token = token;
  }
  setRotatedToken(token: string): void {
    if (!sessionToken(token))
      throw DaemonApiError.fromKey('authentication', 'invalidBearerCredential');
    this.#token = token;
  }
  clearToken(): void {
    this.#token = '';
    this.#approvedPlans = new WeakMap();
    this.#credentialRevision++;
    this.#scopeRevision++;
  }
  async bootstrap(input: string, replaceUrl: (url: string) => void): Promise<SessionGrant | null> {
    const stripped = stripBootstrapUrl(input);
    replaceUrl(stripped.url); // Synchronous removal MUST precede the first request.

    if (new URL(stripped.url).origin !== this.origin)
      throw DaemonApiError.fromKey('identity', 'bootstrapOriginDiffers');

    if (stripped.invalid)
      throw DaemonApiError.fromKey('authentication', 'ambiguousBootstrapCredential');

    if (stripped.code === null) return null;

    return this.exchangeBootstrapCode(stripped.code);
  }
  async exchangeBootstrapCode(code: string, signal?: AbortSignal): Promise<SessionGrant> {
    if (!/^[A-Za-z0-9._~+/=-]{1,4096}$/.test(code))
      throw DaemonApiError.fromKey('authentication', 'invalidBootstrapCredential');
    this.clearToken();
    const response = await this.#request('/api/bootstrap', 'POST', { code }, signal, true);
    const body = await this.#json(response, 8192);
    const grant = sessionGrant(body);

    if (!grant) {
      throw DaemonApiError.fromKey('response', 'invalidBootstrapResponse');
    }

    this.setToken(grant.access_token);

    return grant;
  }
  async rotateSession(signal?: AbortSignal): Promise<SessionGrant> {
    if (this.#rotation) await this.#rotation;

    if (signal?.aborted) throw DaemonApiError.fromKey('http', 'sessionRotationWasInterrupted');
    const revision = this.#credentialRevision;
    const presented = this.#token;
    let release!: () => void;
    this.#rotation = new Promise<void>((resolve) => {
      release = resolve;
    });

    try {
      await this.#waitForIdle(signal);

      if (
        signal?.aborted ||
        revision !== this.#credentialRevision ||
        !presented ||
        presented !== this.#token
      )
        throw DaemonApiError.fromKey('http', 'sessionRotationWasInterrupted');

      const grant = sessionGrant(
        await this.#json(
          await this.#request('/api/session', 'POST', undefined, signal, false, undefined, true),
        ),
      );

      if (!grant) throw DaemonApiError.fromKey('response', 'invalidSessionResponse');

      if (signal?.aborted || revision !== this.#credentialRevision)
        throw DaemonApiError.fromKey('http', 'sessionRotationWasInterrupted');
      this.setRotatedToken(grant.access_token);

      return grant;
    } finally {
      this.#rotation = null;
      release();
    }
  }
  async revokeSession(signal?: AbortSignal): Promise<void> {
    await this.#request('/api/session', 'DELETE', undefined, signal);
  }
  async #waitForIdle(signal?: AbortSignal): Promise<void> {
    if (this.#activeRequests === 0) return;
    await new Promise<void>((resolve, reject) => {
      const done = () => {
        this.#idleWaiters.delete(done);
        signal?.removeEventListener('abort', abort);
        resolve();
      };

      const abort = () => {
        this.#idleWaiters.delete(done);
        reject(DaemonApiError.fromKey('http', 'sessionRotationWasInterrupted'));
      };

      this.#idleWaiters.add(done);
      signal?.addEventListener('abort', abort, { once: true });

      if (signal?.aborted) abort();
    });
  }
  async health(signal?: AbortSignal): Promise<Health> {
    const result = healthSchema.safeParse(
      await this.#json(await this.#request('/api/health', 'GET', undefined, signal)),
    );

    if (!result.success) throw DaemonApiError.fromKey('response', 'invalidDaemonHealthResponse');
    const body = result.data;
    const minimum = body.protocol_min;
    const maximum = body.protocol_max;

    if (
      minimum < 1 ||
      minimum > maximum ||
      minimum > this.protocolVersion ||
      maximum < this.protocolVersion
    ) {
      throw DaemonApiError.fromKey('protocol', 'daemonProtocolMismatch');
    }

    if (this.#daemonId && body.daemon_id !== this.#daemonId) {
      throw DaemonApiError.fromKey('identity', 'daemonIdentityChanged', {
        status: 409,
        daemonIdentityChanged: true,
      });
    }

    if (this.#buildId && body.build_id !== this.#buildId)
      throw DaemonApiError.fromKey('identity', 'daemonPinsChanged', { buildMismatch: true });
    this.#daemonId = body.daemon_id;
    this.#buildId = body.build_id;

    return {
      daemon_id: body.daemon_id,
      build_id: body.build_id,
      protocol_min: minimum,
      protocol_max: maximum,
    };
  }
  async read(path: string, signal?: AbortSignal): Promise<Json> {
    this.#apiUrl(path);
    await this.health(signal);

    return this.#json(await this.#request(path, 'GET', undefined, signal));
  }
  /** Artifact bytes only; consumers must bound and recheck before publishing. */
  async downloadResponse(path: string, signal?: AbortSignal): Promise<Response> {
    if (!/^\/api\/runs\/[0-9a-f-]{36}\/artifacts\/.+/.test(path) || path.includes('?'))
      throw DaemonApiError.fromKey('response', 'invalidArtifactDownloadRoute');
    this.#apiUrl(path);
    await this.health(signal);

    return this.#request(path, 'GET', undefined, signal);
  }
  assertResponseCurrent(response: Response): void {
    if (this.#responseRevisions.get(response) !== this.#credentialRevision || !this.authenticated) {
      void response.body?.cancel().catch(() => undefined);
      throw DaemonApiError.fromKey('authentication', 'downloadCredentialChanged');
    }
  }
  async listRuns(
    options: { limit?: number; offset?: number; state?: RunState; signal?: AbortSignal } = {},
  ): Promise<RunView[]> {
    const limit = options.limit ?? 50;
    const offset = options.offset ?? 0;

    if (
      !Number.isSafeInteger(limit) ||
      limit < 1 ||
      limit > 100 ||
      !Number.isSafeInteger(offset) ||
      offset < 0
    ) {
      throw DaemonApiError.fromKey('response', 'invalidRunPagination');
    }

    const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });

    if (options.state) query.set('state', options.state);
    const value = await this.read(`/api/runs?${query}`, options.signal);
    const result = runListSchema.safeParse(value);

    if (!result.success) throw DaemonApiError.fromKey('response', 'invalidRunList');

    return result.data.runs.map((item) => runView(item));
  }
  async getRun(runId: string, signal?: AbortSignal): Promise<RunView> {
    if (!isUuid(runId)) throw DaemonApiError.fromKey('identity', 'invalidRunIdentity');

    return runView(await this.read(`/api/runs/${runId}`, signal), runId);
  }
  async plan(request: JsonObject, idempotencyKey: string, signal?: AbortSignal): Promise<PlanView> {
    if (!idempotencyKey || idempotencyKey.length > 256 || /[\r\n]/.test(idempotencyKey))
      throw DaemonApiError.fromKey('response', 'invalidIdempotencyKey');
    const revision = this.#scopeRevision;
    this.#requestBody(request, idempotencyKey);
    const snapshot = structuredClone(request);
    freezeJson(snapshot);
    const value = await this.mutate('/api/runs/plan', snapshot, { idempotencyKey, signal });

    if (revision !== this.#scopeRevision)
      throw DaemonApiError.fromKey('authentication', 'requestCredentialChanged');
    const result = planSchema.safeParse(value);

    if (!result.success) throw DaemonApiError.fromKey('response', 'invalidPlanResponse');
    const plan = result.data;
    // Approval is memory-only and belongs to this exact verified connection.
    // A persisted plan must be re-planned with its original idempotency key;
    // never manufacture a DTO to bypass a fresh preview/identity check.
    freezeJson(plan.preview);

    const approved = Object.freeze({
      plan_id: plan.plan_id,
      run_id: plan.run_id,
      request_id: plan.request_id,
      preview: plan.preview,
    });

    if (!this.#daemonId || !this.#buildId)
      throw DaemonApiError.fromKey('identity', 'planLacksAVerifiedDaemonIdentity');
    this.#approvedPlans.set(approved, {
      daemonId: this.#daemonId,
      buildId: this.#buildId,
      revision,
      request: snapshot,
    });

    return approved;
  }
  approvedRequest(plan: PlanView): JsonObject {
    const approval = this.#approvedPlans.get(plan);

    if (
      !this.authenticated ||
      !approval ||
      approval.revision !== this.#scopeRevision ||
      approval.daemonId !== this.#daemonId ||
      approval.buildId !== this.#buildId
    )
      throw DaemonApiError.fromKey('identity', 'planConnectionMismatch');

    return structuredClone(approval.request);
  }
  async submit(plan: PlanView, signal?: AbortSignal): Promise<RunView> {
    this.approvedRequest(plan);

    if (!isUuid(plan.plan_id) || !isUuid(plan.run_id) || !isUuid(plan.request_id))
      throw DaemonApiError.fromKey('identity', 'invalidApprovedPlanIdentity');

    const value = runView(
      await this.mutate('/api/runs', { plan_id: plan.plan_id }, { signal }),
      plan.run_id,
    );

    if (value.plan_id !== plan.plan_id || value.request_id !== plan.request_id)
      throw DaemonApiError.fromKey('identity', 'submissionPlanMismatch');

    return value;
  }
  async cancelRun(runId: string, signal?: AbortSignal): Promise<RunView> {
    if (!isUuid(runId)) throw DaemonApiError.fromKey('identity', 'invalidRunIdentity');

    return runView(await this.mutate(`/api/runs/${runId}/cancel`, undefined, { signal }), runId);
  }
  async mutate(
    path: string,
    body: JsonObject | undefined,
    options: { signal?: AbortSignal; idempotencyKey?: string } = {},
  ): Promise<Json> {
    this.#apiUrl(path);
    this.#requestBody(body, options.idempotencyKey);
    const snapshot = body === undefined ? undefined : structuredClone(body);

    if (snapshot) freezeJson(snapshot);
    const revision = this.#credentialRevision;
    await this.health(options.signal);

    if (revision !== this.#credentialRevision)
      throw DaemonApiError.fromKey('authentication', 'requestCredentialChanged');

    return this.#json(
      await this.#request(path, 'POST', snapshot, options.signal, false, options.idempotencyKey),
    );
  }
  async openEvents(
    runId: string,
    cursor: { attempt: number; seq: number },
    signal?: AbortSignal,
    expectedDaemonId?: string,
  ): Promise<Response> {
    if (
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(runId) ||
      !Number.isSafeInteger(cursor.attempt) ||
      !Number.isSafeInteger(cursor.seq) ||
      cursor.attempt < 0 ||
      cursor.seq < -1 ||
      (cursor.attempt === 0 && cursor.seq !== -1)
    )
      throw DaemonApiError.fromKey('response', 'invalidEventCursor');
    await this.health(signal);

    if (expectedDaemonId !== undefined && expectedDaemonId !== this.#daemonId)
      throw DaemonApiError.fromKey('identity', 'cursorDaemonMismatch');

    const response = await this.#request(
      `/api/runs/${runId}/events?attempt=${cursor.attempt}&seq=${cursor.seq}`,
      'GET',
      undefined,
      signal,
    );

    if (response.headers.get('content-type')?.split(';', 1)[0]?.trim() !== 'text/event-stream') {
      await response.body?.cancel();
      throw DaemonApiError.fromKey('response', 'expectedAnEventStream');
    }

    return response;
  }
  #apiUrl(path: string): URL {
    const url = new URL(path, this.origin);

    if (
      !path.startsWith('/api/') ||
      url.origin !== this.origin ||
      url.username ||
      url.password ||
      url.hash ||
      !url.pathname.startsWith('/api/')
    ) {
      throw DaemonApiError.fromKey('identity', 'apiCrossOrigin');
    }

    if ([...url.searchParams.keys()].some((key) => secretNames.has(key.toLowerCase()))) {
      throw DaemonApiError.fromKey('identity', 'apiUrlCredentials');
    }

    return url;
  }
  #requestBody(body?: JsonObject, idempotencyKey?: string): string | undefined {
    if (idempotencyKey !== undefined && !/^[\x21-\x7e]{1,256}$/.test(idempotencyKey)) {
      throw DaemonApiError.fromKey('response', 'invalidIdempotencyKey');
    }

    if (body === undefined) return undefined;
    validateJson(body);
    const serialized = JSON.stringify(body);

    if (new TextEncoder().encode(serialized).byteLength > 1024 * 1024) {
      throw DaemonApiError.fromKey('response', 'requestExceedsByteLimit');
    }

    return serialized;
  }
  async #json(response: Response, limit?: number): Promise<Json> {
    const revision = this.#responseRevisions.get(response);

    if (revision !== this.#credentialRevision) {
      await response.body?.cancel();
      throw DaemonApiError.fromKey('authentication', 'requestCredentialChanged');
    }

    const body = await boundedJson(response, limit);

    if (revision !== this.#credentialRevision)
      throw DaemonApiError.fromKey('authentication', 'requestCredentialChanged');

    return body;
  }
  async #request(
    path: string,
    method: 'GET' | 'POST' | 'DELETE',
    body?: JsonObject,
    signal?: AbortSignal,
    bootstrap = false,
    idempotencyKey?: string,
    rotation = false,
  ): Promise<Response> {
    if (signal?.aborted) throw DaemonApiError.fromKey('http', 'sessionRotationWasInterrupted');

    if (!rotation && this.#rotation) {
      const revision = this.#credentialRevision;
      await this.#rotation;

      if (signal?.aborted || revision !== this.#credentialRevision)
        throw DaemonApiError.fromKey('http', 'sessionRotationWasInterrupted');
    }

    const url = this.#apiUrl(path);

    if (!bootstrap && !this.#token)
      throw DaemonApiError.fromKey('authentication', 'credentialMissing');
    const headers = new Headers({ Accept: 'application/json' });

    if (!bootstrap) headers.set('Authorization', `Bearer ${this.#token}`);

    if (this.#daemonId && !bootstrap) headers.set('X-WaveDL-Daemon-ID', this.#daemonId);
    const serialized = this.#requestBody(body, idempotencyKey);

    if (body !== undefined) headers.set('Content-Type', 'application/json');

    if (idempotencyKey !== undefined) headers.set('Idempotency-Key', idempotencyKey);
    const deadline = new AbortController();

    const timer = setTimeout(
      () => deadline.abort(DaemonApiError.fromKey('http', 'daemonRequestTimedOut')),
      15000,
    );

    const revision = this.#credentialRevision;
    const presentedToken = this.#token;
    let response: Response;
    this.#activeRequests++;

    try {
      response = await this.fetcher(url.href, {
        method,
        headers,
        body: serialized,
        signal: signal ? AbortSignal.any([signal, deadline.signal]) : deadline.signal,
        credentials: 'omit',
        redirect: 'error',
        cache: 'no-store',
        referrerPolicy: 'no-referrer',
      });
    } finally {
      clearTimeout(timer);
      this.#activeRequests--;

      if (this.#activeRequests === 0) {
        for (const resolve of this.#idleWaiters) resolve();
        this.#idleWaiters.clear();
      }
    }

    if (revision !== this.#credentialRevision) {
      await response.body?.cancel();
      throw DaemonApiError.fromKey('authentication', 'requestCredentialChanged');
    }

    this.#responseRevisions.set(response, revision);

    if (!response.ok) {
      await response.body?.cancel();

      if (response.status === 401) {
        if (presentedToken === this.#token) {
          this.clearToken();
        }

        throw DaemonApiError.fromKey('authentication', 'credentialRejected', { status: 401 });
      }

      const daemonIdentityChanged =
        response.status === 409 &&
        (url.pathname === '/api/health' ||
          url.pathname === '/api/session' ||
          response.headers.get('X-WaveDL-Error-Code') === 'daemon_identity_changed');

      if (daemonIdentityChanged)
        throw DaemonApiError.fromKey('identity', 'daemonIdentityChanged', {
          status: response.status,
          daemonIdentityChanged: true,
        });
      throw DaemonApiError.fromKey('http', 'daemonRequestFailed', {
        status: response.status,
        params: { status: response.status },
      });
    }

    return response;
  }
}
