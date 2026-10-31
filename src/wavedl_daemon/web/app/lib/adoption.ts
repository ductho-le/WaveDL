import { toRaw } from 'vue';
import type { ClientErrorKey, ClientErrorParams } from '../i18n/client-messages';
import { adoptionRequestSchema, type AdoptionRequest } from '../../shared/schemas/adoption-request';
import { CatalogApi, type DataSelection } from './catalog-api';
import type { ConnectionSession } from './connection-session';
import { DaemonApiError, isUuid, object, type Json } from './daemon-api';

export interface AdoptionResult {
  readonly plan_id: string;
  readonly run_id: string;
  readonly request_id: string;
  readonly operation: 'train' | 'test' | 'plot';
  readonly attempt: number;
  readonly backend: 'local';
  readonly state: 'supervision_lost';
  readonly uncertainty: true;
  readonly submission_requested: true;
  readonly cancellation_requested: false;
  readonly backend_status: 'unknown';
  readonly adopted: true;
}

export type AdoptionStatus =
  | 'draft'
  | 'ready'
  | 'pending'
  | 'acknowledgement_unknown'
  | 'rejected'
  | 'confirmed';

export class AdoptionError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly clientMessageKey?: ClientErrorKey,
    readonly clientMessageParams?: ClientErrorParams,
  ) {
    super(message);
    this.name = 'AdoptionError';
  }
}

function response(
  value: Json,
  attempt: number,
): AdoptionResult {
  if (
    !object(value) ||
    Object.keys(value).length !== 12 ||
    !isUuid(value.plan_id) ||
    !isUuid(value.run_id) ||
    !isUuid(value.request_id) ||
    (value.operation !== 'train' && value.operation !== 'test' && value.operation !== 'plot') ||
    value.attempt !== attempt ||
    value.backend !== 'local' ||
    value.state !== 'supervision_lost' ||
    value.uncertainty !== true ||
    value.submission_requested !== true ||
    value.cancellation_requested !== false ||
    value.backend_status !== 'unknown' ||
    value.adopted !== true
  )
    throw DaemonApiError.fromKey('response', 'invalidAdoptionAcknowledgement');

  return Object.freeze({
    plan_id: value.plan_id,
    run_id: value.run_id,
    request_id: value.request_id,
    operation: value.operation,
    attempt,
    backend: 'local',
    state: 'supervision_lost',
    uncertainty: true,
    submission_requested: true,
    cancellation_requested: false,
    backend_status: 'unknown',
    adopted: true,
  });
}

/** Explicit ownership registration only: never submits or cancels a process. */
export class AdoptionWorkflow {
  readonly session: ConnectionSession;
  #source: DataSelection | null = null;
  #request: Readonly<AdoptionRequest> | null = null;
  #key = '';
  #daemonId = '';
  #result: AdoptionResult | null = null;
  #confirmedRevision = -1;
  #status: AdoptionStatus = 'draft';
  constructor(
    session: ConnectionSession,
    readonly keyFactory = () => crypto.randomUUID(),
  ) {
    this.session = toRaw(session);
  }
  get source() {
    return this.#source;
  }
  get request() {
    return this.#request;
  }
  get key() {
    return this.#key;
  }
  get result() {
    return this.#confirmedRevision === this.session.snapshot().revision ? this.#result : null;
  }
  get status() {
    return this.#status;
  }
  prepare(directory: DataSelection, attempt: number): void {
    new CatalogApi(this.session).assertSelection(directory);

    if (directory.kind !== 'directory')
      throw new AdoptionError(
        'Choose a daemon run directory.',
        undefined,
        'chooseADaemonRunDirectory',
      );
    const request = adoptionRequestSchema.parse({ path: directory.path, attempt });

    if (this.#status === 'pending')
      throw new AdoptionError(
        'Wait for the current adoption acknowledgement.',
        undefined,
        'adoptionAwaitingAck',
      );

    const same =
      this.#daemonId === directory.daemonId &&
      this.#request?.path === request.path &&
      this.#request.attempt === request.attempt;

    if (this.#status === 'acknowledgement_unknown' && !same)
      throw new AdoptionError(
        'Revalidate the same directory and attempt before retrying the retained request.',
        undefined,
        'adoptionRevalidationRequired',
      );

    if (!same) {
      this.#key = this.keyFactory();
      this.#request = Object.freeze(request);
      this.#daemonId = directory.daemonId;
      this.#result = null;
    }

    this.#source = directory;

    if (this.#status !== 'acknowledgement_unknown' && !(same && this.result))
      this.#status = 'ready';
  }
  async adopt(): Promise<AdoptionResult> {
    if (!this.#source || !this.#request)
      throw new AdoptionError(
        'Select a daemon directory and attempt first.',
        undefined,
        'adoptionSelectionRequired',
      );
    new CatalogApi(this.session).assertSelection(this.#source);

    if (this.#status === 'pending')
      throw new AdoptionError(
        'An adoption request is already pending.',
        undefined,
        'adoptionAlreadyPending',
      );

    if (this.#result && this.#confirmedRevision === this.session.snapshot().revision)
      return this.#result;
    const request = this.#request;
    const key = this.#key;
    let httpStatus: number | undefined;
    this.#result = null;
    this.#status = 'pending';

    try {
      const result = await this.session.perform(async (api, signal) => {
        try {
          return response(
            await api.mutate('/api/runs/adopt', { ...request }, { idempotencyKey: key, signal }),
            request.attempt,
          );
        } catch (error) {
          if (error instanceof DaemonApiError && error.code === 'http') httpStatus = error.status;
          throw error;
        }
      });

      this.#result = result;
      this.#confirmedRevision = this.session.snapshot().revision;
      this.#status = 'confirmed';

      return result;
    } catch {
      // Service failures may follow a durable registration; retry only the retained request.
      this.#status =
        httpStatus !== undefined && httpStatus < 500 ? 'rejected' : 'acknowledgement_unknown';
      throw new AdoptionError(
        this.#status === 'rejected'
          ? 'The daemon rejected adoption. No ownership confirmation was received.'
          : 'Adoption acknowledgement is unknown. Reconnect and reselect the same directory and attempt, then retry the retained request.',
        httpStatus,
        this.#status === 'rejected' ? 'adoptionRejected' : 'adoptionAckUnknown',
      );
    }
  }
}
