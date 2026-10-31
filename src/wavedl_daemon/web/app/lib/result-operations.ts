import { clientError } from '../i18n/client-messages';
import { resultRequestSchema, type ResultRequest } from '../../shared/schemas/result-request';
import { toRaw } from 'vue';
import { ArtifactClient, type Artifact, type SelectionMetadata } from './artifact-client';
import { CatalogApi, type DataSelection } from './catalog-api';
import type { ConnectionSession } from './connection-session';
import { DaemonApiError, type PlanView, type RunView } from './daemon-api';

export interface CheckpointOption {
  readonly artifact: Artifact;
  readonly classification: SelectionMetadata;
  readonly label: string;
  readonly selectable: boolean;
}

/** Labels come from verified metadata, never filenames or generation ordering. */
export async function checkpointOptions(
  session: ConnectionSession,
  artifacts: readonly Artifact[],
  signal?: AbortSignal,
): Promise<readonly CheckpointOption[]> {
  const client = new ArtifactClient(session);
  const result: CheckpointOption[] = [];

  for (const artifact of artifacts) {
    if (artifact.path !== 'checkpoint_selection.json') continue;
    const classification = await client.selectionMetadata(artifact, signal);

    const selectable =
      classification.kind !== 'cross_validation_summary' &&
      (classification.kind !== 'cross_validation_fold' || classification.fold !== undefined);

    const kind = classification.kind.replaceAll('_', ' ');
    result.push({
      artifact,
      classification,
      selectable,
      label: `${artifact.generation} · attempt ${artifact.attempt} · ${kind}${classification.fold === undefined ? '' : ` ${classification.fold}`}`,
    });
  }

  return result;
}

/** Explicit source publication; no filename, last attempt or host-path inference. */
export class ResultOperations {
  #request: ResultRequest | null = null;
  #fingerprint = '';
  #key = '';
  #plan: PlanView | null = null;
  #epoch = 0;
  #revision = -1;
  readonly session: ConnectionSession;
  constructor(
    session: ConnectionSession,
    readonly keyFactory = () => crypto.randomUUID(),
  ) {
    this.session = toRaw(session);
  }
  get approved(): PlanView | null {
    return this.#revision === this.session.snapshot().revision ? this.#plan : null;
  }
  invalidate() {
    this.#epoch++;
    this.#plan = null;
    this.#request = null;
  }
  async prepare(
    draft: ResultRequest,
    data: DataSelection,
    metadata: Artifact,
    signal?: AbortSignal,
  ) {
    const epoch = ++this.#epoch;
    this.#plan = null;
    this.#request = null;
    new CatalogApi(this.session).assertSelection(data);
    const client = new ArtifactClient(this.session);
    client.assertSelection(metadata);
    const request = resultRequestSchema.parse(draft);

    if (!request.storage_map?.length) delete request.storage_map;

    if (
      data.kind !== 'file' ||
      request.config.data_path !== data.path ||
      request.checkpoint_source.run_id !== metadata.runId ||
      request.checkpoint_source.generation !== metadata.generation
    )
      throw clientError('inferenceSelectionRequired', {});
    const classification = await client.selectionMetadata(metadata, signal);

    if (epoch !== this.#epoch) throw clientError('draftChangedPreviewAgain', {});

    if (
      classification.kind === 'cross_validation_summary' ||
      (classification.kind === 'cross_validation_fold' && classification.fold === undefined)
    )
      throw clientError('cvFoldRequired', {});
    client.assertSelection(metadata);
    new CatalogApi(this.session).assertSelection(data);
    const fingerprint = JSON.stringify(request);

    if (fingerprint !== this.#fingerprint) {
      this.#key = this.keyFactory();
      this.#fingerprint = fingerprint;
    }

    this.#request = request;
    this.#revision = this.session.snapshot().revision;
  }
  async preview(signal?: AbortSignal): Promise<PlanView> {
    if (!this.#request) throw clientError('validateTheInferenceDraftFirst', {});

    if (this.#revision !== this.session.snapshot().revision || !this.session.api)
      throw clientError('inferenceValidationRequired', {});
    const epoch = this.#epoch;
    const request = this.#request;
    const key = this.#key;

    const plan = await this.session.perform((api, scoped) => api.plan(request, key, scoped), {
      signal,
    });

    if (epoch !== this.#epoch) throw clientError('draftChangedPreviewAgain', {});
    this.#revision = this.session.snapshot().revision;
    this.#plan = plan;

    return plan;
  }
  async submit(signal?: AbortSignal): Promise<RunView> {
    const plan = this.approved;

    if (!plan) throw clientError('inferencePreviewRequired', {});

    return this.session.perform((api, scoped) => api.submit(plan, scoped), { signal });
  }
  async status(signal?: AbortSignal): Promise<RunView> {
    const plan = this.approved;

    if (!plan) throw clientError('previewFirst', {});

    return this.session.perform(
      async (api, scoped) => {
        const view = await api.getRun(plan.run_id, scoped);

        if (view.plan_id !== plan.plan_id || view.request_id !== plan.request_id)
          throw DaemonApiError.fromKey('identity', 'inferencePlanMismatch');

        return view;
      },
      { signal },
    );
  }
}
