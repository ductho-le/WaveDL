import { clientError } from '../i18n/client-messages';
import { trainingRequestSchema, type TrainingRequest } from '../../shared/schemas/training-request';
import { CatalogApi, type DataSelection, type ModelSelection } from './catalog-api';
import type { ConnectionSession } from './connection-session';
import { DaemonApiError, type PlanView, type RunView } from './daemon-api';
import { toRaw } from 'vue';

/** Memory-only draft/approval scope; retries never rotate an unchanged request key. */
export class ProjectPlan {
  readonly session: ConnectionSession;
  #request: TrainingRequest | null = null;
  #fingerprint = '';
  #key = '';
  #approved: PlanView | null = null;
  #revision = -1;
  #epoch = 0;
  constructor(
    session: ConnectionSession,
    readonly keyFactory = () => crypto.randomUUID(),
  ) {
    this.session = toRaw(session);
  }
  get approved(): PlanView | null {
    return this.#revision === this.session.snapshot().revision ? this.#approved : null;
  }
  get key(): string {
    return this.#key;
  }
  invalidate(): void {
    this.#epoch++;
    this.#approved = null;
  }
  prepare(draft: TrainingRequest, data: DataSelection, model: ModelSelection): void {
    const catalogue = new CatalogApi(this.session);
    catalogue.assertSelection(data);
    catalogue.assertSelection(model);

    if (
      data.kind !== 'file' ||
      draft.config.data_path !== data.path ||
      draft.config.model !== model.model
    )
      throw clientError('trainingSelectionMismatch', {});
    const request = trainingRequestSchema.parse(draft);

    if (!request.storage_map?.length) delete request.storage_map;
    const fingerprint = JSON.stringify(request);

    if (fingerprint !== this.#fingerprint) {
      this.#fingerprint = fingerprint;
      this.#key = this.keyFactory();
      this.#approved = null;
    }

    this.#request = request;
  }
  async preview(signal?: AbortSignal): Promise<PlanView> {
    if (!this.#request) throw clientError('validateTheTrainingDraftFirst', {});
    const request = this.#request;
    const key = this.#key;
    const epoch = this.#epoch;

    const plan = await this.session.perform((api, scoped) => api.plan(request, key, scoped), {
      signal,
    });

    if (epoch !== this.#epoch || key !== this.#key || request !== this.#request)
      throw clientError('planningDraftChanged', {});
    this.#revision = this.session.snapshot().revision;
    this.#approved = plan;

    return plan;
  }
  async submit(signal?: AbortSignal): Promise<RunView> {
    const plan = this.approved;

    if (!plan) throw clientError('trainingPreviewRequired', {});

    return this.session.perform((api, scoped) => api.submit(plan, scoped), { signal });
  }
  async status(signal?: AbortSignal): Promise<RunView> {
    const plan = this.approved;

    if (!plan) throw clientError('submissionPreviewRequired', {});

    return this.session.perform(
      async (api, scoped) => {
        const run = await api.getRun(plan.run_id, scoped);

        if (run.plan_id !== plan.plan_id || run.request_id !== plan.request_id)
          throw DaemonApiError.fromKey('identity', 'runPlanMismatch');

        return run;
      },
      { signal },
    );
  }
}
