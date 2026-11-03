import { clientError, type ClientMessage } from '../i18n/client-messages';
import { shallowRef, toRaw } from 'vue';
import {
  trainingRequestSchema,
  type TrainingRequest,
} from '../../shared/schemas/training-request';
import {
  inferenceDefaults,
  resultRequestSchema,
  type ResultRequest,
} from '../../shared/schemas/result-request';
import {
  object,
  type DaemonApi,
  type Json,
  type JsonObject,
  type PlanView,
  type RunView,
} from './daemon-api';
import {
  mappedStoragePath,
  normalizedStoragePath,
  storageMapSchema,
} from '../../shared/schemas/storage-mapping';
import type { ConnectionSession } from './connection-session';
import { ArtifactClient, type Artifact, type SelectionMetadata } from './artifact-client';
import { checkpointOptions, type CheckpointOption } from './result-operations';
import {
  approvedTrainingManifestSchema, trainingPlanPreviewSchema, approvedTrainingConfigSchema,
  sourcePlotManifestSchema, approvedPlotManifestSchema,
} from '../../shared/schemas/autoplot';

interface AutoPlotIntent {
  readonly request: ResultRequest;
  readonly idempotencyKey: string;
}

/** Extract only the exact approved manifest, never current form or run-list data. */
export function automaticPlotRequest(
  run: RunView,
  plan: PlanView,
  originalRequest?: JsonObject,
): TrainingRequest {
  if (plan.preview.spec === undefined) throw clientError('missingApprovedManifest', {});
  const parsed = approvedTrainingManifestSchema.safeParse(plan.preview.spec);
  const preview = trainingPlanPreviewSchema.safeParse(plan.preview);

  if (!parsed.success || !preview.success || parsed.data.run_id !== run.run_id || preview.data.target !== run.backend)
    throw clientError('unsupportedTrainingManifest', {});
  const manifest = parsed.data;
  const mappings = storageMapSchema.parse(manifest.storage_map ?? []);
  const parsedConfig = approvedTrainingConfigSchema.safeParse(manifest.resolved_config);

  if (!parsedConfig.success)
    throw clientError('unsupportedTrainingConfig', {});
  const config = parsedConfig.data;

  const runtime = manifest.runtime;

  const normalized = trainingRequestSchema.parse({
    backend: preview.data.target,
    operation: manifest.operation,
    config,
    runtime: {
      python: runtime.python,
      mode: runtime.mode,
      device: runtime.device,
      environment: runtime.environment,
    },
    topology: preview.data.topology,
  });

  if (!originalRequest) {
    if (mappings.length) throw clientError('plotHostRequestRequired', {});

    return normalized;
  }

  const original = trainingRequestSchema.parse(originalRequest);
  const approvedMappings = original.storage_map ?? [];

  const mappingIdentity = (items: typeof mappings): string =>
    JSON.stringify(
      items
        .map((item) => ({
          host_root: normalizedStoragePath(item.host_root),
          execution_root: normalizedStoragePath(item.execution_root),
        }))
        .sort((a, b) => a.host_root.localeCompare(b.host_root)),
    );

  if (
    mappingIdentity(approvedMappings) !== mappingIdentity(mappings) ||
    original.backend !== normalized.backend
  )
    throw clientError('storageApprovalMismatch', {});

  if (!mappings.length) return normalized;

  if (
    mappedStoragePath(original.config.data_path, mappings) !==
      normalizedStoragePath(normalized.config.data_path) ||
    mappedStoragePath(original.runtime.python, mappings) !==
      normalizedStoragePath(normalized.runtime.python)
  )
    throw clientError('hostPathMismatch', {});
  // File registrations are mapped by the daemon just like the dataset. Keep
  // their original host spelling after verifying one forward rewrite; sending
  // normalized execution paths would apply overlapping maps a second time.
  const imports = original.config.import_modules;

  if (
    imports.length !== normalized.config.import_modules.length ||
    imports.some((item, index) => {
      const approved = normalized.config.import_modules[index];

      return item.endsWith('.py')
        ? mappedStoragePath(item, mappings) !== normalizedStoragePath(approved ?? '')
        : item !== approved;
    })
  )
    throw clientError('registrationPathMismatch', {});

  return trainingRequestSchema.parse({
    ...normalized,
    config: {
      ...normalized.config,
      data_path: original.config.data_path,
      import_modules: imports,
    },
    runtime: { ...normalized.runtime, python: original.runtime.python },
    storage_map: original.storage_map,
  });
}

export interface AutomaticPlotState {
  status: 'armed' | 'planning' | 'submitted' | 'error' | 'unavailable' | 'stopped';
  plotRunId: string | null;
  message: ClientMessage;
}

function canonical(value: Json): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;

  if (object(value))
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key] ?? null)}`)
      .join(',')}}`;

  return JSON.stringify(value);
}

const inferenceWeightNames = new Set([
  'model.safetensors',
  'model.bin',
  'pytorch_model.bin',
  'model_weights.pth',
]);

/** A best publication is usable for automatic plots only when its inference state is complete. */
function hasInferenceSource(artifacts: readonly Artifact[], generation: string): boolean {
  const paths = new Set(
    artifacts
      .flatMap((artifact) => artifact.generation === generation ? [artifact.path] : []),
  );

  return (
    paths.has('training_meta.pkl') &&
    paths.has('scaler.pkl') &&
    [...inferenceWeightNames].some((name) => paths.has(name))
  );
}

/** Bounded browser-local continuation of one explicitly approved training run. */
export class AutoPlotController {
  readonly state = shallowRef<AutomaticPlotState>({
    status: 'stopped',
    plotRunId: null,
    message: '',
  });
  readonly session: ConnectionSession;
  #request: TrainingRequest | null = null;
  #run: RunView | null = null;
  #intent: AutoPlotIntent | null = null;
  #plan: PlanView | null = null;
  #sourcePlan: PlanView | null = null;
  #sourceApi: DaemonApi | null = null;
  #sourceFiles: Artifact[] = [];
  #sourceSelection: SelectionMetadata | null = null;
  #scope = new AbortController();
  #revision = -1;
  #daemonId = '';
  #busy = false;
  #sourceAttempt: number | null = null;
  #observedAttempt: number | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #unsubscribe: () => void;
  // Status recovery must never re-arm explicitly disabled/completed workflows.
  // Bound the page-local ledger without evicting deduplication authority.
  #capturedRequests = new Set<string>();
  constructor(
    session: ConnectionSession,
    readonly options: { pollMs?: number; deadlineMs?: number } = {},
  ) {
    this.session = toRaw(session);

    for (const value of [options.pollMs ?? 2000, options.deadlineMs ?? 15000])
      if (!Number.isSafeInteger(value) || value < 1)
        throw clientError('invalidAutomaticPlotDeadline', {});
    this.#unsubscribe = this.session.subscribe((state) => {
      if (
        this.#run &&
        (state.status !== 'connected' ||
          state.revision !== this.#revision ||
          state.health?.daemon_id !== this.#daemonId)
      )
        this.stop();
    });
  }
  capture(run: RunView, plan: PlanView): void {
    if (this.#capturedRequests.has(run.request_id)) return;
    this.stop();
    const state = this.session.snapshot();

    if (state.status !== 'connected' || !state.health) return;

    if (this.#capturedRequests.size >= 256) {
      this.state.value = {
        status: 'unavailable',
        plotRunId: null,
        message: clientError('autoPlotCapacity', {}),
      };

      return;
    }

    this.#capturedRequests.add(run.request_id);

    try {
      const approved = toRaw(plan);
      const api = this.session.api;

      if (!api) throw clientError('autoPlotConnectionRequired', {});
      const request = automaticPlotRequest(run, approved, api.approvedRequest(approved));

      if (request.config.cv > 0) return;
      this.#revision = state.revision;
      this.#sourcePlan = approved;
      this.#sourceApi = api;
      this.#daemonId = state.health.daemon_id;
      this.#scope = new AbortController();

      if (
        !run.submission_requested ||
        run.operation !== 'train' ||
        plan.run_id !== run.run_id ||
        plan.request_id !== run.request_id ||
        plan.plan_id !== run.plan_id
      )
        throw clientError('autoPlotAlreadySubmitted', {});
      this.#request = structuredClone(request);
      this.#run = structuredClone(toRaw(run));
      this.#observedAttempt = run.attempt;
      this.state.value = {
        status: 'armed',
        plotRunId: null,
        message: clientError('autoPlotCheckpointNotice', {}),
      };
      this.#schedule();
    } catch {
      this.stop();
      this.state.value = {
        status: 'unavailable',
        plotRunId: null,
        message: clientError('autoPlotManifestUnsupported', {}),
      };
    }
  }
  #schedule(): void {
    if (this.#timer || this.state.value.status !== 'armed' || this.#scope.signal.aborted) return;
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.#poll();
    }, this.options.pollMs ?? 2000);
  }
  async #poll(): Promise<void> {
    const run = this.#run;
    const signal = this.#scope.signal;

    if (!run || !this.#current(signal)) return;

    try {
      const current = await this.session.perform((api, signal) => api.getRun(run.run_id, signal), {
        signal: this.#scope.signal,
        timeoutMs: this.options.deadlineMs ?? 15000,
      });

      await this.observeFresh(current);
    } catch {
      this.#error(signal);
    }

    this.#schedule();
  }
  #error(signal: AbortSignal): void {
    if (this.#scope.signal === signal && !signal.aborted)
      this.state.value = {
        ...this.state.value,
        status: 'error',
        message: clientError('autoPlotAckUnknown', {}),
      };
  }
  #current(signal: AbortSignal): boolean {
    const state = this.session.snapshot();

    const current =
      !signal.aborted &&
      state.status === 'connected' &&
      state.revision === this.#revision &&
      state.health?.daemon_id === this.#daemonId;

    if (!current) return false;

    try {
      if (!this.#sourcePlan || !this.#sourceApi || this.session.api !== this.#sourceApi)
        throw clientError('sourceApprovalScopeChanged', {});
      this.#sourceApi.approvedRequest(this.#sourcePlan);

      return true;
    } catch {
      this.stop();

      return false;
    }
  }
  async observeFresh(run: RunView): Promise<void> {
    if (this.#busy || this.state.value.status !== 'armed' || !this.#run) return;

    if (!this.#current(this.#scope.signal)) return;

    if (
      run.run_id !== this.#run.run_id ||
      run.plan_id !== this.#run.plan_id ||
      run.request_id !== this.#run.request_id
    )
      return;

    if (run.operation !== this.#run.operation || run.backend !== this.#run.backend) {
      this.stop();

      return;
    }

    if (
      this.#observedAttempt !== null &&
      (run.attempt === null || run.attempt < this.#observedAttempt)
    ) {
      this.stop();

      return;
    }

    this.#observedAttempt = run.attempt;

    if (run.state === 'failed' || run.state === 'cancelled') {
      this.stop();

      return;
    }

    if (run.state !== 'completed' || run.uncertainty || run.attempt === null) return;

    this.#busy = true;
    const parent = this.#scope.signal;
    const signal = AbortSignal.any([parent, AbortSignal.timeout(this.options.deadlineMs ?? 15000)]);

    try {
      const client = new ArtifactClient(this.session);
      const artifacts = await client.list(run.run_id, signal);

      const candidates = artifacts.filter(
        (item) => item.attempt === run.attempt && item.path === 'checkpoint_selection.json',
      );

      candidates.sort((left, right) => {
        const a = BigInt(left.generation.slice(4)),
          b = BigInt(right.generation.slice(4));

        return a < b ? 1 : a > b ? -1 : 0;
      });
      const publications: CheckpointOption[] = [];
      let incompleteBest = false;

      for (const candidate of candidates) {
        if (!this.#current(signal)) return;
        let options: readonly CheckpointOption[];

        try {
          options = await checkpointOptions(this.session, [candidate], signal);
        } catch {
          // Hash-covered malformed metadata can be skipped. Transport, authority
          // and byte-integrity faults revoke the connection, so the scope check
          // below prevents falling back after those failures.
          if (!this.#current(signal)) return;
          continue;
        }

        const option = options[0];

        if (option?.classification.kind === 'best') {
          if (!hasInferenceSource(artifacts, candidate.generation)) {
            incompleteBest = true;
            continue;
          }

          publications.push(option);
          break;
        }
      }

      if (!this.#current(signal)) return;

      if (!publications.length && incompleteBest) {
        this.stop();
        this.state.value = {
          status: 'unavailable',
          plotRunId: null,
          message: clientError('autoPlotCheckpointIncomplete', {}),
        };

        return;
      }

      this.#intent = this.#deriveIntent(run, publications[0]);

      if (!this.#intent) return;
      const selected = publications[0];

      if (!selected) return;
      this.#sourceFiles = artifacts.filter(
        (item) => item.generation === selected.artifact.generation,
      );
      this.#sourceSelection = selected.classification;
      this.#sourceAttempt = run.attempt;

      // Re-read lifecycle after metadata downloads; it may have requeued meanwhile.
      const latest = await this.session.perform((api, scoped) => api.getRun(run.run_id, scoped), {
        signal,
      });

      if (!this.#current(signal)) return;

      if (
        latest.run_id !== run.run_id ||
        latest.plan_id !== run.plan_id ||
        latest.request_id !== run.request_id ||
        latest.backend !== run.backend ||
        latest.operation !== 'train' ||
        latest.cancellation_requested ||
        latest.attempt !== run.attempt ||
        latest.state !== 'completed' ||
        latest.uncertainty
      ) {
        this.stop();

        return;
      }

      for (const option of publications) client.assertSelection(option.artifact);
      await this.#execute(signal);
    } catch {
      this.#error(parent);
    } finally {
      if (this.#scope.signal === parent) this.#busy = false;
    }
  }
  #deriveIntent(run: RunView, selected: CheckpointOption | undefined): AutoPlotIntent | null {
    const captured = this.#request;
    const sourceRun = this.#run;

    if (
      !captured ||
      !sourceRun ||
      !selected?.selectable ||
      selected.classification.kind !== 'best' ||
      selected.artifact.runId !== run.run_id ||
      selected.artifact.attempt !== run.attempt ||
      captured.config.cv > 0 ||
      run.cancellation_requested ||
      run.backend !== captured.backend
    )
      return null;
    // observeFresh has already selected the newest complete, verified best.
    // Once this intent exists, retries execute it directly with the same key.
    const source = selected.artifact;
    const config = captured.config;

    const request: ResultRequest = {
      backend: captured.backend,
      operation: 'plot',
      runtime: captured.runtime,
      topology: { num_machines: 1, processes_per_machine: 1, machine_rank: 0 },
      checkpoint_source: { run_id: run.run_id, generation: source.generation },
      config: {
        ...inferenceDefaults,
        data_path: config.data_path,
        model: config.model,
        import_modules: config.import_modules,
        batch_size: config.batch_size,
        workers: Math.max(0, config.workers),
        plot: true,
      },
    };

    if (captured.storage_map !== undefined) request.storage_map = captured.storage_map;

    return {
      request: resultRequestSchema.parse(request),
      idempotencyKey: `web-autoplot-v1:${sourceRun.run_id}:${sourceRun.request_id}:${source.generation}`,
    };
  }
  async #execute(signal: AbortSignal): Promise<void> {
    if (!this.#intent || !this.#current(signal)) return;
    this.state.value = { ...this.state.value, status: 'planning' };
    const intent = this.#intent;

    if (!this.#plan) {
      const plan = await this.session.perform(
        (api, scoped) => api.plan(intent.request, intent.idempotencyKey, scoped),
        { signal, timeoutMs: this.options.deadlineMs ?? 15000 },
      );

      if (!this.#current(signal)) return;

      if (!(await this.#validatePlotPlan(plan, intent, signal)) || !this.#current(signal)) return;
      this.#plan = plan;
    } else {
      const plan = this.#plan;

      const observed = await this.session.perform(
        (api, scoped) => api.getRun(plan.run_id, scoped),
        { signal },
      );

      if (!this.#current(signal)) return;

      if (observed.plan_id !== plan.plan_id || observed.request_id !== plan.request_id)
        throw clientError('automaticPlotIdentityChanged', {});

      if (observed.submission_requested) {
        this.#complete(observed);

        return;
      }
    }

    if (!this.#current(signal)) return;
    const plan = this.#plan;
    const source = this.#run;

    if (!source) return;

    const sourceNow = await this.session.perform(
      (api, scoped) => api.getRun(source.run_id, scoped),
      { signal, timeoutMs: this.options.deadlineMs ?? 15000 },
    );

    if (!this.#current(signal)) return;

    if (
      sourceNow.plan_id !== source.plan_id ||
      sourceNow.request_id !== source.request_id ||
      sourceNow.attempt !== this.#sourceAttempt ||
      sourceNow.state !== 'completed' ||
      sourceNow.uncertainty ||
      sourceNow.cancellation_requested
    ) {
      this.stop();

      return;
    }

    if (!this.#current(signal)) return;

    const result = await this.session.perform((api, scoped) => api.submit(plan, scoped), {
      signal,
      timeoutMs: this.options.deadlineMs ?? 15000,
    });

    if (this.#current(signal)) this.#complete(result);
  }
  async #validatePlotPlan(
    plan: PlanView,
    intent: AutoPlotIntent,
    signal: AbortSignal,
  ): Promise<boolean> {
    const parent = this.#scope.signal;
    const sourcePlan = this.#sourcePlan;
    const sourceApi = this.#sourceApi;

    const files = Object.fromEntries(
      this.#sourceFiles.map((file) => [file.path, { size: file.size, sha256: file.sha256 }]),
    );

    const selection = this.#sourceSelection ? { ...this.#sourceSelection } : null;

    const current = (): boolean =>
      this.#scope.signal === parent &&
      !signal.aborted &&
      this.#sourcePlan === sourcePlan &&
      this.#sourceApi === sourceApi &&
      this.#current(signal);

    try {
      if (!current()) return false;

      if (!sourcePlan || sourcePlan.preview.spec === undefined || plan.preview.spec === undefined)
        throw clientError('missingApprovedManifests', {});
      const parsedSource = sourcePlotManifestSchema.safeParse(sourcePlan.preview.spec);
      const parsedPlot = approvedPlotManifestSchema.safeParse(plan.preview.spec);

      if (!parsedSource.success || !parsedPlot.success)
        throw clientError('invalidPlotManifest', {});
      const source = parsedSource.data;
      const plot = parsedPlot.data;
      const pin = plot.checkpoint_source;

      const digest = Array.from(
        new Uint8Array(
          await crypto.subtle.digest(
            'SHA-256',
            new TextEncoder().encode(`${JSON.stringify(sourcePlan.preview.spec)}\n`),
          ),
        ),
        (byte) => byte.toString(16).padStart(2, '0'),
      ).join('');

      if (!current()) return false;

      if (
        plot.run_id !== plan.run_id ||
        plan.preview.target !== intent.request.backend ||
        canonical(plan.preview.topology ?? null) !== canonical(intent.request.topology) ||
        canonical(plot.runtime ?? null) !== canonical(source.runtime ?? null) ||
        canonical(plot.storage_map ?? []) !== canonical(source.storage_map ?? []) ||
        plot.resolved_config.data_path !== source.resolved_config.data_path ||
        canonical(plot.resolved_config.import_modules ?? []) !==
          canonical(source.resolved_config.import_modules ?? []) ||
        pin.run_id !== intent.request.checkpoint_source.run_id ||
        pin.generation !== intent.request.checkpoint_source.generation ||
        pin.manifest_sha256 !== digest ||
        canonical(pin.files ?? null) !== canonical(files) ||
        canonical(pin.selection ?? null) !== canonical(selection)
      )
        throw clientError('plotAuthorityChanged', {});

      for (const [key, expected] of Object.entries(intent.request.config)) {
        if (
          key !== 'data_path' &&
          key !== 'import_modules' &&
          canonical(plot.resolved_config[key] ?? null) !== canonical(expected)
        )
          throw clientError('plotIntentChanged', {});
      }

      return true;
    } catch {
      if (!current()) return false;
      this.stop();
      this.state.value = {
        status: 'unavailable',
        plotRunId: null,
        message: clientError('autoPlotAuthorityMismatch', {}),
      };

      return false;
    }
  }
  #complete(run: RunView): void {
    if (
      !this.#plan ||
      run.run_id !== this.#plan.run_id ||
      run.plan_id !== this.#plan.plan_id ||
      run.request_id !== this.#plan.request_id ||
      run.operation !== 'plot' ||
      run.backend !== this.#run?.backend ||
      !run.submission_requested
    )
      throw clientError('autoPlotAckMismatch', {});
    this.state.value = {
      status: 'submitted',
      plotRunId: run.run_id,
      message: clientError('autoPlotAcknowledged', {}),
    };

    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }
  async retry(): Promise<void> {
    if (
      this.#busy ||
      this.state.value.status !== 'error' ||
      !this.#run ||
      !this.#current(this.#scope.signal)
    )
      return;

    if (!this.#intent) {
      this.state.value = { ...this.state.value, status: 'armed' };
      await this.#poll();

      return;
    }

    this.#busy = true;
    const parent = this.#scope.signal;
    const signal = AbortSignal.any([parent, AbortSignal.timeout(this.options.deadlineMs ?? 15000)]);

    try {
      const source = this.#run;

      if (!source) return;

      const current = await this.session.perform(
        (api, scoped) => api.getRun(source.run_id, scoped),
        { signal },
      );

      if (
        current.plan_id !== source.plan_id ||
        current.request_id !== source.request_id ||
        current.attempt !== this.#sourceAttempt ||
        current.state !== 'completed' ||
        current.uncertainty ||
        current.cancellation_requested
      ) {
        this.stop();

        return;
      }

      await this.#execute(signal);
    } catch {
      this.#error(parent);
    } finally {
      if (this.#scope.signal === parent) this.#busy = false;
    }
  }
  stop(): void {
    this.#scope.abort();
    this.#request = null;
    this.#run = null;
    this.#intent = null;
    this.#sourceAttempt = null;
    this.#observedAttempt = null;
    this.#plan = null;
    this.#sourcePlan = null;
    this.#sourceApi = null;
    this.#sourceFiles = [];
    this.#sourceSelection = null;
    this.#busy = false;

    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.state.value = { status: 'stopped', plotRunId: null, message: '' };
  }
  dispose(): void {
    this.stop();
    this.#unsubscribe();
  }
}
