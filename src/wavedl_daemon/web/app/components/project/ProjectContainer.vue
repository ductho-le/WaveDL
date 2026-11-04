<script setup lang="ts">
import { cn } from '@/lib/utils';
import { useForm } from 'vee-validate';
import {
  ArrowRight,
  Check,
  CircleAlert,
  Info,
  OctagonAlert,
  TriangleAlert,
  Undo2,
  X,
} from '@lucide/vue';
import { clientError } from '../../i18n/client-messages';
import { automaticRuntime, hostAccelerator, matchesAutomatic } from '../../lib/host-runtime';
import type { RuntimeSetup } from '../../../shared/schemas/host-profile';
import {
  projectBlockMessageKeys,
  runStateMessageKeys,
  trainingChoiceLabel,
  trainingFieldMessageKeys,
  trainingPresetMessageKeysFor,
} from '../../i18n/dynamic-keys';
import {
  trainingDefaults,
  trainingRequestSchema,
  type TrainingRequest,
} from '../../../shared/schemas/training-request';
import {
  projectBlocks,
  type ProjectBlock,
  type ProjectDraft,
} from '../../../shared/schemas/project-draft';
import { ConnectionSessionError, type ConnectionSession } from '../../lib/connection-session';
import { applyTrainingPreferences, loadTrainingPreferences } from '../../lib/preferences';
import type { PlanView, RunView } from '../../lib/daemon-api';
import {
  CatalogApi,
  type DataSelection,
  type ModelDescription,
  type ModelSelection,
} from '../../lib/catalog-api';
import { ProjectPlan } from '../../lib/project-plan';
import type { DatasetInspection } from '../../lib/dataset-inspection';
import { modelShapeGuidance } from '../../lib/model-shape-guidance';
import { applyTrainingPreset } from '../../lib/training-presets';
import {
  changedFromDefault,
  compactNumber,
  essentialTrainingFields,
  matchingPreset,
  trainingRecipes,
  trainingSections,
  type TrainingFieldName,
} from '../../lib/training-layout';
import { CONFIG_IMPORT_LIMIT, parseTrainingConfig } from '../../lib/training-import';
import {
  clearProjectDraft,
  draftSetup,
  draftRequest,
  loadProjectDraft,
  projectDraft,
  saveProjectDraft,
  validDraftConfig,
} from '../../lib/project-draft';
import { draftConfigFields } from '../../../shared/schemas/project-draft';
import type { ReadinessItem } from './PlanInspector.vue';
import type { PipelineBlock } from './RunPipeline.vue';
import type { DraftSummary } from './StartPanel.vue';

const { locale, t } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();

function stateLabel(state: RunView['state']): string {
  return locale.value === 'en' ? state : t(runStateMessageKeys[state]);
}

const props = defineProps<{ session: ConnectionSession }>();

const emit = defineEmits<{
  submitted: [run: RunView];
  submittedPlan: [run: RunView, plan: PlanView];
}>();

const panelId = useId();
const importInput = useTemplateRef<HTMLInputElement>('importInput');

function daemonId(): string | null {
  return toRaw(props.session).snapshot().health?.daemon_id ?? null;
}

function defaults(): TrainingRequest {
  const id = daemonId();

  return {
    backend: 'local',
    operation: 'train',
    config: id
      ? applyTrainingPreferences(structuredClone(trainingDefaults), loadTrainingPreferences(id))
      : structuredClone(trainingDefaults),
    runtime: { python: '', mode: 'existing', device: 'cpu', environment: {} },
    topology: { num_machines: 1, processes_per_machine: 1, machine_rank: 0 },
  };
}

const { handleSubmit, values, setFieldValue, resetForm, validate } = useForm<TrainingRequest>({
  initialValues: defaults(),
  validationSchema: trainingRequestSchema,
  keepValuesOnUnmount: true,
});

const catalogue = useModelCatalogue(() => props.session);
const host = useHostProfile(() => props.session);

/** Automatic plans the detected Local setup; Manual leaves every runtime value to the user. */
const setup = shallowRef<RuntimeSetup>('automatic');

const data = shallowRef<DataSelection | null>(null);
const datasetInspection = shallowRef<DatasetInspection | null>(null);
const model = shallowRef<ModelSelection | null>(null);
const plan = shallowRef<PlanView | null>(null);
const acknowledged = shallowRef<RunView | null>(null);
const busy = shallowRef(false);
const message = shallowRef('');
const reportedRun = shallowRef<RunView | null>(null);
const connected = shallowRef(false);
const active = shallowRef<ProjectBlock>('dataset');

/** A saved draft offered on the start panel; null once the user chooses how to begin. */
const offered = shallowRef<ProjectDraft | null>(null);
const pendingModel = shallowRef<string | null>(null);
const dragging = shallowRef(false);
const savedAt = shallowRef<number | null>(null);

/** Bumped whenever a connection scope retires; inspectors drop transient UI state. */
const scope = shallowRef(0);

/** Title and detail are read at render time so a visible notice follows the locale. */
interface Notice {
  readonly id: number;
  readonly tone: 'success' | 'info' | 'warning';
  readonly title: () => string;
  readonly detail?: () => string;
  readonly undo?: () => void;
}

const notice = shallowRef<Notice | null>(null);
let noticeTimer: ReturnType<typeof setTimeout> | undefined;

function announce(value: Omit<Notice, 'id'>) {
  clearTimeout(noticeTimer);
  notice.value = { ...value, id: Date.now() };
  noticeTimer = setTimeout(() => (notice.value = null), value.undo ? 12000 : 6000);
}

const displayMessage = computed(() =>
  message.value === 'reported' && reportedRun.value
    ? t('project.daemonReportsStateSubmission', {
        state: stateLabel(reportedRun.value.state),
        submission: t(
          reportedRun.value.submission_requested
            ? 'project.submissionRequested'
            : 'project.notSubmittedYet',
        ),
      })
    : message.value === 'project.submissionAckUnknown'
      ? t('project.submissionOutcomeUnknown')
      : message.value.startsWith('project.')
        ? t(message.value)
        : clientMessage(message.value),
);

/** Messages that block planning read as errors; an unknown outcome is a caution. */
const blockingMessages = [
  'project.planValidationRequired',
  'project.selectionRequired',
  'project.selectionStale',
  'project.planRefused',
];

const messageTone = computed(() =>
  blockingMessages.includes(message.value)
    ? 'destructive'
    : message.value === 'project.submissionAckUnknown'
      ? 'warning'
      : 'default',
);

/** The blocks a blocked plan points at, each one click from its fix. */
const blockedBlocks = computed<ProjectBlock[]>(() => {
  if (messageTone.value !== 'destructive') return [];
  const pending = new Set<ProjectBlock>(issues.value);

  if (!data.value) pending.add('dataset');

  if (!model.value) pending.add('model');

  return projectBlocks.filter((block) => pending.has(block));
});

let workflow = new ProjectPlan(props.session);
let pending: AbortController | null = null;
let epoch = 0;
let unsubscribe: (() => void) | undefined;

/** Set after the daemon acknowledges; the next edit makes the draft worth saving again. */
let submittedDraft = false;

function invalidate() {
  epoch++;
  pending?.abort();
  pending = null;
  busy.value = false;
  plan.value = null;
  workflow.invalidate();
}

function retireScope() {
  scope.value++;
  invalidate();
  data.value = null;
  datasetInspection.value = null;
  model.value = null;
  pendingModel.value = null;
  acknowledged.value = null;
  reportedRun.value = null;
  message.value = '';
}

watch(
  values,
  () => {
    invalidate();
    submittedDraft = false;
  },
  { deep: true, flush: 'sync' },
);

watch(
  () => props.session,
  (session) => {
    unsubscribe?.();
    retireScope();
    workflow = new ProjectPlan(session);
    let revision = -1;
    unsubscribe = toRaw(session).subscribe((state) => {
      if (toRaw(props.session) !== toRaw(session)) return;
      const wasConnected = connected.value;
      connected.value = state.status === 'connected';
      const fresh = state.revision !== revision;

      if (fresh || !connected.value) {
        revision = state.revision;
        retireScope();
      }

      // The revision bumps while still connecting, so offer on the transition to connected:
      // a verified connection offers the saved draft rather than silently replacing it.
      if (connected.value && (fresh || !wasConnected)) {
        const id = daemonId();
        offered.value = id ? loadProjectDraft(id) : null;
      } else if (!connected.value) offered.value = null;
    });
  },
  { immediate: true, flush: 'sync' },
);

// ---- Selections -------------------------------------------------------------------------

function selectData(selection: DataSelection) {
  try {
    new CatalogApi(toRaw(props.session)).assertSelection(selection);
  } catch {
    return;
  }

  data.value = selection;
  datasetInspection.value = null;
  setFieldValue('config.data_path', selection.path);

  if (active.value === 'dataset' && !model.value) active.value = 'model';
}

function inspectData(inspection: DatasetInspection | null, selection: DataSelection | null) {
  if (!selection || selection !== toRaw(data.value)) return;

  try {
    new CatalogApi(toRaw(props.session)).assertSelection(selection);
  } catch {
    return;
  }

  datasetInspection.value = inspection;
}

function selectModel(selection: ModelSelection) {
  try {
    new CatalogApi(toRaw(props.session)).assertSelection(selection);
  } catch {
    return;
  }

  model.value = selection;
  pendingModel.value = null;
  setFieldValue('config.model', selection.model);
}

function chooseModel(description: ModelDescription, pretrained: boolean) {
  const selection = catalogue.select(description);

  if (!selection) return;
  selectModel(selection);
  setFieldValue('config.pretrained', pretrained);
}

// Draft restore and import name a model before the catalogue may have loaded.
watch([() => catalogue.models.value, pendingModel], ([models, id]) => {
  if (!id || !models.length) return;
  const description = models.find((item) => item.id === id);
  const selection = description ? catalogue.select(description) : null;

  if (selection) selectModel(selection);
  else {
    pendingModel.value = null;
    announce({ tone: 'warning', title: () => t('builder.modelMissing', { model: id }) });
  }
});

function applyPreset(id: string) {
  const recipe = trainingRecipes.find((item) => item.id === id);

  if (recipe) setFieldValue('config', applyTrainingPreset(values.config, recipe.config));
}

function resetTraining() {
  const config = structuredClone(trainingDefaults);
  config.data_path = data.value?.path ?? config.data_path;
  config.model = model.value?.model ?? config.model;
  config.pretrained = values.config.pretrained;
  resetForm({ values: { ...values, config } });
}

// ---- Runtime setup ----------------------------------------------------------------------

/** Writes only what differs, so an unchanged setup never revokes an approved plan. */
function applyDetected() {
  const profile = host.profile.value;

  if (!profile || matchesAutomatic(values, profile)) return;
  const expected = automaticRuntime(profile);

  if (values.backend !== expected.backend) setFieldValue('backend', expected.backend);

  if (JSON.stringify(values.runtime) !== JSON.stringify(expected.runtime))
    setFieldValue('runtime', expected.runtime);

  if (JSON.stringify(values.topology) !== JSON.stringify(expected.topology))
    setFieldValue('topology', expected.topology);

  if (values.storage_map?.length) setFieldValue('storage_map', []);
}

// Automatic follows the profile and repairs any reset, restore or import that moved it.
watch(
  () =>
    setup.value === 'automatic' &&
    !!host.profile.value &&
    !matchesAutomatic(values, host.profile.value),
  (drifted) => {
    if (drifted) applyDetected();
  },
  { immediate: true },
);

// ---- Draft persistence ------------------------------------------------------------------

/** Automatic runtime values are detected, not authored, so they alone are not worth saving. */
function meaningful(): boolean {
  const fresh = defaults();
  const { config, ...rest } = toRaw(values);
  const { config: freshConfig, ...freshRest } = fresh;

  return (
    !!data.value ||
    !!model.value ||
    JSON.stringify(config) !== JSON.stringify(freshConfig) ||
    (setup.value === 'manual' && JSON.stringify(rest) !== JSON.stringify(freshRest))
  );
}

let saveTimer: ReturnType<typeof setTimeout> | undefined;

watch(
  [values, data, model, active, setup],
  () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const id = daemonId();

      // Never overwrite a draft the user has not yet chosen to continue or discard.
      if (!id || offered.value || submittedDraft || !connected.value) return;

      if (!meaningful()) return;

      if (
        saveProjectDraft(id, {
          block: active.value,
          data: data.value,
          model: model.value,
          request: values,
          setup: setup.value,
        })
      )
        savedAt.value = Date.now();
    }, 450);
  },
  { deep: true },
);

async function restoreSelections(
  dataset: ProjectDraft['dataset'],
  chosen: ProjectDraft['model'],
): Promise<string[]> {
  const current = epoch;
  const api = new CatalogApi(toRaw(props.session));
  const missing: string[] = [];

  if (dataset) {
    try {
      const selection = await api.resolve(dataset.root, dataset.relativePath);

      if (current === epoch && selection.kind === 'file') selectData(selection);
    } catch {
      missing.push(dataset.relativePath);
    }
  }

  if (chosen?.custom) {
    try {
      const selection = await api.selectCustomModel(chosen.id);

      if (current === epoch) selectModel(selection);
    } catch {
      missing.push(chosen.id);
    }
  } else if (chosen) pendingModel.value = chosen.id;

  return missing;
}

async function continueDraft() {
  const draft = offered.value;

  if (!draft) return;
  offered.value = null;
  setup.value = draftSetup(draft);
  resetForm({ values: draftRequest(draft, defaults()) });
  const missing = await restoreSelections(draft.dataset, draft.model);
  active.value = !data.value ? 'dataset' : !draft.model ? 'model' : draft.block;

  if (missing.length)
    announce({
      tone: 'warning',
      title: () => t('builder.restoreIncomplete'),
      detail: () => missing.join(', '),
    });
}

function startFresh() {
  const id = daemonId();

  if (id) clearProjectDraft(id);
  offered.value = null;
  retireScope();
  setup.value = 'automatic';
  resetForm({ values: defaults() });
  active.value = 'dataset';
  savedAt.value = null;
}

/** After a submission: keep every choice, ready to swap the model and go again. */
function another() {
  invalidate();
  acknowledged.value = null;
  reportedRun.value = null;
  message.value = '';
  submittedDraft = false;
  active.value = 'model';
}

// ---- Import -----------------------------------------------------------------------------

function openImport() {
  importInput.value?.click();
}

async function importFile(file: File) {
  if (file.size > CONFIG_IMPORT_LIMIT) {
    announce({
      tone: 'warning',
      title: () => clientMessage(clientError('configImportTooLarge')),
    });

    return;
  }

  let imported: ReturnType<typeof parseTrainingConfig>;

  try {
    imported = parseTrainingConfig(await file.text());
  } catch (error) {
    announce({
      tone: 'warning',
      title: () => t('builder.importFailed', { file: file.name }),
      detail: () => clientMessage(error instanceof Error ? error : String(error)),
    });

    return;
  }

  const id = daemonId();

  // Field-wise snapshot (no deep clone of reactive state); environment values are copied.
  const before = id
    ? {
        draft: projectDraft(id, {
          block: active.value,
          data: data.value,
          model: model.value,
          request: values,
          setup: setup.value,
        }),
        environment: { ...values.runtime.environment },
        data: data.value,
        model: model.value,
        inspection: datasetInspection.value,
      }
    : null;

  offered.value = null;
  // Fields that choose which code or checkpoint the run executes never arrive by file:
  // like presets, an import cannot retarget executable code or checkpoints.
  const withheld = codeFields.filter((name) => imported.config[name] !== undefined);

  const applied = validDraftConfig(
    Object.fromEntries(
      Object.entries(imported.config).filter(([name]) => !withheld.some((code) => code === name)),
    ),
  );

  const next = {
    ...values.config,
    ...applied,
    data_path: values.config.data_path,
    model: values.config.model,
  };

  // An import that changes nothing must not revoke an approved plan.
  if (JSON.stringify(next) !== JSON.stringify(values.config)) setFieldValue('config', next);
  const missing: string[] = [];

  if (imported.dataPath) {
    try {
      const selection = await new CatalogApi(toRaw(props.session)).locate(imported.dataPath);

      if (selection?.kind === 'file') selectData(selection);
      else missing.push(imported.dataPath);
    } catch {
      missing.push(imported.dataPath);
    }
  }

  if (imported.model) {
    const known = catalogue.models.value.find((item) => item.id === imported.model);

    if (known) chooseModel(known, imported.config.pretrained ?? values.config.pretrained);
    else if (catalogue.models.value.length) {
      try {
        selectModel(await new CatalogApi(toRaw(props.session)).selectCustomModel(imported.model));
      } catch {
        missing.push(imported.model);
      }
    } else pendingModel.value = imported.model;
  }

  active.value = data.value ? 'training' : 'dataset';
  // Fields that choose which code or checkpoint the run executes are never applied quietly.
  announce({
    tone: missing.length || withheld.length ? 'warning' : 'success',
    title: () => t('builder.imported', { file: file.name, count: Object.keys(applied).length }),
    detail: () =>
      [
        withheld.length
          ? t('builder.importCode', { keys: withheld.map(fieldLabel).join(', ') })
          : '',
        missing.length ? t('builder.importMissing', { items: missing.join(', ') }) : '',
        imported.skipped.length
          ? t('builder.importSkipped', { keys: imported.skipped.join(', ') })
          : '',
      ]
        .filter(Boolean)
        .join(' '),
    undo: before
      ? () => {
          const restored = draftRequest(before.draft, defaults());
          resetForm({
            values: {
              ...restored,
              config: {
                ...restored.config,
                data_path: before.data?.path ?? restored.config.data_path,
                model: before.model?.model ?? restored.config.model,
              },
              runtime: { ...restored.runtime, environment: before.environment },
            },
          });
          data.value = before.data;
          model.value = before.model;
          datasetInspection.value = before.inspection;
          notice.value = null;
        }
      : undefined,
  });
}

const codeFields: readonly TrainingFieldName[] = [
  'import_modules',
  'constraint',
  'constraint_file',
  'resume',
];

function pickedFile(event: Event) {
  if (!(event.target instanceof HTMLInputElement)) return;
  const file = event.target.files?.[0];
  event.target.value = '';

  if (file) void importFile(file);
}

let dragDepth = 0;

function hasFiles(event: DragEvent): boolean {
  return !!event.dataTransfer?.types.includes('Files');
}

function dragEnter(event: DragEvent) {
  if (!hasFiles(event)) return;
  dragDepth++;
  dragging.value = true;
}

function dragLeave(event: DragEvent) {
  if (!hasFiles(event)) return;
  dragDepth = Math.max(0, dragDepth - 1);

  if (!dragDepth) dragging.value = false;
}

function drop(event: DragEvent) {
  dragDepth = 0;
  dragging.value = false;
  const file = event.dataTransfer?.files[0];

  if (file && connected.value) void importFile(file);
}

function exportNotice(value: string | Error, ok: boolean, detail?: string) {
  announce({
    tone: ok ? 'success' : 'warning',
    detail: detail ? () => t(detail) : undefined,
    title: () =>
      !(value instanceof Error) && value.startsWith('project.') ? t(value) : clientMessage(value),
  });
}

// ---- Planning workflow (unchanged semantics) ---------------------------------------------

type WorkOperation = 'preview' | 'submit' | 'status';

async function work(
  operation: WorkOperation,
  action: (signal: AbortSignal, isCurrent: () => boolean) => Promise<void>,
) {
  const current = ++epoch;
  pending?.abort();
  pending = new AbortController();
  busy.value = true;
  message.value = '';

  try {
    await action(pending.signal, () => current === epoch);
  } catch (error) {
    if (current === epoch) {
      message.value =
        operation === 'preview' &&
        error instanceof ConnectionSessionError &&
        error.code === 'request' &&
        error.status === 400
          ? 'project.planRefused'
          : 'project.submissionAckUnknown';
    }
  } finally {
    if (current === epoch) {
      busy.value = false;
      pending = null;
    }
  }
}

function preview(event?: Event) {
  const current = epoch;

  return handleSubmit(
    async (draft) => {
      if (current !== epoch) return;

      if (!data.value || !model.value) {
        message.value = 'project.selectionRequired';

        return;
      }

      try {
        workflow.prepare(draft, data.value, model.value);
      } catch {
        message.value = 'project.selectionStale';

        return;
      }

      await work('preview', async (signal, isCurrent) => {
        const result = await workflow.preview(signal);

        if (!isCurrent()) return;
        plan.value = result;
        active.value = 'plan';
      });
    },
    ({ errors }) => {
      if (current !== epoch) return;
      message.value = 'project.planValidationRequired';
      const first = Object.keys(errors)[0];

      if (first) active.value = blockFor(first.split('.'));
    },
  )(event);
}

function submit() {
  return work('submit', async (signal, isCurrent) => {
    const approved = workflow.approved;
    const run = await workflow.submit(signal);

    if (!isCurrent()) return;
    acknowledged.value = run;
    message.value = 'project.daemonAcknowledgedThisExactPlan';
    const id = daemonId();

    if (id) clearProjectDraft(id);
    submittedDraft = true;
    emit('submitted', run);

    if (approved && isCurrent()) emit('submittedPlan', run, approved);
  });
}

function status() {
  return work('status', async (signal, isCurrent) => {
    const approved = workflow.approved;
    const run = await workflow.status(signal);

    if (!isCurrent()) return;
    reportedRun.value = run;
    message.value = 'reported';

    if (run.submission_requested) {
      acknowledged.value = run;
      emit('submitted', run);

      if (approved && isCurrent()) emit('submittedPlan', run, approved);
    }
  });
}

function openAcknowledged() {
  if (connected.value && acknowledged.value) emit('submitted', acknowledged.value);
}

function selectBlock(block: ProjectBlock) {
  // The start panel asks one question first; the strip only previews the draft.
  if (!offered.value) active.value = block;
}

/** The plan on screen is the one the daemon acknowledged; an edit starts a new plan. */
const planSubmitted = computed(
  () => !!plan.value && acknowledged.value?.request_id === plan.value.request_id,
);

function review() {
  if (planSubmitted.value) {
    openAcknowledged();

    return;
  }

  active.value = 'plan';

  if (plan.value) return;

  if (canPreview.value) {
    void preview();

    return;
  }

  const current = epoch;
  void validate().then((result) => {
    if (current !== epoch) return;

    if (!result.valid) message.value = 'project.planValidationRequired';
    else if (!data.value || !model.value) message.value = 'project.selectionRequired';
  });
}

// ---- Derived presentation ---------------------------------------------------------------

function blockFor(path: readonly PropertyKey[]): ProjectBlock {
  const [root, field] = path;

  if (root === 'config')
    return field === 'data_path'
      ? 'dataset'
      : field === 'model' || field === 'pretrained'
        ? 'model'
        : 'training';

  return 'runtime';
}

const issues = computed(() => {
  const parsed = trainingRequestSchema.safeParse(values);

  return parsed.success ? [] : parsed.error.issues.map((issue) => blockFor(issue.path));
});

function issueCount(block: ProjectBlock): number {
  return issues.value.filter((item) => item === block).length;
}

const canPreview = computed(
  () => connected.value && !!data.value && !!model.value && !issues.value.length && !busy.value,
);

const catalogueModel = computed(() =>
  catalogue.models.value.find((item) => item.id === model.value?.model),
);

const shapeStatus = computed(() =>
  catalogueModel.value
    ? modelShapeGuidance(
        datasetInspection.value,
        catalogueModel.value,
        values.config.single_channel,
      ).status
    : 'unknown',
);

function fieldLabel(name: TrainingFieldName): string {
  const keys = trainingFieldMessageKeys[name];

  return keys ? t(keys.label) : name;
}

const presetLabel = computed(() => {
  const id = matchingPreset(values.config);

  if (id === 'defaults') return t('builder.defaultsShort');
  const keys = id ? trainingPresetMessageKeysFor(id) : null;

  return keys ? t(keys.label) : t('builder.customRecipe');
});

// `pretrained` belongs to the Model block, so it is not a Training change.
const trainingFields = [
  ...essentialTrainingFields,
  ...trainingSections.flatMap((section) => section.fields),
];

const changedCount = computed(
  () => trainingFields.filter((name) => changedFromDefault(name, values.config)).length,
);

const deviceLabels: Record<TrainingRequest['runtime']['device'], string> = {
  cpu: 'builder.deviceCpu',
  cuda: 'builder.deviceCuda',
  mps: 'builder.deviceMps',
  auto: 'builder.deviceAuto',
};

const datasetName = computed(
  () => data.value?.path.split(/[\\/]/).filter(Boolean).at(-1) ?? t('builder.nothingSelected'),
);

const modelName = computed(
  () =>
    catalogueModel.value?.displayName.replace(/ \(Pretrained\)$/, '') ??
    model.value?.model ??
    (pendingModel.value ? pendingModel.value : t('builder.nothingSelected')),
);

const pythonName = computed(() => {
  const python = values.runtime.python.trim();

  return python
    ? python.split(/[\\/]/).filter(Boolean).slice(-3).join('/')
    : t('builder.pythonNeeded');
});

const automaticShown = computed(() => setup.value === 'automatic' && !!host.profile.value);

const acceleratorLabel = computed(() => {
  const profile = host.profile.value;

  if (!profile) return '';
  const kind = hostAccelerator(profile);

  if (kind === 'cuda') return profile.accelerators?.cuda[0]?.name ?? t('builder.deviceCuda');

  if (kind === 'mps') return t('builder.appleGpu');

  return kind === 'cpu' ? t('builder.deviceCpu') : t('builder.deviceAuto');
});

const blocks = computed<PipelineBlock[]>(() => [
  {
    id: 'dataset',
    label: t(projectBlockMessageKeys.dataset),
    value: datasetName.value,
    detail: datasetInspection.value
      ? t('builder.datasetDetail', {
          samples: datasetInspection.value.numSamples.toLocaleString(),
          shape: datasetInspection.value.inputShape.join(' × '),
        })
      : data.value
        ? data.value.root
        : t('builder.chooseDataset'),
    state: data.value ? 'complete' : 'pending',
  },
  {
    id: 'model',
    label: t(projectBlockMessageKeys.model),
    value: modelName.value,
    detail: model.value
      ? model.value.source === 'custom'
        ? t('builder.customModelDetail')
        : t(values.config.pretrained ? 'builder.withWeights' : 'builder.fromScratch', {
            params: catalogueModel.value?.paramsM ?? '—',
          })
      : t('builder.chooseModel'),
    state: !model.value
      ? 'pending'
      : model.value.source === 'custom' || shapeStatus.value === 'mismatch' || issueCount('model')
        ? 'attention'
        : 'complete',
  },
  {
    id: 'training',
    label: t(projectBlockMessageKeys.training),
    value: presetLabel.value,
    detail: [
      t('builder.epochsShort', { epochs: values.config.epochs }),
      trainingChoiceLabel('optimizer', values.config.optimizer, t),
      changedCount.value ? t('project.nonDefaultFields', { count: changedCount.value }) : '',
    ]
      .filter(Boolean)
      .join(' · '),
    state: issueCount('training') ? 'attention' : 'complete',
  },
  {
    id: 'runtime',
    label: t(projectBlockMessageKeys.runtime),
    value: automaticShown.value
      ? acceleratorLabel.value
      : setup.value === 'automatic' && host.busy.value
        ? t('builder.detecting')
        : t(deviceLabels[values.runtime.device]),
    detail: automaticShown.value
      ? `${t('builder.setupAutomatic')} · ${host.profile.value?.hostname || t('builder.thisMachine')}`
      : `${t(
          values.backend === 'slurm' ? 'builder.backendSlurm' : 'builder.backendLocal',
        )} · ${pythonName.value}`,
    state: issueCount('runtime') ? 'pending' : 'complete',
  },
  {
    id: 'plan',
    label: t(projectBlockMessageKeys.plan),
    value: planSubmitted.value
      ? t('builder.submitted')
      : plan.value
        ? t('builder.planReady')
        : canPreview.value
          ? t('builder.readyToPlan')
          : t('builder.planPending'),
    detail:
      planSubmitted.value && acknowledged.value
        ? stateLabel(acknowledged.value.state)
        : plan.value
          ? t('builder.awaitingApproval')
          : t('builder.nothingStarts'),
    state: plan.value ? 'complete' : 'pending',
  },
]);

const reviewLabel = computed(() =>
  planSubmitted.value
    ? t('project.openAcknowledgedRun')
    : plan.value
      ? t('builder.reviewPlan')
      : t('builder.createPlan'),
);

const readiness = computed<ReadinessItem[]>(() =>
  blocks.value.flatMap((block) =>
    block.id !== 'plan'
      ? [
          {
            block: block.id,
            label: block.label,
            detail: issueCount(block.id)
              ? t('builder.issueCount', { count: issueCount(block.id) })
              : `${block.value} · ${block.detail}`,
            done: block.state !== 'pending' && !issueCount(block.id),
          },
        ]
      : [],
  ),
);

const draftSummary = computed<DraftSummary | null>(() => {
  const draft = offered.value;

  if (!draft) return null;
  const restored = draftRequest(draft, defaults());
  const presetId = matchingPreset(restored.config);
  const presetKeys = presetId ? trainingPresetMessageKeysFor(presetId) : null;
  const preset = presetId ? trainingRecipes.find((item) => item.id === presetId) : null;

  return {
    savedAt: draft.savedAt,
    steps: [
      {
        label: t(projectBlockMessageKeys.dataset),
        value: draft.dataset?.relativePath.split('/').at(-1) ?? '—',
        done: !!draft.dataset,
      },
      {
        label: t(projectBlockMessageKeys.model),
        value: draft.model?.id ?? '—',
        done: !!draft.model,
      },
      {
        label: t(projectBlockMessageKeys.training),
        value: presetKeys ? t(presetKeys.label) : t('builder.customRecipe'),
        done: true,
      },
      {
        label: t(projectBlockMessageKeys.runtime),
        value:
          draftSetup(draft) === 'automatic'
            ? t('builder.setupAutomatic')
            : t(deviceLabels[draft.runtime.device]),
        done: draftSetup(draft) === 'automatic' || !!draft.runtime.python.trim(),
      },
    ],
    changes: preset
      ? []
      : draftConfigFields
          .filter((name) => changedFromDefault(name, restored.config))
          .slice(0, 4)
          .map((name) => {
            const value = restored.config[name];

            return `${fieldLabel(name)} ${
              value === true || value === false
                ? t(value ? 'builder.on' : 'builder.off')
                : value === null || value === undefined || Array.isArray(value)
                  ? '…'
                  : Number.isFinite(value)
                    ? compactNumber(Number(value))
                    : trainingChoiceLabel(name, String(value), t)
            }`;
          }),
  };
});

const savedRecently = computed(() => savedAt.value !== null);

onBeforeUnmount(() => {
  unsubscribe?.();
  retireScope();
  clearTimeout(saveTimer);
  clearTimeout(noticeTimer);
});

const projectStyles = {
  draftSaved: cn('[animation:draft-saved-in_420ms_cubic-bezier(0.22,_1,_0.36,_1)]'),
  runBuilderMuted: cn(
    'opacity-55 [filter:saturate(0.4)] [transition:opacity_400ms_ease,_filter_400ms_ease]',
  ),
  runBuilderPanel: cn(
    'min-h-96 rounded-[1.5rem] [transition:box-shadow_200ms_ease]',
    "[&[data-dragging='true']]:shadow-[0_0_0_2px_var(--primary)]",
  ),
  runMessage: cn(
    'rounded-[1rem] [animation:run-message-in_360ms_cubic-bezier(0.34,_1.56,_0.64,_1)]',
    "[&[data-tone='destructive']]:shadow-[0_14px_30px_-22px_rgb(220_38_38_/_0.55)]",
  ),
  runMessageFix: cn(
    'inline-flex items-center gap-[0.35rem] py-[0.3rem] px-[0.7rem] rounded-full text-[0.78rem] font-semibold',
    '[background:color-mix(in_srgb,_currentColor_10%,_transparent)] [transition:background-color_150ms_ease]',
    '[&:hover]:[background:color-mix(in_srgb,_currentColor_18%,_transparent)] [&_svg]:size-[0.8rem]',
  ),
  runNotice: cn(
    'flex items-start gap-3 p-[0.75rem_0.75rem_0.75rem_1rem] rounded-[1rem] bg-surface',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_12%,_transparent),_0_16px_36px_-20px_rgb(0_0_0_/_0.35)]',
    "[&[data-tone='success']_.run-notice-icon]:text-success [&[data-tone='warning']_.run-notice-icon]:text-warning",
  ),
  runNoticeEnterActive: cn(
    '[transition:opacity_260ms_ease,_transform_360ms_cubic-bezier(0.34,_1.56,_0.64,_1)]',
  ),
  runNoticeLeaveActive: cn('[transition:opacity_160ms_ease,_transform_160ms_ease]'),
  runNoticeEnterFrom: cn('opacity-0 [transform:translateY(-6px)_scale(0.98)]'),
  runNoticeLeaveTo: cn('opacity-0 [transform:translateY(-6px)_scale(0.98)]'),
  runNoticeIcon: cn('size-[1.1rem] flex-none mt-[0.1rem] text-primary'),
};
</script>
<template>
  <section
    class="run-builder mx-auto flex w-full max-w-6xl flex-col gap-6"
    aria-labelledby="project-title"
    @dragenter="dragEnter"
    @dragleave="dragLeave"
    @dragover.prevent
    @drop.prevent="drop"
  >
    <MotionConfig reduced-motion="user">
      <header class="flex flex-wrap items-end justify-between gap-4">
        <div class="min-w-0">
          <h1 id="project-title" class="text-[2rem] leading-tight font-semibold tracking-[-0.03em]">
            {{ t('project.buildATrainingRun') }}
          </h1>
          <p class="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
            {{ t('builder.subtitle') }}
            <span
              v-if="savedRecently && !offered"
              :class="cn('inline-flex items-center gap-1 text-xs', projectStyles.draftSaved)"
              role="status"
            >
              <Check class="size-3 text-success" aria-hidden="true" />{{ t('builder.draftSaved') }}
            </span>
          </p>
        </div>
        <ConfigMenu
          :draft="values"
          :disabled="busy || !connected"
          @import="openImport"
          @notice="exportNotice"
        />
        <input
          id="training-config-import"
          ref="importInput"
          type="file"
          class="sr-only"
          accept=".yaml,.yml,.json,application/yaml,application/json"
          tabindex="-1"
          :aria-label="t('builder.importConfig')"
          @change="pickedFile"
        />
      </header>

      <Alert v-if="!connected">
        <AlertTitle>{{ t('project.verifiedConnectionRequired') }}</AlertTitle>
        <AlertDescription>{{ t('project.connectionRequired') }}</AlertDescription>
      </Alert>

      <RunPipeline
        :blocks="blocks"
        :active="active"
        :panel-id="panelId"
        :review-label="reviewLabel"
        :review-disabled="!connected || busy || !!offered"
        :review-busy="busy"
        :class="offered ? projectStyles.runBuilderMuted : ''"
        @select="selectBlock"
        @review="review"
      />

      <Transition
        :enter-active-class="projectStyles.runNoticeEnterActive"
        :leave-active-class="projectStyles.runNoticeLeaveActive"
        :enter-from-class="projectStyles.runNoticeEnterFrom"
        :leave-to-class="projectStyles.runNoticeLeaveTo"
        mode="out-in"
      >
        <div
          v-if="notice"
          :key="notice.id"
          :class="projectStyles.runNotice"
          :data-tone="notice.tone"
          role="status"
        >
          <Check
            v-if="notice.tone === 'success'"
            :class="cn('run-notice-icon', projectStyles.runNoticeIcon)"
            aria-hidden="true"
          />
          <CircleAlert
            v-else-if="notice.tone === 'warning'"
            :class="cn('run-notice-icon', projectStyles.runNoticeIcon)"
            aria-hidden="true"
          />
          <Info
            v-else
            :class="cn('run-notice-icon', projectStyles.runNoticeIcon)"
            aria-hidden="true"
          />
          <div class="min-w-0 flex-1">
            <p class="text-sm font-medium">{{ notice.title() }}</p>
            <p v-if="notice.detail" class="mt-0.5 text-xs break-words text-muted-foreground">
              {{ notice.detail() }}
            </p>
          </div>
          <Button v-if="notice.undo" type="button" size="sm" variant="ghost" @click="notice.undo">
            <Undo2 aria-hidden="true" />{{ t('builder.undo') }}
          </Button>
          <Button
            type="button"
            size="icon-sm"
            variant="ghost"
            :aria-label="t('builder.dismiss')"
            @click="notice = null"
          >
            <X aria-hidden="true" />
          </Button>
        </div>
      </Transition>

      <Alert
        v-if="message"
        role="alert"
        :variant="messageTone"
        :class="projectStyles.runMessage"
        :data-tone="messageTone"
      >
        <OctagonAlert v-if="messageTone === 'destructive'" aria-hidden="true" />
        <TriangleAlert v-else-if="messageTone === 'warning'" aria-hidden="true" />
        <Info v-else aria-hidden="true" />
        <AlertTitle>
          {{
            messageTone === 'destructive' ? t('builder.planBlocked') : t('project.planningStatus')
          }}
        </AlertTitle>
        <AlertDescription>
          <p>
            {{
              message === 'project.submissionAckUnknown'
                ? t('project.submissionAckUnknown')
                : displayMessage
            }}
          </p>
          <div v-if="blockedBlocks.length" class="mt-2 flex flex-wrap gap-2">
            <button
              v-for="block in blockedBlocks"
              :key="block"
              type="button"
              :class="projectStyles.runMessageFix"
              @click="selectBlock(block)"
            >
              {{ t(projectBlockMessageKeys[block]) }}
              <span v-if="issueCount(block)" class="opacity-75"
                >· {{ t('builder.issueCount', { count: issueCount(block) }) }}</span
              >
              <ArrowRight aria-hidden="true" />
            </button>
          </div>
        </AlertDescription>
      </Alert>

      <StartPanel
        v-if="offered && draftSummary"
        :summary="draftSummary"
        :dragging="dragging"
        @continue="continueDraft"
        @import="openImport"
        @fresh="startFresh"
      />

      <form
        v-show="!offered"
        :id="panelId"
        role="tabpanel"
        :aria-labelledby="`${panelId}-tab-${active}`"
        :class="projectStyles.runBuilderPanel"
        :data-dragging="dragging"
        novalidate
        @submit.prevent="preview"
      >
        <DatasetInspector
          v-show="active === 'dataset'"
          :session="session"
          :selection="data"
          :inspection="datasetInspection"
          @select="selectData"
          @inspected="inspectData"
        />
        <ModelInspector
          v-show="active === 'model'"
          :session="session"
          :models="catalogue.models.value"
          :busy="catalogue.busy.value"
          :message="catalogue.message.value"
          :selection="model"
          :pretrained="values.config.pretrained"
          :inspection="datasetInspection"
          :single-channel="values.config.single_channel"
          :disabled="busy || !connected"
          @choose="chooseModel"
          @custom="selectModel"
          @reload="catalogue.load"
        />
        <TrainingInspector
          v-show="active === 'training'"
          :scope="scope"
          :disabled="busy"
          @preset="applyPreset"
          @reset="resetTraining"
        />
        <RuntimeInspector
          v-show="active === 'runtime'"
          v-model:setup="setup"
          :profile="host.profile.value"
          :detecting="host.busy.value"
          :disabled="busy"
          @environment="setFieldValue('runtime.environment', $event)"
          @storage="setFieldValue('storage_map', $event)"
          @detect="host.load"
          @detected="applyDetected"
        />
        <PlanInspector
          v-show="active === 'plan'"
          :plan="plan"
          :acknowledged="acknowledged"
          :readiness="readiness"
          :can-preview="canPreview"
          :busy="busy"
          :connected="connected"
          :state-label="acknowledged ? stateLabel(acknowledged.state) : ''"
          @submit="submit"
          @status="status"
          @open="openAcknowledged"
          @stop="invalidate"
          @fresh="another"
          @go="active = $event"
        />
      </form>
    </MotionConfig>
  </section>
</template>
