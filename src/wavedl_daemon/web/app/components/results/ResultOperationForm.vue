<script setup lang="ts">
import type { AcceptableValue } from 'reka-ui';
import { segmentedSelectionSchema, resultExportSchema } from '../../../shared/schemas/workstation-controls';
import { useForm, Field as VeeField } from 'vee-validate';
import { resultRequestSchema } from '../../../shared/schemas/result-request';
import type { ConnectionSession } from '../../lib/connection-session';
import type { Artifact } from '../../lib/artifact-client';
import type { DataSelection } from '../../lib/catalog-api';
import type { PlanView } from '../../lib/daemon-api';
import {
  ResultOperations,
  checkpointOptions,
  type CheckpointOption,
} from '../../lib/result-operations';
import { createResultOperationDefaults } from '../../lib/result-operation-defaults';

const { t } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();

function fieldMessage(message: string): string {
  return message.startsWith('results.') ? t(message) : clientMessage(message);
}

const props = defineProps<{ session: ConnectionSession; artifacts: readonly Artifact[] }>();
const session = toRaw(props.session);
const emit = defineEmits<{ runSelected: [runId: string] }>();
const workflow = new ResultOperations(props.session);
const data = shallowRef<DataSelection | null>(null);
const source = shallowRef<Artifact | null>(null);
const plan = shallowRef<PlanView | null>(null);
const busy = shallowRef(false);
const message = shallowRef('');
const sources = shallowRef<readonly CheckpointOption[]>([]);

const selectionSchema = computed(() =>
  segmentedSelectionSchema(sources.value.flatMap((item) =>
    item.selectable ? [item.artifact.generation] : [],
  )),
);

const sourceMessage = shallowRef('');

const { handleSubmit, values, setFieldValue, setFieldError } = useForm({
  validationSchema: resultRequestSchema,
  initialValues: {
    ...createResultOperationDefaults(),
  },
});

let scope = new AbortController();
let epoch = 0;

function invalidate() {
  epoch++;
  scope.abort();
  scope = new AbortController();
  workflow.invalidate();
  plan.value = null;
  busy.value = false;
}

watch(values, invalidate, { deep: true });

watch(
  () => props.artifacts,
  async (_items, _prior, onCleanup) => {
    invalidate();
    source.value = null;
    sources.value = [];
    sourceMessage.value = 'results.verifying';
    const controller = new AbortController();
    const revision = session.snapshot().revision;
    onCleanup(() => controller.abort());

    try {
      const options = await checkpointOptions(session, props.artifacts, controller.signal);

      if (!controller.signal.aborted && session.api && session.snapshot().revision === revision) {
        sources.value = options;
        sourceMessage.value = options.length ? '' : 'results.noClassification';
      }
    } catch {
      if (!controller.signal.aborted) sourceMessage.value = 'results.classificationFailed';
    }
  },
  { immediate: true },
);

const unsubscribe = toRaw(props.session).subscribe((state) => {
  if (state.status !== 'connected') {
    invalidate();
    source.value = null;
    sources.value = [];
    data.value = null;
  }
});

function chooseSource(value: AcceptableValue) {
  const selection = selectionSchema.value.safeParse(value);

  if (!selection.success) return;

  const selected = sources.value.find(
    (item) => item.selectable && item.artifact.generation === selection.data,
  )?.artifact;

  source.value = selected ?? null;
  setFieldValue('checkpoint_source', {
    run_id: selected?.runId ?? '',
    generation: selected?.generation ?? '',
  });
}

function sourceLabel(item: CheckpointOption): string {
  const kind = item.classification.kind;

  const kindKeys: Readonly<Record<CheckpointOption['classification']['kind'], string>> = {
    best: 'results.best',
    periodic: 'results.periodic',
    interrupted: 'results.interrupted',
    cross_validation_fold: 'results.cvFold',
    cross_validation_summary: 'results.cvSummary',
  };

  return t('results.sourceLabel', {
    generation: item.artifact.generation,
    attempt: item.artifact.attempt,
    kind: t(kindKeys[kind]),
    fold: item.classification.fold === undefined ? '' : ` ${item.classification.fold}`,
  });
}

function chooseData(selected: DataSelection) {
  data.value = selected;
  setFieldValue('config.data_path', selected.path);
}

function exportValue(value: AcceptableValue): 'onnx' | null {
  return resultExportSchema.safeParse(value).data ?? null;
}

async function work(action: (signal: AbortSignal) => Promise<void>) {
  if (busy.value) return;
  const current = epoch;
  busy.value = true;
  message.value = '';

  try {
    await action(scope.signal);
  } catch {
    if (current === epoch) {
      message.value = 'results.unacknowledged';
      setFieldError('backend', 'results.settingsError');
    }
  } finally {
    if (current === epoch) busy.value = false;
  }
}

const preview = handleSubmit(async (draft) => {
  const selectedData = data.value;
  const selectedSource = source.value;

  if (!selectedData || !selectedSource) {
    message.value = 'results.selectionRequired';

    return;
  }

  await work(async (signal) => {
    await workflow.prepare(draft, selectedData, selectedSource, signal);
    plan.value = await workflow.preview(signal);
  });
});

async function submit() {
  await work(async (signal) => {
    const view = await workflow.submit(signal);
    emit('runSelected', view.run_id);
  });
}

async function status() {
  await work(async (signal) => {
    const view = await workflow.status(signal);
    emit('runSelected', view.run_id);
  });
}

onBeforeUnmount(() => {
  unsubscribe();
  invalidate();
});
</script>
<template>
  <section class="flex flex-col gap-5 border-t pt-6" aria-labelledby="inference-title">
    <h2 id="inference-title" class="text-xl font-semibold">{{ t('results.evaluatePlots') }}</h2>
    <InfoChip :label="t('results.explicitSource')" :text="t('results.sourceHelp')" uncertain />
    <BrowseContainer :session="session" @select="chooseData" />
    <form class="flex flex-col gap-5" @submit.prevent="preview">
      <FieldGroup>
        <Field
          ><FieldLabel for="source-generation">{{ t('results.sourceGeneration') }}</FieldLabel
          ><Select :disabled="busy" @update:model-value="chooseSource"
            ><SelectTrigger id="source-generation" :data-value="source?.generation ?? ''"
              ><SelectValue :placeholder="t('results.choosePublication')" /></SelectTrigger
            ><SelectContent
              ><SelectGroup
                ><SelectItem
                  v-for="item in sources"
                  :key="item.artifact.generation"
                  :value="item.artifact.generation"
                  :disabled="!item.selectable"
                >
                  {{ sourceLabel(item) }}
                </SelectItem></SelectGroup
              ></SelectContent
            ></Select
          ></Field
        >
        <VeeField v-slot="{ field, errors }" name="operation"
          ><Field :data-invalid="!!errors.length"
            ><FieldLabel for="result-operation">{{ t('results.operation') }}</FieldLabel
            ><Select
              :model-value="field.value"
              @update:model-value="field.onChange"
              :disabled="busy"
              :aria-invalid="!!errors.length"
              ><SelectTrigger id="result-operation" :data-value="field.value"
                ><SelectValue /></SelectTrigger
              ><SelectContent
                ><SelectGroup
                  ><SelectItem value="test">{{ t('results.evaluate') }}</SelectItem
                  ><SelectItem value="plot">{{
                    t('results.generatePlots')
                  }}</SelectItem></SelectGroup
                ></SelectContent
              ></Select
            ><FieldError v-if="errors.length" :errors="errors.map(fieldMessage)" /></Field
        ></VeeField>
        <VeeField v-slot="{ field, errors }" name="config.export"
          ><Field :data-invalid="!!errors.length"
            ><FieldLabel for="result-export">{{ t('results.modelExport') }}</FieldLabel
            ><Select
              :model-value="field.value ?? ''"
              @update:model-value="(value) => field.onChange(exportValue(value))"
              :disabled="busy"
              :aria-invalid="!!errors.length"
              ><SelectTrigger id="result-export" :data-value="field.value ?? 'none'"
                ><SelectValue :placeholder="t('results.none')" /></SelectTrigger
              ><SelectContent
                ><SelectGroup
                  ><SelectItem value="none">{{ t('results.none') }}</SelectItem
                  ><SelectItem value="onnx">ONNX</SelectItem></SelectGroup
                ></SelectContent
              ></Select
            ><FieldError v-if="errors.length" :errors="errors.map(fieldMessage)" /></Field
        ></VeeField>
        <InferenceFields :disabled="busy" />
        <RuntimeFields :disabled="busy" />
        <EnvironmentFields
          :disabled="busy"
          :environment="values.runtime.environment"
          @update="setFieldValue('runtime.environment', $event)"
        />
        <StorageMappingFields
          :mappings="values.storage_map"
          :backend="values.backend"
          :disabled="busy"
          @update="setFieldValue('storage_map', $event)"
        />
      </FieldGroup>
      <p v-if="sourceMessage" role="status">{{ t(sourceMessage) }}</p>
      <Button type="submit" :disabled="busy || !session.api">{{ t('results.previewPlan') }}</Button>
    </form>
    <p v-if="message" role="alert">
      {{ fieldMessage(message) }}
    </p>
    <PlanApproval v-if="plan" :plan="plan" :busy="busy" @submit="submit" @status="status" />
  </section>
</template>
