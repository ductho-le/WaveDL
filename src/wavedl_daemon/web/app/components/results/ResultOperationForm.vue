<script setup lang="ts">
import type { AcceptableValue } from 'reka-ui';
import { segmentedSelectionSchema, resultExportSchema } from '../../../shared/schemas/workstation-controls';
import { useForm, Field as VeeField } from 'vee-validate';
import { resultRequestSchema } from '../../../shared/schemas/result-request';
import type { ConnectionSession } from '../../lib/connection-session';
import type { Checkpoint } from '../../lib/artifact-client';
import type { DataSelection } from '../../lib/catalog-api';
import type { PlanView } from '../../lib/daemon-api';
import { ResultOperations } from '../../lib/result-operations';
import { createResultOperationDefaults } from '../../lib/result-operation-defaults';

const { t } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();

function fieldMessage(message: string): string {
  return message.startsWith('results.') ? t(message) : clientMessage(message);
}

const props = defineProps<{ session: ConnectionSession; checkpoints: readonly Checkpoint[] }>();
const session = toRaw(props.session);
const emit = defineEmits<{ runSelected: [runId: string] }>();
const workflow = new ResultOperations(props.session);
const data = shallowRef<DataSelection | null>(null);
const source = shallowRef<Checkpoint | null>(null);
const plan = shallowRef<PlanView | null>(null);
const busy = shallowRef(false);
const message = shallowRef('');

const selectionSchema = computed(() =>
  segmentedSelectionSchema(props.checkpoints.map((item) => item.name)),
);

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
  () => props.checkpoints,
  () => {
    invalidate();
    source.value = null;
  },
);

const unsubscribe = toRaw(props.session).subscribe((state) => {
  if (state.status !== 'connected') {
    invalidate();
    source.value = null;
    data.value = null;
  }
});

function chooseSource(value: AcceptableValue) {
  const selection = selectionSchema.value.safeParse(value);

  if (!selection.success) return;

  const selected = props.checkpoints.find((item) => item.name === selection.data);

  source.value = selected ?? null;
  setFieldValue('checkpoint_source', {
    run_id: selected?.runId ?? '',
    checkpoint: selected?.name ?? '',
  });
}

function sourceLabel(item: Checkpoint): string {
  if (item.name === 'best_checkpoint') return t('results.best');

  if (item.name === 'interrupted_checkpoint') return t('results.interrupted');

  return t('results.periodic', { epoch: item.name.match(/^epoch_([0-9]+)_checkpoint$/)?.[1] ?? '' });
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
    workflow.prepare(draft, selectedData, selectedSource);
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
          ><FieldLabel for="source-checkpoint">{{ t('results.sourceCheckpoint') }}</FieldLabel
          ><Select
            :model-value="source?.name ?? ''"
            :disabled="busy"
            @update:model-value="chooseSource"
            ><SelectTrigger id="source-checkpoint" :data-value="source?.name ?? ''"
              ><SelectValue :placeholder="t('results.chooseCheckpoint')" /></SelectTrigger
            ><SelectContent
              ><SelectGroup
                ><SelectItem v-for="item in checkpoints" :key="item.name" :value="item.name">
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
      <Button type="submit" :disabled="busy || !session.api">{{ t('results.previewPlan') }}</Button>
    </form>
    <p v-if="message" role="alert">
      {{ fieldMessage(message) }}
    </p>
    <PlanApproval v-if="plan" :plan="plan" :busy="busy" @submit="submit" @status="status" />
  </section>
</template>
