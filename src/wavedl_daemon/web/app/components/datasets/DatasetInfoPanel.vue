<script setup lang="ts">
import type { ConnectionSession } from '../../lib/connection-session';
import type { DataSelection } from '../../lib/catalog-api';
import { DatasetInspectionApi, type DatasetInspection } from '../../lib/dataset-inspection';

const { t } = useI18n({ useScope: 'global' });

const props = defineProps<{ session: ConnectionSession; selection: DataSelection | null }>();

const emit = defineEmits<{
  inspected: [inspection: DatasetInspection | null, selection: DataSelection | null];
}>();

const info = shallowRef<DatasetInspection | null>(null);
const busy = shallowRef(false);
const connected = shallowRef(false);

const message = shallowRef<'' | 'datasets.inspectionReconnect' | 'datasets.inspectionUnavailable'>(
  '',
);

let controller: AbortController | null = null;
let sequence = 0;
let unsubscribe: (() => void) | undefined;

function clear() {
  sequence++;
  controller?.abort();
  controller = null;
  info.value = null;
  emit('inspected', null, props.selection ? toRaw(props.selection) : null);
  busy.value = false;
}

async function load() {
  clear();
  message.value = '';
  const selection = props.selection ? toRaw(props.selection) : null;

  if (!selection) return;

  if (!connected.value) {
    message.value = 'datasets.inspectionReconnect';

    return;
  }

  const current = sequence;
  const request = new AbortController();
  controller = request;
  busy.value = true;

  try {
    const result = await new DatasetInspectionApi(toRaw(props.session)).inspect(
      selection,
      request.signal,
    );

    if (current === sequence && !request.signal.aborted) {
      info.value = result;
      emit('inspected', result, selection);
    }
  } catch {
    if (current === sequence && !request.signal.aborted)
      message.value = 'datasets.inspectionUnavailable';
  } finally {
    if (current === sequence) {
      busy.value = false;
      controller = null;
    }
  }
}

watch(
  () => props.session,
  (session) => {
    unsubscribe?.();
    clear();
    let revision = toRaw(session).snapshot().revision;
    unsubscribe = toRaw(session).subscribe((state) => {
      connected.value = state.status === 'connected';

      if (state.revision !== revision || !connected.value) {
        revision = state.revision;
        clear();
        message.value = 'datasets.inspectionReconnect';
      }
    });
  },
  { immediate: true },
);

watch(() => [props.selection, props.session], load, { immediate: true });

onBeforeUnmount(() => {
  unsubscribe?.();
  clear();
});
</script>

<template>
  <section class="flex min-w-0 flex-col gap-4" :aria-label="t('datasets.headerTitle')">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <h3>{{ t('datasets.headerTitle') }}</h3>
      <Button
        variant="outline"
        size="sm"
        :disabled="!selection || !connected || busy"
        @click="load"
        >{{ t('datasets.refreshHeader') }}</Button
      >
      <Button v-if="busy" variant="ghost" size="sm" @click="clear">{{
        t('datasets.stopInspecting')
      }}</Button>
    </div>
    <InfoChip :label="t('datasets.headerDetails')" :text="t('datasets.headerHelp')" />
    <p v-if="busy" role="status">{{ t('datasets.headerLoading') }}</p>
    <div v-else-if="message" role="status">
      <InfoChip
        :label="t(message)"
        :text="
          message === 'datasets.inspectionUnavailable'
            ? t('datasets.inspectionDetails')
            : t(message)
        "
        uncertain
      />
    </div>
    <p v-else-if="!selection" role="status">{{ t('datasets.chooseDataset') }}</p>
    <dl v-if="info" class="flex flex-wrap gap-2" :aria-label="t('datasets.infoChips')">
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.format') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="outline">{{ info.format.toUpperCase() }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.samples') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="outline">{{ info.numSamples }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.fileSize') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="outline">{{ t('datasets.unknown') }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.inputKey') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="secondary">{{ info.inputKey }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.inputShape') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="secondary">{{
            info.inputShape.join(' × ') || t('datasets.scalar')
          }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.inputDtype') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="secondary">{{ info.inputDtype }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.outputKey') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="secondary">{{ info.outputKey }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.outputShape') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="secondary">{{
            info.outputShape.join(' × ') || t('datasets.scalar')
          }}</Badge>
        </dd>
      </div>
      <div class="flex min-w-0 max-w-full items-baseline gap-2 rounded-lg border px-3 py-2 text-sm">
        <dt class="text-muted-foreground">{{ t('datasets.outputDtype') }}</dt>
        <dd class="min-w-0 break-all font-medium">
          <Badge variant="secondary">{{ info.outputDtype }}</Badge>
        </dd>
      </div>
    </dl>
    <p v-if="selection" class="break-all text-xs text-muted-foreground">
      {{ selection.relativePath }}
    </p>
  </section>
</template>
