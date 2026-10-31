<script setup lang="ts">
import type { DataSelection } from '../../lib/catalog-api';
import type { ConnectionSession } from '../../lib/connection-session';
import { DatasetPreviewApi, sampleMatrix, type DatasetPreview } from '../../lib/dataset-preview';
import { datasetPreviewMessageKeys, datasetPreviewTabMessageKeys } from '../../i18n/dynamic-keys';

const { t } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();

const props = defineProps<{ session: ConnectionSession; selection: DataSelection | null }>();

const tabs = [
  ['input', datasetPreviewTabMessageKeys.input],
  ['output', datasetPreviewTabMessageKeys.output],
  ['sample', datasetPreviewTabMessageKeys.sample],
  ['correlation', datasetPreviewTabMessageKeys.correlation],
  ['statistics', datasetPreviewTabMessageKeys.statistics],
  ['scatter', datasetPreviewTabMessageKeys.scatter],
] as const;

type Tab = (typeof tabs)[number][0];

type PreviewMessage = { kind: 'key'; key: string } | { kind: 'error'; value: Error };

const activeTab = shallowRef<Tab>('input');
const data = shallowRef<DatasetPreview | null>(null);
const loading = shallowRef(false);
const available = shallowRef(false);
const message = shallowRef<PreviewMessage | null>(null);
let request: AbortController | null = null;
let sequence = 0;
let unsubscribe: (() => void) | null = null;

function revoke() {
  sequence++;
  request?.abort();
  request = null;
  data.value = null;
  loading.value = false;
}

function stop() {
  if (!loading.value) return;
  revoke();
  message.value = { kind: 'key', key: 'datasets.previewStopped' };
}

async function load() {
  revoke();
  message.value = null;
  const selection = props.selection ? toRaw(props.selection) : null;

  if (!selection) return;

  if (!available.value) {
    message.value = { kind: 'key', key: 'datasets.previewReconnect' };

    return;
  }

  const controller = new AbortController();
  request = controller;
  const current = sequence;
  loading.value = true;

  try {
    const result = await new DatasetPreviewApi(toRaw(props.session)).preview(
      selection,
      'all',
      controller.signal,
    );

    if (current === sequence && !controller.signal.aborted) data.value = result;
  } catch (error) {
    if (current === sequence && !controller.signal.aborted)
      message.value =
        error instanceof Error
          ? { kind: 'error', value: error }
          : { kind: 'key', key: 'datasets.previewUnavailable' };
  } finally {
    if (current === sequence) {
      loading.value = false;
      request = null;
    }
  }
}

watch(
  () => props.session,
  (session) => {
    unsubscribe?.();
    revoke();
    const connection = toRaw(session);
    let revision = connection.snapshot().revision;
    unsubscribe = connection.subscribe((state) => {
      available.value = state.status === 'connected';

      if (state.revision !== revision || state.status !== 'connected') {
        revision = state.revision;
        revoke();
        message.value = { kind: 'key', key: 'datasets.previewReconnect' };
      }
    });
  },
  { immediate: true },
);

watch(() => [props.selection, props.session], load, { immediate: true });

onBeforeUnmount(() => {
  unsubscribe?.();
  revoke();
});

function selectTab(value: string | number) {
  const tab = tabs.find(([key]) => key === value);

  if (tab) activeTab.value = tab[0];
}

const histograms = computed(() =>
  activeTab.value === 'input'
    ? (data.value?.input_histograms ?? [])
    : (data.value?.output_histograms ?? []),
);

const plots = computed(() =>
  histograms.value.map((histogram) => ({
    id: histogram.feature_index,
    feature: histogram.feature_index,
    x: histogram.counts.map(
      (_, index) => (histogram.edges[index] ?? 0) / 2 + (histogram.edges[index + 1] ?? 0) / 2,
    ),
    y: histogram.counts,
  })),
);

const scatterPlots = computed(() => {
  const scatter = data.value?.scatter;

  if (!scatter) return [];

  return scatter.feature_indices.map((feature, column) => {
    const points = scatter.x
      .flatMap((row, index) => {
        const x = row[column];
        const y = scatter.y[index];

        return x == null || y == null ? [] : [{ x, y }];
      })
      .sort((a, b) => a.x - b.x);

    return {
      id: feature,
      feature,
      x: points.map((item) => item.x),
      y: points.map((item) => item.y),
    };
  });
});

const matrices = computed(
  () =>
    data.value?.samples.map((sample) => {
      const matrix = sampleMatrix(sample);

      return { sample, matrix, x: matrix[0]?.map((_, index) => index) ?? [], y: matrix[0] ?? [] };
    }) ?? [],
);
</script>

<template>
  <section class="min-w-0 my-6" :aria-label="t('datasets.previewTitle')">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <h3>{{ t('datasets.previewTitle') }}</h3>
      <Button
        variant="outline"
        size="sm"
        :disabled="loading || !selection || !available"
        @click="load"
        >{{ t('datasets.refreshPreview') }}</Button
      >
      <Button v-if="loading" variant="ghost" size="sm" @click="stop">{{
        t('datasets.stopPreview')
      }}</Button>
    </div>
    <p v-if="loading" role="status">{{ t('datasets.previewLoading') }}</p>
    <p v-else-if="message" role="status">
      {{ message.kind === 'key' ? t(message.key) : clientMessage(message.value) }}
    </p>
    <p v-else-if="!selection" role="status">{{ t('datasets.selectDataset') }}</p>
    <template v-if="data">
      <p class="text-muted-foreground text-sm">
        {{ data.format.toUpperCase() }} · {{ data.input_key }} {{ data.input_shape.join(' × ') }} →
        {{ data.output_key }} {{ data.output_shape.join(' × ') }}
      </p>
      <div class="flex flex-wrap gap-2" :aria-label="t('datasets.previewScope')">
        <Badge variant="outline" data-testid="preview-bounds">
          {{
            t('datasets.previewBounds', {
              count: data.sample_indices.length,
              total: data.input_shape[0] ?? 0,
            })
          }}
        </Badge>
        <Badge variant="secondary">{{ t('datasets.previewFeatureCap') }}</Badge>
        <Badge variant="secondary">{{ t('datasets.previewCorrelationCap') }}</Badge>
        <Badge variant="secondary">{{ t('datasets.previewStatisticsCap') }}</Badge>
        <Badge variant="secondary">{{ t('datasets.previewSampleCap') }}</Badge>
        <InfoChip
          :label="t('datasets.previewNotFullChip')"
          :text="t('datasets.previewNotFull')"
          uncertain
        />
      </div>
      <Tabs :model-value="activeTab" @update:model-value="selectTab">
        <TabsList class="flex h-auto flex-wrap" :aria-label="t('datasets.previewView')">
          <TabsTrigger v-for="[key, label] in tabs" :key="key" :value="key">{{
            t(label)
          }}</TabsTrigger>
        </TabsList>
        <TabsContent :key="activeTab" :value="activeTab">
          <div
            v-if="activeTab === 'input' || activeTab === 'output'"
            class="grid grid-cols-[repeat(auto-fit,_minmax(min(100%,_320px),_1fr))] gap-6"
          >
            <PreviewPlot
              v-for="plot in plots"
              :key="plot.id"
              :title="t('datasets.featureTitle', { feature: plot.feature })"
              :x="plot.x"
              :y="plot.y"
              histogram
            />
          </div>
          <div v-else-if="activeTab === 'sample'" class="flex flex-col gap-5">
            <div v-for="{ sample, matrix, x, y } in matrices" :key="sample.sample_index">
              <Badge variant="outline" class="self-start">
                {{
                  t('datasets.sampleHelp', {
                    sample: sample.sample_index,
                    original: sample.feature_shape.join(' × ') || t('datasets.scalar'),
                    preview: sample.shape.join(' × '),
                  })
                }}
              </Badge>
              <PreviewPlot
                v-if="sample.shape.length === 1"
                :x="x"
                :y="y"
                :title="t('datasets.sampleTitle', { sample: sample.sample_index })"
              />
              <PreviewHeatmap
                v-else
                :matrix="matrix"
                :title="t('datasets.sampleTitle', { sample: sample.sample_index })"
              />
            </div>
          </div>
          <template v-else-if="activeTab === 'correlation'">
            <PreviewHeatmap
              v-if="data.correlation.matrix"
              :matrix="data.correlation.matrix"
              :title="t('datasets.correlationTitle')"
              correlation
            />
            <p v-else role="status">
              {{
                t(
                  data.sample_indices.length < 2
                    ? datasetPreviewMessageKeys.correlationInsufficientRows
                    : datasetPreviewMessageKeys.correlationMissing,
                )
              }}
            </p>
            <p class="text-muted-foreground text-xs">
              {{
                t('datasets.featureOrder', {
                  features: data.correlation.feature_indices.join(', '),
                })
              }}
            </p>
          </template>
          <PreviewStatistics
            v-else-if="activeTab === 'statistics'"
            :statistics="data.input_statistics"
          />
          <div
            v-else
            class="grid grid-cols-[repeat(auto-fit,_minmax(min(100%,_320px),_1fr))] gap-6"
          >
            <PreviewPlot
              v-for="plot in scatterPlots"
              :key="plot.id"
              :title="t('datasets.scatterTitle', { feature: plot.feature })"
              :x="plot.x"
              :y="plot.y"
              scatter
            />
          </div>
        </TabsContent>
      </Tabs>
    </template>
  </section>
</template>
