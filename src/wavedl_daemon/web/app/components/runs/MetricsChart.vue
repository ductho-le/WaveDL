<script setup lang="ts">
import { cn } from '@/lib/utils';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { ChartLine, Maximize2 } from '@lucide/vue';
import type { MetricRecord } from '../../lib/metric-history';
import { buildPlotSeries, type PlotChannel, type PlotSeries } from '../../lib/plot-series';
import { plotMetricMessageKeys } from '../../i18n/dynamic-keys';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  chartHeading: cn(
    'flex flex-wrap items-center justify-between [gap:0.75rem_1rem] mb-3 [&_h3]:m-0 [&_h3]:text-base',
    '[&_h3]:font-[650] [&_h3]:tracking-[-0.015em] [&_p]:m-[0.2rem_0_0] [&_p]:text-muted-foreground',
    '[&_p]:text-[0.78rem] [&_p]:tabular-nums',
  ),
  chartState: cn(
    'flex min-h-48 flex-col items-center justify-center gap-2 rounded-[1rem] text-muted-foreground',
    'text-[0.85rem] text-center bg-[color-mix(in_srgb,_var(--foreground)_3%,_transparent)]',
    '[&_svg]:w-[1.4rem] [&_svg]:h-[1.4rem] [&_svg]:opacity-[0.6]',
  ),
};

const { t, locale } = useI18n({ useScope: 'global' });

function metricLabel(field: string): string {
  const key = Object.entries(plotMetricMessageKeys).find(([name]) => name === field)?.[1];

  return key === undefined ? field : t(key);
}

const props = defineProps<{ records: readonly MetricRecord[]; unavailable?: boolean }>();
const channel = shallowRef<PlotChannel>('loss');
const host = useTemplateRef<HTMLDivElement>('host');
const plot = shallowRef<uPlot | null>(null);
const renderError = shallowRef('');
const { isActive } = usePageActive(resumePlot, pausePlot);
let retainedRange: { min: number; max: number } | null = null;
let observer: ResizeObserver | null = null;
let frame: number | null = null;

const projection = computed<{ series: PlotSeries | null; error: string }>(() => {
  try {
    return {
      series: buildPlotSeries(props.unavailable ? [] : props.records, channel.value),
      error: '',
    };
  } catch (error) {
    return {
      series: null,
      error: error instanceof Error ? error.message : t('runs.chartDataError'),
    };
  }
});

const message = computed(() => {
  const raw = projection.value.error || renderError.value;
  const capacity = /^Chart capacity exceeded \((\d+) metric records\)\.$/.exec(raw);

  return capacity ? t('runs.chartCapacity', { count: capacity[1]! }) : raw;
});

function selectChannel(value: string | number) {
  if (value === 'loss' || value === 'r2' || value === 'lr') channel.value = value;
}

function destroyPlot() {
  plot.value?.destroy();
  plot.value = null;
}

function renderPlot() {
  frame = null;

  if (!isActive.value || !host.value) return;
  const series = projection.value.series;

  if (!series || series.pointCount === 0 || props.unavailable) {
    destroyPlot();

    return;
  }

  const width = Math.floor(host.value.getBoundingClientRect().width);

  if (width <= 0) return;

  try {
    renderError.value = '';

    if (plot.value) {
      plot.value.setSize({ width, height: 300 });
      plot.value.setData(series.data);

      return;
    }

    const colors = getComputedStyle(host.value);
    const primary = colors.getPropertyValue('--primary').trim() || '#007aff';
    const secondary = colors.getPropertyValue('--chart-2').trim() || '#7b6dba';
    const muted = colors.getPropertyValue('--muted-foreground').trim() || '#737373';
    const border = colors.getPropertyValue('--border').trim() || '#e5e5e5';
    plot.value = new uPlot(
      {
        width,
        height: 300,
        scales: { x: { time: false } },
        axes: [
          { label: t('runs.step'), stroke: muted, grid: { stroke: border } },
          { stroke: muted, grid: { stroke: border } },
        ],
        series: [
          { label: t('runs.step') },
          ...series.fields.map((field, index) => ({
            label: metricLabel(field),
            stroke: index === 0 ? primary : secondary,
            fill: index === 0 ? (chart: uPlot) => areaFill(chart, primary) : undefined,
            width: 2,
            spanGaps: false,
            points: { show: (chart: uPlot) => chart.data[0].length === 1, size: 5 },
          })),
        ],
        cursor: { drag: { x: true, y: false } },
      },
      series.data,
      host.value,
    );

    if (retainedRange) plot.value.setScale('x', retainedRange);
    retainedRange = null;
  } catch {
    destroyPlot();
    renderError.value = t('runs.chartRenderError');
  }
}

/** A soft wash under the primary series; non-hex theme colours simply skip it. */
function areaFill(chart: uPlot, color: string): CanvasGradient | string {
  if (!/^#[0-9a-f]{6}$/i.test(color)) return 'transparent';
  // The legend asks for its swatch before layout, while the plot box is still empty.
  const { top, height } = chart.bbox;

  if (!Number.isFinite(top) || !Number.isFinite(height) || height <= 0) return `${color}2e`;
  const gradient = chart.ctx.createLinearGradient(0, top, 0, top + height);
  gradient.addColorStop(0, `${color}2e`);
  gradient.addColorStop(1, `${color}00`);

  return gradient;
}

function scheduleRender() {
  if (!isActive.value || frame !== null) return;
  frame = requestAnimationFrame(renderPlot);
}

function resetZoom() {
  const data = projection.value.series?.data;

  if (data) plot.value?.setData(data);
}

watch(
  channel,
  () => {
    destroyPlot();
    scheduleRender();
  },
  { flush: 'post' },
);

watch(projection, scheduleRender, { flush: 'post' });

// Rebuild library-owned text without re-projecting records or losing a user's zoom.
watch(
  locale,
  () => {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    const viewport = plot.value?.scales?.x;

    const range =
      viewport?.min != null && viewport.max != null
        ? { min: viewport.min, max: viewport.max }
        : null;

    destroyPlot();
    renderPlot();

    if (range) plot.value?.setScale('x', range);
  },
  { flush: 'post' },
);

function resumePlot() {
  observer = new ResizeObserver(scheduleRender);

  if (host.value) observer.observe(host.value);
  scheduleRender();
}

function pausePlot() {
  const range = plot.value?.scales.x;
  retainedRange =
    range?.min !== undefined && range.max !== undefined ? { min: range.min, max: range.max } : null;
  observer?.disconnect();
  observer = null;

  if (frame !== null) cancelAnimationFrame(frame);
  frame = null;
  destroyPlot();
}
</script>

<template>
  <section
    class="flex flex-col gap-3 min-w-0 p-[1.25rem_1.4rem] rounded-[1.25rem] bg-surface batch-r-surface-shadow"
    :aria-label="t('runs.runtimeHistory')"
  >
    <Tabs :model-value="channel" @update:model-value="selectChannel">
      <div :class="cn(styles.chartHeading)">
        <div class="min-w-0">
          <h3>{{ t('runs.history') }}</h3>
          <p>{{ t('runs.lineage', { count: projection.series?.pointCount ?? 0 }) }}</p>
        </div>
        <div class="flex items-center gap-[0.35rem]">
          <TabsList class="rounded-[0.65rem]" :aria-label="t('runs.channel')">
            <TabsTrigger value="loss">{{ t('runs.loss') }}</TabsTrigger>
            <TabsTrigger value="r2">{{ t('runs.r2') }}</TabsTrigger>
            <TabsTrigger value="lr">{{ t('runs.lr') }}</TabsTrigger>
          </TabsList>
          <Button
            variant="ghost"
            size="icon"
            class="rounded-full text-muted-foreground"
            :disabled="!plot"
            :aria-label="t('runs.resetZoom')"
            :title="t('runs.resetZoom')"
            @click="resetZoom"
            ><Maximize2 aria-hidden="true"
          /></Button>
        </div>
      </div>
      <TabsContent :value="channel" force-mount class="min-w-0">
        <p v-if="message" role="status" :class="cn(styles.chartState)">{{ message }}</p>
        <p v-else-if="unavailable" role="status" :class="cn(styles.chartState)">
          <InfoChip :label="t('runs.lastKnownRuntime')" :text="t('runs.notLive')" uncertain />
        </p>
        <p v-else-if="!projection.series?.pointCount" role="status" :class="cn(styles.chartState)">
          <ChartLine aria-hidden="true" />
          {{
            t('runs.waitingChannel', {
              channel:
                channel === 'lr'
                  ? t('runs.learningRateLower')
                  : channel === 'loss'
                    ? t('runs.lossLower')
                    : t('runs.r2Lower'),
            })
          }}
        </p>
        <div
          ref="host"
          class="chart-host w-full min-w-0 min-h-85"
          :hidden="unavailable || !!message || !projection.series?.pointCount"
        />
      </TabsContent>
    </Tabs>
    <InfoChip :label="t('runs.chartDetails')" :text="t('runs.chartNote')" />
  </section>
</template>

<style scoped>
.chart-host :deep(.uplot) {
  font-family: inherit;
}
.chart-host :deep(.u-legend) {
  font-size: 0.78rem;
  color: var(--muted-foreground);
}
.chart-host :deep(.u-select) {
  background: color-mix(in srgb, var(--primary) 12%, transparent);
}
</style>
