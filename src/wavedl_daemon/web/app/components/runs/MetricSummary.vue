<script setup lang="ts">
import { cn } from '@/lib/utils';
import { TrendingDown, TrendingUp } from '@lucide/vue';
import type { MetricRecord, MetricSummary } from '../../lib/metric-history';
import { durationText, remainingSeconds } from '../../lib/run-overview';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  progressBand: cn(
    'grid grid-cols-[minmax(0,_1.4fr)_minmax(0,_1fr)] items-center [gap:1rem_1.5rem] p-[1rem_1.1rem]',
    'rounded-[1rem] bg-[color-mix(in_srgb,_var(--primary)_6%,_transparent)]',
    "[&[data-unconfirmed='true']]:bg-[color-mix(in_srgb,_var(--foreground)_4%,_transparent)]",
    '[&:not(:has(.progress-main))]:grid-cols-[minmax(0,_1fr)] @max-[34rem]:grid-cols-[minmax(0,_1fr)]',
  ),
  progressStats: cn(
    'grid grid-cols-[repeat(auto-fit,_minmax(6rem,_1fr))] gap-3 m-0 [&_dt]:text-muted-foreground',
    '[&_dt]:text-[0.72rem] [&_dd]:m-[0.15rem_0_0] [&_dd]:text-[0.95rem] [&_dd]:font-semibold',
    '[&_dd]:tabular-nums [&_dd]:whitespace-nowrap',
  ),
  kpiTile: cn(
    'relative grid grid-cols-[minmax(0,_1fr)_auto]',
    "[grid-template-areas:'label_trend'_'value_value'_'spark_spark'] items-center [gap:0.2rem_0.5rem]",
    'min-w-0 p-[0.85rem_0.95rem_0.6rem] rounded-[1rem]',
    'bg-[color-mix(in_srgb,_var(--foreground)_3.5%,_transparent)] [&_dt]:[grid-area:label]',
    '[&_dt]:overflow-hidden [&_dt]:text-muted-foreground [&_dt]:text-xs [&_dt]:text-ellipsis',
    '[&_dt]:whitespace-nowrap [&_dd]:[grid-area:value] [&_dd]:m-0 [&_dd]:overflow-hidden',
    '[&_dd]:text-[1.45rem] [&_dd]:font-[650] [&_dd]:tabular-nums [&_dd]:tracking-[-0.03em]',
    '[&_dd]:text-ellipsis [&_dd]:whitespace-nowrap',
  ),
  kpiTrend: cn(
    '[grid-area:trend] inline-flex items-center gap-[0.2rem] p-[0.1rem_0.4rem] rounded-full',
    'text-muted-foreground text-[0.68rem] font-semibold tabular-nums',
    'bg-[color-mix(in_srgb,_var(--foreground)_6%,_transparent)] [&_svg]:w-3 [&_svg]:h-3',
    "[&[data-tone='good']]:text-success",
    "[&[data-tone='good']]:bg-[color-mix(in_srgb,_var(--success)_13%,_transparent)]",
    "[&[data-tone='bad']]:text-warning",
    "[&[data-tone='bad']]:bg-[color-mix(in_srgb,_var(--warning)_13%,_transparent)]",
  ),
  metricSpec: cn(
    '[&_summary]:w-fit [&_summary]:text-muted-foreground [&_summary]:text-[0.78rem]',
    '[&_summary]:font-semibold [&_summary]:cursor-pointer',
    '[&_summary:focus-visible]:[outline:2px_solid_var(--ring)]!',
    '[&_summary:focus-visible]:[outline-offset:2px]! [&_summary:focus-visible]:rounded-[0.25rem]',
  ),
  metricSpecList: cn(
    'grid grid-cols-[repeat(auto-fit,_minmax(min(100%,_15rem),_1fr))] gap-x-7 m-[0.5rem_0_0]',
    '[&_>_div]:flex [&_>_div]:items-baseline [&_>_div]:justify-between [&_>_div]:gap-4',
    '[&_>_div]:p-[0.55rem_0]',
    '[&_>_div]:[border-bottom:1px_solid_color-mix(in_srgb,_var(--border)_60%,_transparent)]!',
    '[&_dt]:text-muted-foreground [&_dt]:text-[0.82rem] [&_dd]:m-0 [&_dd]:text-[0.88rem] [&_dd]:font-[550]',
    '[&_dd]:tabular-nums [&_dd]:whitespace-nowrap',
  ),
};

const { t, locale } = useI18n({ useScope: 'global' });

const props = defineProps<{
  summary: MetricSummary;
  unconfirmed?: boolean;
  records?: readonly MetricRecord[];
}>();

const gradient = `${useId()}-spark`;

// Whether a rising value is good news; null means direction carries no verdict.
const channels = [
  { name: 'train_loss', risingIsBetter: false },
  { name: 'val_loss', risingIsBetter: false },
  { name: 'r2', risingIsBetter: true },
  { name: 'lr', risingIsBetter: null },
] as const;

function series(index: number): (number | null)[] {
  const channel = channels[index]?.name;

  if (!channel) return [];

  return (props.records ?? []).slice(-60).map((record) => {
    const value = record.metrics[channel] ?? (record.name === channel ? record.value : null);

    return value !== null && value !== undefined && Number.isFinite(value) ? value : null;
  });
}

function sparkline(index: number): { line: string; area: string } | null {
  const values = series(index);
  const numeric = values.filter((value): value is number => value !== null);

  if (numeric.length < 2) return null;
  const min = Math.min(...numeric);
  const span = Math.max(...numeric) - min || 1;
  const x = (position: number) => (position / Math.max(values.length - 1, 1)) * 120;
  const y = (value: number) => 30 - ((value - min) / span) * 26;

  const line = values
    .map((value, position) =>
      value === null
        ? ''
        : `${position === 0 || values[position - 1] === null ? 'M' : 'L'}${x(position)},${y(value)}`,
    )
    .join(' ');

  // The fill follows the last unbroken stretch so gaps never shade missing data.
  let start = values.length - 1;

  while (start > 0 && values[start - 1] !== null) start--;
  const tail = values.slice(start);

  const area =
    tail.length > 1
      ? `M${x(start)},32 ${tail.map((value, offset) => `L${x(start + offset)},${y(value ?? min)}`).join(' ')} L${x(values.length - 1)},32 Z`
      : '';

  return { line, area };
}

function trend(
  index: number,
): { text: string; tone: 'good' | 'bad' | 'neutral'; rising: boolean } | null {
  const numeric = series(index).filter((value): value is number => value !== null);
  const latest = numeric[numeric.length - 1];
  const previous = numeric[numeric.length - 2];
  const channel = channels[index];

  if (latest === undefined || previous === undefined || !channel || latest === previous)
    return null;
  const rising = latest > previous;

  const options: Intl.NumberFormatOptions = {
    maximumSignificantDigits: 2,
    signDisplay: 'always',
  };

  if (channel.name !== 'r2' && previous !== 0) options.style = 'percent';
  const format = new Intl.NumberFormat(locale.value, options);

  const change =
    channel.name === 'r2' || previous === 0
      ? latest - previous
      : (latest - previous) / Math.abs(previous);

  return {
    text: format.format(change),
    rising,
    tone:
      channel.risingIsBetter === null
        ? 'neutral'
        : rising === channel.risingIsBetter
          ? 'good'
          : 'bad',
  };
}

function display(value: number | null | undefined, suffix = ''): string {
  return value == null
    ? t('runs.unavailable')
    : `${new Intl.NumberFormat(locale.value, { maximumSignificantDigits: 4 }).format(value)}${suffix}`;
}

const metrics = computed(() => {
  const current = props.summary.current;

  if (!current) return [];

  const exactValues = [
    current.trainLoss,
    current.validationLoss,
    current.r2,
    current.learningRate,
    current.pearson,
    current.gradientNorm,
    current.mae,
    current.epochSeconds,
    current.bestR2,
    current.elapsedSeconds,
    current.patience ? `${current.patience.used} / ${current.patience.maximum}` : null,
  ];

  return [
    { label: t('runs.trainLoss'), value: display(current.trainLoss) },
    { label: t('runs.valLoss'), value: display(current.validationLoss) },
    { label: t('runs.r2'), value: display(current.r2) },
    { label: t('runs.lr'), value: display(current.learningRate) },
    { label: t('runs.pearson'), value: display(current.pearson) },
    { label: t('runs.gradNorm'), value: display(current.gradientNorm) },
    { label: t('runs.mae'), value: display(current.mae) },
    { label: t('runs.epochDuration'), value: display(current.epochSeconds, ' s') },
    { label: t('runs.bestR2'), value: display(current.bestR2) },
    { label: t('runs.elapsed'), value: display(current.elapsedSeconds, ' s') },
    {
      label: t('runs.patience'),
      value: current.patience
        ? `${current.patience.used} / ${current.patience.maximum}`
        : t('runs.unavailable'),
    },
  ].map((metric, index) => ({ ...metric, exact: exactValues[index] }));
});

const kpis = computed(() =>
  metrics.value.slice(0, 4).map((metric, index) => {
    const change = trend(index);

    return {
      ...metric,
      spark: sparkline(index),
      trend: change && {
        ...change,
        label: [
          t('runs.changeFromPrevious', { value: change.text }),
          change.tone === 'good'
            ? t('runs.improving')
            : change.tone === 'bad'
              ? t('runs.worsening')
              : '',
        ]
          .filter(Boolean)
          .join(', '),
      },
    };
  }),
);

const percent = computed(() =>
  props.summary.progress
    ? new Intl.NumberFormat(locale.value, { style: 'percent', maximumFractionDigits: 0 }).format(
        Math.min(1, Math.max(0, props.summary.progress.fraction)),
      )
    : '',
);

const elapsed = computed(() => props.summary.current?.elapsedSeconds ?? null);

// Last-known evidence must not read as a live countdown.
const remaining = computed(() =>
  props.unconfirmed ? null : remainingSeconds(props.summary.progress, elapsed.value),
);
</script>
<template>
  <section
    class="@container flex flex-col gap-[1.1rem] min-w-0 p-[1.25rem_1.4rem_1.4rem] rounded-[1.25rem] bg-surface batch-r-surface-shadow"
    :aria-label="t('runs.currentMetrics')"
  >
    <header
      class="flex flex-wrap items-center justify-between gap-2 [&_h3]:text-base [&_h3]:font-[650] [&_h3]:tracking-[-0.015em]"
    >
      <h3>{{ unconfirmed ? t('runs.lastKnownRuntime') : t('runs.currentRuntime') }}</h3>
      <InfoChip
        v-if="unconfirmed"
        :label="t('runs.lastKnownRuntime')"
        :text="t('runs.lastKnownNote')"
        uncertain
      />
      <InfoChip v-else :label="t('runs.currentRuntime')" :text="t('runs.latestNote')" uncertain />
    </header>
    <div
      v-if="summary.progress || summary.best !== null"
      :class="cn(styles.progressBand)"
      :data-unconfirmed="unconfirmed ? 'true' : undefined"
    >
      <div v-if="summary.progress" class="progress-main flex flex-col gap-[0.6rem] min-w-0">
        <div class="flex items-baseline justify-between gap-3">
          <span class="text-[1.2rem] font-[650] tabular-nums tracking-[-0.02em]">{{
            t('runs.epoch', { epoch: summary.progress.epoch, total: summary.progress.totalEpochs })
          }}</span>
          <span
            class="text-muted-foreground text-[0.85rem] font-[550] tabular-nums"
            aria-hidden="true"
            >{{ percent }}</span
          >
        </div>
        <Progress
          class="h-[0.45rem]"
          :model-value="summary.progress.fraction * 100"
          :aria-label="t('runs.epochProgress')"
        />
      </div>
      <dl :class="cn(styles.progressStats)">
        <div v-if="elapsed !== null">
          <dt>{{ t('runs.elapsed') }}</dt>
          <dd>{{ durationText(elapsed, locale) }}</dd>
        </div>
        <div v-if="remaining !== null" :title="t('runs.remainingNote')">
          <dt>{{ t('runs.remaining') }}</dt>
          <dd>≈ {{ durationText(remaining, locale) }}</dd>
        </div>
        <div v-if="summary.best !== null">
          <dt>{{ t('runs.bestValShort') }}</dt>
          <dd>
            <p :title="String(summary.best)">
              <span class="sr-only">{{ t('runs.bestVal', { value: display(summary.best) }) }}</span
              ><span aria-hidden="true">{{ display(summary.best) }}</span>
            </p>
          </dd>
        </div>
      </dl>
    </div>
    <dl
      v-if="kpis.length"
      class="grid grid-cols-[repeat(auto-fit,_minmax(min(100%,_8.5rem),_1fr))] gap-3 m-0"
    >
      <div v-for="(metric, index) in kpis" :key="metric.label" :class="cn(styles.kpiTile)">
        <dt>{{ metric.label }}</dt>
        <dd :title="metric.exact == null ? undefined : String(metric.exact)">
          {{ metric.value }}
        </dd>
        <span
          v-if="metric.trend"
          :class="cn(styles.kpiTrend)"
          :data-tone="metric.trend.tone"
          role="img"
          :aria-label="metric.trend.label"
          ><TrendingUp v-if="metric.trend.rising" aria-hidden="true" /><TrendingDown
            v-else
            aria-hidden="true"
          /><span aria-hidden="true">{{ metric.trend.text }}</span></span
        >
        <svg
          v-if="metric.spark"
          class="sparkline [grid-area:spark] block w-full h-9 mt-[0.15rem] text-primary"
          viewBox="0 0 120 32"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <linearGradient :id="`${gradient}-${index}`" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stop-color="currentColor" stop-opacity="0.22" />
              <stop offset="100%" stop-color="currentColor" stop-opacity="0" />
            </linearGradient>
          </defs>
          <path
            v-if="metric.spark.area"
            :d="metric.spark.area"
            :fill="`url(#${gradient}-${index})`"
          />
          <path
            :d="metric.spark.line"
            fill="none"
            stroke="currentColor"
            stroke-width="1.75"
            stroke-linecap="round"
            stroke-linejoin="round"
            vector-effect="non-scaling-stroke"
          />
        </svg>
      </div>
    </dl>
    <details v-if="metrics.length > 4" :class="cn(styles.metricSpec)" open>
      <summary>{{ t('runs.allMetrics') }}</summary>
      <dl :class="cn(styles.metricSpecList)">
        <div v-for="metric in metrics.slice(4)" :key="metric.label">
          <dt>{{ metric.label }}</dt>
          <dd :title="metric.exact == null ? undefined : String(metric.exact)">
            {{ metric.value }}
          </dd>
        </div>
      </dl>
    </details>
    <p
      v-if="summary.latest && !summary.current"
      class="tabular-nums"
      :title="String(summary.latest.value)"
    >
      {{ summary.latest.name }}: {{ display(summary.latest.value) }} {{ summary.latest.unit }}
    </p>
    <div
      v-if="summary.current"
      class="grid grid-cols-[repeat(auto-fit,_minmax(min(100%,_16rem),_1fr))] gap-3"
    >
      <PerParameterMetrics :parameters="summary.current.parameters" />
      <MetricHealthPanel :health="summary.current.health" />
    </div>
    <Empty v-if="!summary.latest && !summary.progress" class="py-8">
      <EmptyDescription>{{ t('runs.waitingRuntime') }}</EmptyDescription>
    </Empty>
  </section>
</template>
