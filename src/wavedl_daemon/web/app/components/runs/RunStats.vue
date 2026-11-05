<script setup lang="ts">
import { cn } from '@/lib/utils';
import { TrendingDown, TrendingUp } from '@lucide/vue';
import type { MetricRecord, MetricSummary } from '../../lib/metric-history';
import { buildPlotSeries, minimumPoint } from '../../lib/plot-series';

type StatChannel = 'train_loss' | 'val_loss' | 'r2' | 'lr';

const styles = {
  strip: cn('grid shrink-0 grid-cols-[repeat(auto-fit,minmax(7rem,1fr))] py-3'),
  stat: cn('min-w-0 border-l px-4 first:border-l-0 [&_dt]:truncate [&_dt]:text-xs [&_dt]:text-muted-foreground'),
  trend: cn(
    'inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[0.65rem] font-medium tabular-nums',
    'bg-muted text-muted-foreground [&_svg]:size-3',
    "[&[data-tone='good']]:bg-success/10 [&[data-tone='good']]:text-success",
    "[&[data-tone='bad']]:bg-warning/10 [&[data-tone='bad']]:text-warning",
  ),
};

const { t, locale } = useI18n({ useScope: 'global' });
const props = defineProps<{ summary: MetricSummary; records: readonly MetricRecord[] }>();

const channels: readonly { name: StatChannel; risingIsBetter: boolean | null }[] = [
  { name: 'val_loss', risingIsBetter: false },
  { name: 'train_loss', risingIsBetter: false },
  { name: 'r2', risingIsBetter: true },
  { name: 'lr', risingIsBetter: null },
];

function series(index: number): (number | null)[] {
  const channel = channels[index]?.name;

  if (!channel) return [];

  return props.records.slice(-60).map((record) => {
    const value = (channel === 'r2' ? record.metrics.r2 ?? record.metrics.val_r2 : record.metrics[channel])
      ?? (record.name === channel ? record.value : null);

    return value !== null && value !== undefined && Number.isFinite(value) ? value : null;
  });
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
  const change = channel.name === 'r2' || previous === 0
    ? latest - previous
    : (latest - previous) / Math.abs(previous);

  return {
    text: format.format(change),
    rising,
    tone: channel.risingIsBetter === null
      ? 'neutral'
      : rising === channel.risingIsBetter ? 'good' : 'bad',
  };
}

function display(value: number | null | undefined): string {
  return value == null
    ? t('runs.unavailable')
    : new Intl.NumberFormat(locale.value, { maximumSignificantDigits: 4 }).format(value);
}

const best = computed(() => {
  try {
    return minimumPoint(buildPlotSeries(props.records, 'loss'), 'val_loss');
  } catch {
    return null;
  }
});

const stats = computed(() => {
  const current = props.summary.current;
  const values = [
    { label: t('runs.valLoss'), value: current?.validationLoss, channel: 0 },
    { label: t('runs.trainLoss'), value: current?.trainLoss, channel: 1 },
    { label: t('runs.r2'), value: current?.r2, channel: 2 },
    { label: t('runs.bestValShort'), value: best.value?.value, epoch: best.value?.step },
    { label: t('runs.lr'), value: current?.learningRate, channel: 3 },
  ];

  return values.map((stat) => {
    const change = stat.channel === undefined ? null : trend(stat.channel);

    return {
      ...stat,
      text: display(stat.value),
      trend: change && {
        ...change,
        label: [
          t('runs.changeFromPrevious', { value: change.text }),
          change.tone === 'good' ? t('runs.improving') : change.tone === 'bad' ? t('runs.worsening') : '',
        ].filter(Boolean).join(', '),
      },
    };
  });
});
</script>
<template>
  <dl :class="styles.strip">
    <div v-for="stat in stats" :key="stat.label" :class="styles.stat">
      <dt :title="stat.label">{{ stat.label }}</dt>
      <dd class="mt-1 flex h-12 flex-wrap content-start items-baseline gap-x-2 gap-y-0.5">
        <span class="truncate text-lg font-semibold tracking-tight tabular-nums" :title="stat.value == null ? undefined : String(stat.value)">{{ stat.text }}</span>
        <span v-if="stat.epoch !== undefined" class="text-xs text-muted-foreground tabular-nums">{{ t('runs.atEpoch', { epoch: stat.epoch }) }}</span>
        <span v-if="stat.trend" :class="styles.trend" :data-tone="stat.trend.tone" role="img" :aria-label="stat.trend.label">
          <TrendingUp v-if="stat.trend.rising" aria-hidden="true" /><TrendingDown v-else aria-hidden="true" />
          <span aria-hidden="true">{{ stat.trend.text }}</span>
        </span>
      </dd>
    </div>
  </dl>
</template>
