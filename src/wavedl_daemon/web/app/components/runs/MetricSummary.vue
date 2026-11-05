<script setup lang="ts">
import { cn } from '@/lib/utils';
import type { MetricSummary } from '../../lib/metric-history';

const styles = {
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
}>();

const numberFormat = computed(
  () => new Intl.NumberFormat(locale.value, { maximumSignificantDigits: 4 }),
);

function formatted(value: number | null | undefined, suffix = ''): string {
  return value == null ? t('runs.unavailable') : `${numberFormat.value.format(value)}${suffix}`;
}

const metrics = computed(() => {
  const current = props.summary.current;

  if (!current) return [];

  const patience = current.patience
    ? `${current.patience.used} / ${current.patience.maximum}`
    : null;

  return [
    { label: t('runs.pearson'), exact: current.pearson, value: formatted(current.pearson) },
    {
      label: t('runs.gradNorm'),
      exact: current.gradientNorm,
      value: formatted(current.gradientNorm),
    },
    { label: t('runs.mae'), exact: current.mae, value: formatted(current.mae) },
    {
      label: t('runs.epochDuration'),
      exact: current.epochSeconds,
      value: formatted(current.epochSeconds, ' s'),
    },
    { label: t('runs.bestR2'), exact: current.bestR2, value: formatted(current.bestR2) },
    {
      label: t('runs.elapsed'),
      exact: current.elapsedSeconds,
      value: formatted(current.elapsedSeconds, ' s'),
    },
    { label: t('runs.patience'), exact: patience, value: patience ?? t('runs.unavailable') },
  ];
});
</script>
<template>
  <section
    class="@container flex min-w-0 flex-col gap-6"
    :aria-label="unconfirmed ? t('runs.lastKnownRuntime') : t('runs.currentRuntime')"
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
    <div class="grid grid-cols-[repeat(auto-fit,_minmax(min(100%,_16rem),_1fr))] gap-6">
      <PerParameterMetrics :parameters="summary.current?.parameters ?? []" />
      <MetricHealthPanel
        :health="summary.current?.health ?? []"
        class="border-t pt-6 @min-[33.5rem]:border-t-0 @min-[33.5rem]:border-l @min-[33.5rem]:pt-0 @min-[33.5rem]:pl-6"
      />
    </div>
    <section v-if="metrics.length" :aria-label="t('runs.allMetrics')">
      <h4 class="text-[0.85rem] font-[650]">{{ t('runs.allMetrics') }}</h4>
      <dl :class="cn(styles.metricSpecList)">
        <div v-for="metric in metrics" :key="metric.label">
          <dt>{{ metric.label }}</dt>
          <dd :title="metric.exact == null ? undefined : String(metric.exact)">
            {{ metric.value }}
          </dd>
        </div>
      </dl>
    </section>
    <p
      v-if="summary.latest && !summary.current"
      class="tabular-nums"
      :title="String(summary.latest.value)"
    >
      {{ summary.latest.name }}:
      {{
        summary.latest.value == null
          ? t('runs.unavailable')
          : numberFormat.format(summary.latest.value)
      }}
      {{ summary.latest.unit }}
    </p>
    <Empty v-if="!summary.latest && !summary.progress" class="py-8">
      <EmptyDescription>{{ t('runs.waitingRuntime') }}</EmptyDescription>
    </Empty>
  </section>
</template>
