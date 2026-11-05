<script setup lang="ts">
import { cn } from '@/lib/utils';
import type { MetricSummary } from '../../lib/metric-history';

const styles = {
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
}>();

const numberFormat = computed(() => new Intl.NumberFormat(locale.value, { maximumSignificantDigits: 4 }));

const metrics = computed(() => {
  const current = props.summary.current;

  if (!current) return [];

  return [
    { label: t('runs.trainLoss'), exact: current.trainLoss },
    { label: t('runs.valLoss'), exact: current.validationLoss },
    { label: t('runs.r2'), exact: current.r2 },
    { label: t('runs.lr'), exact: current.learningRate },
    { label: t('runs.pearson'), exact: current.pearson },
    { label: t('runs.gradNorm'), exact: current.gradientNorm },
    { label: t('runs.mae'), exact: current.mae },
    { label: t('runs.epochDuration'), exact: current.epochSeconds, suffix: ' s' },
    { label: t('runs.bestR2'), exact: current.bestR2 },
    { label: t('runs.elapsed'), exact: current.elapsedSeconds, suffix: ' s' },
    {
      label: t('runs.patience'),
      exact: current.patience ? `${current.patience.used} / ${current.patience.maximum}` : null,
    },
  ].map((metric) => ({
    ...metric,
    value: metric.exact == null
      ? t('runs.unavailable')
      : typeof metric.exact === 'number'
        ? `${numberFormat.value.format(metric.exact)}${metric.suffix ?? ''}`
        : metric.exact,
  }));
});
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
    <details v-if="metrics.length" :class="cn(styles.metricSpec)" open>
      <summary>{{ t('runs.allMetrics') }}</summary>
      <dl :class="cn(styles.metricSpecList)">
        <div v-for="metric in metrics" :key="metric.label">
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
      {{ summary.latest.name }}: {{ summary.latest.value == null ? t('runs.unavailable') : numberFormat.format(summary.latest.value) }} {{ summary.latest.unit }}
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
