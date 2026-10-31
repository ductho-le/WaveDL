<script setup lang="ts">
import { cn } from '@/lib/utils';
import type { RuntimeMetricSummary } from '../../lib/metric-history';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  parameterRow: cn(
    'grid grid-cols-[minmax(0,_1fr)_auto] [gap:0.25rem_0.75rem] text-[0.82rem] [&_dt]:overflow-hidden',
    '[&_dt]:text-muted-foreground [&_dt]:text-ellipsis [&_dt]:whitespace-nowrap [&_dd]:m-0',
    '[&_dd]:font-[550] [&_dd]:tabular-nums',
  ),
};

const { t, locale } = useI18n({ useScope: 'global' });
const props = defineProps<{ parameters: RuntimeMetricSummary['parameters'] }>();

// Bars compare parameters with each other only; the largest error fills the track.
const largest = computed(() =>
  Math.max(
    0,
    ...props.parameters.map((parameter) =>
      parameter.value !== null && Number.isFinite(parameter.value) ? Math.abs(parameter.value) : 0,
    ),
  ),
);

function share(value: number | null): string {
  if (value === null || !largest.value) return '0%';

  return `${Math.max(2, (Math.abs(value) / largest.value) * 100)}%`;
}
</script>
<template>
  <section
    :aria-label="t('runs.perParam')"
    class="flex flex-col gap-[0.6rem] min-w-0 p-[0.95rem_1rem] rounded-[1rem] bg-[color-mix(in_srgb,_var(--foreground)_3.5%,_transparent)]"
  >
    <h4 class="text-[0.85rem] font-[650]">{{ t('runs.perParam') }}</h4>
    <dl class="flex flex-col gap-[0.55rem] m-0">
      <div v-for="parameter in parameters" :key="parameter.name" :class="cn(styles.parameterRow)">
        <dt>{{ parameter.name }}</dt>
        <dd :title="parameter.value === null ? undefined : String(parameter.value)">
          {{
            parameter.value === null
              ? t('runs.unavailable')
              : new Intl.NumberFormat(locale, { maximumSignificantDigits: 4 }).format(
                  parameter.value,
                )
          }}
        </dd>
        <span
          class="col-span-full h-[0.3rem] overflow-hidden rounded-full bg-[color-mix(in_srgb,_var(--foreground)_7%,_transparent)]"
          aria-hidden="true"
          ><span
            class="block h-full rounded-[inherit] bg-primary [transition:width_420ms_cubic-bezier(0.22,_1,_0.36,_1)]"
            :style="{ width: share(parameter.value) }"
        /></span>
      </div>
    </dl>
    <p v-if="!parameters.length" class="text-muted-foreground text-sm">
      {{ t('runs.noPerParam') }}
    </p>
  </section>
</template>
