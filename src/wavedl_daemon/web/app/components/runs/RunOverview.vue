<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import { Activity, CircleCheck, Layers, TriangleAlert } from '@lucide/vue';
import type { RunView } from '../../lib/daemon-api';
import { groupRuns, runCounts } from '../../lib/run-overview';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  runOverview: cn(
    'flex flex-col gap-5 min-w-0 p-[clamp(1.25rem,_3vw,_2.5rem)] rounded-[1.5rem]',
    '[background:radial-gradient(_70%_60%_at_50%_0%,_color-mix(in_srgb,_var(--primary)_9%,_transparent),_transparent_70%_),_color-mix(in_srgb,_var(--surface)_55%,_transparent)]',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_10%,_transparent)]',
  ),
  runOverviewTile: cn(
    '[--tone:var(--muted-foreground)] flex flex-col gap-[0.6rem] p-[1rem_1.1rem] rounded-[1.1rem]',
    'bg-surface',
    'batch-r-surface-shadow',
    "[&[data-tone='live']]:[--tone:var(--primary)] [&[data-tone='completed']]:[--tone:var(--success)]",
    "[&[data-tone='attention']]:[--tone:var(--warning)] [&_dt]:flex [&_dt]:items-center",
    '[&_dt]:gap-[0.4rem] [&_dt]:text-muted-foreground [&_dt]:text-[0.78rem] [&_dt]:font-[550]',
    '[&_dt_svg]:w-[0.95rem] [&_dt_svg]:h-[0.95rem] [&_dt_svg]:text-(--tone) [&_dd]:m-0 [&_dd]:text-[2rem]',
    '[&_dd]:font-[650] [&_dd]:tabular-nums [&_dd]:tracking-[-0.03em] [&_dd]:leading-[1]',
  ),
  runOverviewCard: cn(
    'relative flex flex-col gap-3 min-w-0 p-[1rem_1.1rem] rounded-[1.1rem] bg-surface',
    'batch-r-surface-shadow',
    '[transition:transform_220ms_cubic-bezier(0.22,_1,_0.36,_1),_box-shadow_220ms_ease]',
    '[&:hover]:[transform:translateY(-2px)]',
    '[&:hover]:shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_12%,_transparent),_0_14px_30px_-18px_rgb(0_0_0_/_0.35)]',
    'motion-reduce:transform-none motion-reduce:transition-none motion-reduce:[&:hover]:transform-none',
    'motion-reduce:[&:hover]:transition-none',
  ),
  runOverviewOpen: cn(
    'grid gap-[0.2rem] min-w-0 p-0 border-0 text-[inherit] text-left [background:none] cursor-pointer',
    "[&::after]:[content:''] [&::after]:absolute [&::after]:inset-0 [&::after]:rounded-[1.1rem]",
    '[&:focus-visible]:[outline:none]! [&:focus-visible::after]:[outline:2px_solid_var(--ring)]!',
    '[&:focus-visible::after]:[outline-offset:2px]! [&_strong]:overflow-hidden [&_strong]:font-semibold',
    '[&_strong]:text-ellipsis [&_strong]:whitespace-nowrap [&_span]:overflow-hidden',
    '[&_span]:text-muted-foreground [&_span]:text-[0.78rem] [&_span]:text-ellipsis',
    '[&_span]:whitespace-nowrap',
  ),
};

const { t } = useI18n({ useScope: 'global' });
const props = defineProps<{ runs: readonly RunView[]; loading: boolean; now: Date }>();

defineEmits<{ select: [runId: string] }>();

const labels = useRunLabels();
const counts = computed(() => runCounts(props.runs));

const tiles = computed(() => [
  { key: 'live', label: t('runs.groupLive'), value: counts.value.live, icon: Activity },
  {
    key: 'completed',
    label: t('runs.completed'),
    value: counts.value.completed,
    icon: CircleCheck,
  },
  {
    key: 'attention',
    label: t('runs.filterAttention'),
    value: counts.value.attention,
    icon: TriangleAlert,
  },
  { key: 'all', label: t('runs.filterAll'), value: counts.value.all, icon: Layers },
]);

// Live runs lead; otherwise the newest finished ones are the likeliest next look.
const spotlight = computed(() => {
  const groups = groupRuns(props.runs, props.now);
  const live = groups.find((group) => group.key === 'live');

  return {
    title: live ? t('runs.groupLive') : t('runs.mostRecent'),
    runs: (live?.runs ?? groups.flatMap((group) => group.runs)).slice(0, 4),
  };
});

const spring = { type: 'spring', stiffness: 260, damping: 26 } as const;
</script>
<template>
  <section :class="cn(styles.runOverview)" :aria-label="t('runs.overviewTitle')">
    <motion.header
      :initial="{ opacity: 0, y: 14 }"
      :animate="{ opacity: 1, y: 0 }"
      :transition="spring"
    >
      <h2 class="text-[clamp(1.4rem,_2.6vw,_1.85rem)] font-[650] tracking-[-0.03em]">
        {{ t('runs.overviewTitle') }}
      </h2>
      <p class="max-w-[44ch] mt-[0.35rem] text-muted-foreground">{{ t('runs.overviewHint') }}</p>
    </motion.header>
    <dl class="grid grid-cols-[repeat(auto-fit,_minmax(min(100%,_9.5rem),_1fr))] gap-3 m-0">
      <motion.div
        v-for="(tile, index) in tiles"
        :key="tile.key"
        :class="cn(styles.runOverviewTile)"
        :data-tone="tile.key"
        :initial="{ opacity: 0, y: 14 }"
        :animate="{ opacity: 1, y: 0 }"
        :transition="{ ...spring, delay: 0.04 * (index + 1) }"
      >
        <dt><component :is="tile.icon" aria-hidden="true" />{{ tile.label }}</dt>
        <dd>{{ loading && !runs.length ? '–' : tile.value }}</dd>
      </motion.div>
    </dl>
    <section v-if="spotlight.runs.length" aria-labelledby="run-overview-spotlight">
      <h3
        id="run-overview-spotlight"
        class="mb-[0.6rem] text-muted-foreground text-[0.7rem] font-[650] tracking-[0.08em] uppercase"
      >
        {{ spotlight.title }}
      </h3>
      <ul
        class="grid grid-cols-[repeat(auto-fill,_minmax(min(100%,_16rem),_1fr))] gap-3 m-0 p-0 list-none"
      >
        <motion.li
          v-for="(run, index) in spotlight.runs"
          :key="run.run_id"
          :class="cn(styles.runOverviewCard)"
          :initial="{ opacity: 0, y: 14 }"
          :animate="{ opacity: 1, y: 0 }"
          :transition="{ ...spring, delay: 0.2 + 0.05 * index }"
        >
          <button
            type="button"
            :class="cn(styles.runOverviewOpen)"
            @click="$emit('select', run.run_id)"
          >
            <strong>{{ labels.title(run) }}</strong>
            <span>{{ labels.meta(run, now) }}</span>
          </button>
          <span class="relative z-1 [align-self:flex-start]"><RunStateChip :run="run" /></span>
        </motion.li>
      </ul>
    </section>
  </section>
</template>
