<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import { ChevronLeft, ChevronRight, RotateCw, Search, SearchX } from '@lucide/vue';
import type { RunView } from '../../lib/daemon-api';
import {
  groupRuns,
  runCounts,
  runFilterOf,
  runMatches,
  type RunFilter,
  type RunGroupKey,
} from '../../lib/run-overview';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  runList: cn(
    'flex flex-col gap-3 min-w-0 max-w-full p-[1rem_0.75rem_0.75rem] rounded-[1.25rem] bg-surface',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_12%,_transparent),_0_1px_2px_rgb(0_0_0_/_0.04),_0_18px_40px_-28px_rgb(0_0_0_/_0.35)]',
  ),
  runSearchInput: cn(
    'w-full h-9 p-[0_0.75rem_0_2.15rem] border-0 rounded-[0.7rem] text-foreground text-sm',
    'bg-[color-mix(in_srgb,_var(--foreground)_6%,_transparent)] [outline:none]!',
    '[transition:box-shadow_160ms_ease] [&::placeholder]:text-muted-foreground',
    '[&:focus-visible]:shadow-[0_0_0_3px_color-mix(in_srgb,_var(--primary)_45%,_transparent)]',
  ),
  runFilter: cn(
    '[&_.segmented-control]:flex [&_.segmented-control]:w-full [&_.segmented-control-item]:flex-1',
    '[&_.segmented-control-item]:px-[0.4rem] [&_.segmented-control-item]:tabular-nums',
  ),
  runListNone: cn(
    'flex flex-col items-center gap-[0.35rem] p-[2rem_1rem] text-muted-foreground text-[0.85rem]',
    'text-center [&_svg]:w-6 [&_svg]:h-6 [&_svg]:mb-1',
  ),
  runItem: cn(
    'relative grid grid-cols-[auto_minmax(0,_1fr)_auto] grid-rows-[auto_auto_auto] items-center',
    '[gap:0.12rem_0.6rem] min-w-0 p-[0.6rem_0.65rem] rounded-[0.8rem]',
    '[transition:background-color_140ms_ease]',
    '[&:hover]:bg-[color-mix(in_srgb,_var(--foreground)_4.5%,_transparent)]',
    "[&[data-selected='true']:hover]:bg-transparent",
  ),
  runSelection: cn(
    'absolute inset-0 rounded-[inherit] bg-[color-mix(in_srgb,_var(--primary)_11%,_var(--surface))]',
    'shadow-[inset_0_0_0_1px_color-mix(in_srgb,_var(--primary)_22%,_transparent)]',
  ),
  runRow: cn(
    'static grid col-span-full [grid-row:1_/_-1] grid-cols-subgrid grid-rows-subgrid items-center min-w-0',
    "p-0 border-0 text-[inherit] text-left [background:none] cursor-pointer [&::after]:[content:'']",
    '[&::after]:absolute [&::after]:inset-0 [&::after]:rounded-[0.8rem] [&:focus-visible]:[outline:none]!',
    '[&:focus-visible::after]:[outline:2px_solid_var(--ring)]!',
    '[&:focus-visible::after]:[outline-offset:-2px]!',
  ),
  runItemState: cn(
    'relative z-1 col-start-3 row-start-1 [transform-origin:right_center] [scale:0.9]',
    '[&_.run-state-chip]:whitespace-nowrap [&_.run-state-chip]:flex-nowrap',
  ),
  runGlyph: cn(
    'relative col-start-1 row-start-1 w-[0.55rem] h-[0.55rem] rounded-full',
    "bg-[color-mix(in_srgb,_var(--foreground)_22%,_transparent)] [&[data-state='running']]:bg-primary",
    "[&[data-state='submitted']]:bg-primary [&[data-state='queued']]:bg-primary",
    "[&[data-state='pending']]:bg-primary [&[data-state='running']::after]:[content:'']",
    "[&[data-state='running']::after]:absolute [&[data-state='running']::after]:inset-[-3px]",
    "[&[data-state='running']::after]:rounded-[inherit]",
    "[&[data-state='running']::after]:[border:2px_solid_var(--primary)]!",
    "[&[data-state='running']::after]:[animation:run-glyph-pulse_1.8s_cubic-bezier(0.22,_1,_0.36,_1)_infinite]",
    "[&[data-state='completed']]:bg-success [&[data-state='failed']]:bg-destructive",
    "[&[data-state='held']]:bg-warning [&[data-state='stopping']]:bg-warning",
    String.raw`[&[data-state='submission\_unknown']]:bg-warning [&[data-state='terminal\_unknown']]:bg-warning`,
    String.raw`[&[data-state='supervision\_lost']]:bg-warning`,
    "motion-reduce:[&[data-state='running']::after]:animate-none",
  ),
  identity: cn(
    'contents [&_>_*]:relative [&_strong]:col-start-2 [&_strong]:row-start-1 [&_strong]:min-w-0',
    '[&_strong]:overflow-hidden [&_strong]:text-[0.9rem] [&_strong]:font-semibold',
    '[&_strong]:tracking-[-0.01em] [&_strong]:text-ellipsis [&_strong]:whitespace-nowrap',
    '[&_small]:[grid-column:2_/_-1] [&_small]:row-start-3 [&_small]:min-w-0 [&_small]:overflow-hidden',
    '[&_small]:text-[color-mix(in_srgb,_var(--muted-foreground)_80%,_transparent)]',
    "[&_small]:[font:0.68rem/1.4_ui-monospace,_'SF_Mono',_monospace] [&_small]:text-ellipsis",
    '[&_small]:whitespace-nowrap',
  ),
  runListFooter: cn(
    'flex flex-wrap items-center justify-between gap-2 p-[0.5rem_0.25rem_0]',
    '[border-top:1px_solid_color-mix(in_srgb,_var(--border)_70%,_transparent)]! [&:empty]:hidden',
  ),
};

const { t, locale } = useI18n({ useScope: 'global' });
const labels = useRunLabels();

const props = defineProps<{
  runs: readonly RunView[];
  loading: boolean;
  page: number;
  staleRunIds?: ReadonlySet<string>;
  lastRefreshed?: string | null;
  selectedRunId?: string | null;
  now?: Date;
}>();

defineEmits<{ select: [runId: string]; refresh: []; page: [offset: number] }>();

const query = defineModel<string>('query', { default: '' });
const filter = defineModel<RunFilter>('filter', { default: 'all' });
const list = useTemplateRef<HTMLElement>('list');

const updatedLabel = computed(() => {
  const date = new Date(props.lastRefreshed ?? '');

  return Number.isNaN(date.getTime())
    ? t('runs.unavailable')
    : t('runs.updatedTime', {
        time: new Intl.DateTimeFormat(locale.value === 'vi' ? 'vi-VN' : 'en-US', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }).format(date),
      });
});

const counts = computed(() => runCounts(props.runs));

const filterOptions = computed(() =>
  (
    [
      ['all', t('runs.filterAll')],
      ['live', t('runs.filterLive')],
      ['completed', t('runs.filterDone')],
      ['attention', t('runs.filterAttention')],
    ] as const
  ).map(([value, label]) => ({
    value,
    label: counts.value[value] ? `${label} ${counts.value[value]}` : label,
  })),
);

const groupLabels = computed<Record<RunGroupKey, string>>(() => ({
  live: t('runs.groupLive'),
  today: t('runs.groupToday'),
  yesterday: t('runs.groupYesterday'),
  week: t('runs.groupWeek'),
  earlier: t('runs.groupEarlier'),
}));

const visible = computed(() =>
  props.runs.filter(
    (run) =>
      (filter.value === 'all' || runFilterOf(run) === filter.value) &&
      runMatches(run, query.value, [labels.operation(run)]),
  ),
);

const clock = computed(() => props.now ?? new Date());
const groups = computed(() => groupRuns(visible.value, clock.value));
const filtered = computed(() => filter.value !== 'all' || query.value.trim() !== '');

function clearFilters() {
  query.value = '';
  filter.value = 'all';
}

/** Arrow keys move between rows like a sidebar; Enter and Space stay native. */
function move(event: KeyboardEvent, step: number) {
  const rows = Array.from(list.value?.querySelectorAll<HTMLButtonElement>('button.run-row') ?? []);
  const index = rows.findIndex((row) => row === event.currentTarget);
  const next = rows[Math.min(rows.length - 1, Math.max(0, index + step))];

  if (!next) return;
  event.preventDefault();
  next.focus();
}

function enterList(event: KeyboardEvent) {
  const first = list.value?.querySelector<HTMLButtonElement>('button.run-row');

  if (!first) return;
  event.preventDefault();
  first.focus();
}
</script>
<template>
  <section :class="cn(styles.runList)" :aria-label="t('runs.runs')">
    <header class="flex items-center justify-between gap-2 px-[0.35rem]">
      <h2 class="flex items-baseline gap-2 text-[1.15rem] font-[650] tracking-[-0.02em]">
        {{ t('runs.runs')
        }}<span
          v-if="runs.length"
          class="text-muted-foreground text-[0.8rem] font-medium tabular-nums"
          >{{ runs.length }}</span
        >
      </h2>
      <Button
        variant="ghost"
        size="icon"
        class="rounded-full text-muted-foreground"
        :disabled="loading"
        :aria-label="t('runs.refresh')"
        :title="t('runs.refresh')"
        @click="$emit('refresh')"
        ><RotateCw
          :class="{ 'animate-spin motion-reduce:animate-none': loading }"
          aria-hidden="true"
      /></Button>
    </header>
    <label class="relative flex items-center">
      <Search
        class="absolute left-[0.7rem] w-[0.95rem] h-[0.95rem] text-muted-foreground pointer-events-none"
        aria-hidden="true"
      />
      <span class="sr-only">{{ t('runs.searchRuns') }}</span>
      <input
        v-model="query"
        type="search"
        :class="cn(styles.runSearchInput)"
        :placeholder="t('runs.searchPlaceholder')"
        :aria-describedby="runs.length ? 'run-list-keys' : undefined"
        @keydown.down="enterList"
      />
    </label>
    <div :class="cn(styles.runFilter)">
      <SegmentedControl v-model="filter" :options="filterOptions" :label="t('runs.filterRuns')" />
    </div>
    <p id="run-list-keys" class="sr-only">{{ t('runs.keyboardHint') }}</p>
    <p
      v-if="loading && !runs.length"
      role="status"
      class="p-[1rem_0.5rem] text-muted-foreground text-sm"
    >
      {{ t('runs.loadingRuns') }}
    </p>
    <Empty v-else-if="!runs.length" role="status" class="py-8"
      ><EmptyDescription>{{ t('runs.noRuns') }}</EmptyDescription></Empty
    >
    <div v-else-if="!visible.length" role="status" :class="cn(styles.runListNone)">
      <SearchX aria-hidden="true" />
      <p>{{ t('runs.noMatches') }}</p>
      <Button v-if="filtered" variant="link" size="sm" @click="clearFilters">{{
        t('runs.clearFilters')
      }}</Button>
    </div>
    <div
      ref="list"
      class="flex flex-auto flex-col gap-3 min-h-0 -mx-1 px-1 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
    >
      <LayoutGroup>
        <section
          v-for="group in groups"
          :key="group.key"
          :aria-labelledby="`run-group-${group.key}`"
        >
          <h3
            :id="`run-group-${group.key}`"
            class="sticky top-0 z-2 p-[0.35rem_0.5rem_0.3rem] text-muted-foreground text-[0.68rem] font-[650] tracking-[0.07em] uppercase bg-surface"
          >
            {{ groupLabels[group.key] }}
          </h3>
          <ul class="items flex flex-col gap-0.5 m-0 p-0 list-none min-w-0">
            <li
              v-for="run in group.runs"
              :key="run.run_id"
              :class="cn(styles.runItem)"
              :data-selected="run.run_id === selectedRunId ? 'true' : undefined"
            >
              <motion.span
                v-if="run.run_id === selectedRunId"
                layout-id="run-list-selection"
                :class="cn(styles.runSelection)"
                :transition="{ type: 'spring', stiffness: 520, damping: 40 }"
                aria-hidden="true"
              />
              <button
                type="button"
                :class="cn('run-row', styles.runRow)"
                :aria-current="run.run_id === selectedRunId ? 'true' : undefined"
                @click="$emit('select', run.run_id)"
                @keydown.down="move($event, 1)"
                @keydown.up="move($event, -1)"
              >
                <span :class="cn(styles.runGlyph)" :data-state="run.state" aria-hidden="true" />
                <span :class="cn('identity', styles.identity)"
                  ><strong>{{ labels.title(run) }}</strong
                  ><span
                    class="[grid-column:2_/_-1] row-start-2 min-w-0 overflow-hidden text-muted-foreground text-[0.76rem] text-ellipsis whitespace-nowrap"
                    >{{ labels.meta(run, clock) }}</span
                  ><small>{{ run.run_id }}</small></span
                >
              </button>
              <span :class="cn(styles.runItemState)"
                ><RunStateChip :run="run" :stale="staleRunIds?.has(run.run_id)"
              /></span>
            </li>
          </ul>
        </section>
      </LayoutGroup>
    </div>
    <footer :class="cn(styles.runListFooter)">
      <InfoChip
        v-if="lastRefreshed"
        :label="updatedLabel"
        :text="t('runs.listSnapshot', { time: lastRefreshed })"
      />
      <div
        v-if="page > 0 || runs.length >= 50"
        class="flex items-center gap-1 text-muted-foreground text-[0.8rem] tabular-nums"
        data-testid="run-pagination"
      >
        <Button
          variant="ghost"
          size="icon"
          :disabled="loading || page === 0"
          :aria-label="t('runs.previous')"
          :title="t('runs.previous')"
          @click="$emit('page', Math.max(0, page - 50))"
          ><ChevronLeft aria-hidden="true" /></Button
        ><span>{{ t('runs.page', { page: Math.floor(page / 50) + 1 }) }}</span
        ><Button
          variant="ghost"
          size="icon"
          :disabled="loading || runs.length < 50"
          :aria-label="t('runs.next')"
          :title="t('runs.next')"
          @click="$emit('page', page + 50)"
          ><ChevronRight aria-hidden="true"
        /></Button>
      </div>
    </footer>
  </section>
</template>
