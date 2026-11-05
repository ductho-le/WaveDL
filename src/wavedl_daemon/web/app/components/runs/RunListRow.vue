<script setup lang="ts">
import { cn } from '@/lib/utils';
import { CircleX } from '@lucide/vue';
import type { RunView } from '../../lib/daemon-api';
import { relativeTime, runCreatedAt, type RunColumn } from '../../lib/run-overview';
import { runStateMessageKeys } from '../../i18n/dynamic-keys';

const styles = {
  row: cn(
    'h-[38px] cursor-pointer text-sm transition-colors motion-reduce:transition-none',
    'hover:bg-muted/50 focus-within:outline-2 focus-within:-outline-offset-2 focus-within:outline-ring',
    'data-[selected=true]:bg-primary/10 data-[selected=true]:shadow-[inset_3px_0_0_var(--primary)]',
  ),
  cell: cn('h-[38px] min-w-0 overflow-hidden whitespace-nowrap border-b border-border/50 px-2'),
  glyph: cn(
    'relative block size-[0.55rem] shrink-0 rounded-full bg-foreground/20',
    'data-[state=running]:bg-primary data-[state=submitted]:bg-primary data-[state=queued]:bg-primary data-[state=pending]:bg-primary',
    'data-[state=completed]:bg-success data-[state=failed]:bg-destructive',
    'data-[state=held]:bg-warning data-[state=stopping]:bg-warning',
    'data-[state=submission_unknown]:bg-warning data-[state=terminal_unknown]:bg-warning data-[state=supervision_lost]:bg-warning',
    "[&[data-state='running']::after]:content-[''] [&[data-state='running']::after]:absolute",
    "[&[data-state='running']::after]:inset-[-3px] [&[data-state='running']::after]:rounded-[inherit]",
    "[&[data-state='running']::after]:border-2 [&[data-state='running']::after]:border-primary",
    "[&[data-state='running']::after]:[animation:run-glyph-pulse_1.8s_cubic-bezier(0.22,_1,_0.36,_1)_infinite]",
    "motion-reduce:[&[data-state='running']::after]:animate-none",
  ),
  state: cn(
    'flex items-center [&_.run-state-chip]:whitespace-nowrap [&_.run-state-chip]:flex-nowrap',
    '[&_.run-state-chip]:max-h-7 [&_.run-state-chip]:overflow-hidden',
  ),
};

const props = defineProps<{
  run: RunView;
  columns: readonly RunColumn[];
  folded: boolean;
  selected: boolean;
  current: boolean;
  stale: boolean;
  now: Date;
}>();

const emit = defineEmits<{
  select: [runId: string];
  move: [event: KeyboardEvent, step: -1 | 1];
}>();

const { t, locale } = useI18n({ useScope: 'global' });
const labels = useRunLabels();
const created = computed(() => runCreatedAt(props.run));
const exactTime = computed(() => created.value
  ? new Intl.DateTimeFormat(locale.value, { dateStyle: 'medium', timeStyle: 'short' }).format(created.value)
  : undefined);
const shortTime = computed(() => created.value ? relativeTime(created.value, props.now, locale.value, 'short') : '–');
const longTime = computed(() => created.value ? relativeTime(created.value, props.now, locale.value) : '–');
const stateLabel = computed(() => `${t(runStateMessageKeys[props.run.state])}${props.stale ? t('runs.stale') : ''}${props.run.uncertainty ? t('runs.uncertain') : ''}`);
const progressLabel = computed(() => props.run.progress
  ? t('runs.progressComplete', { percent: new Intl.NumberFormat(locale.value, { style: 'percent', maximumFractionDigits: 0 }).format(props.run.progress.fraction) })
  : t('runs.progressUnknown'));
const metadata = computed(() => props.columns.filter((column) => column.id !== 'name'));

function selectCell(event: MouseEvent) {
  if (event.target instanceof Element && event.target.closest('button, [role="note"], a, input')) return;
  emit('select', props.run.run_id);
}
</script>

<template>
  <tr :class="styles.row" :data-run-id="run.run_id" :data-selected="selected" @click="selectCell">
    <td class="p-0">
      <div :class="cn(styles.cell, 'flex items-center justify-center px-0')">
        <span :class="styles.glyph" :data-state="run.state" aria-hidden="true" />
        <span class="sr-only">{{ stateLabel }}</span>
      </div>
    </td>
    <td class="p-0">
      <div :class="cn(styles.cell, 'grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2')">
        <button
          type="button"
          class="run-row min-w-0 truncate text-left font-semibold tracking-[-0.01em] outline-none"
          :data-run-id="run.run_id"
          :aria-current="current ? 'true' : undefined"
          :title="labels.title(run)"
          @click="emit('select', run.run_id)"
          @keydown.down="emit('move', $event, 1)"
          @keydown.up="emit('move', $event, -1)"
        >
          {{ labels.title(run) }}<span class="sr-only">, {{ stateLabel }}</span>
        </button>
        <span data-trailing :aria-hidden="!folded" :class="cn('inline-flex shrink-0 items-center gap-1.5 text-[0.7rem] text-muted-foreground tabular-nums', stale && 'opacity-60')">
          <time class="shrink-0 whitespace-nowrap" :datetime="created?.toISOString()" :title="exactTime">{{ shortTime }}</time>
          <svg
            v-if="run.state === 'running'"
            viewBox="0 0 16 16"
            role="img"
            :aria-label="`${progressLabel}${stale ? t('runs.stale') : ''}`"
            :class="cn('size-4 shrink-0 text-primary', !run.progress && 'animate-spin motion-reduce:animate-none')"
          >
            <title>{{ progressLabel }}{{ stale ? t('runs.stale') : '' }}</title>
            <circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="2" class="opacity-20" />
            <circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" pathLength="1" :stroke-dasharray="`${run.progress?.fraction ?? 0.25} 1`" transform="rotate(-90 8 8)" />
          </svg>
          <CircleX v-else-if="run.state === 'failed'" class="size-4 shrink-0 text-destructive" role="img" :aria-label="t('runs.failed')" />
          <span v-else class="size-4 shrink-0" aria-hidden="true" />
        </span>
      </div>
    </td>
    <td v-for="column in metadata" :key="column.id" data-metadata class="p-0" :aria-hidden="folded" :inert="folded">
      <div v-if="column.id === 'state'" :class="cn(styles.cell, styles.state)">
        <RunStateChip :run="run" :stale="stale" />
      </div>
      <div v-else :class="cn(styles.cell, 'flex items-center text-muted-foreground')">
        <time v-if="column.id === 'created'" class="truncate" :datetime="created?.toISOString()" :title="exactTime">{{ longTime }}</time>
        <span v-else class="truncate" :title="String(column.value(run) ?? '–')">{{ column.value(run) ?? '–' }}</span>
      </div>
    </td>
  </tr>
</template>
