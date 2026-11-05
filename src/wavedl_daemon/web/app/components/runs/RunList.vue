<script setup lang="ts">
import { cn } from '@/lib/utils';
import { AnimatePresence, motion, useReducedMotion } from 'motion-v';
import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, RotateCw, Search, SearchX } from '@lucide/vue';
import type { RunView } from '../../lib/daemon-api';
import { runStateMessageKeys } from '../../i18n/dynamic-keys';
import {
  groupRuns, pathName, runCounts, runCreatedAt, runFilterOf, runMatches,
  type RunColumn, type RunColumnId, type RunColumnRegistry, type RunFilter,
  type RunGroupKey, type RunSort,
} from '../../lib/run-overview';

const FOLD_DURATION = 340;
const styles = {
  root: cn(
    'flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-hidden',
    '[&_[data-metadata]]:transition-opacity [&_[data-metadata]]:duration-[120ms] [&_[data-metadata]]:delay-[220ms]',
    '[&_[data-trailing]]:opacity-0 [&_[data-trailing]]:transition-opacity [&_[data-trailing]]:duration-[80ms]',
    '[&[data-compact=true]_[data-metadata]]:opacity-0 [&[data-compact=true]_[data-metadata]]:delay-0',
    '[&[data-compact=true]_[data-trailing]]:opacity-100 [&[data-compact=true]_[data-trailing]]:delay-[260ms]',
    '[&[data-stacked=true]_[data-metadata]]:hidden',
    '[&[data-animate=false]_[data-metadata]]:transition-none [&[data-animate=false]_[data-trailing]]:transition-none',
    'motion-reduce:[&_[data-metadata]]:transition-none motion-reduce:[&_[data-trailing]]:transition-none',
  ),
  search: cn(
    'h-9 w-full rounded-[0.7rem] border-0 bg-foreground/5 py-0 pr-3 pl-[2.15rem] text-sm text-foreground outline-none',
    'placeholder:text-muted-foreground focus-visible:ring-[3px] focus-visible:ring-ring/45',
  ),
  card: cn(
    'flex min-h-0 shrink-0 flex-col overflow-hidden rounded-[1.25rem] border border-border/60 bg-surface',
    'shadow-[0_1px_2px_rgb(0_0_0_/_0.04)]',
  ),
  header: cn('sticky top-0 z-2 h-8 bg-surface text-left text-xs font-semibold text-muted-foreground'),
  footer: cn('flex h-11 shrink-0 items-center justify-center gap-1 overflow-hidden border-t border-border/60 px-2 text-xs text-muted-foreground tabular-nums whitespace-nowrap'),
};

type CollectionColumns = { [Key in RunColumnId]: RunColumnRegistry[Key] & { weight: number } };
type PendingFocus = { kind: 'detail'; runId: string } | { kind: 'row'; runId: string };

const props = defineProps<{
  runs: readonly RunView[];
  loading: boolean;
  page: number;
  folded: boolean;
  stacked: boolean;
  selectedRunId: string | null;
  staleRunIds?: ReadonlySet<string>;
  lastRefreshed?: string | null;
  now: Date;
}>();
const emit = defineEmits<{
  select: [runId: string];
  refresh: [];
  page: [offset: number];
  focusDetail: [];
}>();
defineSlots<{ detail(): unknown }>();

const { t, locale } = useI18n({ useScope: 'global' });
const labels = useRunLabels();
const { isActive } = usePageActive();
const reducedMotion = useReducedMotion();
const query = shallowRef('');
const filter = shallowRef<RunFilter>('all');
const sort = shallowRef<RunSort>({ column: 'created', direction: 'desc' });
const stage = useTemplateRef<HTMLElement>('stage');
const list = useTemplateRef<HTMLElement>('list');
const search = useTemplateRef<HTMLInputElement>('search');
const detailRegion = useTemplateRef<HTMLElement>('detailRegion');
const { width } = useElementSize(() => isActive.value ? stage.value : null);
const gutter = shallowRef(0);
const entered = shallowRef(false);
const resizing = shallowRef(false);
const highlighted = shallowRef<string | null>(null);
let resizeFrame: number | undefined;
let focusTimer: ReturnType<typeof setTimeout> | undefined;
let highlightTimer: ReturnType<typeof setTimeout> | undefined;
let pendingFocus: PendingFocus | null = null;

const compact = computed(() => props.folded || props.stacked);
const animate = computed(() => entered.value && isActive.value && !resizing.value && !props.stacked && !reducedMotion.value);
const listWidth = computed(() => props.stacked ? width.value : Math.round(Math.max(250, width.value * 0.3)));
const cardWidth = computed(() => props.folded && !props.stacked ? listWidth.value : width.value);
const detailWidth = computed(() => props.stacked ? width.value : Math.max(0, width.value - listWidth.value - 16));
const tableWidth = computed(() => Math.max(0, width.value - 2 - gutter.value));
const identityWidth = computed(() => Math.max(0, listWidth.value - 2 - gutter.value - 28));
const metadataWidth = computed(() => Math.max(0, tableWidth.value - identityWidth.value - 28));
const cardTransition = computed(() => ({ duration: animate.value ? 0.25 : 0, delay: animate.value ? 0.09 : 0, ease: [0.22, 1, 0.36, 1] }));
const seamTransition = computed(() => ({ duration: animate.value ? 0.17 : 0, delay: animate.value ? 0.17 : 0, ease: [0.22, 1, 0.36, 1] }));
const seamExit = computed(() => ({
  opacity: [null, 0, 0], x: [null, 32, 32],
  transition: { duration: animate.value ? 0.34 : 0, times: [0, 0.35, 1], ease: [0.22, 1, 0.36, 1] },
}));

const registry = computed<CollectionColumns>(() => ({
  name: { id: 'name', labelKey: 'runs.columnName', kind: 'text', initialDirection: 'asc', weight: 0, value: labels.title },
  model: { id: 'model', labelKey: 'runs.columnModel', kind: 'text', initialDirection: 'asc', weight: 14, value: (run) => run.summary?.model || null },
  dataset: { id: 'dataset', labelKey: 'runs.columnDataset', kind: 'text', initialDirection: 'asc', weight: 18, value: (run) => pathName(run.summary?.data_path) || null },
  operation: { id: 'operation', labelKey: 'runs.columnType', kind: 'text', initialDirection: 'asc', weight: 10, value: labels.operation },
  epochs: { id: 'epochs', labelKey: 'runs.columnEpochs', kind: 'number', initialDirection: 'asc', weight: 8, value: (run) => run.summary?.epochs ?? null },
  cv: { id: 'cv', labelKey: 'runs.columnCv', kind: 'number', initialDirection: 'asc', weight: 6, value: (run) => run.summary?.cv ?? null },
  created: { id: 'created', labelKey: 'runs.columnCreated', kind: 'number', initialDirection: 'desc', weight: 20, value: (run) => runCreatedAt(run)?.getTime() ?? null },
  state: { id: 'state', labelKey: 'runs.columnState', kind: 'text', initialDirection: 'asc', weight: 24, value: (run) => t(runStateMessageKeys[run.state]) },
}));
const columns = computed(() => Object.values(registry.value));
const collator = computed(() => new Intl.Collator(locale.value, { numeric: true, sensitivity: 'base' }));
const counts = computed(() => runCounts(props.runs));
const filterOptions = computed<{ value: RunFilter; label: string }[]>(() => [
  { value: 'all', label: `${t('runs.filterAll')} ${counts.value.all}` },
  { value: 'live', label: `${t('runs.filterLive')} ${counts.value.live}` },
  { value: 'completed', label: `${t('runs.filterDone')} ${counts.value.completed}` },
  { value: 'attention', label: `${t('runs.filterAttention')} ${counts.value.attention}` },
]);
const groupLabels = computed<Record<RunGroupKey, string>>(() => ({
  live: t('runs.groupLive'), today: t('runs.groupToday'), yesterday: t('runs.groupYesterday'), week: t('runs.groupWeek'), earlier: t('runs.groupEarlier'),
}));
const visible = computed(() => props.runs.filter((run) =>
  (filter.value === 'all' || runFilterOf(run) === filter.value) && runMatches(run, query.value, [labels.operation(run)]),
));
const grouped = computed(() => sort.value.column === 'created' && sort.value.direction === 'desc');
const groups = computed<{ key: RunGroupKey | null; runs: RunView[] }[]>(() => grouped.value
  ? groupRuns(visible.value, props.now)
  : [{ key: null, runs: [...visible.value].sort(compareRuns) }]);
const orderedRuns = computed(() => groups.value.flatMap((group) => group.runs));
const rows = computed(() => {
  const starts = new Map<string, string>();
  for (const group of groups.value) {
    const first = group.runs[0];
    if (first && group.key) starts.set(first.run_id, groupLabels.value[group.key]);
  }
  return orderedRuns.value.map((run) => ({ run, group: starts.get(run.run_id) }));
});
const filtered = computed(() => filter.value !== 'all' || query.value.trim() !== '');
const keyboardHintId = `${useId()}-run-keys`;
const updatedLabel = computed(() => {
  const date = new Date(props.lastRefreshed ?? '');
  return Number.isNaN(date.getTime()) ? t('runs.unavailable') : t('runs.updatedTime', {
    time: new Intl.DateTimeFormat(locale.value, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date),
  });
});

function compareRuns(a: RunView, b: RunView): number {
  const column = registry.value[sort.value.column];
  const left = column.value(a);
  const right = column.value(b);
  if (left === null || left === '') return right === null || right === '' ? a.run_id.localeCompare(b.run_id) : 1;
  if (right === null || right === '') return -1;
  let order = 0;
  if (column.kind === 'number') {
    const aValue = column.value(a);
    const bValue = column.value(b);
    if (aValue !== null && bValue !== null) order = aValue - bValue;
  } else {
    const aValue = column.value(a);
    const bValue = column.value(b);
    if (aValue !== null && bValue !== null) order = collator.value.compare(aValue, bValue);
  }
  return order * (sort.value.direction === 'asc' ? 1 : -1) || a.run_id.localeCompare(b.run_id);
}

function changeSort(column: RunColumn) {
  sort.value = { column: column.id, direction: sort.value.column === column.id
    ? sort.value.direction === 'asc' ? 'desc' : 'asc' : column.initialDirection };
}

function sortLabel(column: RunColumn): string {
  const direction = sort.value.column === column.id
    ? sort.value.direction === 'asc' ? 'desc' : 'asc' : column.initialDirection;
  return t(direction === 'asc' ? 'runs.sortAscending' : 'runs.sortDescending', { column: t(column.labelKey) });
}

function clearFilters() {
  query.value = '';
  filter.value = 'all';
}

function rowButton(runId: string): HTMLButtonElement | undefined {
  return Array.from(list.value?.querySelectorAll<HTMLButtonElement>('button.run-row') ?? []).find((row) => row.dataset.runId === runId);
}

function focusRow(runId: string) {
  const row = rowButton(runId);
  const body = list.value;
  if (!row || !body) {
    search.value?.focus({ preventScroll: true });
    return;
  }
  const bounds = row.getBoundingClientRect();
  const viewport = body.getBoundingClientRect();
  if (bounds.top < viewport.top + 32) body.scrollTop -= viewport.top + 32 - bounds.top;
  else if (bounds.bottom > viewport.bottom) body.scrollTop += bounds.bottom - viewport.bottom;
  row.focus({ preventScroll: true });
}

function move(event: KeyboardEvent, step: -1 | 1) {
  if (!isActive.value || event.defaultPrevented || event.isComposing) return;
  const target = event.currentTarget;
  const id = target instanceof HTMLElement ? target.dataset.runId : undefined;
  const index = orderedRuns.value.findIndex((run) => run.run_id === id);
  const next = index < 0 ? undefined : orderedRuns.value[Math.min(orderedRuns.value.length - 1, Math.max(0, index + step))];
  if (!next) return;
  event.preventDefault();
  if (props.folded) emit('select', next.run_id);
  else focusRow(next.run_id);
}

function moveFromDetail(event: KeyboardEvent) {
  if (!isActive.value || !props.folded || event.defaultPrevented || event.isComposing || (event.key !== 'ArrowDown' && event.key !== 'ArrowUp')) return;
  if (!(event.target instanceof Element) || !detailRegion.value?.contains(event.target)) return;
  if (event.target.closest('input, textarea, select, button, a, [contenteditable], [aria-haspopup], [role="button"], [role="note"], [role="tab"], [role="slider"], [role="combobox"], [role="menu"], [role="listbox"], [data-slot="tooltip-trigger"], [data-slot="popover-content"], .uplot')) return;
  const index = orderedRuns.value.findIndex((run) => run.run_id === props.selectedRunId);
  const next = index < 0 ? undefined : orderedRuns.value[Math.min(orderedRuns.value.length - 1, Math.max(0, index + (event.key === 'ArrowDown' ? 1 : -1)))];
  if (!next) return;
  event.preventDefault();
  emit('select', next.run_id);
}

function enterList(event: KeyboardEvent) {
  const first = orderedRuns.value[0];
  if (!isActive.value || !first || event.defaultPrevented || event.isComposing) return;
  event.preventDefault();
  focusRow(first.run_id);
}

function clearFocusWork() {
  clearTimeout(focusTimer);
  clearTimeout(highlightTimer);
  pendingFocus = null;
  highlighted.value = null;
}

function onCardAnimationComplete() {
  const card = list.value?.parentElement;
  if (card && Math.abs(card.offsetWidth - cardWidth.value) < 1) settleFocus();
}

function settleFocus() {
  clearTimeout(focusTimer);
  const intent = pendingFocus;
  pendingFocus = null;
  if (!intent || !isActive.value) return;
  if (intent.kind === 'detail') {
    if (props.selectedRunId === intent.runId) emit('focusDetail');
  } else if (props.selectedRunId === null) {
    focusRow(intent.runId);
    highlighted.value = intent.runId;
    highlightTimer = setTimeout(() => { highlighted.value = null; }, 700);
  }
}

async function requestFocus(intent: PendingFocus, wait: boolean) {
  clearFocusWork();
  pendingFocus = intent;
  if (intent.kind === 'row') highlighted.value = intent.runId;
  await nextTick();
  if (pendingFocus !== intent) return;
  if (props.stacked && !props.folded && list.value && list.value.offsetWidth > 0) {
    gutter.value = Math.max(0, list.value.offsetWidth - list.value.clientWidth);
  }
  if (wait) focusTimer = setTimeout(settleFocus, FOLD_DURATION + 30);
  else settleFocus();
}

watch(() => props.selectedRunId, (id, previous) => {
  if (!id) detailRegion.value?.setAttribute('inert', '');
  else detailRegion.value?.removeAttribute('inert');
  if (id) void requestFocus({ kind: 'detail', runId: id }, previous === null && animate.value);
  else if (previous) void requestFocus({ kind: 'row', runId: previous }, animate.value);
}, { flush: 'sync' });

watch([width, () => props.stacked, isActive], async ([measured, , active], _previous, onCleanup) => {
  const intent = pendingFocus;
  clearFocusWork();
  if (resizeFrame !== undefined) cancelAnimationFrame(resizeFrame);
  let cancelled = false;
  onCleanup(() => {
    cancelled = true;
    if (resizeFrame !== undefined) cancelAnimationFrame(resizeFrame);
  });
  resizing.value = true;
  if (!active || measured <= 0) return;
  const firstEntry = !entered.value;
  await nextTick();
  if (cancelled) return;
  if (list.value && list.value.offsetWidth > 0) gutter.value = Math.max(0, list.value.offsetWidth - list.value.clientWidth);
  resizeFrame = requestAnimationFrame(() => {
    resizing.value = false;
    entered.value = true;
    if (intent) void requestFocus(intent, false);
    else if (firstEntry && props.selectedRunId) void requestFocus({ kind: 'detail', runId: props.selectedRunId }, false);
  });
});

onScopeDispose(() => {
  clearFocusWork();
  if (resizeFrame !== undefined) cancelAnimationFrame(resizeFrame);
});
</script>

<template>
  <section :class="styles.root" :aria-label="t('runs.runs')" :data-folded="folded" :data-compact="compact" :data-stacked="stacked" :data-animate="animate" @keydown="moveFromDetail">
    <header v-show="!(stacked && folded)" class="flex shrink-0 flex-wrap items-center gap-2">
      <label class="relative flex min-w-0 flex-[1_1_13rem] items-center">
        <Search class="pointer-events-none absolute left-[0.7rem] size-4 text-muted-foreground" aria-hidden="true" />
        <span class="sr-only">{{ t('runs.searchRuns') }}</span>
        <input ref="search" v-model="query" type="search" :class="styles.search" :placeholder="t('runs.searchPlaceholder')" :aria-describedby="keyboardHintId" @keydown.down="enterList" />
      </label>
      <div class="min-w-0 max-w-full [&_.segmented-control-item]:shrink-0 [&_.segmented-control-item]:tabular-nums">
        <SegmentedControl v-model="filter" :options="filterOptions" :label="t('runs.filterRuns')" />
      </div>
      <div class="ml-auto flex shrink-0 items-center gap-1">
        <InfoChip v-if="lastRefreshed" :label="updatedLabel" :text="t('runs.listSnapshot', { time: lastRefreshed })" />
        <Button variant="ghost" size="icon" class="rounded-full text-muted-foreground" :disabled="loading" :aria-label="t('runs.refresh')" :title="t('runs.refresh')" @click="emit('refresh')">
          <RotateCw :class="{ 'animate-spin motion-reduce:animate-none': loading }" aria-hidden="true" />
        </Button>
      </div>
      <p :id="keyboardHintId" class="sr-only">{{ t('runs.keyboardHint') }}</p>
    </header>
    <div ref="stage" class="relative flex min-h-0 min-w-0 flex-1 flex-nowrap overflow-hidden">
      <motion.div
        v-show="!(stacked && folded)"
        :class="styles.card"
        :initial="false"
        :animate="{ width: cardWidth || '100%' }"
        :transition="cardTransition"
        @animation-complete="onCardAnimationComplete"
      >
        <div ref="list" class="min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain [scrollbar-gutter:stable]">
          <table class="table-fixed border-collapse" :style="{ width: tableWidth ? `${tableWidth}px` : '100%' }">
            <caption class="sr-only">{{ t('runs.runs') }}</caption>
            <colgroup>
              <col style="width: 28px" />
              <col v-for="column in columns" :key="column.id" :data-metadata="column.id !== 'name' ? '' : undefined" :style="{ width: `${column.id === 'name' ? identityWidth : metadataWidth * column.weight / 100}px` }" />
            </colgroup>
            <thead :class="styles.header">
              <tr class="h-8">
                <th scope="col" class="h-8 p-0"><span class="sr-only">{{ t('runs.columnStatus') }}</span></th>
                <th v-for="column in columns" :key="column.id" scope="col" class="h-8 overflow-hidden p-0" :data-metadata="column.id !== 'name' ? '' : undefined" :aria-hidden="compact && column.id !== 'name'" :inert="compact && column.id !== 'name'" :aria-sort="sort.column === column.id ? sort.direction === 'asc' ? 'ascending' : 'descending' : 'none'">
                  <button type="button" class="flex h-8 w-full min-w-0 items-center gap-1 overflow-hidden px-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring" :aria-label="sortLabel(column)" :title="sortLabel(column)" @click="changeSort(column)">
                    <span class="truncate">{{ t(column.labelKey) }}</span>
                    <component :is="sort.direction === 'asc' ? ArrowUp : ArrowDown" v-if="sort.column === column.id" class="size-3 shrink-0" aria-hidden="true" />
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              <template v-for="entry in rows" :key="entry.run.run_id">
                <tr v-if="entry.group" class="h-6">
                  <td :colspan="stacked ? 2 : columns.length + 1" class="h-6 p-0"><div class="flex h-6 items-center overflow-hidden px-2 text-[0.65rem] font-semibold tracking-[0.07em] whitespace-nowrap text-muted-foreground uppercase">{{ entry.group }}</div></td>
                </tr>
                <RunListRow :run="entry.run" :columns="columns" :folded="compact" :selected="entry.run.run_id === selectedRunId || entry.run.run_id === highlighted" :current="entry.run.run_id === selectedRunId" :stale="staleRunIds?.has(entry.run.run_id) ?? false" :now="now" @select="emit('select', $event)" @move="move" />
              </template>
            </tbody>
          </table>
          <p v-if="loading && !runs.length" role="status" class="p-6 text-sm text-muted-foreground">{{ t('runs.loadingRuns') }}</p>
          <Empty v-else-if="!runs.length" role="status" class="py-8"><EmptyDescription>{{ t('runs.noRuns') }}</EmptyDescription></Empty>
          <div v-else-if="!visible.length" role="status" class="flex flex-col items-center gap-2 px-3 py-8 text-center text-sm text-muted-foreground">
            <SearchX class="size-6" aria-hidden="true" /><p>{{ t('runs.noMatches') }}</p>
            <Button v-if="filtered" variant="link" size="sm" @click="clearFilters">{{ t('runs.clearFilters') }}</Button>
          </div>
        </div>
        <footer :class="styles.footer">
          <div v-if="page > 0 || runs.length >= 50" class="flex shrink-0 items-center gap-1" data-testid="run-pagination">
            <Button variant="ghost" size="icon" class="size-8" :disabled="loading || page === 0" :aria-label="t('runs.previous')" :title="t('runs.previous')" @click="emit('page', Math.max(0, page - 50))"><ChevronLeft aria-hidden="true" /></Button>
            <span>{{ t('runs.page', { page: Math.floor(page / 50) + 1 }) }}</span>
            <Button variant="ghost" size="icon" class="size-8" :disabled="loading || runs.length < 50" :aria-label="t('runs.next')" :title="t('runs.next')" @click="emit('page', page + 50)"><ChevronRight aria-hidden="true" /></Button>
          </div>
        </footer>
      </motion.div>
      <AnimatePresence :initial="false">
        <motion.div v-if="folded" key="detail" data-testid="run-detail-seam" class="min-h-0 shrink-0 overflow-x-hidden overflow-y-auto overscroll-contain [scrollbar-gutter:stable]" :style="{ width: `${detailWidth}px`, marginLeft: stacked ? '0px' : '16px' }" :initial="{ opacity: animate ? 0 : 1, x: animate ? 32 : 0 }" :animate="{ opacity: 1, x: 0 }" :exit="seamExit" :transition="seamTransition">
          <div ref="detailRegion" class="min-w-0 px-px pb-1"><slot name="detail" /></div>
        </motion.div>
      </AnimatePresence>
    </div>
  </section>
</template>
