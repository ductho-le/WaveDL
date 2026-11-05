<script setup lang="ts">
import { ArrowLeft, FileStack, RefreshCw, Search } from '@lucide/vue';
import { cn } from '@/lib/utils';
import type { ConnectionSession } from '../../lib/connection-session';
import { CatalogApi, type BrowseEntry, type DataSelection } from '../../lib/catalog-api';
import type { DatasetInspection } from '../../lib/dataset-inspection';
import { pathName, relativeTime } from '../../lib/run-overview';
import type { DatasetBrowserColumn } from '../../composables/useDatasetBrowser';

const props = withDefaults(defineProps<{
  session: ConnectionSession;
  allowDirectory?: boolean;
  label?: string;
  selected?: DataSelection | null;
}>(), { allowDirectory: false, selected: null });
const emit = defineEmits<{
  select: [selection: DataSelection];
  inspected: [inspection: DatasetInspection | null, selection: DataSelection | null];
}>();
const { t, locale } = useI18n({ useScope: 'global' });
const { roots, columns, suggestions, candidate, busy, resolving, message, reload, browse, chooseFile, suggest } = useDatasetBrowser(
  () => props.session, () => props.allowDirectory,
);
const { info, busy: inspecting, failed } = useDatasetInspection(() => props.session, () => candidate.value);
const container = useTemplateRef<HTMLElement>('container');
const { width } = useElementSize(container);
const narrow = computed(() => width.value > 0 && width.value < 896);
const mobileColumn = shallowRef(0);
const previewOpen = shallowRef(false);
let previewTrigger: HTMLElement | null = null;
const query = shallowRef('');
let focusIntent = 0;
const id = useId();
const clock = useNow({ scheduler: (tick) => useIntervalFn(tick, 60_000) });
const label = computed(() => props.label ?? t('datasets.pathPickerTitle'));
const needle = computed(() => query.value.trim().toLocaleLowerCase());
const matchingSuggestions = computed(() => suggestions.value.filter((item) => !needle.value || item.path.toLocaleLowerCase().includes(needle.value)));
const locations = computed<BrowseEntry[]>(() => roots.value.map((name) => ({ name, kind: 'directory' })));
const activePage = computed(() => columns.value[mobileColumn.value - 1]?.page ?? null);
const { filteredPage, busy: filtering, failed: filterFailed, offset: filterOffset } = useDatasetFolderFilter(
  () => props.session, () => activePage.value, () => query.value,
);
const shownColumns = computed(() => columns.value.map((column, index) => {
  const page = index === mobileColumn.value - 1 && filteredPage.value ? filteredPage.value : column.page;

  return { ...column, page, entries: page?.entries ?? [] };
}));
const styles = {
  browser: cn('flex min-w-0 overflow-x-auto rounded-xl border bg-surface'),
  column: cn('h-[23rem] w-52 shrink-0 border-r'),
  suggestion: cn(
    'flex min-w-0 flex-col gap-3 rounded-xl border p-3.5 text-left transition-colors',
    'bg-[color-mix(in_srgb,var(--foreground)_2%,transparent)] hover:bg-[color-mix(in_srgb,var(--primary)_6%,transparent)]',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    'aria-pressed:border-primary aria-pressed:bg-[color-mix(in_srgb,var(--primary)_8%,transparent)]',
  ),
};

function child(column: DatasetBrowserColumn, name: string): string {
  return [column.path, name].filter(Boolean).join('/');
}

function selectedEntry(column: DatasetBrowserColumn, index: number): string | null {
  const next = columns.value[index + 1];

  if (next?.root === column.root) return next.path.slice(column.path ? column.path.length + 1 : 0).split('/')[0] ?? null;
  const current = candidate.value ?? props.selected;
  const parent = current?.relativePath.split('/').slice(0, -1).join('/');

  return current?.root === column.root && parent === column.path ? pathName(current.relativePath) : null;
}

async function focusColumn(index: number) {
  mobileColumn.value = Math.max(0, index);
  await nextTick();
  const pane = container.value?.querySelector<HTMLElement>(`[data-column="${mobileColumn.value}"]`);
  const option = pane?.querySelector<HTMLButtonElement>('[role="option"][tabindex="0"]');
  const target = option ?? pane?.querySelector<HTMLElement>('[role="listbox"][tabindex="0"]');
  target?.focus();
  pane?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

async function openFolder(root: string, path: string, index: number, offset = 0) {
  const intent = ++focusIntent;
  query.value = '';
  mobileColumn.value = index + 1;
  await browse(root, path, index, offset);

  if (focusIntent !== intent) return;
  await focusColumn(index + 1);
}

function choose(column: DatasetBrowserColumn, index: number, entry: BrowseEntry) {
  if (entry.kind === 'directory') void openFolder(column.root, child(column, entry.name), index + 1);
  else {
    focusIntent++;
    mobileColumn.value = index + 1;
    openPreview();
    void chooseFile(column.root, child(column, entry.name));
  }
}

function openPreview() {
  previewTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  previewOpen.value = true;
}

function restorePreviewFocus(event: Event) {
  if (!previewTrigger?.isConnected || !previewTrigger.getClientRects().length) return;
  event.preventDefault();
  previewTrigger.focus({ preventScroll: true });
}

function back(index: number) {
  focusIntent++;
  query.value = '';
  void focusColumn(index);
}

function paginate(column: DatasetBrowserColumn, index: number, offset: number) {
  if (needle.value && mobileColumn.value === index + 1) filterOffset.value = offset;
  else void openFolder(column.root, column.path, index, offset);
}

function confirm() {
  const selection = candidate.value;

  if (!selection || inspecting.value || resolving.value) return;

  try {
    new CatalogApi(toRaw(props.session)).assertSelection(selection);
  } catch {
    return;
  }

  previewOpen.value = false;
  emit('select', selection);
  emit('inspected', info.value, selection);
}

watch(roots, () => {
  mobileColumn.value = 0;
  query.value = '';
  previewOpen.value = false;
  focusIntent++;
});
watch(narrow, (value) => { if (!value) previewOpen.value = false; });
</script>

<template>
  <section ref="container" class="flex min-w-0 flex-col gap-4" :aria-label="label">
    <div class="flex items-center gap-2">
      <div class="relative min-w-0 flex-1">
        <Search class="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
        <Input :id="`${id}-filter`" v-model="query" type="search" class="h-10 rounded-xl pl-9" :aria-label="t('datasets.filterLabel')" :aria-describedby="`${id}-filter-help`" :placeholder="t('datasets.filterPlaceholder')" data-testid="dataset-path-search" />
      </div>
      <Button type="button" variant="ghost" size="icon" :aria-label="t('datasets.reloadRoots')" @click="reload"><RefreshCw aria-hidden="true" /></Button>
    </div>
    <p :id="`${id}-filter-help`" class="sr-only">{{ t('datasets.filterHelp') }}</p>
    <section v-if="matchingSuggestions.length" :aria-labelledby="`${id}-suggested`">
      <h3 :id="`${id}-suggested`" class="mb-2 text-xs font-semibold text-muted-foreground">{{ t('datasets.suggested') }}</h3>
      <div class="grid grid-cols-1 gap-2 min-[30rem]:grid-cols-2" :class="!narrow && 'grid-cols-4!'">
        <button v-for="item in matchingSuggestions" :key="item.path" type="button" :class="styles.suggestion" :aria-pressed="candidate?.path === item.selection.path" :title="item.path" @click="focusIntent++; suggest(item.selection); openPreview()">
          <span class="flex w-full items-center gap-2"><FileStack class="size-4 shrink-0 text-primary" aria-hidden="true" /><span class="min-w-0 flex-1 truncate text-sm font-medium">{{ item.name }}</span><Badge variant="outline" class="text-[0.6rem]">{{ item.format }}</Badge></span>
          <span class="text-xs text-muted-foreground">{{ t('datasets.usedInRuns', { count: item.count }, item.count) }}<template v-if="item.lastUsed"> · {{ relativeTime(item.lastUsed, clock, locale) }}</template></span>
        </button>
      </div>
    </section>
    <p v-if="message || filterFailed" role="alert" class="text-sm text-muted-foreground">{{ t('datasets.browseUnavailable') }}</p>
    <div v-if="narrow && mobileColumn > 0" class="flex items-center justify-between gap-2">
      <Button type="button" variant="ghost" size="sm" @click="back(mobileColumn - 1)"><ArrowLeft aria-hidden="true" />{{ t('datasets.back') }}</Button>
      <Button v-if="allowDirectory && candidate?.kind === 'directory'" type="button" variant="outline" size="sm" @click="openPreview">{{ t('datasets.useFolder') }}</Button>
    </div>
    <div :class="styles.browser">
      <DatasetBrowserColumn v-show="!narrow || mobileColumn === 0" data-column="0" :class="cn(styles.column, narrow && 'w-full border-r-0')" :title="t('datasets.locations')" :entries="locations" :selected="columns[0]?.root ?? null" locations :busy="busy && !roots.length" @choose="openFolder($event.name, '', 0)" />
      <DatasetBrowserColumn
        v-for="(column, index) in shownColumns" v-show="!narrow || mobileColumn === index + 1"
        :key="`${column.root}/${column.path}`" :data-column="index + 1"
        :class="cn(styles.column, narrow && 'w-full border-r-0')"
        :title="column.path ? pathName(column.path) : column.root" :entries="column.entries"
        :selected="selectedEntry(column, index)" :page="column.page" :busy="(busy && !column.page) || (filtering && mobileColumn === index + 1)"
        @choose="choose(column, index, $event)" @back="back(index)"
        @page="paginate(column, index, $event)"
      />
      <div v-if="!narrow" class="sticky right-0 min-h-[23rem] w-72 shrink-0 grow border-l bg-surface">
        <DatasetSelectionPreview :selection="candidate" :inspection="info" :busy="inspecting || resolving" :failed="failed" @confirm="confirm" />
      </div>
    </div>
    <p v-if="!roots.length && !busy && !message" class="text-sm text-muted-foreground" role="status">{{ t('datasets.noRoots') }}</p>
    <Sheet :open="narrow && previewOpen && (!!candidate || resolving)" @update:open="previewOpen = $event">
      <SheetContent side="bottom" class="max-h-[85dvh] overflow-y-auto rounded-t-2xl" @close-auto-focus="restorePreviewFocus">
        <SheetHeader class="sr-only"><SheetTitle>{{ t('datasets.filePreview') }}</SheetTitle><SheetDescription>{{ t('datasets.confirmHelp') }}</SheetDescription></SheetHeader>
        <DatasetSelectionPreview :selection="candidate" :inspection="info" :busy="inspecting || resolving" :failed="failed" @confirm="confirm" />
      </SheetContent>
    </Sheet>
  </section>
</template>
