<script setup lang="ts">
import { ChevronLeft, ChevronRight, File, Folder, HardDrive } from '@lucide/vue';
import { cn } from '@/lib/utils';
import { datasetFormat } from '../../lib/dataset-suggestions';
import type { BrowseEntry, BrowsePage } from '../../lib/catalog-api';

const props = defineProps<{
  title: string;
  entries: readonly BrowseEntry[];
  selected: string | null;
  locations?: boolean;
  page?: BrowsePage | null;
  busy?: boolean;
}>();
const emit = defineEmits<{
  choose: [entry: BrowseEntry];
  back: [];
  page: [offset: number];
}>();
const { t } = useI18n({ useScope: 'global' });
const list = useTemplateRef<HTMLElement>('list');
const active = shallowRef<string | null>(null);
const selectable = computed(() => props.entries.filter((entry) => enabled(entry)));
const tabStop = computed(() => {
  const preferred = active.value ?? props.selected;

  return selectable.value.find((entry) => entry.name === preferred)?.name ?? selectable.value[0]?.name;
});

const styles = {
  row: cn(
    'flex min-h-10 w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm',
    'transition-colors hover:bg-[color-mix(in_srgb,var(--foreground)_5%,transparent)]',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset',
    'aria-selected:bg-[color-mix(in_srgb,var(--primary)_12%,transparent)] aria-selected:text-primary',
    'disabled:cursor-default disabled:text-muted-foreground disabled:opacity-45 disabled:hover:bg-transparent',
  ),
};

function enabled(entry: BrowseEntry): boolean {
  return entry.kind === 'directory' || datasetFormat(entry.name) !== null;
}

function focus(name?: string) {
  const buttons = list.value?.querySelectorAll<HTMLButtonElement>('[role="option"]');
  const target = Array.from(buttons ?? []).find((button) => button.dataset.name === name && !button.disabled);
  target?.focus();
}

function keydown(event: KeyboardEvent, entry: BrowseEntry) {
  if (event.isComposing || event.defaultPrevented) return;
  const index = selectable.value.findIndex((item) => item.name === entry.name);
  let next: BrowseEntry | undefined;

  if (event.key === 'ArrowDown') next = selectable.value[Math.min(index + 1, selectable.value.length - 1)];
  else if (event.key === 'ArrowUp') next = selectable.value[Math.max(index - 1, 0)];
  else if (event.key === 'Home') next = selectable.value[0];
  else if (event.key === 'End') next = selectable.value.at(-1);
  else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    emit('back');
  } else if (event.key === 'ArrowRight' && entry.kind === 'directory') {
    event.preventDefault();
    emit('choose', entry);
  }

  if (next) {
    event.preventDefault();
    active.value = next.name;
    focus(next.name);
  }
}

</script>

<template>
  <section class="flex min-w-0 flex-col" :aria-label="title">
    <h3 class="truncate border-b px-3 py-3 text-xs font-semibold text-muted-foreground" :title="title">{{ title }}</h3>
    <div class="min-h-64 flex-1 overflow-y-auto p-1.5">
      <div v-if="busy" class="space-y-3 p-2" role="status" :aria-label="t('datasets.browseLoading')">
        <Skeleton v-for="index in 5" :key="index" class="h-7 w-full" />
      </div>
      <div v-else ref="list" role="listbox" :aria-label="title" :aria-busy="busy" :tabindex="selectable.length ? undefined : 0" @keydown.left.self.prevent="emit('back')">
        <button
          v-for="entry in entries"
          :key="entry.name"
          type="button"
          role="option"
          :data-name="entry.name"
          :aria-selected="entry.name === selected"
          :aria-disabled="!enabled(entry)"
          :disabled="!enabled(entry)"
          :tabindex="entry.name === tabStop ? 0 : -1"
          :class="styles.row"
          :title="entry.name"
          @focus="active = entry.name"
          @keydown="keydown($event, entry)"
          @click="emit('choose', entry)"
        >
          <HardDrive v-if="locations" class="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          <Folder v-else-if="entry.kind === 'directory'" class="size-4 shrink-0 text-primary" aria-hidden="true" />
          <File v-else class="size-4 shrink-0" aria-hidden="true" />
          <span class="min-w-0 flex-1 truncate">{{ entry.name }}</span>
          <ChevronRight v-if="entry.kind === 'directory'" class="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
        </button>
        <p v-if="!entries.length" class="px-2 py-5 text-sm text-muted-foreground" role="status">{{ t('datasets.emptyDirectory') }}</p>
      </div>
    </div>
    <nav v-if="page && page.total > page.limit" class="flex items-center justify-between gap-1 border-t p-2" :aria-label="t('datasets.entries')">
      <Button type="button" variant="ghost" size="icon-sm" :disabled="busy || page.offset === 0" :aria-label="t('datasets.previousEntries')" @click="emit('page', Math.max(0, page.offset - page.limit))"><ChevronLeft aria-hidden="true" /></Button>
      <span class="text-[0.65rem] tabular-nums text-muted-foreground">{{ t('datasets.entryCount', { start: page.offset + 1, end: Math.min(page.offset + page.limit, page.total), total: page.total }) }}</span>
      <Button type="button" variant="ghost" size="icon-sm" :disabled="busy || page.offset + page.limit >= page.total" :aria-label="t('datasets.nextEntries')" @click="emit('page', page.offset + page.limit)"><ChevronRight aria-hidden="true" /></Button>
    </nav>
  </section>
</template>
