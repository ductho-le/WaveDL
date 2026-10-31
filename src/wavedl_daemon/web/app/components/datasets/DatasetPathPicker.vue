<script setup lang="ts">
import { cn } from '@/lib/utils';
import {
  ArrowUp,
  Check,
  ChevronLeft,
  ChevronRight,
  File,
  Folder,
  FolderOpen,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldCheck,
} from '@lucide/vue';
import type { ConnectionSession } from '../../lib/connection-session';
import { CatalogApi, type BrowsePage, type DataSelection } from '../../lib/catalog-api';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  pathBrowserSuggestions: cn(
    'absolute top-[calc(100%_+_0.35rem)] right-0 left-0 z-30 max-h-72 overflow-y-auto p-[0.35rem]',
    'rounded-[0.9rem] bg-popover',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_14%,_transparent),_0_18px_40px_-16px_rgb(0_0_0_/_0.35)]',
  ),
  pathBrowserSuggestion: cn(
    'flex w-full min-h-9 items-center gap-[0.6rem] p-[0.35rem_0.6rem] rounded-[0.55rem] text-[0.85rem]',
    "text-left [&[aria-selected='true']]:bg-[color-mix(in_srgb,_var(--primary)_12%,_transparent)]",
  ),
  pathBrowserRoot: cn(
    'inline-flex items-center gap-[0.4rem] p-[0.3rem_0.75rem] rounded-full text-[0.8rem] font-medium',
    'bg-[color-mix(in_srgb,_var(--foreground)_5%,_transparent)]',
    '[transition:background-color_150ms_ease,_color_150ms_ease] [&_svg]:w-[0.85rem] [&_svg]:h-[0.85rem]',
    '[&:hover]:bg-[color-mix(in_srgb,_var(--foreground)_9%,_transparent)]',
    "[&[data-active='true']]:text-primary-foreground [&[data-active='true']]:bg-primary",
  ),
  pathBrowserBar: cn(
    'flex items-center gap-[0.35rem] p-[0.35rem_0.6rem_0.35rem_0.35rem]',
    '[border-bottom:0.5px_solid_color-mix(in_srgb,_var(--foreground)_12%,_transparent)]!',
    'bg-[color-mix(in_srgb,_var(--foreground)_2.5%,_transparent)]',
  ),
  pathBrowserCrumb: cn(
    'inline-flex items-center gap-[0.3rem] p-[0.2rem_0.45rem] rounded-[0.45rem] text-[0.8rem]',
    'whitespace-nowrap [&_svg]:w-[0.8rem] [&_svg]:h-[0.8rem]',
    '[&:is(button):hover]:bg-[color-mix(in_srgb,_var(--foreground)_7%,_transparent)]',
    '[&[data-current]]:font-semibold',
  ),
  pathBrowserRow: cn(
    'flex w-full min-h-[2.6rem] min-w-0 flex-1 items-center gap-[0.7rem] p-[0.35rem_0.6rem]',
    'rounded-[0.6rem] text-sm text-left [transition:background-color_120ms_ease]',
    '[&:hover:not(:disabled)]:bg-[color-mix(in_srgb,_var(--foreground)_5%,_transparent)]',
    '[&:focus-visible]:[outline:2px_solid_var(--primary)]! [&:focus-visible]:[outline-offset:-2px]!',
    '[&[data-selected]]:bg-[color-mix(in_srgb,_var(--primary)_11%,_transparent)]',
    '[&:hover_.path-browser-use]:opacity-[1] [&:focus-visible_.path-browser-use]:opacity-[1]',
  ),
  pathBrowserIcon: cn(
    'grid size-7 flex-none place-items-center rounded-[0.5rem] text-muted-foreground',
    'bg-[color-mix(in_srgb,_var(--foreground)_5%,_transparent)] [&_svg]:w-[0.95rem] [&_svg]:h-[0.95rem]',
    "[&[data-kind='directory']]:text-[#3b82f6]",
    "[&[data-kind='directory']]:bg-[color-mix(in_srgb,_#3b82f6_13%,_transparent)]",
    "[&[data-kind='file']]:text-success",
    "[&[data-kind='file']]:bg-[color-mix(in_srgb,_var(--success)_13%,_transparent)]",
  ),
  pathBrowserExt: cn(
    'flex-none p-[0.05rem_0.4rem] rounded-[0.35rem] text-muted-foreground text-[0.65rem] font-[650]',
    'tracking-[0.04em] bg-[color-mix(in_srgb,_var(--foreground)_6%,_transparent)]',
  ),
  pathBrowserEmpty: cn(
    'flex flex-col items-center gap-2 p-[2.25rem_1rem] text-muted-foreground text-[0.85rem] text-center',
    '[&_svg]:w-[1.6rem] [&_svg]:h-[1.6rem] [&_svg]:opacity-[0.6]',
  ),
};

const { t } = useI18n({ useScope: 'global' });

const props = withDefaults(
  defineProps<{
    session: ConnectionSession;
    allowDirectory?: boolean;
    label?: string;
    /** The current choice, marked in the listing when its folder is open. */
    selected?: DataSelection | null;
  }>(),
  { allowDirectory: false, selected: null },
);

const emit = defineEmits<{ select: [selection: DataSelection] }>();

type BrowseMessage = '' | 'datasets.browseUnavailable' | 'datasets.chooseFile';

type SearchOption = {
  id: string;
  label: string;
  detail: string;
  kind: 'root' | 'directory' | 'file';
  root: string;
  path: string;
};

const id = useId();
const roots = shallowRef<readonly string[]>([]);
const page = shallowRef<BrowsePage | null>(null);
const root = shallowRef('');
const path = shallowRef('');
const query = shallowRef('');
const activeIndex = shallowRef(0);
const focused = shallowRef(false);
const busy = shallowRef(false);
const message = shallowRef<BrowseMessage>('');
let revision = 0;
let request: AbortController | null = null;
let unsubscribe: (() => void) | undefined;

const pathParts = computed(() => (path.value ? path.value.split('/') : []));

const suggestions = computed<SearchOption[]>(() => {
  const search = query.value.trim().toLocaleLowerCase();

  const rootOptions = roots.value
    .flatMap((item) => !search || item.toLocaleLowerCase().includes(search) ? [{
      id: `root:${item}`,
      label: item,
      detail: t('datasets.configuredRoot'),
      kind: 'root' as const,
      root: item,
      path: '',
    }] : []);

  const entryOptions = (page.value?.entries ?? [])
    .flatMap((entry) => {
      const entryPath = [path.value, entry.name].filter(Boolean).join('/');

      if (search && !`${root.value}/${entryPath}`.toLocaleLowerCase().includes(search)) return [];

      return [{
        id: `entry:${root.value}:${entryPath}`,
        label: entry.name,
        detail: entry.kind === 'directory' ? t('datasets.folder') : t('datasets.file'),
        kind: entry.kind,
        root: root.value,
        path: entryPath,
      }];
    });

  const ordered = root.value ? [...rootOptions, ...entryOptions] : rootOptions;

  return ordered.slice(0, 24);
});

const label = computed(() => props.label ?? t('datasets.pathPickerTitle'));
const searchId = `${id}-dataset-path-search`;
const optionListId = `${id}-dataset-path-options`;

function cancel() {
  revision++;
  request?.abort();
  request = null;
  busy.value = false;
}

async function work<T>(
  callback: (api: CatalogApi, signal: AbortSignal) => Promise<T>,
  accept: (value: T) => void,
) {
  cancel();
  const current = revision;
  const controller = new AbortController();
  request = controller;
  busy.value = true;
  message.value = '';

  try {
    const value = await callback(new CatalogApi(toRaw(props.session)), controller.signal);

    if (current === revision && !controller.signal.aborted) accept(value);
  } catch {
    if (current === revision && !controller.signal.aborted)
      message.value = 'datasets.browseUnavailable';
  } finally {
    if (current === revision) {
      busy.value = false;
      request = null;
    }
  }
}

function browse(nextRoot = root.value, nextPath = path.value, offset = 0) {
  root.value = nextRoot;
  path.value = nextPath;
  query.value = '';
  page.value = null;

  return work(
    (api, signal) => api.browse(nextRoot, nextPath, offset, signal),
    (value) => {
      page.value = value;
    },
  );
}

function loadRoots() {
  roots.value = [];
  page.value = null;
  root.value = '';
  path.value = '';
  query.value = '';

  return work(
    (api, signal) => api.roots(signal),
    (value) => {
      roots.value = value;
    },
  );
}

function child(name: string) {
  return [path.value, name].filter(Boolean).join('/');
}

function up() {
  return browse(root.value, pathParts.value.slice(0, -1).join('/'));
}

function extension(name: string): string {
  const dot = name.lastIndexOf('.');

  return dot > 0 ? name.slice(dot + 1).toUpperCase() : '';
}

function isSelected(name: string): boolean {
  return props.selected?.root === root.value && props.selected.relativePath === child(name);
}

function resolvePath(selectedRoot: string, selectedPath: string) {
  return work(
    (api, signal) => api.resolve(selectedRoot, selectedPath, signal),
    (selection) => {
      if (selection.kind === 'directory' && !props.allowDirectory) {
        message.value = 'datasets.chooseFile';

        return;
      }

      emit('select', selection);
      focused.value = false;
    },
  );
}

function choose(option: SearchOption) {
  focused.value = false;

  if (option.kind === 'root') return browse(option.root, '');

  if (option.kind === 'directory') return browse(option.root, option.path);

  return resolvePath(option.root, option.path);
}

function chooseTypedPath() {
  const typed = query.value.trim();
  const matchingRoot = roots.value.find((item) => item === typed);

  if (matchingRoot) return browse(matchingRoot, '');

  if (!root.value || !typed) return;
  const prefix = `${root.value}/`;
  const selectedPath = typed.startsWith(prefix) ? typed.slice(prefix.length) : typed;

  if (!selectedPath) return browse(root.value, '');
  focused.value = false;

  return work(
    (api, signal) => api.resolve(root.value, selectedPath, signal),
    (selection) => {
      if (selection.kind === 'directory' && !props.allowDirectory) {
        void browse(root.value, selectedPath);
      } else emit('select', selection);
    },
  );
}

function keydown(event: KeyboardEvent) {
  if (event.key === 'ArrowDown' && suggestions.value.length) {
    event.preventDefault();
    focused.value = true;
    activeIndex.value = (activeIndex.value + 1) % suggestions.value.length;
  } else if (event.key === 'ArrowUp' && suggestions.value.length) {
    event.preventDefault();
    focused.value = true;
    activeIndex.value =
      (activeIndex.value - 1 + suggestions.value.length) % suggestions.value.length;
  } else if (event.key === 'Enter') {
    event.preventDefault();
    const active = focused.value ? suggestions.value[activeIndex.value] : undefined;

    if (active) void choose(active);
    else void chooseTypedPath();
  } else if (event.key === 'Escape') {
    focused.value = false;
  }
}

watch(query, () => {
  activeIndex.value = 0;
});

watch(
  () => toRaw(props.session),
  (session) => {
    unsubscribe?.();
    cancel();
    roots.value = [];
    page.value = null;
    root.value = '';
    path.value = '';
    let scope = -1;
    let connected = false;
    unsubscribe = session.subscribe((state) => {
      if (state.revision === scope && connected && state.status === 'connected') return;
      scope = state.revision;
      connected = state.status === 'connected';
      cancel();
      roots.value = [];
      page.value = null;
      root.value = '';
      path.value = '';
      query.value = '';

      if (state.status === 'connected') void loadRoots();
      else if (state.status !== 'disconnected') message.value = 'datasets.browseUnavailable';
    });
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  unsubscribe?.();
  cancel();
});
</script>

<template>
  <section class="flex min-w-0 flex-col gap-3" :aria-label="label">
    <div class="flex items-center gap-1">
      <div class="relative min-w-0 flex-1">
        <Search
          class="absolute top-1/2 left-3 size-4 text-muted-foreground pointer-events-none -translate-y-1/2"
          aria-hidden="true"
        />
        <Input
          :id="searchId"
          v-model="query"
          role="combobox"
          class="h-10 rounded-xl bg-surface pl-9"
          :aria-label="t('datasets.searchRootsAndPaths')"
          autocomplete="off"
          aria-autocomplete="list"
          :aria-controls="optionListId"
          :aria-expanded="focused && suggestions.length > 0"
          :aria-activedescendant="
            focused && suggestions[activeIndex]
              ? `${id}-dataset-path-option-${activeIndex}`
              : undefined
          "
          :placeholder="t('datasets.searchRootsPlaceholder')"
          :disabled="busy || !roots.length"
          data-testid="dataset-path-search"
          @focus="focused = true"
          @blur="focused = false"
          @keydown="keydown"
        />
        <ul
          v-if="focused && suggestions.length"
          :id="optionListId"
          role="listbox"
          :class="cn(styles.pathBrowserSuggestions)"
          :aria-label="t('datasets.searchResults')"
        >
          <li v-for="(option, index) in suggestions" :key="option.id" role="presentation">
            <button
              :id="`${id}-dataset-path-option-${index}`"
              type="button"
              role="option"
              :aria-selected="index === activeIndex"
              :data-value="option.kind === 'root' ? option.root : `${option.root}/${option.path}`"
              :class="cn(styles.pathBrowserSuggestion)"
              @mousedown.prevent
              @mouseenter="activeIndex = index"
              @click="choose(option)"
            >
              <HardDrive
                v-if="option.kind === 'root'"
                class="size-4 flex-none text-muted-foreground [&[data-kind='directory']]:text-[#3b82f6]"
                aria-hidden="true"
              />
              <Folder
                v-else-if="option.kind === 'directory'"
                class="size-4 flex-none text-muted-foreground [&[data-kind='directory']]:text-[#3b82f6]"
                data-kind="directory"
                aria-hidden="true"
              />
              <File
                v-else
                class="size-4 flex-none text-muted-foreground [&[data-kind='directory']]:text-[#3b82f6]"
                data-kind="file"
                aria-hidden="true"
              />
              <span class="min-w-0 flex-1 truncate">{{ option.label }}</span>
              <span class="shrink-0 text-xs text-muted-foreground">{{ option.detail }}</span>
            </button>
          </li>
        </ul>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        class="rounded-xl"
        :disabled="busy"
        :aria-label="t('datasets.reloadRoots')"
        :title="t('datasets.reloadRoots')"
        @click="loadRoots"
      >
        <RefreshCw aria-hidden="true" />
      </Button>
    </div>

    <div
      v-if="root && roots.length > 1"
      class="flex flex-wrap gap-[0.4rem]"
      role="group"
      :aria-label="t('datasets.locations')"
    >
      <button
        v-for="item in roots"
        :key="item"
        type="button"
        :class="cn(styles.pathBrowserRoot)"
        :data-active="item === root"
        :aria-pressed="item === root"
        :disabled="busy"
        @click="browse(item, '')"
      >
        <HardDrive aria-hidden="true" />{{ item }}
      </button>
    </div>

    <div
      class="overflow-hidden rounded-[0.9rem] bg-surface shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_13%,_transparent)]"
    >
      <div v-if="root" :class="cn(styles.pathBrowserBar)">
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          :disabled="busy || !pathParts.length"
          :aria-label="t('datasets.upOneFolder')"
          :title="t('datasets.upOneFolder')"
          @click="up"
        >
          <ArrowUp aria-hidden="true" />
        </Button>
        <Breadcrumb class="min-w-0 flex-1" :aria-label="t('datasets.pathBreadcrumb')">
          <BreadcrumbList class="flex-nowrap gap-0.5 overflow-x-auto sm:gap-0.5">
            <BreadcrumbItem>
              <button
                v-if="pathParts.length"
                type="button"
                :class="cn(styles.pathBrowserCrumb)"
                :disabled="busy"
                @click="browse(root, '')"
              >
                <HardDrive aria-hidden="true" />{{ root }}
              </button>
              <BreadcrumbPage v-else :class="cn(styles.pathBrowserCrumb)" data-current>
                <HardDrive aria-hidden="true" />{{ root }}
              </BreadcrumbPage>
            </BreadcrumbItem>
            <template v-for="(part, index) in pathParts" :key="`${index}-${part}`">
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <button
                  v-if="index < pathParts.length - 1"
                  type="button"
                  :class="cn(styles.pathBrowserCrumb)"
                  :disabled="busy"
                  @click="browse(root, pathParts.slice(0, index + 1).join('/'))"
                >
                  {{ part }}
                </button>
                <BreadcrumbPage v-else :class="cn(styles.pathBrowserCrumb)" data-current>{{
                  part
                }}</BreadcrumbPage>
              </BreadcrumbItem>
            </template>
          </BreadcrumbList>
        </Breadcrumb>
        <LoaderCircle
          v-if="busy"
          class="size-4 flex-none animate-spin text-muted-foreground"
          aria-hidden="true"
        />
      </div>

      <p v-if="busy" role="status" class="sr-only">{{ t('datasets.browseLoading') }}</p>
      <Alert v-if="message" role="alert" variant="warning" class="m-3 w-auto text-sm">
        <AlertDescription>{{ t(message) }}</AlertDescription>
      </Alert>

      <div
        v-if="busy && !page && !roots.length"
        class="flex flex-col gap-1.5 p-3"
        aria-hidden="true"
      >
        <Skeleton v-for="index in 4" :key="index" class="h-9 w-full rounded-lg" />
      </div>
      <div
        v-else-if="!roots.length && !busy && !message"
        role="status"
        :class="cn(styles.pathBrowserEmpty)"
      >
        <HardDrive aria-hidden="true" />
        <p>{{ t('datasets.noRoots') }}</p>
      </div>
      <!-- No location open yet: the locations themselves are the first listing. -->
      <ul
        v-else-if="!root && roots.length"
        class="flex max-h-88 flex-col overflow-y-auto p-[0.35rem]"
        :aria-label="t('datasets.locations')"
      >
        <li v-for="item in roots" :key="item">
          <button
            type="button"
            :class="cn(styles.pathBrowserRow)"
            :disabled="busy"
            @click="browse(item, '')"
          >
            <span :class="cn(styles.pathBrowserIcon)" data-kind="root"
              ><HardDrive aria-hidden="true"
            /></span>
            <span class="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{{
              item
            }}</span>
            <ChevronRight
              class="w-[0.95rem] h-[0.95rem] flex-none text-muted-foreground"
              aria-hidden="true"
            />
          </button>
        </li>
      </ul>
      <div v-else-if="busy && !page" class="flex flex-col gap-1.5 p-3" aria-hidden="true">
        <Skeleton v-for="index in 5" :key="index" class="h-9 w-full rounded-lg" />
      </div>
      <div
        v-else-if="page && !page.entries.length && !busy"
        role="status"
        :class="cn(styles.pathBrowserEmpty)"
      >
        <FolderOpen aria-hidden="true" />
        <p>{{ t('datasets.emptyDirectory') }}</p>
      </div>
      <ul
        v-if="root && page?.entries.length"
        class="flex max-h-88 flex-col overflow-y-auto p-[0.35rem]"
        :aria-label="t('datasets.entries')"
      >
        <li v-for="entry in page.entries" :key="entry.name" class="flex items-center gap-1">
          <button
            v-if="entry.kind === 'directory'"
            type="button"
            :class="cn(styles.pathBrowserRow)"
            :disabled="busy"
            @click="browse(root, child(entry.name))"
          >
            <span :class="cn(styles.pathBrowserIcon)" data-kind="directory"
              ><Folder aria-hidden="true"
            /></span>
            <span class="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{{
              entry.name
            }}</span>
            <ChevronRight
              class="w-[0.95rem] h-[0.95rem] flex-none text-muted-foreground"
              aria-hidden="true"
            />
          </button>
          <button
            v-else
            type="button"
            :class="cn(styles.pathBrowserRow)"
            :data-selected="isSelected(entry.name) || undefined"
            :disabled="busy"
            :aria-label="t('datasets.selectEntry', { kind: t('datasets.file'), name: entry.name })"
            @click="resolvePath(root, child(entry.name))"
          >
            <span :class="cn(styles.pathBrowserIcon)" data-kind="file"
              ><File aria-hidden="true"
            /></span>
            <span class="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{{
              entry.name
            }}</span>
            <span v-if="extension(entry.name)" :class="cn(styles.pathBrowserExt)">{{
              extension(entry.name)
            }}</span>
            <Check
              v-if="isSelected(entry.name)"
              class="size-4 flex-none text-primary [stroke-width:2.5]"
              aria-hidden="true"
            />
            <span
              v-else
              class="path-browser-use flex-none text-primary text-[0.78rem] font-semibold opacity-[0] [transition:opacity_120ms_ease] [@media(hover:none)]:opacity-[1]"
              >{{ t('datasets.use') }}</span
            >
          </button>
          <Button
            v-if="entry.kind === 'directory' && allowDirectory"
            type="button"
            variant="ghost"
            size="sm"
            :disabled="busy"
            :aria-label="
              t('datasets.selectEntry', { kind: t('datasets.folder'), name: entry.name })
            "
            @click="resolvePath(root, child(entry.name))"
          >
            {{ t('datasets.selectKind', { kind: t('datasets.folder') }) }}
          </Button>
        </li>
      </ul>
      <div
        v-if="page && page.total > page.limit"
        class="flex items-center justify-end gap-1 p-[0.3rem_0.5rem] [border-top:0.5px_solid_color-mix(in_srgb,_var(--foreground)_12%,_transparent)]!"
      >
        <span class="text-xs text-muted-foreground tabular-nums">
          {{
            t('datasets.entryCount', {
              start: page.offset + 1,
              end: page.offset + page.entries.length,
              total: page.total,
            })
          }}
        </span>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          :disabled="busy || page.offset === 0"
          :aria-label="t('datasets.previousEntries')"
          @click="browse(root, path, Math.max(0, page.offset - page.limit))"
        >
          <ChevronLeft aria-hidden="true" />
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          :disabled="busy || page.offset + page.limit >= page.total"
          :aria-label="t('datasets.nextEntries')"
          @click="browse(root, path, page.offset + page.limit)"
        >
          <ChevronRight aria-hidden="true" />
        </Button>
      </div>
    </div>

    <div class="flex flex-wrap items-center gap-2">
      <p
        class="inline-flex flex-1 items-center gap-[0.35rem] text-muted-foreground text-xs [&_svg]:w-[0.85rem] [&_svg]:h-[0.85rem] [&_svg]:flex-none"
      >
        <ShieldCheck aria-hidden="true" />{{ t('datasets.browseSafety') }}
      </p>
      <Button
        v-if="allowDirectory && page"
        type="button"
        variant="outline"
        size="sm"
        :disabled="busy"
        data-testid="dataset-select-directory"
        @click="resolvePath(root, path)"
      >
        {{ t('datasets.selectDirectory') }}
      </Button>
      <Button v-if="busy" type="button" variant="ghost" size="sm" @click="cancel">
        {{ t('datasets.stopBrowsing') }}
      </Button>
    </div>
  </section>
</template>
