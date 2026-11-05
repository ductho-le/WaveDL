import { CatalogApi, type BrowseEntry, type BrowsePage } from '../lib/catalog-api';
import type { ConnectionSession } from '../lib/connection-session';

/** Search one folder, including its other pages, never its descendants. */
export function useDatasetFolderFilter(
  session: () => ConnectionSession,
  page: () => BrowsePage | null,
  query: () => string,
) {
  const entries = shallowRef<readonly BrowseEntry[]>([]);
  const busy = shallowRef(false);
  const failed = shallowRef(false);
  const offset = shallowRef(0);
  const needle = computed(() => query().trim().toLocaleLowerCase());
  const matching = computed(() => entries.value.filter((entry) => entry.name.toLocaleLowerCase().includes(needle.value)));
  const filteredPage = computed<BrowsePage | null>(() => {
    const current = page();

    if (!current || !needle.value) return null;

    return {
      ...current,
      offset: offset.value,
      total: matching.value.length,
      entries: matching.value.slice(offset.value, offset.value + current.limit),
    };
  });

  watch(query, () => { offset.value = 0; });
  watch([session, page, () => !!needle.value], async (_current, _previous, onCleanup) => {
    const current = page();
    offset.value = 0;
    entries.value = current?.entries ?? [];
    busy.value = false;
    failed.value = false;

    if (!current || !needle.value || current.total <= current.limit) return;
    const controller = new AbortController();
    onCleanup(() => controller.abort());
    busy.value = true;
    const pages = new Map<number, readonly BrowseEntry[]>([[current.offset, current.entries]]);

    try {
      const api = new CatalogApi(toRaw(session()));

      for (let start = 0; start < current.total; start += current.limit) {
        if (start === current.offset) continue;
        const result = await api.browse(current.root, current.path, start, controller.signal);

        if (controller.signal.aborted) return;
        pages.set(start, result.entries);
      }
    } catch {
      if (!controller.signal.aborted) failed.value = true;
    } finally {
      if (!controller.signal.aborted) {
        const ordered = [...pages.entries()].sort(([a], [b]) => a - b).flatMap(([, values]) => values);
        const unique = new Map(ordered.map((entry) => [entry.name, entry]));
        entries.value = [...unique.values()];
        busy.value = false;
      }
    }
  }, { immediate: true });

  return { filteredPage, busy, failed, offset };
}
