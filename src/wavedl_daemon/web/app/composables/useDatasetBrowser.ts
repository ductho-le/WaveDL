import { CatalogApi, type BrowsePage, type DataSelection } from '../lib/catalog-api';
import type { ConnectionSession } from '../lib/connection-session';
import type { RunView } from '../lib/daemon-api';
import { datasetFormat, datasetSuggestions, type DatasetSuggestion } from '../lib/dataset-suggestions';

export interface DatasetBrowserColumn {
  readonly root: string;
  readonly path: string;
  readonly page: BrowsePage | null;
}

export interface ResolvedDatasetSuggestion extends DatasetSuggestion {
  readonly selection: DataSelection;
}

export function useDatasetBrowser(session: () => ConnectionSession, allowDirectory: () => boolean) {
  const roots = shallowRef<readonly string[]>([]);
  const columns = shallowRef<readonly DatasetBrowserColumn[]>([]);
  const suggestions = shallowRef<readonly ResolvedDatasetSuggestion[]>([]);
  const candidate = shallowRef<DataSelection | null>(null);
  const busy = shallowRef(false);
  const resolving = shallowRef(false);
  const message = shallowRef(false);
  let scope = new AbortController();
  let navigation: AbortController | null = null;
  let resolution: AbortController | null = null;
  let unsubscribe: (() => void) | undefined;
  let generation = 0;

  function reset() {
    generation++;
    scope.abort();
    navigation?.abort();
    resolution?.abort();
    scope = new AbortController();
    roots.value = [];
    columns.value = [];
    suggestions.value = [];
    candidate.value = null;
    busy.value = false;
    resolving.value = false;
    message.value = false;
  }

  async function loadSuggestions(signal: AbortSignal) {
    try {
      const runs: RunView[] = [];
      const currentSession = toRaw(session());
      let offset = 0;

      while (!signal.aborted) {
        const page = await currentSession.perform(
          (api, requestSignal) => api.listRuns({ limit: 100, offset, signal: requestSignal }),
          { signal },
        );
        runs.push(...page);

        if (page.length < 100) break;
        offset += page.length;
      }

      const api = new CatalogApi(currentSession);
      const resolved: ResolvedDatasetSuggestion[] = [];

      for (const suggestion of datasetSuggestions(runs)) {
        if (signal.aborted) return;

        try {
          const selection = await api.locate(suggestion.path, signal);

          if (selection?.kind === 'file' && datasetFormat(selection.path))
            resolved.push({ ...suggestion, selection });
        } catch {
          if (signal.aborted) return;
        }

        if (resolved.length === 4) break;
      }

      if (!signal.aborted) suggestions.value = resolved;
    } catch {
      // Suggestions are optional; history or paths may no longer be available.
    }
  }

  async function reload() {
    reset();
    const signal = scope.signal;
    busy.value = true;
    void loadSuggestions(signal);

    try {
      const result = await new CatalogApi(toRaw(session())).roots(signal);

      if (!signal.aborted) roots.value = result;
    } catch {
      if (!signal.aborted) message.value = true;
    } finally {
      if (!signal.aborted) busy.value = false;
    }
  }

  async function resolve(root: string, path: string) {
    resolution?.abort();
    candidate.value = null;
    resolving.value = true;
    message.value = false;
    const controller = new AbortController();
    resolution = controller;
    const signal = AbortSignal.any([scope.signal, controller.signal]);

    try {
      const selection = await new CatalogApi(toRaw(session())).resolve(root, path, signal);

      if (!signal.aborted && (selection.kind === 'file' || allowDirectory()))
        candidate.value = selection;
    } catch {
      if (!signal.aborted) message.value = true;
    } finally {
      if (!signal.aborted) resolving.value = false;
    }
  }

  function chooseFile(root: string, path: string) {
    navigation?.abort();
    busy.value = false;

    return resolve(root, path);
  }

  function suggest(selection: DataSelection) {
    navigation?.abort();
    resolution?.abort();
    busy.value = false;
    resolving.value = false;
    message.value = false;
    candidate.value = selection;
  }

  async function browse(root: string, path: string, index: number, offset = 0) {
    navigation?.abort();
    resolution?.abort();
    resolving.value = false;
    candidate.value = null;
    const controller = new AbortController();
    navigation = controller;
    const signal = AbortSignal.any([scope.signal, controller.signal]);
    const current = generation;
    columns.value = [...columns.value.slice(0, index), { root, path, page: null }];
    busy.value = true;
    message.value = false;

    try {
      const page = await new CatalogApi(toRaw(session())).browse(root, path, offset, signal);

      if (signal.aborted || current !== generation) return;
      columns.value = [...columns.value.slice(0, index), { root, path, page }];

      if (allowDirectory()) void resolve(root, path);
    } catch {
      if (!signal.aborted && current === generation) message.value = true;
    } finally {
      if (!signal.aborted && current === generation) busy.value = false;
    }
  }

  watch(session, (current) => {
    unsubscribe?.();
    let revision = -1;
    let connected = false;
    unsubscribe = toRaw(current).subscribe((state) => {
      if (state.revision === revision && connected === (state.status === 'connected')) return;
      revision = state.revision;
      connected = state.status === 'connected';

      if (connected) void reload();
      else reset();
    });
  }, { immediate: true });

  onBeforeUnmount(() => {
    unsubscribe?.();
    reset();
  });

  return { roots, columns, suggestions, candidate, busy, resolving, message, reload, browse, chooseFile, suggest };
}
