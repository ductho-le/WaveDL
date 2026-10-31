import { clientError, type ClientMessage } from '../i18n/client-messages';
import { DaemonApi, DaemonApiError, type RunView } from '../lib/daemon-api';
import { CursorStore } from '../lib/event-stream';
import { RunMonitor, type RunMonitorSnapshot } from '../lib/run-monitor';

export function useRuns(
  api: DaemonApi,
  reportFailure: (error: DaemonApiError) => void,
  initiallyPaused = false,
) {
  const runs = shallowRef<readonly RunView[]>([]);
  const selected = shallowRef<RunMonitorSnapshot | null>(null);
  const loading = shallowRef(false);
  const message = shallowRef<ClientMessage | Error>('');
  const page = shallowRef(0);
  const staleRunIds = shallowRef<ReadonlySet<string>>(new Set());
  const lastRefreshed = shallowRef<string | null>(null);
  let detailRevision = 0;
  let listAbort: AbortController | null = null;
  let monitor: RunMonitor | null = null;
  let unsubscribe: (() => void) | null = null;
  let disposed = false;
  let paused = initiallyPaused;

  async function refresh(offset = page.value): Promise<void> {
    listAbort?.abort();
    const controller = new AbortController();
    listAbort = controller;
    loading.value = true;
    const detailAtStart = detailRevision;

    try {
      const result = await api.listRuns({
        limit: 50,
        offset,
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]),
      });

      if (!controller.signal.aborted && !disposed) {
        const detail = selected.value;
        runs.value =
          detail?.run && detail.freshness === 'fresh' && detailRevision !== detailAtStart
            ? result.map((run) => (run.run_id === detail.run?.run_id ? detail.run : run))
            : result;
        staleRunIds.value =
          detail?.run && detail.freshness === 'stale' ? new Set([detail.run.run_id]) : new Set();
        lastRefreshed.value = new Date().toISOString();
        page.value = offset;
        message.value = '';
      }
    } catch (error) {
      if (!controller.signal.aborted && !disposed) {
        message.value =
          error instanceof DaemonApiError ? error : clientError('unableToRefreshRuns');
        staleRunIds.value = new Set(runs.value.map((run) => run.run_id));

        if (error instanceof DaemonApiError) reportFailure(error);
      }
    } finally {
      if (listAbort === controller && !disposed) loading.value = false;
    }
  }

  function select(runId: string): void {
    unsubscribe?.();
    monitor?.stop();
    const daemonId = api.daemonId;
    monitor = new RunMonitor(
      runId,
      {
        getRun: (id, signal) =>
          api.getRun(id, AbortSignal.any([signal, AbortSignal.timeout(15000)])),
        openEvents: (id, cursor, signal) => api.openEvents(id, cursor, signal, daemonId),
        cancelRun: (id, signal) =>
          api.cancelRun(id, AbortSignal.any([signal, AbortSignal.timeout(15000)])),
      },
      2000,
      reportFailure,
      daemonId
        ? new CursorStore(daemonId, runId, {
            getItem: (key) => globalThis.localStorage.getItem(key),
            setItem: (key, value) => globalThis.localStorage.setItem(key, value),
          })
        : null,
    );
    unsubscribe = monitor.subscribe((snapshot) => {
      const previous = selected.value;
      selected.value = snapshot;

      if (
        snapshot.run &&
        (snapshot.run !== previous?.run || snapshot.freshness !== previous.freshness)
      ) {
        detailRevision++;
        const stale = new Set(staleRunIds.value);

        if (snapshot.freshness === 'fresh') {
          runs.value = runs.value.map((run) =>
            run.run_id === snapshot.run?.run_id ? snapshot.run : run,
          );
          stale.delete(snapshot.run.run_id);
        } else {
          stale.add(snapshot.run.run_id);
        }

        staleRunIds.value = stale;
      }
    });

    if (!paused) void monitor.start();
  }

  function replay(): void {
    if (monitor) select(monitor.runId);
  }

  async function cancel(): Promise<void> {
    await monitor?.cancel();
  }

  function close(): void {
    unsubscribe?.();
    unsubscribe = null;
    monitor?.stop();
    monitor = null;
    selected.value = null;
  }

  function pause(): void {
    paused = true;
    listAbort?.abort();
    listAbort = null;
    monitor?.pause();
    loading.value = false;
  }

  function resume(): void {
    paused = false;

    if (lastRefreshed.value === null) void refresh();
    const current = monitor;

    if (current) {
      void current.resume(() => !paused && !disposed && monitor === current);
      void current.start();
    }
  }

  onScopeDispose(() => {
    disposed = true;
    listAbort?.abort();
    close();
  });

  if (!paused) void refresh();

  return {
    runs,
    selected,
    loading,
    message,
    page,
    staleRunIds,
    lastRefreshed,
    refresh,
    select,
    replay,
    cancel,
    close,
    pause,
    resume,
  };
}
