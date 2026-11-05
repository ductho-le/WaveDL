import type { DataSelection } from '../lib/catalog-api';
import type { ConnectionSession } from '../lib/connection-session';
import { DatasetInspectionApi, type DatasetInspection } from '../lib/dataset-inspection';

/** Header requests never outlive their selection or connection revision. */
export function useDatasetInspection(
  session: () => ConnectionSession,
  selection: () => DataSelection | null,
) {
  const info = shallowRef<DatasetInspection | null>(null);
  const busy = shallowRef(false);
  const failed = shallowRef(false);
  const revision = shallowRef(0);
  let unsubscribe: (() => void) | undefined;

  watch(session, (current) => {
    unsubscribe?.();
    unsubscribe = toRaw(current).subscribe(() => { revision.value++; });
  }, { immediate: true });

  watch([session, selection, revision], async (_current, _previous, onCleanup) => {
    const current = selection();
    info.value = null;
    failed.value = false;
    busy.value = false;

    if (!current || current.kind !== 'file') return;
    const controller = new AbortController();
    onCleanup(() => controller.abort());
    busy.value = true;

    try {
      const result = await new DatasetInspectionApi(toRaw(session())).inspect(
        toRaw(current), controller.signal,
      );

      if (!controller.signal.aborted) info.value = result;
    } catch {
      if (!controller.signal.aborted) failed.value = true;
    } finally {
      if (!controller.signal.aborted) busy.value = false;
    }
  }, { immediate: true, flush: 'sync' });

  onBeforeUnmount(() => unsubscribe?.());

  return { info, busy, failed };
}
