import type { RunView } from '../lib/daemon-api';
import { pathName, relativeTime, runCreatedAt } from '../lib/run-overview';
import { runOperationMessageKeys } from '../i18n/dynamic-keys';

/** One naming rule for a run everywhere it is listed or opened. */
export function useRunLabels() {
  const { t, locale } = useI18n({ useScope: 'global' });

  function operation(run: RunView): string {
    return t(runOperationMessageKeys[run.operation]);
  }

  function title(run: RunView): string {
    return run.summary?.name ?? run.summary?.model ?? operation(run);
  }

  function created(run: RunView, now: Date): string | null {
    const date = runCreatedAt(run);

    return date ? relativeTime(date, now, locale.value) : null;
  }

  /** Model (when the title is a name), dataset, operation and age. */
  function meta(run: RunView, now: Date): string {
    return [
      run.summary?.name ? run.summary.model : null,
      pathName(run.summary?.data_path) || null,
      operation(run),
      created(run, now),
    ]
      .filter((part): part is string => part !== null && part !== undefined && part !== '')
      .join(' · ');
  }

  return { operation, title, created, meta };
}
