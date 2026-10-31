import type { RunState, RunView } from './daemon-api';

/** Presentation-only grouping of one run page; lifecycle truth stays with the daemon. */
export type RunFilter = 'all' | 'live' | 'completed' | 'attention';

export type RunGroupKey = 'live' | 'today' | 'yesterday' | 'week' | 'earlier';

export interface RunGroup {
  key: RunGroupKey;
  runs: RunView[];
}

const LIVE_STATES: ReadonlySet<RunState> = new Set([
  'pending',
  'submitted',
  'queued',
  'held',
  'running',
  'stopping',
]);

const ATTENTION_STATES: ReadonlySet<RunState> = new Set([
  'failed',
  'submission_unknown',
  'terminal_unknown',
  'observation_unavailable',
  'supervision_lost',
]);

const DAY = 86_400_000;

export function runFilterOf(run: RunView): Exclude<RunFilter, 'all'> | null {
  if (LIVE_STATES.has(run.state)) return 'live';

  if (ATTENTION_STATES.has(run.state)) return 'attention';

  return run.state === 'completed' ? 'completed' : null;
}

export function runCounts(runs: readonly RunView[]): Record<RunFilter, number> {
  const counts: Record<RunFilter, number> = {
    all: runs.length,
    live: 0,
    completed: 0,
    attention: 0,
  };

  for (const run of runs) {
    const filter = runFilterOf(run);

    if (filter) counts[filter]++;
  }

  return counts;
}

export function runCreatedAt(run: RunView): Date | null {
  const date = new Date(run.created_at ?? '');

  return Number.isNaN(date.getTime()) ? null : date;
}

/** Final path segment of a host path, for either separator; empty when there is none. */
export function pathName(path: string | null | undefined): string {
  return (
    (path ?? '')
      .replace(/[\\/]+$/, '')
      .split(/[\\/]/)
      .pop() ?? ''
  );
}

export function shortRunId(runId: string): string {
  return runId.slice(0, 8);
}

/** Case-insensitive match over the run identity and caller-supplied visible labels. */
export function runMatches(run: RunView, query: string, labels: readonly string[] = []): boolean {
  const needle = query.trim().toLowerCase();

  if (!needle) return true;

  return [run.run_id, run.summary?.model, pathName(run.summary?.data_path), ...labels].some(
    (value) => value != null && value.toLowerCase().includes(needle),
  );
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/**
 * Live runs first, then calendar buckets newest-first. The daemon pages oldest-first, so
 * undated runs keep their reversed page order inside "earlier".
 */
export function groupRuns(runs: readonly RunView[], now: Date): RunGroup[] {
  const today = startOfDay(now);
  const order: RunGroupKey[] = ['live', 'today', 'yesterday', 'week', 'earlier'];

  const buckets = new Map<RunGroupKey, { run: RunView; time: number; index: number }[]>(
    order.map((key) => [key, []]),
  );

  runs.forEach((run, index) => {
    const time = runCreatedAt(run)?.getTime() ?? Number.NEGATIVE_INFINITY;

    const key: RunGroupKey = LIVE_STATES.has(run.state)
      ? 'live'
      : time >= today
        ? 'today'
        : time >= today - DAY
          ? 'yesterday'
          : time >= today - 6 * DAY
            ? 'week'
            : 'earlier';

    buckets.get(key)?.push({ run, time, index });
  });

  return order.flatMap((key) => {
    const entries = buckets.get(key) ?? [];

    if (!entries.length) return [];
    entries.sort((a, b) => b.time - a.time || b.index - a.index);

    return [{ key, runs: entries.map((entry) => entry.run) }];
  });
}

/** "3 minutes ago"-style text, choosing the largest sensible unit. */
export function relativeTime(date: Date, now: Date, locale: string): string {
  const seconds = Math.round((date.getTime() - now.getTime()) / 1000);
  const format = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });

  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ['day', 86_400],
    ['hour', 3_600],
    ['minute', 60],
  ];

  for (const [unit, size] of units) {
    if (Math.abs(seconds) >= size) return format.format(Math.round(seconds / size), unit);
  }

  return format.format(0, 'second');
}

/** Two most significant units of a duration, e.g. "1h 12m" or "4m 5s". */
export function durationText(totalSeconds: number, locale: string): string {
  const seconds = Math.max(0, Math.round(totalSeconds));

  const parts: [Intl.NumberFormatOptions['unit'], number][] = [
    ['hour', Math.floor(seconds / 3600)],
    ['minute', Math.floor((seconds % 3600) / 60)],
    ['second', seconds % 60],
  ];

  const first = parts.findIndex(([, value]) => value > 0);

  return parts
    .slice(first === -1 ? 2 : first, first === -1 ? 3 : first + 2)
    .map(([unit, value]) =>
      new Intl.NumberFormat(locale, { style: 'unit', unit, unitDisplay: 'narrow' }).format(value),
    )
    .join(' ');
}

/** Remaining training time from the mean observed epoch; null until it is meaningful. */
export function remainingSeconds(
  progress: { epoch: number; totalEpochs: number } | null,
  elapsedSeconds: number | null,
): number | null {
  if (!progress || elapsedSeconds === null || progress.epoch < 1) return null;
  const remaining = progress.totalEpochs - progress.epoch;

  if (remaining <= 0 || !Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) return null;

  return (elapsedSeconds / progress.epoch) * remaining;
}
