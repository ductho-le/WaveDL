import { clientRangeError } from '../i18n/client-messages';
import type { MetricRecord } from './metric-history';
import type { RuntimeMetricField } from '../../shared/schemas/metric-history';

export type PlotChannel = 'loss' | 'r2' | 'lr';

export const PLOT_CHANNELS = ['loss', 'r2', 'lr'] as const;
export const MAX_PLOT_RECORDS = 200_000;

const fields: Record<PlotChannel, readonly RuntimeMetricField[]> = {
  loss: ['train_loss', 'val_loss'],
  r2: ['r2', 'val_r2'],
  lr: ['lr'],
};

export const PLOT_LABELS = {
  train_loss: 'Training loss',
  val_loss: 'Validation loss',
  r2: 'Training R²',
  val_r2: 'Validation R²',
  lr: 'Learning rate',
} satisfies Record<string, string>;

export interface PlotPoint {
  step: number;
  values: Partial<Record<RuntimeMetricField, number | null>>;
}

export interface PlotSeries {
  points: readonly PlotPoint[];
  fields: readonly RuntimeMetricField[];
  pointCount: number;
}

interface Cell {
  value: number | null;
  attempt: number;
  seq: number;
}

/** Align a causal, active projection without interpolating absent or null values.
 * Rebuild on restore: stale branch tails must disappear, not remain frozen.
 * O(N + S log S), O(S * F) storage; F is at most two fixed channels.
 */
export function buildPlotSeries(
  records: readonly MetricRecord[],
  channel: PlotChannel,
): PlotSeries {
  if (records.length > MAX_PLOT_RECORDS)
    throw clientRangeError('chartCapacityExceeded', { limit: MAX_PLOT_RECORDS });
  const selected = fields[channel];
  const rows = new Map<number, Map<string, Cell>>();

  for (const record of records) {
    if (!record.active || !Number.isSafeInteger(record.step) || record.step < 0) continue;

    for (const field of selected) {
      const value = Object.hasOwn(record.metrics, field)
        ? record.metrics[field]
        : record.name === field
          ? record.value
          : undefined;

      if (value === undefined) continue;
      let row = rows.get(record.step);

      if (!row) {
        row = new Map();
        rows.set(record.step, row);
      }

      const previous = row.get(field);

      if (
        !previous ||
        record.attempt > previous.attempt ||
        (record.attempt === previous.attempt && record.seq > previous.seq)
      )
        row.set(field, { value, attempt: record.attempt, seq: record.seq });
    }
  }

  const steps = [...rows.keys()].sort((a, b) => a - b);

  const points = steps.map((step): PlotPoint => ({
    step,
    values: Object.fromEntries(
      selected.map((field) => [field, rows.get(step)?.get(field)?.value ?? null]),
    ),
  }));

  return {
    points,
    fields: selected.filter((field) => points.some((point) => point.values[field] != null)),
    pointCount: steps.length,
  };
}

export function minimumPoint(
  series: PlotSeries,
  field: RuntimeMetricField,
): { step: number; value: number } | null {
  let minimum: { step: number; value: number } | null = null;

  for (const point of series.points) {
    const value = point.values[field];

    if (value != null && (minimum === null || value < minimum.value))
      minimum = { step: point.step, value };
  }

  return minimum;
}
