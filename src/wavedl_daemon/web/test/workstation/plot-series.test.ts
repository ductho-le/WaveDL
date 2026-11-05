import { expect, test } from 'vitest';
import type { MetricRecord } from '../../app/lib/metric-history';
import { buildPlotSeries, minimumPoint } from '../../app/lib/plot-series';

function record(overrides: Partial<MetricRecord>): MetricRecord {
  return {
    runId: 'run',
    attempt: 1,
    seq: 1,
    step: 1,
    name: 'epoch',
    value: null,
    unit: '',
    metrics: {},
    active: true,
    ...overrides,
  };
}

test('aligns epochs by step, keeps the latest attempt, skips inactive records', () => {
  const series = buildPlotSeries([
    record({ attempt: 1, seq: 5, step: 2, metrics: { val_loss: 0.4 } }),
    record({ attempt: 2, seq: 1, step: 2, metrics: { train_loss: 0.5, val_loss: 0.3 } }),
    record({ attempt: 1, seq: 1, step: 1, metrics: { train_loss: 0.9 } }),
    record({ attempt: 2, seq: 2, step: 3, metrics: { val_loss: 0.1 }, active: false }),
  ], 'loss');

  expect(series.points).toEqual([
    { step: 1, values: { train_loss: 0.9, val_loss: null } },
    { step: 2, values: { train_loss: 0.5, val_loss: 0.3 } },
  ]);
  expect(series.pointCount).toBe(2);
});

test('lists only fields that have at least one value', () => {
  expect(buildPlotSeries([record({ seq: 1, step: 1, metrics: { val_r2: 0.8 } })], 'r2').fields).toEqual(['val_r2']);
});

test('minimumPoint finds the lowest value at its earliest epoch', () => {
  const series = buildPlotSeries([
    record({ seq: 1, step: 1, metrics: { val_loss: 0.5 } }),
    record({ seq: 2, step: 2, metrics: { val_loss: 0.2 } }),
    record({ seq: 3, step: 3, metrics: { val_loss: 0.2 } }),
  ], 'loss');

  expect(minimumPoint(series, 'val_loss')).toEqual({ step: 2, value: 0.2 });
  expect(minimumPoint(buildPlotSeries([], 'loss'), 'val_loss')).toBeNull();
});
