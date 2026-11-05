import { expect, test } from 'vitest';
import type { RunView } from '../../app/lib/daemon-api';
import { datasetFormat, datasetInputShape, datasetOutputValues, datasetSuggestions } from '../../app/lib/dataset-suggestions';

function run(id: string, path: string | null, created: string | null = null): RunView {
  return {
    run_id: id,
    plan_id: id,
    request_id: id,
    backend: 'local',
    operation: 'train',
    attempt: 1,
    state: 'completed',
    uncertainty: false,
    submission_requested: true,
    cancellation_requested: false,
    evidence: null,
    created_at: created,
    summary: { name: null, model: null, data_path: path, epochs: null, cv: null, source_run_id: null },
  };
}

test('groups run paths newest-first and counts each run once', () => {
  const suggestions = datasetSuggestions([
    run('1', '/data/old.npz', '2026-01-01'),
    run('2', '/data/new.hdf5', '2026-04-01'),
    run('3', '/data/old.npz', '2026-03-01'),
    run('3', '/data/old.npz', '2026-03-01'),
  ]);

  expect(suggestions.map((item) => [item.name, item.count, item.format])).toEqual([
    ['new.hdf5', 1, 'H5'], ['old.npz', 2, 'NPZ'],
  ]);
  expect(suggestions[1]?.lastUsed?.toISOString()).toBe('2026-03-01T00:00:00.000Z');
});

test('ignores unsupported files and missing summaries, preserves distinct full paths', () => {
  expect(datasetSuggestions([
    run('1', '/data/readme.txt'), run('2', null), run('3', '/a/train.mat'), run('4', '/b/train.mat'),
  ]).map((item) => item.path)).toEqual(['/b/train.mat', '/a/train.mat']);
});

test('handles windows paths and invalid or missing dates', () => {
  const suggestions = datasetSuggestions([
    run('1', 'C:\\data\\train.NPZ', 'invalid'), run('2', 'C:/data/train.NPZ'),
    run('3', '/data/dated.h5', '2026-01-01'),
  ]);

  expect(suggestions[0]?.name).toBe('dated.h5');
  expect(suggestions[1]?.name).toBe('train.NPZ');
  expect(suggestions[1]?.count).toBe(2);
  expect(suggestions[1]?.lastUsed).toBeNull();
});

test('keeps all candidates so unresolved suggestions can be replaced by older paths', () => {
  expect(datasetSuggestions(Array.from({ length: 7 }, (_, index) => run(String(index), `/data/${index}.npz`)))).toHaveLength(7);
  expect(datasetSuggestions([])).toEqual([]);
});

test('formats supported file types and scalar or multidimensional sample shapes', () => {
  expect(datasetFormat('train.HDF5')).toBe('H5');
  expect(datasetFormat('train.npz.zip')).toBeNull();
  expect(datasetInputShape([])).toBe('1');
  expect(datasetInputShape([2, 1024])).toBe('2 × 1024');
  expect(datasetOutputValues([])).toBe(1);
  expect(datasetOutputValues([2, 3])).toBe(6);
  expect(datasetOutputValues([0])).toBeNull();
  expect(datasetOutputValues([Number.MAX_SAFE_INTEGER, 2])).toBeNull();
});
