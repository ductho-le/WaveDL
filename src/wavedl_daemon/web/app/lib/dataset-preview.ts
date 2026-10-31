import type { z } from 'zod';
import { CatalogApi, type DataSelection } from './catalog-api';
import { ConnectionSession, ConnectionSessionError } from './connection-session';
import { DaemonApiError, type Json } from './daemon-api';
import { datasetPreviewSchema, previewValuesSchema, type PreviewValues, PREVIEW_VIEWS } from '../../shared/schemas/dataset-preview';

export { PREVIEW_VIEWS } from '../../shared/schemas/dataset-preview';

export type PreviewView = (typeof PREVIEW_VIEWS)[number];

export type DatasetPreview = z.infer<typeof datasetPreviewSchema>;

export type PreviewSample = DatasetPreview['samples'][number];

const product = (values: readonly number[]) => values.reduce((result, value) => result * value, 1);

function boundedIndices(values: readonly number[], width: number, maximum: number) {
  return (
    values.length <= maximum &&
    values.every(
      (value, index) => value < width && (index === 0 || value > (values[index - 1] ?? -1)),
    )
  );
}

function nested(value: PreviewValues, dimensions: readonly number[]): boolean {
  if (!dimensions.length)
    return !Array.isArray(value) && previewValuesSchema.safeParse(value).success;

  return (
    Array.isArray(value) &&
    value.length === dimensions[0] &&
    value.every((item) => nested(item, dimensions.slice(1)))
  );
}

export function parseDatasetPreview(
  value: Json,
  selection: Pick<DataSelection, 'root' | 'relativePath'>,
  view: PreviewView,
): DatasetPreview {
  const fail = (): never => {
    throw DaemonApiError.fromKey('response', 'invalidDatasetPreview');
  };

  let bytes: number;

  try {
    bytes = new TextEncoder().encode(JSON.stringify(value)).length;
  } catch {
    return fail();
  }

  if (bytes > 1048576) return fail();
  const parsed = datasetPreviewSchema.safeParse(value);

  if (!parsed.success) return fail();
  const data = parsed.data;
  const cap = data.limits;
  const format = selection.relativePath.split('.').at(-1)?.toLowerCase();

  if (
    data.format !== format ||
    bytes > cap.max_output_bytes ||
    data.selection.root !== selection.root ||
    data.selection.path !== selection.relativePath ||
    data.view !== view
  )
    return fail();
  const rows = data.input_shape[0] ?? 0;
  const xwidth = product(data.input_shape.slice(1));
  const ywidth = product(data.output_shape.slice(1));
  const count = data.sample_indices.length;

  if (
    rows !== data.output_shape[0] ||
    product(data.input_shape) > cap.max_elements ||
    product(data.output_shape) > cap.max_elements ||
    xwidth > cap.max_input_features ||
    ywidth > cap.max_output_features ||
    data.identity.st_size > cap.max_file_bytes ||
    !boundedIndices(data.sample_indices, rows, cap.max_sample_rows)
  )
    return fail();

  for (const [histograms, width] of [
    [data.input_histograms, xwidth],
    [data.output_histograms, ywidth],
  ] as const) {
    if (
      !boundedIndices(
        histograms.map((item) => item.feature_index),
        width,
        Math.min(8, width),
      )
    )
      return fail();

    for (const histogram of histograms)
      if (
        histogram.counts.reduce((sum, item) => sum + item, 0) > count ||
        !histogram.edges.every(
          (item, index) => index === 0 || item >= (histogram.edges[index - 1] ?? item),
        )
      )
        return fail();
  }

  const correlation = data.correlation;

  if (
    !correlation.feature_indices.length ||
    !boundedIndices(correlation.feature_indices, ywidth, 30) ||
    (count <= 1 || correlation.feature_indices.length <= 1
      ? correlation.matrix !== null
      : !nested(correlation.matrix, [
          correlation.feature_indices.length,
          correlation.feature_indices.length,
        ]))
  )
    return fail();

  if (correlation.matrix?.some((row) => row.some((value) => value !== null && Math.abs(value) > 1)))
    return fail();
  const stats = data.input_statistics;

  if (
    !stats.feature_indices.length ||
    !boundedIndices(stats.feature_indices, xwidth, 40) ||
    stats.means.length !== stats.feature_indices.length ||
    stats.stds.length !== stats.feature_indices.length ||
    stats.stds.some((item) => item !== null && item < 0)
  )
    return fail();

  if (
    !data.scatter.feature_indices.length ||
    !boundedIndices(data.scatter.feature_indices, xwidth, 8) ||
    !nested(data.scatter.x, [count, data.scatter.feature_indices.length]) ||
    data.scatter.y.length !== count ||
    data.samples.length > count
  )
    return fail();
  let elements = 0;
  const sampleIds = new Set<number>();

  for (const sample of data.samples) {
    const original = data.input_shape.slice(1);
    elements += product(sample.shape);

    if (
      sampleIds.has(sample.sample_index) ||
      !data.sample_indices.includes(sample.sample_index) ||
      JSON.stringify(sample.feature_shape) !== JSON.stringify(original) ||
      sample.shape.length !== Math.max(1, original.length) ||
      sample.shape.some((size, index) => size > (original[index] ?? 1)) ||
      !nested(sample.values, sample.shape)
    )
      return fail();
    sampleIds.add(sample.sample_index);
  }

  if (elements > cap.max_preview_elements) return fail();

  return data;
}

export class DatasetPreviewApi {
  constructor(readonly session: ConnectionSession) {}
  preview(
    selection: DataSelection,
    view: PreviewView,
    signal?: AbortSignal,
  ): Promise<DatasetPreview> {
    try {
      new CatalogApi(this.session).assertSelection(selection);

      if (
        selection.kind !== 'file' ||
        !/\.(npz|mat|h5|hdf5)$/i.test(selection.relativePath) ||
        !PREVIEW_VIEWS.includes(view)
      )
        throw ConnectionSessionError.fromKey('request', 'datasetSelectionRequired');
    } catch (error) {
      return Promise.reject(error);
    }

    return this.session.perform(
      async (api, requestSignal) => {
        new CatalogApi(this.session).assertSelection(selection);

        const body = await api.mutate(
          '/api/datasets/preview',
          { root: selection.root, path: selection.relativePath, view },
          { signal: requestSignal },
        );

        return parseDatasetPreview(body, selection, view);
      },
      { signal, timeoutMs: 60000 },
    );
  }
}

export function sampleMatrix(sample: PreviewSample): (number | null)[][] {
  const flat: (number | null)[] = [];

  const visit = (value: PreviewValues): void => {
    if (Array.isArray(value)) value.forEach(visit);
    else flat.push(value);
  };

  visit(sample.values);
  const width = sample.shape.at(-1) ?? 1;

  return Array.from({ length: flat.length / width }, (_, row) =>
    flat.slice(row * width, (row + 1) * width),
  );
}
