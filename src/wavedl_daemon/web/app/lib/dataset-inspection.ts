import { CatalogApi, type DataSelection } from './catalog-api';
import { ConnectionSession, ConnectionSessionError } from './connection-session';
import { DaemonApiError, type Json } from './daemon-api';
import { datasetInspectionSchema } from '../../shared/schemas/dataset-inspection';

export interface DatasetInspection {
  format: string;
  numSamples: number;
  inputKey: string;
  outputKey: string;
  inputShape: number[];
  outputShape: number[];
  inputDtype: string;
  outputDtype: string;
  fileSize: null;
}

export function parseDatasetInspection(
  value: Json,
  selection: Pick<DataSelection, 'relativePath'>,
): DatasetInspection {
  const fail = (): never => {
    throw DaemonApiError.fromKey('response', 'datasetHeaderUnsafe');
  };

  let bytes: number;

  try {
    bytes = new TextEncoder().encode(JSON.stringify(value)).length;
  } catch {
    return fail();
  }

  if (bytes > 65536) return fail();
  const parsed = datasetInspectionSchema.safeParse(value);

  if (!parsed.success) return fail();
  const metadata = parsed.data;

  if (metadata.format !== selection.relativePath.split('.').at(-1)?.toLowerCase()) return fail();
  const variables = new Map(metadata.variables.map((item) => [item.name, item]));

  if (
    variables.size !== metadata.variables.length ||
    metadata.variables.reduce((total, item) => total + item.declared_bytes, 0) > 8 * 1024 ** 3
  )
    return fail();

  for (const item of metadata.variables) {
    if (item.shape.length ? item.sample_count !== item.shape[0] : item.sample_count !== null)
      return fail();
  }

  // Priority is shared with wavedl.core.data_keys and the Qt metadata consumer.
  const inputs = ['input_train', 'input_test', 'X', 'data', 'inputs', 'features', 'x']
    .map((key) => variables.get(key))
    .find(Boolean);

  const outputs = ['output_train', 'output_test', 'Y', 'labels', 'outputs', 'targets', 'y']
    .map((key) => variables.get(key))
    .find(Boolean);

  if (
    !inputs ||
    !outputs ||
    !inputs.dtype ||
    !outputs.dtype ||
    !inputs.shape.length ||
    !outputs.shape.length ||
    inputs.shape.length > 8 ||
    outputs.shape.length > 8 ||
    inputs.shape.some((size) => size < 1) ||
    outputs.shape.some((size) => size < 1) ||
    inputs.sample_count !== outputs.sample_count ||
    inputs.sample_count === null
  )
    return fail();

  return {
    format: metadata.format,
    numSamples: inputs.sample_count,
    inputKey: inputs.name,
    outputKey: outputs.name,
    inputShape: inputs.shape.slice(1),
    outputShape: outputs.shape.slice(1).length ? outputs.shape.slice(1) : [1],
    inputDtype: inputs.dtype,
    outputDtype: outputs.dtype,
    fileSize: null,
  };
}

export class DatasetInspectionApi {
  constructor(readonly session: ConnectionSession) {}
  inspect(selection: DataSelection, signal?: AbortSignal): Promise<DatasetInspection> {
    try {
      new CatalogApi(this.session).assertSelection(selection);

      if (selection.kind !== 'file' || !/\.(npz|mat|h5|hdf5)$/i.test(selection.relativePath))
        throw ConnectionSessionError.fromKey('request', 'datasetSelectionRequired');
    } catch (error) {
      return Promise.reject(error);
    }

    return this.session.perform(
      async (api, requestSignal) => {
        new CatalogApi(this.session).assertSelection(selection);

        const body = await api.mutate(
          '/api/datasets/inspect',
          { root: selection.root, path: selection.relativePath },
          { signal: requestSignal },
        );

        return parseDatasetInspection(body, selection);
      },
      { signal, timeoutMs: 60000 },
    );
  }
}
